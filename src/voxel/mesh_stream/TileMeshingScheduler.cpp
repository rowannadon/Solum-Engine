#include "solum_engine/voxel/MeshManager.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iterator>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "solum_engine/resources/Constants.h"
#include "solum_engine/voxel/World.h"

namespace {
constexpr int kMinPrefetchChunks = 4;

struct FootprintDistanceRange {
    int32_t minDistanceChunks = 0;
    int32_t maxDistanceChunks = 0;
};

int32_t minDistanceToInterval(int32_t value, int32_t minValue, int32_t maxValue) {
    if (value < minValue) {
        return minValue - value;
    }
    if (value > maxValue) {
        return value - maxValue;
    }
    return 0;
}

int32_t maxDistanceToInterval(int32_t value, int32_t minValue, int32_t maxValue) {
    const int64_t distanceToMin = std::llabs(static_cast<int64_t>(value) - static_cast<int64_t>(minValue));
    const int64_t distanceToMax = std::llabs(static_cast<int64_t>(value) - static_cast<int64_t>(maxValue));
    return static_cast<int32_t>(std::max(distanceToMin, distanceToMax));
}

FootprintDistanceRange footprintDistanceRangeForCell(int32_t cellX,
                                                     int32_t cellY,
                                                     int32_t spanChunks,
                                                     const ChunkCoord& centerChunk) {
    const int32_t minChunkX = cellX * spanChunks;
    const int32_t maxChunkX = minChunkX + spanChunks - 1;
    const int32_t minChunkY = cellY * spanChunks;
    const int32_t maxChunkY = minChunkY + spanChunks - 1;

    const int32_t minDx = minDistanceToInterval(centerChunk.v.x, minChunkX, maxChunkX);
    const int32_t minDy = minDistanceToInterval(centerChunk.v.y, minChunkY, maxChunkY);
    const int32_t maxDx = maxDistanceToInterval(centerChunk.v.x, minChunkX, maxChunkX);
    const int32_t maxDy = maxDistanceToInterval(centerChunk.v.y, minChunkY, maxChunkY);

    return FootprintDistanceRange{
        std::max(minDx, minDy),
        std::max(maxDx, maxDy)
    };
}
}  // namespace

void MeshManager::scheduleTilesAround(const ChunkCoord& centerChunk,
                                      const glm::vec3& playerWorldPosition,
                                      float sseProjectionScale,
                                      const ChunkCoord* previousCenterChunk,
                                      int32_t centerShiftChunks) {
    struct ScheduledTileLod {
        int32_t distanceSq = 0;
        TileLodCoord coord{};
        jobsystem::Priority priority = jobsystem::Priority::Low;
        bool forceRemesh = false;
        int32_t activeWindowExtraChunks = 0;
    };

    std::vector<ScheduledTileLod> primaryJobsToSchedule;
    std::vector<ScheduledTileLod> backfillJobsToSchedule;
    std::unordered_map<MeshTileCoord, int8_t> desiredUpdatesByTile;
    std::unordered_set<MeshTileCoord> tilesToProcess;

    const int32_t maxRadiusChunks = std::max(0, maxConfiguredRadius());
    const int32_t clampedCenterShift = std::min(centerShiftChunks, 2);
    const int32_t prefetchChunks = std::max(kMinPrefetchChunks, clampedCenterShift * meshTileSizeChunks_);
    const int32_t scheduleOuterRadiusChunks = maxRadiusChunks + prefetchChunks;
    const int8_t maxLod = static_cast<int8_t>(config_.lodChunkRadii.size() - 1);

    auto computeTileBounds = [this, scheduleOuterRadiusChunks](const ChunkCoord& chunkCoord) {
        const int32_t minTileX = floor_div(
            chunkCoord.v.x - scheduleOuterRadiusChunks - (meshTileSizeChunks_ - 1),
            meshTileSizeChunks_
        );
        const int32_t maxTileX = floor_div(chunkCoord.v.x + scheduleOuterRadiusChunks, meshTileSizeChunks_);
        const int32_t minTileY = floor_div(
            chunkCoord.v.y - scheduleOuterRadiusChunks - (meshTileSizeChunks_ - 1),
            meshTileSizeChunks_
        );
        const int32_t maxTileY = floor_div(chunkCoord.v.y + scheduleOuterRadiusChunks, meshTileSizeChunks_);
        return std::array<int32_t, 4>{minTileX, maxTileX, minTileY, maxTileY};
    };

    const auto currentBounds = computeTileBounds(centerChunk);
    const int32_t minTileX = currentBounds[0];
    const int32_t maxTileX = currentBounds[1];
    const int32_t minTileY = currentBounds[2];
    const int32_t maxTileY = currentBounds[3];

    const bool hadPreviousCenter = (previousCenterChunk != nullptr);
    const bool treatAsLargeJump = !hadPreviousCenter || centerShiftChunks >= (meshTileSizeChunks_ * 4);

    if (!treatAsLargeJump) {
        const auto previousBounds = computeTileBounds(*previousCenterChunk);
        for (int32_t tileY = minTileY; tileY <= maxTileY; ++tileY) {
            for (int32_t tileX = minTileX; tileX <= maxTileX; ++tileX) {
                const MeshTileCoord tileCoord{tileX, tileY};
                if (!tileInBounds(tileCoord,
                                  previousBounds[0],
                                  previousBounds[1],
                                  previousBounds[2],
                                  previousBounds[3])) {
                    tilesToProcess.insert(tileCoord);
                }
            }
        }
    } else {
        for (int32_t tileY = minTileY; tileY <= maxTileY; ++tileY) {
            for (int32_t tileX = minTileX; tileX <= maxTileX; ++tileX) {
                tilesToProcess.insert(MeshTileCoord{tileX, tileY});
            }
        }
    }

    // Incremental sweep over the active window to avoid scanning every tile each update.
    int32_t sweepStartIndex = 0;
    int32_t sweepCount = 0;
    int32_t sweepWidth = 0;
    int32_t sweepHeight = 0;
    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        if (!hasLodRefreshScanCenter_ || !(centerChunk == lodRefreshScanCenterChunk_)) {
            lodRefreshScanCenterChunk_ = centerChunk;
            hasLodRefreshScanCenter_ = true;
            if (treatAsLargeJump) {
                lodRefreshScanNextIndex_ = 0;
            }
        }

        sweepWidth = (maxTileX - minTileX) + 1;
        sweepHeight = (maxTileY - minTileY) + 1;
        const int32_t sweepTotal = std::max(0, sweepWidth * sweepHeight);
        sweepStartIndex = std::clamp(lodRefreshScanNextIndex_, 0, std::max(0, sweepTotal - 1));
        const int32_t baseBudget = 128;
        const int32_t shiftBudget = std::max(0, centerShiftChunks) * 64;
        const int32_t sweepBudget = baseBudget + shiftBudget;
        sweepCount = std::min(sweepTotal, sweepBudget);
        lodRefreshScanNextIndex_ = (lodRefreshScanNextIndex_ + sweepCount) % std::max(1, sweepTotal);
    }

    for (int32_t i = 0; i < sweepCount; ++i) {
        const int32_t wrappedIndex = (sweepStartIndex + i) % std::max(1, sweepWidth * sweepHeight);
        const int32_t localY = wrappedIndex / std::max(1, sweepWidth);
        const int32_t localX = wrappedIndex % std::max(1, sweepWidth);
        tilesToProcess.insert(MeshTileCoord{
            minTileX + localX,
            minTileY + localY
        });
    }

    // Always refresh a small near-player tile neighborhood each update.
    const int32_t centerTileX = floor_div(centerChunk.v.x, meshTileSizeChunks_);
    const int32_t centerTileY = floor_div(centerChunk.v.y, meshTileSizeChunks_);
    const int32_t nearTileRadius = 1;
    for (int32_t dy = -nearTileRadius; dy <= nearTileRadius; ++dy) {
        for (int32_t dx = -nearTileRadius; dx <= nearTileRadius; ++dx) {
            const MeshTileCoord nearTile{centerTileX + dx, centerTileY + dy};
            if (tileInBounds(nearTile, minTileX, maxTileX, minTileY, maxTileY)) {
                tilesToProcess.insert(nearTile);
            }
        }
    }

    std::unordered_map<MeshTileCoord, int8_t> previousDesiredByTile;
    {
        std::shared_lock<std::shared_mutex> lock(meshMutex_);
        previousDesiredByTile.reserve(tilesToProcess.size());
        for (const MeshTileCoord& tileCoord : tilesToProcess) {
            const auto tileIt = meshTiles_.find(tileCoord);
            if (tileIt == meshTiles_.end()) {
                continue;
            }
            previousDesiredByTile.emplace(tileCoord, tileIt->second.desiredLod);
        }
    }

    for (const MeshTileCoord& tileCoord : tilesToProcess) {
        if (!tileInBounds(tileCoord, minTileX, maxTileX, minTileY, maxTileY)) {
            continue;
        }

        const FootprintDistanceRange distances = footprintDistanceRangeForCell(
            tileCoord.x,
            tileCoord.y,
            meshTileSizeChunks_,
            centerChunk
        );
        if (distances.minDistanceChunks > scheduleOuterRadiusChunks) {
            desiredUpdatesByTile[tileCoord] = -1;
            continue;
        }

        const int8_t visibleDesired = desiredLodForTile(
            tileCoord,
            centerChunk,
            playerWorldPosition,
            sseProjectionScale,
            0
        );
        const int8_t prefetchDesired = desiredLodForTile(
            tileCoord,
            centerChunk,
            playerWorldPosition,
            sseProjectionScale,
            prefetchChunks
        );
        const int8_t candidateDesired = (visibleDesired >= 0) ? visibleDesired : prefetchDesired;
        const auto previousDesiredIt = previousDesiredByTile.find(tileCoord);
        const int8_t previousDesired = (previousDesiredIt != previousDesiredByTile.end())
            ? previousDesiredIt->second
            : -1;
        const int8_t baseDesired = applyLodHysteresis(
            tileCoord,
            candidateDesired,
            previousDesired,
            playerWorldPosition,
            sseProjectionScale
        );
        desiredUpdatesByTile[tileCoord] = baseDesired;
        if (baseDesired < 0) {
            continue;
        }

        const int32_t distanceSq = distances.minDistanceChunks * distances.minDistanceChunks;
        const int32_t lodMax = std::min<int32_t>(maxLod, static_cast<int32_t>(baseDesired) + 1);
        const int32_t activeWindowExtraChunks = prefetchChunks + meshTileSizeChunks_;

        primaryJobsToSchedule.push_back(ScheduledTileLod{
            distanceSq,
            TileLodCoord{tileCoord, static_cast<uint8_t>(baseDesired)},
            priorityFromLodLevel(static_cast<uint8_t>(baseDesired)),
            false,
            activeWindowExtraChunks
        });

        for (int32_t lod = static_cast<int32_t>(baseDesired) + 1; lod <= lodMax; ++lod) {
            backfillJobsToSchedule.push_back(ScheduledTileLod{
                distanceSq,
                TileLodCoord{tileCoord, static_cast<uint8_t>(lod)},
                jobsystem::Priority::Low,
                false,
                activeWindowExtraChunks
            });
        }
    }

    auto sortScheduledJobs = [](std::vector<ScheduledTileLod>& jobs) {
        std::sort(jobs.begin(), jobs.end(), [](const ScheduledTileLod& a, const ScheduledTileLod& b) {
            if (a.distanceSq != b.distanceSq) {
                return a.distanceSq < b.distanceSq;
            }
            if (!(a.coord.tile == b.coord.tile)) {
                return a.coord.tile < b.coord.tile;
            }
            return a.coord.lodLevel < b.coord.lodLevel;
        });
    };
    sortScheduledJobs(primaryJobsToSchedule);
    sortScheduledJobs(backfillJobsToSchedule);

    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);

        for (const auto& [tileCoord, desiredLod] : desiredUpdatesByTile) {
            if (desiredLod < 0) {
                auto tileIt = meshTiles_.find(tileCoord);
                if (tileIt != meshTiles_.end()) {
                    tileIt->second.desiredLod = -1;
                }
                continue;
            }
            MeshTileState& tileState = meshTiles_[tileCoord];
            tileState.desiredLod = desiredLod;
        }

        for (auto& [tileCoord, tileState] : meshTiles_) {
            if (!tileInBounds(tileCoord, minTileX, maxTileX, minTileY, maxTileY)) {
                tileState.desiredLod = -1;
            }
        }

        refreshRenderedLodsLocked();

        const int32_t pruneExtraChunks = prefetchChunks + meshTileSizeChunks_;
        for (auto it = meshTiles_.begin(); it != meshTiles_.end();) {
            if (isTileWithinActiveWindowLocked(it->first, pruneExtraChunks)) {
                ++it;
                continue;
            }

            bool hasPendingForTile = false;
            for (const TileLodCellCoord& pending : pendingTileJobs_) {
                if (pending.tileLod.tile == it->first) {
                    hasPendingForTile = true;
                    break;
                }
            }

            if (hasPendingForTile) {
                ++it;
                continue;
            }

            it = meshTiles_.erase(it);
        }
    }

    for (const ScheduledTileLod& scheduled : primaryJobsToSchedule) {
        scheduleTileLodMeshing(
            scheduled.coord,
            scheduled.priority,
            scheduled.forceRemesh,
            scheduled.activeWindowExtraChunks
        );
    }

    for (const ScheduledTileLod& scheduled : backfillJobsToSchedule) {
        scheduleTileLodMeshing(
            scheduled.coord,
            scheduled.priority,
            scheduled.forceRemesh,
            scheduled.activeWindowExtraChunks
        );
    }
}

void MeshManager::scheduleRemeshForNewColumns(const ColumnCoord& centerColumn) {
    constexpr std::size_t kRemeshColumnsPerUpdate = 512;
    const uint64_t processedRevision = processedWorldGenerationRevision_.load(std::memory_order_acquire);
    std::vector<ColumnCoord> generatedColumns;
    const uint64_t nextRevision = world_.copyGeneratedColumnsSince(
        processedRevision,
        generatedColumns,
        kRemeshColumnsPerUpdate
    );
    if (nextRevision == processedRevision) {
        return;
    }
    processedWorldGenerationRevision_.store(nextRevision, std::memory_order_release);

    if (generatedColumns.empty()) {
        return;
    }

    const int32_t remeshRadius = std::max(0, maxConfiguredRadius() + meshTileSizeChunks_ + kMinPrefetchChunks);
    std::unordered_set<MeshTileCoord> tilesToRemesh;

    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        for (const ColumnCoord& coord : generatedColumns) {
            const int32_t dx = std::abs(coord.v.x - centerColumn.v.x);
            const int32_t dy = std::abs(coord.v.y - centerColumn.v.y);
            if (dx > remeshRadius || dy > remeshRadius) {
                continue;
            }

            if (!knownGeneratedColumns_.insert(coord).second) {
                continue;
            }

            const int32_t tileX = floor_div(coord.v.x, meshTileSizeChunks_);
            const int32_t tileY = floor_div(coord.v.y, meshTileSizeChunks_);
            const int32_t localX = floor_mod(coord.v.x, meshTileSizeChunks_);
            const int32_t localY = floor_mod(coord.v.y, meshTileSizeChunks_);
            tilesToRemesh.insert(MeshTileCoord{tileX, tileY});

            const bool touchesLeftEdge = (localX == 0);
            const bool touchesRightEdge = (localX == meshTileSizeChunks_ - 1);
            const bool touchesBottomEdge = (localY == 0);
            const bool touchesTopEdge = (localY == meshTileSizeChunks_ - 1);

            if (touchesLeftEdge) {
                tilesToRemesh.insert(MeshTileCoord{tileX - 1, tileY});
            }
            if (touchesRightEdge) {
                tilesToRemesh.insert(MeshTileCoord{tileX + 1, tileY});
            }
            if (touchesBottomEdge) {
                tilesToRemesh.insert(MeshTileCoord{tileX, tileY - 1});
            }
            if (touchesTopEdge) {
                tilesToRemesh.insert(MeshTileCoord{tileX, tileY + 1});
            }

            if (touchesLeftEdge && touchesBottomEdge) {
                tilesToRemesh.insert(MeshTileCoord{tileX - 1, tileY - 1});
            }
            if (touchesLeftEdge && touchesTopEdge) {
                tilesToRemesh.insert(MeshTileCoord{tileX - 1, tileY + 1});
            }
            if (touchesRightEdge && touchesBottomEdge) {
                tilesToRemesh.insert(MeshTileCoord{tileX + 1, tileY - 1});
            }
            if (touchesRightEdge && touchesTopEdge) {
                tilesToRemesh.insert(MeshTileCoord{tileX + 1, tileY + 1});
            }
        }
    }

    if (tilesToRemesh.empty()) {
        return;
    }

    ChunkCoord seamCenterChunk{};
    glm::vec3 seamPlayerWorldPosition{
        static_cast<float>(centerColumn.v.x * cfg::CHUNK_SIZE),
        static_cast<float>(centerColumn.v.y * cfg::CHUNK_SIZE),
        static_cast<float>(cfg::COLUMN_HEIGHT_BLOCKS) * 0.5f
    };
    float seamSseProjectionScale = config_.lodSseFallbackProjectionScale;
    {
        std::shared_lock<std::shared_mutex> lock(meshMutex_);
        seamCenterChunk = hasLastScheduledCenter_
            ? lastScheduledCenterChunk_
            : ChunkCoord{centerColumn.v.x, centerColumn.v.y, 0};
        if (hasLastSseProjectionScale_) {
            seamPlayerWorldPosition = lastPlayerWorldPosition_;
            seamSseProjectionScale = lastSseProjectionScale_;
        }
    }

    const int8_t maxLod = static_cast<int8_t>(config_.lodChunkRadii.size() - 1);
    for (const MeshTileCoord& tileCoord : tilesToRemesh) {
        const int8_t visibleDesired = desiredLodForTile(
            tileCoord,
            seamCenterChunk,
            seamPlayerWorldPosition,
            seamSseProjectionScale,
            0
        );
        const int8_t prefetchDesired = desiredLodForTile(
            tileCoord,
            seamCenterChunk,
            seamPlayerWorldPosition,
            seamSseProjectionScale,
            kMinPrefetchChunks
        );
        const int8_t baseDesired = (visibleDesired >= 0) ? visibleDesired : prefetchDesired;
        if (baseDesired < 0) {
            continue;
        }

        const int32_t lodMax = std::min<int32_t>(maxLod, static_cast<int32_t>(baseDesired) + 1);
        const int32_t activeWindowExtraChunks = kMinPrefetchChunks + meshTileSizeChunks_;

        scheduleTileLodMeshing(
            TileLodCoord{tileCoord, static_cast<uint8_t>(baseDesired)},
            priorityFromLodLevel(static_cast<uint8_t>(baseDesired)),
            true,
            activeWindowExtraChunks
        );

        for (int32_t lod = static_cast<int32_t>(baseDesired) + 1; lod <= lodMax; ++lod) {
            scheduleTileLodMeshing(
                TileLodCoord{tileCoord, static_cast<uint8_t>(lod)},
                jobsystem::Priority::Low,
                true,
                activeWindowExtraChunks
            );
        }
    }
}

void MeshManager::scheduleTileLodMeshing(const TileLodCoord& coord,
                                         jobsystem::Priority priority,
                                         bool forceRemesh,
                                         int32_t activeWindowExtraChunks) {
    if (!isTileFootprintGenerated(coord.tile)) {
        return;
    }

    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        MeshTileState& tileState = meshTiles_[coord.tile];
        MeshTileLodState& lodState = tileState.lodStates[coord.lodLevel];
        const int32_t cellsPerAxis = cellCountPerAxisForLod(coord.lodLevel);
        lodState.expectedCellCount = cellsPerAxis * cellsPerAxis;
    }

    const int32_t cellsPerAxis = cellCountPerAxisForLod(coord.lodLevel);
    for (int32_t cellY = 0; cellY < cellsPerAxis; ++cellY) {
        for (int32_t cellX = 0; cellX < cellsPerAxis; ++cellX) {
            scheduleTileLodCellMeshing(
                TileLodCellCoord{
                    coord,
                    static_cast<uint16_t>(cellX),
                    static_cast<uint16_t>(cellY)
                },
                priority,
                forceRemesh,
                activeWindowExtraChunks
            );
        }
    }
}

void MeshManager::scheduleTileLodCellMeshing(const TileLodCellCoord& coord,
                                             jobsystem::Priority priority,
                                             bool forceRemesh,
                                             int32_t activeWindowExtraChunks) {
    if (!isTileFootprintGenerated(coord.tileLod.tile)) {
        return;
    }

    const int32_t clampedActiveWindowExtraChunks = std::max(0, activeWindowExtraChunks);
    const uint32_t cellKey = packCellKey(coord.cellX, coord.cellY);

    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        if (pendingTileJobs_.find(coord) != pendingTileJobs_.end()) {
            if (forceRemesh) {
                deferredRemeshTileLods_.insert(coord);
            }
            return;
        }

        if (!isTileWithinActiveWindowLocked(coord.tileLod.tile, clampedActiveWindowExtraChunks)) {
            return;
        }

        const auto tileIt = meshTiles_.find(coord.tileLod.tile);
        if (tileIt != meshTiles_.end() && !forceRemesh) {
            const auto lodIt = tileIt->second.lodStates.find(coord.tileLod.lodLevel);
            if (lodIt != tileIt->second.lodStates.end() &&
                lodIt->second.cellMeshes.find(cellKey) != lodIt->second.cellMeshes.end()) {
                return;
            }
        }

        pendingTileJobs_.insert(coord);
    }

    try {
        jobs_.schedule(
            priority,
            [this, coord, clampedActiveWindowExtraChunks]() -> MeshGenerationResult {
                {
                    std::shared_lock<std::shared_mutex> lock(meshMutex_);
                    if (!isTileWithinActiveWindowLocked(coord.tileLod.tile, clampedActiveWindowExtraChunks)) {
                        return MeshGenerationResult{coord, {}, false};
                    }
                }

                if (!isTileFootprintGenerated(coord.tileLod.tile)) {
                    return MeshGenerationResult{coord, {}, false};
                }

                const uint8_t lodLevel = coord.tileLod.lodLevel;
                const int32_t spanChunks = static_cast<int32_t>(chunkSpanForLod(lodLevel));
                const int32_t tileOriginChunkX = coord.tileLod.tile.x * meshTileSizeChunks_;
                const int32_t tileOriginChunkY = coord.tileLod.tile.y * meshTileSizeChunks_;
                const int32_t baseCellX = floor_div(tileOriginChunkX, spanChunks);
                const int32_t baseCellY = floor_div(tileOriginChunkY, spanChunks);
                const int32_t cellsPerAxis = std::max(1, meshTileSizeChunks_ / spanChunks);
                const int32_t zCount = chunkZCountForLod(lodLevel);
                const int32_t cellSpanLodCells = cellSpanLodCellsForLod(lodLevel);

                const int32_t localStartX = static_cast<int32_t>(coord.cellX) * cellSpanLodCells;
                const int32_t localStartY = static_cast<int32_t>(coord.cellY) * cellSpanLodCells;
                const int32_t localEndX = std::min(cellsPerAxis, localStartX + cellSpanLodCells);
                const int32_t localEndY = std::min(cellsPerAxis, localStartY + cellSpanLodCells);

                std::vector<Meshlet> meshlets;
                std::unordered_map<ColumnCoord, uint32_t> emptyMaskCache;
                const int32_t cacheColumnsX = std::max(1, (localEndX - localStartX) * spanChunks);
                const int32_t cacheColumnsY = std::max(1, (localEndY - localStartY) * spanChunks);
                emptyMaskCache.reserve(static_cast<size_t>(cacheColumnsX * cacheColumnsY));

                for (int32_t y = localStartY; y < localEndY; ++y) {
                    for (int32_t x = localStartX; x < localEndX; ++x) {
                        for (int32_t z = 0; z < zCount; ++z) {
                            const ChunkCoord cellCoord{
                                baseCellX + x,
                                baseCellY + y,
                                z
                            };
                            if (isLodCellAllAir(cellCoord, lodLevel, emptyMaskCache)) {
                                continue;
                            }

                            std::vector<Meshlet> cellMeshlets = meshLodCell(cellCoord, lodLevel);
                            if (!cellMeshlets.empty()) {
                                meshlets.insert(
                                    meshlets.end(),
                                    std::make_move_iterator(cellMeshlets.begin()),
                                    std::make_move_iterator(cellMeshlets.end())
                                );
                            }
                        }
                    }
                }

                return MeshGenerationResult{
                    coord,
                    std::move(meshlets),
                    true
                };
            },
            [this, coord](jobsystem::JobResult<MeshGenerationResult>&& result) {
                if (!result.success()) {
                    std::unique_lock<std::shared_mutex> lock(meshMutex_);
                    pendingTileJobs_.erase(coord);
                    deferredRemeshTileLods_.erase(coord);
                    return;
                }

                MeshGenerationResult meshResult = std::move(result).value();
                if (!meshResult.meshed) {
                    std::unique_lock<std::shared_mutex> lock(meshMutex_);
                    pendingTileJobs_.erase(coord);
                    deferredRemeshTileLods_.erase(coord);
                    return;
                }

                std::unique_lock<std::shared_mutex> lock(meshMutex_);
                pendingTileJobs_.erase(coord);
                if (shuttingDown_.load(std::memory_order_acquire)) {
                    deferredRemeshTileLods_.erase(coord);
                    return;
                }

                const MeshTileCoord tileCoord = meshResult.coord.tileLod.tile;
                auto& completedForTile = completedTileResultsByTile_[tileCoord];
                completedForTile.push_back(CompletedTileCellResult{
                    meshResult.coord,
                    std::move(meshResult.meshlets)
                });
                if (completedTileResultQueued_.insert(tileCoord).second) {
                    completedTileResultOrder_.push_back(tileCoord);
                }
            }
        );
    } catch (const std::exception&) {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        pendingTileJobs_.erase(coord);
        deferredRemeshTileLods_.erase(coord);
    }
}
