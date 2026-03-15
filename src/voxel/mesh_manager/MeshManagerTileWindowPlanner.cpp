#include "solum_engine/voxel/MeshManager.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <mutex>
#include <queue>
#include <utility>

#include "solum_engine/voxel/World.h"
#include "solum_engine/voxel/mesh_manager/TileFootprintUtils.h"

namespace {
constexpr std::size_t kWorldRevisionBatch = 2048u;

using mesh_manager::tile_footprint::FootprintDistanceRange;
using mesh_manager::tile_footprint::footprintDistanceRangeForCell;

jobsystem::Priority primaryPriorityForDistance(int32_t distanceChunks, int32_t tileSizeChunks) {
    const int32_t tileDistance = std::max(1, tileSizeChunks);
    if (distanceChunks <= tileDistance) {
        return jobsystem::Priority::Critical;
    }
    if (distanceChunks <= (tileDistance * 2)) {
        return jobsystem::Priority::High;
    }
    return jobsystem::Priority::Normal;
}

std::size_t maxPendingMeshJobs(std::size_t workerCount) {
    const std::size_t clampedWorkers = std::max<std::size_t>(workerCount, 1u);
    return std::max<std::size_t>(48u, clampedWorkers * 6u);
}
}  // namespace

void MeshManager::scheduleTilesAround(const ChunkCoord& centerChunk,
                                      const glm::vec3& playerWorldPosition,
                                      float sseProjectionScale,
                                      int32_t centerShiftChunks) {
    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        resetTileQueuesLocked(centerChunk, playerWorldPosition, sseProjectionScale, centerShiftChunks);
    }
    pumpTileQueues();
}

void MeshManager::resetTileQueuesLocked(const ChunkCoord& centerChunk,
                                        const glm::vec3& playerWorldPosition,
                                        float sseProjectionScale,
                                        int32_t centerShiftChunks) {
    const int32_t maxRadiusChunks = std::max(0, maxConfiguredRadius());
    planningPrefetchChunks_ = std::max(4, std::max(0, centerShiftChunks));
    const int32_t scheduleOuterRadiusChunks = maxRadiusChunks + planningPrefetchChunks_;

    planningMinTileX_ = floor_div(
        centerChunk.v.x - scheduleOuterRadiusChunks - (meshTileSizeChunks_ - 1),
        meshTileSizeChunks_
    );
    planningMaxTileX_ = floor_div(centerChunk.v.x + scheduleOuterRadiusChunks, meshTileSizeChunks_);
    planningMinTileY_ = floor_div(
        centerChunk.v.y - scheduleOuterRadiusChunks - (meshTileSizeChunks_ - 1),
        meshTileSizeChunks_
    );
    planningMaxTileY_ = floor_div(centerChunk.v.y + scheduleOuterRadiusChunks, meshTileSizeChunks_);
    currentCenterTile_ = MeshTileCoord{
        floor_div(centerChunk.v.x, meshTileSizeChunks_),
        floor_div(centerChunk.v.y, meshTileSizeChunks_)
    };
    maxVisibleFrontierRing_ = std::max(
        std::max(std::abs(currentCenterTile_.x - planningMinTileX_), std::abs(currentCenterTile_.x - planningMaxTileX_)),
        std::max(std::abs(currentCenterTile_.y - planningMinTileY_), std::abs(currentCenterTile_.y - planningMaxTileY_))
    );

    ++tileQueueCenterVersion_;
    ++tileQueueSequence_;
    visibleFrontierRing_ = 0;
    currentVisibleRingInitialized_ = false;
    currentVisibleRingOutstandingTiles_.clear();
    queuedVisibleTiles_.clear();
    waitingVisibleTiles_.clear();
    queuedVisibleTileHeap_ = decltype(queuedVisibleTileHeap_){};

    pruneMeshTilesOutsideWindowLocked();

    for (auto& [tileCoord, tileState] : meshTiles_) {
        if (!tileInBounds(tileCoord, planningMinTileX_, planningMaxTileX_, planningMinTileY_, planningMaxTileY_)) {
            continue;
        }
        tileState.queuedVisibleLod = -1;
        tileState.pendingVisibleSlices = 0u;
        tileState.waitingForVisibleFootprint = false;

        const int8_t previousDesired = tileState.desiredLod;
        const int8_t candidateDesired = desiredLodForTile(
            tileCoord,
            centerChunk,
            playerWorldPosition,
            sseProjectionScale,
            0
        );
        tileState.desiredLod = applyLodHysteresis(
            tileCoord,
            candidateDesired,
            previousDesired,
            playerWorldPosition,
            sseProjectionScale
        );
        if (refreshSelectedLodLocked(tileCoord, tileState)) {
            selectionSnapshotDirty_ = true;
        }
    }
}

void MeshManager::pruneMeshTilesOutsideWindowLocked() {
    for (auto it = meshTiles_.begin(); it != meshTiles_.end();) {
        if (tileInBounds(it->first, planningMinTileX_, planningMaxTileX_, planningMinTileY_, planningMaxTileY_)) {
            ++it;
            continue;
        }

        queuedVisibleTiles_.erase(it->first);
        waitingVisibleTiles_.erase(it->first);
        currentVisibleRingOutstandingTiles_.erase(it->first);

        for (const auto& [lod, slices] : it->second.lodStates) {
            for (const auto& [zSlice, lodState] : slices) {
                if (!lodState.resident) {
                    continue;
                }
                queueTileLodRemovalLocked(MeshTileLodKey{
                    MeshTileSliceCoord{it->first, zSlice},
                    lod
                });
            }
        }
        it = meshTiles_.erase(it);
        selectionSnapshotDirty_ = true;
    }
}

void MeshManager::ensureVisibleFrontierLocked() {
    while (visibleFrontierRing_ <= maxVisibleFrontierRing_) {
        if (!currentVisibleRingInitialized_) {
            currentVisibleRingOutstandingTiles_.clear();
            initializeVisibleRingLocked(visibleFrontierRing_);
            currentVisibleRingInitialized_ = true;
        }

        if (!currentVisibleRingOutstandingTiles_.empty()) {
            return;
        }

        ++visibleFrontierRing_;
        currentVisibleRingInitialized_ = false;
    }
}

bool MeshManager::initializeVisibleRingLocked(int32_t ring) {
    bool anyTilesInRing = false;

    auto processTile = [&](const MeshTileCoord& tileCoord) {
        if (!tileInBounds(tileCoord, planningMinTileX_, planningMaxTileX_, planningMinTileY_, planningMaxTileY_)) {
            return;
        }

        anyTilesInRing = true;
        MeshTileState& tileState = meshTiles_[tileCoord];
        tileState.queuedVisibleLod = -1;
        tileState.pendingVisibleSlices = 0u;
        tileState.waitingForVisibleFootprint = false;

        for (auto lodIt = tileState.lodStates.begin(); lodIt != tileState.lodStates.end();) {
            if (static_cast<int32_t>(lodIt->first) >= config_.lodLevelCount) {
                for (const auto& [zSlice, lodState] : lodIt->second) {
                    if (!lodState.resident) {
                        continue;
                    }
                    queueTileLodRemovalLocked(MeshTileLodKey{
                        MeshTileSliceCoord{tileCoord, zSlice},
                        lodIt->first
                    });
                }
                lodIt = tileState.lodStates.erase(lodIt);
            } else {
                ++lodIt;
            }
        }

        const int8_t previousDesired = tileState.desiredLod;
        const int8_t candidateDesired = desiredLodForTile(
            tileCoord,
            lastScheduledCenterChunk_,
            lastPlayerWorldPosition_,
            lastSseProjectionScale_,
            0
        );
        const int8_t desired = applyLodHysteresis(
            tileCoord,
            candidateDesired,
            previousDesired,
            lastPlayerWorldPosition_,
            lastSseProjectionScale_
        );
        tileState.desiredLod = desired;
        if (refreshSelectedLodLocked(tileCoord, tileState)) {
            selectionSnapshotDirty_ = true;
        }
        if (desired < 0) {
            return;
        }

        if (isTileDisplayReadyLocked(tileCoord, tileState)) {
            return;
        }

        currentVisibleRingOutstandingTiles_.insert(tileCoord);

        const uint16_t pendingTile = pendingSliceCountForTileLocked(tileCoord);
        if (pendingTile > 0u) {
            tileState.queuedVisibleLod = desired;
            tileState.pendingVisibleSlices = pendingTile;
            return;
        }

        if (!isTileFootprintGenerated(tileCoord)) {
            tileState.waitingForVisibleFootprint = true;
            waitingVisibleTiles_.insert(tileCoord);
            return;
        }

        const FootprintDistanceRange distances = footprintDistanceRangeForCell(
            tileCoord.x,
            tileCoord.y,
            meshTileSizeChunks_,
            lastScheduledCenterChunk_
        );
        const int32_t distanceChunks = distances.minDistanceChunks;
        enqueueVisibleTileLocked(tileCoord, ring, distanceChunks * distanceChunks);
    };

    if (ring == 0) {
        processTile(currentCenterTile_);
        return anyTilesInRing;
    }

    const int32_t ringMinX = currentCenterTile_.x - ring;
    const int32_t ringMaxX = currentCenterTile_.x + ring;
    const int32_t ringMinY = currentCenterTile_.y - ring;
    const int32_t ringMaxY = currentCenterTile_.y + ring;

    for (int32_t x = ringMinX; x <= ringMaxX; ++x) {
        processTile(MeshTileCoord{x, ringMinY});
    }
    for (int32_t x = ringMinX; x <= ringMaxX; ++x) {
        processTile(MeshTileCoord{x, ringMaxY});
    }
    for (int32_t y = ringMinY + 1; y < ringMaxY; ++y) {
        processTile(MeshTileCoord{ringMinX, y});
    }
    for (int32_t y = ringMinY + 1; y < ringMaxY; ++y) {
        processTile(MeshTileCoord{ringMaxX, y});
    }

    return anyTilesInRing;
}

void MeshManager::enqueueVisibleTileLocked(const MeshTileCoord& tile, int32_t ring, int32_t distanceSq) {
    if (!queuedVisibleTiles_.insert(tile).second) {
        return;
    }

    MeshTileState& tileState = meshTiles_[tile];
    tileState.waitingForVisibleFootprint = false;
    waitingVisibleTiles_.erase(tile);
    queuedVisibleTileHeap_.push(QueuedVisibleTileEntry{
        tile,
        ring,
        distanceSq,
        tileQueueCenterVersion_,
        tileQueueSequence_++
    });
}

void MeshManager::markVisibleTileReadyLocked(const MeshTileCoord& tile) {
    currentVisibleRingOutstandingTiles_.erase(tile);
    queuedVisibleTiles_.erase(tile);
    waitingVisibleTiles_.erase(tile);

    auto tileIt = meshTiles_.find(tile);
    if (tileIt == meshTiles_.end()) {
        return;
    }

    tileIt->second.queuedVisibleLod = -1;
    tileIt->second.pendingVisibleSlices = 0u;
    tileIt->second.waitingForVisibleFootprint = false;
}

void MeshManager::noteVisibleTileAttemptFinishedLocked(const MeshTileCoord& tile) {
    auto tileIt = meshTiles_.find(tile);
    if (tileIt == meshTiles_.end()) {
        return;
    }

    MeshTileState& tileState = tileIt->second;
    if (tileState.queuedVisibleLod < 0) {
        return;
    }

    tileState.queuedVisibleLod = -1;
    tileState.pendingVisibleSlices = 0u;

    if (currentVisibleRingOutstandingTiles_.find(tile) == currentVisibleRingOutstandingTiles_.end()) {
        tileState.waitingForVisibleFootprint = false;
        waitingVisibleTiles_.erase(tile);
        return;
    }

    if (tileState.desiredLod < 0 || isTileDisplayReadyLocked(tile, tileState)) {
        markVisibleTileReadyLocked(tile);
        return;
    }

    const uint16_t pendingTile = pendingSliceCountForTileLocked(tile);
    if (pendingTile > 0u) {
        tileState.queuedVisibleLod = tileState.desiredLod;
        tileState.pendingVisibleSlices = pendingTile;
        tileState.waitingForVisibleFootprint = false;
        waitingVisibleTiles_.erase(tile);
        return;
    }

    if (!isTileFootprintGenerated(tile)) {
        tileState.waitingForVisibleFootprint = true;
        waitingVisibleTiles_.insert(tile);
        return;
    }

    const int32_t ring = std::max(std::abs(tile.x - currentCenterTile_.x), std::abs(tile.y - currentCenterTile_.y));
    const FootprintDistanceRange distances = footprintDistanceRangeForCell(
        tile.x,
        tile.y,
        meshTileSizeChunks_,
        lastScheduledCenterChunk_
    );
    const int32_t distanceChunks = distances.minDistanceChunks;
    enqueueVisibleTileLocked(tile, ring, distanceChunks * distanceChunks);
}

void MeshManager::wakeVisibleTilesForGeneratedColumns(const std::vector<ColumnCoord>& generatedColumns) {
    std::unordered_set<MeshTileCoord> candidateTiles;
    candidateTiles.reserve(generatedColumns.size());
    for (const ColumnCoord& coord : generatedColumns) {
        candidateTiles.insert(MeshTileCoord{
            floor_div(coord.v.x, meshTileSizeChunks_),
            floor_div(coord.v.y, meshTileSizeChunks_)
        });
    }

    std::unique_lock<std::shared_mutex> lock(meshMutex_);
    for (const MeshTileCoord& tile : candidateTiles) {
        if (!tileInBounds(tile, planningMinTileX_, planningMaxTileX_, planningMinTileY_, planningMaxTileY_)) {
            continue;
        }
        if (currentVisibleRingOutstandingTiles_.find(tile) == currentVisibleRingOutstandingTiles_.end()) {
            continue;
        }
        if (queuedVisibleTiles_.find(tile) != queuedVisibleTiles_.end()) {
            continue;
        }

        auto tileIt = meshTiles_.find(tile);
        if (tileIt == meshTiles_.end() || tileIt->second.desiredLod < 0) {
            continue;
        }
        if (pendingSliceCountForTileLocked(tile) > 0u) {
            tileIt->second.queuedVisibleLod = tileIt->second.desiredLod;
            continue;
        }
        if (!isTileFootprintGenerated(tile)) {
            continue;
        }

        const int32_t ring = std::max(std::abs(tile.x - currentCenterTile_.x), std::abs(tile.y - currentCenterTile_.y));
        const FootprintDistanceRange distances = footprintDistanceRangeForCell(
            tile.x,
            tile.y,
            meshTileSizeChunks_,
            lastScheduledCenterChunk_
        );
        const int32_t distanceChunks = distances.minDistanceChunks;
        enqueueVisibleTileLocked(tile, ring, distanceChunks * distanceChunks);
    }
}

void MeshManager::pumpTileQueues() {
    struct PendingDispatch {
        TileLodCoord coord{};
        jobsystem::Priority priority = jobsystem::Priority::Low;
        bool forceRemesh = false;
    };

    std::vector<PendingDispatch> dispatches;
    std::vector<MeshTileCoord> visibleAttempts;
    bool repump = false;
    const int32_t activeWindowExtraChunks = planningPrefetchChunks_ + meshTileSizeChunks_;

    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        ensureVisibleFrontierLocked();

        std::size_t pendingCount = pendingTileLodJobs_.size() + pendingPriorityTileLodJobs_.size();
        const std::size_t maxPending = maxPendingMeshJobs(jobs_.worker_count());
        std::size_t remainingBudget = (pendingCount < maxPending) ? (maxPending - pendingCount) : 0u;

        while (remainingBudget > 0u) {
            ensureVisibleFrontierLocked();
            bool queuedVisibleWork = false;
            while (remainingBudget > 0u && !queuedVisibleTileHeap_.empty()) {
                const QueuedVisibleTileEntry entry = queuedVisibleTileHeap_.top();
                queuedVisibleTileHeap_.pop();

                if (entry.centerVersion != tileQueueCenterVersion_) {
                    continue;
                }
                if (queuedVisibleTiles_.erase(entry.tile) == 0u) {
                    continue;
                }
                if (currentVisibleRingOutstandingTiles_.find(entry.tile) == currentVisibleRingOutstandingTiles_.end()) {
                    continue;
                }

                auto tileIt = meshTiles_.find(entry.tile);
                if (tileIt == meshTiles_.end()) {
                    currentVisibleRingOutstandingTiles_.erase(entry.tile);
                    continue;
                }

                MeshTileState& tileState = tileIt->second;
                if (tileState.desiredLod < 0) {
                    markVisibleTileReadyLocked(entry.tile);
                    repump = true;
                    continue;
                }
                if (isTileDisplayReadyLocked(entry.tile, tileState)) {
                    markVisibleTileReadyLocked(entry.tile);
                    repump = true;
                    continue;
                }

                const uint16_t pendingTile = pendingSliceCountForTileLocked(entry.tile);
                if (pendingTile > 0u) {
                    tileState.queuedVisibleLod = tileState.desiredLod;
                    tileState.pendingVisibleSlices = pendingTile;
                    continue;
                }

                if (!isTileFootprintGenerated(entry.tile)) {
                    tileState.waitingForVisibleFootprint = true;
                    waitingVisibleTiles_.insert(entry.tile);
                    continue;
                }

                const FootprintDistanceRange distances = footprintDistanceRangeForCell(
                    entry.tile.x,
                    entry.tile.y,
                    meshTileSizeChunks_,
                    lastScheduledCenterChunk_
                );
                const int32_t distanceChunks = distances.minDistanceChunks;
                const jobsystem::Priority priority = primaryPriorityForDistance(distanceChunks, meshTileSizeChunks_);

                std::size_t beforeDispatchCount = dispatches.size();
                for (int32_t lod = 0; lod < config_.lodLevelCount; ++lod) {
                    const auto lodIt = tileState.lodStates.find(static_cast<uint8_t>(lod));
                    for (int32_t zSlice = 0; zSlice < meshTileSliceCount_; ++zSlice) {
                        const bool resident = lodIt != tileState.lodStates.end() &&
                            lodIt->second.contains(zSlice) &&
                            lodIt->second.at(zSlice).resident;
                        if (resident) {
                            continue;
                        }
                        dispatches.push_back(PendingDispatch{
                            TileLodCoord{MeshTileSliceCoord{entry.tile, zSlice}, static_cast<uint8_t>(lod)},
                            priority,
                            false
                        });
                    }
                }

                const std::size_t added = dispatches.size() - beforeDispatchCount;
                if (added == 0u) {
                    if (isTileDisplayReadyLocked(entry.tile, tileState)) {
                        markVisibleTileReadyLocked(entry.tile);
                    } else {
                        noteVisibleTileAttemptFinishedLocked(entry.tile);
                    }
                    repump = true;
                    continue;
                }

                tileState.queuedVisibleLod = tileState.desiredLod;
                tileState.pendingVisibleSlices = 0u;
                tileState.waitingForVisibleFootprint = false;
                waitingVisibleTiles_.erase(entry.tile);
                visibleAttempts.push_back(entry.tile);
                queuedVisibleWork = true;
                remainingBudget = (added >= remainingBudget) ? 0u : (remainingBudget - added);
            }

            if (!queuedVisibleWork) {
                break;
            }
        }
    }

    for (const PendingDispatch& dispatch : dispatches) {
        scheduleTileLodMeshing(
            dispatch.coord,
            dispatch.priority,
            dispatch.forceRemesh,
            activeWindowExtraChunks
        );
    }

    if (!visibleAttempts.empty()) {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        for (const MeshTileCoord& tile : visibleAttempts) {
            auto tileIt = meshTiles_.find(tile);
            if (tileIt == meshTiles_.end() || tileIt->second.queuedVisibleLod < 0) {
                continue;
            }

            const uint16_t pending = pendingSliceCountForTileLocked(tile);
            tileIt->second.pendingVisibleSlices = pending;
            if (pending > 0u) {
                continue;
            }

            if (isTileDisplayReadyLocked(tile, tileIt->second)) {
                markVisibleTileReadyLocked(tile);
            } else {
                noteVisibleTileAttemptFinishedLocked(tile);
            }
            repump = true;
        }
    }

    if (repump) {
        pumpTileQueues();
    }
}

void MeshManager::scheduleRemeshForNewColumns(const ColumnCoord& centerColumn) {
    struct ScheduledTileLod {
        TileLodCoord coord{};
        int32_t frontierDepth = std::numeric_limits<int32_t>::max();
        int32_t distanceSq = 0;
        uint8_t tier = 1u;
        jobsystem::Priority priority = jobsystem::Priority::Low;
    };

    const uint64_t processedRevision = processedWorldGenerationRevision_.load(std::memory_order_acquire);
    std::vector<ColumnCoord> generatedColumns;
    const uint64_t nextRevision = world_.copyGeneratedColumnsSince(
        processedRevision,
        generatedColumns,
        kWorldRevisionBatch
    );
    if (nextRevision == processedRevision) {
        return;
    }
    processedWorldGenerationRevision_.store(nextRevision, std::memory_order_release);

    if (generatedColumns.empty()) {
        return;
    }

    wakeVisibleTilesForGeneratedColumns(generatedColumns);

    std::unordered_set<MeshTileCoord> tilesToRemesh;
    const int32_t remeshRadius = std::max(0, maxConfiguredRadius() + meshTileSizeChunks_ + 4);

    for (const ColumnCoord& coord : generatedColumns) {
        const int32_t dx = std::abs(coord.v.x - centerColumn.v.x);
        const int32_t dy = std::abs(coord.v.y - centerColumn.v.y);
        if (dx > remeshRadius || dy > remeshRadius) {
            continue;
        }

        const int32_t tileX = floor_div(coord.v.x, meshTileSizeChunks_);
        const int32_t tileY = floor_div(coord.v.y, meshTileSizeChunks_);
        for (int32_t oy = -1; oy <= 1; ++oy) {
            for (int32_t ox = -1; ox <= 1; ++ox) {
                tilesToRemesh.insert(MeshTileCoord{tileX + ox, tileY + oy});
            }
        }
    }

    std::vector<ScheduledTileLod> jobsToSchedule;
    jobsToSchedule.reserve(
        tilesToRemesh.size() *
        static_cast<std::size_t>(config_.lodLevelCount) *
        static_cast<std::size_t>(meshTileSliceCount_)
    );

    const ChunkCoord centerChunk = hasLastScheduledCenter_
        ? lastScheduledCenterChunk_
        : ChunkCoord{centerColumn.v.x, centerColumn.v.y, 0};

    const glm::vec3 playerWorldPosition = lastPlayerWorldPosition_;
    const float sseProjectionScale = hasLastSseProjectionScale_
        ? lastSseProjectionScale_
        : config_.lodSseFallbackProjectionScale;

    std::unordered_map<MeshTileCoord, int32_t> remeshDepth;
    if (!tilesToRemesh.empty()) {
        MeshTileCoord seed{};
        bool hasSeed = false;
        int32_t bestDistance = std::numeric_limits<int32_t>::max();
        for (const MeshTileCoord& tile : tilesToRemesh) {
            const int32_t dx = std::abs(tile.x - floor_div(centerChunk.v.x, meshTileSizeChunks_));
            const int32_t dy = std::abs(tile.y - floor_div(centerChunk.v.y, meshTileSizeChunks_));
            const int32_t distance = std::max(dx, dy);
            if (!hasSeed || distance < bestDistance) {
                hasSeed = true;
                bestDistance = distance;
                seed = tile;
            }
        }

        if (hasSeed) {
            std::queue<std::pair<MeshTileCoord, int32_t>> frontier;
            frontier.push(std::make_pair(seed, 0));
            remeshDepth.emplace(seed, 0);

            static constexpr std::array<glm::ivec2, 4> kCardinalNeighbors{
                glm::ivec2{1, 0},
                glm::ivec2{-1, 0},
                glm::ivec2{0, 1},
                glm::ivec2{0, -1}
            };

            while (!frontier.empty()) {
                const auto [current, depth] = frontier.front();
                frontier.pop();

                for (const glm::ivec2& n : kCardinalNeighbors) {
                    const MeshTileCoord neighbor{current.x + n.x, current.y + n.y};
                    if (tilesToRemesh.find(neighbor) == tilesToRemesh.end()) {
                        continue;
                    }
                    if (!remeshDepth.emplace(neighbor, depth + 1).second) {
                        continue;
                    }
                    frontier.push(std::make_pair(neighbor, depth + 1));
                }
            }
        }
    }

    {
        std::shared_lock<std::shared_mutex> lock(meshMutex_);
        for (const MeshTileCoord& tile : tilesToRemesh) {
            const auto tileIt = meshTiles_.find(tile);

            const int8_t desired = (tileIt != meshTiles_.end() && tileIt->second.desiredLod >= 0)
                ? tileIt->second.desiredLod
                : desiredLodForTile(tile, centerChunk, playerWorldPosition, sseProjectionScale, 0);
            if (desired < 0) {
                continue;
            }

            const FootprintDistanceRange distances = footprintDistanceRangeForCell(
                tile.x,
                tile.y,
                meshTileSizeChunks_,
                centerChunk
            );
            const int32_t distanceChunks = distances.minDistanceChunks;
            const int32_t distanceSq = distanceChunks * distanceChunks;
            const int32_t frontierDepth = remeshDepth.contains(tile)
                ? remeshDepth.at(tile)
                : std::numeric_limits<int32_t>::max();

            const jobsystem::Priority primaryPriority =
                primaryPriorityForDistance(distanceChunks, meshTileSizeChunks_);

            for (int32_t lod = 0; lod < config_.lodLevelCount; ++lod) {
                for (int32_t zSlice = 0; zSlice < meshTileSliceCount_; ++zSlice) {
                    jobsToSchedule.push_back(ScheduledTileLod{
                        TileLodCoord{MeshTileSliceCoord{tile, zSlice}, static_cast<uint8_t>(lod)},
                        frontierDepth,
                        distanceSq,
                        0u,
                        primaryPriority
                    });
                }
            }
        }
    }

    std::stable_sort(jobsToSchedule.begin(), jobsToSchedule.end(), [](const ScheduledTileLod& a, const ScheduledTileLod& b) {
        if (a.frontierDepth != b.frontierDepth) {
            return a.frontierDepth < b.frontierDepth;
        }
        if (a.distanceSq != b.distanceSq) {
            return a.distanceSq < b.distanceSq;
        }
        if (a.tier != b.tier) {
            return a.tier < b.tier;
        }
        if (!(a.coord.tile == b.coord.tile)) {
            return a.coord.tile < b.coord.tile;
        }
        return a.coord.lodLevel < b.coord.lodLevel;
    });

    std::size_t pendingCount = 0u;
    {
        std::shared_lock<std::shared_mutex> lock(meshMutex_);
        pendingCount = pendingTileLodJobs_.size() + pendingPriorityTileLodJobs_.size();
    }
    const std::size_t maxPending = maxPendingMeshJobs(jobs_.worker_count());
    std::size_t jobIndex = 0u;
    while (jobIndex < jobsToSchedule.size() && pendingCount < maxPending) {
        const ScheduledTileLod& scheduled = jobsToSchedule[jobIndex++];
        scheduleTileLodMeshing(
            scheduled.coord,
            scheduled.priority,
            true,
            meshTileSizeChunks_ + 4
        );

        std::shared_lock<std::shared_mutex> lock(meshMutex_);
        pendingCount = pendingTileLodJobs_.size() + pendingPriorityTileLodJobs_.size();
    }

    pumpTileQueues();
}
