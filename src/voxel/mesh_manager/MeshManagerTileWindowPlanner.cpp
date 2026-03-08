#include "solum_engine/voxel/MeshManager.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <utility>

#include "solum_engine/voxel/mesh_manager/TileFootprintUtils.h"
#include "solum_engine/voxel/World.h"

namespace {
constexpr std::size_t kWorldRevisionBatch = 2048u;

using mesh_manager::tile_footprint::FootprintDistanceRange;
using mesh_manager::tile_footprint::footprintDistanceRangeForCell;

bool hasRenderablePackedData(const std::shared_ptr<const PackedMeshletData>& packed) {
    return packed != nullptr && !packed->metadata.empty();
}

jobsystem::Priority demotePriority(jobsystem::Priority priority) {
    switch (priority) {
    case jobsystem::Priority::Critical:
        return jobsystem::Priority::High;
    case jobsystem::Priority::High:
        return jobsystem::Priority::Normal;
    case jobsystem::Priority::Normal:
        return jobsystem::Priority::Low;
    case jobsystem::Priority::Low:
    default:
        return jobsystem::Priority::Low;
    }
}

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
    return std::max<std::size_t>(24u, clampedWorkers * 3u);
}
}  // namespace

void MeshManager::scheduleTilesAround(const ChunkCoord& centerChunk,
                                      const glm::vec3& playerWorldPosition,
                                      float sseProjectionScale,
                                      const ChunkCoord* previousCenterChunk,
                                      int32_t centerShiftChunks) {
    struct ScheduledTileLod {
        TileLodCoord coord{};
        int32_t frontierDepth = std::numeric_limits<int32_t>::max();
        int32_t distanceSq = 0;
        uint8_t tier = 1u;
        bool forceRemesh = false;
        jobsystem::Priority priority = jobsystem::Priority::Low;
    };

    const int32_t maxRadiusChunks = std::max(0, maxConfiguredRadius());
    const int32_t prefetchChunks = std::max(4, std::max(0, centerShiftChunks));
    const int32_t scheduleOuterRadiusChunks = maxRadiusChunks + prefetchChunks;

    const int32_t minTileX = floor_div(
        centerChunk.v.x - scheduleOuterRadiusChunks - (meshTileSizeChunks_ - 1),
        meshTileSizeChunks_
    );
    const int32_t maxTileX = floor_div(centerChunk.v.x + scheduleOuterRadiusChunks, meshTileSizeChunks_);
    const int32_t minTileY = floor_div(
        centerChunk.v.y - scheduleOuterRadiusChunks - (meshTileSizeChunks_ - 1),
        meshTileSizeChunks_
    );
    const int32_t maxTileY = floor_div(centerChunk.v.y + scheduleOuterRadiusChunks, meshTileSizeChunks_);

    // Build a contiguous generated-tile frontier rooted near the camera so meshing expands
    // outward cleanly instead of jumping to disconnected generated islands.
    std::unordered_set<MeshTileCoord> generatedTiles;
    for (int32_t tileY = minTileY; tileY <= maxTileY; ++tileY) {
        for (int32_t tileX = minTileX; tileX <= maxTileX; ++tileX) {
            const MeshTileCoord tileCoord{tileX, tileY};
            if (isTileFootprintGenerated(tileCoord)) {
                generatedTiles.insert(tileCoord);
            }
        }
    }

    std::unordered_map<MeshTileCoord, int32_t> reachableTileDepth;
    if (!generatedTiles.empty()) {
        const MeshTileCoord centerTile{
            floor_div(centerChunk.v.x, meshTileSizeChunks_),
            floor_div(centerChunk.v.y, meshTileSizeChunks_)
        };

        MeshTileCoord seed = centerTile;
        if (generatedTiles.find(seed) == generatedTiles.end()) {
            int32_t bestDistance = std::numeric_limits<int32_t>::max();
            for (const MeshTileCoord& tile : generatedTiles) {
                const int32_t dx = std::abs(tile.x - centerTile.x);
                const int32_t dy = std::abs(tile.y - centerTile.y);
                const int32_t chebyshev = std::max(dx, dy);
                if (chebyshev < bestDistance) {
                    bestDistance = chebyshev;
                    seed = tile;
                }
            }
        }

        std::queue<std::pair<MeshTileCoord, int32_t>> frontier;
        frontier.push(std::make_pair(seed, 0));
        reachableTileDepth.emplace(seed, 0);

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
                if (generatedTiles.find(neighbor) == generatedTiles.end()) {
                    continue;
                }
                if (!reachableTileDepth.emplace(neighbor, depth + 1).second) {
                    continue;
                }
                frontier.push(std::make_pair(neighbor, depth + 1));
            }
        }
    }

    std::vector<ScheduledTileLod> jobsToSchedule;
    jobsToSchedule.reserve(
        static_cast<std::size_t>((maxTileX - minTileX + 1) * (maxTileY - minTileY + 1) * config_.lodLevelCount) *
        static_cast<std::size_t>(meshTileSliceCount_)
    );

    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);

        for (auto it = meshTiles_.begin(); it != meshTiles_.end();) {
            if (tileInBounds(it->first, minTileX, maxTileX, minTileY, maxTileY)) {
                ++it;
                continue;
            }

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

        const MeshTileCoord centerTile{
            floor_div(centerChunk.v.x, meshTileSizeChunks_),
            floor_div(centerChunk.v.y, meshTileSizeChunks_)
        };

        std::vector<MeshTileCoord> reachableOrdered;
        reachableOrdered.reserve(reachableTileDepth.size());
        for (const auto& [tileCoord, _] : reachableTileDepth) {
            reachableOrdered.push_back(tileCoord);
        }
        std::sort(reachableOrdered.begin(), reachableOrdered.end(), [&](const MeshTileCoord& a, const MeshTileCoord& b) {
            const int32_t aFrontierDepth = reachableTileDepth.at(a);
            const int32_t bFrontierDepth = reachableTileDepth.at(b);
            if (aFrontierDepth != bFrontierDepth) {
                return aFrontierDepth < bFrontierDepth;
            }
            const int32_t adx = std::abs(a.x - centerTile.x);
            const int32_t ady = std::abs(a.y - centerTile.y);
            const int32_t bdx = std::abs(b.x - centerTile.x);
            const int32_t bdy = std::abs(b.y - centerTile.y);
            const int32_t aDistance = std::max(adx, ady);
            const int32_t bDistance = std::max(bdx, bdy);
            if (aDistance != bDistance) {
                return aDistance < bDistance;
            }
            return a < b;
        });

        std::vector<MeshTileCoord> retainedInBoundsOrdered;
        retainedInBoundsOrdered.reserve(meshTiles_.size());
        for (const auto& [tileCoord, _] : meshTiles_) {
            if (!tileInBounds(tileCoord, minTileX, maxTileX, minTileY, maxTileY)) {
                continue;
            }
            if (reachableTileDepth.find(tileCoord) != reachableTileDepth.end()) {
                continue;
            }
            retainedInBoundsOrdered.push_back(tileCoord);
        }
        std::sort(retainedInBoundsOrdered.begin(), retainedInBoundsOrdered.end(), [&](const MeshTileCoord& a, const MeshTileCoord& b) {
            const int32_t adx = std::abs(a.x - centerTile.x);
            const int32_t ady = std::abs(a.y - centerTile.y);
            const int32_t bdx = std::abs(b.x - centerTile.x);
            const int32_t bdy = std::abs(b.y - centerTile.y);
            const int32_t aDistance = std::max(adx, ady);
            const int32_t bDistance = std::max(bdx, bdy);
            if (aDistance != bDistance) {
                return aDistance < bDistance;
            }
            return a < b;
        });

        std::vector<MeshTileCoord> tilesToProcess;
        tilesToProcess.reserve(reachableOrdered.size() + retainedInBoundsOrdered.size());
        tilesToProcess.insert(tilesToProcess.end(), reachableOrdered.begin(), reachableOrdered.end());
        tilesToProcess.insert(tilesToProcess.end(), retainedInBoundsOrdered.begin(), retainedInBoundsOrdered.end());

        for (const MeshTileCoord& tileCoord : tilesToProcess) {
            MeshTileState& tileState = meshTiles_[tileCoord];
            // Drop any stale high LODs that can overlap neighboring tiles.
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
                centerChunk,
                playerWorldPosition,
                sseProjectionScale,
                0
            );
            const int8_t desired = applyLodHysteresis(
                tileCoord,
                candidateDesired,
                previousDesired,
                playerWorldPosition,
                sseProjectionScale
            );
            tileState.desiredLod = desired;
            if (desired < 0) {
                continue;
            }

            const FootprintDistanceRange distances = footprintDistanceRangeForCell(
                tileCoord.x,
                tileCoord.y,
                meshTileSizeChunks_,
                centerChunk
            );
            const int32_t distanceChunks = distances.minDistanceChunks;
            const int32_t distanceSq = distanceChunks * distanceChunks;
            const auto frontierIt = reachableTileDepth.find(tileCoord);
            const int32_t frontierDepth = (frontierIt != reachableTileDepth.end())
                ? frontierIt->second
                : std::numeric_limits<int32_t>::max();

            const jobsystem::Priority primaryPriority =
                primaryPriorityForDistance(distanceChunks, meshTileSizeChunks_);
            const jobsystem::Priority secondaryPriority = demotePriority(primaryPriority);

            bool desiredNeedsRepair = false;
            const auto desiredLodIt = tileState.lodStates.find(static_cast<uint8_t>(desired));
            if (desiredLodIt != tileState.lodStates.end()) {
                bool desiredAnyResident = false;
                bool desiredAnyRenderable = false;
                for (int32_t zSlice = 0; zSlice < meshTileSliceCount_; ++zSlice) {
                    const auto sliceIt = desiredLodIt->second.find(zSlice);
                    if (sliceIt == desiredLodIt->second.end() || !sliceIt->second.resident) {
                        continue;
                    }
                    desiredAnyResident = true;
                    if (hasRenderablePackedData(sliceIt->second.culledPacked) ||
                        hasRenderablePackedData(sliceIt->second.doubleSidedPacked)) {
                        desiredAnyRenderable = true;
                    }
                }

                bool hasNonEmptyAlternateLod = false;
                for (const auto& [lodLevel, lodSlices] : tileState.lodStates) {
                    if (lodLevel == static_cast<uint8_t>(desired)) {
                        continue;
                    }
                    for (const auto& [_, lodState] : lodSlices) {
                        if (lodState.resident &&
                            (hasRenderablePackedData(lodState.culledPacked) ||
                             hasRenderablePackedData(lodState.doubleSidedPacked))) {
                            hasNonEmptyAlternateLod = true;
                            break;
                        }
                    }
                    if (hasNonEmptyAlternateLod) {
                        break;
                    }
                }

                if (desiredAnyResident && !desiredAnyRenderable && hasNonEmptyAlternateLod) {
                    constexpr uint64_t kEmptyLodRepairCooldownRevisions = 32u;
                    const uint64_t currentRevision = meshRevision_.load(std::memory_order_acquire);
                    uint64_t oldestDesiredRevision = currentRevision;
                    bool hasDesiredRevision = false;
                    for (const auto& [_, lodState] : desiredLodIt->second) {
                        if (!lodState.resident) {
                            continue;
                        }
                        oldestDesiredRevision = hasDesiredRevision
                            ? std::min(oldestDesiredRevision, lodState.revision)
                            : lodState.revision;
                        hasDesiredRevision = true;
                    }
                    if (hasDesiredRevision) {
                        desiredNeedsRepair =
                            currentRevision >= (oldestDesiredRevision + kEmptyLodRepairCooldownRevisions);
                    }
                }
            }

            // Primary job: currently best-fit LOD for this tile.
            for (int32_t zSlice = 0; zSlice < meshTileSliceCount_; ++zSlice) {
                jobsToSchedule.push_back(ScheduledTileLod{
                    TileLodCoord{MeshTileSliceCoord{tileCoord, zSlice}, static_cast<uint8_t>(desired)},
                    frontierDepth,
                    distanceSq,
                    0u,
                    desiredNeedsRepair,
                    primaryPriority
                });
            }

            std::vector<int32_t> supplementalLods;
            supplementalLods.reserve(static_cast<std::size_t>(config_.lodLevelCount - 1));
            for (int32_t lod = 0; lod < config_.lodLevelCount; ++lod) {
                if (lod == desired) {
                    continue;
                }
                supplementalLods.push_back(lod);
            }
            std::stable_sort(
                supplementalLods.begin(),
                supplementalLods.end(),
                [desired](int32_t a, int32_t b) {
                    const int32_t ad = std::abs(a - desired);
                    const int32_t bd = std::abs(b - desired);
                    if (ad != bd) {
                        return ad < bd;
                    }
                    // Prefer coarser supplemental LODs first.
                    return a > b;
                }
            );

            for (int32_t lod : supplementalLods) {
                for (int32_t zSlice = 0; zSlice < meshTileSliceCount_; ++zSlice) {
                    jobsToSchedule.push_back(ScheduledTileLod{
                        TileLodCoord{MeshTileSliceCoord{tileCoord, zSlice}, static_cast<uint8_t>(lod)},
                        frontierDepth,
                        distanceSq,
                        1u,
                        false,
                        secondaryPriority
                    });
                }
            }
        }

        if (refreshSelectedLodsLocked()) {
            selectionSnapshotDirty_ = true;
        }
    }

    std::stable_sort(jobsToSchedule.begin(), jobsToSchedule.end(), [&](const ScheduledTileLod& a, const ScheduledTileLod& b) {
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
            scheduled.forceRemesh,
            prefetchChunks + meshTileSizeChunks_
        );

        std::shared_lock<std::shared_mutex> lock(meshMutex_);
        pendingCount = pendingTileLodJobs_.size() + pendingPriorityTileLodJobs_.size();
    }

    (void)previousCenterChunk;
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
            const int8_t selected = (tileIt != meshTiles_.end()) ? tileIt->second.selectedLod : -1;
            const int8_t targetLod = (selected >= 0) ? selected : desired;
            if (targetLod < 0) {
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

            for (int32_t zSlice = 0; zSlice < meshTileSliceCount_; ++zSlice) {
                jobsToSchedule.push_back(ScheduledTileLod{
                    TileLodCoord{MeshTileSliceCoord{tile, zSlice}, static_cast<uint8_t>(targetLod)},
                    frontierDepth,
                    distanceSq,
                    0u,
                    primaryPriority
                });
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
}

