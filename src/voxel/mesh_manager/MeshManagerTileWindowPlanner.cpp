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
        refreshSelectedLodLocked(tileCoord, tileState);
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
        if (it->second.selectedLod >= 0) {
            for (int32_t zSlice = 0; zSlice < meshTileSliceCount_; ++zSlice) {
                pendingSelectionChanges_[MeshTileSliceCoord{it->first, zSlice}] = -1;
            }
        }
        it = meshTiles_.erase(it);
    }
}

void MeshManager::ensureVisibleFrontierLocked() {
    while (visibleFrontierRing_ <= maxVisibleFrontierRing_) {
        if (!currentVisibleRingInitialized_) {
            // Don't clear outstanding tiles — keep waiting tiles from previous
            // rings so they can still be woken by wakeVisibleTilesForGeneratedColumns.
            initializeVisibleRingLocked(visibleFrontierRing_);
            currentVisibleRingInitialized_ = true;
        }

        // Only block frontier advancement on tiles that are actively being
        // dispatched or queued — NOT on tiles merely waiting for their
        // footprint to be generated.  This prevents a single tile with an
        // incomplete footprint (e.g. at the world-generation boundary) from
        // starving every tile in every outer ring.
        bool hasActiveOutstanding = false;
        for (const auto& tile : currentVisibleRingOutstandingTiles_) {
            if (!waitingVisibleTiles_.contains(tile)) {
                hasActiveOutstanding = true;
                break;
            }
        }

        if (hasActiveOutstanding) {
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
        refreshSelectedLodLocked(tileCoord, tileState);
        if (desired < 0) {
            return;
        }

        if (isTileDisplayReadyLocked(tileCoord, tileState)) {
            return;
        }

        currentVisibleRingOutstandingTiles_.insert(tileCoord);
        advanceVisibleTileLocked(tileCoord, false, nullptr, nullptr, nullptr);
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
}

void MeshManager::noteVisibleTileAttemptFinishedLocked(const MeshTileCoord& tile) {
    if (!currentVisibleRingOutstandingTiles_.contains(tile)) {
        queuedVisibleTiles_.erase(tile);
        waitingVisibleTiles_.erase(tile);
        return;
    }

    advanceVisibleTileLocked(tile, false, nullptr, nullptr, nullptr);
}

void MeshManager::waitVisibleTileForFootprintLocked(const MeshTileCoord& tile) {
    queuedVisibleTiles_.erase(tile);
    waitingVisibleTiles_.insert(tile);
}

void MeshManager::advanceVisibleTileLocked(const MeshTileCoord& tile,
                                          bool dispatchNow,
                                          std::vector<PendingMeshDispatch>* dispatches,
                                          std::vector<MeshTileCoord>* dispatchedTiles,
                                          bool* repump) {
    auto tileIt = meshTiles_.find(tile);
    if (tileIt == meshTiles_.end()) {
        markVisibleTileReadyLocked(tile);
        return;
    }

    MeshTileState& tileState = tileIt->second;
    if (tileState.desiredLod < 0 || isTileDisplayReadyLocked(tile, tileState)) {
        markVisibleTileReadyLocked(tile);
        if (repump != nullptr) {
            *repump = true;
        }
        return;
    }

    if (!isTileFootprintGenerated(tile)) {
        waitVisibleTileForFootprintLocked(tile);
        return;
    }

    waitingVisibleTiles_.erase(tile);
    for (int32_t lod = 0; lod < config_.lodLevelCount; ++lod) {
        if (pendingSliceCountForLodLocked(tile, static_cast<uint8_t>(lod)) > 0u) {
            return;
        }
    }

    if (!dispatchNow) {
        enqueueVisibleTileLocked(tile, visibleRingForTileLocked(tile), visibleDistanceSqForTileLocked(tile));
        return;
    }

    const FootprintDistanceRange distances = footprintDistanceRangeForCell(
        tile.x,
        tile.y,
        meshTileSizeChunks_,
        lastScheduledCenterChunk_
    );
    const jobsystem::Priority priority =
        primaryPriorityForDistance(distances.minDistanceChunks, meshTileSizeChunks_);

    std::size_t beforeDispatchCount = (dispatches != nullptr) ? dispatches->size() : 0u;
    for (int32_t lod = 0; lod < config_.lodLevelCount; ++lod) {
        const uint8_t lodLevel = static_cast<uint8_t>(lod);
        const auto lodIt = tileState.lodStates.find(lodLevel);
        for (int32_t zSlice = 0; zSlice < meshTileSliceCount_; ++zSlice) {
            const bool resident = lodIt != tileState.lodStates.end() &&
                lodIt->second.contains(zSlice) &&
                lodIt->second.at(zSlice).resident;
            if (resident || dispatches == nullptr) {
                continue;
            }
            dispatches->push_back(PendingMeshDispatch{
                TileLodCoord{MeshTileSliceCoord{tile, zSlice}, lodLevel},
                priority,
                false,
                true
            });
        }
    }

    const std::size_t added = (dispatches != nullptr) ? (dispatches->size() - beforeDispatchCount) : 0u;
    if (added == 0u) {
        if (isTileDisplayReadyLocked(tile, tileState)) {
            markVisibleTileReadyLocked(tile);
        } else {
            enqueueVisibleTileLocked(tile, visibleRingForTileLocked(tile), visibleDistanceSqForTileLocked(tile));
        }
        if (repump != nullptr) {
            *repump = true;
        }
        return;
    }

    if (dispatchedTiles != nullptr) {
        dispatchedTiles->push_back(tile);
    }
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
        advanceVisibleTileLocked(tile, false, nullptr, nullptr, nullptr);
    }
}

void MeshManager::pumpTileQueues() {
    std::vector<PendingMeshDispatch> dispatches;
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

                const std::size_t beforeDispatchCount = dispatches.size();
                advanceVisibleTileLocked(entry.tile, true, &dispatches, &visibleAttempts, &repump);
                const std::size_t added = dispatches.size() - beforeDispatchCount;
                queuedVisibleWork = queuedVisibleWork || (added > 0u);
                remainingBudget = (added >= remainingBudget) ? 0u : (remainingBudget - added);
            }

            if (!queuedVisibleWork) {
                break;
            }
        }
    }

    for (const PendingMeshDispatch& dispatch : dispatches) {
        scheduleTileLodMeshing(
            dispatch.coord,
            dispatch.priority,
            dispatch.forceRemesh,
            activeWindowExtraChunks,
            dispatch.usePriorityQueue
        );
    }

    if (!visibleAttempts.empty()) {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        for (const MeshTileCoord& tile : visibleAttempts) {
            if (!currentVisibleRingOutstandingTiles_.contains(tile)) {
                continue;
            }
            advanceVisibleTileLocked(tile, false, nullptr, nullptr, &repump);
        }
    }

    if (repump) {
        pumpTileQueues();
    }
}

void MeshManager::scheduleRemeshForNewColumns(const ColumnCoord& centerColumn) {
    struct ScheduledTileLod {
        TileLodCoord coord{};
        int32_t distanceSq = 0;
        jobsystem::Priority priority = jobsystem::Priority::Low;
        bool usePriorityQueue = false;
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
            const bool visibleOutstanding = currentVisibleRingOutstandingTiles_.contains(tile);
            const jobsystem::Priority primaryPriority = visibleOutstanding
                ? jobsystem::Priority::Critical
                : primaryPriorityForDistance(distanceChunks, meshTileSizeChunks_);

            for (int32_t lod = 0; lod < config_.lodLevelCount; ++lod) {
                for (int32_t zSlice = 0; zSlice < meshTileSliceCount_; ++zSlice) {
                    jobsToSchedule.push_back(ScheduledTileLod{
                        TileLodCoord{MeshTileSliceCoord{tile, zSlice}, static_cast<uint8_t>(lod)},
                        distanceSq,
                        primaryPriority,
                        visibleOutstanding
                    });
                }
            }
        }
    }

    std::stable_sort(jobsToSchedule.begin(), jobsToSchedule.end(), [](const ScheduledTileLod& a, const ScheduledTileLod& b) {
        if (a.distanceSq != b.distanceSq) {
            return a.distanceSq < b.distanceSq;
        }
        if (!(a.coord.tile == b.coord.tile)) {
            return a.coord.tile < b.coord.tile;
        }
        return a.coord.lodLevel < b.coord.lodLevel;
    });

    // Schedule all affected tiles without a pending-job cap.
    // scheduleTileLodMeshing already handles dedup (deferred remesh if
    // already pending, footprint check, active-window check), so the actual
    // number of *new* jobs submitted is naturally bounded.  Dropping tiles
    // here would permanently lose remesh requests because the generation
    // revision has already been advanced past these columns.
    for (const ScheduledTileLod& scheduled : jobsToSchedule) {
        scheduleTileLodMeshing(
            scheduled.coord,
            scheduled.priority,
            true,
            meshTileSizeChunks_ + 4,
            scheduled.usePriorityQueue
        );
    }

    pumpTileQueues();
}
