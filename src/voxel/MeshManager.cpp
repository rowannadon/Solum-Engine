#include "solum_engine/voxel/MeshManager.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <mutex>
#include <utility>

#include "solum_engine/resources/Constants.h"
#include "solum_engine/voxel/mesh_manager/TileFootprintUtils.h"
#include "solum_engine/voxel/World.h"

namespace {
constexpr int kMaxLodShift = 30;

using mesh_manager::tile_footprint::FootprintDistanceRange;
using mesh_manager::tile_footprint::footprintDistanceRangeForCell;

bool isPowerOfTwo(int32_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

int32_t floorPowerOfTwo(int32_t value) {
    if (value <= 1) {
        return 1;
    }

    int32_t floored = 1;
    while (floored <= (value / 2)) {
        floored <<= 1;
    }
    return floored;
}

int32_t pow2ClampedShift(int32_t shift) {
    const int32_t clampedShift = std::clamp(shift, 0, kMaxLodShift);
    return (1 << clampedShift);
}

}  // namespace

MeshManager::MeshManager(const World& world, std::shared_ptr<const BlockModelLibrary> blockModelLibrary)
    : MeshManager(world, Config{}, std::move(blockModelLibrary)) {}

MeshManager::MeshManager(const World& world, Config config, std::shared_ptr<const BlockModelLibrary> blockModelLibrary)
    : world_(world),
      blockModelLibrary_(std::move(blockModelLibrary)),
      config_(std::move(config)),
      jobs_(config_.jobConfig),
      priorityJobs_([this]() {
          jobsystem::JobSystem::Config priorityConfig = config_.jobConfig;
          const std::size_t configuredWorkers =
              (priorityConfig.worker_threads > 0) ? priorityConfig.worker_threads : 1u;
          priorityConfig.worker_threads = std::clamp<std::size_t>(configuredWorkers, 1u, 2u);
          return priorityConfig;
      }()) {
    sanitizeConfig(config_);
    meshTileSizeChunks_ = std::max(1, config_.meshTileSizeChunks);
    meshTileHeightChunks_ = std::max(1, config_.meshTileHeightChunks);
    meshTileSliceCount_ = std::max(1, cfg::COLUMN_HEIGHT / meshTileHeightChunks_);
    processedWorldPlayerEditRevision_.store(world_.playerEditChunkRevision(), std::memory_order_release);
    processedWorldLightingRevision_.store(world_.lightingChunkRevision(), std::memory_order_release);
    processedWorldGenerationRevision_.store(world_.generationRevision(), std::memory_order_release);
}

MeshManager::~MeshManager() {
    shuttingDown_.store(true, std::memory_order_release);
    jobs_.wait_for_idle();
    priorityJobs_.wait_for_idle();
    jobs_.stop();
    priorityJobs_.stop();
}

MeshManager::PlanningUpdate MeshManager::updatePlanningInputs(const glm::vec3& playerWorldPosition,
                                                             float sseProjectionScale) {
    PlanningUpdate update;
    const float safeSseProjectionScale =
        (std::isfinite(sseProjectionScale) && sseProjectionScale > 0.0f)
        ? sseProjectionScale
        : config_.lodSseFallbackProjectionScale;
    update.safeSseProjectionScale = safeSseProjectionScale;

    const BlockCoord playerBlock{
        static_cast<int32_t>(std::floor(playerWorldPosition.x)),
        static_cast<int32_t>(std::floor(playerWorldPosition.y)),
        static_cast<int32_t>(std::floor(playerWorldPosition.z))
    };
    update.centerChunk = block_to_chunk(playerBlock);
    update.centerColumn = chunk_to_column(update.centerChunk);

    ChunkCoord previousCenterChunk{};
    bool hadPreviousCenter = false;
    bool centerChanged = false;

    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        const bool projectionChanged =
            !hasLastSseProjectionScale_ || std::abs(lastSseProjectionScale_ - safeSseProjectionScale) > 0.01f;
        const float replanDistanceBlocks = std::max(1.0f, 0.5f * static_cast<float>(cfg::CHUNK_SIZE));
        bool playerMovedForLodPlanning = !hasLastPlanningPlayerWorldPosition_;
        if (hasLastPlanningPlayerWorldPosition_) {
            const float dx = playerWorldPosition.x - lastPlanningPlayerWorldPosition_.x;
            const float dy = playerWorldPosition.y - lastPlanningPlayerWorldPosition_.y;
            playerMovedForLodPlanning = ((dx * dx) + (dy * dy)) >= (replanDistanceBlocks * replanDistanceBlocks);
        }

        lastPlayerWorldPosition_ = playerWorldPosition;
        hasLastPlayerWorldPosition_ = true;
        lastSseProjectionScale_ = safeSseProjectionScale;
        hasLastSseProjectionScale_ = true;

        if (!hasLastScheduledCenter_ || !(update.centerChunk == lastScheduledCenterChunk_)) {
            hadPreviousCenter = hasLastScheduledCenter_;
            previousCenterChunk = lastScheduledCenterChunk_;
            lastScheduledCenterChunk_ = update.centerChunk;
            hasLastScheduledCenter_ = true;
            centerChanged = true;
        }

        update.planningResetRequired = centerChanged || projectionChanged || playerMovedForLodPlanning;
        if (update.planningResetRequired) {
            lastPlanningPlayerWorldPosition_ = playerWorldPosition;
            hasLastPlanningPlayerWorldPosition_ = true;
        }

        update.shouldPumpPendingWork =
            !pendingTileLodJobs_.empty() ||
            !pendingPriorityTileLodJobs_.empty() ||
            !queuedVisibleTileHeap_.empty() ||
            !currentVisibleRingOutstandingTiles_.empty() ||
            !waitingVisibleTiles_.empty();
    }

    update.centerShiftChunks = (centerChanged && hadPreviousCenter)
        ? std::max(
            std::abs(update.centerChunk.v.x - previousCenterChunk.v.x),
            std::abs(update.centerChunk.v.y - previousCenterChunk.v.y))
        : 0;
    return update;
}

MeshManager::WorldChangeFlags MeshManager::drainWorldChangeFeeds(const ColumnCoord& centerColumn) {
    WorldChangeFlags changeFlags;

    changeFlags.playerEdits =
        world_.playerEditChunkRevision() != processedWorldPlayerEditRevision_.load(std::memory_order_acquire);
    if (changeFlags.playerEdits) {
        scheduleRemeshForPlayerEditedChunks(centerColumn);
    }

    changeFlags.lighting =
        world_.lightingChunkRevision() != processedWorldLightingRevision_.load(std::memory_order_acquire);
    if (changeFlags.lighting) {
        scheduleRemeshForLightingChangedChunks(centerColumn);
    }

    changeFlags.generation =
        world_.generationRevision() != processedWorldGenerationRevision_.load(std::memory_order_acquire);
    return changeFlags;
}

void MeshManager::maybeResetTilePlanning(const PlanningUpdate& update, const glm::vec3& playerWorldPosition) {
    if (!update.planningResetRequired) {
        return;
    }

    scheduleTilesAround(
        update.centerChunk,
        playerWorldPosition,
        update.safeSseProjectionScale,
        update.centerShiftChunks
    );
}

void MeshManager::pumpPendingVisibleWork(const PlanningUpdate& update, const WorldChangeFlags& changeFlags) {
    if (!update.planningResetRequired &&
        !changeFlags.playerEdits &&
        !changeFlags.lighting &&
        !changeFlags.generation) {
        if (update.shouldPumpPendingWork) {
            pumpTileQueues();
        }
        return;
    }
}

void MeshManager::updatePlayerPosition(const glm::vec3& playerWorldPosition, float sseProjectionScale) {
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return;
    }

    const PlanningUpdate update = updatePlanningInputs(playerWorldPosition, sseProjectionScale);
    const WorldChangeFlags changeFlags = drainWorldChangeFeeds(update.centerColumn);
    pumpPendingVisibleWork(update, changeFlags);
    maybeResetTilePlanning(update, playerWorldPosition);

    if (changeFlags.generation) {
        scheduleRemeshForNewColumns(update.centerColumn);
    }
}

bool MeshManager::scheduleTileLodMeshing(const TileLodCoord& coord,
                                         jobsystem::Priority priority,
                                         bool forceRemesh,
                                         int32_t activeWindowExtraChunks,
                                         bool usePriorityQueue) {
    if (!isTileFootprintGenerated(coord.tile.tile)) {
        return false;
    }

    bool scheduled = false;
    uint64_t jobGeneration = 0u;
    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        if (!isTileWithinActiveWindowLocked(coord.tile.tile, activeWindowExtraChunks)) {
            return false;
        }

        const bool normalPending = pendingTileLodJobs_.find(coord) != pendingTileLodJobs_.end();
        const bool priorityPending = pendingPriorityTileLodJobs_.find(coord) != pendingPriorityTileLodJobs_.end();
        if (priorityPending || normalPending) {
            if (forceRemesh) {
                deferredRemeshTileLods_.insert(coord);
            }
            return false;
        }

        MeshTileState& tileState = meshTiles_[coord.tile.tile];
        auto& lodState = tileState.lodStates[coord.lodLevel][coord.tile.z];
        if (lodState.resident && !forceRemesh) {
            return false;
        }

        if (usePriorityQueue) {
            pendingPriorityTileLodJobs_.insert(coord);
        } else {
            pendingTileLodJobs_.insert(coord);
        }
        jobGeneration = ++nextTileLodJobGeneration_;
        tileLodJobGeneration_[coord] = jobGeneration;
        scheduled = true;
    }

    try {
        jobsystem::JobSystem& targetJobs = usePriorityQueue ? priorityJobs_ : jobs_;
        targetJobs.schedule(
            priority,
            [this, coord, activeWindowExtraChunks]() -> MeshGenerationResult {
                {
                    std::shared_lock<std::shared_mutex> lock(meshMutex_);
                    if (!isTileWithinActiveWindowLocked(coord.tile.tile, activeWindowExtraChunks)) {
                        return MeshGenerationResult{coord, {}, false};
                    }
                }

                if (!isTileFootprintGenerated(coord.tile.tile)) {
                    return MeshGenerationResult{coord, {}, false};
                }

                return MeshGenerationResult{coord, meshTileLod(coord), true};
            },
            [this, coord, usePriorityQueue, jobGeneration](jobsystem::JobResult<MeshGenerationResult>&& result) {
                bool rescheduleDeferred = false;
                bool pumpQueues = false;
                {
                    std::unique_lock<std::shared_mutex> lock(meshMutex_);
                    if (usePriorityQueue) {
                        pendingPriorityTileLodJobs_.erase(coord);
                    } else {
                        pendingTileLodJobs_.erase(coord);
                    }

                    auto tileIt = meshTiles_.find(coord.tile.tile);
                    const bool isVisibleAttempt =
                        currentVisibleRingOutstandingTiles_.contains(coord.tile.tile);
                    const auto jobGenerationIt = tileLodJobGeneration_.find(coord);
                    const bool staleGeneration =
                        jobGenerationIt == tileLodJobGeneration_.end() ||
                        jobGenerationIt->second != jobGeneration;

                    if (staleGeneration || !result.success() || shuttingDown_.load(std::memory_order_acquire)) {
                        if (isVisibleAttempt) {
                            noteVisibleTileAttemptFinishedLocked(coord.tile.tile);
                            pumpQueues = true;
                        }
                    } else {
                        MeshGenerationResult meshResult = std::move(result).value();
                        if (!meshResult.meshed) {
                            if (isVisibleAttempt) {
                                noteVisibleTileAttemptFinishedLocked(coord.tile.tile);
                                pumpQueues = true;
                            }
                        } else if (tileIt != meshTiles_.end()) {
                            MeshTileLodState& lodState = tileIt->second.lodStates[coord.lodLevel][coord.tile.z];
                            lodState.culledPacked = std::make_shared<PackedMeshletData>(
                                packMeshletsForUpload(meshResult.meshOutput.culledMeshlets)
                            );
                            lodState.doubleSidedPacked = std::make_shared<PackedMeshletData>(
                                packMeshletsForUpload(meshResult.meshOutput.doubleSidedMeshlets)
                            );
                            lodState.resident = true;
                            lodState.revision = meshRevision_.fetch_add(1, std::memory_order_acq_rel) + 1u;
                            queueTileLodUploadLocked(MeshTileLodKey{coord.tile, coord.lodLevel}, usePriorityQueue);
                            lodState.uploadQueued = true;

                            if (tileIt->second.preferLod0DuringRemesh &&
                                allTileLodsResidentLocked(tileIt->second)) {
                                tileIt->second.preferLod0DuringRemesh = false;
                            }

                            refreshSelectedLodLocked(coord.tile.tile, tileIt->second);

                            if (isVisibleAttempt) {
                                bool tileStillPending = false;
                                for (int32_t lod = 0; lod < config_.lodLevelCount; ++lod) {
                                    if (pendingSliceCountForLodLocked(
                                            coord.tile.tile,
                                            static_cast<uint8_t>(lod)
                                        ) > 0u) {
                                        tileStillPending = true;
                                        break;
                                    }
                                }

                                if (!tileStillPending) {
                                    noteVisibleTileAttemptFinishedLocked(coord.tile.tile);
                                    pumpQueues = true;
                                }
                            }

                            const auto deferredIt = deferredRemeshTileLods_.find(coord);
                            if (deferredIt != deferredRemeshTileLods_.end()) {
                                deferredRemeshTileLods_.erase(deferredIt);
                                rescheduleDeferred = true;
                            }
                        }
                    }

                    if (pendingTileLodJobs_.find(coord) == pendingTileLodJobs_.end() &&
                        pendingPriorityTileLodJobs_.find(coord) == pendingPriorityTileLodJobs_.end()) {
                        deferredRemeshTileLods_.erase(coord);
                    }
                }

                if (rescheduleDeferred) {
                    scheduleTileLodMeshing(
                        coord,
                        priorityFromLodLevel(coord.lodLevel),
                        true,
                        meshTileSizeChunks_ + 4,
                        usePriorityQueue
                    );
                }
                if (pumpQueues) {
                    pumpTileQueues();
                }
            }
        );
    } catch (const std::exception&) {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        if (usePriorityQueue) {
            pendingPriorityTileLodJobs_.erase(coord);
        } else {
            pendingTileLodJobs_.erase(coord);
        }
        if (currentVisibleRingOutstandingTiles_.contains(coord.tile.tile)) {
            noteVisibleTileAttemptFinishedLocked(coord.tile.tile);
        }
        if (pendingTileLodJobs_.find(coord) == pendingTileLodJobs_.end() &&
            pendingPriorityTileLodJobs_.find(coord) == pendingPriorityTileLodJobs_.end()) {
            deferredRemeshTileLods_.erase(coord);
        }
        return false;
    }

    return scheduled;
}

bool MeshManager::isTileWithinActiveWindowLocked(const MeshTileCoord& tileCoord, int32_t extraChunks) const {
    if (!hasLastScheduledCenter_) {
        return true;
    }

    const int32_t radiusChunks = std::max(0, maxConfiguredRadius() + extraChunks);
    const FootprintDistanceRange distances = footprintDistanceRangeForCell(
        tileCoord.x,
        tileCoord.y,
        meshTileSizeChunks_,
        lastScheduledCenterChunk_
    );

    return distances.minDistanceChunks <= radiusChunks;
}

bool MeshManager::isTileFootprintGenerated(const MeshTileCoord& tileCoord) const {
    const int32_t baseChunkX = tileCoord.x * meshTileSizeChunks_;
    const int32_t baseChunkY = tileCoord.y * meshTileSizeChunks_;

    for (int32_t dy = 0; dy < meshTileSizeChunks_; ++dy) {
        for (int32_t dx = 0; dx < meshTileSizeChunks_; ++dx) {
            if (!world_.isColumnGenerated(ColumnCoord{baseChunkX + dx, baseChunkY + dy})) {
                return false;
            }
        }
    }

    return true;
}

bool MeshManager::lodFullyResidentLocked(const MeshTileState& state, int32_t lodLevel) const {
    if (lodLevel < 0) {
        return false;
    }

    const auto lodIt = state.lodStates.find(static_cast<uint8_t>(lodLevel));
    if (lodIt == state.lodStates.end()) {
        return false;
    }
    for (int32_t zSlice = 0; zSlice < meshTileSliceCount_; ++zSlice) {
        const auto sliceIt = lodIt->second.find(zSlice);
        if (sliceIt == lodIt->second.end() || !sliceIt->second.resident) {
            return false;
        }
    }
    return true;
}

bool MeshManager::allTileLodsResidentLocked(const MeshTileState& state) const {
    for (int32_t lod = 0; lod < config_.lodLevelCount; ++lod) {
        if (!lodFullyResidentLocked(state, lod)) {
            return false;
        }
    }
    return config_.lodLevelCount > 0;
}

uint8_t MeshManager::pendingSliceCountForLodLocked(const MeshTileCoord& tileCoord, uint8_t lodLevel) const {
    uint8_t count = 0u;
    for (int32_t zSlice = 0; zSlice < meshTileSliceCount_; ++zSlice) {
        const TileLodCoord coord{MeshTileSliceCoord{tileCoord, zSlice}, lodLevel};
        if (pendingTileLodJobs_.find(coord) != pendingTileLodJobs_.end() ||
            pendingPriorityTileLodJobs_.find(coord) != pendingPriorityTileLodJobs_.end()) {
            ++count;
        }
    }
    return count;
}

bool MeshManager::canDisplayLod0DuringRemeshLocked(const MeshTileCoord& tileCoord, const MeshTileState& state) const {
    if (!state.preferLod0DuringRemesh) {
        return false;
    }
    return lodFullyResidentLocked(state, 0) &&
           pendingSliceCountForLodLocked(tileCoord, 0u) == 0u;
}

bool MeshManager::isTileDisplayReadyLocked(const MeshTileCoord& tileCoord, const MeshTileState& state) const {
    return allTileLodsResidentLocked(state) || canDisplayLod0DuringRemeshLocked(tileCoord, state);
}

bool MeshManager::hasVisibleQueueWorkLocked() const {
    if (!queuedVisibleTileHeap_.empty() || !currentVisibleRingOutstandingTiles_.empty()) {
        return true;
    }
    return visibleFrontierRing_ <= maxVisibleFrontierRing_;
}

int32_t MeshManager::visibleRingForTileLocked(const MeshTileCoord& tileCoord) const {
    return std::max(std::abs(tileCoord.x - currentCenterTile_.x), std::abs(tileCoord.y - currentCenterTile_.y));
}

int32_t MeshManager::visibleDistanceSqForTileLocked(const MeshTileCoord& tileCoord) const {
    const FootprintDistanceRange distances = footprintDistanceRangeForCell(
        tileCoord.x,
        tileCoord.y,
        meshTileSizeChunks_,
        lastScheduledCenterChunk_
    );
    const int32_t distanceChunks = distances.minDistanceChunks;
    return distanceChunks * distanceChunks;
}

int32_t MeshManager::cellCountPerAxisForLod(uint8_t lodLevel) const {
    const int32_t spanChunks = std::max(1, chunkSpanForLod(lodLevel));
    return std::max(1, (meshTileSizeChunks_ + spanChunks - 1) / spanChunks);
}

int32_t MeshManager::cellCountPerZForLod(uint8_t lodLevel) const {
    const int32_t spanChunks = std::max(1, chunkSpanForLod(lodLevel));
    return std::max(1, (meshTileHeightChunks_ + spanChunks - 1) / spanChunks);
}

uint64_t MeshManager::meshRevision() const noexcept {
    return meshRevision_.load(std::memory_order_acquire);
}

bool MeshManager::tileInBounds(const MeshTileCoord& tileCoord,
                               int32_t minTileX,
                               int32_t maxTileX,
                               int32_t minTileY,
                               int32_t maxTileY) {
    return tileCoord.x >= minTileX &&
           tileCoord.x <= maxTileX &&
           tileCoord.y >= minTileY &&
           tileCoord.y <= maxTileY;
}

int32_t MeshManager::maxConfiguredRadius() const {
    return std::max(0, config_.activeChunkRadius);
}

int32_t MeshManager::chunkSpanForLod(uint8_t lodLevel) {
    const int32_t extraShift = std::max(
        0,
        static_cast<int32_t>(lodLevel) - static_cast<int32_t>(Chunk::MAX_MIP_LEVEL)
    );
    return pow2ClampedShift(extraShift);
}

int32_t MeshManager::chunkZCountForLod(uint8_t lodLevel) {
    const int32_t spanChunks = static_cast<int32_t>(chunkSpanForLod(lodLevel));
    return std::max(1, cfg::COLUMN_HEIGHT / spanChunks);
}

jobsystem::Priority MeshManager::priorityFromLodLevel(uint8_t lodLevel) {
    // Fallback priority used for deferred remesh retries.
    if (lodLevel == 0u) {
        return jobsystem::Priority::High;
    }
    if (lodLevel == 1u) {
        return jobsystem::Priority::Normal;
    }
    return jobsystem::Priority::Low;
}

void MeshManager::sanitizeConfig(Config& config) {
    config.meshTileSizeChunks = std::max(1, config.meshTileSizeChunks);
    if (!isPowerOfTwo(config.meshTileSizeChunks)) {
        config.meshTileSizeChunks = floorPowerOfTwo(config.meshTileSizeChunks);
    }
    config.meshTileHeightChunks = std::max(1, config.meshTileHeightChunks);
    if (!isPowerOfTwo(config.meshTileHeightChunks)) {
        config.meshTileHeightChunks = floorPowerOfTwo(config.meshTileHeightChunks);
    }
    config.meshTileHeightChunks = std::min(config.meshTileHeightChunks, cfg::COLUMN_HEIGHT);
    while (config.meshTileHeightChunks > 1 &&
           (cfg::COLUMN_HEIGHT % config.meshTileHeightChunks) != 0) {
        config.meshTileHeightChunks >>= 1;
    }

    config.lodLevelCount = std::max(1, config.lodLevelCount);
    const int32_t maxLodLevelCountBySpan = kMaxLodShift + 1;
    config.lodLevelCount = std::clamp(config.lodLevelCount, 1, maxLodLevelCountBySpan);

    config.activeChunkRadius = std::max(0, config.activeChunkRadius);

    if (!std::isfinite(config.lodSseTargetPixels) || config.lodSseTargetPixels <= 0.0f) {
        config.lodSseTargetPixels = 1.0f;
    }
    if (!std::isfinite(config.lodSseHysteresisPixels) || config.lodSseHysteresisPixels < 0.0f) {
        config.lodSseHysteresisPixels = 0.25f;
    }
    if (!std::isfinite(config.lodSseMinDepthBlocks) || config.lodSseMinDepthBlocks <= 0.0f) {
        config.lodSseMinDepthBlocks = 4.0f;
    }
    if (!std::isfinite(config.lodSseFallbackProjectionScale) ||
        config.lodSseFallbackProjectionScale <= 0.0f) {
        config.lodSseFallbackProjectionScale = 390.0f;
    }
}
