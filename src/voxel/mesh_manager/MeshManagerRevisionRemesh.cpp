#include "solum_engine/voxel/MeshManager.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <shared_mutex>
#include <unordered_set>
#include <vector>

#include "solum_engine/voxel/World.h"
#include "solum_engine/voxel/mesh_manager/TileFootprintUtils.h"

namespace {
constexpr std::size_t kPlayerEditRevisionBatch = 1024u;

using mesh_manager::tile_footprint::FootprintDistanceRange;
using mesh_manager::tile_footprint::footprintDistanceRangeForCell;
}  // namespace

void MeshManager::scheduleRemeshForChangedChunks(const ColumnCoord& centerColumn,
                                                 const std::vector<ChunkCoord>& changedChunks) {
    struct ScheduledTileLod {
        TileLodCoord coord{};
        int32_t distanceSq = 0;
        jobsystem::Priority priority = jobsystem::Priority::Critical;
    };

    if (changedChunks.empty()) {
        return;
    }

    std::unordered_set<MeshTileSliceCoord> slicesToRemesh;
    const int32_t remeshRadius = std::max(0, maxConfiguredRadius() + meshTileSizeChunks_ + 4);
    for (const ChunkCoord& coord : changedChunks) {
        const int32_t dx = std::abs(coord.v.x - centerColumn.v.x);
        const int32_t dy = std::abs(coord.v.y - centerColumn.v.y);
        if (dx > remeshRadius || dy > remeshRadius) {
            continue;
        }

        const int32_t tileX = floor_div(coord.v.x, meshTileSizeChunks_);
        const int32_t tileY = floor_div(coord.v.y, meshTileSizeChunks_);
        const int32_t baseSliceZ = floor_div(coord.v.z, meshTileHeightChunks_);
        const int32_t zMin = std::max(0, baseSliceZ - 1);
        const int32_t zMax = std::min(meshTileSliceCount_ - 1, baseSliceZ + 1);
        for (int32_t zSlice = zMin; zSlice <= zMax; ++zSlice) {
            slicesToRemesh.insert(MeshTileSliceCoord{MeshTileCoord{tileX, tileY}, zSlice});
        }
    }

    if (slicesToRemesh.empty()) {
        return;
    }

    const ChunkCoord centerChunk = hasLastScheduledCenter_
        ? lastScheduledCenterChunk_
        : ChunkCoord{centerColumn.v.x, centerColumn.v.y, 0};
    const glm::vec3 playerWorldPosition = lastPlayerWorldPosition_;
    const float sseProjectionScale = hasLastSseProjectionScale_
        ? lastSseProjectionScale_
        : config_.lodSseFallbackProjectionScale;

    std::vector<ScheduledTileLod> jobsToSchedule;
    jobsToSchedule.reserve(slicesToRemesh.size());

    {
        std::shared_lock<std::shared_mutex> lock(meshMutex_);
        for (const MeshTileSliceCoord& tileSlice : slicesToRemesh) {
            const MeshTileCoord& tile = tileSlice.tile;
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

            jobsToSchedule.push_back(ScheduledTileLod{
                TileLodCoord{tileSlice, static_cast<uint8_t>(targetLod)},
                distanceSq,
                jobsystem::Priority::Critical
            });
        }
    }

    std::stable_sort(jobsToSchedule.begin(), jobsToSchedule.end(), [](const ScheduledTileLod& a, const ScheduledTileLod& b) {
        if (a.distanceSq != b.distanceSq) {
            return a.distanceSq < b.distanceSq;
        }
        if (!(a.coord.tile == b.coord.tile)) {
            return a.coord.tile < b.coord.tile;
        }
        if (a.coord.lodLevel != b.coord.lodLevel) {
            return a.coord.lodLevel < b.coord.lodLevel;
        }
        return false;
    });

    for (const ScheduledTileLod& scheduled : jobsToSchedule) {
        scheduleTileLodMeshing(
            scheduled.coord,
            scheduled.priority,
            true,
            meshTileSizeChunks_ + 2,
            true
        );
    }
}

void MeshManager::scheduleRemeshForPlayerEditedChunks(const ColumnCoord& centerColumn) {
    const uint64_t processedRevision = processedWorldPlayerEditRevision_.load(std::memory_order_acquire);
    std::vector<ChunkCoord> editedChunks;
    const uint64_t nextRevision = world_.copyPlayerEditedChunksSince(
        processedRevision,
        editedChunks,
        kPlayerEditRevisionBatch
    );
    if (nextRevision == processedRevision) {
        return;
    }

    processedWorldPlayerEditRevision_.store(nextRevision, std::memory_order_release);
    scheduleRemeshForChangedChunks(centerColumn, editedChunks);
}

void MeshManager::scheduleRemeshForLightingChangedChunks(const ColumnCoord& centerColumn) {
    const uint64_t processedRevision = processedWorldLightingRevision_.load(std::memory_order_acquire);
    std::vector<ChunkCoord> changedChunks;
    const uint64_t nextRevision = world_.copyLightingChangedChunksSince(
        processedRevision,
        changedChunks,
        kPlayerEditRevisionBatch
    );
    if (nextRevision == processedRevision) {
        return;
    }

    processedWorldLightingRevision_.store(nextRevision, std::memory_order_release);
    scheduleRemeshForChangedChunks(centerColumn, changedChunks);
}
