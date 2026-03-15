#include "solum_engine/voxel/MeshManager.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "solum_engine/voxel/World.h"
#include "solum_engine/voxel/mesh_manager/TileFootprintUtils.h"

namespace {
constexpr std::size_t kPlayerEditRevisionBatch = 1024u;

using mesh_manager::tile_footprint::FootprintDistanceRange;
using mesh_manager::tile_footprint::footprintDistanceRangeForCell;

uint8_t mipMaskForLod(uint8_t lodLevel) {
    const uint8_t clampedLod = std::min<uint8_t>(lodLevel, Chunk::MAX_MIP_LEVEL);
    return static_cast<uint8_t>(1u << clampedLod);
}

bool lodAffectedByMipMask(uint8_t lodLevel, uint8_t changedMipMask) {
    return (changedMipMask & mipMaskForLod(lodLevel)) != 0u;
}

uint8_t fullLodMipMask(int32_t lodLevelCount) {
    uint8_t mask = 0u;
    for (int32_t lod = 0; lod < lodLevelCount; ++lod) {
        mask |= mipMaskForLod(static_cast<uint8_t>(lod));
    }
    return mask;
}

jobsystem::Priority remeshPriorityForDistance(int32_t distanceChunks, int32_t tileSizeChunks) {
    const int32_t tileDistance = std::max(1, tileSizeChunks);
    if (distanceChunks <= tileDistance) {
        return jobsystem::Priority::Critical;
    }
    if (distanceChunks <= (tileDistance * 2)) {
        return jobsystem::Priority::High;
    }
    return jobsystem::Priority::Normal;
}
}  // namespace

void MeshManager::scheduleRemeshForChangedChunks(const ColumnCoord& centerColumn,
                                                 const std::vector<WorldChunkEdit>& changedChunks) {
    struct ScheduledTileLod {
        TileLodCoord coord{};
        int32_t distanceSq = 0;
        uint8_t tier = 1u;
        jobsystem::Priority priority = jobsystem::Priority::Low;
        bool usePriorityQueue = false;
    };

    if (changedChunks.empty()) {
        return;
    }

    std::unordered_map<TileLodCoord, ScheduledTileLod> jobsByCoord;
    const int32_t remeshRadius = std::max(0, maxConfiguredRadius() + meshTileSizeChunks_ + 4);
    for (const WorldChunkEdit& edit : changedChunks) {
        if (edit.changedMipMask == 0u) {
            continue;
        }

        const ChunkCoord& coord = edit.coord;
        const int32_t dx = std::abs(coord.v.x - centerColumn.v.x);
        const int32_t dy = std::abs(coord.v.y - centerColumn.v.y);
        if (dx > remeshRadius || dy > remeshRadius) {
            continue;
        }

        for (int32_t lod = 0; lod < config_.lodLevelCount; ++lod) {
            const uint8_t lodLevel = static_cast<uint8_t>(lod);
            if (!lodAffectedByMipMask(lodLevel, edit.changedMipMask)) {
                continue;
            }

            const int32_t spanChunks = std::max(1, chunkSpanForLod(lodLevel));
            const int32_t cellMinChunkX = floor_div(coord.v.x, spanChunks) * spanChunks;
            const int32_t cellMinChunkY = floor_div(coord.v.y, spanChunks) * spanChunks;
            const int32_t cellMinChunkZ = floor_div(coord.v.z, spanChunks) * spanChunks;
            const int32_t cellMaxChunkX = cellMinChunkX + spanChunks - 1;
            const int32_t cellMaxChunkY = cellMinChunkY + spanChunks - 1;
            const int32_t cellMaxChunkZ = std::min(cfg::COLUMN_HEIGHT - 1, cellMinChunkZ + spanChunks - 1);

            const int32_t tileMinX = floor_div(cellMinChunkX, meshTileSizeChunks_);
            const int32_t tileMaxX = floor_div(cellMaxChunkX, meshTileSizeChunks_);
            const int32_t tileMinY = floor_div(cellMinChunkY, meshTileSizeChunks_);
            const int32_t tileMaxY = floor_div(cellMaxChunkY, meshTileSizeChunks_);
            const int32_t sliceMin = std::max(0, floor_div(cellMinChunkZ, meshTileHeightChunks_) - 1);
            const int32_t sliceMax = std::min(
                meshTileSliceCount_ - 1,
                floor_div(cellMaxChunkZ, meshTileHeightChunks_) + 1
            );

            for (int32_t tileY = tileMinY; tileY <= tileMaxY; ++tileY) {
                for (int32_t tileX = tileMinX; tileX <= tileMaxX; ++tileX) {
                    for (int32_t zSlice = sliceMin; zSlice <= sliceMax; ++zSlice) {
                        const TileLodCoord key{
                            MeshTileSliceCoord{MeshTileCoord{tileX, tileY}, zSlice},
                            lodLevel
                        };
                        jobsByCoord.try_emplace(key, ScheduledTileLod{key});
                    }
                }
            }
        }
    }

    if (jobsByCoord.empty()) {
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
    jobsToSchedule.reserve(jobsByCoord.size());

    {
        std::shared_lock<std::shared_mutex> lock(meshMutex_);
        for (auto& [coord, scheduled] : jobsByCoord) {
            const MeshTileCoord& tile = coord.tile.tile;
            const auto tileIt = meshTiles_.find(tile);

            const int8_t desired = (tileIt != meshTiles_.end() && tileIt->second.desiredLod >= 0)
                ? tileIt->second.desiredLod
                : desiredLodForTile(tile, centerChunk, playerWorldPosition, sseProjectionScale, 0);
            if (desired < 0) {
                continue;
            }

            const int8_t selected = (tileIt != meshTiles_.end()) ? tileIt->second.selectedLod : -1;
            const FootprintDistanceRange distances = footprintDistanceRangeForCell(
                tile.x,
                tile.y,
                meshTileSizeChunks_,
                centerChunk
            );
            const int32_t distanceChunks = distances.minDistanceChunks;
            scheduled.distanceSq = distanceChunks * distanceChunks;
            scheduled.usePriorityQueue =
                (desired == static_cast<int8_t>(coord.lodLevel)) ||
                (selected == static_cast<int8_t>(coord.lodLevel));
            scheduled.tier = scheduled.usePriorityQueue ? 0u : 1u;
            scheduled.priority = scheduled.usePriorityQueue
                ? remeshPriorityForDistance(distanceChunks, meshTileSizeChunks_)
                : priorityFromLodLevel(coord.lodLevel);

            jobsToSchedule.push_back(scheduled);
        }
    }

    std::stable_sort(jobsToSchedule.begin(), jobsToSchedule.end(), [](const ScheduledTileLod& a, const ScheduledTileLod& b) {
        if (a.tier != b.tier) {
            return a.tier < b.tier;
        }
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
            scheduled.usePriorityQueue
        );
    }
}

void MeshManager::scheduleRemeshForPlayerEditedChunks(const ColumnCoord& centerColumn) {
    const uint64_t processedRevision = processedWorldPlayerEditRevision_.load(std::memory_order_acquire);
    std::vector<WorldChunkEdit> editedChunks;
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
    std::vector<WorldChunkEdit> lightingEdits;
    lightingEdits.reserve(changedChunks.size());
    const uint8_t changedMipMask = fullLodMipMask(config_.lodLevelCount);
    for (const ChunkCoord& coord : changedChunks) {
        lightingEdits.push_back(WorldChunkEdit{coord, changedMipMask});
    }
    scheduleRemeshForChangedChunks(centerColumn, lightingEdits);
}
