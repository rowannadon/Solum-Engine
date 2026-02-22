#include "solum_engine/voxel/MeshManager.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iterator>
#include <mutex>
#include <utility>

#include "solum_engine/resources/Constants.h"
#include "solum_engine/voxel/ChunkMesher.h"
#include "solum_engine/voxel/World.h"

namespace {
constexpr int kChunkExtent = cfg::CHUNK_SIZE;
constexpr int kPaddedChunkExtent = cfg::CHUNK_SIZE + 2;
constexpr int kPaddedChunkArea = kPaddedChunkExtent * kPaddedChunkExtent;
constexpr int kPaddedChunkVoxelCount = kPaddedChunkExtent * kPaddedChunkExtent * kPaddedChunkExtent;
constexpr int kMinPrefetchChunks = 4;

BlockMaterial airBlock() {
    static const BlockMaterial kAir = UnpackedBlockMaterial{}.pack();
    return kAir;
}

BlockMaterial unknownCullingBlock() {
    static const BlockMaterial kSolid = UnpackedBlockMaterial{1, 0, Direction::PlusZ, 0}.pack();
    return kSolid;
}

struct PaddedChunkBlockSource final : IBlockSource {
    BlockCoord origin{};
    std::array<BlockMaterial, kPaddedChunkVoxelCount> blocks{};

    static constexpr int index(int x, int y, int z) {
        return (x * kPaddedChunkArea) + (y * kPaddedChunkExtent) + z;
    }

    BlockMaterial getBlock(const BlockCoord& coord) const override {
        const int lx = coord.v.x - origin.v.x;
        const int ly = coord.v.y - origin.v.y;
        const int lz = coord.v.z - origin.v.z;
        if (lx < 0 || ly < 0 || lz < 0 ||
            lx >= kPaddedChunkExtent ||
            ly >= kPaddedChunkExtent ||
            lz >= kPaddedChunkExtent) {
            return airBlock();
        }

        return blocks[static_cast<size_t>(index(lx, ly, lz))];
    }
};

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

struct MeshManager::MeshGenerationResult {
    TileLodCoord coord;
    std::vector<Meshlet> meshlets;
    bool meshed = false;
};

MeshManager::MeshManager(const World& world)
    : MeshManager(world, Config{}) {}

MeshManager::MeshManager(const World& world, Config config)
    : world_(world),
      config_(std::move(config)),
      jobs_(config_.jobConfig) {
    sanitizeConfig(config_);
    const uint8_t maxConfiguredLod = static_cast<uint8_t>(config_.lodChunkRadii.size() - 1);
    meshTileSizeChunks_ = std::max(1, static_cast<int32_t>(chunkSpanForLod(maxConfiguredLod)));
}

MeshManager::~MeshManager() {
    shuttingDown_.store(true, std::memory_order_release);
    jobs_.wait_for_idle();
    jobs_.stop();
}

void MeshManager::waitForIdle() {
    jobs_.wait_for_idle();
}

void MeshManager::updatePlayerPosition(const glm::vec3& playerWorldPosition) {
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return;
    }

    const BlockCoord playerBlock{
        static_cast<int32_t>(std::floor(playerWorldPosition.x)),
        static_cast<int32_t>(std::floor(playerWorldPosition.y)),
        static_cast<int32_t>(std::floor(playerWorldPosition.z))
    };
    const ChunkCoord centerChunk = block_to_chunk(playerBlock);
    const ColumnCoord centerColumn = chunk_to_column(centerChunk);

    ChunkCoord previousCenterChunk{};
    bool hadPreviousCenter = false;
    bool centerChanged = false;
    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        if (!hasLastScheduledCenter_ || !(centerChunk == lastScheduledCenterChunk_)) {
            hadPreviousCenter = hasLastScheduledCenter_;
            previousCenterChunk = lastScheduledCenterChunk_;
            lastScheduledCenterChunk_ = centerChunk;
            hasLastScheduledCenter_ = true;
            centerChanged = true;
        }
    }

    if (centerChanged) {
        const int32_t centerShiftChunks = hadPreviousCenter
            ? std::max(
                  std::abs(centerChunk.v.x - previousCenterChunk.v.x),
                  std::abs(centerChunk.v.y - previousCenterChunk.v.y))
            : 0;
        scheduleTilesAround(centerChunk, centerShiftChunks);
    }

    scheduleRemeshForNewColumns(centerColumn);
}

void MeshManager::scheduleTilesAround(const ChunkCoord& centerChunk, int32_t centerShiftChunks) {
    struct ScheduledTileLod {
        int32_t distanceSq = 0;
        TileLodCoord coord{};
        bool forceRemesh = false;
        int32_t activeWindowExtraChunks = 0;
    };

    std::vector<ScheduledTileLod> jobsToSchedule;
    std::unordered_map<MeshTileCoord, int8_t> desiredLodByTile;

    const int32_t maxRadiusChunks = std::max(0, maxConfiguredRadius());
    const int32_t clampedCenterShift = std::min(centerShiftChunks, 2);
    const int32_t prefetchChunks = std::max(kMinPrefetchChunks, clampedCenterShift * meshTileSizeChunks_);
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
    const int8_t maxLod = static_cast<int8_t>(config_.lodChunkRadii.size() - 1);

    for (int32_t tileY = minTileY; tileY <= maxTileY; ++tileY) {
        for (int32_t tileX = minTileX; tileX <= maxTileX; ++tileX) {
            const MeshTileCoord tileCoord{tileX, tileY};
            const FootprintDistanceRange distances = footprintDistanceRangeForCell(
                tileX,
                tileY,
                meshTileSizeChunks_,
                centerChunk
            );
            if (distances.minDistanceChunks > scheduleOuterRadiusChunks) {
                continue;
            }

            const int8_t visibleDesired = desiredLodForTile(tileCoord, centerChunk, 0);
            const int8_t prefetchDesired = desiredLodForTile(tileCoord, centerChunk, prefetchChunks);
            const int8_t baseDesired = (visibleDesired >= 0) ? visibleDesired : prefetchDesired;
            if (baseDesired < 0) {
                continue;
            }

            desiredLodByTile[tileCoord] = baseDesired;

            const int32_t distanceSq = distances.minDistanceChunks * distances.minDistanceChunks;
            const int32_t lodMin = std::max<int32_t>(0, static_cast<int32_t>(baseDesired) - 1);
            const int32_t lodMax = std::min<int32_t>(maxLod, static_cast<int32_t>(baseDesired) + 1);
            const int32_t activeWindowExtraChunks = prefetchChunks + meshTileSizeChunks_;

            for (int32_t lod = lodMin; lod <= lodMax; ++lod) {
                jobsToSchedule.push_back(ScheduledTileLod{
                    distanceSq,
                    TileLodCoord{tileCoord, static_cast<uint8_t>(lod)},
                    false,
                    activeWindowExtraChunks
                });
            }
        }
    }

    std::sort(jobsToSchedule.begin(), jobsToSchedule.end(), [](const ScheduledTileLod& a, const ScheduledTileLod& b) {
        if (a.coord.lodLevel != b.coord.lodLevel) {
            return a.coord.lodLevel < b.coord.lodLevel;
        }
        if (a.distanceSq != b.distanceSq) {
            return a.distanceSq < b.distanceSq;
        }
        return a.coord < b.coord;
    });

    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);

        for (const auto& [tileCoord, desiredLod] : desiredLodByTile) {
            MeshTileState& tileState = meshTiles_[tileCoord];
            tileState.desiredLod = desiredLod;
        }

        for (auto& [tileCoord, tileState] : meshTiles_) {
            if (desiredLodByTile.find(tileCoord) == desiredLodByTile.end()) {
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
            for (const TileLodCoord& pending : pendingTileJobs_) {
                if (pending.tile == it->first) {
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

    for (const ScheduledTileLod& scheduled : jobsToSchedule) {
        scheduleTileLodMeshing(
            scheduled.coord,
            priorityFromLodLevel(scheduled.coord.lodLevel),
            scheduled.forceRemesh,
            scheduled.activeWindowExtraChunks
        );
    }
}

void MeshManager::scheduleRemeshForNewColumns(const ColumnCoord& centerColumn) {
    const int32_t radius = std::max(0, maxConfiguredRadius() + meshTileSizeChunks_);
    std::vector<ColumnCoord> generatedColumns;
    generatedColumns.reserve(512);

    auto tryCollectGenerated = [this, &generatedColumns](const ColumnCoord& coord) {
        if (world_.isColumnGenerated(coord)) {
            generatedColumns.push_back(coord);
        }
    };

    // Keep near-player remesh responsive.
    const int32_t nearRadius = std::min(radius, meshTileSizeChunks_);
    for (int32_t dy = -nearRadius; dy <= nearRadius; ++dy) {
        for (int32_t dx = -nearRadius; dx <= nearRadius; ++dx) {
            tryCollectGenerated(ColumnCoord{
                centerColumn.v.x + dx,
                centerColumn.v.y + dy
            });
        }
    }

    // Incremental background scan over the full active window to avoid large per-frame spikes.
    int32_t scanStartIndex = 0;
    int32_t scanCount = 0;
    int32_t diameter = (radius * 2) + 1;
    ColumnCoord scanCenter = centerColumn;
    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        if (!hasRemeshScanCenter_ || !(centerColumn == remeshScanCenter_)) {
            remeshScanCenter_ = centerColumn;
            hasRemeshScanCenter_ = true;
            remeshScanNextIndex_ = 0;
        }

        scanCenter = remeshScanCenter_;
        const int32_t totalCells = diameter * diameter;
        scanStartIndex = std::clamp(remeshScanNextIndex_, 0, std::max(0, totalCells - 1));
        const int32_t scanBudget = 512;
        scanCount = std::min(scanBudget, totalCells);
        remeshScanNextIndex_ = (remeshScanNextIndex_ + scanCount) % std::max(1, totalCells);
    }

    for (int32_t i = 0; i < scanCount; ++i) {
        const int32_t wrappedIndex = (scanStartIndex + i) % (diameter * diameter);
        const int32_t localY = wrappedIndex / diameter;
        const int32_t localX = wrappedIndex % diameter;
        const int32_t dx = localX - radius;
        const int32_t dy = localY - radius;
        tryCollectGenerated(ColumnCoord{
            scanCenter.v.x + dx,
            scanCenter.v.y + dy
        });
    }

    if (generatedColumns.empty()) {
        return;
    }

    std::unordered_set<MeshTileCoord> tilesToRemesh;

    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        for (const ColumnCoord& coord : generatedColumns) {
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
    {
        std::shared_lock<std::shared_mutex> lock(meshMutex_);
        seamCenterChunk = hasLastScheduledCenter_
            ? lastScheduledCenterChunk_
            : ChunkCoord{centerColumn.v.x, centerColumn.v.y, 0};
    }

    const int8_t maxLod = static_cast<int8_t>(config_.lodChunkRadii.size() - 1);
    for (const MeshTileCoord& tileCoord : tilesToRemesh) {
        const int8_t visibleDesired = desiredLodForTile(tileCoord, seamCenterChunk, 0);
        const int8_t prefetchDesired = desiredLodForTile(tileCoord, seamCenterChunk, kMinPrefetchChunks);
        const int8_t baseDesired = (visibleDesired >= 0) ? visibleDesired : prefetchDesired;
        if (baseDesired < 0) {
            continue;
        }

        const int32_t lodMin = std::max<int32_t>(0, static_cast<int32_t>(baseDesired) - 1);
        const int32_t lodMax = std::min<int32_t>(maxLod, static_cast<int32_t>(baseDesired) + 1);
        const int32_t activeWindowExtraChunks = kMinPrefetchChunks + meshTileSizeChunks_;

        for (int32_t lod = lodMin; lod <= lodMax; ++lod) {
            scheduleTileLodMeshing(
                TileLodCoord{tileCoord, static_cast<uint8_t>(lod)},
                priorityFromLodLevel(static_cast<uint8_t>(lod)),
                true,
                activeWindowExtraChunks
            );
        }
    }
}

std::vector<Meshlet> MeshManager::meshLodCell(const ChunkCoord& cellCoord, uint8_t lodLevel) const {
    const uint8_t mipLevel = std::min<uint8_t>(lodLevel, Chunk::MAX_MIP_LEVEL);
    const uint8_t voxelScale = static_cast<uint8_t>(1u << mipLevel);

    ChunkMesher mesher;
    const BlockCoord sectionOriginMip{
        cellCoord.v.x * cfg::CHUNK_SIZE,
        cellCoord.v.y * cfg::CHUNK_SIZE,
        cellCoord.v.z * cfg::CHUNK_SIZE
    };
    const BlockCoord paddedOriginMip{
        sectionOriginMip.v.x - 1,
        sectionOriginMip.v.y - 1,
        sectionOriginMip.v.z - 1
    };

    PaddedChunkBlockSource snapshot;
    snapshot.origin = paddedOriginMip;
    snapshot.blocks.fill(airBlock());

    const glm::ivec3 paddedExtent{
        kPaddedChunkExtent,
        kPaddedChunkExtent,
        kPaddedChunkExtent
    };
    WorldSection worldSection = world_.createSection(paddedOriginMip, paddedExtent, mipLevel);
    std::vector<WorldSection::Sample> sectionSamples;
    worldSection.copySamples(sectionSamples);

    const int32_t worldHeightAtMip = cfg::COLUMN_HEIGHT_BLOCKS >> mipLevel;
    const size_t paddedYZArea =
        static_cast<size_t>(kPaddedChunkExtent) * static_cast<size_t>(kPaddedChunkExtent);

    for (int x = 0; x < kPaddedChunkExtent; ++x) {
        for (int y = 0; y < kPaddedChunkExtent; ++y) {
            for (int z = 0; z < kPaddedChunkExtent; ++z) {
                const size_t sampleIndex =
                    (static_cast<size_t>(x) * paddedYZArea) +
                    (static_cast<size_t>(y) * static_cast<size_t>(kPaddedChunkExtent)) +
                    static_cast<size_t>(z);
                const WorldSection::Sample& sample = sectionSamples[sampleIndex];
                const BlockCoord coordToCopy{
                    paddedOriginMip.v.x + x,
                    paddedOriginMip.v.y + y,
                    paddedOriginMip.v.z + z
                };

                BlockMaterial block = sample.block;
                if (!sample.known) {
                    if (coordToCopy.v.z >= 0 && coordToCopy.v.z < worldHeightAtMip) {
                        block = unknownCullingBlock();
                    } else {
                        block = airBlock();
                    }
                }
                snapshot.blocks[static_cast<size_t>(PaddedChunkBlockSource::index(x, y, z))] = block;
            }
        }
    }

    const glm::ivec3 sectionExtent{kChunkExtent, kChunkExtent, kChunkExtent};
    const glm::ivec3 meshletOrigin{
        sectionOriginMip.v.x * voxelScale,
        sectionOriginMip.v.y * voxelScale,
        sectionOriginMip.v.z * voxelScale
    };
    return mesher.mesh(
        snapshot,
        sectionOriginMip,
        sectionExtent,
        meshletOrigin,
        voxelScale
    );
}

void MeshManager::scheduleTileLodMeshing(const TileLodCoord& coord,
                                         jobsystem::Priority priority,
                                         bool forceRemesh,
                                         int32_t activeWindowExtraChunks) {
    if (!isTileFootprintGenerated(coord.tile)) {
        return;
    }

    const int32_t clampedActiveWindowExtraChunks = std::max(0, activeWindowExtraChunks);

    bool shouldSchedule = false;
    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        if (pendingTileJobs_.find(coord) != pendingTileJobs_.end()) {
            if (forceRemesh) {
                deferredRemeshTileLods_.insert(coord);
            }
            return;
        }

        if (!isTileWithinActiveWindowLocked(coord.tile, clampedActiveWindowExtraChunks)) {
            return;
        }

        const auto tileIt = meshTiles_.find(coord.tile);
        if (tileIt != meshTiles_.end() &&
            !forceRemesh &&
            tileIt->second.lodMeshes.find(coord.lodLevel) != tileIt->second.lodMeshes.end()) {
            return;
        }

        pendingTileJobs_.insert(coord);
        shouldSchedule = true;
    }

    if (!shouldSchedule) {
        return;
    }

    try {
        jobs_.schedule(
            priority,
            [this, coord, clampedActiveWindowExtraChunks]() -> MeshGenerationResult {
                {
                    std::shared_lock<std::shared_mutex> lock(meshMutex_);
                    if (!isTileWithinActiveWindowLocked(coord.tile, clampedActiveWindowExtraChunks)) {
                        return MeshGenerationResult{coord, {}, false};
                    }
                }

                if (!isTileFootprintGenerated(coord.tile)) {
                    return MeshGenerationResult{coord, {}, false};
                }

                const int32_t spanChunks = static_cast<int32_t>(chunkSpanForLod(coord.lodLevel));
                const int32_t tileOriginChunkX = coord.tile.x * meshTileSizeChunks_;
                const int32_t tileOriginChunkY = coord.tile.y * meshTileSizeChunks_;
                const int32_t baseCellX = floor_div(tileOriginChunkX, spanChunks);
                const int32_t baseCellY = floor_div(tileOriginChunkY, spanChunks);
                const int32_t cellsPerAxis = std::max(1, meshTileSizeChunks_ / spanChunks);
                const int32_t zCount = chunkZCountForLod(coord.lodLevel);

                std::vector<Meshlet> meshlets;

                for (int32_t y = 0; y < cellsPerAxis; ++y) {
                    for (int32_t x = 0; x < cellsPerAxis; ++x) {
                        for (int32_t z = 0; z < zCount; ++z) {
                            const ChunkCoord cellCoord{
                                baseCellX + x,
                                baseCellY + y,
                                z
                            };
                            std::vector<Meshlet> cellMeshlets = meshLodCell(cellCoord, coord.lodLevel);
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
                onTileLodMeshed(meshResult.coord, std::move(meshResult.meshlets));
            }
        );
    } catch (const std::exception&) {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        pendingTileJobs_.erase(coord);
        deferredRemeshTileLods_.erase(coord);
    }
}

void MeshManager::onTileLodMeshed(const TileLodCoord& coord, std::vector<Meshlet>&& meshlets) {
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return;
    }

    bool needsDeferredRemesh = false;
    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        pendingTileJobs_.erase(coord);

        MeshTileState& tileState = meshTiles_[coord.tile];
        tileState.lodMeshes[coord.lodLevel] = std::move(meshlets);

        auto deferredIt = deferredRemeshTileLods_.find(coord);
        if (deferredIt != deferredRemeshTileLods_.end()) {
            deferredRemeshTileLods_.erase(deferredIt);
            needsDeferredRemesh = true;
        }

        refreshRenderedLodsLocked();
    }

    meshRevision_.fetch_add(1, std::memory_order_acq_rel);

    if (needsDeferredRemesh) {
        const int32_t activeWindowExtraChunks = kMinPrefetchChunks + meshTileSizeChunks_;
        scheduleTileLodMeshing(
            coord,
            priorityFromLodLevel(coord.lodLevel),
            true,
            activeWindowExtraChunks
        );
    }
}

std::vector<Meshlet> MeshManager::copyMeshlets() const {
    std::shared_lock<std::shared_mutex> lock(meshMutex_);

    struct SelectedTileMesh {
        MeshTileCoord tile{};
        uint8_t lod = 0;
        const std::vector<Meshlet>* meshlets = nullptr;
    };

    std::vector<SelectedTileMesh> selected;
    selected.reserve(meshTiles_.size());

    for (const auto& [tileCoord, tileState] : meshTiles_) {
        const int8_t chosenLod = chooseRenderableLodForTileLocked(tileState);
        if (chosenLod < 0) {
            continue;
        }

        const auto meshIt = tileState.lodMeshes.find(static_cast<uint8_t>(chosenLod));
        if (meshIt == tileState.lodMeshes.end()) {
            continue;
        }

        selected.push_back(SelectedTileMesh{
            tileCoord,
            static_cast<uint8_t>(chosenLod),
            &meshIt->second
        });
    }

    std::sort(selected.begin(), selected.end(), [](const SelectedTileMesh& a, const SelectedTileMesh& b) {
        if (a.tile == b.tile) {
            return a.lod < b.lod;
        }
        return a.tile < b.tile;
    });

    size_t totalMeshletCount = 0;
    for (const SelectedTileMesh& entry : selected) {
        totalMeshletCount += entry.meshlets->size();
    }

    std::vector<Meshlet> meshlets;
    meshlets.reserve(totalMeshletCount);

    for (const SelectedTileMesh& entry : selected) {
        meshlets.insert(meshlets.end(), entry.meshlets->begin(), entry.meshlets->end());
    }

    return meshlets;
}

std::vector<Meshlet> MeshManager::copyMeshletsAround(const ColumnCoord& centerColumn, int32_t columnRadius) const {
    std::shared_lock<std::shared_mutex> lock(meshMutex_);

    struct SelectedTileMesh {
        MeshTileCoord tile{};
        uint8_t lod = 0;
        const std::vector<Meshlet>* meshlets = nullptr;
    };

    std::vector<SelectedTileMesh> selected;
    selected.reserve(meshTiles_.size());

    const int32_t clampedRadius = std::max(0, columnRadius);
    const int32_t minColumnX = centerColumn.v.x - clampedRadius;
    const int32_t maxColumnX = centerColumn.v.x + clampedRadius;
    const int32_t minColumnY = centerColumn.v.y - clampedRadius;
    const int32_t maxColumnY = centerColumn.v.y + clampedRadius;

    auto intersectsView = [this, minColumnX, maxColumnX, minColumnY, maxColumnY](const MeshTileCoord& tileCoord) {
        const int32_t tileMinX = tileCoord.x * meshTileSizeChunks_;
        const int32_t tileMaxX = tileMinX + meshTileSizeChunks_ - 1;
        const int32_t tileMinY = tileCoord.y * meshTileSizeChunks_;
        const int32_t tileMaxY = tileMinY + meshTileSizeChunks_ - 1;
        return !(tileMaxX < minColumnX || tileMinX > maxColumnX ||
                 tileMaxY < minColumnY || tileMinY > maxColumnY);
    };

    for (const auto& [tileCoord, tileState] : meshTiles_) {
        if (!intersectsView(tileCoord)) {
            continue;
        }

        const int8_t chosenLod = chooseRenderableLodForTileLocked(tileState);
        if (chosenLod < 0) {
            continue;
        }

        const auto meshIt = tileState.lodMeshes.find(static_cast<uint8_t>(chosenLod));
        if (meshIt == tileState.lodMeshes.end()) {
            continue;
        }

        selected.push_back(SelectedTileMesh{
            tileCoord,
            static_cast<uint8_t>(chosenLod),
            &meshIt->second
        });
    }

    std::sort(selected.begin(), selected.end(), [](const SelectedTileMesh& a, const SelectedTileMesh& b) {
        if (a.tile == b.tile) {
            return a.lod < b.lod;
        }
        return a.tile < b.tile;
    });

    size_t totalMeshletCount = 0;
    for (const SelectedTileMesh& entry : selected) {
        totalMeshletCount += entry.meshlets->size();
    }

    std::vector<Meshlet> meshlets;
    meshlets.reserve(totalMeshletCount);

    for (const SelectedTileMesh& entry : selected) {
        meshlets.insert(meshlets.end(), entry.meshlets->begin(), entry.meshlets->end());
    }

    return meshlets;
}

uint64_t MeshManager::meshRevision() const noexcept {
    return meshRevision_.load(std::memory_order_acquire);
}

bool MeshManager::hasPendingJobs() const {
    std::shared_lock<std::shared_mutex> lock(meshMutex_);
    return !pendingTileJobs_.empty() || !deferredRemeshTileLods_.empty();
}

int8_t MeshManager::desiredLodForTile(const MeshTileCoord& tileCoord,
                                      const ChunkCoord& centerChunk,
                                      int32_t extraChunks) const {
    const FootprintDistanceRange distances = footprintDistanceRangeForCell(
        tileCoord.x,
        tileCoord.y,
        meshTileSizeChunks_,
        centerChunk
    );

    for (size_t lodIndex = 0; lodIndex < config_.lodChunkRadii.size(); ++lodIndex) {
        const int32_t outerRadiusChunks = std::max(0, config_.lodChunkRadii[lodIndex] + extraChunks);
        const int32_t innerRadiusChunks = (lodIndex == 0)
            ? -1
            : (config_.lodChunkRadii[lodIndex - 1] - extraChunks);

        if (distances.maxDistanceChunks > innerRadiusChunks &&
            distances.minDistanceChunks <= outerRadiusChunks) {
            return static_cast<int8_t>(lodIndex);
        }
    }

    return -1;
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

int8_t MeshManager::chooseRenderableLodForTileLocked(const MeshTileState& state) const {
    auto hasMesh = [&state](int32_t lod) {
        if (lod < 0) {
            return false;
        }
        return state.lodMeshes.find(static_cast<uint8_t>(lod)) != state.lodMeshes.end();
    };

    if (state.desiredLod >= 0 && hasMesh(state.desiredLod)) {
        return state.desiredLod;
    }

    if (state.renderedLod >= 0 && hasMesh(state.renderedLod)) {
        return state.renderedLod;
    }

    const int32_t lodCount = static_cast<int32_t>(config_.lodChunkRadii.size());
    if (state.desiredLod >= 0) {
        for (int32_t lod = static_cast<int32_t>(state.desiredLod) + 1; lod < lodCount; ++lod) {
            if (hasMesh(lod)) {
                return static_cast<int8_t>(lod);
            }
        }
        for (int32_t lod = static_cast<int32_t>(state.desiredLod) - 1; lod >= 0; --lod) {
            if (hasMesh(lod)) {
                return static_cast<int8_t>(lod);
            }
        }
    }

    int8_t coarsest = -1;
    for (const auto& [lod, _] : state.lodMeshes) {
        if (static_cast<int8_t>(lod) > coarsest) {
            coarsest = static_cast<int8_t>(lod);
        }
    }
    return coarsest;
}

void MeshManager::refreshRenderedLodsLocked() {
    for (auto& [_, tileState] : meshTiles_) {
        tileState.renderedLod = chooseRenderableLodForTileLocked(tileState);
    }
}

int32_t MeshManager::maxConfiguredRadius() const {
    if (config_.lodChunkRadii.empty()) {
        return 0;
    }
    return config_.lodChunkRadii.back();
}

uint8_t MeshManager::chunkSpanForLod(uint8_t lodLevel) {
    const uint8_t clamped = std::min<uint8_t>(lodLevel, Chunk::MAX_MIP_LEVEL);
    return static_cast<uint8_t>(1u << clamped);
}

int32_t MeshManager::chunkZCountForLod(uint8_t lodLevel) {
    const int32_t spanChunks = static_cast<int32_t>(chunkSpanForLod(lodLevel));
    return std::max(1, cfg::COLUMN_HEIGHT / spanChunks);
}

jobsystem::Priority MeshManager::priorityFromLodLevel(uint8_t lodLevel) {
    if (lodLevel == 0) {
        return jobsystem::Priority::Critical;
    }
    if (lodLevel == 1) {
        return jobsystem::Priority::High;
    }
    if (lodLevel == 2) {
        return jobsystem::Priority::Normal;
    }
    return jobsystem::Priority::Low;
}

void MeshManager::sanitizeConfig(Config& config) {
    config.lodChunkRadii.erase(
        std::remove_if(config.lodChunkRadii.begin(), config.lodChunkRadii.end(), [](int32_t radius) {
            return radius <= 0;
        }),
        config.lodChunkRadii.end()
    );

    if (config.lodChunkRadii.empty()) {
        config.lodChunkRadii.push_back(4);
    }

    std::sort(config.lodChunkRadii.begin(), config.lodChunkRadii.end());
    config.lodChunkRadii.erase(
        std::unique(config.lodChunkRadii.begin(), config.lodChunkRadii.end()),
        config.lodChunkRadii.end()
    );

    const size_t maxLodLevels = static_cast<size_t>(Chunk::MAX_MIP_LEVEL) + 1;
    if (config.lodChunkRadii.size() > maxLodLevels) {
        config.lodChunkRadii.resize(maxLodLevels);
    }
}
