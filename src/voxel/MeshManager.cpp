#include "solum_engine/voxel/MeshManager.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
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
constexpr int kMinPrefetchChunks = 2;

BlockMaterial airBlock() {
    static const BlockMaterial kAir = UnpackedBlockMaterial{}.pack();
    return kAir;
}

BlockMaterial unknownCullingBlock() {
    static const BlockMaterial kSolid = UnpackedBlockMaterial{1, 0, Direction::PlusZ, 0}.pack();
    return kSolid;
}

BlockMaterial culledSolidBlock() {
    static const BlockMaterial kCulled = UnpackedBlockMaterial{
        ChunkMesher::kCulledSolidBlockId,
        0,
        Direction::PlusZ,
        0
    }.pack();
    return kCulled;
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

constexpr std::array<glm::ivec2, 4> kNeighborColumnOffsets = {
    glm::ivec2{1, 0},
    glm::ivec2{-1, 0},
    glm::ivec2{0, 1},
    glm::ivec2{0, -1}
};

constexpr std::array<glm::ivec2, 9> kRemeshCellOffsets = {
    glm::ivec2{-1, -1}, glm::ivec2{0, -1}, glm::ivec2{1, -1},
    glm::ivec2{-1,  0}, glm::ivec2{0,  0}, glm::ivec2{1,  0},
    glm::ivec2{-1,  1}, glm::ivec2{0,  1}, glm::ivec2{1,  1}
};
}  // namespace

struct MeshManager::MeshGenerationResult {
    LODChunkCoord coord;
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
        scheduleLodRingsAround(centerChunk, centerShiftChunks);
    }

    // Poll for newly generated columns and remesh affected LOD cells.
    scheduleRemeshForNewColumns(centerColumn);
}

void MeshManager::scheduleLodRingsAround(const ChunkCoord& centerChunk, int32_t centerShiftChunks) {
    struct ScheduledChunk {
        int32_t distanceSq = 0;
        LODChunkCoord coord{};
        bool forceRemesh = false;
        int32_t activeWindowExtraChunks = 0;
    };

    std::vector<ScheduledChunk> chunks;

    for (size_t lodIndex = 0; lodIndex < config_.lodChunkRadii.size(); ++lodIndex) {
        const uint8_t lodLevel = static_cast<uint8_t>(lodIndex);
        const int32_t spanChunks = static_cast<int32_t>(chunkSpanForLod(lodLevel));
        const int32_t outerRadiusChunks = std::max(0, config_.lodChunkRadii[lodIndex]);
        const int32_t innerRadiusChunks = (lodIndex == 0)
            ? -1
            : std::max(0, config_.lodChunkRadii[lodIndex - 1]);

        const int32_t prefetchChunks = std::max(kMinPrefetchChunks, centerShiftChunks * spanChunks);
        const int32_t scheduleOuterRadiusChunks = outerRadiusChunks + prefetchChunks;
        const int32_t scheduleInnerRadiusChunks = (lodIndex == 0)
            ? -1
            : (innerRadiusChunks - prefetchChunks);

        const int32_t minCellX = floor_div(
            centerChunk.v.x - scheduleOuterRadiusChunks - (spanChunks - 1),
            spanChunks
        );
        const int32_t maxCellX = floor_div(centerChunk.v.x + scheduleOuterRadiusChunks, spanChunks);
        const int32_t minCellY = floor_div(
            centerChunk.v.y - scheduleOuterRadiusChunks - (spanChunks - 1),
            spanChunks
        );
        const int32_t maxCellY = floor_div(centerChunk.v.y + scheduleOuterRadiusChunks, spanChunks);
        const int32_t zCount = chunkZCountForLod(lodLevel);

        for (int32_t cellY = minCellY; cellY <= maxCellY; ++cellY) {
            for (int32_t cellX = minCellX; cellX <= maxCellX; ++cellX) {
                const FootprintDistanceRange distances = footprintDistanceRangeForCell(
                    cellX,
                    cellY,
                    spanChunks,
                    centerChunk
                );
                if (distances.maxDistanceChunks <= scheduleInnerRadiusChunks ||
                    distances.minDistanceChunks > scheduleOuterRadiusChunks) {
                    continue;
                }

                const bool isVisibleRingCell =
                    distances.maxDistanceChunks > innerRadiusChunks &&
                    distances.minDistanceChunks <= outerRadiusChunks;
                bool forceRemeshVisibleCell = false;
                if (isVisibleRingCell && centerShiftChunks > 0) {
                    bool touchesInnerBoundary = false;
                    if (lodIndex > 0) {
                        const int32_t minInnerChangingDistance = std::max(0, innerRadiusChunks - centerShiftChunks);
                        const int32_t maxInnerChangingDistance = innerRadiusChunks + centerShiftChunks;
                        touchesInnerBoundary =
                            distances.maxDistanceChunks >= minInnerChangingDistance &&
                            distances.minDistanceChunks <= maxInnerChangingDistance;
                    }

                    const int32_t minOuterChangingDistance = std::max(0, outerRadiusChunks - centerShiftChunks);
                    const int32_t maxOuterChangingDistance = outerRadiusChunks + centerShiftChunks;
                    const bool touchesOuterBoundary =
                        distances.maxDistanceChunks >= minOuterChangingDistance &&
                        distances.minDistanceChunks <= maxOuterChangingDistance;

                    forceRemeshVisibleCell = touchesInnerBoundary || touchesOuterBoundary;
                }
                const int32_t distanceSq = distances.minDistanceChunks * distances.minDistanceChunks;

                for (int32_t z = 0; z < zCount; ++z) {
                    chunks.push_back(ScheduledChunk{
                        distanceSq,
                        LODChunkCoord{
                            ChunkCoord{cellX, cellY, z},
                            lodLevel
                        },
                        forceRemeshVisibleCell,
                        prefetchChunks + spanChunks
                    });
                }
            }
        }
    }

    std::sort(chunks.begin(), chunks.end(), [](const ScheduledChunk& a, const ScheduledChunk& b) {
        if (a.distanceSq != b.distanceSq) {
            return a.distanceSq < b.distanceSq;
        }
        return a.coord < b.coord;
    });

    for (const ScheduledChunk& scheduled : chunks) {
        jobsystem::Priority priority = priorityFromDistanceSq(scheduled.distanceSq);
        if (scheduled.forceRemesh &&
            priority != jobsystem::Priority::Critical &&
            priority != jobsystem::Priority::High) {
            priority = jobsystem::Priority::High;
        }

        scheduleChunkMeshing(
            scheduled.coord,
            priority,
            scheduled.forceRemesh,
            centerChunk,
            scheduled.activeWindowExtraChunks
        );
    }
}

void MeshManager::scheduleRemeshForNewColumns(const ColumnCoord& centerColumn) {
    const int32_t radius = std::max(0, maxConfiguredRadius() + 1);
    std::vector<ColumnCoord> generatedColumns;
    const int32_t diameter = (radius * 2) + 1;
    generatedColumns.reserve(static_cast<size_t>(diameter * diameter));

    for (int32_t dy = -radius; dy <= radius; ++dy) {
        for (int32_t dx = -radius; dx <= radius; ++dx) {
            const ColumnCoord coord{
                centerColumn.v.x + dx,
                centerColumn.v.y + dy
            };
            if (world_.isColumnGenerated(coord)) {
                generatedColumns.push_back(coord);
            }
        }
    }

    if (generatedColumns.empty()) {
        return;
    }

    std::unordered_set<LODChunkCoord> chunksToRemesh;
    auto queueColumnLodCells = [this, &chunksToRemesh](const ColumnCoord& columnCoord) {
        for (size_t lodIndex = 0; lodIndex < config_.lodChunkRadii.size(); ++lodIndex) {
            const uint8_t lodLevel = static_cast<uint8_t>(lodIndex);
            const int32_t spanChunks = static_cast<int32_t>(chunkSpanForLod(lodLevel));
            const int32_t baseCellX = floor_div(columnCoord.v.x, spanChunks);
            const int32_t baseCellY = floor_div(columnCoord.v.y, spanChunks);
            const int32_t zCount = chunkZCountForLod(lodLevel);

            for (const glm::ivec2& offset : kRemeshCellOffsets) {
                const int32_t cellX = baseCellX + offset.x;
                const int32_t cellY = baseCellY + offset.y;
                for (int32_t z = 0; z < zCount; ++z) {
                    chunksToRemesh.insert(LODChunkCoord{
                        ChunkCoord{cellX, cellY, z},
                        lodLevel
                    });
                }
            }
        }
    };

    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        for (const ColumnCoord& coord : generatedColumns) {
            if (!knownGeneratedColumns_.insert(coord).second) {
                continue;
            }

            queueColumnLodCells(coord);

            for (const glm::ivec2& offset : kNeighborColumnOffsets) {
                const ColumnCoord neighbor{
                    coord.v.x + offset.x,
                    coord.v.y + offset.y
                };
                if (knownGeneratedColumns_.find(neighbor) != knownGeneratedColumns_.end()) {
                    queueColumnLodCells(neighbor);
                }
            }
        }
    }

    ChunkCoord seamCenterChunk{};
    {
        std::shared_lock<std::shared_mutex> lock(meshMutex_);
        seamCenterChunk = hasLastScheduledCenter_
            ? lastScheduledCenterChunk_
            : ChunkCoord{centerColumn.v.x, centerColumn.v.y, 0};
    }

    for (const LODChunkCoord& lodChunk : chunksToRemesh) {
        const int32_t activeWindowExtraChunks =
            std::max(1, kMinPrefetchChunks + static_cast<int32_t>(chunkSpanForLod(lodChunk.lodLevel)));
        scheduleChunkMeshing(
            lodChunk,
            jobsystem::Priority::Normal,
            true,
            seamCenterChunk,
            activeWindowExtraChunks
        );
    }
}

void MeshManager::scheduleChunkMeshing(const LODChunkCoord& coord,
                                       jobsystem::Priority priority,
                                       bool forceRemesh,
                                       const ChunkCoord& centerChunkForSeams,
                                       int32_t activeWindowExtraChunks) {
    if (!isFootprintGenerated(coord)) {
        return;
    }

    const int32_t clampedActiveWindowExtraChunks = std::max(0, activeWindowExtraChunks);

    bool shouldSchedule = false;
    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        if (pendingMeshJobs_.find(coord) != pendingMeshJobs_.end()) {
            if (forceRemesh) {
                deferredRemeshChunks_.insert(coord);
            }
            return;
        }

        if (!isWithinActiveWindowLocked(coord, clampedActiveWindowExtraChunks)) {
            return;
        }

        if (!forceRemesh && chunkMeshes_.find(coord) != chunkMeshes_.end()) {
            return;
        }

        pendingMeshJobs_.insert(coord);
        shouldSchedule = true;
    }

    if (!shouldSchedule) {
        return;
    }

    try {
        jobs_.schedule(
            priority,
            [this, coord, centerChunkForSeams, clampedActiveWindowExtraChunks]() -> MeshGenerationResult {
                {
                    std::shared_lock<std::shared_mutex> lock(meshMutex_);
                    if (!isWithinActiveWindowLocked(coord, clampedActiveWindowExtraChunks)) {
                        return MeshGenerationResult{coord, {}, false};
                    }
                }

                if (!isFootprintGenerated(coord)) {
                    return MeshGenerationResult{coord, {}, false};
                }

                const uint8_t mipLevel = coord.lodLevel;
                const uint8_t voxelScale = static_cast<uint8_t>(1u << mipLevel);

                ChunkMesher mesher;
                const BlockCoord sectionOriginMip{
                    coord.coord.v.x * cfg::CHUNK_SIZE,
                    coord.coord.v.y * cfg::CHUNK_SIZE,
                    coord.coord.v.z * cfg::CHUNK_SIZE
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
                                // Keep vertical world bounds exposed, but suppress faces against
                                // not-yet-generated lateral neighbors to avoid edge flicker/remesh churn.
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

                {
                    const size_t lodIndex = static_cast<size_t>(coord.lodLevel);
                    if (lodIndex < config_.lodChunkRadii.size()) {
                        const int32_t innerRadiusChunks = (lodIndex == 0)
                            ? -1
                            : std::max(0, config_.lodChunkRadii[lodIndex - 1]);
                        const int32_t outerRadiusChunks = std::max(0, config_.lodChunkRadii[lodIndex]);
                        const int32_t chunkSizeAtMip = std::max(1, cfg::CHUNK_SIZE >> mipLevel);

                        std::array<int32_t, kPaddedChunkExtent> chunkXForPaddedX{};
                        std::array<int32_t, kPaddedChunkExtent> chunkYForPaddedY{};
                        for (int x = 0; x < kPaddedChunkExtent; ++x) {
                            chunkXForPaddedX[static_cast<size_t>(x)] =
                                floor_div(paddedOriginMip.v.x + x, chunkSizeAtMip);
                        }
                        for (int y = 0; y < kPaddedChunkExtent; ++y) {
                            chunkYForPaddedY[static_cast<size_t>(y)] =
                                floor_div(paddedOriginMip.v.y + y, chunkSizeAtMip);
                        }

                        for (int x = 1; x <= cfg::CHUNK_SIZE; ++x) {
                            const int32_t chunkX = chunkXForPaddedX[static_cast<size_t>(x)];
                            for (int y = 1; y <= cfg::CHUNK_SIZE; ++y) {
                                const int32_t chunkY = chunkYForPaddedY[static_cast<size_t>(y)];
                                const int32_t distanceChunks = std::max(
                                    std::abs(chunkX - centerChunkForSeams.v.x),
                                    std::abs(chunkY - centerChunkForSeams.v.y)
                                );
                                const bool carveForInnerBoundary =
                                    innerRadiusChunks >= 0 && distanceChunks <= innerRadiusChunks;
                                const bool carveForOuterBoundary = distanceChunks > outerRadiusChunks;
                                if (!carveForInnerBoundary && !carveForOuterBoundary) {
                                    continue;
                                }

                                for (int z = 1; z <= cfg::CHUNK_SIZE; ++z) {
                                    snapshot.blocks[static_cast<size_t>(PaddedChunkBlockSource::index(x, y, z))] =
                                        culledSolidBlock();
                                }
                            }
                        }
                    }
                }

                const glm::ivec3 sectionExtent{kChunkExtent, kChunkExtent, kChunkExtent};
                const glm::ivec3 meshletOrigin{
                    sectionOriginMip.v.x * voxelScale,
                    sectionOriginMip.v.y * voxelScale,
                    sectionOriginMip.v.z * voxelScale
                };
                std::vector<Meshlet> meshlets = mesher.mesh(
                    snapshot,
                    sectionOriginMip,
                    sectionExtent,
                    meshletOrigin,
                    voxelScale
                );

                return MeshGenerationResult{
                    coord,
                    std::move(meshlets),
                    true
                };
            },
            [this, coord](jobsystem::JobResult<MeshGenerationResult>&& result) {
                if (!result.success()) {
                    std::unique_lock<std::shared_mutex> lock(meshMutex_);
                    pendingMeshJobs_.erase(coord);
                    deferredRemeshChunks_.erase(coord);
                    return;
                }

                MeshGenerationResult meshResult = std::move(result).value();
                if (!meshResult.meshed) {
                    std::unique_lock<std::shared_mutex> lock(meshMutex_);
                    pendingMeshJobs_.erase(coord);
                    deferredRemeshChunks_.erase(coord);
                    return;
                }
                onChunkMeshed(meshResult.coord, std::move(meshResult.meshlets));
            }
        );
    } catch (const std::exception&) {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        pendingMeshJobs_.erase(coord);
        deferredRemeshChunks_.erase(coord);
    }
}

void MeshManager::onChunkMeshed(const LODChunkCoord& coord, std::vector<Meshlet>&& meshlets) {
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return;
    }

    bool needsDeferredRemesh = false;
    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        pendingMeshJobs_.erase(coord);
        chunkMeshes_[coord] = std::move(meshlets);

        auto deferredIt = deferredRemeshChunks_.find(coord);
        if (deferredIt != deferredRemeshChunks_.end()) {
            deferredRemeshChunks_.erase(deferredIt);
            needsDeferredRemesh = true;
        }
    }

    meshRevision_.fetch_add(1, std::memory_order_acq_rel);

    if (needsDeferredRemesh) {
        ChunkCoord seamCenterChunk{};
        {
            std::shared_lock<std::shared_mutex> lock(meshMutex_);
            const int32_t spanChunks = static_cast<int32_t>(chunkSpanForLod(coord.lodLevel));
            seamCenterChunk = hasLastScheduledCenter_
                ? lastScheduledCenterChunk_
                : ChunkCoord{
                      coord.coord.v.x * spanChunks,
                      coord.coord.v.y * spanChunks,
                      0
                  };
        }
        const int32_t activeWindowExtraChunks =
            std::max(1, kMinPrefetchChunks + static_cast<int32_t>(chunkSpanForLod(coord.lodLevel)));
        scheduleChunkMeshing(
            coord,
            jobsystem::Priority::High,
            true,
            seamCenterChunk,
            activeWindowExtraChunks
        );
    }
}

std::vector<Meshlet> MeshManager::copyMeshlets() const {
    std::shared_lock<std::shared_mutex> lock(meshMutex_);

    std::vector<const std::pair<const LODChunkCoord, std::vector<Meshlet>>*> orderedChunks;
    orderedChunks.reserve(chunkMeshes_.size());
    for (const auto& entry : chunkMeshes_) {
        orderedChunks.push_back(&entry);
    }

    std::sort(orderedChunks.begin(), orderedChunks.end(), [](const auto* a, const auto* b) {
        return a->first < b->first;
    });

    size_t totalMeshletCount = 0;
    for (const auto* chunkEntry : orderedChunks) {
        totalMeshletCount += chunkEntry->second.size();
    }

    std::vector<Meshlet> meshlets;
    meshlets.reserve(totalMeshletCount);

    for (const auto* chunkEntry : orderedChunks) {
        meshlets.insert(meshlets.end(), chunkEntry->second.begin(), chunkEntry->second.end());
    }

    return meshlets;
}

std::vector<Meshlet> MeshManager::copyMeshletsAround(const ColumnCoord& centerColumn, int32_t columnRadius) const {
    std::shared_lock<std::shared_mutex> lock(meshMutex_);

    std::vector<const std::pair<const LODChunkCoord, std::vector<Meshlet>>*> orderedChunks;
    orderedChunks.reserve(chunkMeshes_.size());
    const int32_t clampedRadius = std::max(0, columnRadius);
    const ChunkCoord centerChunk{centerColumn.v.x, centerColumn.v.y, 0};

    const int32_t minColumnX = centerColumn.v.x - clampedRadius;
    const int32_t maxColumnX = centerColumn.v.x + clampedRadius;
    const int32_t minColumnY = centerColumn.v.y - clampedRadius;
    const int32_t maxColumnY = centerColumn.v.y + clampedRadius;

    for (const auto& entry : chunkMeshes_) {
        if (!isInDesiredRingForCenter(entry.first, centerChunk, 0)) {
            continue;
        }

        const uint8_t lodLevel = entry.first.lodLevel;
        const int32_t spanChunks = static_cast<int32_t>(chunkSpanForLod(lodLevel));
        const int32_t meshMinX = entry.first.coord.v.x * spanChunks;
        const int32_t meshMaxX = meshMinX + spanChunks - 1;
        const int32_t meshMinY = entry.first.coord.v.y * spanChunks;
        const int32_t meshMaxY = meshMinY + spanChunks - 1;

        if (meshMaxX < minColumnX || meshMinX > maxColumnX ||
            meshMaxY < minColumnY || meshMinY > maxColumnY) {
            continue;
        }

        orderedChunks.push_back(&entry);
    }

    std::sort(orderedChunks.begin(), orderedChunks.end(), [](const auto* a, const auto* b) {
        return a->first < b->first;
    });

    size_t totalMeshletCount = 0;
    for (const auto* chunkEntry : orderedChunks) {
        totalMeshletCount += chunkEntry->second.size();
    }

    std::vector<Meshlet> meshlets;
    meshlets.reserve(totalMeshletCount);

    for (const auto* chunkEntry : orderedChunks) {
        meshlets.insert(meshlets.end(), chunkEntry->second.begin(), chunkEntry->second.end());
    }

    return meshlets;
}

uint64_t MeshManager::meshRevision() const noexcept {
    return meshRevision_.load(std::memory_order_acquire);
}

bool MeshManager::hasPendingJobs() const {
    std::shared_lock<std::shared_mutex> lock(meshMutex_);
    return !pendingMeshJobs_.empty() || !deferredRemeshChunks_.empty();
}

bool MeshManager::isInDesiredRingForCenter(const LODChunkCoord& coord,
                                           const ChunkCoord& centerChunk,
                                           int32_t extraChunks) const {
    const size_t lodIndex = static_cast<size_t>(coord.lodLevel);
    if (lodIndex >= config_.lodChunkRadii.size()) {
        return false;
    }

    const int32_t spanChunks = static_cast<int32_t>(chunkSpanForLod(coord.lodLevel));
    const int32_t zCount = chunkZCountForLod(coord.lodLevel);
    if (coord.coord.v.z < 0 || coord.coord.v.z >= zCount) {
        return false;
    }

    const int32_t outerRadiusChunks = std::max(0, config_.lodChunkRadii[lodIndex] + extraChunks);
    const int32_t innerRadiusChunks = (lodIndex == 0)
        ? -1
        : (config_.lodChunkRadii[lodIndex - 1] - extraChunks);

    const FootprintDistanceRange distances = footprintDistanceRangeForCell(
        coord.coord.v.x,
        coord.coord.v.y,
        spanChunks,
        centerChunk
    );
    return distances.maxDistanceChunks > innerRadiusChunks &&
           distances.minDistanceChunks <= outerRadiusChunks;
}

bool MeshManager::isWithinActiveWindowLocked(const LODChunkCoord& coord, int32_t extraChunks) const {
    if (!hasLastScheduledCenter_) {
        return true;
    }
    return isInDesiredRingForCenter(coord, lastScheduledCenterChunk_, extraChunks);
}

bool MeshManager::isFootprintGenerated(const LODChunkCoord& coord) const {
    const int32_t spanChunks = static_cast<int32_t>(chunkSpanForLod(coord.lodLevel));
    const int32_t baseChunkX = coord.coord.v.x * spanChunks;
    const int32_t baseChunkY = coord.coord.v.y * spanChunks;

    for (int32_t dy = 0; dy < spanChunks; ++dy) {
        for (int32_t dx = 0; dx < spanChunks; ++dx) {
            if (!world_.isColumnGenerated(ColumnCoord{baseChunkX + dx, baseChunkY + dy})) {
                return false;
            }
        }
    }

    return true;
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

jobsystem::Priority MeshManager::priorityFromDistanceSq(int32_t distanceSq) {
    if (distanceSq <= 0) {
        return jobsystem::Priority::Critical;
    }
    if (distanceSq <= 4) {
        return jobsystem::Priority::High;
    }
    if (distanceSq <= 25) {
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
