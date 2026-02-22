#include "solum_engine/voxel/World.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <utility>

#include "solum_engine/resources/Constants.h"
#include "solum_engine/voxel/Column.h"
#include "solum_engine/voxel/Region.h"
#include "solum_engine/voxel/TerrainGenerator.h"

namespace {
constexpr int kChunkExtent = cfg::CHUNK_SIZE;
constexpr int kPaddedChunkExtent = cfg::CHUNK_SIZE + 2;
constexpr int kPaddedChunkArea = kPaddedChunkExtent * kPaddedChunkExtent;
constexpr int kPaddedChunkVoxelCount = kPaddedChunkExtent * kPaddedChunkExtent * kPaddedChunkExtent;

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
}  // namespace

struct World::ColumnGenerationResult {
    ColumnCoord coord;
    Column column;
    bool generated = false;
};

struct World::MeshGenerationResult {
    ChunkCoord coord;
    std::vector<Meshlet> meshlets;
    bool meshed = false;
};

WorldSection::WorldSection(const World& world, const BlockCoord& origin, const glm::ivec3& extent)
    : world_(world), origin_(origin), extent_(extent) {}

BlockMaterial WorldSection::getBlock(const BlockCoord& coord) const {
    return world_.getBlock(coord);
}

BlockMaterial WorldSection::getLocalBlock(int32_t x, int32_t y, int32_t z) const {
    return world_.getBlock(BlockCoord{
        origin_.v.x + x,
        origin_.v.y + y,
        origin_.v.z + z
    });
}

World::World()
    : World(Config{}) {}

World::World(Config config)
    : config_(std::move(config)),
      jobs_(config_.jobConfig) {}

World::~World() {
    shuttingDown_.store(true, std::memory_order_release);
    jobs_.wait_for_idle();
    jobs_.stop();
}

void World::waitForIdle() {
    jobs_.wait_for_idle();
}

BlockMaterial World::getBlock(const BlockCoord& coord) const {
    BlockMaterial block = airBlock();
    tryGetBlock(coord, block);
    return block;
}

bool World::tryGetBlock(const BlockCoord& coord, BlockMaterial& outBlock) const {
    std::shared_lock<std::shared_mutex> lock(worldMutex_);
    return tryGetBlockLocked(coord, outBlock);
}

bool World::tryGetBlockLocked(const BlockCoord& coord, BlockMaterial& outBlock) const {
    if (coord.v.z < 0 || coord.v.z >= cfg::COLUMN_HEIGHT_BLOCKS) {
        outBlock = airBlock();
        return false;
    }

    const ChunkCoord chunkCoord = block_to_chunk(coord);
    if (chunkCoord.v.z < 0 || chunkCoord.v.z >= cfg::COLUMN_HEIGHT) {
        outBlock = airBlock();
        return false;
    }

    const ColumnCoord columnCoord = chunk_to_column(chunkCoord);
    const RegionCoord regionCoord = column_to_region(columnCoord);

    // A region may exist while many of its columns are still ungenerated.
    // Treat those columns as unknown so meshing can apply boundary policy.
    if (generatedColumns_.find(columnCoord) == generatedColumns_.end()) {
        outBlock = airBlock();
        return false;
    }

    const auto regionIt = regions_.find(regionCoord);
    if (regionIt == regions_.end() || regionIt->second == nullptr) {
        outBlock = airBlock();
        return false;
    }

    const glm::ivec2 localColumn = column_local_in_region(columnCoord);
    const glm::ivec3 localBlock = block_local_in_chunk(coord);
    const Column& column = regionIt->second->getColumn(
        static_cast<uint8_t>(localColumn.x),
        static_cast<uint8_t>(localColumn.y)
    );

    outBlock = column.getChunk(static_cast<uint8_t>(chunkCoord.v.z)).getBlock(
        static_cast<uint8_t>(localBlock.x),
        static_cast<uint8_t>(localBlock.y),
        static_cast<uint8_t>(localBlock.z)
    );
    return true;
}

WorldSection World::createSection(const BlockCoord& origin, const glm::ivec3& extent) const {
    return WorldSection(*this, origin, extent);
}

void World::updatePlayerPosition(const glm::vec3& playerWorldPosition) {
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return;
    }

    const BlockCoord playerBlock{
        static_cast<int32_t>(std::floor(playerWorldPosition.x)),
        static_cast<int32_t>(std::floor(playerWorldPosition.y)),
        static_cast<int32_t>(std::floor(playerWorldPosition.z))
    };
    const ColumnCoord centerColumn = chunk_to_column(block_to_chunk(playerBlock));

    ColumnCoord previousCenter{};
    bool hadPreviousCenter = false;
    {
        std::unique_lock<std::shared_mutex> lock(worldMutex_);
        if (hasLastScheduledCenter_ && centerColumn == lastScheduledCenter_) {
            return;
        }

        hadPreviousCenter = hasLastScheduledCenter_;
        previousCenter = lastScheduledCenter_;
        lastScheduledCenter_ = centerColumn;
        hasLastScheduledCenter_ = true;
    }

    if (!hadPreviousCenter) {
        scheduleColumnsAround(centerColumn);
        return;
    }

    scheduleColumnsDelta(previousCenter, centerColumn);
}

void World::scheduleColumnsAround(const ColumnCoord& centerColumn) {
    const int32_t radius = std::max(0, config_.columnLoadRadius);

    std::vector<std::pair<int32_t, ColumnCoord>> columns;
    const int32_t diameter = (radius * 2) + 1;
    columns.reserve(static_cast<size_t>(diameter * diameter));

    for (int32_t dy = -radius; dy <= radius; ++dy) {
        for (int32_t dx = -radius; dx <= radius; ++dx) {
            const ColumnCoord coord{
                centerColumn.v.x + dx,
                centerColumn.v.y + dy
            };
            const int32_t distanceSq = (dx * dx) + (dy * dy);
            columns.emplace_back(distanceSq, coord);
        }
    }

    std::sort(columns.begin(), columns.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    for (const auto& [distanceSq, coord] : columns) {
        scheduleColumnGeneration(coord, priorityFromDistanceSq(distanceSq));
    }
}

void World::scheduleColumnsDelta(const ColumnCoord& previousCenter, const ColumnCoord& newCenter) {
    const int32_t radius = std::max(0, config_.columnLoadRadius);
    const int32_t previousMinX = previousCenter.v.x - radius;
    const int32_t previousMaxX = previousCenter.v.x + radius;
    const int32_t previousMinY = previousCenter.v.y - radius;
    const int32_t previousMaxY = previousCenter.v.y + radius;

    const int32_t newMinX = newCenter.v.x - radius;
    const int32_t newMaxX = newCenter.v.x + radius;
    const int32_t newMinY = newCenter.v.y - radius;
    const int32_t newMaxY = newCenter.v.y + radius;

    const bool noOverlap =
        newMaxX < previousMinX || newMinX > previousMaxX ||
        newMaxY < previousMinY || newMinY > previousMaxY;
    if (noOverlap) {
        scheduleColumnsAround(newCenter);
        return;
    }

    for (int32_t y = newMinY; y <= newMaxY; ++y) {
        for (int32_t x = newMinX; x <= newMaxX; ++x) {
            if (x >= previousMinX && x <= previousMaxX &&
                y >= previousMinY && y <= previousMaxY) {
                continue;
            }

            const ColumnCoord coord{x, y};
            const int32_t dx = x - newCenter.v.x;
            const int32_t dy = y - newCenter.v.y;
            const int32_t distanceSq = (dx * dx) + (dy * dy);
            scheduleColumnGeneration(coord, priorityFromDistanceSq(distanceSq));
        }
    }
}

void World::scheduleColumnGeneration(const ColumnCoord& coord, jobsystem::Priority priority) {
    bool shouldSchedule = false;
    {
        std::unique_lock<std::shared_mutex> lock(worldMutex_);
        if (!isColumnGeneratedLocked(coord) && pendingColumnJobs_.find(coord) == pendingColumnJobs_.end()) {
            if (!isWithinActiveWindowLocked(coord, 1)) {
                return;
            }
            pendingColumnJobs_.insert(coord);
            shouldSchedule = true;
        }
    }

    if (!shouldSchedule) {
        return;
    }

    try {
        jobs_.schedule(
            priority,
            [this, coord]() -> ColumnGenerationResult {
                {
                    std::shared_lock<std::shared_mutex> lock(worldMutex_);
                    if (!isWithinActiveWindowLocked(coord, 1)) {
                        return ColumnGenerationResult{
                            coord,
                            Column{},
                            false
                        };
                    }
                }

                TerrainGenerator generator;
                Column generatedColumn;

                const ChunkCoord columnBaseChunk = column_local_to_chunk(coord, 0);
                const BlockCoord columnOrigin = chunk_to_block_origin(columnBaseChunk);
                generator.generateColumn(columnOrigin.v, generatedColumn);

                return ColumnGenerationResult{
                    coord,
                    std::move(generatedColumn),
                    true
                };
            },
            [this, coord](jobsystem::JobResult<ColumnGenerationResult>&& result) {
                if (!result.success()) {
                    std::unique_lock<std::shared_mutex> lock(worldMutex_);
                    pendingColumnJobs_.erase(coord);
                    return;
                }

                ColumnGenerationResult generated = std::move(result).value();
                if (!generated.generated) {
                    std::unique_lock<std::shared_mutex> lock(worldMutex_);
                    pendingColumnJobs_.erase(coord);
                    return;
                }
                onColumnGenerated(generated.coord, std::move(generated.column));
            }
        );
    } catch (const std::exception&) {
        std::unique_lock<std::shared_mutex> lock(worldMutex_);
        pendingColumnJobs_.erase(coord);
    }
}

void World::onColumnGenerated(const ColumnCoord& coord, Column&& column) {
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return;
    }

    std::unordered_set<ChunkCoord> chunksToRemesh;
    {
        std::unique_lock<std::shared_mutex> lock(worldMutex_);
        pendingColumnJobs_.erase(coord);

        Region* region = getOrCreateRegionLocked(column_to_region(coord));
        if (region == nullptr) {
            return;
        }

        const glm::ivec2 localColumn = column_local_in_region(coord);
        region->getColumn(
            static_cast<uint8_t>(localColumn.x),
            static_cast<uint8_t>(localColumn.y)
        ) = std::move(column);

        generatedColumns_.insert(coord);

        auto queueColumnChunks = [&chunksToRemesh](const ColumnCoord& columnCoord) {
            for (int32_t chunkZ = 0; chunkZ < cfg::COLUMN_HEIGHT; ++chunkZ) {
                chunksToRemesh.insert(column_local_to_chunk(columnCoord, chunkZ));
            }
        };

        queueColumnChunks(coord);

        static constexpr std::array<glm::ivec2, 4> kNeighbors = {
            glm::ivec2{1, 0},
            glm::ivec2{-1, 0},
            glm::ivec2{0, 1},
            glm::ivec2{0, -1}
        };

        for (const glm::ivec2& neighborOffset : kNeighbors) {
            const ColumnCoord neighbor{
                coord.v.x + neighborOffset.x,
                coord.v.y + neighborOffset.y
            };
            if (generatedColumns_.find(neighbor) != generatedColumns_.end()) {
                queueColumnChunks(neighbor);
            }
        }
    }

    for (const ChunkCoord& chunkCoord : chunksToRemesh) {
        scheduleChunkMeshing(chunkCoord, jobsystem::Priority::Normal, true);
    }
}

void World::scheduleChunkMeshing(const ChunkCoord& coord,
                                 jobsystem::Priority priority,
                                 bool forceRemesh) {
    bool shouldSchedule = false;
    {
        std::unique_lock<std::shared_mutex> lock(worldMutex_);
        if (pendingMeshJobs_.find(coord) != pendingMeshJobs_.end()) {
            if (forceRemesh) {
                deferredRemeshChunks_.insert(coord);
            }
            return;
        }

        if (!isColumnGeneratedLocked(chunk_to_column(coord))) {
            return;
        }

        if (!isWithinActiveWindowLocked(chunk_to_column(coord), 1)) {
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
            [this, coord]() -> MeshGenerationResult {
                {
                    std::shared_lock<std::shared_mutex> lock(worldMutex_);
                    if (!isWithinActiveWindowLocked(chunk_to_column(coord), 1)) {
                        return MeshGenerationResult{
                            coord,
                            {},
                            false
                        };
                    }
                }

                ChunkMesher mesher;
                const BlockCoord chunkOrigin = chunk_to_block_origin(coord);
                const BlockCoord paddedOrigin{
                    chunkOrigin.v.x - 1,
                    chunkOrigin.v.y - 1,
                    chunkOrigin.v.z - 1
                };

                PaddedChunkBlockSource snapshot;
                snapshot.origin = paddedOrigin;
                snapshot.blocks.fill(airBlock());

                {
                    std::shared_lock<std::shared_mutex> lock(worldMutex_);
                    for (int x = 0; x < kPaddedChunkExtent; ++x) {
                        for (int y = 0; y < kPaddedChunkExtent; ++y) {
                            for (int z = 0; z < kPaddedChunkExtent; ++z) {
                                const BlockCoord coordToCopy{
                                    paddedOrigin.v.x + x,
                                    paddedOrigin.v.y + y,
                                    paddedOrigin.v.z + z
                                };

                                BlockMaterial block = airBlock();
                                if (!tryGetBlockLocked(coordToCopy, block)) {
                                    // Keep vertical world bounds exposed, but suppress faces against
                                    // not-yet-generated lateral neighbors to avoid edge flicker/remesh churn.
                                    if (coordToCopy.v.z >= 0 && coordToCopy.v.z < cfg::COLUMN_HEIGHT_BLOCKS) {
                                        block = unknownCullingBlock();
                                    } else {
                                        block = airBlock();
                                    }
                                }
                                snapshot.blocks[static_cast<size_t>(PaddedChunkBlockSource::index(x, y, z))] = block;
                            }
                        }
                    }
                }

                const glm::ivec3 chunkExtent{kChunkExtent, kChunkExtent, kChunkExtent};
                std::vector<Meshlet> meshlets = mesher.mesh(
                    snapshot,
                    chunkOrigin,
                    chunkExtent,
                    chunkOrigin.v
                );

                return MeshGenerationResult{
                    coord,
                    std::move(meshlets),
                    true
                };
            },
            [this, coord](jobsystem::JobResult<MeshGenerationResult>&& result) {
                if (!result.success()) {
                    std::unique_lock<std::shared_mutex> lock(worldMutex_);
                    pendingMeshJobs_.erase(coord);
                    deferredRemeshChunks_.erase(coord);
                    return;
                }

                MeshGenerationResult meshResult = std::move(result).value();
                if (!meshResult.meshed) {
                    std::unique_lock<std::shared_mutex> lock(worldMutex_);
                    pendingMeshJobs_.erase(coord);
                    deferredRemeshChunks_.erase(coord);
                    return;
                }
                onChunkMeshed(meshResult.coord, std::move(meshResult.meshlets));
            }
        );
    } catch (const std::exception&) {
        std::unique_lock<std::shared_mutex> lock(worldMutex_);
        pendingMeshJobs_.erase(coord);
        deferredRemeshChunks_.erase(coord);
    }
}

void World::onChunkMeshed(const ChunkCoord& coord, std::vector<Meshlet>&& meshlets) {
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return;
    }

    bool needsDeferredRemesh = false;
    {
        std::unique_lock<std::shared_mutex> lock(worldMutex_);
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
        scheduleChunkMeshing(coord, jobsystem::Priority::High, true);
    }
}

std::vector<Meshlet> World::copyMeshlets() const {
    std::shared_lock<std::shared_mutex> lock(worldMutex_);

    std::vector<const std::pair<const ChunkCoord, std::vector<Meshlet>>*> orderedChunks;
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

std::vector<Meshlet> World::copyMeshletsAround(const ColumnCoord& centerColumn, int32_t columnRadius) const {
    std::shared_lock<std::shared_mutex> lock(worldMutex_);

    std::vector<const std::pair<const ChunkCoord, std::vector<Meshlet>>*> orderedChunks;
    orderedChunks.reserve(chunkMeshes_.size());
    const int32_t clampedRadius = std::max(0, columnRadius);

    for (const auto& entry : chunkMeshes_) {
        const ColumnCoord column = chunk_to_column(entry.first);
        const int32_t dx = std::abs(column.v.x - centerColumn.v.x);
        const int32_t dy = std::abs(column.v.y - centerColumn.v.y);
        if (dx > clampedRadius || dy > clampedRadius) {
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

uint64_t World::meshRevision() const noexcept {
    return meshRevision_.load(std::memory_order_acquire);
}

bool World::hasPendingJobs() const {
    std::shared_lock<std::shared_mutex> lock(worldMutex_);
    return !pendingColumnJobs_.empty() || !pendingMeshJobs_.empty() || !deferredRemeshChunks_.empty();
}

bool World::isColumnGeneratedLocked(const ColumnCoord& coord) const {
    return generatedColumns_.find(coord) != generatedColumns_.end();
}

bool World::isWithinActiveWindowLocked(const ColumnCoord& coord, int32_t extraRadius) const {
    if (!hasLastScheduledCenter_) {
        return true;
    }

    const int32_t radius = std::max(0, config_.columnLoadRadius + extraRadius);
    const int32_t dx = std::abs(coord.v.x - lastScheduledCenter_.v.x);
    const int32_t dy = std::abs(coord.v.y - lastScheduledCenter_.v.y);
    return dx <= radius && dy <= radius;
}

Region* World::getOrCreateRegionLocked(const RegionCoord& coord) {
    auto it = regions_.find(coord);
    if (it != regions_.end()) {
        return it->second.get();
    }

    auto [insertedIt, inserted] = regions_.emplace(coord, std::make_unique<Region>(coord));
    if (!inserted) {
        std::cerr << "Failed to insert region at " << coord << '\n';
        return nullptr;
    }
    return insertedIt->second.get();
}

jobsystem::Priority World::priorityFromDistanceSq(int32_t distanceSq) {
    if (distanceSq <= 0) {
        return jobsystem::Priority::Critical;
    }
    if (distanceSq <= 2) {
        return jobsystem::Priority::High;
    }
    if (distanceSq <= 8) {
        return jobsystem::Priority::Normal;
    }
    return jobsystem::Priority::Low;
}
