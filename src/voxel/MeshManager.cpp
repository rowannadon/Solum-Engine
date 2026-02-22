#include "solum_engine/voxel/MeshManager.h"

#include <algorithm>
#include <array>
#include <cmath>
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

constexpr std::array<glm::ivec2, 4> kNeighborOffsets = {
    glm::ivec2{1, 0},
    glm::ivec2{-1, 0},
    glm::ivec2{0, 1},
    glm::ivec2{0, -1}
};
}  // namespace

struct MeshManager::MeshGenerationResult {
    ChunkCoord coord;
    std::vector<Meshlet> meshlets;
    bool meshed = false;
};

MeshManager::MeshManager(const World& world)
    : MeshManager(world, Config{}) {}

MeshManager::MeshManager(const World& world, Config config)
    : world_(world),
      config_(std::move(config)),
      jobs_(config_.jobConfig) {}

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
    const ColumnCoord centerColumn = chunk_to_column(block_to_chunk(playerBlock));

    ColumnCoord previousCenter{};
    bool hadPreviousCenter = false;
    bool centerChanged = false;
    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        if (!hasLastScheduledCenter_ || !(centerColumn == lastScheduledCenter_)) {
            hadPreviousCenter = hasLastScheduledCenter_;
            previousCenter = lastScheduledCenter_;
            lastScheduledCenter_ = centerColumn;
            hasLastScheduledCenter_ = true;
            centerChanged = true;
        }
    }

    if (centerChanged) {
        if (!hadPreviousCenter) {
            scheduleChunksAround(centerColumn);
        } else {
            scheduleChunksDelta(previousCenter, centerColumn);
        }
    }

    // Poll for columns that finished generation since the previous frame so
    // chunk boundaries can be remeshed once both sides exist.
    scheduleRemeshForNewColumns(centerColumn);
}

void MeshManager::scheduleChunksAround(const ColumnCoord& centerColumn) {
    const int32_t radius = std::max(0, config_.columnMeshRadius);

    std::vector<std::pair<int32_t, ChunkCoord>> chunks;
    const int32_t diameter = (radius * 2) + 1;
    chunks.reserve(static_cast<size_t>(diameter * diameter * cfg::COLUMN_HEIGHT));

    for (int32_t dy = -radius; dy <= radius; ++dy) {
        for (int32_t dx = -radius; dx <= radius; ++dx) {
            const int32_t distanceSq = (dx * dx) + (dy * dy);
            const ColumnCoord column{
                centerColumn.v.x + dx,
                centerColumn.v.y + dy
            };
            for (int32_t chunkZ = 0; chunkZ < cfg::COLUMN_HEIGHT; ++chunkZ) {
                chunks.emplace_back(distanceSq, column_local_to_chunk(column, chunkZ));
            }
        }
    }

    std::sort(chunks.begin(), chunks.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    for (const auto& [distanceSq, coord] : chunks) {
        scheduleChunkMeshing(coord, priorityFromDistanceSq(distanceSq), false);
    }
}

void MeshManager::scheduleChunksDelta(const ColumnCoord& previousCenter, const ColumnCoord& newCenter) {
    const int32_t radius = std::max(0, config_.columnMeshRadius);
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
        scheduleChunksAround(newCenter);
        return;
    }

    for (int32_t y = newMinY; y <= newMaxY; ++y) {
        for (int32_t x = newMinX; x <= newMaxX; ++x) {
            if (x >= previousMinX && x <= previousMaxX &&
                y >= previousMinY && y <= previousMaxY) {
                continue;
            }

            const int32_t dx = x - newCenter.v.x;
            const int32_t dy = y - newCenter.v.y;
            const int32_t distanceSq = (dx * dx) + (dy * dy);
            const ColumnCoord column{x, y};
            for (int32_t chunkZ = 0; chunkZ < cfg::COLUMN_HEIGHT; ++chunkZ) {
                scheduleChunkMeshing(
                    column_local_to_chunk(column, chunkZ),
                    priorityFromDistanceSq(distanceSq),
                    false
                );
            }
        }
    }
}

void MeshManager::scheduleRemeshForNewColumns(const ColumnCoord& centerColumn) {
    const int32_t radius = std::max(0, config_.columnMeshRadius + 1);
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

    std::unordered_set<ChunkCoord> chunksToRemesh;
    auto queueColumnChunks = [&chunksToRemesh](const ColumnCoord& columnCoord) {
        for (int32_t chunkZ = 0; chunkZ < cfg::COLUMN_HEIGHT; ++chunkZ) {
            chunksToRemesh.insert(column_local_to_chunk(columnCoord, chunkZ));
        }
    };

    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        for (const ColumnCoord& coord : generatedColumns) {
            if (!knownGeneratedColumns_.insert(coord).second) {
                continue;
            }

            queueColumnChunks(coord);

            for (const glm::ivec2& offset : kNeighborOffsets) {
                const ColumnCoord neighbor{
                    coord.v.x + offset.x,
                    coord.v.y + offset.y
                };
                if (knownGeneratedColumns_.find(neighbor) != knownGeneratedColumns_.end()) {
                    queueColumnChunks(neighbor);
                }
            }
        }
    }

    for (const ChunkCoord& chunkCoord : chunksToRemesh) {
        scheduleChunkMeshing(chunkCoord, jobsystem::Priority::Normal, true);
    }
}

void MeshManager::scheduleChunkMeshing(const ChunkCoord& coord,
                                       jobsystem::Priority priority,
                                       bool forceRemesh) {
    if (!world_.isColumnGenerated(chunk_to_column(coord))) {
        return;
    }

    bool shouldSchedule = false;
    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        if (pendingMeshJobs_.find(coord) != pendingMeshJobs_.end()) {
            if (forceRemesh) {
                deferredRemeshChunks_.insert(coord);
            }
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
                    std::shared_lock<std::shared_mutex> lock(meshMutex_);
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

                const glm::ivec3 paddedExtent{
                    kPaddedChunkExtent,
                    kPaddedChunkExtent,
                    kPaddedChunkExtent
                };
                WorldSection worldSection = world_.createSection(paddedOrigin, paddedExtent);
                std::vector<WorldSection::Sample> sectionSamples;
                worldSection.copySamples(sectionSamples);
                const size_t paddedYZArea = static_cast<size_t>(kPaddedChunkExtent) * static_cast<size_t>(kPaddedChunkExtent);

                for (int x = 0; x < kPaddedChunkExtent; ++x) {
                    for (int y = 0; y < kPaddedChunkExtent; ++y) {
                        for (int z = 0; z < kPaddedChunkExtent; ++z) {
                            const size_t sampleIndex =
                                (static_cast<size_t>(x) * paddedYZArea) +
                                (static_cast<size_t>(y) * static_cast<size_t>(kPaddedChunkExtent)) +
                                static_cast<size_t>(z);
                            const WorldSection::Sample& sample = sectionSamples[sampleIndex];
                            const BlockCoord coordToCopy{
                                paddedOrigin.v.x + x,
                                paddedOrigin.v.y + y,
                                paddedOrigin.v.z + z
                            };

                            BlockMaterial block = sample.block;
                            if (!sample.known) {
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

void MeshManager::onChunkMeshed(const ChunkCoord& coord, std::vector<Meshlet>&& meshlets) {
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
        scheduleChunkMeshing(coord, jobsystem::Priority::High, true);
    }
}

std::vector<Meshlet> MeshManager::copyMeshlets() const {
    std::shared_lock<std::shared_mutex> lock(meshMutex_);

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

std::vector<Meshlet> MeshManager::copyMeshletsAround(const ColumnCoord& centerColumn, int32_t columnRadius) const {
    std::shared_lock<std::shared_mutex> lock(meshMutex_);

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

uint64_t MeshManager::meshRevision() const noexcept {
    return meshRevision_.load(std::memory_order_acquire);
}

bool MeshManager::hasPendingJobs() const {
    std::shared_lock<std::shared_mutex> lock(meshMutex_);
    return !pendingMeshJobs_.empty() || !deferredRemeshChunks_.empty();
}

bool MeshManager::isWithinActiveWindowLocked(const ColumnCoord& coord, int32_t extraRadius) const {
    if (!hasLastScheduledCenter_) {
        return true;
    }

    const int32_t radius = std::max(0, config_.columnMeshRadius + extraRadius);
    const int32_t dx = std::abs(coord.v.x - lastScheduledCenter_.v.x);
    const int32_t dy = std::abs(coord.v.y - lastScheduledCenter_.v.y);
    return dx <= radius && dy <= radius;
}

jobsystem::Priority MeshManager::priorityFromDistanceSq(int32_t distanceSq) {
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
