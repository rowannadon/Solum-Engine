#include "solum_engine/voxel/World.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <mutex>
#include <utility>
#include <vector>

#include "solum_engine/resources/Constants.h"
#include "solum_engine/voxel/Column.h"
#include "solum_engine/voxel/Region.h"
#include "solum_engine/voxel/TerrainGenerator.h"

namespace {
BlockMaterial airBlock() {
    static const BlockMaterial kAir = UnpackedBlockMaterial{}.pack();
    return kAir;
}
}  // namespace

struct World::ColumnGenerationResult {
    ColumnCoord coord;
    Column column;
    bool generated = false;
};

WorldSection::WorldSection(const World& world,
                           const BlockCoord& origin,
                           const glm::ivec3& extent,
                           uint8_t mipLevel)
    : world_(world),
      origin_(origin),
      extent_(extent),
      mipLevel_(std::min<uint8_t>(mipLevel, Chunk::MAX_MIP_LEVEL)) {}

BlockMaterial WorldSection::getBlock(const BlockCoord& coord) const {
    return world_.getBlock(coord, mipLevel_);
}

bool WorldSection::tryGetBlock(const BlockCoord& coord, BlockMaterial& outBlock) const {
    return world_.tryGetBlock(coord, outBlock, mipLevel_);
}

BlockMaterial WorldSection::getLocalBlock(int32_t x, int32_t y, int32_t z) const {
    return world_.getBlock(BlockCoord{
        origin_.v.x + x,
        origin_.v.y + y,
        origin_.v.z + z
    }, mipLevel_);
}

bool WorldSection::tryGetLocalBlock(int32_t x, int32_t y, int32_t z, BlockMaterial& outBlock) const {
    return world_.tryGetBlock(BlockCoord{
        origin_.v.x + x,
        origin_.v.y + y,
        origin_.v.z + z
    }, outBlock, mipLevel_);
}

void WorldSection::copySamples(std::vector<Sample>& outSamples) const {
    if (extent_.x <= 0 || extent_.y <= 0 || extent_.z <= 0) {
        outSamples.clear();
        return;
    }

    const size_t yzArea = static_cast<size_t>(extent_.y) * static_cast<size_t>(extent_.z);
    const size_t sampleCount = static_cast<size_t>(extent_.x) * yzArea;
    outSamples.resize(sampleCount);

    std::shared_lock<std::shared_mutex> lock(world_.worldMutex_);
    for (int32_t x = 0; x < extent_.x; ++x) {
        for (int32_t y = 0; y < extent_.y; ++y) {
            for (int32_t z = 0; z < extent_.z; ++z) {
                const size_t index =
                    (static_cast<size_t>(x) * yzArea) +
                    (static_cast<size_t>(y) * static_cast<size_t>(extent_.z)) +
                    static_cast<size_t>(z);
                Sample sample;
                const BlockCoord coord{
                    origin_.v.x + x,
                    origin_.v.y + y,
                    origin_.v.z + z
                };
                sample.known = world_.tryGetBlockLocked(coord, sample.block, mipLevel_);
                outSamples[index] = sample;
            }
        }
    }
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
    return getBlock(coord, 0);
}

BlockMaterial World::getBlock(const BlockCoord& coord, uint8_t mipLevel) const {
    BlockMaterial block = airBlock();
    tryGetBlock(coord, block, mipLevel);
    return block;
}

bool World::tryGetBlock(const BlockCoord& coord, BlockMaterial& outBlock) const {
    return tryGetBlock(coord, outBlock, 0);
}

bool World::tryGetBlock(const BlockCoord& coord, BlockMaterial& outBlock, uint8_t mipLevel) const {
    std::shared_lock<std::shared_mutex> lock(worldMutex_);
    return tryGetBlockLocked(coord, outBlock, mipLevel);
}

bool World::isColumnGenerated(const ColumnCoord& coord) const {
    std::shared_lock<std::shared_mutex> lock(worldMutex_);
    return isColumnGeneratedLocked(coord);
}

uint64_t World::generationRevision() const {
    return generationRevision_.load(std::memory_order_acquire);
}

void World::copyGeneratedColumns(std::vector<ColumnCoord>& outColumns) const {
    std::shared_lock<std::shared_mutex> lock(worldMutex_);
    outColumns.clear();
    outColumns.reserve(generatedColumns_.size());
    for (const ColumnCoord& coord : generatedColumns_) {
        outColumns.push_back(coord);
    }
    std::sort(outColumns.begin(), outColumns.end());
}

void World::copyGeneratedColumnsAround(const ColumnCoord& centerColumn,
                                       int32_t radius,
                                       std::vector<ColumnCoord>& outColumns) const {
    const int32_t clampedRadius = std::max(0, radius);
    const int32_t diameter = (clampedRadius * 2) + 1;
    const size_t maxInWindow = static_cast<size_t>(diameter) * static_cast<size_t>(diameter);

    std::shared_lock<std::shared_mutex> lock(worldMutex_);
    outColumns.clear();
    outColumns.reserve(std::min(maxInWindow, generatedColumns_.size()));

    for (const ColumnCoord& coord : generatedColumns_) {
        const int32_t dx = std::abs(coord.v.x - centerColumn.v.x);
        const int32_t dy = std::abs(coord.v.y - centerColumn.v.y);
        if (dx <= clampedRadius && dy <= clampedRadius) {
            outColumns.push_back(coord);
        }
    }

    std::sort(outColumns.begin(), outColumns.end());
}

bool World::tryGetBlockLocked(const BlockCoord& coord,
                              BlockMaterial& outBlock,
                              uint8_t mipLevel) const {
    const uint8_t clampedMip = std::min<uint8_t>(mipLevel, Chunk::MAX_MIP_LEVEL);
    const int32_t chunkSizeAtMip = static_cast<int32_t>(Chunk::mipSize(clampedMip));
    const int32_t worldHeightAtMip = cfg::COLUMN_HEIGHT_BLOCKS >> clampedMip;

    if (coord.v.z < 0 || coord.v.z >= worldHeightAtMip) {
        outBlock = airBlock();
        return false;
    }

    const ChunkCoord chunkCoord{
        floor_div(coord.v.x, chunkSizeAtMip),
        floor_div(coord.v.y, chunkSizeAtMip),
        floor_div(coord.v.z, chunkSizeAtMip)
    };
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
    const glm::ivec3 localBlock{
        floor_mod(coord.v.x, chunkSizeAtMip),
        floor_mod(coord.v.y, chunkSizeAtMip),
        floor_mod(coord.v.z, chunkSizeAtMip)
    };
    const Column& column = regionIt->second->getColumn(
        static_cast<uint8_t>(localColumn.x),
        static_cast<uint8_t>(localColumn.y)
    );

    outBlock = column.getChunk(static_cast<uint8_t>(chunkCoord.v.z)).getBlock(
        static_cast<uint8_t>(localBlock.x),
        static_cast<uint8_t>(localBlock.y),
        static_cast<uint8_t>(localBlock.z),
        clampedMip
    );
    return true;
}

WorldSection World::createSection(const BlockCoord& origin, const glm::ivec3& extent) const {
    return createSection(origin, extent, 0);
}

WorldSection World::createSection(const BlockCoord& origin,
                                  const glm::ivec3& extent,
                                  uint8_t mipLevel) const {
    return WorldSection(*this, origin, extent, mipLevel);
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

    // Fast path for unchanged center without taking the write lock. Worker mesh jobs
    // hold shared locks frequently; avoiding a per-frame writer lock reduces stalls.
    {
        std::shared_lock<std::shared_mutex> lock(worldMutex_);
        if (hasLastScheduledCenter_ && centerColumn == lastScheduledCenter_) {
            return;
        }
    }

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

    const auto insertedResult = generatedColumns_.insert(coord);
    if (insertedResult.second) {
        generationRevision_.fetch_add(1, std::memory_order_release);
    }
}

bool World::hasPendingJobs() const {
    std::shared_lock<std::shared_mutex> lock(worldMutex_);
    return !pendingColumnJobs_.empty();
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
