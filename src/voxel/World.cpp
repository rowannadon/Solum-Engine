#include "solum_engine/voxel/World.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
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

uint8_t unlitPackedLight() {
    static constexpr uint8_t kUnlit = Chunk::packLight(0u, 0u);
    return kUnlit;
}

constexpr int kNeighborhoodColumns = 3;
constexpr int kRelightPaddingColumns = 1;
constexpr int kNeighborhoodSizeBlocks = cfg::CHUNK_SIZE * kNeighborhoodColumns;
constexpr int kNeighborhoodAreaBlocks = kNeighborhoodSizeBlocks * kNeighborhoodSizeBlocks;
constexpr int kNeighborhoodVolumeBlocks = kNeighborhoodAreaBlocks * cfg::COLUMN_HEIGHT_BLOCKS;
constexpr int kCenterOffsetBlocks = cfg::CHUNK_SIZE * kRelightPaddingColumns;
constexpr int kCenterColumnVoxelCount = cfg::CHUNK_SIZE * cfg::CHUNK_SIZE * cfg::COLUMN_HEIGHT_BLOCKS;

size_t neighborhoodVoxelIndex(int x, int y, int z) {
    return (static_cast<size_t>(z) * static_cast<size_t>(kNeighborhoodAreaBlocks)) +
           (static_cast<size_t>(y) * static_cast<size_t>(kNeighborhoodSizeBlocks)) +
           static_cast<size_t>(x);
}

size_t centerColumnVoxelIndex(int x, int y, int z) {
    return (static_cast<size_t>(z) * static_cast<size_t>(cfg::CHUNK_SIZE) * static_cast<size_t>(cfg::CHUNK_SIZE)) +
           (static_cast<size_t>(y) * static_cast<size_t>(cfg::CHUNK_SIZE)) +
           static_cast<size_t>(x);
}

int32_t distanceSqToCenter(const ColumnCoord& coord, const ColumnCoord& center) {
    const int64_t dx = static_cast<int64_t>(coord.v.x) - static_cast<int64_t>(center.v.x);
    const int64_t dy = static_cast<int64_t>(coord.v.y) - static_cast<int64_t>(center.v.y);
    const int64_t distanceSq = (dx * dx) + (dy * dy);
    if (distanceSq > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        return std::numeric_limits<int32_t>::max();
    }
    return static_cast<int32_t>(distanceSq);
}
}  // namespace

struct World::ColumnGenerationResult {
    ColumnCoord coord;
    Column column;
    bool generated = false;
};

struct World::ColumnRelightResult {
    ColumnCoord coord;
    std::vector<uint8_t> packedLights;
    bool relit = false;
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

uint8_t WorldSection::getPackedLight(const BlockCoord& coord) const {
    uint8_t packedLight = unlitPackedLight();
    world_.tryGetPackedLight(coord, packedLight, mipLevel_);
    return packedLight;
}

bool WorldSection::tryGetBlock(const BlockCoord& coord, BlockMaterial& outBlock) const {
    return world_.tryGetBlock(coord, outBlock, mipLevel_);
}

bool WorldSection::tryGetPackedLight(const BlockCoord& coord, uint8_t& outPackedLight) const {
    return world_.tryGetPackedLight(coord, outPackedLight, mipLevel_);
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
      jobs_(config_.jobConfig) {
    const std::size_t configuredMaxInFlight = config_.maxInFlightColumnJobs;
    const std::size_t workerCount = std::max<std::size_t>(std::size_t{1}, jobs_.worker_count());
    const std::size_t autoMaxInFlight = workerCount * 2;
    maxInFlightColumnJobs_ = std::max<std::size_t>(
        std::size_t{1},
        (configuredMaxInFlight > 0) ? configuredMaxInFlight : autoMaxInFlight
    );
    maxInFlightRelightJobs_ = std::clamp<std::size_t>(workerCount / 3u, std::size_t{1}, std::size_t{2});
}

World::~World() {
    shuttingDown_.store(true, std::memory_order_release);
    jobs_.wait_for_idle();
    jobs_.stop();
}

BlockMaterial World::getBlock(const BlockCoord& coord) const {
    return getBlock(coord, 0);
}

uint8_t World::getPackedLight(const BlockCoord& coord) const {
    uint8_t packedLight = unlitPackedLight();
    tryGetPackedLight(coord, packedLight);
    return packedLight;
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

bool World::tryGetPackedLight(const BlockCoord& coord, uint8_t& outPackedLight) const {
    return tryGetPackedLight(coord, outPackedLight, 0);
}

bool World::tryGetPackedLight(const BlockCoord& coord, uint8_t& outPackedLight, uint8_t mipLevel) const {
    std::shared_lock<std::shared_mutex> lock(worldMutex_);
    return tryGetPackedLightLocked(coord, outPackedLight, mipLevel);
}

void World::markColumnLightingDirty(const ColumnCoord& coord) {
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return;
    }

    {
        std::unique_lock<std::shared_mutex> lock(worldMutex_);
        invalidateRelightNeighborhoodLocked(coord);
        enqueueRelightNeighborhoodLocked(coord);
    }
    pumpColumnRelightQueue();
}

bool World::isColumnGenerated(const ColumnCoord& coord) const {
    std::shared_lock<std::shared_mutex> lock(worldMutex_);
    return isColumnGeneratedLocked(coord);
}

bool World::tryGetColumnEmptyChunkMask(const ColumnCoord& coord, uint32_t& outMask) const {
    std::shared_lock<std::shared_mutex> lock(worldMutex_);
    if (!isColumnGeneratedLocked(coord)) {
        outMask = 0u;
        return false;
    }

    const RegionCoord regionCoord = column_to_region(coord);
    const auto regionIt = regions_.find(regionCoord);
    if (regionIt == regions_.end() || regionIt->second == nullptr) {
        outMask = 0u;
        return false;
    }

    const glm::ivec2 localColumn = column_local_in_region(coord);
    const Column& column = regionIt->second->getColumn(
        static_cast<uint8_t>(localColumn.x),
        static_cast<uint8_t>(localColumn.y)
    );
    outMask = column.getEmptyChunkMask();
    return true;
}

uint64_t World::generationRevision() const {
    return generationRevision_.load(std::memory_order_acquire);
}

uint64_t World::copyGeneratedColumnsSince(uint64_t afterRevision,
                                          std::vector<ColumnCoord>& outColumns,
                                          std::size_t maxCount) const {
    std::shared_lock<std::shared_mutex> lock(worldMutex_);
    const uint64_t currentRevision = static_cast<uint64_t>(generatedColumnHistory_.size());
    const uint64_t clampedRevision = std::min(afterRevision, currentRevision);
    const size_t startIndex = static_cast<size_t>(clampedRevision);
    const size_t available = generatedColumnHistory_.size() - startIndex;
    const size_t count = std::min(maxCount, available);

    outColumns.clear();
    outColumns.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        outColumns.push_back(generatedColumnHistory_[startIndex + i]);
    }

    return clampedRevision + static_cast<uint64_t>(count);
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

bool World::tryGetPackedLightLocked(const BlockCoord& coord,
                                    uint8_t& outPackedLight,
                                    uint8_t mipLevel) const {
    const uint8_t clampedMip = std::min<uint8_t>(mipLevel, Chunk::MAX_MIP_LEVEL);
    const int32_t chunkSizeAtMip = static_cast<int32_t>(Chunk::mipSize(clampedMip));
    const int32_t worldHeightAtMip = cfg::COLUMN_HEIGHT_BLOCKS >> clampedMip;

    if (coord.v.z < 0 || coord.v.z >= worldHeightAtMip) {
        outPackedLight = unlitPackedLight();
        return false;
    }

    const ChunkCoord chunkCoord{
        floor_div(coord.v.x, chunkSizeAtMip),
        floor_div(coord.v.y, chunkSizeAtMip),
        floor_div(coord.v.z, chunkSizeAtMip)
    };
    if (chunkCoord.v.z < 0 || chunkCoord.v.z >= cfg::COLUMN_HEIGHT) {
        outPackedLight = unlitPackedLight();
        return false;
    }

    const ColumnCoord columnCoord = chunk_to_column(chunkCoord);
    const RegionCoord regionCoord = column_to_region(columnCoord);

    if (generatedColumns_.find(columnCoord) == generatedColumns_.end()) {
        outPackedLight = unlitPackedLight();
        return false;
    }

    const auto regionIt = regions_.find(regionCoord);
    if (regionIt == regions_.end() || regionIt->second == nullptr) {
        outPackedLight = unlitPackedLight();
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

    outPackedLight = column.getChunk(static_cast<uint8_t>(chunkCoord.v.z)).getPackedLight(
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

    std::vector<ScheduledColumnJob> jobsToSchedule;
    std::vector<ColumnCoord> relightJobsToSchedule;
    {
        // Fast path for unchanged center without taking the write lock. Worker mesh jobs
        // hold shared locks frequently; avoiding a per-frame writer lock reduces stalls.
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

        lastScheduledCenter_ = centerColumn;
        hasLastScheduledCenter_ = true;
        ++queueCenterVersion_;

        // Discard unscheduled work from an old center so movement can immediately
        // shift generation focus to the newest player position.
        queuedColumnJobs_.clear();
        queuedColumnHeap_ = decltype(queuedColumnHeap_){};

        refillQueuedColumnsLocked();
        pruneQueuedColumnsOutsideActiveWindowLocked();
        pruneQueuedRelightColumnsOutsideActiveWindowLocked();
        seedQueuedRelightColumnsLocked();
        collectColumnJobsToScheduleLocked(jobsToSchedule);
        collectRelightJobsToScheduleLocked(relightJobsToSchedule);
    }
    dispatchScheduledRelightJobs(std::move(relightJobsToSchedule));
    dispatchScheduledColumnJobs(std::move(jobsToSchedule));
}

void World::enqueueColumnGenerationLocked(const ColumnCoord& coord) {
    if (!isWithinActiveWindowLocked(coord, 0)) {
        return;
    }
    if (isColumnGeneratedLocked(coord)) {
        return;
    }
    if (pendingColumnJobs_.find(coord) != pendingColumnJobs_.end()) {
        return;
    }
    if (queuedColumnJobs_.find(coord) != queuedColumnJobs_.end()) {
        return;
    }
    queuedColumnJobs_.insert(coord);
    const int32_t distanceSq = hasLastScheduledCenter_
        ? distanceSqToCenter(coord, lastScheduledCenter_)
        : 0;
    queuedColumnHeap_.push(QueuedColumnEntry{
        coord,
        distanceSq,
        queueCenterVersion_,
        queueSequence_++
    });
}

void World::enqueueColumnRelightLocked(const ColumnCoord& coord) {
    if (!isWithinActiveWindowLocked(coord, 1)) {
        return;
    }
    if (!isColumnGeneratedLocked(coord)) {
        return;
    }
    if (pendingRelightJobs_.find(coord) != pendingRelightJobs_.end()) {
        return;
    }
    if (queuedRelightJobs_.find(coord) != queuedRelightJobs_.end()) {
        return;
    }
    if (resolvedLightingColumns_.find(coord) != resolvedLightingColumns_.end()) {
        return;
    }

    queuedRelightJobs_.insert(coord);
    queuedRelightOrder_.push_back(coord);
}

void World::invalidateRelightNeighborhoodLocked(const ColumnCoord& coord) {
    for (int32_t oy = -1; oy <= 1; ++oy) {
        for (int32_t ox = -1; ox <= 1; ++ox) {
            const ColumnCoord target{coord.v.x + ox, coord.v.y + oy};
            resolvedLightingColumns_.erase(target);
        }
    }
}

void World::enqueueRelightNeighborhoodLocked(const ColumnCoord& coord) {
    for (int32_t oy = -1; oy <= 1; ++oy) {
        for (int32_t ox = -1; ox <= 1; ++ox) {
            enqueueColumnRelightLocked(ColumnCoord{coord.v.x + ox, coord.v.y + oy});
        }
    }
}

bool World::hasRelightNeighborhoodLocked(const ColumnCoord& coord) const {
    for (int32_t oy = -1; oy <= 1; ++oy) {
        for (int32_t ox = -1; ox <= 1; ++ox) {
            const ColumnCoord neighbor{coord.v.x + ox, coord.v.y + oy};
            if (generatedColumns_.find(neighbor) == generatedColumns_.end()) {
                return false;
            }
        }
    }
    return true;
}

std::size_t World::desiredQueuedColumnCountLocked() const {
    const int32_t radius = std::max(0, config_.columnLoadRadius);
    const uint64_t diameter = (static_cast<uint64_t>(radius) * 2u) + 1u;
    const uint64_t maxWindowColumns64 = diameter * diameter;
    const std::size_t maxWindowColumns = static_cast<std::size_t>(
        std::min<uint64_t>(maxWindowColumns64, std::numeric_limits<std::size_t>::max())
    );

    // Keep a short look-ahead backlog to reduce idle worker time, but cap queue
    // growth so center shifts can be reflected immediately.
    constexpr std::size_t kQueueLookAheadMultiplier = 4;
    const std::size_t lookAheadOutstanding =
        (maxInFlightColumnJobs_ > (std::numeric_limits<std::size_t>::max() / kQueueLookAheadMultiplier))
            ? std::numeric_limits<std::size_t>::max()
            : (maxInFlightColumnJobs_ * kQueueLookAheadMultiplier);
    const std::size_t desiredOutstanding = std::max(maxInFlightColumnJobs_, lookAheadOutstanding);

    const std::size_t pendingCount = pendingColumnJobs_.size();
    const std::size_t desiredQueued =
        (desiredOutstanding > pendingCount) ? (desiredOutstanding - pendingCount) : 0;
    return std::min(desiredQueued, maxWindowColumns);
}

void World::refillQueuedColumnsLocked() {
    if (!hasLastScheduledCenter_) {
        return;
    }

    const std::size_t desiredQueued = desiredQueuedColumnCountLocked();
    if (queuedColumnJobs_.size() >= desiredQueued) {
        return;
    }

    std::size_t remainingToQueue = desiredQueued - queuedColumnJobs_.size();
    const int32_t radius = std::max(0, config_.columnLoadRadius);
    const ColumnCoord center = lastScheduledCenter_;

    auto tryEnqueue = [&](int32_t x, int32_t y) {
        if (remainingToQueue == 0) {
            return;
        }
        const std::size_t queuedBefore = queuedColumnJobs_.size();
        enqueueColumnGenerationLocked(ColumnCoord{x, y});
        if (queuedColumnJobs_.size() > queuedBefore) {
            --remainingToQueue;
        }
    };

    if (remainingToQueue > 0) {
        tryEnqueue(center.v.x, center.v.y);
    }

    for (int32_t ring = 1; ring <= radius && remainingToQueue > 0; ++ring) {
        const int32_t minX = center.v.x - ring;
        const int32_t maxX = center.v.x + ring;
        const int32_t minY = center.v.y - ring;
        const int32_t maxY = center.v.y + ring;

        for (int32_t x = minX; x <= maxX && remainingToQueue > 0; ++x) {
            tryEnqueue(x, minY);
            tryEnqueue(x, maxY);
        }
        for (int32_t y = minY + 1; y <= maxY - 1 && remainingToQueue > 0; ++y) {
            tryEnqueue(minX, y);
            tryEnqueue(maxX, y);
        }
    }
}

void World::pruneQueuedColumnsOutsideActiveWindowLocked() {
    // Bounded heap cleanup to cap per-pump overhead when radius grows.
    constexpr size_t kPruneBudget = 256;
    size_t processed = 0;
    while (processed < kPruneBudget && !queuedColumnHeap_.empty()) {
        const QueuedColumnEntry top = queuedColumnHeap_.top();

        auto queuedIt = queuedColumnJobs_.find(top.coord);
        if (queuedIt == queuedColumnJobs_.end()) {
            queuedColumnHeap_.pop();
            ++processed;
            continue;
        }

        if (!isWithinActiveWindowLocked(top.coord, 0) ||
            isColumnGeneratedLocked(top.coord) ||
            pendingColumnJobs_.find(top.coord) != pendingColumnJobs_.end()) {
            queuedColumnJobs_.erase(queuedIt);
            queuedColumnHeap_.pop();
            ++processed;
            continue;
        }

        if (top.centerVersion != queueCenterVersion_) {
            queuedColumnHeap_.pop();
            queuedColumnHeap_.push(QueuedColumnEntry{
                top.coord,
                hasLastScheduledCenter_ ? distanceSqToCenter(top.coord, lastScheduledCenter_) : 0,
                queueCenterVersion_,
                queueSequence_++
            });
            ++processed;
            continue;
        }

        break;
    }
}

void World::pruneQueuedRelightColumnsOutsideActiveWindowLocked() {
    if (queuedRelightOrder_.empty()) {
        return;
    }

    // Keep relight prune work bounded so large queues cannot stall the main thread.
    constexpr std::size_t kPruneBudget = 256;
    std::size_t processed = 0;
    while (processed < kPruneBudget && !queuedRelightOrder_.empty()) {
        const ColumnCoord coord = queuedRelightOrder_.front();
        queuedRelightOrder_.pop_front();
        ++processed;

        const auto queuedIt = queuedRelightJobs_.find(coord);
        if (queuedIt == queuedRelightJobs_.end()) {
            continue;
        }

        if (!isWithinActiveWindowLocked(coord, 1) || !isColumnGeneratedLocked(coord)) {
            queuedRelightJobs_.erase(queuedIt);
            continue;
        }
        if (resolvedLightingColumns_.find(coord) != resolvedLightingColumns_.end()) {
            queuedRelightJobs_.erase(queuedIt);
            continue;
        }

        queuedRelightOrder_.push_back(coord);
    }
}

void World::seedQueuedRelightColumnsLocked() {
    if (!hasLastScheduledCenter_ || !queuedRelightOrder_.empty()) {
        return;
    }

    const int32_t radius = std::max(0, config_.columnLoadRadius);
    const int32_t minX = lastScheduledCenter_.v.x - radius;
    const int32_t maxX = lastScheduledCenter_.v.x + radius;
    const int32_t minY = lastScheduledCenter_.v.y - radius;
    const int32_t maxY = lastScheduledCenter_.v.y + radius;

    for (int32_t y = minY; y <= maxY; ++y) {
        for (int32_t x = minX; x <= maxX; ++x) {
            enqueueColumnRelightLocked(ColumnCoord{x, y});
        }
    }
}

void World::collectRelightJobsToScheduleLocked(std::vector<ColumnCoord>& outJobs) {
    outJobs.clear();
    if (maxInFlightRelightJobs_ == 0) {
        return;
    }

    const std::size_t relightInFlightBudget = maxInFlightRelightJobs_;
    if (pendingRelightJobs_.size() >= relightInFlightBudget) {
        return;
    }

    std::size_t remainingSlots = relightInFlightBudget - pendingRelightJobs_.size();
    constexpr std::size_t kScheduleAttemptBudget = 128;
    std::size_t attemptsRemaining = std::min<std::size_t>(queuedRelightOrder_.size(), kScheduleAttemptBudget);

    while (remainingSlots > 0 && attemptsRemaining > 0 && !queuedRelightOrder_.empty()) {
        const ColumnCoord coord = queuedRelightOrder_.front();
        queuedRelightOrder_.pop_front();
        queuedRelightJobs_.erase(coord);
        --attemptsRemaining;

        if (!isWithinActiveWindowLocked(coord, 1) || !isColumnGeneratedLocked(coord)) {
            continue;
        }
        if (resolvedLightingColumns_.find(coord) != resolvedLightingColumns_.end()) {
            continue;
        }

        if (!hasRelightNeighborhoodLocked(coord)) {
            queuedRelightJobs_.insert(coord);
            queuedRelightOrder_.push_back(coord);
            continue;
        }

        if (pendingRelightJobs_.insert(coord).second) {
            outJobs.push_back(coord);
            --remainingSlots;
        }
    }
}

void World::collectColumnJobsToScheduleLocked(std::vector<ScheduledColumnJob>& outJobs) {
    while (pendingColumnJobs_.size() < maxInFlightColumnJobs_ && !queuedColumnHeap_.empty()) {
        const QueuedColumnEntry top = queuedColumnHeap_.top();
        queuedColumnHeap_.pop();

        auto queuedIt = queuedColumnJobs_.find(top.coord);
        if (queuedIt == queuedColumnJobs_.end()) {
            continue;
        }

        if (!isWithinActiveWindowLocked(top.coord, 0) ||
            isColumnGeneratedLocked(top.coord) ||
            pendingColumnJobs_.find(top.coord) != pendingColumnJobs_.end()) {
            queuedColumnJobs_.erase(queuedIt);
            continue;
        }

        if (top.centerVersion != queueCenterVersion_) {
            queuedColumnHeap_.push(QueuedColumnEntry{
                top.coord,
                hasLastScheduledCenter_ ? distanceSqToCenter(top.coord, lastScheduledCenter_) : 0,
                queueCenterVersion_,
                queueSequence_++
            });
            continue;
        }

        queuedColumnJobs_.erase(queuedIt);
        pendingColumnJobs_.insert(top.coord);
        outJobs.push_back(ScheduledColumnJob{
            top.coord,
            priorityFromDistanceSq(top.distanceSq)
        });
    }
}

void World::dispatchScheduledColumnJobs(std::vector<ScheduledColumnJob>&& jobsToSchedule) {
    for (const ScheduledColumnJob& scheduled : jobsToSchedule) {
        const ColumnCoord coord = scheduled.coord;
        try {
            jobs_.schedule(
                scheduled.priority,
                [this, coord]() -> ColumnGenerationResult {
                    {
                        std::shared_lock<std::shared_mutex> lock(worldMutex_);
                        if (!isWithinActiveWindowLocked(coord, 0)) {
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
                        {
                            std::unique_lock<std::shared_mutex> lock(worldMutex_);
                            pendingColumnJobs_.erase(coord);
                        }
                        pumpColumnGenerationQueue();
                        return;
                    }

                    ColumnGenerationResult generated = std::move(result).value();
                    if (!generated.generated) {
                        {
                            std::unique_lock<std::shared_mutex> lock(worldMutex_);
                            pendingColumnJobs_.erase(coord);
                        }
                        pumpColumnGenerationQueue();
                        return;
                    }
                    onColumnGenerated(generated.coord, std::move(generated.column));
                    pumpColumnGenerationQueue();
                }
            );
        } catch (const std::exception&) {
            {
                std::unique_lock<std::shared_mutex> lock(worldMutex_);
                pendingColumnJobs_.erase(coord);
                if (!shuttingDown_.load(std::memory_order_acquire) &&
                    isWithinActiveWindowLocked(coord, 0) &&
                    !isColumnGeneratedLocked(coord)) {
                    queuedColumnJobs_.insert(coord);
                    const int32_t distanceSq = hasLastScheduledCenter_
                        ? distanceSqToCenter(coord, lastScheduledCenter_)
                        : 0;
                    queuedColumnHeap_.push(QueuedColumnEntry{
                        coord,
                        distanceSq,
                        queueCenterVersion_,
                        queueSequence_++
                    });
                }
            }
        }
    }
}

void World::dispatchScheduledRelightJobs(std::vector<ColumnCoord>&& jobsToSchedule) {
    for (const ColumnCoord& coord : jobsToSchedule) {
        try {
            jobs_.schedule(
                jobsystem::Priority::Critical,
                [this, coord]() -> ColumnRelightResult {
                    ColumnRelightResult result{};
                    result.coord = coord;
                    result.packedLights.assign(static_cast<size_t>(kCenterColumnVoxelCount), unlitPackedLight());

                    std::vector<uint8_t> neighborhoodSolid(static_cast<size_t>(kNeighborhoodVolumeBlocks), 0u);
                    std::vector<uint8_t> centerBlockLight(static_cast<size_t>(kCenterColumnVoxelCount), 0u);

                    {
                        std::shared_lock<std::shared_mutex> lock(worldMutex_);
                        if (shuttingDown_.load(std::memory_order_acquire) ||
                            !isWithinActiveWindowLocked(coord, 1) ||
                            !hasRelightNeighborhoodLocked(coord)) {
                            return result;
                        }

                        auto getColumnLocked = [this](const ColumnCoord& columnCoord) -> const Column* {
                            const RegionCoord regionCoord = column_to_region(columnCoord);
                            const auto regionIt = regions_.find(regionCoord);
                            if (regionIt == regions_.end() || regionIt->second == nullptr) {
                                return nullptr;
                            }

                            const glm::ivec2 localColumn = column_local_in_region(columnCoord);
                            return &regionIt->second->getColumn(
                                static_cast<uint8_t>(localColumn.x),
                                static_cast<uint8_t>(localColumn.y)
                            );
                        };

                        std::array<const Column*, kNeighborhoodColumns * kNeighborhoodColumns> columns{};
                        for (int ny = 0; ny < kNeighborhoodColumns; ++ny) {
                            for (int nx = 0; nx < kNeighborhoodColumns; ++nx) {
                                const ColumnCoord neighborCoord{
                                    coord.v.x + nx - kRelightPaddingColumns,
                                    coord.v.y + ny - kRelightPaddingColumns
                                };
                                columns[static_cast<size_t>(ny * kNeighborhoodColumns + nx)] = getColumnLocked(neighborCoord);
                                if (columns[static_cast<size_t>(ny * kNeighborhoodColumns + nx)] == nullptr) {
                                    return result;
                                }
                            }
                        }

                        for (int ny = 0; ny < kNeighborhoodColumns; ++ny) {
                            for (int nx = 0; nx < kNeighborhoodColumns; ++nx) {
                                const Column& column = *columns[static_cast<size_t>(ny * kNeighborhoodColumns + nx)];
                                const int xOffset = nx * cfg::CHUNK_SIZE;
                                const int yOffset = ny * cfg::CHUNK_SIZE;
                                const bool isCenterColumn = (nx == kRelightPaddingColumns) && (ny == kRelightPaddingColumns);

                                for (int z = 0; z < cfg::COLUMN_HEIGHT_BLOCKS; ++z) {
                                    for (int y = 0; y < cfg::CHUNK_SIZE; ++y) {
                                        for (int x = 0; x < cfg::CHUNK_SIZE; ++x) {
                                            const int sampleX = xOffset + x;
                                            const int sampleY = yOffset + y;
                                            const size_t sampleIndex = neighborhoodVoxelIndex(sampleX, sampleY, z);

                                            const BlockMaterial block = column.getBlock(
                                                static_cast<uint8_t>(x),
                                                static_cast<uint8_t>(y),
                                                static_cast<uint16_t>(z)
                                            );
                                            neighborhoodSolid[sampleIndex] = (block.unpack().id != 0u) ? 1u : 0u;

                                            if (!isCenterColumn) {
                                                continue;
                                            }

                                            const size_t centerIndex = centerColumnVoxelIndex(x, y, z);
                                            const uint8_t packedLight = column.getPackedLight(
                                                static_cast<uint8_t>(x),
                                                static_cast<uint8_t>(y),
                                                static_cast<uint16_t>(z)
                                            );
                                            centerBlockLight[centerIndex] = Chunk::unpackBlockLight(packedLight);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    std::vector<uint8_t> skyFromAbove(static_cast<size_t>(kNeighborhoodAreaBlocks), 15u);
                    std::vector<uint8_t> levelSky(static_cast<size_t>(kNeighborhoodAreaBlocks), 0u);
                    std::vector<uint8_t> levelSolid(static_cast<size_t>(kNeighborhoodAreaBlocks), 0u);
                    std::vector<int> queue;
                    queue.reserve(static_cast<size_t>(kNeighborhoodAreaBlocks));

                    static constexpr std::array<std::array<int, 2>, 4> kCardinalOffsets{{
                        {{1, 0}},
                        {{-1, 0}},
                        {{0, 1}},
                        {{0, -1}},
                    }};

                    for (int z = cfg::COLUMN_HEIGHT_BLOCKS - 1; z >= 0; --z) {
                        queue.clear();
                        size_t queueHead = 0u;

                        for (int y = 0; y < kNeighborhoodSizeBlocks; ++y) {
                            for (int x = 0; x < kNeighborhoodSizeBlocks; ++x) {
                                const size_t index2D = static_cast<size_t>(y * kNeighborhoodSizeBlocks + x);
                                const bool solid = neighborhoodSolid[neighborhoodVoxelIndex(x, y, z)] != 0u;
                                levelSolid[index2D] = solid ? 1u : 0u;
                                if (solid) {
                                    levelSky[index2D] = 0u;
                                    continue;
                                }

                                const uint8_t seed = skyFromAbove[index2D];
                                levelSky[index2D] = seed;
                                if (seed > 1u) {
                                    queue.push_back(static_cast<int>(index2D));
                                }
                            }
                        }

                        while (queueHead < queue.size()) {
                            const int current = queue[queueHead++];
                            const uint8_t light = levelSky[static_cast<size_t>(current)];
                            if (light <= 1u) {
                                continue;
                            }

                            const uint8_t propagated = static_cast<uint8_t>(light - 1u);
                            const int cx = current % kNeighborhoodSizeBlocks;
                            const int cy = current / kNeighborhoodSizeBlocks;
                            for (const auto& offset : kCardinalOffsets) {
                                const int nx = cx + offset[0];
                                const int ny = cy + offset[1];
                                if (nx < 0 || ny < 0 || nx >= kNeighborhoodSizeBlocks || ny >= kNeighborhoodSizeBlocks) {
                                    continue;
                                }

                                const size_t neighborIndex = static_cast<size_t>(ny * kNeighborhoodSizeBlocks + nx);
                                if (levelSolid[neighborIndex] != 0u) {
                                    continue;
                                }
                                if (propagated <= levelSky[neighborIndex]) {
                                    continue;
                                }

                                levelSky[neighborIndex] = propagated;
                                queue.push_back(static_cast<int>(neighborIndex));
                            }
                        }

                        for (int y = 0; y < kNeighborhoodSizeBlocks; ++y) {
                            for (int x = 0; x < kNeighborhoodSizeBlocks; ++x) {
                                const size_t index2D = static_cast<size_t>(y * kNeighborhoodSizeBlocks + x);
                                const uint8_t sky = (levelSolid[index2D] != 0u) ? 0u : levelSky[index2D];
                                skyFromAbove[index2D] = sky;

                                if (x < kCenterOffsetBlocks || x >= (kCenterOffsetBlocks + cfg::CHUNK_SIZE) ||
                                    y < kCenterOffsetBlocks || y >= (kCenterOffsetBlocks + cfg::CHUNK_SIZE)) {
                                    continue;
                                }

                                const int localX = x - kCenterOffsetBlocks;
                                const int localY = y - kCenterOffsetBlocks;
                                const size_t centerIndex = centerColumnVoxelIndex(localX, localY, z);
                                result.packedLights[centerIndex] = Chunk::packLight(sky, centerBlockLight[centerIndex]);
                            }
                        }
                    }

                    result.relit = true;
                    return result;
                },
                [this, coord](jobsystem::JobResult<ColumnRelightResult>&& result) {
                    bool repump = false;
                    {
                        std::unique_lock<std::shared_mutex> lock(worldMutex_);
                        pendingRelightJobs_.erase(coord);

                        if (!result.success() || shuttingDown_.load(std::memory_order_acquire)) {
                            repump = true;
                        } else {
                            ColumnRelightResult relight = std::move(result).value();
                            if (!relight.relit) {
                                if (isWithinActiveWindowLocked(coord, 1) && isColumnGeneratedLocked(coord)) {
                                    enqueueColumnRelightLocked(coord);
                                }
                                repump = true;
                            } else if (isColumnGeneratedLocked(coord)) {
                                Region* region = getOrCreateRegionLocked(column_to_region(coord));
                                if (region != nullptr) {
                                    const glm::ivec2 localColumn = column_local_in_region(coord);
                                    Column& centerColumn = region->getColumn(
                                        static_cast<uint8_t>(localColumn.x),
                                        static_cast<uint8_t>(localColumn.y)
                                    );

                                    bool changed = false;
                                    for (int chunkZ = 0; chunkZ < cfg::COLUMN_HEIGHT; ++chunkZ) {
                                        Chunk& chunk = centerColumn.getChunk(static_cast<uint8_t>(chunkZ));
                                        const uint8_t* chunkLights =
                                            relight.packedLights.data() + (static_cast<size_t>(chunkZ) * Chunk::VOLUME);
                                        if (chunk.setPackedLightVolume(chunkLights, Chunk::VOLUME)) {
                                            changed = true;
                                        }
                                    }

                                    resolvedLightingColumns_.insert(coord);
                                    if (changed) {
                                        generatedColumnHistory_.push_back(coord);
                                        generationRevision_.fetch_add(1, std::memory_order_release);
                                    }
                                }
                                repump = true;
                            } else {
                                repump = true;
                            }
                        }
                    }

                    if (repump) {
                        pumpColumnRelightQueue();
                    }
                }
            );
        } catch (const std::exception&) {
            std::unique_lock<std::shared_mutex> lock(worldMutex_);
            pendingRelightJobs_.erase(coord);
            if (!shuttingDown_.load(std::memory_order_acquire) &&
                isWithinActiveWindowLocked(coord, 1) &&
                isColumnGeneratedLocked(coord)) {
                enqueueColumnRelightLocked(coord);
            }
        }
    }
}

void World::pumpColumnGenerationQueue() {
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return;
    }

    std::vector<ScheduledColumnJob> jobsToSchedule;
    std::vector<ColumnCoord> relightJobsToSchedule;
    {
        std::unique_lock<std::shared_mutex> lock(worldMutex_);
        pruneQueuedColumnsOutsideActiveWindowLocked();
        pruneQueuedRelightColumnsOutsideActiveWindowLocked();
        refillQueuedColumnsLocked();
        seedQueuedRelightColumnsLocked();
        collectColumnJobsToScheduleLocked(jobsToSchedule);
        collectRelightJobsToScheduleLocked(relightJobsToSchedule);
    }
    dispatchScheduledRelightJobs(std::move(relightJobsToSchedule));
    dispatchScheduledColumnJobs(std::move(jobsToSchedule));
}

void World::pumpColumnRelightQueue() {
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return;
    }

    std::vector<ColumnCoord> jobsToSchedule;
    {
        std::unique_lock<std::shared_mutex> lock(worldMutex_);
        pruneQueuedRelightColumnsOutsideActiveWindowLocked();
        seedQueuedRelightColumnsLocked();
        collectRelightJobsToScheduleLocked(jobsToSchedule);
    }
    dispatchScheduledRelightJobs(std::move(jobsToSchedule));
}

void World::onColumnGenerated(const ColumnCoord& coord, Column&& column) {
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return;
    }

    std::unique_lock<std::shared_mutex> lock(worldMutex_);
    pendingColumnJobs_.erase(coord);
    if (!isWithinActiveWindowLocked(coord, 0)) {
        return;
    }

    // Keep occupancy metadata coherent even if a generator path bypasses Column::setBlock.
    column.rebuildEmptyChunkMask();

    Region* region = getOrCreateRegionLocked(column_to_region(coord));
    if (region == nullptr) {
        return;
    }

    const glm::ivec2 localColumn = column_local_in_region(coord);
    region->getColumn(
        static_cast<uint8_t>(localColumn.x),
        static_cast<uint8_t>(localColumn.y)
    ) = std::move(column);

    invalidateRelightNeighborhoodLocked(coord);

    const auto insertedResult = generatedColumns_.insert(coord);
    if (insertedResult.second) {
        generatedColumnHistory_.push_back(coord);
        generationRevision_.fetch_add(1, std::memory_order_release);
    }

    enqueueRelightNeighborhoodLocked(coord);
}

bool World::hasPendingJobs() const {
    std::shared_lock<std::shared_mutex> lock(worldMutex_);
    return !pendingColumnJobs_.empty() ||
           !queuedColumnJobs_.empty() ||
           !pendingRelightJobs_.empty() ||
           !queuedRelightOrder_.empty();
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
        return jobsystem::Priority::High;
    }
    if (distanceSq <= 2) {
        return jobsystem::Priority::Normal;
    }
    if (distanceSq <= 8) {
        return jobsystem::Priority::Low;
    }
    return jobsystem::Priority::Low;
}
