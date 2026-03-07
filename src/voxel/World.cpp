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
#include "solum_engine/voxel/MaterialLightProperties.h"
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

int32_t distanceSqToCenter(const ColumnCoord& coord, const ColumnCoord& center) {
    const int64_t dx = static_cast<int64_t>(coord.v.x) - static_cast<int64_t>(center.v.x);
    const int64_t dy = static_cast<int64_t>(coord.v.y) - static_cast<int64_t>(center.v.y);
    const int64_t distanceSq = (dx * dx) + (dy * dy);
    if (distanceSq > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        return std::numeric_limits<int32_t>::max();
    }
    return static_cast<int32_t>(distanceSq);
}

constexpr int kChunkExtent = cfg::CHUNK_SIZE;
constexpr int kChunkArea = kChunkExtent * kChunkExtent;
constexpr uint32_t kAllColumnChunkBits = Column::allChunksEmptyMask();
constexpr std::array<glm::ivec2, 4> kHorizontalOffsets = {
    glm::ivec2{1, 0},
    glm::ivec2{-1, 0},
    glm::ivec2{0, 1},
    glm::ivec2{0, -1},
};
constexpr std::array<glm::ivec3, 6> kCardinalOffsets = {
    glm::ivec3{1, 0, 0},
    glm::ivec3{-1, 0, 0},
    glm::ivec3{0, 1, 0},
    glm::ivec3{0, -1, 0},
    glm::ivec3{0, 0, 1},
    glm::ivec3{0, 0, -1},
};

constexpr int chunkLocalIndex(int x, int y, int z) {
    return (z * kChunkArea) + (y * kChunkExtent) + x;
}

uint8_t attenuateLight(uint8_t light, uint8_t loss) {
    if (light == 0u || loss == MaterialLightProperties::kOpaqueLightLoss || loss >= light) {
        return 0u;
    }
    return static_cast<uint8_t>(light - loss);
}
}  // namespace

struct World::ColumnGenerationResult {
    ColumnCoord coord;
    Column column;
    bool generated = false;
};

struct World::ChunkPropagationResult {
    ChunkCoord coord;
    bool propagated = false;
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
      jobs_(config_.jobConfig),
      chunkPropagationJobs_([this]() {
          jobsystem::JobSystem::Config propagationConfig = config_.jobConfig;
          const std::size_t configuredWorkers =
              (propagationConfig.worker_threads > 0) ? propagationConfig.worker_threads : 1u;
          propagationConfig.worker_threads = std::clamp<std::size_t>(configuredWorkers, 1u, 2u);
          return propagationConfig;
      }()) {
    const std::size_t configuredMaxInFlight = config_.maxInFlightColumnJobs;
    const std::size_t workerCount = std::max<std::size_t>(std::size_t{1}, jobs_.worker_count());
    const std::size_t autoMaxInFlight = workerCount * 2;
    maxInFlightColumnJobs_ = std::max<std::size_t>(
        std::size_t{1},
        (configuredMaxInFlight > 0) ? configuredMaxInFlight : autoMaxInFlight
    );
    maxInFlightChunkPropagationJobs_ = std::max<std::size_t>(
        std::size_t{1},
        chunkPropagationJobs_.worker_count()
    );
}

World::~World() {
    shuttingDown_.store(true, std::memory_order_release);
    jobs_.wait_for_idle();
    chunkPropagationJobs_.wait_for_idle();
    jobs_.stop();
    chunkPropagationJobs_.stop();
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

bool World::tryGetBlockAndPackedLight(const BlockCoord& coord,
                                      BlockMaterial& outBlock,
                                      uint8_t& outPackedLight,
                                      uint8_t mipLevel) const {
    std::shared_lock<std::shared_mutex> lock(worldMutex_);
    return tryGetBlockAndPackedLightLocked(coord, outBlock, outPackedLight, mipLevel);
}

void World::sampleBlockAndLightVolume(const BlockCoord& origin,
                                      const glm::ivec3& extent,
                                      const glm::ivec3& stride,
                                      uint8_t mipLevel,
                                      BlockMaterial* outBlocks,
                                      uint8_t* outPackedLights,
                                      uint8_t* outKnownMask) const {
    if (outBlocks == nullptr || outPackedLights == nullptr) {
        return;
    }

    const int32_t width = std::max(0, extent.x);
    const int32_t height = std::max(0, extent.y);
    const int32_t depth = std::max(0, extent.z);
    if (width == 0 || height == 0 || depth == 0) {
        return;
    }

    const int32_t strideX = std::max(1, stride.x);
    const int32_t strideY = std::max(1, stride.y);
    const int32_t strideZ = std::max(1, stride.z);
    const int32_t rowStride = height * depth;
    std::shared_lock<std::shared_mutex> lock(worldMutex_);
    for (int32_t x = 0; x < width; ++x) {
        for (int32_t y = 0; y < height; ++y) {
            for (int32_t z = 0; z < depth; ++z) {
                const int32_t index = (x * rowStride) + (y * depth) + z;
                const BlockCoord coord{
                    origin.v.x + (x * strideX),
                    origin.v.y + (y * strideY),
                    origin.v.z + (z * strideZ)
                };
                BlockMaterial block = airBlock();
                uint8_t packedLight = unlitPackedLight();
                const bool known = tryGetBlockAndPackedLightLocked(coord, block, packedLight, mipLevel);
                outBlocks[index] = block;
                outPackedLights[index] = packedLight;
                if (outKnownMask != nullptr) {
                    outKnownMask[index] = known ? 1u : 0u;
                }
            }
        }
    }
}

bool World::tryGetPackedLight(const BlockCoord& coord, uint8_t& outPackedLight) const {
    return tryGetPackedLight(coord, outPackedLight, 0);
}

bool World::tryGetPackedLight(const BlockCoord& coord, uint8_t& outPackedLight, uint8_t mipLevel) const {
    std::shared_lock<std::shared_mutex> lock(worldMutex_);
    return tryGetPackedLightLocked(coord, outPackedLight, mipLevel);
}

bool World::breakBlock(const BlockCoord& coord) {
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return false;
    }
    bool changed = false;
    {
        std::unique_lock<std::shared_mutex> lock(worldMutex_);
        changed = applyBlockEditLocked(coord, airBlock(), false, true);
    }
    if (changed) {
        pumpChunkPropagationQueue();
    }
    return changed;
}

bool World::placeBlock(const BlockCoord& coord, const BlockMaterial& block) {
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return false;
    }
    bool changed = false;
    {
        std::unique_lock<std::shared_mutex> lock(worldMutex_);
        changed = applyBlockEditLocked(coord, block, true, false);
    }
    if (changed) {
        pumpChunkPropagationQueue();
    }
    return changed;
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

uint64_t World::playerEditRevision() const {
    return playerEditRevision_.load(std::memory_order_acquire);
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

uint64_t World::copyPlayerEditedColumnsSince(uint64_t afterRevision,
                                             std::vector<ColumnCoord>& outColumns,
                                             std::size_t maxCount) const {
    std::shared_lock<std::shared_mutex> lock(worldMutex_);
    const uint64_t currentRevision = static_cast<uint64_t>(playerEditedColumnHistory_.size());
    const uint64_t clampedRevision = std::min(afterRevision, currentRevision);
    const size_t startIndex = static_cast<size_t>(clampedRevision);
    const size_t available = playerEditedColumnHistory_.size() - startIndex;
    const size_t count = std::min(maxCount, available);

    outColumns.clear();
    outColumns.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        outColumns.push_back(playerEditedColumnHistory_[startIndex + i]);
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
    uint8_t ignoredPackedLight = unlitPackedLight();
    return tryGetBlockAndPackedLightLocked(coord, outBlock, ignoredPackedLight, mipLevel);
}

bool World::tryGetBlockAndPackedLightLocked(const BlockCoord& coord,
                                            BlockMaterial& outBlock,
                                            uint8_t& outPackedLight,
                                            uint8_t mipLevel) const {
    const uint8_t clampedMip = std::min<uint8_t>(mipLevel, Chunk::MAX_MIP_LEVEL);
    const int32_t chunkSizeAtMip = static_cast<int32_t>(Chunk::mipSize(clampedMip));
    const int32_t worldHeightAtMip = cfg::COLUMN_HEIGHT_BLOCKS >> clampedMip;

    if (coord.v.z < 0 || coord.v.z >= worldHeightAtMip) {
        outBlock = airBlock();
        outPackedLight = unlitPackedLight();
        return false;
    }

    const ChunkCoord chunkCoord{
        floor_div(coord.v.x, chunkSizeAtMip),
        floor_div(coord.v.y, chunkSizeAtMip),
        floor_div(coord.v.z, chunkSizeAtMip)
    };
    if (chunkCoord.v.z < 0 || chunkCoord.v.z >= cfg::COLUMN_HEIGHT) {
        outBlock = airBlock();
        outPackedLight = unlitPackedLight();
        return false;
    }

    const ColumnCoord columnCoord = chunk_to_column(chunkCoord);
    const RegionCoord regionCoord = column_to_region(columnCoord);

    // A region may exist while many of its columns are still ungenerated.
    // Treat those columns as unknown so meshing can apply boundary policy.
    if (generatedColumns_.find(columnCoord) == generatedColumns_.end()) {
        outBlock = airBlock();
        outPackedLight = unlitPackedLight();
        return false;
    }

    const auto regionIt = regions_.find(regionCoord);
    if (regionIt == regions_.end() || regionIt->second == nullptr) {
        outBlock = airBlock();
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
    const Chunk& chunk = column.getChunk(static_cast<uint8_t>(chunkCoord.v.z));

    outBlock = chunk.getBlock(
        static_cast<uint8_t>(localBlock.x),
        static_cast<uint8_t>(localBlock.y),
        static_cast<uint8_t>(localBlock.z),
        clampedMip
    );
    outPackedLight = chunk.getPackedLight(
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
    BlockMaterial ignoredBlock = airBlock();
    return tryGetBlockAndPackedLightLocked(coord, ignoredBlock, outPackedLight, mipLevel);
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
        collectColumnJobsToScheduleLocked(jobsToSchedule);
    }
    dispatchScheduledColumnJobs(std::move(jobsToSchedule));
}

void World::enqueueColumnGenerationLocked(const ColumnCoord& coord) {
    if (!isWithinActiveWindowLocked(coord, 0)) {
        return;
    }
    if (isColumnSkycastCompleteLocked(coord)) {
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
            isColumnSkycastCompleteLocked(top.coord) ||
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

void World::collectColumnJobsToScheduleLocked(std::vector<ScheduledColumnJob>& outJobs) {
    while (pendingColumnJobs_.size() < maxInFlightColumnJobs_ && !queuedColumnHeap_.empty()) {
        const QueuedColumnEntry top = queuedColumnHeap_.top();
        queuedColumnHeap_.pop();

        auto queuedIt = queuedColumnJobs_.find(top.coord);
        if (queuedIt == queuedColumnJobs_.end()) {
            continue;
        }

        if (!isWithinActiveWindowLocked(top.coord, 0) ||
            isColumnSkycastCompleteLocked(top.coord) ||
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
                        pumpChunkPropagationQueue();
                        return;
                    }

                    ColumnGenerationResult generated = std::move(result).value();
                    if (!generated.generated) {
                        {
                            std::unique_lock<std::shared_mutex> lock(worldMutex_);
                            pendingColumnJobs_.erase(coord);
                        }
                        pumpColumnGenerationQueue();
                        pumpChunkPropagationQueue();
                        return;
                    }
                    onColumnGenerated(generated.coord, std::move(generated.column));
                    pumpColumnGenerationQueue();
                    pumpChunkPropagationQueue();
                }
            );
        } catch (const std::exception&) {
            {
                std::unique_lock<std::shared_mutex> lock(worldMutex_);
                pendingColumnJobs_.erase(coord);
                if (!shuttingDown_.load(std::memory_order_acquire) &&
                    isWithinActiveWindowLocked(coord, 0) &&
                    !isColumnSkycastCompleteLocked(coord)) {
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

void World::pumpColumnGenerationQueue() {
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return;
    }

    std::vector<ScheduledColumnJob> jobsToSchedule;
    {
        std::unique_lock<std::shared_mutex> lock(worldMutex_);
        pruneQueuedColumnsOutsideActiveWindowLocked();
        refillQueuedColumnsLocked();
        collectColumnJobsToScheduleLocked(jobsToSchedule);
    }
    dispatchScheduledColumnJobs(std::move(jobsToSchedule));
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

    skycastColumns_.insert(coord);
    ColumnLightingState& lightingState = columnLightingStates_[coord];
    lightingState.propagatedChunkMask = 0u;
    lightingState.queuedChunkMask = 0u;

    enqueueChunkPropagationCandidatesLocked(coord);
}

bool World::applyBlockEditLocked(const BlockCoord& coord,
                                 const BlockMaterial& newBlock,
                                 bool requireCurrentAir,
                                 bool requireCurrentSolid) {
    if (coord.v.z < 0 || coord.v.z >= cfg::COLUMN_HEIGHT_BLOCKS) {
        return false;
    }
    const ChunkCoord chunkCoord = block_to_chunk(coord);
    if (chunkCoord.v.z < 0 || chunkCoord.v.z >= cfg::COLUMN_HEIGHT) {
        return false;
    }

    const ColumnCoord columnCoord = chunk_to_column(chunkCoord);
    if (!isColumnSkycastCompleteLocked(columnCoord)) {
        return false;
    }

    Column* column = tryGetSkycastColumnLocked(columnCoord);
    if (column == nullptr) {
        return false;
    }

    const uint8_t localX = static_cast<uint8_t>(floor_mod(coord.v.x, cfg::CHUNK_SIZE));
    const uint8_t localY = static_cast<uint8_t>(floor_mod(coord.v.y, cfg::CHUNK_SIZE));
    const uint16_t localZ = static_cast<uint16_t>(floor_mod(coord.v.z, cfg::COLUMN_HEIGHT_BLOCKS));
    const BlockMaterial currentBlock = column->getBlock(localX, localY, localZ);
    if (currentBlock == newBlock) {
        return false;
    }

    const bool currentIsAir = (currentBlock.unpack().id == 0u);
    if (requireCurrentAir && !currentIsAir) {
        return false;
    }
    if (requireCurrentSolid && currentIsAir) {
        return false;
    }

    column->setBlock(localX, localY, localZ, newBlock);

    std::unordered_set<ColumnCoord> geometryDirtyColumns;
    geometryDirtyColumns.insert(columnCoord);
    if (localX == 0u) {
        geometryDirtyColumns.insert(ColumnCoord{columnCoord.v.x - 1, columnCoord.v.y});
    } else if (localX == static_cast<uint8_t>(cfg::CHUNK_SIZE - 1)) {
        geometryDirtyColumns.insert(ColumnCoord{columnCoord.v.x + 1, columnCoord.v.y});
    }
    if (localY == 0u) {
        geometryDirtyColumns.insert(ColumnCoord{columnCoord.v.x, columnCoord.v.y - 1});
    } else if (localY == static_cast<uint8_t>(cfg::CHUNK_SIZE - 1)) {
        geometryDirtyColumns.insert(ColumnCoord{columnCoord.v.x, columnCoord.v.y + 1});
    }

    for (const ColumnCoord& dirtyColumn : geometryDirtyColumns) {
        if (!isColumnSkycastCompleteLocked(dirtyColumn)) {
            continue;
        }
        generatedColumns_.insert(dirtyColumn);
        playerEditedColumnHistory_.push_back(dirtyColumn);
        playerEditRevision_.fetch_add(1, std::memory_order_release);
        generatedColumnHistory_.push_back(dirtyColumn);
        generationRevision_.fetch_add(1, std::memory_order_release);
    }

    for (int32_t oy = -1; oy <= 1; ++oy) {
        for (int32_t ox = -1; ox <= 1; ++ox) {
            int32_t zMin = std::max(0, chunkCoord.v.z - 1);
            const int32_t zMax = std::min(cfg::COLUMN_HEIGHT - 1, chunkCoord.v.z + 1);
            // A vertical edit can affect skylight for the entire shaft below.
            if (ox == 0 && oy == 0) {
                zMin = 0;
            }

            for (int32_t nz = zMin; nz <= zMax; ++nz) {
                const ChunkCoord affectedChunk{
                    chunkCoord.v.x + ox,
                    chunkCoord.v.y + oy,
                    nz
                };
                const ColumnCoord affectedColumn = chunk_to_column(affectedChunk);
                if (!isColumnSkycastCompleteLocked(affectedColumn)) {
                    continue;
                }

                auto stateIt = columnLightingStates_.find(affectedColumn);
                if (stateIt == columnLightingStates_.end()) {
                    continue;
                }

                const uint8_t chunkZ = static_cast<uint8_t>(nz);
                const uint32_t bit = (1u << static_cast<uint32_t>(chunkZ));
                ColumnLightingState& state = stateIt->second;
                const bool alreadyQueued = (state.queuedChunkMask & bit) != 0u;
                state.propagatedChunkMask &= ~bit;

                if (pendingChunkPropagationJobs_.find(affectedChunk) != pendingChunkPropagationJobs_.end()) {
                    state.queuedChunkMask |= bit;
                    if (!alreadyQueued) {
                        queuedChunkPropagationJobs_.push_front(affectedChunk);
                    }
                    priorityChunkPropagationJobs_.insert(affectedChunk);
                    continue;
                }

                if (!canPropagateChunkLocked(affectedChunk)) {
                    continue;
                }

                state.queuedChunkMask |= bit;
                if (!alreadyQueued) {
                    queuedChunkPropagationJobs_.push_front(affectedChunk);
                }
                priorityChunkPropagationJobs_.insert(affectedChunk);
            }
        }
    }

    return true;
}

void World::enqueueChunkPropagationCandidatesLocked(const ColumnCoord& coord) {
    auto enqueueForColumn = [&](const ColumnCoord& candidateColumn) {
        if (!isColumnSkycastCompleteLocked(candidateColumn)) {
            return;
        }
        for (int32_t z = 0; z < cfg::COLUMN_HEIGHT; ++z) {
            enqueueChunkPropagationIfReadyLocked(ChunkCoord{
                candidateColumn.v.x,
                candidateColumn.v.y,
                z
            });
        }
    };

    enqueueForColumn(coord);
    for (const glm::ivec2& offset : kHorizontalOffsets) {
        enqueueForColumn(ColumnCoord{
            coord.v.x + offset.x,
            coord.v.y + offset.y
        });
    }
}

void World::enqueueChunkPropagationIfReadyLocked(const ChunkCoord& coord) {
    if (!canPropagateChunkLocked(coord)) {
        return;
    }

    if (coord.v.z < 0 || coord.v.z >= cfg::COLUMN_HEIGHT) {
        return;
    }

    const ColumnCoord columnCoord = chunk_to_column(coord);
    auto stateIt = columnLightingStates_.find(columnCoord);
    if (stateIt == columnLightingStates_.end()) {
        return;
    }

    const uint8_t chunkZ = static_cast<uint8_t>(coord.v.z);
    const uint32_t bit = (1u << static_cast<uint32_t>(chunkZ));
    ColumnLightingState& state = stateIt->second;
    if ((state.propagatedChunkMask & bit) != 0u ||
        (state.queuedChunkMask & bit) != 0u ||
        pendingChunkPropagationJobs_.find(coord) != pendingChunkPropagationJobs_.end()) {
        return;
    }

    state.queuedChunkMask |= bit;
    queuedChunkPropagationJobs_.push_back(coord);
}

void World::collectChunkPropagationJobsLocked(std::vector<ChunkCoord>& outChunks) {
    while (pendingChunkPropagationJobs_.size() < maxInFlightChunkPropagationJobs_ &&
           !queuedChunkPropagationJobs_.empty()) {
        const ChunkCoord coord = queuedChunkPropagationJobs_.front();
        queuedChunkPropagationJobs_.pop_front();
        if (coord.v.z < 0 || coord.v.z >= cfg::COLUMN_HEIGHT) {
            continue;
        }

        const ColumnCoord columnCoord = chunk_to_column(coord);
        auto stateIt = columnLightingStates_.find(columnCoord);
        if (stateIt == columnLightingStates_.end()) {
            priorityChunkPropagationJobs_.erase(coord);
            continue;
        }

        const uint8_t chunkZ = static_cast<uint8_t>(coord.v.z);
        const uint32_t bit = (1u << static_cast<uint32_t>(chunkZ));
        ColumnLightingState& state = stateIt->second;
        if ((state.queuedChunkMask & bit) == 0u) {
            priorityChunkPropagationJobs_.erase(coord);
            continue;
        }

        state.queuedChunkMask &= ~bit;
        if ((state.propagatedChunkMask & bit) != 0u) {
            priorityChunkPropagationJobs_.erase(coord);
            continue;
        }
        if (!canPropagateChunkLocked(coord)) {
            priorityChunkPropagationJobs_.erase(coord);
            continue;
        }

        pendingChunkPropagationJobs_.insert(coord);
        outChunks.push_back(coord);
    }
}

void World::dispatchChunkPropagationJobs(std::vector<ChunkCoord>&& chunksToSchedule) {
    for (const ChunkCoord& coord : chunksToSchedule) {
        const int32_t distanceSq = hasLastScheduledCenter_
            ? distanceSqToCenter(chunk_to_column(coord), lastScheduledCenter_)
            : 0;
        const bool highPriority = (priorityChunkPropagationJobs_.erase(coord) > 0u);
        const jobsystem::Priority priority = highPriority
            ? jobsystem::Priority::Critical
            : priorityFromDistanceSq(distanceSq);

        try {
            chunkPropagationJobs_.schedule(
                priority,
                [this, coord]() -> ChunkPropagationResult {
                    return ChunkPropagationResult{
                        coord,
                        propagateChunkLighting(coord)
                    };
                },
                [this, coord, highPriority](jobsystem::JobResult<ChunkPropagationResult>&& result) {
                    {
                        std::unique_lock<std::shared_mutex> lock(worldMutex_);
                        pendingChunkPropagationJobs_.erase(coord);

                        bool propagated = false;
                        if (result.success()) {
                            ChunkPropagationResult propagationResult = std::move(result).value();
                            propagated = propagationResult.propagated;
                        }

                        if (!propagated && !shuttingDown_.load(std::memory_order_acquire)) {
                            const ColumnCoord columnCoord = chunk_to_column(coord);
                            const auto stateIt = columnLightingStates_.find(columnCoord);
                            if (stateIt != columnLightingStates_.end()) {
                                const uint8_t chunkZ = static_cast<uint8_t>(coord.v.z);
                                const uint32_t bit = (1u << static_cast<uint32_t>(chunkZ));
                                if ((stateIt->second.propagatedChunkMask & bit) == 0u &&
                                    (stateIt->second.queuedChunkMask & bit) == 0u &&
                                    canPropagateChunkLocked(coord)) {
                                    stateIt->second.queuedChunkMask |= bit;
                                    queuedChunkPropagationJobs_.push_back(coord);
                                    if (highPriority) {
                                        priorityChunkPropagationJobs_.insert(coord);
                                    }
                                }
                            }
                        }
                    }
                    pumpChunkPropagationQueue();
                }
            );
        } catch (const std::exception&) {
            std::unique_lock<std::shared_mutex> lock(worldMutex_);
            pendingChunkPropagationJobs_.erase(coord);
            if (!shuttingDown_.load(std::memory_order_acquire) && canPropagateChunkLocked(coord)) {
                const ColumnCoord columnCoord = chunk_to_column(coord);
                auto stateIt = columnLightingStates_.find(columnCoord);
                if (stateIt != columnLightingStates_.end()) {
                    const uint8_t chunkZ = static_cast<uint8_t>(coord.v.z);
                    const uint32_t bit = (1u << static_cast<uint32_t>(chunkZ));
                    if ((stateIt->second.propagatedChunkMask & bit) == 0u &&
                        (stateIt->second.queuedChunkMask & bit) == 0u) {
                        stateIt->second.queuedChunkMask |= bit;
                        queuedChunkPropagationJobs_.push_back(coord);
                        if (highPriority) {
                            priorityChunkPropagationJobs_.insert(coord);
                        }
                    }
                }
            }
        }
    }
}

void World::pumpChunkPropagationQueue() {
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return;
    }

    std::vector<ChunkCoord> chunksToSchedule;
    {
        std::unique_lock<std::shared_mutex> lock(worldMutex_);
        collectChunkPropagationJobsLocked(chunksToSchedule);
    }
    dispatchChunkPropagationJobs(std::move(chunksToSchedule));
}

bool World::propagateChunkLighting(const ChunkCoord& coord) {
    if (coord.v.z < 0 || coord.v.z >= cfg::COLUMN_HEIGHT) {
        return false;
    }

    struct ChunkPropagationSnapshot {
        std::array<uint8_t, Chunk::VOLUME> sky{};
        std::array<uint8_t, Chunk::VOLUME> blockLight{};
        std::array<uint8_t, Chunk::VOLUME> blockLightLoss{};
        std::array<uint8_t, Chunk::VOLUME> skyVerticalLoss{};
        std::array<uint8_t, Chunk::VOLUME> blocksLightMask{};
    };

    const ColumnCoord columnCoord = chunk_to_column(coord);
    const uint8_t chunkZ = static_cast<uint8_t>(coord.v.z);
    const uint32_t bit = (1u << static_cast<uint32_t>(chunkZ));
    ChunkPropagationSnapshot snapshot{};
    std::vector<int> floodQueue;
    floodQueue.reserve(Chunk::VOLUME);

    {
        std::shared_lock<std::shared_mutex> lock(worldMutex_);
        if (!canPropagateChunkLocked(coord)) {
            return false;
        }

        const auto stateIt = columnLightingStates_.find(columnCoord);
        if (stateIt == columnLightingStates_.end()) {
            return false;
        }
        if ((stateIt->second.propagatedChunkMask & bit) != 0u) {
            return true;
        }

        const Column* centerColumn = tryGetSkycastColumnLocked(columnCoord);
        if (centerColumn == nullptr) {
            return false;
        }
        const Chunk& centerChunk = centerColumn->getChunk(chunkZ);

        for (int z = 0; z < kChunkExtent; ++z) {
            for (int y = 0; y < kChunkExtent; ++y) {
                for (int x = 0; x < kChunkExtent; ++x) {
                    const int index = chunkLocalIndex(x, y, z);
                    const BlockMaterial block = centerChunk.getBlock(
                        static_cast<uint8_t>(x),
                        static_cast<uint8_t>(y),
                        static_cast<uint8_t>(z)
                    );
                    const uint16_t materialId = block.unpack().id;
                    const bool blocksLight = MaterialLightProperties::blocksLight(materialId);
                    const uint8_t packedLight = centerChunk.getPackedLight(
                        static_cast<uint8_t>(x),
                        static_cast<uint8_t>(y),
                        static_cast<uint8_t>(z)
                    );
                    snapshot.blocksLightMask[static_cast<size_t>(index)] = blocksLight ? 1u : 0u;
                    snapshot.blockLightLoss[static_cast<size_t>(index)] = MaterialLightProperties::blockLightStepLoss(materialId);
                    snapshot.skyVerticalLoss[static_cast<size_t>(index)] = MaterialLightProperties::skyLightVerticalLoss(materialId);
                    snapshot.sky[static_cast<size_t>(index)] = 0u;
                    snapshot.blockLight[static_cast<size_t>(index)] = Chunk::unpackBlockLight(packedLight);
                }
            }
        }

        const Column* plusXColumn = tryGetSkycastColumnLocked(ColumnCoord{columnCoord.v.x + 1, columnCoord.v.y});
        const Column* minusXColumn = tryGetSkycastColumnLocked(ColumnCoord{columnCoord.v.x - 1, columnCoord.v.y});
        const Column* plusYColumn = tryGetSkycastColumnLocked(ColumnCoord{columnCoord.v.x, columnCoord.v.y + 1});
        const Column* minusYColumn = tryGetSkycastColumnLocked(ColumnCoord{columnCoord.v.x, columnCoord.v.y - 1});
        const Chunk* plusXChunk = (plusXColumn != nullptr) ? &plusXColumn->getChunk(chunkZ) : nullptr;
        const Chunk* minusXChunk = (minusXColumn != nullptr) ? &minusXColumn->getChunk(chunkZ) : nullptr;
        const Chunk* plusYChunk = (plusYColumn != nullptr) ? &plusYColumn->getChunk(chunkZ) : nullptr;
        const Chunk* minusYChunk = (minusYColumn != nullptr) ? &minusYColumn->getChunk(chunkZ) : nullptr;
        const Chunk* plusZChunk =
            (chunkZ + 1u < static_cast<uint8_t>(cfg::COLUMN_HEIGHT))
            ? &centerColumn->getChunk(static_cast<uint8_t>(chunkZ + 1u))
            : nullptr;

        auto enqueueBoundarySeed = [&](int lx,
                                       int ly,
                                       int lz,
                                       const Chunk* neighbor,
                                       int nx,
                                       int ny,
                                       int nz,
                                       bool verticalStep) {
            const int localIndex = chunkLocalIndex(lx, ly, lz);
            if (snapshot.blocksLightMask[static_cast<size_t>(localIndex)] != 0u) {
                return;
            }
            if (neighbor == nullptr) {
                return;
            }
            const BlockMaterial neighborBlock = neighbor->getBlock(
                static_cast<uint8_t>(nx),
                static_cast<uint8_t>(ny),
                static_cast<uint8_t>(nz)
            );
            if (MaterialLightProperties::blocksLight(neighborBlock.unpack().id)) {
                return;
            }
            const uint8_t neighborSky = Chunk::unpackSkyLight(neighbor->getPackedLight(
                static_cast<uint8_t>(nx),
                static_cast<uint8_t>(ny),
                static_cast<uint8_t>(nz)
            ));
            const uint8_t loss = verticalStep
                ? snapshot.skyVerticalLoss[static_cast<size_t>(localIndex)]
                : snapshot.blockLightLoss[static_cast<size_t>(localIndex)];
            const uint8_t candidate = attenuateLight(
                neighborSky,
                loss
            );
            if (candidate == 0u) {
                return;
            }

            uint8_t& current = snapshot.sky[static_cast<size_t>(localIndex)];
            if (candidate <= current) {
                return;
            }

            current = candidate;
            floodQueue.push_back(localIndex);
        };

        if (chunkZ == static_cast<uint8_t>(cfg::COLUMN_HEIGHT - 1)) {
            for (int y = 0; y < kChunkExtent; ++y) {
                for (int x = 0; x < kChunkExtent; ++x) {
                    const int localIndex = chunkLocalIndex(x, y, kChunkExtent - 1);
                    if (snapshot.blocksLightMask[static_cast<size_t>(localIndex)] != 0u) {
                        continue;
                    }

                    const uint8_t candidate = attenuateLight(
                        15u,
                        snapshot.skyVerticalLoss[static_cast<size_t>(localIndex)]
                    );
                    if (candidate == 0u) {
                        continue;
                    }

                    uint8_t& current = snapshot.sky[static_cast<size_t>(localIndex)];
                    if (candidate <= current) {
                        continue;
                    }
                    current = candidate;
                    floodQueue.push_back(localIndex);
                }
            }
        }

        for (int z = 0; z < kChunkExtent; ++z) {
            for (int y = 0; y < kChunkExtent; ++y) {
                enqueueBoundarySeed(kChunkExtent - 1, y, z, plusXChunk, 0, y, z, false);
                enqueueBoundarySeed(0, y, z, minusXChunk, kChunkExtent - 1, y, z, false);
            }
        }
        for (int z = 0; z < kChunkExtent; ++z) {
            for (int x = 0; x < kChunkExtent; ++x) {
                enqueueBoundarySeed(x, kChunkExtent - 1, z, plusYChunk, x, 0, z, false);
                enqueueBoundarySeed(x, 0, z, minusYChunk, x, kChunkExtent - 1, z, false);
            }
        }
        for (int y = 0; y < kChunkExtent; ++y) {
            for (int x = 0; x < kChunkExtent; ++x) {
                enqueueBoundarySeed(x, y, kChunkExtent - 1, plusZChunk, x, y, 0, true);
            }
        }
    }

    std::size_t queueHead = 0u;
    while (queueHead < floodQueue.size()) {
        const int index = floodQueue[queueHead++];
        const uint8_t current = snapshot.sky[static_cast<size_t>(index)];
        if (current == 0u) {
            continue;
        }

        const int x = index % kChunkExtent;
        const int y = (index / kChunkExtent) % kChunkExtent;
        const int z = index / kChunkArea;

        for (const glm::ivec3& offset : kCardinalOffsets) {
            if (offset.z > 0) {
                continue;
            }
            const int nx = x + offset.x;
            const int ny = y + offset.y;
            const int nz = z + offset.z;
            if (nx < 0 || ny < 0 || nz < 0 ||
                nx >= kChunkExtent || ny >= kChunkExtent || nz >= kChunkExtent) {
                continue;
            }

            const int neighborIndex = chunkLocalIndex(nx, ny, nz);
            if (snapshot.blocksLightMask[static_cast<size_t>(neighborIndex)] != 0u) {
                continue;
            }
            const uint8_t loss = (offset.z != 0)
                ? snapshot.skyVerticalLoss[static_cast<size_t>(neighborIndex)]
                : snapshot.blockLightLoss[static_cast<size_t>(neighborIndex)];
            const uint8_t propagated = attenuateLight(
                current,
                loss
            );
            if (propagated == 0u) {
                continue;
            }
            uint8_t& neighborSky = snapshot.sky[static_cast<size_t>(neighborIndex)];
            if (propagated <= neighborSky) {
                continue;
            }

            neighborSky = propagated;
            floodQueue.push_back(neighborIndex);
        }
    }

    std::array<uint8_t, Chunk::VOLUME> propagatedPackedLight{};
    for (int z = 0; z < kChunkExtent; ++z) {
        for (int y = 0; y < kChunkExtent; ++y) {
            for (int x = 0; x < kChunkExtent; ++x) {
                const int index = chunkLocalIndex(x, y, z);
                const bool blocksLight = snapshot.blocksLightMask[static_cast<size_t>(index)] != 0u;
                uint8_t sky = snapshot.sky[static_cast<size_t>(index)];
                if (blocksLight) {
                    sky = 0u;
                }

                propagatedPackedLight[static_cast<size_t>(index)] = Chunk::packLight(
                    sky,
                    snapshot.blockLight[static_cast<size_t>(index)]
                );
            }
        }
    }

    {
        std::unique_lock<std::shared_mutex> lock(worldMutex_);
        if (!canPropagateChunkLocked(coord)) {
            return false;
        }

        auto stateIt = columnLightingStates_.find(columnCoord);
        if (stateIt == columnLightingStates_.end()) {
            return false;
        }

        if ((stateIt->second.propagatedChunkMask & bit) != 0u) {
            return true;
        }

        Column* centerColumn = tryGetSkycastColumnLocked(columnCoord);
        if (centerColumn == nullptr) {
            return false;
        }

        const bool wasGenerated = generatedColumns_.find(columnCoord) != generatedColumns_.end();
        Chunk& centerChunk = centerColumn->getChunk(chunkZ);
        bool chunkLightChanged = false;
        for (int z = 0; z < kChunkExtent && !chunkLightChanged; ++z) {
            for (int y = 0; y < kChunkExtent && !chunkLightChanged; ++y) {
                for (int x = 0; x < kChunkExtent; ++x) {
                    const int index = chunkLocalIndex(x, y, z);
                    if (centerChunk.getPackedLight(
                            static_cast<uint8_t>(x),
                            static_cast<uint8_t>(y),
                            static_cast<uint8_t>(z)) !=
                        propagatedPackedLight[static_cast<size_t>(index)]) {
                        chunkLightChanged = true;
                        break;
                    }
                }
            }
        }

        centerChunk.setPackedLightVolume(propagatedPackedLight);
        stateIt->second.propagatedChunkMask |= bit;

        if (chunkLightChanged && chunkZ > 0u) {
            const ChunkCoord belowCoord{coord.v.x, coord.v.y, coord.v.z - 1};
            const ColumnCoord belowColumnCoord = chunk_to_column(belowCoord);
            const auto belowStateIt = columnLightingStates_.find(belowColumnCoord);
            if (belowStateIt != columnLightingStates_.end()) {
                const uint8_t belowChunkZ = static_cast<uint8_t>(belowCoord.v.z);
                const uint32_t belowBit = (1u << static_cast<uint32_t>(belowChunkZ));
                ColumnLightingState& belowState = belowStateIt->second;
                const bool belowAlreadyQueued = (belowState.queuedChunkMask & belowBit) != 0u;
                belowState.propagatedChunkMask &= ~belowBit;

                if (pendingChunkPropagationJobs_.find(belowCoord) != pendingChunkPropagationJobs_.end()) {
                    belowState.queuedChunkMask |= belowBit;
                    if (!belowAlreadyQueued) {
                        queuedChunkPropagationJobs_.push_front(belowCoord);
                    }
                    priorityChunkPropagationJobs_.insert(belowCoord);
                } else if (canPropagateChunkLocked(belowCoord)) {
                    belowState.queuedChunkMask |= belowBit;
                    if (!belowAlreadyQueued) {
                        queuedChunkPropagationJobs_.push_front(belowCoord);
                    }
                    priorityChunkPropagationJobs_.insert(belowCoord);
                }
            }
        }

        if (stateIt->second.propagatedChunkMask == kAllColumnChunkBits) {
            if (generatedColumns_.insert(columnCoord).second) {
                generatedColumnHistory_.push_back(columnCoord);
                generationRevision_.fetch_add(1, std::memory_order_release);
            } else if (wasGenerated) {
                generatedColumnHistory_.push_back(columnCoord);
                generationRevision_.fetch_add(1, std::memory_order_release);
            }
        } else if (wasGenerated) {
            generatedColumnHistory_.push_back(columnCoord);
            generationRevision_.fetch_add(1, std::memory_order_release);
        }
    }

    return true;
}

bool World::canPropagateChunkLocked(const ChunkCoord& coord) const {
    if (coord.v.z < 0 || coord.v.z >= cfg::COLUMN_HEIGHT) {
        return false;
    }

    const ColumnCoord columnCoord = chunk_to_column(coord);
    if (!isColumnSkycastCompleteLocked(columnCoord)) {
        return false;
    }

    for (const glm::ivec2& offset : kHorizontalOffsets) {
        const ColumnCoord neighborCoord{
            columnCoord.v.x + offset.x,
            columnCoord.v.y + offset.y
        };
        if (!isColumnSkycastCompleteLocked(neighborCoord)) {
            return false;
        }
    }

    return true;
}

bool World::isColumnSkycastCompleteLocked(const ColumnCoord& coord) const {
    return skycastColumns_.find(coord) != skycastColumns_.end();
}

Column* World::tryGetSkycastColumnLocked(const ColumnCoord& coord) {
    if (!isColumnSkycastCompleteLocked(coord)) {
        return nullptr;
    }

    const RegionCoord regionCoord = column_to_region(coord);
    const auto regionIt = regions_.find(regionCoord);
    if (regionIt == regions_.end() || regionIt->second == nullptr) {
        return nullptr;
    }

    const glm::ivec2 localColumn = column_local_in_region(coord);
    return &regionIt->second->getColumn(
        static_cast<uint8_t>(localColumn.x),
        static_cast<uint8_t>(localColumn.y)
    );
}

const Column* World::tryGetSkycastColumnLocked(const ColumnCoord& coord) const {
    if (!isColumnSkycastCompleteLocked(coord)) {
        return nullptr;
    }

    const RegionCoord regionCoord = column_to_region(coord);
    const auto regionIt = regions_.find(regionCoord);
    if (regionIt == regions_.end() || regionIt->second == nullptr) {
        return nullptr;
    }

    const glm::ivec2 localColumn = column_local_in_region(coord);
    return &regionIt->second->getColumn(
        static_cast<uint8_t>(localColumn.x),
        static_cast<uint8_t>(localColumn.y)
    );
}

bool World::hasPendingJobs() const {
    std::shared_lock<std::shared_mutex> lock(worldMutex_);
    return !pendingColumnJobs_.empty() ||
           !queuedColumnJobs_.empty() ||
           !pendingChunkPropagationJobs_.empty() ||
           !queuedChunkPropagationJobs_.empty();
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
