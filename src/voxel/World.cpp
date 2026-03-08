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

constexpr uint8_t kLightingDirtyTopology = 1u << 0u;
constexpr uint8_t kLightingDirtyBoundary = 1u << 1u;

uint64_t hashMix(uint64_t seed, uint64_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6u) + (seed >> 2u);
    return seed;
}
}  // namespace

struct World::ColumnGenerationResult {
    ColumnCoord coord;
    Column column;
    bool generated = false;
};

struct World::ChunkPropagationResult {
    ChunkCoord coord;
    uint64_t targetEpoch = 0u;
    bool highPriority = false;
    bool propagated = false;
    bool lightChanged = false;
    uint64_t solveSignature = 0u;
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
        const ChunkCoord editedChunk = block_to_chunk(coord);
        tryApplyImmediateLightingAround(editedChunk);
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
        const ChunkCoord editedChunk = block_to_chunk(coord);
        tryApplyImmediateLightingAround(editedChunk);
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

uint64_t World::lightingRevision() const {
    return lightingRevision_.load(std::memory_order_acquire);
}

uint64_t World::playerEditChunkRevision() const {
    return playerEditChunkRevision_.load(std::memory_order_acquire);
}

uint64_t World::lightingChunkRevision() const {
    return lightingChunkRevision_.load(std::memory_order_acquire);
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

uint64_t World::copyPlayerEditedChunksSince(uint64_t afterRevision,
                                            std::vector<ChunkCoord>& outChunks,
                                            std::size_t maxCount) const {
    std::shared_lock<std::shared_mutex> lock(worldMutex_);
    const uint64_t currentRevision = static_cast<uint64_t>(playerEditedChunkHistory_.size());
    const uint64_t clampedRevision = std::min(afterRevision, currentRevision);
    const size_t startIndex = static_cast<size_t>(clampedRevision);
    const size_t available = playerEditedChunkHistory_.size() - startIndex;
    const size_t count = std::min(maxCount, available);

    outChunks.clear();
    outChunks.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        outChunks.push_back(playerEditedChunkHistory_[startIndex + i]);
    }

    return clampedRevision + static_cast<uint64_t>(count);
}

uint64_t World::copyLightingChangedColumnsSince(uint64_t afterRevision,
                                                std::vector<ColumnCoord>& outColumns,
                                                std::size_t maxCount) const {
    std::shared_lock<std::shared_mutex> lock(worldMutex_);
    const uint64_t currentRevision = static_cast<uint64_t>(lightingChangedColumnHistory_.size());
    const uint64_t clampedRevision = std::min(afterRevision, currentRevision);
    const size_t startIndex = static_cast<size_t>(clampedRevision);
    const size_t available = lightingChangedColumnHistory_.size() - startIndex;
    const size_t count = std::min(maxCount, available);

    outColumns.clear();
    outColumns.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        outColumns.push_back(lightingChangedColumnHistory_[startIndex + i]);
    }

    return clampedRevision + static_cast<uint64_t>(count);
}

uint64_t World::copyLightingChangedChunksSince(uint64_t afterRevision,
                                               std::vector<ChunkCoord>& outChunks,
                                               std::size_t maxCount) const {
    std::shared_lock<std::shared_mutex> lock(worldMutex_);
    const uint64_t currentRevision = static_cast<uint64_t>(lightingChangedChunkHistory_.size());
    const uint64_t clampedRevision = std::min(afterRevision, currentRevision);
    const size_t startIndex = static_cast<size_t>(clampedRevision);
    const size_t available = lightingChangedChunkHistory_.size() - startIndex;
    const size_t count = std::min(maxCount, available);

    outChunks.clear();
    outChunks.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        outChunks.push_back(lightingChangedChunkHistory_[startIndex + i]);
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
    for (int32_t z = 0; z < cfg::COLUMN_HEIGHT; ++z) {
        const ChunkCoord chunkCoord{coord.v.x, coord.v.y, z};
        LightingChunkState& state = lightingChunkStates_[chunkCoord];
        if (state.topologyEpoch == 0u) {
            state.topologyEpoch = 1u;
        } else {
            ++state.topologyEpoch;
        }
        state.lightingEpoch = 0u;
        state.queuedEpoch = 0u;
        state.inFlightEpoch = 0u;
        state.dirtyFlags = static_cast<uint8_t>(kLightingDirtyTopology | kLightingDirtyBoundary);
        state.lastSolveSignature = 0u;
    }
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

    std::unordered_set<ChunkCoord> geometryDirtyChunks;
    for (const ColumnCoord& dirtyColumn : geometryDirtyColumns) {
        geometryDirtyChunks.insert(ChunkCoord{dirtyColumn.v.x, dirtyColumn.v.y, chunkCoord.v.z});
    }
    if (localZ == 0u) {
        for (const ColumnCoord& dirtyColumn : geometryDirtyColumns) {
            if (chunkCoord.v.z > 0) {
                geometryDirtyChunks.insert(ChunkCoord{dirtyColumn.v.x, dirtyColumn.v.y, chunkCoord.v.z - 1});
            }
        }
    } else if (localZ == static_cast<uint16_t>(cfg::CHUNK_SIZE - 1)) {
        for (const ColumnCoord& dirtyColumn : geometryDirtyColumns) {
            if (chunkCoord.v.z + 1 < cfg::COLUMN_HEIGHT) {
                geometryDirtyChunks.insert(ChunkCoord{dirtyColumn.v.x, dirtyColumn.v.y, chunkCoord.v.z + 1});
            }
        }
    }

    for (const ChunkCoord& dirtyChunk : geometryDirtyChunks) {
        if (dirtyChunk.v.z < 0 || dirtyChunk.v.z >= cfg::COLUMN_HEIGHT) {
            continue;
        }
        if (!isColumnSkycastCompleteLocked(chunk_to_column(dirtyChunk))) {
            continue;
        }
        playerEditedChunkHistory_.push_back(dirtyChunk);
        playerEditChunkRevision_.fetch_add(1, std::memory_order_release);
    }

    for (int32_t oy = -1; oy <= 1; ++oy) {
        for (int32_t ox = -1; ox <= 1; ++ox) {
            int32_t zMin = std::max(0, chunkCoord.v.z - 1);
            const int32_t zMax = std::min(cfg::COLUMN_HEIGHT - 1, chunkCoord.v.z + 1);
            if (ox == 0 && oy == 0) {
                zMin = 0;
            }

            for (int32_t nz = zMin; nz <= zMax; ++nz) {
                bumpChunkTopologyEpochLocked(
                    ChunkCoord{chunkCoord.v.x + ox, chunkCoord.v.y + oy, nz},
                    true,
                    static_cast<uint8_t>(kLightingDirtyTopology | kLightingDirtyBoundary)
                );
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
            bumpChunkTopologyEpochLocked(
                ChunkCoord{candidateColumn.v.x, candidateColumn.v.y, z},
                false,
                kLightingDirtyBoundary
            );
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

void World::enqueueChunkPropagationIfReadyLocked(const ChunkCoord& coord, bool highPriority) {
    if (coord.v.z < 0 || coord.v.z >= cfg::COLUMN_HEIGHT) {
        return;
    }
    if (!isChunkKnownLocked(coord)) {
        return;
    }
    LightingChunkState& state = lightingChunkStates_[coord];
    if (state.topologyEpoch == 0u) {
        state.topologyEpoch = 1u;
    }
    const uint64_t targetEpoch = state.topologyEpoch;

    if (state.lightingEpoch >= targetEpoch &&
        state.dirtyFlags == 0u) {
        return;
    }

    if (state.inFlightEpoch >= targetEpoch) {
        return;
    }

    if (state.queuedEpoch >= targetEpoch && !highPriority) {
        return;
    }

    if (!canPropagateChunkLocked(coord)) {
        return;
    }

    state.queuedEpoch = targetEpoch;
    const ChunkPropagationTask task{coord, targetEpoch, highPriority};
    if (highPriority) {
        queuedChunkPropagationJobs_.push_front(task);
    } else {
        queuedChunkPropagationJobs_.push_back(task);
    }
}

void World::collectChunkPropagationJobsLocked(std::vector<ChunkPropagationTask>& outChunks) {
    std::size_t retryBudget = queuedChunkPropagationJobs_.size();
    while (pendingChunkPropagationJobs_.size() < maxInFlightChunkPropagationJobs_ &&
           !queuedChunkPropagationJobs_.empty() &&
           retryBudget > 0u) {
        --retryBudget;
        const ChunkPropagationTask task = queuedChunkPropagationJobs_.front();
        queuedChunkPropagationJobs_.pop_front();
        const ChunkCoord coord = task.coord;
        if (coord.v.z < 0 || coord.v.z >= cfg::COLUMN_HEIGHT) {
            continue;
        }
        if (!isChunkKnownLocked(coord)) {
            continue;
        }

        auto stateIt = lightingChunkStates_.find(coord);
        if (stateIt == lightingChunkStates_.end()) {
            continue;
        }
        LightingChunkState& state = stateIt->second;
        const uint64_t latestEpoch = state.topologyEpoch;
        const uint64_t targetEpoch = std::max(task.targetEpoch, latestEpoch);

        if (state.lightingEpoch >= targetEpoch && state.dirtyFlags == 0u) {
            state.queuedEpoch = 0u;
            continue;
        }
        if (!canPropagateChunkLocked(coord)) {
            queuedChunkPropagationJobs_.push_back(ChunkPropagationTask{
                coord,
                targetEpoch,
                task.highPriority
            });
            continue;
        }

        if (state.inFlightEpoch >= targetEpoch) {
            continue;
        }

        state.queuedEpoch = 0u;
        state.inFlightEpoch = targetEpoch;
        pendingChunkPropagationJobs_[coord] = targetEpoch;
        outChunks.push_back(ChunkPropagationTask{coord, targetEpoch, task.highPriority});
    }
}

void World::dispatchChunkPropagationJobs(std::vector<ChunkPropagationTask>&& chunksToSchedule) {
    for (const ChunkPropagationTask& task : chunksToSchedule) {
        const ChunkCoord coord = task.coord;
        const uint64_t targetEpoch = task.targetEpoch;
        const int32_t distanceSq = hasLastScheduledCenter_
            ? distanceSqToCenter(chunk_to_column(coord), lastScheduledCenter_)
            : 0;
        const jobsystem::Priority priority = task.highPriority
            ? jobsystem::Priority::Critical
            : priorityFromDistanceSq(distanceSq);

        try {
            chunkPropagationJobs_.schedule(
                priority,
                [this, coord, targetEpoch, task]() -> ChunkPropagationResult {
                    bool lightChanged = false;
                    uint64_t solveSignature = 0u;
                    const bool propagated = propagateChunkLighting(
                        coord,
                        targetEpoch,
                        &lightChanged,
                        &solveSignature
                    );
                    return ChunkPropagationResult{
                        coord,
                        targetEpoch,
                        task.highPriority,
                        propagated,
                        lightChanged,
                        solveSignature
                    };
                },
                [this, coord, targetEpoch, task](jobsystem::JobResult<ChunkPropagationResult>&& result) {
                    {
                        std::unique_lock<std::shared_mutex> lock(worldMutex_);
                        auto pendingIt = pendingChunkPropagationJobs_.find(coord);
                        if (pendingIt != pendingChunkPropagationJobs_.end() &&
                            pendingIt->second == targetEpoch) {
                            pendingChunkPropagationJobs_.erase(pendingIt);
                        }

                        auto stateIt = lightingChunkStates_.find(coord);
                        if (stateIt != lightingChunkStates_.end() &&
                            stateIt->second.inFlightEpoch == targetEpoch) {
                            stateIt->second.inFlightEpoch = 0u;
                        }

                        bool propagated = false;
                        if (result.success()) {
                            ChunkPropagationResult propagationResult = std::move(result).value();
                            propagated = propagationResult.propagated;
                        }

                        if (!propagated && !shuttingDown_.load(std::memory_order_acquire)) {
                            auto stateRetryIt = lightingChunkStates_.find(coord);
                            if (stateRetryIt != lightingChunkStates_.end() &&
                                stateRetryIt->second.lightingEpoch < stateRetryIt->second.topologyEpoch) {
                                enqueueChunkPropagationIfReadyLocked(coord, task.highPriority);
                            }
                        }
                    }
                    pumpChunkPropagationQueue();
                }
            );
        } catch (const std::exception&) {
            std::unique_lock<std::shared_mutex> lock(worldMutex_);
            auto pendingIt = pendingChunkPropagationJobs_.find(coord);
            if (pendingIt != pendingChunkPropagationJobs_.end() &&
                pendingIt->second == targetEpoch) {
                pendingChunkPropagationJobs_.erase(pendingIt);
            }
            auto stateIt = lightingChunkStates_.find(coord);
            if (stateIt != lightingChunkStates_.end() &&
                stateIt->second.inFlightEpoch == targetEpoch) {
                stateIt->second.inFlightEpoch = 0u;
            }
            if (!shuttingDown_.load(std::memory_order_acquire)) {
                enqueueChunkPropagationIfReadyLocked(coord, task.highPriority);
            }
        }
    }
}

void World::pumpChunkPropagationQueue() {
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return;
    }

    std::vector<ChunkPropagationTask> chunksToSchedule;
    {
        std::unique_lock<std::shared_mutex> lock(worldMutex_);
        collectChunkPropagationJobsLocked(chunksToSchedule);
    }
    dispatchChunkPropagationJobs(std::move(chunksToSchedule));
}

void World::bumpChunkTopologyEpochLocked(const ChunkCoord& coord,
                                         bool highPriority,
                                         uint8_t dirtyFlags) {
    if (!isChunkKnownLocked(coord)) {
        return;
    }
    LightingChunkState& state = lightingChunkStates_[coord];
    if (state.topologyEpoch == 0u) {
        state.topologyEpoch = 1u;
    } else {
        ++state.topologyEpoch;
    }
    state.dirtyFlags |= dirtyFlags;
    enqueueChunkPropagationIfReadyLocked(coord, highPriority);
}

World::LightingChunkState* World::tryGetLightingChunkStateLocked(const ChunkCoord& coord) {
    auto it = lightingChunkStates_.find(coord);
    if (it == lightingChunkStates_.end()) {
        return nullptr;
    }
    return &it->second;
}

const World::LightingChunkState* World::tryGetLightingChunkStateLocked(const ChunkCoord& coord) const {
    auto it = lightingChunkStates_.find(coord);
    if (it == lightingChunkStates_.end()) {
        return nullptr;
    }
    return &it->second;
}

bool World::isChunkKnownLocked(const ChunkCoord& coord) const {
    if (coord.v.z < 0 || coord.v.z >= cfg::COLUMN_HEIGHT) {
        return false;
    }
    const ColumnCoord columnCoord = chunk_to_column(coord);
    if (!isColumnSkycastCompleteLocked(columnCoord)) {
        return false;
    }
    return tryGetSkycastColumnLocked(columnCoord) != nullptr;
}

uint64_t World::computeChunkSolveSignatureLocked(const ChunkCoord& coord) const {
    if (!isChunkKnownLocked(coord)) {
        return 0u;
    }

    const ColumnCoord columnCoord = chunk_to_column(coord);
    const uint8_t chunkZ = static_cast<uint8_t>(coord.v.z);
    const Column* centerColumn = tryGetSkycastColumnLocked(columnCoord);
    if (centerColumn == nullptr) {
        return 0u;
    }
    const Chunk& centerChunk = centerColumn->getChunk(chunkZ);

    uint64_t signature = 0xcbf29ce484222325ull;
    for (int z = 0; z < kChunkExtent; ++z) {
        for (int y = 0; y < kChunkExtent; ++y) {
            for (int x = 0; x < kChunkExtent; ++x) {
                const BlockMaterial block = centerChunk.getBlock(
                    static_cast<uint8_t>(x),
                    static_cast<uint8_t>(y),
                    static_cast<uint8_t>(z)
                );
                const uint64_t packedLight = static_cast<uint64_t>(centerChunk.getPackedLight(
                    static_cast<uint8_t>(x),
                    static_cast<uint8_t>(y),
                    static_cast<uint8_t>(z)
                ));
                signature = hashMix(signature, static_cast<uint64_t>(block.data));
                signature = hashMix(signature, packedLight);
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
    const Chunk* minusZChunk =
        (chunkZ > 0u)
        ? &centerColumn->getChunk(static_cast<uint8_t>(chunkZ - 1u))
        : nullptr;

    auto hashBoundary = [&signature](const Chunk* chunk,
                                     uint8_t fixedAxis,
                                     bool axisX,
                                     bool axisY,
                                     bool axisZ) {
        if (chunk == nullptr) {
            signature = hashMix(signature, 0xDEADBEEFull);
            return;
        }
        for (int b = 0; b < kChunkExtent; ++b) {
            for (int a = 0; a < kChunkExtent; ++a) {
                const uint8_t x = axisX ? fixedAxis : static_cast<uint8_t>(a);
                const uint8_t y = axisY ? fixedAxis : static_cast<uint8_t>(axisX ? a : b);
                const uint8_t z = axisZ ? fixedAxis : static_cast<uint8_t>(b);
                signature = hashMix(signature, static_cast<uint64_t>(chunk->getPackedLight(x, y, z)));
            }
        }
    };

    hashBoundary(plusXChunk, 0u, true, false, false);
    hashBoundary(minusXChunk, static_cast<uint8_t>(kChunkExtent - 1), true, false, false);
    hashBoundary(plusYChunk, 0u, false, true, false);
    hashBoundary(minusYChunk, static_cast<uint8_t>(kChunkExtent - 1), false, true, false);
    hashBoundary(plusZChunk, 0u, false, false, true);
    hashBoundary(minusZChunk, static_cast<uint8_t>(kChunkExtent - 1), false, false, true);

    return signature;
}

bool World::tryApplyImmediateLightingAround(const ChunkCoord& centerChunk) {
    std::array<ChunkCoord, 7> immediateChunks = {
        centerChunk,
        ChunkCoord{centerChunk.v.x + 1, centerChunk.v.y, centerChunk.v.z},
        ChunkCoord{centerChunk.v.x - 1, centerChunk.v.y, centerChunk.v.z},
        ChunkCoord{centerChunk.v.x, centerChunk.v.y + 1, centerChunk.v.z},
        ChunkCoord{centerChunk.v.x, centerChunk.v.y - 1, centerChunk.v.z},
        ChunkCoord{centerChunk.v.x, centerChunk.v.y, centerChunk.v.z + 1},
        ChunkCoord{centerChunk.v.x, centerChunk.v.y, centerChunk.v.z - 1}
    };

    bool anyApplied = false;
    for (const ChunkCoord& chunkCoord : immediateChunks) {
        uint64_t targetEpoch = 0u;
        {
            std::shared_lock<std::shared_mutex> lock(worldMutex_);
            const LightingChunkState* state = tryGetLightingChunkStateLocked(chunkCoord);
            if (state == nullptr) {
                continue;
            }
            targetEpoch = state->topologyEpoch;
        }

        bool lightChanged = false;
        if (propagateChunkLighting(chunkCoord, targetEpoch, &lightChanged, nullptr)) {
            anyApplied = true;
        }
    }

    return anyApplied;
}

bool World::propagateChunkLighting(const ChunkCoord& coord,
                                   uint64_t targetEpoch,
                                   bool* outLightChanged,
                                   uint64_t* outSolveSignature) {
    if (outLightChanged != nullptr) {
        *outLightChanged = false;
    }
    if (outSolveSignature != nullptr) {
        *outSolveSignature = 0u;
    }

    if (coord.v.z < 0 || coord.v.z >= cfg::COLUMN_HEIGHT || targetEpoch == 0u) {
        return false;
    }

    struct ChunkPropagationSnapshot {
        std::array<uint8_t, Chunk::VOLUME> sky{};
        std::array<uint8_t, Chunk::VOLUME> blockLight{};
        std::array<uint8_t, Chunk::VOLUME> emissive{};
        std::array<uint8_t, Chunk::VOLUME> oldPackedLight{};
        std::array<uint8_t, Chunk::VOLUME> blockLightLoss{};
        std::array<uint8_t, Chunk::VOLUME> skyVerticalLoss{};
        std::array<uint8_t, Chunk::VOLUME> blocksLightMask{};
    };

    const ColumnCoord columnCoord = chunk_to_column(coord);
    const uint8_t chunkZ = static_cast<uint8_t>(coord.v.z);
    ChunkPropagationSnapshot snapshot{};
    std::vector<int> skyQueue;
    std::vector<int> blockQueue;
    skyQueue.reserve(Chunk::VOLUME);
    blockQueue.reserve(Chunk::VOLUME);
    uint64_t snapshotSignature = 0u;

    {
        std::shared_lock<std::shared_mutex> lock(worldMutex_);
        if (!canPropagateChunkLocked(coord) || !isChunkKnownLocked(coord)) {
            return false;
        }

        const LightingChunkState* state = tryGetLightingChunkStateLocked(coord);
        if (state == nullptr || state->topologyEpoch != targetEpoch) {
            return false;
        }

        snapshotSignature = computeChunkSolveSignatureLocked(coord);

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
                    const uint8_t packedLight = centerChunk.getPackedLight(
                        static_cast<uint8_t>(x),
                        static_cast<uint8_t>(y),
                        static_cast<uint8_t>(z)
                    );
                    const bool blocksLight = MaterialLightProperties::blocksLight(materialId);
                    const uint8_t emissive = MaterialLightProperties::emissiveLight(materialId);
                    snapshot.blocksLightMask[static_cast<size_t>(index)] = blocksLight ? 1u : 0u;
                    snapshot.blockLightLoss[static_cast<size_t>(index)] = MaterialLightProperties::blockLightStepLoss(materialId);
                    snapshot.skyVerticalLoss[static_cast<size_t>(index)] = MaterialLightProperties::skyLightVerticalLoss(materialId);
                    snapshot.sky[static_cast<size_t>(index)] = 0u;
                    snapshot.oldPackedLight[static_cast<size_t>(index)] = packedLight;
                    snapshot.emissive[static_cast<size_t>(index)] = emissive;
                    snapshot.blockLight[static_cast<size_t>(index)] = emissive;
                    if (emissive > 0u) {
                        blockQueue.push_back(index);
                    }
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
        const Chunk* minusZChunk =
            (chunkZ > 0u)
            ? &centerColumn->getChunk(static_cast<uint8_t>(chunkZ - 1u))
            : nullptr;

        auto seedSkyFromNeighbor = [&](int lx,
                                       int ly,
                                       int lz,
                                       const Chunk* neighbor,
                                       int nx,
                                       int ny,
                                       int nz,
                                       uint8_t loss) {
            const int localIndex = chunkLocalIndex(lx, ly, lz);
            if (snapshot.blocksLightMask[static_cast<size_t>(localIndex)] != 0u) {
                return;
            }
            if (neighbor == nullptr) {
                return;
            }
            const uint8_t neighborSky = Chunk::unpackSkyLight(neighbor->getPackedLight(
                static_cast<uint8_t>(nx),
                static_cast<uint8_t>(ny),
                static_cast<uint8_t>(nz)
            ));
            const uint8_t candidate = attenuateLight(neighborSky, loss);
            if (candidate == 0u) {
                return;
            }
            uint8_t& current = snapshot.sky[static_cast<size_t>(localIndex)];
            if (candidate <= current) {
                return;
            }
            current = candidate;
            skyQueue.push_back(localIndex);
        };

        auto seedBlockFromNeighbor = [&](int lx,
                                         int ly,
                                         int lz,
                                         const Chunk* neighbor,
                                         int nx,
                                         int ny,
                                         int nz) {
            const int localIndex = chunkLocalIndex(lx, ly, lz);
            if (snapshot.blocksLightMask[static_cast<size_t>(localIndex)] != 0u &&
                snapshot.emissive[static_cast<size_t>(localIndex)] == 0u) {
                return;
            }
            if (neighbor == nullptr) {
                return;
            }
            const uint8_t neighborBlock = Chunk::unpackBlockLight(neighbor->getPackedLight(
                static_cast<uint8_t>(nx),
                static_cast<uint8_t>(ny),
                static_cast<uint8_t>(nz)
            ));
            const uint8_t candidate = attenuateLight(
                neighborBlock,
                snapshot.blockLightLoss[static_cast<size_t>(localIndex)]
            );
            if (candidate == 0u) {
                return;
            }
            uint8_t& current = snapshot.blockLight[static_cast<size_t>(localIndex)];
            if (candidate <= current) {
                return;
            }
            current = candidate;
            blockQueue.push_back(localIndex);
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
                    skyQueue.push_back(localIndex);
                }
            }
        }

        for (int z = 0; z < kChunkExtent; ++z) {
            for (int y = 0; y < kChunkExtent; ++y) {
                seedSkyFromNeighbor(
                    kChunkExtent - 1, y, z,
                    plusXChunk, 0, y, z,
                    snapshot.blockLightLoss[static_cast<size_t>(chunkLocalIndex(kChunkExtent - 1, y, z))]
                );
                seedSkyFromNeighbor(
                    0, y, z,
                    minusXChunk, kChunkExtent - 1, y, z,
                    snapshot.blockLightLoss[static_cast<size_t>(chunkLocalIndex(0, y, z))]
                );
                seedBlockFromNeighbor(kChunkExtent - 1, y, z, plusXChunk, 0, y, z);
                seedBlockFromNeighbor(0, y, z, minusXChunk, kChunkExtent - 1, y, z);
            }
        }
        for (int z = 0; z < kChunkExtent; ++z) {
            for (int x = 0; x < kChunkExtent; ++x) {
                seedSkyFromNeighbor(
                    x, kChunkExtent - 1, z,
                    plusYChunk, x, 0, z,
                    snapshot.blockLightLoss[static_cast<size_t>(chunkLocalIndex(x, kChunkExtent - 1, z))]
                );
                seedSkyFromNeighbor(
                    x, 0, z,
                    minusYChunk, x, kChunkExtent - 1, z,
                    snapshot.blockLightLoss[static_cast<size_t>(chunkLocalIndex(x, 0, z))]
                );
                seedBlockFromNeighbor(x, kChunkExtent - 1, z, plusYChunk, x, 0, z);
                seedBlockFromNeighbor(x, 0, z, minusYChunk, x, kChunkExtent - 1, z);
            }
        }
        for (int y = 0; y < kChunkExtent; ++y) {
            for (int x = 0; x < kChunkExtent; ++x) {
                seedSkyFromNeighbor(
                    x, y, kChunkExtent - 1,
                    plusZChunk, x, y, 0,
                    snapshot.skyVerticalLoss[static_cast<size_t>(chunkLocalIndex(x, y, kChunkExtent - 1))]
                );
                const uint8_t upwardLoss = static_cast<uint8_t>(std::min<uint16_t>(
                    15u,
                    static_cast<uint16_t>(snapshot.blockLightLoss[static_cast<size_t>(chunkLocalIndex(x, y, 0))]) + 1u
                ));
                seedSkyFromNeighbor(
                    x, y, 0,
                    minusZChunk, x, y, kChunkExtent - 1,
                    upwardLoss
                );
                seedBlockFromNeighbor(x, y, kChunkExtent - 1, plusZChunk, x, y, 0);
                seedBlockFromNeighbor(x, y, 0, minusZChunk, x, y, kChunkExtent - 1);
            }
        }
    }

    std::size_t skyQueueHead = 0u;
    while (skyQueueHead < skyQueue.size()) {
        const int index = skyQueue[skyQueueHead++];
        const uint8_t current = snapshot.sky[static_cast<size_t>(index)];
        if (current == 0u) {
            continue;
        }

        const int x = index % kChunkExtent;
        const int y = (index / kChunkExtent) % kChunkExtent;
        const int z = index / kChunkArea;

        for (const glm::ivec3& offset : kCardinalOffsets) {
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

            uint8_t loss = snapshot.blockLightLoss[static_cast<size_t>(neighborIndex)];
            if (offset.z < 0) {
                loss = snapshot.skyVerticalLoss[static_cast<size_t>(neighborIndex)];
            } else if (offset.z > 0) {
                const uint16_t upwardLossBase = static_cast<uint16_t>(
                    snapshot.blockLightLoss[static_cast<size_t>(neighborIndex)]
                );
                loss = static_cast<uint8_t>(std::min<uint16_t>(15u, upwardLossBase + 1u));
            }

            const uint8_t propagated = attenuateLight(current, loss);
            if (propagated == 0u) {
                continue;
            }

            uint8_t& neighborSky = snapshot.sky[static_cast<size_t>(neighborIndex)];
            if (propagated <= neighborSky) {
                continue;
            }
            neighborSky = propagated;
            skyQueue.push_back(neighborIndex);
        }
    }

    std::size_t blockQueueHead = 0u;
    while (blockQueueHead < blockQueue.size()) {
        const int index = blockQueue[blockQueueHead++];
        const uint8_t current = snapshot.blockLight[static_cast<size_t>(index)];
        if (current == 0u) {
            continue;
        }

        const int x = index % kChunkExtent;
        const int y = (index / kChunkExtent) % kChunkExtent;
        const int z = index / kChunkArea;

        for (const glm::ivec3& offset : kCardinalOffsets) {
            const int nx = x + offset.x;
            const int ny = y + offset.y;
            const int nz = z + offset.z;
            if (nx < 0 || ny < 0 || nz < 0 ||
                nx >= kChunkExtent || ny >= kChunkExtent || nz >= kChunkExtent) {
                continue;
            }

            const int neighborIndex = chunkLocalIndex(nx, ny, nz);
            if (snapshot.blocksLightMask[static_cast<size_t>(neighborIndex)] != 0u &&
                snapshot.emissive[static_cast<size_t>(neighborIndex)] == 0u) {
                continue;
            }

            const uint8_t propagated = attenuateLight(
                current,
                snapshot.blockLightLoss[static_cast<size_t>(neighborIndex)]
            );
            if (propagated == 0u) {
                continue;
            }
            uint8_t& neighborBlock = snapshot.blockLight[static_cast<size_t>(neighborIndex)];
            if (propagated <= neighborBlock) {
                continue;
            }

            neighborBlock = propagated;
            blockQueue.push_back(neighborIndex);
        }
    }

    std::array<uint8_t, Chunk::VOLUME> propagatedPackedLight{};
    for (int z = 0; z < kChunkExtent; ++z) {
        for (int y = 0; y < kChunkExtent; ++y) {
            for (int x = 0; x < kChunkExtent; ++x) {
                const int index = chunkLocalIndex(x, y, z);
                const bool blocksLight = snapshot.blocksLightMask[static_cast<size_t>(index)] != 0u;
                uint8_t sky = snapshot.sky[static_cast<size_t>(index)];
                uint8_t block = snapshot.blockLight[static_cast<size_t>(index)];
                if (blocksLight) {
                    sky = 0u;
                    if (snapshot.emissive[static_cast<size_t>(index)] == 0u) {
                        block = 0u;
                    }
                }
                propagatedPackedLight[static_cast<size_t>(index)] = Chunk::packLight(sky, block);
            }
        }
    }

    {
        std::unique_lock<std::shared_mutex> lock(worldMutex_);
        if (!canPropagateChunkLocked(coord) || !isChunkKnownLocked(coord)) {
            return false;
        }

        LightingChunkState* state = tryGetLightingChunkStateLocked(coord);
        if (state == nullptr || state->topologyEpoch != targetEpoch) {
            return false;
        }

        const uint64_t commitSignature = computeChunkSolveSignatureLocked(coord);
        if (commitSignature != snapshotSignature) {
            state->dirtyFlags |= kLightingDirtyBoundary;
            return false;
        }

        Column* centerColumn = tryGetSkycastColumnLocked(columnCoord);
        if (centerColumn == nullptr) {
            return false;
        }
        Chunk& centerChunk = centerColumn->getChunk(chunkZ);

        bool chunkLightChanged = false;
        bool plusXChanged = false;
        bool minusXChanged = false;
        bool plusYChanged = false;
        bool minusYChanged = false;
        bool plusZChanged = false;
        bool minusZChanged = false;

        for (int z = 0; z < kChunkExtent; ++z) {
            for (int y = 0; y < kChunkExtent; ++y) {
                for (int x = 0; x < kChunkExtent; ++x) {
                    const int index = chunkLocalIndex(x, y, z);
                    const uint8_t oldPacked = snapshot.oldPackedLight[static_cast<size_t>(index)];
                    const uint8_t newPacked = propagatedPackedLight[static_cast<size_t>(index)];
                    if (oldPacked == newPacked) {
                        continue;
                    }

                    chunkLightChanged = true;
                    if (x == kChunkExtent - 1) {
                        plusXChanged = true;
                    }
                    if (x == 0) {
                        minusXChanged = true;
                    }
                    if (y == kChunkExtent - 1) {
                        plusYChanged = true;
                    }
                    if (y == 0) {
                        minusYChanged = true;
                    }
                    if (z == kChunkExtent - 1) {
                        plusZChanged = true;
                    }
                    if (z == 0) {
                        minusZChanged = true;
                    }
                }
            }
        }

        centerChunk.setPackedLightVolume(propagatedPackedLight);
        state->lightingEpoch = targetEpoch;
        state->lastSolveSignature = snapshotSignature;
        state->dirtyFlags = 0u;

        const bool wasGenerated = generatedColumns_.find(columnCoord) != generatedColumns_.end();
        if (chunkLightChanged && wasGenerated) {
            lightingChangedColumnHistory_.push_back(columnCoord);
            lightingRevision_.fetch_add(1, std::memory_order_release);
            lightingChangedChunkHistory_.push_back(coord);
            lightingChunkRevision_.fetch_add(1, std::memory_order_release);
        }

        auto queueDependent = [&](const ChunkCoord& dependentCoord) {
            bumpChunkTopologyEpochLocked(dependentCoord, true, kLightingDirtyBoundary);
        };

        if (plusXChanged) {
            queueDependent(ChunkCoord{coord.v.x + 1, coord.v.y, coord.v.z});
        }
        if (minusXChanged) {
            queueDependent(ChunkCoord{coord.v.x - 1, coord.v.y, coord.v.z});
        }
        if (plusYChanged) {
            queueDependent(ChunkCoord{coord.v.x, coord.v.y + 1, coord.v.z});
        }
        if (minusYChanged) {
            queueDependent(ChunkCoord{coord.v.x, coord.v.y - 1, coord.v.z});
        }
        if (plusZChanged) {
            queueDependent(ChunkCoord{coord.v.x, coord.v.y, coord.v.z + 1});
        }
        if (minusZChanged) {
            queueDependent(ChunkCoord{coord.v.x, coord.v.y, coord.v.z - 1});
        }

        bool columnConverged = true;
        for (int32_t z = 0; z < cfg::COLUMN_HEIGHT; ++z) {
            const ChunkCoord chunkCoord{columnCoord.v.x, columnCoord.v.y, z};
            const LightingChunkState* chunkState = tryGetLightingChunkStateLocked(chunkCoord);
            if (chunkState == nullptr || chunkState->lightingEpoch < chunkState->topologyEpoch) {
                columnConverged = false;
                break;
            }
        }
        if (columnConverged && generatedColumns_.insert(columnCoord).second) {
            generatedColumnHistory_.push_back(columnCoord);
            generationRevision_.fetch_add(1, std::memory_order_release);
        }

        if (outLightChanged != nullptr) {
            *outLightChanged = chunkLightChanged;
        }
        if (outSolveSignature != nullptr) {
            *outSolveSignature = snapshotSignature;
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
