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
#include "solum_engine/voxel/mesh_stream/MeshSnapshotBuilder.h"

namespace {
constexpr int kChunkExtent = cfg::CHUNK_SIZE;
constexpr int kPaddedChunkExtent = cfg::CHUNK_SIZE + 2;
constexpr int kPaddedChunkArea = kPaddedChunkExtent * kPaddedChunkExtent;
constexpr int kPaddedChunkVoxelCount = kPaddedChunkExtent * kPaddedChunkExtent * kPaddedChunkExtent;
constexpr int kMinPrefetchChunks = 4;
constexpr int kMaxLodShift = 30;

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

float minDistanceToInterval(float value, float minValue, float maxValue) {
    if (value < minValue) {
        return minValue - value;
    }
    if (value > maxValue) {
        return value - maxValue;
    }
    return 0.0f;
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

bool isPowerOfTwo(int32_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

int32_t floorPowerOfTwo(int32_t value) {
    if (value <= 1) {
        return 1;
    }

    int32_t floored = 1;
    while (floored <= (value / 2)) {
        floored <<= 1;
    }
    return floored;
}

int32_t integerLog2(int32_t value) {
    int32_t result = 0;
    while (value > 1) {
        value >>= 1;
        ++result;
    }
    return result;
}

int32_t pow2ClampedShift(int32_t shift) {
    const int32_t clampedShift = std::clamp(shift, 0, kMaxLodShift);
    return (1 << clampedShift);
}

}  // namespace

MeshManager::MeshManager(const World& world)
    : MeshManager(world, Config{}) {}

MeshManager::MeshManager(const World& world, Config config)
    : world_(world),
      config_(std::move(config)),
      jobs_(config_.jobConfig) {
    sanitizeConfig(config_);
    meshTileSizeChunks_ = std::max(1, config_.meshTileSizeChunks);
    processedWorldGenerationRevision_.store(world_.generationRevision(), std::memory_order_release);
}

MeshManager::~MeshManager() {
    shuttingDown_.store(true, std::memory_order_release);
    jobs_.wait_for_idle();
    jobs_.stop();
}

void MeshManager::updatePlayerPosition(const glm::vec3& playerWorldPosition, float sseProjectionScale) {
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return;
    }

    const float safeSseProjectionScale =
        (std::isfinite(sseProjectionScale) && sseProjectionScale > 0.0f)
        ? sseProjectionScale
        : config_.lodSseFallbackProjectionScale;

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
    bool sseScaleChanged = false;
    constexpr float kSseScaleChangeAbsoluteThreshold = 0.01f;
    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        if (hasLastSseProjectionScale_) {
            const float scaleDelta = std::abs(safeSseProjectionScale - lastSseProjectionScale_);
            sseScaleChanged = scaleDelta > kSseScaleChangeAbsoluteThreshold;
        } else {
            sseScaleChanged = true;
        }

        lastPlayerWorldPosition_ = playerWorldPosition;
        lastSseProjectionScale_ = safeSseProjectionScale;
        hasLastSseProjectionScale_ = true;

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
        scheduleTilesAround(
            centerChunk,
            playerWorldPosition,
            safeSseProjectionScale,
            hadPreviousCenter ? &previousCenterChunk : nullptr,
            centerShiftChunks
        );
    } else if (sseScaleChanged) {
        // A projection-scale change (e.g. framebuffer resize/FOV change) invalidates
        // SSE-based LOD selection across the active window, so force a full refresh.
        scheduleTilesAround(
            centerChunk,
            playerWorldPosition,
            safeSseProjectionScale,
            nullptr,
            meshTileSizeChunks_ * 4
        );
    } else {
        // Continuously re-evaluate a budgeted subset of tiles so LOD can respond to
        // camera motion within the same chunk and runtime tuning of SSE parameters.
        scheduleTilesAround(
            centerChunk,
            playerWorldPosition,
            safeSseProjectionScale,
            &centerChunk,
            0
        );
    }

    const uint64_t worldRevision = world_.generationRevision();
    const uint64_t processedRevision = processedWorldGenerationRevision_.load(std::memory_order_acquire);
    if (worldRevision != processedRevision) {
        scheduleRemeshForNewColumns(centerColumn);
    }

    // Limit integration of completed meshing to one tile per update call.
    applyCompletedTileResultsBudgeted();
}

std::vector<Meshlet> MeshManager::meshLodCell(const ChunkCoord& cellCoord, uint8_t lodLevel) const {
    const uint8_t mipLevel = std::min<uint8_t>(lodLevel, Chunk::MAX_MIP_LEVEL);
    const int32_t extraLodShift = std::max(
        0,
        static_cast<int32_t>(lodLevel) - static_cast<int32_t>(Chunk::MAX_MIP_LEVEL)
    );
    const int32_t sampleStrideMip = pow2ClampedShift(extraLodShift);
    const uint32_t baseVoxelScale = static_cast<uint32_t>(1u << mipLevel);
    const uint32_t voxelScale = baseVoxelScale * static_cast<uint32_t>(sampleStrideMip);

    ChunkMesher mesher;
    const BlockCoord sectionOriginSample{
        cellCoord.v.x * cfg::CHUNK_SIZE,
        cellCoord.v.y * cfg::CHUNK_SIZE,
        cellCoord.v.z * cfg::CHUNK_SIZE
    };
    const BlockCoord paddedOriginSample{
        sectionOriginSample.v.x - 1,
        sectionOriginSample.v.y - 1,
        sectionOriginSample.v.z - 1
    };

    PaddedChunkBlockSource snapshot;
    snapshot.origin = paddedOriginSample;
    snapshot.blocks.fill(airBlock());

    const int32_t worldHeightAtMip = cfg::COLUMN_HEIGHT_BLOCKS >> mipLevel;

    for (int x = 0; x < kPaddedChunkExtent; ++x) {
        for (int y = 0; y < kPaddedChunkExtent; ++y) {
            for (int z = 0; z < kPaddedChunkExtent; ++z) {
                const BlockCoord sampleCoord{
                    paddedOriginSample.v.x + x,
                    paddedOriginSample.v.y + y,
                    paddedOriginSample.v.z + z
                };
                const BlockCoord coordToCopy{
                    sampleCoord.v.x * sampleStrideMip,
                    sampleCoord.v.y * sampleStrideMip,
                    sampleCoord.v.z * sampleStrideMip
                };

                BlockMaterial block = airBlock();
                if (!world_.tryGetBlock(coordToCopy, block, mipLevel)) {
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
        sectionOriginSample.v.x * static_cast<int32_t>(voxelScale),
        sectionOriginSample.v.y * static_cast<int32_t>(voxelScale),
        sectionOriginSample.v.z * static_cast<int32_t>(voxelScale)
    };
    return mesher.mesh(
        snapshot,
        sectionOriginSample,
        sectionExtent,
        meshletOrigin,
        voxelScale
    );
}

void MeshManager::applyCompletedTileResultsBudgeted() {
    std::vector<CompletedTileCellResult> completedForTile;
    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        while (!completedTileResultOrder_.empty()) {
            const MeshTileCoord tileCoord = completedTileResultOrder_.front();
            completedTileResultOrder_.pop_front();
            completedTileResultQueued_.erase(tileCoord);

            auto completedIt = completedTileResultsByTile_.find(tileCoord);
            if (completedIt == completedTileResultsByTile_.end() || completedIt->second.empty()) {
                continue;
            }

            completedForTile = std::move(completedIt->second);
            completedTileResultsByTile_.erase(completedIt);
            break;
        }
    }

    if (completedForTile.empty()) {
        return;
    }

    std::sort(
        completedForTile.begin(),
        completedForTile.end(),
        [](const CompletedTileCellResult& a, const CompletedTileCellResult& b) {
            if (a.coord.tileLod.lodLevel != b.coord.tileLod.lodLevel) {
                return a.coord.tileLod.lodLevel < b.coord.tileLod.lodLevel;
            }
            if (a.coord.cellY != b.coord.cellY) {
                return a.coord.cellY < b.coord.cellY;
            }
            return a.coord.cellX < b.coord.cellX;
        }
    );

    for (CompletedTileCellResult& completed : completedForTile) {
        onTileLodCellMeshed(completed.coord, std::move(completed.meshlets));
    }
}

void MeshManager::onTileLodCellMeshed(const TileLodCellCoord& coord, std::vector<Meshlet>&& meshlets) {
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return;
    }

    bool needsDeferredRemesh = false;
    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        pendingTileJobs_.erase(coord);

        MeshTileState& tileState = meshTiles_[coord.tileLod.tile];
        MeshTileLodState& lodState = tileState.lodStates[coord.tileLod.lodLevel];
        lodState.cellMeshes[packCellKey(coord.cellX, coord.cellY)] = std::move(meshlets);
        const int32_t cellsPerAxis = cellCountPerAxisForLod(coord.tileLod.lodLevel);
        lodState.expectedCellCount = cellsPerAxis * cellsPerAxis;

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
        scheduleTileLodCellMeshing(
            coord,
            priorityFromLodLevel(coord.tileLod.lodLevel),
            true,
            activeWindowExtraChunks
        );
    }
}

std::vector<Meshlet> MeshManager::copyMeshletsAround(const ColumnCoord& centerColumn, int32_t columnRadius) const {
    std::shared_lock<std::shared_mutex> lock(meshMutex_);

    std::vector<MeshSnapshotTile> selected;
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

        const auto lodIt = tileState.lodStates.find(static_cast<uint8_t>(chosenLod));
        if (lodIt == tileState.lodStates.end()) {
            continue;
        }

        MeshSnapshotTile snapshotTile{};
        snapshotTile.tile = tileCoord;
        snapshotTile.lod = static_cast<uint8_t>(chosenLod);
        for (const auto& [_, cellMeshlets] : lodIt->second.cellMeshes) {
            snapshotTile.meshlets.insert(
                snapshotTile.meshlets.end(),
                cellMeshlets.begin(),
                cellMeshlets.end()
            );
        }
        selected.push_back(std::move(snapshotTile));
    }

    return MeshSnapshotBuilder::build(std::move(selected), meshTileSizeChunks_);
}

uint64_t MeshManager::meshRevision() const noexcept {
    return meshRevision_.load(std::memory_order_acquire);
}

bool MeshManager::hasPendingJobs() const {
    std::shared_lock<std::shared_mutex> lock(meshMutex_);
    return !pendingTileJobs_.empty() ||
           !deferredRemeshTileLods_.empty() ||
           !completedTileResultsByTile_.empty() ||
           !completedTileResultOrder_.empty();
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

bool MeshManager::isLodCellAllAir(const ChunkCoord& cellCoord,
                                  uint8_t lodLevel,
                                  std::unordered_map<ColumnCoord, uint32_t>& emptyMaskCache) const {
    const int32_t spanChunks = static_cast<int32_t>(chunkSpanForLod(lodLevel));
    const int32_t zStart = cellCoord.v.z * spanChunks;
    if (zStart < 0 || zStart >= cfg::COLUMN_HEIGHT) {
        return true;
    }

    const int32_t zEnd = std::min<int32_t>(cfg::COLUMN_HEIGHT, zStart + spanChunks);
    const int32_t zCount = std::max(0, zEnd - zStart);
    if (zCount <= 0) {
        return true;
    }

    uint32_t zMask = (zCount >= 32)
        ? 0xFFFFFFFFu
        : ((1u << static_cast<uint32_t>(zCount)) - 1u);
    zMask <<= static_cast<uint32_t>(zStart);

    const int32_t baseColumnX = cellCoord.v.x * spanChunks;
    const int32_t baseColumnY = cellCoord.v.y * spanChunks;

    for (int32_t dy = 0; dy < spanChunks; ++dy) {
        for (int32_t dx = 0; dx < spanChunks; ++dx) {
            const ColumnCoord columnCoord{baseColumnX + dx, baseColumnY + dy};
            uint32_t emptyMask = 0u;

            const auto cacheIt = emptyMaskCache.find(columnCoord);
            if (cacheIt != emptyMaskCache.end()) {
                emptyMask = cacheIt->second;
            } else {
                if (!world_.tryGetColumnEmptyChunkMask(columnCoord, emptyMask)) {
                    return false;
                }
                emptyMaskCache.emplace(columnCoord, emptyMask);
            }

            if ((emptyMask & zMask) != zMask) {
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
        const auto lodIt = state.lodStates.find(static_cast<uint8_t>(lod));
        if (lodIt == state.lodStates.end()) {
            return false;
        }
        if (lodIt->second.expectedCellCount <= 0) {
            return false;
        }
        return static_cast<int32_t>(lodIt->second.cellMeshes.size()) >= lodIt->second.expectedCellCount;
    };

    if (state.desiredLod >= 0 && hasMesh(state.desiredLod)) {
        return state.desiredLod;
    }

    if (state.renderedLod >= 0 && hasMesh(state.renderedLod)) {
        return state.renderedLod;
    }

    const int32_t lodCount = config_.lodLevelCount;
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
    for (const auto& [lod, lodState] : state.lodStates) {
        if (lodState.expectedCellCount <= 0) {
            continue;
        }
        if (static_cast<int32_t>(lodState.cellMeshes.size()) < lodState.expectedCellCount) {
            continue;
        }
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

int32_t MeshManager::cellSpanChunksForLod(uint8_t lodLevel) const {
    return (lodLevel == 0u) ? 1 : 2;
}

int32_t MeshManager::cellSpanLodCellsForLod(uint8_t lodLevel) const {
    const int32_t spanChunks = static_cast<int32_t>(chunkSpanForLod(lodLevel));
    const int32_t targetCellSpanChunks = cellSpanChunksForLod(lodLevel);
    return std::max(1, targetCellSpanChunks / std::max(1, spanChunks));
}

int32_t MeshManager::cellCountPerAxisForLod(uint8_t lodLevel) const {
    const int32_t spanChunks = static_cast<int32_t>(chunkSpanForLod(lodLevel));
    const int32_t lodCellsPerAxis = std::max(1, meshTileSizeChunks_ / std::max(1, spanChunks));
    const int32_t cellSpanLodCells = cellSpanLodCellsForLod(lodLevel);
    return std::max(1, (lodCellsPerAxis + cellSpanLodCells - 1) / cellSpanLodCells);
}

uint32_t MeshManager::packCellKey(uint16_t cellX, uint16_t cellY) const {
    return (static_cast<uint32_t>(cellY) << 16u) | static_cast<uint32_t>(cellX);
}

bool MeshManager::tileInBounds(const MeshTileCoord& tileCoord,
                               int32_t minTileX,
                               int32_t maxTileX,
                               int32_t minTileY,
                               int32_t maxTileY) {
    return tileCoord.x >= minTileX &&
           tileCoord.x <= maxTileX &&
           tileCoord.y >= minTileY &&
           tileCoord.y <= maxTileY;
}

int32_t MeshManager::maxConfiguredRadius() const {
    return std::max(0, config_.activeChunkRadius);
}

int32_t MeshManager::chunkSpanForLod(uint8_t lodLevel) {
    return pow2ClampedShift(static_cast<int32_t>(lodLevel));
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
    config.meshTileSizeChunks = std::max(1, config.meshTileSizeChunks);
    if (!isPowerOfTwo(config.meshTileSizeChunks)) {
        config.meshTileSizeChunks = floorPowerOfTwo(config.meshTileSizeChunks);
    }
    const int32_t tileSizeLog2 = integerLog2(config.meshTileSizeChunks);

    config.lodLevelCount = std::max(1, config.lodLevelCount);
    // Limit generated LODs so the coarsest voxel at that LOD does not exceed one tile width.
    // Additional LODs (excluding LOD0) cap at 4 + log2(tileSizeChunks): 1->4, 2->5, 4->6, etc.
    const int32_t maxAdditionalLods = static_cast<int32_t>(Chunk::MAX_MIP_LEVEL) + tileSizeLog2;
    const int32_t maxLodLevelCountForTile = maxAdditionalLods + 1;
    const int32_t maxLodLevelCountBySpan = kMaxLodShift + 1;
    config.lodLevelCount = std::clamp(
        config.lodLevelCount,
        1,
        std::min(maxLodLevelCountForTile, maxLodLevelCountBySpan)
    );

    config.activeChunkRadius = std::max(0, config.activeChunkRadius);

    if (!std::isfinite(config.lodSseTargetPixels) || config.lodSseTargetPixels <= 0.0f) {
        config.lodSseTargetPixels = 1.0f;
    }
    if (!std::isfinite(config.lodSseHysteresisPixels) || config.lodSseHysteresisPixels < 0.0f) {
        config.lodSseHysteresisPixels = 0.25f;
    }
    if (!std::isfinite(config.lodSseMinDepthBlocks) || config.lodSseMinDepthBlocks <= 0.0f) {
        config.lodSseMinDepthBlocks = 4.0f;
    }
    if (!std::isfinite(config.lodSseFallbackProjectionScale) ||
        config.lodSseFallbackProjectionScale <= 0.0f) {
        config.lodSseFallbackProjectionScale = 390.0f;
    }
}
