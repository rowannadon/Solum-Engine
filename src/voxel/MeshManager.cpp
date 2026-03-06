#include "solum_engine/voxel/MeshManager.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iterator>
#include <limits>
#include <mutex>
#include <queue>
#include <utility>

#include "solum_engine/resources/Constants.h"
#include "solum_engine/voxel/ChunkMesher.h"
#include "solum_engine/voxel/World.h"

namespace {
constexpr int kChunkExtent = cfg::CHUNK_SIZE;
constexpr int kPaddedChunkExtent = cfg::CHUNK_SIZE + 2;
constexpr int kPaddedChunkArea = kPaddedChunkExtent * kPaddedChunkExtent;
constexpr int kPaddedChunkVoxelCount = kPaddedChunkExtent * kPaddedChunkExtent * kPaddedChunkExtent;
constexpr int kMaxLodShift = 30;
constexpr std::size_t kWorldRevisionBatch = 2048u;

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
    std::array<uint8_t, kPaddedChunkVoxelCount> lights{};

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

    uint8_t getPackedLight(const BlockCoord& coord) const override {
        const int lx = coord.v.x - origin.v.x;
        const int ly = coord.v.y - origin.v.y;
        const int lz = coord.v.z - origin.v.z;
        if (lx < 0 || ly < 0 || lz < 0 ||
            lx >= kPaddedChunkExtent ||
            ly >= kPaddedChunkExtent ||
            lz >= kPaddedChunkExtent) {
            return Chunk::packLight(0u, 0u);
        }

        return lights[static_cast<size_t>(index(lx, ly, lz))];
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

jobsystem::Priority demotePriority(jobsystem::Priority priority) {
    switch (priority) {
    case jobsystem::Priority::Critical:
        return jobsystem::Priority::High;
    case jobsystem::Priority::High:
        return jobsystem::Priority::Normal;
    case jobsystem::Priority::Normal:
        return jobsystem::Priority::Low;
    case jobsystem::Priority::Low:
    default:
        return jobsystem::Priority::Low;
    }
}

jobsystem::Priority primaryPriorityForDistance(int32_t distanceChunks, int32_t tileSizeChunks) {
    const int32_t tileDistance = std::max(1, tileSizeChunks);
    if (distanceChunks <= tileDistance) {
        return jobsystem::Priority::Critical;
    }
    if (distanceChunks <= (tileDistance * 2)) {
        return jobsystem::Priority::High;
    }
    return jobsystem::Priority::Normal;
}

std::size_t maxPendingMeshJobs(std::size_t workerCount) {
    const std::size_t clampedWorkers = std::max<std::size_t>(workerCount, 1u);
    return std::max<std::size_t>(24u, clampedWorkers * 3u);
}

const BlockModelQuadRef* selectModelQuadRef(const BlockModelLibrary* blockModelLibrary,
                                            uint16_t materialId,
                                            uint32_t faceDirection) {
    if (blockModelLibrary == nullptr || blockModelLibrary->models.empty() || faceDirection >= 6u) {
        return nullptr;
    }

    uint16_t modelIndex = blockModelLibrary->materialToModel[materialId];

    const BlockModelDefinition* model = blockModelLibrary->modelByIndex(modelIndex);
    if (model == nullptr) {
        model = blockModelLibrary->modelByIndex(blockModelLibrary->fallbackModelIndex);
    }
    if (model == nullptr) {
        return nullptr;
    }

    auto resolveRef = [blockModelLibrary](uint32_t refIndex) -> const BlockModelQuadRef* {
        if (refIndex >= blockModelLibrary->quadRefs.size()) {
            return nullptr;
        }
        return &blockModelLibrary->quadRefs[refIndex];
    };

    if (!model->cullableQuadRefs[faceDirection].empty()) {
        if (const BlockModelQuadRef* ref = resolveRef(model->cullableQuadRefs[faceDirection][0])) {
            return ref;
        }
    }

    if (!model->nonCullableQuadRefs.empty()) {
        if (const BlockModelQuadRef* ref = resolveRef(model->nonCullableQuadRefs[0])) {
            return ref;
        }
    }

    for (uint32_t face = 0u; face < 6u; ++face) {
        if (!model->cullableQuadRefs[face].empty()) {
            if (const BlockModelQuadRef* ref = resolveRef(model->cullableQuadRefs[face][0])) {
                return ref;
            }
        }
    }

    return nullptr;
}

void appendSkirtQuad(std::vector<Meshlet>& targetMeshlets,
                     uint32_t faceDirection,
                     const glm::ivec3& origin,
                     uint32_t voxelScale,
                     uint16_t materialId,
                     const BlockModelLibrary* blockModelLibrary) {
    Meshlet skirt{};
    skirt.origin = origin;
    skirt.faceDirection = faceDirection;
    skirt.voxelScale = std::max(voxelScale, 1u);
    skirt.packedQuadLocalOffsets[0] = packMeshletLocalOffset(0u, 0u, 0u);
    skirt.quadMaterialIds[0] = materialId;
    skirt.quadAoData[0] = packMeshletQuadAoData(3u, 3u, 3u, 3u, false);
    skirt.quadLightData[0] = Chunk::packLight(15u, 0u);
    const BlockModelQuadRef* quadRef = selectModelQuadRef(blockModelLibrary, materialId, faceDirection);
    skirt.quadModelQuadIndices[0] = (quadRef != nullptr) ? quadRef->gpuQuadIndex : faceDirection;
    skirt.quadUsesVoxelAo[0] = 0u;
    if (quadRef != nullptr) {
        skirt.localBoundsMin = quadRef->minCorner;
        skirt.localBoundsMax = quadRef->maxCorner;
    } else {
        skirt.localBoundsMin = glm::vec3(0.0f);
        skirt.localBoundsMax = glm::vec3(1.0f);
    }
    skirt.hasCustomBounds = true;
    skirt.quadCount = 1u;
    targetMeshlets.push_back(skirt);
}

void appendAlwaysOnTileSkirts(ChunkMeshOutput& meshOutput,
                              const MeshTileCoord& tile,
                              int32_t meshTileSizeChunks,
                              uint8_t lodLevel,
                              const BlockModelLibrary* blockModelLibrary) {
    if (lodLevel == 0u ||
        (meshOutput.culledMeshlets.empty() && meshOutput.doubleSidedMeshlets.empty())) {
        return;
    }

    std::vector<Meshlet> culledSkirtMeshlets;
    std::vector<Meshlet> doubleSidedSkirtMeshlets;
    const int32_t tileMinX = tile.x * meshTileSizeChunks * cfg::CHUNK_SIZE;
    const int32_t tileMinY = tile.y * meshTileSizeChunks * cfg::CHUNK_SIZE;
    const int32_t tileMaxX = tileMinX + meshTileSizeChunks * cfg::CHUNK_SIZE;
    const int32_t tileMaxY = tileMinY + meshTileSizeChunks * cfg::CHUNK_SIZE;

    auto processMeshlets = [&](const std::vector<Meshlet>& meshlets) {
        for (const Meshlet& meshlet : meshlets) {
            if (meshlet.faceDirection != Direction::PlusZ || meshlet.quadCount == 0u) {
                continue;
            }

            const uint32_t voxelScale = std::max(meshlet.voxelScale, 1u);
            for (uint32_t quadIndex = 0; quadIndex < meshlet.quadCount; ++quadIndex) {
                const uint16_t packed = meshlet.packedQuadLocalOffsets[quadIndex];
                const uint16_t materialId = meshlet.quadMaterialIds[quadIndex];
                const uint32_t localX = static_cast<uint32_t>(packed & 0x1Fu);
                const uint32_t localY = static_cast<uint32_t>((packed >> 5u) & 0x1Fu);
                const uint32_t localZ = static_cast<uint32_t>((packed >> 10u) & 0x1Fu);

                const int32_t worldX = meshlet.origin.x + static_cast<int32_t>(localX * voxelScale);
                const int32_t worldY = meshlet.origin.y + static_cast<int32_t>(localY * voxelScale);
                const int32_t worldZ = meshlet.origin.z + static_cast<int32_t>(localZ * voxelScale);
                const bool materialDoubleSided = (blockModelLibrary != nullptr) &&
                    blockModelLibrary->isMaterialDoubleSided(materialId);
                std::vector<Meshlet>& targetMeshlets = materialDoubleSided
                    ? doubleSidedSkirtMeshlets
                    : culledSkirtMeshlets;

                if (worldX == tileMinX) {
                    appendSkirtQuad(
                        targetMeshlets,
                        Direction::MinusX,
                        glm::ivec3(worldX, worldY, worldZ),
                        voxelScale,
                        materialId,
                        blockModelLibrary
                    );
                }
                if ((worldX + static_cast<int32_t>(voxelScale)) == tileMaxX) {
                    appendSkirtQuad(
                        targetMeshlets,
                        Direction::PlusX,
                        glm::ivec3(worldX, worldY, worldZ),
                        voxelScale,
                        materialId,
                        blockModelLibrary
                    );
                }
                if (worldY == tileMinY) {
                    appendSkirtQuad(
                        targetMeshlets,
                        Direction::MinusY,
                        glm::ivec3(worldX, worldY, worldZ),
                        voxelScale,
                        materialId,
                        blockModelLibrary
                    );
                }
                if ((worldY + static_cast<int32_t>(voxelScale)) == tileMaxY) {
                    appendSkirtQuad(
                        targetMeshlets,
                        Direction::PlusY,
                        glm::ivec3(worldX, worldY, worldZ),
                        voxelScale,
                        materialId,
                        blockModelLibrary
                    );
                }
            }
        }
    };

    processMeshlets(meshOutput.culledMeshlets);
    processMeshlets(meshOutput.doubleSidedMeshlets);

    meshOutput.culledMeshlets.insert(
        meshOutput.culledMeshlets.end(),
        std::make_move_iterator(culledSkirtMeshlets.begin()),
        std::make_move_iterator(culledSkirtMeshlets.end())
    );
    meshOutput.doubleSidedMeshlets.insert(
        meshOutput.doubleSidedMeshlets.end(),
        std::make_move_iterator(doubleSidedSkirtMeshlets.begin()),
        std::make_move_iterator(doubleSidedSkirtMeshlets.end())
    );
}
}  // namespace

MeshManager::MeshManager(const World& world, std::shared_ptr<const BlockModelLibrary> blockModelLibrary)
    : MeshManager(world, Config{}, std::move(blockModelLibrary)) {}

MeshManager::MeshManager(const World& world, Config config, std::shared_ptr<const BlockModelLibrary> blockModelLibrary)
    : world_(world),
      blockModelLibrary_(std::move(blockModelLibrary)),
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
    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
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

    const int32_t centerShiftChunks = (centerChanged && hadPreviousCenter)
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

    const uint64_t worldRevision = world_.generationRevision();
    const uint64_t processedRevision = processedWorldGenerationRevision_.load(std::memory_order_acquire);
    if (worldRevision != processedRevision) {
        scheduleRemeshForNewColumns(centerColumn);
    }
}

std::vector<MeshTileLodUpload> MeshManager::consumePendingTileLodUploads(std::size_t maxCount) {
    std::vector<MeshTileLodUpload> uploads;
    uploads.reserve(maxCount);

    std::unique_lock<std::shared_mutex> lock(meshMutex_);
    while (!pendingUploadOrder_.empty() && uploads.size() < maxCount) {
        const MeshTileLodKey key = pendingUploadOrder_.front();
        pendingUploadOrder_.pop_front();
        pendingUploadSet_.erase(key);

        const auto tileIt = meshTiles_.find(key.tile);
        if (tileIt == meshTiles_.end()) {
            continue;
        }
        const auto lodIt = tileIt->second.lodStates.find(key.lod);
        if (lodIt == tileIt->second.lodStates.end() || !lodIt->second.resident) {
            continue;
        }

        MeshTileLodUpload upload{};
        upload.key = key;
        upload.culledMeshlets = lodIt->second.culledMeshlets;
        upload.doubleSidedMeshlets = lodIt->second.doubleSidedMeshlets;
        upload.revision = lodIt->second.revision;
        lodIt->second.uploadQueued = false;
        uploads.push_back(std::move(upload));
    }

    return uploads;
}

std::vector<MeshTileLodKey> MeshManager::consumePendingTileLodRemovals(std::size_t maxCount) {
    std::vector<MeshTileLodKey> removals;
    removals.reserve(maxCount);

    std::unique_lock<std::shared_mutex> lock(meshMutex_);
    while (!pendingRemovalOrder_.empty() && removals.size() < maxCount) {
        const MeshTileLodKey key = pendingRemovalOrder_.front();
        pendingRemovalOrder_.pop_front();
        pendingRemovalSet_.erase(key);
        removals.push_back(key);
    }

    return removals;
}

bool MeshManager::consumeSelectionSnapshot(uint64_t& outRevision,
                                           std::vector<MeshTileSelectionEntry>& outSelection) {
    std::unique_lock<std::shared_mutex> lock(meshMutex_);
    if (!selectionSnapshotDirty_) {
        return false;
    }

    outSelection.clear();
    outSelection.reserve(meshTiles_.size());
    for (const auto& [tileCoord, tileState] : meshTiles_) {
        if (tileState.selectedLod < 0) {
            continue;
        }
        outSelection.push_back(MeshTileSelectionEntry{tileCoord, tileState.selectedLod});
    }
    std::sort(outSelection.begin(), outSelection.end(), [](const MeshTileSelectionEntry& a, const MeshTileSelectionEntry& b) {
        return a.tile < b.tile;
    });

    ++selectionRevision_;
    outRevision = selectionRevision_;
    selectionSnapshotDirty_ = false;
    return true;
}

void MeshManager::queueTileLodUploadLocked(const MeshTileLodKey& key) {
    if (pendingUploadSet_.insert(key).second) {
        pendingUploadOrder_.push_back(key);
    }
}

void MeshManager::queueTileLodRemovalLocked(const MeshTileLodKey& key) {
    if (pendingRemovalSet_.insert(key).second) {
        pendingRemovalOrder_.push_back(key);
    }
}

void MeshManager::scheduleTilesAround(const ChunkCoord& centerChunk,
                                      const glm::vec3& playerWorldPosition,
                                      float sseProjectionScale,
                                      const ChunkCoord* previousCenterChunk,
                                      int32_t centerShiftChunks) {
    struct ScheduledTileLod {
        TileLodCoord coord{};
        int32_t frontierDepth = std::numeric_limits<int32_t>::max();
        int32_t distanceSq = 0;
        uint8_t tier = 1u;
        bool forceRemesh = false;
        jobsystem::Priority priority = jobsystem::Priority::Low;
    };

    const int32_t maxRadiusChunks = std::max(0, maxConfiguredRadius());
    const int32_t prefetchChunks = std::max(4, std::max(0, centerShiftChunks));
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

    // Build a contiguous generated-tile frontier rooted near the camera so meshing expands
    // outward cleanly instead of jumping to disconnected generated islands.
    std::unordered_set<MeshTileCoord> generatedTiles;
    for (int32_t tileY = minTileY; tileY <= maxTileY; ++tileY) {
        for (int32_t tileX = minTileX; tileX <= maxTileX; ++tileX) {
            const MeshTileCoord tileCoord{tileX, tileY};
            if (isTileFootprintGenerated(tileCoord)) {
                generatedTiles.insert(tileCoord);
            }
        }
    }

    std::unordered_map<MeshTileCoord, int32_t> reachableTileDepth;
    if (!generatedTiles.empty()) {
        const MeshTileCoord centerTile{
            floor_div(centerChunk.v.x, meshTileSizeChunks_),
            floor_div(centerChunk.v.y, meshTileSizeChunks_)
        };

        MeshTileCoord seed = centerTile;
        if (generatedTiles.find(seed) == generatedTiles.end()) {
            int32_t bestDistance = std::numeric_limits<int32_t>::max();
            for (const MeshTileCoord& tile : generatedTiles) {
                const int32_t dx = std::abs(tile.x - centerTile.x);
                const int32_t dy = std::abs(tile.y - centerTile.y);
                const int32_t chebyshev = std::max(dx, dy);
                if (chebyshev < bestDistance) {
                    bestDistance = chebyshev;
                    seed = tile;
                }
            }
        }

        std::queue<std::pair<MeshTileCoord, int32_t>> frontier;
        frontier.push(std::make_pair(seed, 0));
        reachableTileDepth.emplace(seed, 0);

        static constexpr std::array<glm::ivec2, 4> kCardinalNeighbors{
            glm::ivec2{1, 0},
            glm::ivec2{-1, 0},
            glm::ivec2{0, 1},
            glm::ivec2{0, -1}
        };

        while (!frontier.empty()) {
            const auto [current, depth] = frontier.front();
            frontier.pop();

            for (const glm::ivec2& n : kCardinalNeighbors) {
                const MeshTileCoord neighbor{current.x + n.x, current.y + n.y};
                if (generatedTiles.find(neighbor) == generatedTiles.end()) {
                    continue;
                }
                if (!reachableTileDepth.emplace(neighbor, depth + 1).second) {
                    continue;
                }
                frontier.push(std::make_pair(neighbor, depth + 1));
            }
        }
    }

    std::vector<ScheduledTileLod> jobsToSchedule;
    jobsToSchedule.reserve(static_cast<std::size_t>((maxTileX - minTileX + 1) * (maxTileY - minTileY + 1) * config_.lodLevelCount));

    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);

        for (auto it = meshTiles_.begin(); it != meshTiles_.end();) {
            if (tileInBounds(it->first, minTileX, maxTileX, minTileY, maxTileY) &&
                reachableTileDepth.find(it->first) != reachableTileDepth.end()) {
                ++it;
                continue;
            }

            for (const auto& [lod, lodState] : it->second.lodStates) {
                if (lodState.resident) {
                    queueTileLodRemovalLocked(MeshTileLodKey{it->first, lod});
                }
            }
            it = meshTiles_.erase(it);
            selectionSnapshotDirty_ = true;
        }

        const MeshTileCoord centerTile{
            floor_div(centerChunk.v.x, meshTileSizeChunks_),
            floor_div(centerChunk.v.y, meshTileSizeChunks_)
        };

        std::vector<MeshTileCoord> reachableOrdered;
        reachableOrdered.reserve(reachableTileDepth.size());
        for (const auto& [tileCoord, _] : reachableTileDepth) {
            reachableOrdered.push_back(tileCoord);
        }
        std::sort(reachableOrdered.begin(), reachableOrdered.end(), [&](const MeshTileCoord& a, const MeshTileCoord& b) {
            const int32_t aFrontierDepth = reachableTileDepth.at(a);
            const int32_t bFrontierDepth = reachableTileDepth.at(b);
            if (aFrontierDepth != bFrontierDepth) {
                return aFrontierDepth < bFrontierDepth;
            }
            const int32_t adx = std::abs(a.x - centerTile.x);
            const int32_t ady = std::abs(a.y - centerTile.y);
            const int32_t bdx = std::abs(b.x - centerTile.x);
            const int32_t bdy = std::abs(b.y - centerTile.y);
            const int32_t aDistance = std::max(adx, ady);
            const int32_t bDistance = std::max(bdx, bdy);
            if (aDistance != bDistance) {
                return aDistance < bDistance;
            }
            return a < b;
        });

        for (const MeshTileCoord& tileCoord : reachableOrdered) {
            MeshTileState& tileState = meshTiles_[tileCoord];
            // Drop any stale high LODs that can overlap neighboring tiles.
            for (auto lodIt = tileState.lodStates.begin(); lodIt != tileState.lodStates.end();) {
                if (static_cast<int32_t>(lodIt->first) >= config_.lodLevelCount) {
                    if (lodIt->second.resident) {
                        queueTileLodRemovalLocked(MeshTileLodKey{tileCoord, lodIt->first});
                    }
                    lodIt = tileState.lodStates.erase(lodIt);
                } else {
                    ++lodIt;
                }
            }

            const int8_t previousDesired = tileState.desiredLod;
            const int8_t candidateDesired = desiredLodForTile(
                tileCoord,
                centerChunk,
                playerWorldPosition,
                sseProjectionScale,
                0
            );
            const int8_t desired = applyLodHysteresis(
                tileCoord,
                candidateDesired,
                previousDesired,
                playerWorldPosition,
                sseProjectionScale
            );
            tileState.desiredLod = desired;
            if (desired < 0) {
                continue;
            }

            const FootprintDistanceRange distances = footprintDistanceRangeForCell(
                tileCoord.x,
                tileCoord.y,
                meshTileSizeChunks_,
                centerChunk
            );
            const int32_t distanceChunks = distances.minDistanceChunks;
            const int32_t distanceSq = distanceChunks * distanceChunks;
            const int32_t frontierDepth = reachableTileDepth.at(tileCoord);

            const jobsystem::Priority primaryPriority =
                primaryPriorityForDistance(distanceChunks, meshTileSizeChunks_);
            const jobsystem::Priority secondaryPriority = demotePriority(primaryPriority);

            bool desiredNeedsRepair = false;
            const auto desiredLodIt = tileState.lodStates.find(static_cast<uint8_t>(desired));
            if (desiredLodIt != tileState.lodStates.end() &&
                desiredLodIt->second.resident &&
                desiredLodIt->second.culledMeshlets.empty() &&
                desiredLodIt->second.doubleSidedMeshlets.empty()) {
                bool hasNonEmptyAlternateLod = false;
                for (const auto& [lodLevel, lodState] : tileState.lodStates) {
                    if (lodLevel == static_cast<uint8_t>(desired)) {
                        continue;
                    }
                    if (lodState.resident &&
                        (!lodState.culledMeshlets.empty() || !lodState.doubleSidedMeshlets.empty())) {
                        hasNonEmptyAlternateLod = true;
                        break;
                    }
                }

                if (hasNonEmptyAlternateLod) {
                    constexpr uint64_t kEmptyLodRepairCooldownRevisions = 32u;
                    const uint64_t currentRevision = meshRevision_.load(std::memory_order_acquire);
                    desiredNeedsRepair =
                        currentRevision >= (desiredLodIt->second.revision + kEmptyLodRepairCooldownRevisions);
                }
            }

            // Primary job: currently best-fit LOD for this tile.
            jobsToSchedule.push_back(ScheduledTileLod{
                TileLodCoord{tileCoord, static_cast<uint8_t>(desired)},
                frontierDepth,
                distanceSq,
                0u,
                desiredNeedsRepair,
                primaryPriority
            });

            std::vector<int32_t> supplementalLods;
            supplementalLods.reserve(static_cast<std::size_t>(config_.lodLevelCount - 1));
            for (int32_t lod = 0; lod < config_.lodLevelCount; ++lod) {
                if (lod == desired) {
                    continue;
                }
                supplementalLods.push_back(lod);
            }
            std::stable_sort(
                supplementalLods.begin(),
                supplementalLods.end(),
                [desired](int32_t a, int32_t b) {
                    const int32_t ad = std::abs(a - desired);
                    const int32_t bd = std::abs(b - desired);
                    if (ad != bd) {
                        return ad < bd;
                    }
                    // Prefer coarser supplemental LODs first.
                    return a > b;
                }
            );

            for (int32_t lod : supplementalLods) {
                jobsToSchedule.push_back(ScheduledTileLod{
                    TileLodCoord{tileCoord, static_cast<uint8_t>(lod)},
                    frontierDepth,
                    distanceSq,
                    1u,
                    false,
                    secondaryPriority
                });
            }
        }

        if (refreshSelectedLodsLocked()) {
            selectionSnapshotDirty_ = true;
        }
    }

    std::stable_sort(jobsToSchedule.begin(), jobsToSchedule.end(), [&](const ScheduledTileLod& a, const ScheduledTileLod& b) {
        if (a.frontierDepth != b.frontierDepth) {
            return a.frontierDepth < b.frontierDepth;
        }
        if (a.distanceSq != b.distanceSq) {
            return a.distanceSq < b.distanceSq;
        }
        if (a.tier != b.tier) {
            return a.tier < b.tier;
        }
        if (!(a.coord.tile == b.coord.tile)) {
            return a.coord.tile < b.coord.tile;
        }
        return false;
    });

    std::size_t pendingCount = 0u;
    {
        std::shared_lock<std::shared_mutex> lock(meshMutex_);
        pendingCount = pendingTileLodJobs_.size();
    }
    const std::size_t maxPending = maxPendingMeshJobs(jobs_.worker_count());
    std::size_t jobIndex = 0u;
    while (jobIndex < jobsToSchedule.size() && pendingCount < maxPending) {
        const ScheduledTileLod& scheduled = jobsToSchedule[jobIndex++];
        scheduleTileLodMeshing(
            scheduled.coord,
            scheduled.priority,
            scheduled.forceRemesh,
            prefetchChunks + meshTileSizeChunks_
        );

        std::shared_lock<std::shared_mutex> lock(meshMutex_);
        pendingCount = pendingTileLodJobs_.size();
    }

    (void)previousCenterChunk;
}

void MeshManager::scheduleRemeshForNewColumns(const ColumnCoord& centerColumn) {
    struct ScheduledTileLod {
        TileLodCoord coord{};
        int32_t frontierDepth = std::numeric_limits<int32_t>::max();
        int32_t distanceSq = 0;
        uint8_t tier = 1u;
        jobsystem::Priority priority = jobsystem::Priority::Low;
    };

    const uint64_t processedRevision = processedWorldGenerationRevision_.load(std::memory_order_acquire);
    std::vector<ColumnCoord> generatedColumns;
    const uint64_t nextRevision = world_.copyGeneratedColumnsSince(
        processedRevision,
        generatedColumns,
        kWorldRevisionBatch
    );
    if (nextRevision == processedRevision) {
        return;
    }
    processedWorldGenerationRevision_.store(nextRevision, std::memory_order_release);

    if (generatedColumns.empty()) {
        return;
    }

    std::unordered_set<MeshTileCoord> tilesToRemesh;
    const int32_t remeshRadius = std::max(0, maxConfiguredRadius() + meshTileSizeChunks_ + 4);

    for (const ColumnCoord& coord : generatedColumns) {
        const int32_t dx = std::abs(coord.v.x - centerColumn.v.x);
        const int32_t dy = std::abs(coord.v.y - centerColumn.v.y);
        if (dx > remeshRadius || dy > remeshRadius) {
            continue;
        }

        const int32_t tileX = floor_div(coord.v.x, meshTileSizeChunks_);
        const int32_t tileY = floor_div(coord.v.y, meshTileSizeChunks_);
        for (int32_t oy = -1; oy <= 1; ++oy) {
            for (int32_t ox = -1; ox <= 1; ++ox) {
                tilesToRemesh.insert(MeshTileCoord{tileX + ox, tileY + oy});
            }
        }
    }

    std::vector<ScheduledTileLod> jobsToSchedule;
    jobsToSchedule.reserve(tilesToRemesh.size() * static_cast<std::size_t>(config_.lodLevelCount));

    const ChunkCoord centerChunk = hasLastScheduledCenter_
        ? lastScheduledCenterChunk_
        : ChunkCoord{centerColumn.v.x, centerColumn.v.y, 0};

    const glm::vec3 playerWorldPosition = lastPlayerWorldPosition_;
    const float sseProjectionScale = hasLastSseProjectionScale_
        ? lastSseProjectionScale_
        : config_.lodSseFallbackProjectionScale;

    std::unordered_map<MeshTileCoord, int32_t> remeshDepth;
    if (!tilesToRemesh.empty()) {
        MeshTileCoord seed{};
        bool hasSeed = false;
        int32_t bestDistance = std::numeric_limits<int32_t>::max();
        for (const MeshTileCoord& tile : tilesToRemesh) {
            const int32_t dx = std::abs(tile.x - floor_div(centerChunk.v.x, meshTileSizeChunks_));
            const int32_t dy = std::abs(tile.y - floor_div(centerChunk.v.y, meshTileSizeChunks_));
            const int32_t distance = std::max(dx, dy);
            if (!hasSeed || distance < bestDistance) {
                hasSeed = true;
                bestDistance = distance;
                seed = tile;
            }
        }

        if (hasSeed) {
            std::queue<std::pair<MeshTileCoord, int32_t>> frontier;
            frontier.push(std::make_pair(seed, 0));
            remeshDepth.emplace(seed, 0);

            static constexpr std::array<glm::ivec2, 4> kCardinalNeighbors{
                glm::ivec2{1, 0},
                glm::ivec2{-1, 0},
                glm::ivec2{0, 1},
                glm::ivec2{0, -1}
            };

            while (!frontier.empty()) {
                const auto [current, depth] = frontier.front();
                frontier.pop();

                for (const glm::ivec2& n : kCardinalNeighbors) {
                    const MeshTileCoord neighbor{current.x + n.x, current.y + n.y};
                    if (tilesToRemesh.find(neighbor) == tilesToRemesh.end()) {
                        continue;
                    }
                    if (!remeshDepth.emplace(neighbor, depth + 1).second) {
                        continue;
                    }
                    frontier.push(std::make_pair(neighbor, depth + 1));
                }
            }
        }
    }

    {
        std::shared_lock<std::shared_mutex> lock(meshMutex_);
        for (const MeshTileCoord& tile : tilesToRemesh) {
            const auto tileIt = meshTiles_.find(tile);
            if (tileIt == meshTiles_.end()) {
                continue;
            }

            const int8_t desired = (tileIt->second.desiredLod >= 0)
                ? tileIt->second.desiredLod
                : desiredLodForTile(tile, centerChunk, playerWorldPosition, sseProjectionScale, 0);
            if (desired < 0) {
                continue;
            }

            const FootprintDistanceRange distances = footprintDistanceRangeForCell(
                tile.x,
                tile.y,
                meshTileSizeChunks_,
                centerChunk
            );
            const int32_t distanceChunks = distances.minDistanceChunks;
            const int32_t distanceSq = distanceChunks * distanceChunks;
            const int32_t frontierDepth = remeshDepth.contains(tile)
                ? remeshDepth.at(tile)
                : std::numeric_limits<int32_t>::max();

            const jobsystem::Priority primaryPriority =
                primaryPriorityForDistance(distanceChunks, meshTileSizeChunks_);
            const jobsystem::Priority secondaryPriority = demotePriority(primaryPriority);

            jobsToSchedule.push_back(ScheduledTileLod{
                TileLodCoord{tile, static_cast<uint8_t>(desired)},
                frontierDepth,
                distanceSq,
                0u,
                primaryPriority
            });

            std::vector<int32_t> supplementalLods;
            supplementalLods.reserve(static_cast<std::size_t>(config_.lodLevelCount - 1));
            for (int32_t lod = 0; lod < config_.lodLevelCount; ++lod) {
                if (lod == desired) {
                    continue;
                }
                supplementalLods.push_back(lod);
            }
            std::stable_sort(
                supplementalLods.begin(),
                supplementalLods.end(),
                [desired](int32_t a, int32_t b) {
                    const int32_t ad = std::abs(a - desired);
                    const int32_t bd = std::abs(b - desired);
                    if (ad != bd) {
                        return ad < bd;
                    }
                    return a > b;
                }
            );

            for (int32_t lod : supplementalLods) {
                jobsToSchedule.push_back(ScheduledTileLod{
                    TileLodCoord{tile, static_cast<uint8_t>(lod)},
                    frontierDepth,
                    distanceSq,
                    1u,
                    secondaryPriority
                });
            }
        }
    }

    std::stable_sort(jobsToSchedule.begin(), jobsToSchedule.end(), [](const ScheduledTileLod& a, const ScheduledTileLod& b) {
        if (a.frontierDepth != b.frontierDepth) {
            return a.frontierDepth < b.frontierDepth;
        }
        if (a.distanceSq != b.distanceSq) {
            return a.distanceSq < b.distanceSq;
        }
        if (a.tier != b.tier) {
            return a.tier < b.tier;
        }
        if (!(a.coord.tile == b.coord.tile)) {
            return a.coord.tile < b.coord.tile;
        }
        return false;
    });

    std::size_t pendingCount = 0u;
    {
        std::shared_lock<std::shared_mutex> lock(meshMutex_);
        pendingCount = pendingTileLodJobs_.size();
    }
    const std::size_t maxPending = maxPendingMeshJobs(jobs_.worker_count());
    std::size_t jobIndex = 0u;
    while (jobIndex < jobsToSchedule.size() && pendingCount < maxPending) {
        const ScheduledTileLod& scheduled = jobsToSchedule[jobIndex++];
        scheduleTileLodMeshing(
            scheduled.coord,
            scheduled.priority,
            true,
            meshTileSizeChunks_ + 4
        );

        std::shared_lock<std::shared_mutex> lock(meshMutex_);
        pendingCount = pendingTileLodJobs_.size();
    }
}

void MeshManager::scheduleTileLodMeshing(const TileLodCoord& coord,
                                         jobsystem::Priority priority,
                                         bool forceRemesh,
                                         int32_t activeWindowExtraChunks) {
    if (!isTileFootprintGenerated(coord.tile)) {
        return;
    }

    {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        if (!isTileWithinActiveWindowLocked(coord.tile, activeWindowExtraChunks)) {
            return;
        }

        if (pendingTileLodJobs_.find(coord) != pendingTileLodJobs_.end()) {
            if (forceRemesh) {
                deferredRemeshTileLods_.insert(coord);
            }
            return;
        }

        MeshTileState& tileState = meshTiles_[coord.tile];
        auto& lodState = tileState.lodStates[coord.lodLevel];
        if (lodState.resident && !forceRemesh) {
            return;
        }

        pendingTileLodJobs_.insert(coord);
    }

    try {
        jobs_.schedule(
            priority,
            [this, coord, activeWindowExtraChunks]() -> MeshGenerationResult {
                {
                    std::shared_lock<std::shared_mutex> lock(meshMutex_);
                    if (!isTileWithinActiveWindowLocked(coord.tile, activeWindowExtraChunks)) {
                        return MeshGenerationResult{coord, {}, false};
                    }
                }

                if (!isTileFootprintGenerated(coord.tile)) {
                    return MeshGenerationResult{coord, {}, false};
                }

                return MeshGenerationResult{coord, meshTileLod(coord), true};
            },
            [this, coord](jobsystem::JobResult<MeshGenerationResult>&& result) {
                bool rescheduleDeferred = false;
                {
                    std::unique_lock<std::shared_mutex> lock(meshMutex_);
                    pendingTileLodJobs_.erase(coord);

                    if (!result.success() || shuttingDown_.load(std::memory_order_acquire)) {
                        deferredRemeshTileLods_.erase(coord);
                        return;
                    }

                    MeshGenerationResult meshResult = std::move(result).value();
                    if (!meshResult.meshed) {
                        deferredRemeshTileLods_.erase(coord);
                        return;
                    }

                    auto tileIt = meshTiles_.find(coord.tile);
                    if (tileIt == meshTiles_.end()) {
                        deferredRemeshTileLods_.erase(coord);
                        return;
                    }

                    MeshTileLodState& lodState = tileIt->second.lodStates[coord.lodLevel];
                    lodState.culledMeshlets = std::move(meshResult.meshOutput.culledMeshlets);
                    lodState.doubleSidedMeshlets = std::move(meshResult.meshOutput.doubleSidedMeshlets);
                    lodState.resident = true;
                    lodState.revision = meshRevision_.fetch_add(1, std::memory_order_acq_rel) + 1u;
                    queueTileLodUploadLocked(MeshTileLodKey{coord.tile, coord.lodLevel});
                    lodState.uploadQueued = true;

                    if (refreshSelectedLodsLocked()) {
                        selectionSnapshotDirty_ = true;
                    }

                    const auto deferredIt = deferredRemeshTileLods_.find(coord);
                    if (deferredIt != deferredRemeshTileLods_.end()) {
                        deferredRemeshTileLods_.erase(deferredIt);
                        rescheduleDeferred = true;
                    }
                }

                if (rescheduleDeferred) {
                    scheduleTileLodMeshing(
                        coord,
                        priorityFromLodLevel(coord.lodLevel),
                        true,
                        meshTileSizeChunks_ + 4
                    );
                }
            }
        );
    } catch (const std::exception&) {
        std::unique_lock<std::shared_mutex> lock(meshMutex_);
        pendingTileLodJobs_.erase(coord);
        deferredRemeshTileLods_.erase(coord);
    }
}

ChunkMeshOutput MeshManager::meshTileLod(const TileLodCoord& coord) const {
    const uint8_t lodLevel = coord.lodLevel;
    const int32_t spanChunks = std::max(1, chunkSpanForLod(lodLevel));
    const int32_t tileOriginChunkX = coord.tile.x * meshTileSizeChunks_;
    const int32_t tileOriginChunkY = coord.tile.y * meshTileSizeChunks_;
    const int32_t baseCellX = floor_div(tileOriginChunkX, spanChunks);
    const int32_t baseCellY = floor_div(tileOriginChunkY, spanChunks);
    const int32_t cellsPerAxis = cellCountPerAxisForLod(lodLevel);
    const int32_t zCount = chunkZCountForLod(lodLevel);

    ChunkMeshOutput meshOutput{};
    std::unordered_map<ColumnCoord, uint32_t> emptyMaskCache;
    emptyMaskCache.reserve(static_cast<size_t>(meshTileSizeChunks_ * meshTileSizeChunks_));

    for (int32_t y = 0; y < cellsPerAxis; ++y) {
        for (int32_t x = 0; x < cellsPerAxis; ++x) {
            for (int32_t z = 0; z < zCount; ++z) {
                const ChunkCoord cellCoord{baseCellX + x, baseCellY + y, z};
                if (isLodCellAllAir(cellCoord, lodLevel, emptyMaskCache)) {
                    continue;
                }

                ChunkMeshOutput cellMeshOutput = meshLodCell(cellCoord, lodLevel);
                if (!cellMeshOutput.culledMeshlets.empty()) {
                    meshOutput.culledMeshlets.insert(
                        meshOutput.culledMeshlets.end(),
                        std::make_move_iterator(cellMeshOutput.culledMeshlets.begin()),
                        std::make_move_iterator(cellMeshOutput.culledMeshlets.end())
                    );
                }
                if (!cellMeshOutput.doubleSidedMeshlets.empty()) {
                    meshOutput.doubleSidedMeshlets.insert(
                        meshOutput.doubleSidedMeshlets.end(),
                        std::make_move_iterator(cellMeshOutput.doubleSidedMeshlets.begin()),
                        std::make_move_iterator(cellMeshOutput.doubleSidedMeshlets.end())
                    );
                }
            }
        }
    }

    appendAlwaysOnTileSkirts(meshOutput, coord.tile, meshTileSizeChunks_, lodLevel, blockModelLibrary_.get());
    return meshOutput;
}

ChunkMeshOutput MeshManager::meshLodCell(const ChunkCoord& cellCoord, uint8_t lodLevel) const {
    const uint8_t mipLevel = std::min<uint8_t>(lodLevel, Chunk::MAX_MIP_LEVEL);
    const int32_t extraLodShift = std::max(
        0,
        static_cast<int32_t>(lodLevel) - static_cast<int32_t>(Chunk::MAX_MIP_LEVEL)
    );
    const int32_t sampleStrideMip = pow2ClampedShift(extraLodShift);
    const uint32_t baseVoxelScale = static_cast<uint32_t>(1u << mipLevel);
    const uint32_t voxelScale = baseVoxelScale * static_cast<uint32_t>(sampleStrideMip);

    ChunkMesher mesher(blockModelLibrary_);
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
    snapshot.lights.fill(Chunk::packLight(0u, 0u));

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
                uint8_t packedLight = Chunk::packLight(0u, 0u);
                if (!world_.tryGetBlock(coordToCopy, block, mipLevel)) {
                    if (coordToCopy.v.z >= 0 && coordToCopy.v.z < worldHeightAtMip) {
                        block = unknownCullingBlock();
                    } else {
                        block = airBlock();
                    }
                } else {
                    world_.tryGetPackedLight(coordToCopy, packedLight, mipLevel);
                }
                const size_t index = static_cast<size_t>(PaddedChunkBlockSource::index(x, y, z));
                snapshot.blocks[index] = block;
                snapshot.lights[index] = packedLight;
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
    auto hasResidentMesh = [&state](int32_t lod) {
        if (lod < 0) {
            return false;
        }
        const auto lodIt = state.lodStates.find(static_cast<uint8_t>(lod));
        return lodIt != state.lodStates.end() && lodIt->second.resident;
    };
    auto hasRenderableMesh = [&state](int32_t lod) {
        if (lod < 0) {
            return false;
        }
        const auto lodIt = state.lodStates.find(static_cast<uint8_t>(lod));
        return lodIt != state.lodStates.end() &&
               lodIt->second.resident &&
               (!lodIt->second.culledMeshlets.empty() || !lodIt->second.doubleSidedMeshlets.empty());
    };

    if (state.desiredLod >= 0) {
        // Prefer desired LOD if available, then coarser fallbacks first.
        for (int32_t lod = state.desiredLod; lod < config_.lodLevelCount; ++lod) {
            if (hasRenderableMesh(lod)) {
                return static_cast<int8_t>(lod);
            }
        }
        for (int32_t lod = state.desiredLod - 1; lod >= 0; --lod) {
            if (hasRenderableMesh(lod)) {
                return static_cast<int8_t>(lod);
            }
        }
        // If every resident option is empty, keep the prior resident fallback behavior.
        for (int32_t lod = state.desiredLod; lod < config_.lodLevelCount; ++lod) {
            if (hasResidentMesh(lod)) {
                return static_cast<int8_t>(lod);
            }
        }
        for (int32_t lod = state.desiredLod - 1; lod >= 0; --lod) {
            if (hasResidentMesh(lod)) {
                return static_cast<int8_t>(lod);
            }
        }
        return -1;
    }

    // No desired LOD yet: prefer coarsest available.
    for (int32_t lod = config_.lodLevelCount - 1; lod >= 0; --lod) {
        if (hasRenderableMesh(lod)) {
            return static_cast<int8_t>(lod);
        }
    }
    for (int32_t lod = config_.lodLevelCount - 1; lod >= 0; --lod) {
        if (hasResidentMesh(lod)) {
            return static_cast<int8_t>(lod);
        }
    }
    return -1;
}

bool MeshManager::refreshSelectedLodsLocked() {
    bool changed = false;
    for (auto& [_, tileState] : meshTiles_) {
        const int8_t selected = chooseRenderableLodForTileLocked(tileState);
        if (selected != tileState.selectedLod) {
            tileState.selectedLod = selected;
            changed = true;
        }
    }
    return changed;
}

int32_t MeshManager::cellCountPerAxisForLod(uint8_t lodLevel) const {
    const int32_t spanChunks = std::max(1, chunkSpanForLod(lodLevel));
    return std::max(1, (meshTileSizeChunks_ + spanChunks - 1) / spanChunks);
}

uint64_t MeshManager::meshRevision() const noexcept {
    return meshRevision_.load(std::memory_order_acquire);
}

bool MeshManager::hasPendingJobs() const {
    std::shared_lock<std::shared_mutex> lock(meshMutex_);
    return !pendingTileLodJobs_.empty() ||
           !deferredRemeshTileLods_.empty() ||
           !pendingUploadOrder_.empty() ||
           !pendingRemovalOrder_.empty() ||
           selectionSnapshotDirty_;
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
    // Fallback priority used for deferred remesh retries.
    if (lodLevel == 0u) {
        return jobsystem::Priority::High;
    }
    if (lodLevel == 1u) {
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
    // Cap to non-overlapping tile LODs: once chunk span exceeds tile span, adjacent tiles
    // map to the same coarse cell and overlap.
    const int32_t maxLodLevelCountForTile = tileSizeLog2 + 1;
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
