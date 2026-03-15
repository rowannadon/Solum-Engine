#include "solum_engine/voxel/World.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <shared_mutex>

#include "solum_engine/resources/Constants.h"
#include "solum_engine/voxel/Column.h"
#include "solum_engine/voxel/Region.h"

namespace {
BlockMaterial airBlock() {
    static const BlockMaterial kAir = UnpackedBlockMaterial{}.pack();
    return kAir;
}

uint8_t unlitPackedLight() {
    static constexpr uint8_t kUnlit = Chunk::packLight(0u, 0u);
    return kUnlit;
}
}  // namespace

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
    buildMeshingBlockVolumeSnapshot(
        origin,
        extent,
        stride,
        mipLevel,
        outBlocks,
        outPackedLights,
        outKnownMask
    );
}

void World::buildMeshingBlockVolumeSnapshot(const BlockCoord& origin,
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
    const uint8_t clampedMip = std::min<uint8_t>(mipLevel, Chunk::MAX_MIP_LEVEL);
    const int32_t chunkSizeAtMip = static_cast<int32_t>(Chunk::mipSize(clampedMip));
    const int32_t worldHeightAtMip = cfg::COLUMN_HEIGHT_BLOCKS >> clampedMip;

    std::shared_lock<std::shared_mutex> lock(worldMutex_);
    ColumnCoord cachedColumnCoord{};
    const Column* cachedColumn = nullptr;
    bool hasCachedColumn = false;

    auto refreshColumnCache = [&](const ColumnCoord& columnCoord) {
        cachedColumnCoord = columnCoord;
        hasCachedColumn = true;
        cachedColumn = nullptr;

        if (generatedColumns_.find(columnCoord) == generatedColumns_.end()) {
            return;
        }

        const RegionCoord regionCoord = column_to_region(columnCoord);
        const auto regionIt = regions_.find(regionCoord);
        if (regionIt == regions_.end() || regionIt->second == nullptr) {
            return;
        }

        const glm::ivec2 localColumn = column_local_in_region(columnCoord);
        cachedColumn = &regionIt->second->getColumn(
            static_cast<uint8_t>(localColumn.x),
            static_cast<uint8_t>(localColumn.y)
        );
    };

    for (int32_t x = 0; x < width; ++x) {
        for (int32_t y = 0; y < height; ++y) {
            for (int32_t z = 0; z < depth; ++z) {
                const int32_t index = (x * rowStride) + (y * depth) + z;
                const BlockCoord coord{
                    origin.v.x + (x * strideX),
                    origin.v.y + (y * strideY),
                    origin.v.z + (z * strideZ)
                };
                if (coord.v.z < 0 || coord.v.z >= worldHeightAtMip) {
                    outBlocks[index] = airBlock();
                    outPackedLights[index] = unlitPackedLight();
                    if (outKnownMask != nullptr) {
                        outKnownMask[index] = 0u;
                    }
                    continue;
                }

                const ChunkCoord chunkCoord{
                    floor_div(coord.v.x, chunkSizeAtMip),
                    floor_div(coord.v.y, chunkSizeAtMip),
                    floor_div(coord.v.z, chunkSizeAtMip)
                };
                if (chunkCoord.v.z < 0 || chunkCoord.v.z >= cfg::COLUMN_HEIGHT) {
                    outBlocks[index] = airBlock();
                    outPackedLights[index] = unlitPackedLight();
                    if (outKnownMask != nullptr) {
                        outKnownMask[index] = 0u;
                    }
                    continue;
                }

                const ColumnCoord columnCoord = chunk_to_column(chunkCoord);
                if (!hasCachedColumn || !(cachedColumnCoord == columnCoord)) {
                    refreshColumnCache(columnCoord);
                }

                if (cachedColumn == nullptr) {
                    outBlocks[index] = airBlock();
                    outPackedLights[index] = unlitPackedLight();
                    if (outKnownMask != nullptr) {
                        outKnownMask[index] = 0u;
                    }
                    continue;
                }

                const glm::ivec3 localBlock{
                    floor_mod(coord.v.x, chunkSizeAtMip),
                    floor_mod(coord.v.y, chunkSizeAtMip),
                    floor_mod(coord.v.z, chunkSizeAtMip)
                };
                const Chunk& chunk = cachedColumn->getChunk(static_cast<uint8_t>(chunkCoord.v.z));
                outBlocks[index] = chunk.getBlock(
                    static_cast<uint8_t>(localBlock.x),
                    static_cast<uint8_t>(localBlock.y),
                    static_cast<uint8_t>(localBlock.z),
                    clampedMip
                );
                outPackedLights[index] = chunk.getPackedLight(
                    static_cast<uint8_t>(localBlock.x),
                    static_cast<uint8_t>(localBlock.y),
                    static_cast<uint8_t>(localBlock.z),
                    clampedMip
                );
                if (outKnownMask != nullptr) {
                    outKnownMask[index] = 1u;
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
    return copyRevisionEventsSinceLocked(generatedColumnEvents_, afterRevision, outColumns, maxCount);
}

uint64_t World::copyPlayerEditedChunksSince(uint64_t afterRevision,
                                            std::vector<WorldChunkEdit>& outChunks,
                                            std::size_t maxCount) const {
    std::shared_lock<std::shared_mutex> lock(worldMutex_);
    return copyRevisionEventsSinceLocked(playerEditedChunkEvents_, afterRevision, outChunks, maxCount);
}

uint64_t World::copyLightingChangedChunksSince(uint64_t afterRevision,
                                               std::vector<ChunkCoord>& outChunks,
                                               std::size_t maxCount) const {
    std::shared_lock<std::shared_mutex> lock(worldMutex_);
    return copyRevisionEventsSinceLocked(lightingChangedChunkEvents_, afterRevision, outChunks, maxCount);
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
