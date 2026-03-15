#include "solum_engine/voxel/MeshManager.h"

#include <algorithm>
#include <array>
#include <iterator>

#include "solum_engine/resources/Constants.h"
#include "solum_engine/voxel/World.h"

namespace {
constexpr int kPaddedChunkExtent = cfg::CHUNK_SIZE + 2;
constexpr int kPaddedChunkArea = kPaddedChunkExtent * kPaddedChunkExtent;
constexpr int kPaddedChunkVoxelCount = kPaddedChunkExtent * kPaddedChunkExtent * kPaddedChunkExtent;
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

int32_t pow2ClampedShift(int32_t shift) {
    const int32_t clampedShift = std::clamp(shift, 0, kMaxLodShift);
    return (1 << clampedShift);
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
    skirt.quadLightData[0] = packMeshletQuadLightPair(
        Chunk::packLight(15u, 0u),
        Chunk::packLight(15u, 0u)
    );
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

ChunkMeshOutput MeshManager::meshTileLod(const TileLodCoord& coord) const {
    const uint8_t lodLevel = coord.lodLevel;
    const int32_t spanChunks = std::max(1, chunkSpanForLod(lodLevel));
    const int32_t tileOriginChunkX = coord.tile.tile.x * meshTileSizeChunks_;
    const int32_t tileOriginChunkY = coord.tile.tile.y * meshTileSizeChunks_;
    const int32_t tileOriginChunkZ = coord.tile.z * meshTileHeightChunks_;
    const int32_t baseCellX = floor_div(tileOriginChunkX, spanChunks);
    const int32_t baseCellY = floor_div(tileOriginChunkY, spanChunks);
    const int32_t baseCellZ = floor_div(tileOriginChunkZ, spanChunks);
    const int32_t cellsPerAxis = cellCountPerAxisForLod(lodLevel);
    const int32_t cellsPerZ = cellCountPerZForLod(lodLevel);
    const int32_t zCount = chunkZCountForLod(lodLevel);

    ChunkMeshOutput meshOutput{};
    std::unordered_map<ColumnCoord, uint32_t> emptyMaskCache;
    emptyMaskCache.reserve(static_cast<size_t>(meshTileSizeChunks_ * meshTileSizeChunks_));

    for (int32_t y = 0; y < cellsPerAxis; ++y) {
        for (int32_t x = 0; x < cellsPerAxis; ++x) {
            for (int32_t z = 0; z < cellsPerZ; ++z) {
                const int32_t cellZ = baseCellZ + z;
                if (cellZ < 0 || cellZ >= zCount) {
                    continue;
                }

                const ChunkCoord cellCoord{baseCellX + x, baseCellY + y, cellZ};
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

    appendAlwaysOnTileSkirts(meshOutput, coord.tile.tile, meshTileSizeChunks_, lodLevel, blockModelLibrary_.get());
    return meshOutput;
}

ChunkMeshOutput MeshManager::meshLodCell(const ChunkCoord& cellCoord, uint8_t lodLevel) const {
    const uint8_t mipLevel = std::min<uint8_t>(lodLevel, Chunk::MAX_MIP_LEVEL);
    const int32_t spanChunks = std::max(1, chunkSpanForLod(lodLevel));
    const int32_t extraLodShift = std::max(
        0,
        static_cast<int32_t>(lodLevel) - static_cast<int32_t>(Chunk::MAX_MIP_LEVEL)
    );
    const int32_t sampleStrideMip = pow2ClampedShift(extraLodShift);
    const int32_t chunkSizeAtMip = std::max(1, static_cast<int32_t>(Chunk::mipSize(mipLevel)));
    const uint32_t baseVoxelScale = static_cast<uint32_t>(1u << mipLevel);
    const uint32_t voxelScale = baseVoxelScale * static_cast<uint32_t>(sampleStrideMip);

    ChunkMesher mesher(blockModelLibrary_);
    const BlockCoord sectionOriginSample{
        cellCoord.v.x * spanChunks * chunkSizeAtMip,
        cellCoord.v.y * spanChunks * chunkSizeAtMip,
        cellCoord.v.z * spanChunks * chunkSizeAtMip
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
    std::array<uint8_t, kPaddedChunkVoxelCount> knownMask{};
    knownMask.fill(0u);

    const int32_t worldHeightAtMip = cfg::COLUMN_HEIGHT_BLOCKS >> mipLevel;
    const BlockCoord paddedOriginCopy{
        paddedOriginSample.v.x * sampleStrideMip,
        paddedOriginSample.v.y * sampleStrideMip,
        paddedOriginSample.v.z * sampleStrideMip
    };
    world_.buildMeshingBlockVolumeSnapshot(
        paddedOriginCopy,
        glm::ivec3{kPaddedChunkExtent, kPaddedChunkExtent, kPaddedChunkExtent},
        glm::ivec3{sampleStrideMip, sampleStrideMip, sampleStrideMip},
        mipLevel,
        snapshot.blocks.data(),
        snapshot.lights.data(),
        knownMask.data()
    );

    for (int x = 0; x < kPaddedChunkExtent; ++x) {
        for (int y = 0; y < kPaddedChunkExtent; ++y) {
            for (int z = 0; z < kPaddedChunkExtent; ++z) {
                const size_t index = static_cast<size_t>(PaddedChunkBlockSource::index(x, y, z));
                if (knownMask[index] != 0u) {
                    continue;
                }

                const int32_t worldZ = paddedOriginCopy.v.z + (z * sampleStrideMip);
                if (worldZ >= 0 && worldZ < worldHeightAtMip) {
                    snapshot.blocks[index] = unknownCullingBlock();
                } else {
                    snapshot.blocks[index] = airBlock();
                }
            }
        }
    }

    const glm::ivec3 sectionExtent{
        chunkSizeAtMip,
        chunkSizeAtMip,
        chunkSizeAtMip
    };
    const glm::ivec3 meshletOrigin{
        sectionOriginSample.v.x * static_cast<int32_t>(baseVoxelScale),
        sectionOriginSample.v.y * static_cast<int32_t>(baseVoxelScale),
        sectionOriginSample.v.z * static_cast<int32_t>(baseVoxelScale)
    };
    return mesher.mesh(
        snapshot,
        sectionOriginSample,
        sectionExtent,
        meshletOrigin,
        voxelScale
    );
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
