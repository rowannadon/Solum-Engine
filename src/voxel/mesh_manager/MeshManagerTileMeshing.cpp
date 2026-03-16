#include "solum_engine/voxel/MeshManager.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <unordered_set>

#include "solum_engine/resources/Constants.h"
#include "solum_engine/voxel/World.h"

namespace {
constexpr int kPaddedChunkExtent = cfg::CHUNK_SIZE + 2;
constexpr int kPaddedChunkArea = kPaddedChunkExtent * kPaddedChunkExtent;
constexpr int kPaddedChunkVoxelCount = kPaddedChunkExtent * kPaddedChunkExtent * kPaddedChunkExtent;
constexpr int kMaxLodShift = 30;
constexpr uint8_t kDefaultSeamPackedLight = Chunk::packLight(15u, 0u);

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

bool isMaterialAoOccluder(const BlockModelLibrary* blockModelLibrary, uint16_t materialId) {
    if (materialId == 0u) {
        return false;
    }
    if (blockModelLibrary == nullptr) {
        return true;
    }
    return blockModelLibrary->isMaterialAoOccluder(materialId);
}

uint32_t oppositeDirection(uint32_t dir) {
    switch (dir) {
        case Direction::PlusX: return Direction::MinusX;
        case Direction::MinusX: return Direction::PlusX;
        case Direction::PlusY: return Direction::MinusY;
        case Direction::MinusY: return Direction::PlusY;
        case Direction::PlusZ: return Direction::MinusZ;
        default: return Direction::PlusZ;
    }
}

uint8_t maxPackedLight(uint8_t a, uint8_t b) {
    return Chunk::packLight(
        std::max(Chunk::unpackSkyLight(a), Chunk::unpackSkyLight(b)),
        std::max(Chunk::unpackBlockLight(a), Chunk::unpackBlockLight(b))
    );
}

constexpr std::array<std::array<std::array<glm::ivec3, 3>, 4>, 6> kAoStates = {{
    {{
        {glm::ivec3(1, -1, 0), glm::ivec3(1, 0, -1), glm::ivec3(1, -1, -1)},
        {glm::ivec3(1, 1, 0), glm::ivec3(1, 0, -1), glm::ivec3(1, 1, -1)},
        {glm::ivec3(1, -1, 0), glm::ivec3(1, 0, 1), glm::ivec3(1, -1, 1)},
        {glm::ivec3(1, 1, 0), glm::ivec3(1, 0, 1), glm::ivec3(1, 1, 1)},
    }},
    {{
        {glm::ivec3(-1, -1, 0), glm::ivec3(-1, 0, -1), glm::ivec3(-1, -1, -1)},
        {glm::ivec3(-1, -1, 0), glm::ivec3(-1, 0, 1), glm::ivec3(-1, -1, 1)},
        {glm::ivec3(-1, 1, 0), glm::ivec3(-1, 0, -1), glm::ivec3(-1, 1, -1)},
        {glm::ivec3(-1, 1, 0), glm::ivec3(-1, 0, 1), glm::ivec3(-1, 1, 1)},
    }},
    {{
        {glm::ivec3(-1, 1, 0), glm::ivec3(0, 1, -1), glm::ivec3(-1, 1, -1)},
        {glm::ivec3(-1, 1, 0), glm::ivec3(0, 1, 1), glm::ivec3(-1, 1, 1)},
        {glm::ivec3(1, 1, 0), glm::ivec3(0, 1, -1), glm::ivec3(1, 1, -1)},
        {glm::ivec3(1, 1, 0), glm::ivec3(0, 1, 1), glm::ivec3(1, 1, 1)},
    }},
    {{
        {glm::ivec3(-1, -1, 0), glm::ivec3(0, -1, -1), glm::ivec3(-1, -1, -1)},
        {glm::ivec3(1, -1, 0), glm::ivec3(0, -1, -1), glm::ivec3(1, -1, -1)},
        {glm::ivec3(-1, -1, 0), glm::ivec3(0, -1, 1), glm::ivec3(-1, -1, 1)},
        {glm::ivec3(1, -1, 0), glm::ivec3(0, -1, 1), glm::ivec3(1, -1, 1)},
    }},
    {{
        {glm::ivec3(-1, 0, 1), glm::ivec3(0, -1, 1), glm::ivec3(-1, -1, 1)},
        {glm::ivec3(1, 0, 1), glm::ivec3(0, -1, 1), glm::ivec3(1, -1, 1)},
        {glm::ivec3(-1, 0, 1), glm::ivec3(0, 1, 1), glm::ivec3(-1, 1, 1)},
        {glm::ivec3(1, 0, 1), glm::ivec3(0, 1, 1), glm::ivec3(1, 1, 1)},
    }},
    {{
        {glm::ivec3(-1, 0, -1), glm::ivec3(0, -1, -1), glm::ivec3(-1, -1, -1)},
        {glm::ivec3(-1, 0, -1), glm::ivec3(0, 1, -1), glm::ivec3(-1, 1, -1)},
        {glm::ivec3(1, 0, -1), glm::ivec3(0, -1, -1), glm::ivec3(1, -1, -1)},
        {glm::ivec3(1, 0, -1), glm::ivec3(0, 1, -1), glm::ivec3(1, 1, -1)},
    }},
}};

uint8_t vertexAo(bool side1, bool side2, bool corner) {
    if (side1 && side2) {
        return 0u;
    }
    return static_cast<uint8_t>(3u - static_cast<uint8_t>(side1) - static_cast<uint8_t>(side2) - static_cast<uint8_t>(corner));
}

struct FaceCoordKey {
    glm::ivec3 origin{0};
    uint32_t direction = 0u;

    friend bool operator==(const FaceCoordKey& a, const FaceCoordKey& b) {
        return a.direction == b.direction && a.origin == b.origin;
    }
};

struct FaceCoordKeyHash {
    size_t operator()(const FaceCoordKey& key) const noexcept {
#if SIZE_MAX > UINT32_MAX
        constexpr size_t kGoldenRatio = 0x9e3779b97f4a7c15ull;
#else
        constexpr size_t kGoldenRatio = 0x9e3779b9u;
#endif
        size_t seed = std::hash<int32_t>{}(key.origin.x);
        seed ^= std::hash<int32_t>{}(key.origin.y) + kGoldenRatio + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int32_t>{}(key.origin.z) + kGoldenRatio + (seed << 6) + (seed >> 2);
        seed ^= std::hash<uint32_t>{}(key.direction) + kGoldenRatio + (seed << 6) + (seed >> 2);
        return seed;
    }
};

bool isBoundarySideDirection(uint32_t faceDirection) {
    return faceDirection == Direction::MinusX ||
        faceDirection == Direction::PlusX ||
        faceDirection == Direction::MinusY ||
        faceDirection == Direction::PlusY;
}

void appendSeamQuad(std::vector<Meshlet>& targetMeshlets,
                    uint32_t faceDirection,
                    const glm::ivec3& origin,
                    uint32_t voxelScale,
                    uint16_t materialId,
                    uint16_t packedLight,
                    uint16_t packedAoData) {
    Meshlet seam{};
    seam.origin = origin;
    seam.faceDirection = faceDirection;
    seam.voxelScale = std::max(voxelScale, 1u);
    seam.flags = MESHLET_FLAG_SEAM;
    seam.packedQuadLocalOffsets[0] = packMeshletLocalOffset(0u, 0u, 0u);
    seam.quadMaterialIds[0] = materialId;
    seam.quadAoData[0] = packedAoData;
    seam.quadLightData[0] = packedLight;
    seam.quadModelQuadIndices[0] = faceDirection;
    seam.quadUsesVoxelAo[0] = 1u;
    seam.localBoundsMin = glm::vec3(0.0f);
    seam.localBoundsMax = glm::vec3(1.0f);
    seam.hasCustomBounds = true;
    seam.quadCount = 1u;
    targetMeshlets.push_back(seam);
}

bool trySamplePackedLightAtMip(const World& world,
                               const glm::ivec3& mesherCoord,
                               int32_t sampleStrideMip,
                               uint8_t mipLevel,
                               uint8_t& outPackedLight) {
    const BlockCoord worldMipCoord{
        mesherCoord.x * sampleStrideMip,
        mesherCoord.y * sampleStrideMip,
        mesherCoord.z * sampleStrideMip
    };
    return world.tryGetPackedLight(worldMipCoord, outPackedLight, mipLevel);
}

bool isSolidAtMip(const World& world,
                  const BlockModelLibrary* blockModelLibrary,
                  const glm::ivec3& mesherCoord,
                  int32_t sampleStrideMip,
                  uint8_t mipLevel) {
    BlockMaterial block = airBlock();
    const BlockCoord worldMipCoord{
        mesherCoord.x * sampleStrideMip,
        mesherCoord.y * sampleStrideMip,
        mesherCoord.z * sampleStrideMip
    };
    if (!world.tryGetBlock(worldMipCoord, block, mipLevel)) {
        // Treat unknown blocks as solid (occluding), consistent with how
        // unknownCullingBlock() works in the LOD cell meshing path.  This
        // prevents bright AO seam lines and incorrect full-bright sky light
        // boosts at tile boundaries whose neighbours haven't generated yet.
        return true;
    }
    return isMaterialAoOccluder(blockModelLibrary, block.unpack().id);
}

uint16_t computeSeamPackedAoData(const World& world,
                                 const BlockModelLibrary* blockModelLibrary,
                                 uint32_t faceDirection,
                                 const glm::ivec3& blockCoordMesher,
                                 int32_t sampleStrideMip,
                                 uint8_t mipLevel) {
    std::array<uint8_t, 4> ao{};
    for (uint32_t corner = 0; corner < 4; ++corner) {
        const glm::ivec3 side1Coord = blockCoordMesher + kAoStates[faceDirection][corner][0];
        const glm::ivec3 side2Coord = blockCoordMesher + kAoStates[faceDirection][corner][1];
        const glm::ivec3 cornerCoord = blockCoordMesher + kAoStates[faceDirection][corner][2];

        ao[corner] = vertexAo(
            isSolidAtMip(world, blockModelLibrary, side1Coord, sampleStrideMip, mipLevel),
            isSolidAtMip(world, blockModelLibrary, side2Coord, sampleStrideMip, mipLevel),
            isSolidAtMip(world, blockModelLibrary, cornerCoord, sampleStrideMip, mipLevel)
        );
    }

    const bool flipped = (static_cast<uint32_t>(ao[1]) + static_cast<uint32_t>(ao[2])) >
        (static_cast<uint32_t>(ao[0]) + static_cast<uint32_t>(ao[3]));
    return packMeshletQuadAoData(ao[0], ao[1], ao[2], ao[3], flipped);
}

void collectBoundarySideFaces(const std::vector<Meshlet>& meshlets,
                              std::unordered_set<FaceCoordKey, FaceCoordKeyHash>& outFaces) {
    for (const Meshlet& meshlet : meshlets) {
        if (!isBoundarySideDirection(meshlet.faceDirection) || meshlet.quadCount == 0u) {
            continue;
        }

        const uint32_t voxelScale = std::max(meshlet.voxelScale, 1u);
        for (uint32_t quadIndex = 0; quadIndex < meshlet.quadCount; ++quadIndex) {
            if (meshlet.quadUsesVoxelAo[quadIndex] == 0u) {
                continue;
            }

            const glm::uvec3 local = unpackMeshletLocalOffset(meshlet.packedQuadLocalOffsets[quadIndex]);
            outFaces.insert(FaceCoordKey{
                glm::ivec3(
                    meshlet.origin.x + static_cast<int32_t>(local.x * voxelScale),
                    meshlet.origin.y + static_cast<int32_t>(local.y * voxelScale),
                    meshlet.origin.z + static_cast<int32_t>(local.z * voxelScale)
                ),
                meshlet.faceDirection
            });
        }
    }
}

void appendTileSeamStrips(ChunkMeshOutput& meshOutput,
                          const World& world,
                          const MeshTileCoord& tile,
                          int32_t meshTileSizeChunks,
                          uint8_t lodLevel,
                          const BlockModelLibrary* blockModelLibrary) {
    if (lodLevel == 0u ||
        (meshOutput.culledMeshlets.empty() && meshOutput.doubleSidedMeshlets.empty())) {
        return;
    }

    const uint8_t mipLevel = std::min<uint8_t>(lodLevel, Chunk::MAX_MIP_LEVEL);
    const int32_t extraLodShift = std::max(
        0,
        static_cast<int32_t>(lodLevel) - static_cast<int32_t>(Chunk::MAX_MIP_LEVEL)
    );
    const int32_t sampleStrideMip = pow2ClampedShift(extraLodShift);
    const int32_t tileMinX = tile.x * meshTileSizeChunks * cfg::CHUNK_SIZE;
    const int32_t tileMinY = tile.y * meshTileSizeChunks * cfg::CHUNK_SIZE;
    const int32_t tileMaxX = tileMinX + meshTileSizeChunks * cfg::CHUNK_SIZE;
    const int32_t tileMaxY = tileMinY + meshTileSizeChunks * cfg::CHUNK_SIZE;

    std::unordered_set<FaceCoordKey, FaceCoordKeyHash> existingBoundaryFaces;
    existingBoundaryFaces.reserve(meshOutput.culledMeshlets.size() + meshOutput.doubleSidedMeshlets.size());
    collectBoundarySideFaces(meshOutput.culledMeshlets, existingBoundaryFaces);
    collectBoundarySideFaces(meshOutput.doubleSidedMeshlets, existingBoundaryFaces);

    std::vector<Meshlet> culledSeamMeshlets;
    std::vector<Meshlet> doubleSidedSeamMeshlets;

    auto processMeshlets = [&](const std::vector<Meshlet>& meshlets) {
        for (const Meshlet& meshlet : meshlets) {
            if (meshlet.faceDirection != Direction::PlusZ || meshlet.quadCount == 0u) {
                continue;
            }

            const uint32_t voxelScale = std::max(meshlet.voxelScale, 1u);
            const int32_t voxelScaleI = static_cast<int32_t>(voxelScale);
            for (uint32_t quadIndex = 0; quadIndex < meshlet.quadCount; ++quadIndex) {
                if (meshlet.quadUsesVoxelAo[quadIndex] == 0u) {
                    continue;
                }

                const glm::uvec3 local = unpackMeshletLocalOffset(meshlet.packedQuadLocalOffsets[quadIndex]);
                const uint16_t materialId = meshlet.quadMaterialIds[quadIndex];
                const uint16_t fallbackLight = meshlet.quadLightData[quadIndex];
                const int32_t worldX = meshlet.origin.x + static_cast<int32_t>(local.x * voxelScale);
                const int32_t worldY = meshlet.origin.y + static_cast<int32_t>(local.y * voxelScale);
                const int32_t worldZ = meshlet.origin.z + static_cast<int32_t>(local.z * voxelScale);
                const glm::ivec3 seamOrigin{worldX, worldY, worldZ};
                const glm::ivec3 blockCoordMesher{
                    floor_div(worldX, voxelScaleI),
                    floor_div(worldY, voxelScaleI),
                    floor_div(worldZ, voxelScaleI)
                };

                auto appendForDirection = [&](uint32_t faceDirection) {
                    const FaceCoordKey key{seamOrigin, faceDirection};
                    if (existingBoundaryFaces.contains(key)) {
                        return;
                    }

                    const glm::ivec3 frontCoord = blockCoordMesher + ChunkMesher::directionOffsets[faceDirection];
                    const glm::ivec3 backCoord = blockCoordMesher + ChunkMesher::directionOffsets[oppositeDirection(faceDirection)];
                    const uint8_t fallbackFront = static_cast<uint8_t>(fallbackLight & 0xFFu);
                    const uint8_t fallbackBack = static_cast<uint8_t>((fallbackLight >> 8u) & 0xFFu);
                    uint8_t frontPackedLight = fallbackFront;
                    uint8_t backPackedLight = fallbackBack;
                    const bool frontLightKnown = trySamplePackedLightAtMip(
                        world,
                        frontCoord,
                        sampleStrideMip,
                        mipLevel,
                        frontPackedLight
                    );
                    const bool backLightKnown = trySamplePackedLightAtMip(
                        world,
                        backCoord,
                        sampleStrideMip,
                        mipLevel,
                        backPackedLight
                    );
                    if (frontLightKnown) {
                        frontPackedLight = maxPackedLight(frontPackedLight, fallbackFront);
                    }
                    if (backLightKnown) {
                        backPackedLight = maxPackedLight(backPackedLight, fallbackBack);
                    }

                    // Only apply full-bright sky light boost when the front
                    // block is *known* to be air.  isSolidAtMip now returns
                    // true for unknown blocks, so this naturally guards
                    // against boosting seams whose neighbours haven't loaded.
                    if (!isSolidAtMip(world, blockModelLibrary, frontCoord, sampleStrideMip, mipLevel)) {
                        frontPackedLight = maxPackedLight(frontPackedLight, kDefaultSeamPackedLight);
                    }
                    const uint16_t packedLight = packMeshletQuadLightPair(frontPackedLight, backPackedLight);
                    const uint16_t packedAoData = computeSeamPackedAoData(
                        world,
                        blockModelLibrary,
                        faceDirection,
                        blockCoordMesher,
                        sampleStrideMip,
                        mipLevel
                    );

                    std::vector<Meshlet>& targetMeshlets =
                        (blockModelLibrary != nullptr && blockModelLibrary->isMaterialDoubleSided(materialId))
                        ? doubleSidedSeamMeshlets
                        : culledSeamMeshlets;
                    appendSeamQuad(
                        targetMeshlets,
                        faceDirection,
                        seamOrigin,
                        voxelScale,
                        materialId,
                        packedLight,
                        packedAoData
                    );
                    existingBoundaryFaces.insert(key);
                };

                if (worldX == tileMinX) {
                    appendForDirection(Direction::MinusX);
                }
                if ((worldX + voxelScaleI) == tileMaxX) {
                    appendForDirection(Direction::PlusX);
                }
                if (worldY == tileMinY) {
                    appendForDirection(Direction::MinusY);
                }
                if ((worldY + voxelScaleI) == tileMaxY) {
                    appendForDirection(Direction::PlusY);
                }
            }
        }
    };

    processMeshlets(meshOutput.culledMeshlets);
    processMeshlets(meshOutput.doubleSidedMeshlets);

    meshOutput.culledMeshlets.insert(
        meshOutput.culledMeshlets.end(),
        std::make_move_iterator(culledSeamMeshlets.begin()),
        std::make_move_iterator(culledSeamMeshlets.end())
    );
    meshOutput.doubleSidedMeshlets.insert(
        meshOutput.doubleSidedMeshlets.end(),
        std::make_move_iterator(doubleSidedSeamMeshlets.begin()),
        std::make_move_iterator(doubleSidedSeamMeshlets.end())
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

    appendTileSeamStrips(meshOutput, world_, coord.tile.tile, meshTileSizeChunks_, lodLevel, blockModelLibrary_.get());
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
