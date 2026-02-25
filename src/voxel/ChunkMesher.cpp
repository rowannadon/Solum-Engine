#include "solum_engine/voxel/ChunkMesher.h"
#include "solum_engine/resources/Constants.h"
#include "solum_engine/resources/Coords.h"

#include <algorithm>

namespace {
    constexpr uint16_t kAirBlockId = 0u;

    constexpr int kChunkSize = Chunk::SIZE;
    constexpr int kChunkSizePadded = kChunkSize + 2;
    constexpr int kPaddedPlaneArea = kChunkSizePadded * kChunkSizePadded;
    constexpr int kPaddedBlockCount = kChunkSizePadded * kChunkSizePadded * kChunkSizePadded;

    inline bool IsSolidForCulling(BlockMaterial blockID) {
        return blockID.unpack().id != kAirBlockId;
    }

    constexpr std::array<std::array<std::array<glm::ivec3, 3>, 4>, 6> kAoStates = {{
        // PlusX
        {{
            {glm::ivec3(1, -1, 0), glm::ivec3(1, 0, -1), glm::ivec3(1, -1, -1)},
            {glm::ivec3(1, 1, 0), glm::ivec3(1, 0, -1), glm::ivec3(1, 1, -1)},
            {glm::ivec3(1, -1, 0), glm::ivec3(1, 0, 1), glm::ivec3(1, -1, 1)},
            {glm::ivec3(1, 1, 0), glm::ivec3(1, 0, 1), glm::ivec3(1, 1, 1)},
        }},
        // MinusX
        {{
            {glm::ivec3(-1, -1, 0), glm::ivec3(-1, 0, -1), glm::ivec3(-1, -1, -1)},
            {glm::ivec3(-1, -1, 0), glm::ivec3(-1, 0, 1), glm::ivec3(-1, -1, 1)},
            {glm::ivec3(-1, 1, 0), glm::ivec3(-1, 0, -1), glm::ivec3(-1, 1, -1)},
            {glm::ivec3(-1, 1, 0), glm::ivec3(-1, 0, 1), glm::ivec3(-1, 1, 1)},
        }},
        // PlusY
        {{
            {glm::ivec3(-1, 1, 0), glm::ivec3(0, 1, -1), glm::ivec3(-1, 1, -1)},
            {glm::ivec3(-1, 1, 0), glm::ivec3(0, 1, 1), glm::ivec3(-1, 1, 1)},
            {glm::ivec3(1, 1, 0), glm::ivec3(0, 1, -1), glm::ivec3(1, 1, -1)},
            {glm::ivec3(1, 1, 0), glm::ivec3(0, 1, 1), glm::ivec3(1, 1, 1)},
        }},
        // MinusY
        {{
            {glm::ivec3(-1, -1, 0), glm::ivec3(0, -1, -1), glm::ivec3(-1, -1, -1)},
            {glm::ivec3(1, -1, 0), glm::ivec3(0, -1, -1), glm::ivec3(1, -1, -1)},
            {glm::ivec3(-1, -1, 0), glm::ivec3(0, -1, 1), glm::ivec3(-1, -1, 1)},
            {glm::ivec3(1, -1, 0), glm::ivec3(0, -1, 1), glm::ivec3(1, -1, 1)},
        }},
        // PlusZ
        {{
            {glm::ivec3(-1, 0, 1), glm::ivec3(0, -1, 1), glm::ivec3(-1, -1, 1)},
            {glm::ivec3(1, 0, 1), glm::ivec3(0, -1, 1), glm::ivec3(1, -1, 1)},
            {glm::ivec3(-1, 0, 1), glm::ivec3(0, 1, 1), glm::ivec3(-1, 1, 1)},
            {glm::ivec3(1, 0, 1), glm::ivec3(0, 1, 1), glm::ivec3(1, 1, 1)},
        }},
        // MinusZ
        {{
            {glm::ivec3(-1, 0, -1), glm::ivec3(0, -1, -1), glm::ivec3(-1, -1, -1)},
            {glm::ivec3(-1, 0, -1), glm::ivec3(0, 1, -1), glm::ivec3(-1, 1, -1)},
            {glm::ivec3(1, 0, -1), glm::ivec3(0, -1, -1), glm::ivec3(1, -1, -1)},
            {glm::ivec3(1, 0, -1), glm::ivec3(0, 1, -1), glm::ivec3(1, 1, -1)},
        }},
    }};

    uint8_t vertexAO(bool side1, bool side2, bool corner) {
        if (side1 && side2) {
            return 0u;
        }
        return static_cast<uint8_t>(3u - static_cast<uint8_t>(side1) - static_cast<uint8_t>(side2) - static_cast<uint8_t>(corner));
    }

    template <typename SolidSampler>
    uint16_t computePackedQuadAoData(uint32_t dir, const glm::ivec3& blockCoord, const SolidSampler& isSolid) {
        std::array<uint8_t, 4> ao{};
        for (uint32_t corner = 0; corner < 4; ++corner) {
            const glm::ivec3 side1Coord = blockCoord + kAoStates[dir][corner][0];
            const glm::ivec3 side2Coord = blockCoord + kAoStates[dir][corner][1];
            const glm::ivec3 cornerCoord = blockCoord + kAoStates[dir][corner][2];

            ao[corner] = vertexAO(
                isSolid(side1Coord),
                isSolid(side2Coord),
                isSolid(cornerCoord)
            );
        }

        // Mesh uses diagonal 1-2 when unflipped and 0-3 when flipped.
        const bool flipped = (static_cast<uint32_t>(ao[1]) + static_cast<uint32_t>(ao[2])) >
                             (static_cast<uint32_t>(ao[0]) + static_cast<uint32_t>(ao[3]));
        return packMeshletQuadAoData(ao[0], ao[1], ao[2], ao[3], flipped);
    }

    std::vector<Meshlet> flattenMeshlets(const std::array<std::vector<Meshlet>, 6>& meshletsByDirection) {
        size_t totalMeshletCount = 0;
        for (const auto& dirMeshlets : meshletsByDirection) {
            totalMeshletCount += dirMeshlets.size();
        }

        std::vector<Meshlet> meshlets;
        meshlets.reserve(totalMeshletCount);

        for (const auto& dirMeshlets : meshletsByDirection) {
            meshlets.insert(meshlets.end(), dirMeshlets.begin(), dirMeshlets.end());
        }

        return meshlets;
    }
}

std::vector<Meshlet> ChunkMesher::mesh(const Chunk& chunk,
                                       const ChunkCoord& coord,
                                       const std::vector<const Chunk*>& neighbors) const {
    // We use a flat array of uint32_t to store the unpacked IDs for cache-friendly access
    std::array<BlockMaterial, kPaddedBlockCount> paddedBlockData;
    UnpackedBlockMaterial air{0, 0, Direction::PlusX, 0};
    paddedBlockData.fill(air.pack()); // Fill with air by default

    // Helper to get 1D index for the 3D padded array
    auto paddedIndex = [&](int x, int y, int z) {
        return (x * kPaddedPlaneArea) + (y * kChunkSizePadded) + z;
    };

    // 1. Unpack the central chunk into the padded array
    for (int x = 0; x < kChunkSize; ++x) {
        for (int y = 0; y < kChunkSize; ++y) {
            for (int z = 0; z < kChunkSize; ++z) {
                paddedBlockData[paddedIndex(x + 1, y + 1, z + 1)] = chunk.getBlock(x, y, z);
            }
        }
    }

    // 2. Unpack the neighbor boundaries into the padded array
    // Directions match the directionOffsets array: +X, -X, +Y, -Y, +Z, -Z
    for (size_t dir = 0; dir < std::min(neighbors.size(), static_cast<size_t>(6)); ++dir) {
        const Chunk* neighbor = neighbors[dir];
        if (!neighbor) continue;

        for (int i = 0; i < kChunkSize; ++i) {
            for (int j = 0; j < kChunkSize; ++j) {
                switch (dir) {
                    case 0: // PlusX: Neighbor's x=0 maps to padded x=17
                        paddedBlockData[paddedIndex(kChunkSize + 1, i + 1, j + 1)] = neighbor->getBlock(0, i, j);
                        break;
                    case 1: // MinusX: Neighbor's x=15 maps to padded x=0
                        paddedBlockData[paddedIndex(0, i + 1, j + 1)] = neighbor->getBlock(kChunkSize - 1, i, j);
                        break;
                    case 2: // PlusY: Neighbor's y=0 maps to padded y=17
                        paddedBlockData[paddedIndex(i + 1, kChunkSize + 1, j + 1)] = neighbor->getBlock(i, 0, j);
                        break;
                    case 3: // MinusY: Neighbor's y=15 maps to padded y=0
                        paddedBlockData[paddedIndex(i + 1, 0, j + 1)] = neighbor->getBlock(i, kChunkSize - 1, j);
                        break;
                    case 4: // PlusZ: Neighbor's z=0 maps to padded z=17
                        paddedBlockData[paddedIndex(i + 1, j + 1, kChunkSize + 1)] = neighbor->getBlock(i, j, 0);
                        break;
                    case 5: // MinusZ: Neighbor's z=15 maps to padded z=0
                        paddedBlockData[paddedIndex(i + 1, j + 1, 0)] = neighbor->getBlock(i, j, kChunkSize - 1);
                        break;
                }
            }
        }
    }

    // 3. Generate Meshlets
    BlockCoord chunkOrigin = chunk_to_block_origin(coord);
    std::array<std::vector<Meshlet>, 6> meshletsByDirection;

    auto appendQuad = [&](uint32_t dir,
                          uint32_t x,
                          uint32_t y,
                          uint32_t z,
                          uint16_t materialId,
                          uint16_t packedAoData) {
        auto& dirMeshlets = meshletsByDirection[dir];
        if (dirMeshlets.empty() || dirMeshlets.back().quadCount >= MESHLET_QUAD_CAPACITY) {
            Meshlet meshlet{};
            meshlet.origin = chunkOrigin.v;
            meshlet.faceDirection = dir;
            dirMeshlets.push_back(meshlet);
        }

        Meshlet& activeMeshlet = dirMeshlets.back();
        activeMeshlet.packedQuadLocalOffsets[activeMeshlet.quadCount] = packMeshletLocalOffset(x, y, z);
        activeMeshlet.quadMaterialIds[activeMeshlet.quadCount] = materialId;
        activeMeshlet.quadAoData[activeMeshlet.quadCount] = packedAoData;
        activeMeshlet.quadCount += 1;
    };

    auto isSolidAtPadded = [&paddedBlockData, &paddedIndex](const glm::ivec3& coord) {
        return IsSolidForCulling(paddedBlockData[paddedIndex(coord.x, coord.y, coord.z)]);
    };

    // Iterate through the actual chunk boundaries inside the padded array
    for (int x = 0; x < kChunkSize; ++x) {
        for (int y = 0; y < kChunkSize; ++y) {
            for (int z = 0; z < kChunkSize; ++z) {
                const int paddedX = x + 1;
                const int paddedY = y + 1;
                const int paddedZ = z + 1;

                const BlockMaterial blockID = paddedBlockData[paddedIndex(paddedX, paddedY, paddedZ)];
                const uint16_t materialId = blockID.unpack().id;
                if (materialId == kAirBlockId || materialId == ChunkMesher::kCulledSolidBlockId) {
                    continue;
                }

                // Check all 6 faces for visibility
                for (uint32_t dir = 0; dir < 6; ++dir) {
                    const glm::ivec3& offset = directionOffsets[dir];
                    const int neighborX = paddedX + offset.x;
                    const int neighborY = paddedY + offset.y;
                    const int neighborZ = paddedZ + offset.z;
                    
                    BlockMaterial neighborBlockID = paddedBlockData[paddedIndex(neighborX, neighborY, neighborZ)];
                    
                    if (IsSolidForCulling(neighborBlockID)) {
                        continue; // Face is occluded
                    }

                    const uint16_t packedAoData = computePackedQuadAoData(
                        dir,
                        glm::ivec3{paddedX, paddedY, paddedZ},
                        isSolidAtPadded
                    );

                    appendQuad(
                        dir,
                        static_cast<uint32_t>(x),
                        static_cast<uint32_t>(y),
                        static_cast<uint32_t>(z),
                        materialId,
                        packedAoData
                    );
                }
            }
        }
    }

    return flattenMeshlets(meshletsByDirection);
}

std::vector<Meshlet> ChunkMesher::mesh(const IBlockSource& source,
                                       const BlockCoord& sectionOrigin,
                                       const glm::ivec3& sectionExtent,
                                       const glm::ivec3& meshletOrigin,
                                       uint32_t voxelScale) const {
    if (sectionExtent.x <= 0 || sectionExtent.y <= 0 || sectionExtent.z <= 0) {
        return {};
    }

    if (sectionExtent.x > 32 || sectionExtent.y > 32 || sectionExtent.z > 32) {
        return {};
    }

    std::array<std::vector<Meshlet>, 6> meshletsByDirection;

    auto appendQuad = [&](uint32_t dir,
                          uint32_t x,
                          uint32_t y,
                          uint32_t z,
                          uint16_t materialId,
                          uint16_t packedAoData) {
        auto& dirMeshlets = meshletsByDirection[dir];
        if (dirMeshlets.empty() || dirMeshlets.back().quadCount >= MESHLET_QUAD_CAPACITY) {
            Meshlet meshlet{};
            meshlet.origin = meshletOrigin;
            meshlet.faceDirection = dir;
            meshlet.voxelScale = std::max(voxelScale, 1u);
            dirMeshlets.push_back(meshlet);
        }

        Meshlet& activeMeshlet = dirMeshlets.back();
        activeMeshlet.packedQuadLocalOffsets[activeMeshlet.quadCount] = packMeshletLocalOffset(x, y, z);
        activeMeshlet.quadMaterialIds[activeMeshlet.quadCount] = materialId;
        activeMeshlet.quadAoData[activeMeshlet.quadCount] = packedAoData;
        activeMeshlet.quadCount += 1;
    };

    auto isSolidAtCoord = [&source](const glm::ivec3& coord) {
        return IsSolidForCulling(source.getBlock(BlockCoord{coord.x, coord.y, coord.z}));
    };

    for (int x = 0; x < sectionExtent.x; ++x) {
        for (int y = 0; y < sectionExtent.y; ++y) {
            for (int z = 0; z < sectionExtent.z; ++z) {
                const BlockCoord blockCoord{
                    sectionOrigin.v.x + x,
                    sectionOrigin.v.y + y,
                    sectionOrigin.v.z + z
                };

                const BlockMaterial blockID = source.getBlock(blockCoord);
                const uint16_t materialId = blockID.unpack().id;
                if (materialId == kAirBlockId || materialId == ChunkMesher::kCulledSolidBlockId) {
                    continue;
                }

                for (uint32_t dir = 0; dir < 6; ++dir) {
                    const glm::ivec3& offset = directionOffsets[dir];
                    const BlockCoord neighborCoord{
                        blockCoord.v.x + offset.x,
                        blockCoord.v.y + offset.y,
                        blockCoord.v.z + offset.z
                    };

                    const BlockMaterial neighborBlockID = source.getBlock(neighborCoord);
                    if (IsSolidForCulling(neighborBlockID)) {
                        continue;
                    }

                    const uint16_t packedAoData = computePackedQuadAoData(
                        dir,
                        blockCoord.v,
                        isSolidAtCoord
                    );

                    appendQuad(
                        dir,
                        static_cast<uint32_t>(x),
                        static_cast<uint32_t>(y),
                        static_cast<uint32_t>(z),
                        materialId,
                        packedAoData
                    );
                }
            }
        }
    }

    return flattenMeshlets(meshletsByDirection);
}
