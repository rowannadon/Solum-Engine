#include "solum_engine/voxel/ChunkMesher.h"
#include "solum_engine/resources/Constants.h"
#include "solum_engine/resources/Coords.h"
#include "solum_engine/voxel/BlockMaterial.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace {
    constexpr int kChunkSize = CHUNK_SIZE;
    constexpr int kChunkSizePadded = CHUNK_SIZE_P;
    constexpr int kChunkPlaneArea = kChunkSize * kChunkSize;
    constexpr int kPaddedPlaneArea = kChunkSizePadded * kChunkSizePadded;
    constexpr int kPaddedBlockCount = kChunkSizePadded * kChunkSizePadded * kChunkSizePadded;

    inline bool IsSolid(const BlockMaterial& block) {
        return ((block.data >> 16u) & 0xFFFFu) != 0u;
    }
}

std::vector<Meshlet> ChunkMesher::mesh(Chunk& chunk, const std::vector<Chunk*>& neighbors) {
    std::array<BlockMaterial, kPaddedBlockCount> paddedBlockData;
    paddedBlockData.fill(BlockMaterial{ 0 });

    BlockCoord chunkOrigin = chunk_to_block_origin(chunk.pos);

    auto paddedIndex = [&](int x, int y, int z) {
        return (x * kPaddedPlaneArea) + (y * kChunkSizePadded) + z;
    };

    const BlockMaterial* chunkData = chunk.getBlockData();
    auto paddedPtr = paddedBlockData.data();

    for (int x = 0; x < kChunkSize; ++x) {
        const BlockMaterial* srcPlane = chunkData + x * kChunkPlaneArea;
        for (int y = 0; y < kChunkSize; ++y) {
            BlockMaterial* dstRow = paddedPtr + paddedIndex(x + 1, y + 1, 1);
            const BlockMaterial* srcRow = srcPlane + y * kChunkSize;
            std::memcpy(dstRow, srcRow, kChunkSize * sizeof(BlockMaterial));
        }
    }

    std::array<Chunk*, 6> neighborChunks{};
    neighborChunks.fill(nullptr);
    const size_t neighborCount = std::min(neighbors.size(), neighborChunks.size());
    for (size_t i = 0; i < neighborCount; ++i) {
        neighborChunks[i] = neighbors[i];
    }

    auto copyXPlane = [&](int destX, int srcX, const BlockMaterial* src) {
        const BlockMaterial* srcPlane = src + srcX * kChunkPlaneArea;
        for (int y = 0; y < kChunkSize; ++y) {
            BlockMaterial* dstRow = paddedPtr + paddedIndex(destX, y + 1, 1);
            const BlockMaterial* srcRow = srcPlane + y * kChunkSize;
            std::memcpy(dstRow, srcRow, kChunkSize * sizeof(BlockMaterial));
        }
    };

    auto copyYPlane = [&](int destY, int srcY, const BlockMaterial* src) {
        for (int x = 0; x < kChunkSize; ++x) {
            const BlockMaterial* srcRow = src + x * kChunkPlaneArea + srcY * kChunkSize;
            BlockMaterial* dstRow = paddedPtr + paddedIndex(x + 1, destY, 1);
            std::memcpy(dstRow, srcRow, kChunkSize * sizeof(BlockMaterial));
        }
    };

    auto copyZPlane = [&](int destZ, int srcZ, const BlockMaterial* src) {
        for (int x = 0; x < kChunkSize; ++x) {
            const BlockMaterial* srcPlane = src + x * kChunkPlaneArea;
            for (int y = 0; y < kChunkSize; ++y) {
                paddedPtr[paddedIndex(x + 1, y + 1, destZ)] = srcPlane[y * kChunkSize + srcZ];
            }
        }
    };

    for (int dir = 0; dir < 6; ++dir) {
        Chunk* neighbor = neighborChunks[dir];
        if (!neighbor) {
            continue;
        }

        const BlockMaterial* neighborData = neighbor->getBlockData();
        switch (static_cast<Direction>(dir)) {
        case Direction::PlusX:
            copyXPlane(kChunkSize + 1, 0, neighborData);
            break;
        case Direction::MinusX:
            copyXPlane(0, kChunkSize - 1, neighborData);
            break;
        case Direction::PlusY:
            copyYPlane(kChunkSize + 1, 0, neighborData);
            break;
        case Direction::MinusY:
            copyYPlane(0, kChunkSize - 1, neighborData);
            break;
        case Direction::PlusZ:
            copyZPlane(kChunkSize + 1, 0, neighborData);
            break;
        case Direction::MinusZ:
            copyZPlane(0, kChunkSize - 1, neighborData);
            break;
        }
    }

    std::array<std::vector<Meshlet>, 6> meshletsByDirection;

    auto appendQuad = [&](uint32_t dir, uint32_t x, uint32_t y, uint32_t z) {
        auto& dirMeshlets = meshletsByDirection[dir];
        if (dirMeshlets.empty() || dirMeshlets.back().quadCount >= MESHLET_QUAD_CAPACITY) {
            Meshlet meshlet{};
            meshlet.origin = chunkOrigin.v;
            meshlet.faceDirection = dir;
            dirMeshlets.push_back(meshlet);
        }

        Meshlet& activeMeshlet = dirMeshlets.back();
        activeMeshlet.packedQuadLocalOffsets[activeMeshlet.quadCount] = packMeshletLocalOffset(x, y, z);
        activeMeshlet.quadCount += 1;
    };

    for (int x = 0; x < kChunkSize; ++x) {
        for (int y = 0; y < kChunkSize; ++y) {
            for (int z = 0; z < kChunkSize; ++z) {
                const BlockMaterial block = chunkData[x * kChunkPlaneArea + y * kChunkSize + z];
                if (!IsSolid(block)) {
                    continue;
                }

                const int paddedX = x + 1;
                const int paddedY = y + 1;
                const int paddedZ = z + 1;

                for (uint32_t dir = 0; dir < 6; ++dir) {
                    const glm::ivec3 offset = directionOffsets[dir];
                    const int neighborX = paddedX + offset.x;
                    const int neighborY = paddedY + offset.y;
                    const int neighborZ = paddedZ + offset.z;
                    const BlockMaterial neighborBlock = paddedBlockData[paddedIndex(neighborX, neighborY, neighborZ)];
                    if (IsSolid(neighborBlock)) {
                        continue;
                    }

                    appendQuad(dir, static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(z));
                }
            }
        }
    }

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
