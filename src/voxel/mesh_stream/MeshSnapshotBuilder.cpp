#include "solum_engine/voxel/mesh_stream/MeshSnapshotBuilder.h"

#include <algorithm>
#include <iterator>
#include <unordered_map>

#include "solum_engine/resources/Constants.h"

namespace {
void appendSkirtQuad(std::vector<Meshlet>& skirtMeshlets,
                     uint32_t faceDirection,
                     const glm::ivec3& origin,
                     uint32_t voxelScale,
                     uint16_t materialId) {
    Meshlet skirt{};
    skirt.origin = origin;
    skirt.faceDirection = faceDirection;
    skirt.voxelScale = std::max(voxelScale, 1u);
    skirt.packedQuadLocalOffsets[0] = packMeshletLocalOffset(0u, 0u, 0u);
    skirt.quadMaterialIds[0] = materialId;
    skirt.quadAoData[0] = packMeshletQuadAoData(3u, 3u, 3u, 3u, false);
    skirt.quadCount = 1u;
    skirtMeshlets.push_back(skirt);
}
}  // namespace

std::vector<Meshlet> MeshSnapshotBuilder::build(std::vector<MeshSnapshotTile> selectedTiles,
                                                int32_t meshTileSizeChunks) {
    std::sort(selectedTiles.begin(), selectedTiles.end(), [](const MeshSnapshotTile& a, const MeshSnapshotTile& b) {
        if (a.tile == b.tile) {
            return a.lod < b.lod;
        }
        return a.tile < b.tile;
    });

    size_t totalMeshletCount = 0u;
    for (const MeshSnapshotTile& tile : selectedTiles) {
        totalMeshletCount += tile.meshlets.size();
    }

    std::unordered_map<MeshTileCoord, uint8_t> selectedLodByTile;
    selectedLodByTile.reserve(selectedTiles.size());
    for (const MeshSnapshotTile& tile : selectedTiles) {
        selectedLodByTile[tile.tile] = tile.lod;
    }

    std::vector<Meshlet> skirtMeshlets;
    for (const MeshSnapshotTile& entry : selectedTiles) {
        if (entry.lod == 0u) {
            continue;
        }

        const auto isFinerNeighbor = [&selectedLodByTile, &entry](int32_t dx, int32_t dy) {
            const auto neighborIt = selectedLodByTile.find(MeshTileCoord{entry.tile.x + dx, entry.tile.y + dy});
            return neighborIt != selectedLodByTile.end() && neighborIt->second < entry.lod;
        };

        const bool skirtPlusX = isFinerNeighbor(+1, 0);
        const bool skirtMinusX = isFinerNeighbor(-1, 0);
        const bool skirtPlusY = isFinerNeighbor(0, +1);
        const bool skirtMinusY = isFinerNeighbor(0, -1);
        if (!skirtPlusX && !skirtMinusX && !skirtPlusY && !skirtMinusY) {
            continue;
        }

        const int32_t tileMinX = entry.tile.x * meshTileSizeChunks * cfg::CHUNK_SIZE;
        const int32_t tileMinY = entry.tile.y * meshTileSizeChunks * cfg::CHUNK_SIZE;
        const int32_t tileMaxX = tileMinX + meshTileSizeChunks * cfg::CHUNK_SIZE;
        const int32_t tileMaxY = tileMinY + meshTileSizeChunks * cfg::CHUNK_SIZE;

        for (const Meshlet& meshlet : entry.meshlets) {
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

                if (skirtMinusX && worldX == tileMinX) {
                    appendSkirtQuad(
                        skirtMeshlets,
                        Direction::MinusX,
                        glm::ivec3(worldX, worldY, worldZ),
                        voxelScale,
                        materialId
                    );
                }
                if (skirtPlusX && (worldX + static_cast<int32_t>(voxelScale)) == tileMaxX) {
                    appendSkirtQuad(
                        skirtMeshlets,
                        Direction::PlusX,
                        glm::ivec3(worldX, worldY, worldZ),
                        voxelScale,
                        materialId
                    );
                }
                if (skirtMinusY && worldY == tileMinY) {
                    appendSkirtQuad(
                        skirtMeshlets,
                        Direction::MinusY,
                        glm::ivec3(worldX, worldY, worldZ),
                        voxelScale,
                        materialId
                    );
                }
                if (skirtPlusY && (worldY + static_cast<int32_t>(voxelScale)) == tileMaxY) {
                    appendSkirtQuad(
                        skirtMeshlets,
                        Direction::PlusY,
                        glm::ivec3(worldX, worldY, worldZ),
                        voxelScale,
                        materialId
                    );
                }
            }
        }
    }

    totalMeshletCount += skirtMeshlets.size();
    std::vector<Meshlet> meshlets;
    meshlets.reserve(totalMeshletCount);

    for (MeshSnapshotTile& entry : selectedTiles) {
        meshlets.insert(
            meshlets.end(),
            std::make_move_iterator(entry.meshlets.begin()),
            std::make_move_iterator(entry.meshlets.end())
        );
    }
    meshlets.insert(
        meshlets.end(),
        std::make_move_iterator(skirtMeshlets.begin()),
        std::make_move_iterator(skirtMeshlets.end())
    );
    return meshlets;
}
