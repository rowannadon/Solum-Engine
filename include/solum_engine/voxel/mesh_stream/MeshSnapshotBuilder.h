#pragma once

#include <cstdint>
#include <vector>

#include "solum_engine/render/MeshletTypes.h"
#include "solum_engine/voxel/MeshManager.h"

struct MeshSnapshotTile {
    MeshTileCoord tile{};
    uint8_t lod = 0;
    std::vector<Meshlet> meshlets;
};

class MeshSnapshotBuilder {
public:
    static std::vector<Meshlet> build(
        std::vector<MeshSnapshotTile> selectedTiles,
        int32_t meshTileSizeChunks
    );
};
