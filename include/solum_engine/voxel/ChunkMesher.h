#pragma once

#include "solum_engine/voxel/Chunk.h"
#include "solum_engine/resources/Coords.h"
#include "solum_engine/render/MeshletTypes.h"
#include <glm/glm.hpp>

#include <array>
#include <vector>

class ChunkMesher {
public:
    std::vector<Meshlet> mesh(const Chunk& chunk, const ChunkCoord& coord, const std::vector<const Chunk*>& neighbors);

    static constexpr std::array<glm::ivec3, 6> directionOffsets = {
        glm::ivec3(1, 0, 0),   // PlusX
        glm::ivec3(-1, 0, 0),  // MinusX
        glm::ivec3(0, 1, 0),   // PlusY
        glm::ivec3(0, -1, 0),  // MinusY
        glm::ivec3(0, 0, 1),   // PlusZ
        glm::ivec3(0, 0, -1),  // MinusZ
    };
};