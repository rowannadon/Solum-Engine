#pragma once

#include "solum_engine/voxel/Chunk.h"
#include "solum_engine/resources/Coords.h"
#include "solum_engine/render/MeshletTypes.h"
#include <glm/glm.hpp>

#include <array>
#include <vector>

class IBlockSource {
public:
    virtual ~IBlockSource() = default;
    virtual BlockMaterial getBlock(const BlockCoord& coord) const = 0;
};

class ChunkMesher {
public:
    std::vector<Meshlet> mesh(const Chunk& chunk, const ChunkCoord& coord, const std::vector<const Chunk*>& neighbors) const;
    std::vector<Meshlet> mesh(const IBlockSource& source,
                              const BlockCoord& sectionOrigin,
                              const glm::ivec3& sectionExtent,
                              const glm::ivec3& meshletOrigin) const;

    static constexpr std::array<glm::ivec3, 6> directionOffsets = {
        glm::ivec3(1, 0, 0),   // PlusX
        glm::ivec3(-1, 0, 0),  // MinusX
        glm::ivec3(0, 1, 0),   // PlusY
        glm::ivec3(0, -1, 0),  // MinusY
        glm::ivec3(0, 0, 1),   // PlusZ (up in z-up world)
        glm::ivec3(0, 0, -1),  // MinusZ (down in z-up world)
    };
};
