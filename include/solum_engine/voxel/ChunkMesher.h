#include "solum_engine/voxel/Chunk.h"
#include "solum_engine/render/VertexAttributes.h"

#include <iostream>


class ChunkMesher {
	std::vector<VertexAttributes> mesh(Chunk& chunk, std::vector<Chunk*> neighbors);

    std::array<glm::ivec3, 6> directionOffsets = {
        glm::ivec3(1, 0, 0),
        glm::ivec3(-1, 0, 0),
        glm::ivec3(0, 1, 0),
        glm::ivec3(0, -1, 0),
        glm::ivec3(0, 0, 1),
        glm::ivec3(0, 0, -1),
    };

    std::array<glm::ivec3, 6> oppositeDirectionOffsets = {
        glm::ivec3(-1, 0, 0),
        glm::ivec3(1, 0, 0),
        glm::ivec3(0, -1, 0),
        glm::ivec3(0, 1, 0),
        glm::ivec3(0, 0, -1),
        glm::ivec3(0, 0, 1),
    };
};