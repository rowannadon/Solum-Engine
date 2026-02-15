#pragma once

#include "solum_engine/voxel/Chunk.h"
#include "solum_engine/render/MeshletTypes.h"

#include <array>
#include <vector>

class ChunkMesher {
public:
	std::vector<Meshlet> mesh(Chunk& chunk, const std::vector<Chunk*>& neighbors);

	static constexpr std::array<glm::ivec3, 6> directionOffsets = {
		glm::ivec3(1, 0, 0),
		glm::ivec3(-1, 0, 0),
		glm::ivec3(0, 1, 0),
		glm::ivec3(0, -1, 0),
		glm::ivec3(0, 0, 1),
		glm::ivec3(0, 0, -1),
	};

};
