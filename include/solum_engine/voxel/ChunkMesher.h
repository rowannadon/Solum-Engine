#pragma once

#include "solum_engine/voxel/Chunk.h"
#include "solum_engine/render/VertexAttributes.h"
#include "solum_engine/voxel/ChunkMeshes.h"

#include <array>
#include <vector>

class ChunkMesher {
public:
	MeshData mesh(Chunk& chunk, const std::vector<Chunk*>& neighbors);
	const std::vector<uint32_t>& getIndices() const { return indices; }

private:
	std::vector<uint32_t> indices;
	std::vector<VertexAttributes> vertices;

	static constexpr std::array<glm::ivec3, 6> directionOffsets = {
		glm::ivec3(1, 0, 0),
		glm::ivec3(-1, 0, 0),
		glm::ivec3(0, 1, 0),
		glm::ivec3(0, -1, 0),
		glm::ivec3(0, 0, 1),
		glm::ivec3(0, 0, -1),
	};

	static constexpr std::array<glm::ivec3, 6> oppositeDirectionOffsets = {
		glm::ivec3(-1, 0, 0),
		glm::ivec3(1, 0, 0),
		glm::ivec3(0, -1, 0),
		glm::ivec3(0, 1, 0),
		glm::ivec3(0, 0, -1),
		glm::ivec3(0, 0, 1),
	};
};
