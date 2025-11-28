#include "solum_engine/voxel/ChunkMesher.h"
#include "solum_engine/resources/Constants.h"
#include "solum_engine/voxel/BlockMaterial.h"


std::vector<VertexAttributes> ChunkMesher::mesh(Chunk& chunk, std::vector<Chunk*> neighbors) {
	// generate padded block data using neighbor data
	BlockMaterial paddedBlockData[CHUNK_SIZE_P * CHUNK_SIZE_P * CHUNK_SIZE_P];

	for (int i = 0; i < 6; i++) {
		Chunk *neighbor = neighbors.at(i);

		Direction d = static_cast<Direction>(i);

		// TODO
	}

	// perform meshing
	// iterate through the array and check neighboring for solid (make sure to iterate in best optimized row/column/plane order), create vertex and index data
}