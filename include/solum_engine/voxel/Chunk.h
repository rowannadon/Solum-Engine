#pragma once

#include <vector>
#include "solum_engine/voxel/BlockMaterial.h"
#include "solum_engine/resources/Constants.h"
#include "solum_engine/resources/Coords.h"
#include <glm/glm.hpp>

class ChunkMesher;

class Chunk {
    friend class ChunkMesher;
    ChunkCoord pos;
    std::array<BlockMaterial, CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE> data;
    
public:
    //Chunk(glm::ivec3 pos);
    bool setBlock(glm::ivec3 pos, UnpackedBlockMaterial mat);
    UnpackedBlockMaterial getBlock(glm::ivec3 pos);

private:
    bool validatePos(glm::ivec3 pos);
    int getIndex(glm::ivec3 pos);
};
