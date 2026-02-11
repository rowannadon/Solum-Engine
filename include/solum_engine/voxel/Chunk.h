#pragma once

#include <vector>
#include "solum_engine/voxel/BlockMaterial.h"
#include "solum_engine/voxel/BlockStorage.h"
#include "solum_engine/voxel/LightStorage.h"
#include "solum_engine/resources/Constants.h"
#include "solum_engine/resources/Coords.h"
#include <glm/glm.hpp>

class ChunkMesher;

class Chunk {
    ChunkCoord pos;
    std::array<BlockMaterial, CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE> data;

    friend class ChunkMesher;
    
public:
    Chunk(ChunkCoord p);
    bool setBlock(BlockCoord pos, UnpackedBlockMaterial mat);
    UnpackedBlockMaterial getBlock(BlockCoord pos);

private:
    bool validatePos(BlockCoord pos);
    int getIndex(BlockCoord pos);
};
