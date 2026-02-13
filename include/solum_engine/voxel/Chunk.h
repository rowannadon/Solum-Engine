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
    BlockStorage blocks;

    friend class ChunkMesher;
    
public:
    Chunk(ChunkCoord p);
    bool setBlock(BlockCoord pos, UnpackedBlockMaterial mat);
    UnpackedBlockMaterial getBlock(BlockCoord pos);
    BlockMaterial* getBlockData();

};
