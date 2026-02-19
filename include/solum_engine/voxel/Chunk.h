#pragma once

#include <vector>
#include "solum_engine/voxel/BlockMaterial.h"
#include "solum_engine/voxel/BlockStorage.h"
#include "solum_engine/voxel/ChunkState.h"
#include "solum_engine/resources/Constants.h"
#include "solum_engine/resources/Coords.h"
#include <glm/glm.hpp>

class ChunkMesher;
class Chunk {
private:
    ChunkCoord coord_;
    ChunkState state;
    BlockStorage blocks;

    friend class ChunkMesher;
    
public:
    explicit Chunk(ChunkCoord coord) : coord_(coord) {};
    bool setBlock(BlockCoord pos, UnpackedBlockMaterial mat);
    UnpackedBlockMaterial getBlock(BlockCoord pos);
    BlockMaterial* getBlockData();
    ChunkCoord getCoord() {
        return coord_;
    }
};
