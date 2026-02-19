#include "solum_engine/voxel/Chunk.h"

bool Chunk::setBlock(BlockCoord pos, UnpackedBlockMaterial mat) {
    bool success = blocks.setBlock(pos, mat);
    // mark blocks dirty
    return success;
}

UnpackedBlockMaterial Chunk::getBlock(BlockCoord pos) {
    return blocks.getBlock(pos);
}

BlockMaterial* Chunk::getBlockData() {
    return blocks.getData();
}