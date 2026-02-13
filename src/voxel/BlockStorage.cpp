#include "solum_engine/voxel/BlockStorage.h"


bool BlockStorage::setBlock(BlockCoord pos, UnpackedBlockMaterial mat) {
    if (!validatePos(pos)) {
        return false;
    }

    int index = getIndex(pos);
    data[index] = mat.pack();
    return true;
}

UnpackedBlockMaterial BlockStorage::getBlock(BlockCoord pos) {
    if (!validatePos(pos)) {
        UnpackedBlockMaterial m;
        m.id = 0;
        return m;
    }

    int index = getIndex(pos);
    BlockMaterial mat = data[index];
    return mat.unpack();
}

bool BlockStorage::validatePos(BlockCoord pos) {
    if (pos.x() < 0 || pos.y() < 0 || pos.z() < 0) {
        return false;
    }
    if (pos.x() > CHUNK_SIZE - 1 || pos.y() > CHUNK_SIZE - 1 || pos.z() > CHUNK_SIZE - 1) {
        return false;
    }
    return true;
}

int BlockStorage::getIndex(BlockCoord pos) {
    return (pos.x() * CHUNK_SIZE * CHUNK_SIZE) + (pos.y() * CHUNK_SIZE) + pos.z();
}