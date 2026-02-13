#include "solum_engine/voxel/BlockStorage.h"

bool BlockStorage::setBlock(BlockCoord pos, UnpackedBlockMaterial mat) {
    if (!validatePos(pos)) {
        return false;
    }

    const std::size_t index = getIndex(pos);
    data_[index] = mat.pack();
    return true;
}

UnpackedBlockMaterial BlockStorage::getBlock(BlockCoord pos) const {
    if (!validatePos(pos)) {
        return {};
    }

    const std::size_t index = getIndex(pos);
    return data_[index].unpack();
}

bool BlockStorage::validatePos(BlockCoord pos) const {
    return pos.x() >= 0 && pos.y() >= 0 && pos.z() >= 0 &&
        pos.x() < CHUNK_SIZE && pos.y() < CHUNK_SIZE && pos.z() < CHUNK_SIZE;
}

std::size_t BlockStorage::getIndex(BlockCoord pos) const {
    return static_cast<std::size_t>((pos.x() * CHUNK_SIZE * CHUNK_SIZE) + (pos.y() * CHUNK_SIZE) + pos.z());
}
