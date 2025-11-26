#include "solum_engine/voxel/Chunk.h"

Chunk::Chunk() {
    data.resize(CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE);
};

bool Chunk::setBlock(glm::ivec3 pos, UnpackedBlockMaterial mat) {
    if (!validatePos(pos)) {
        return false;
    }

    int index = getIndex(pos);
    data[index] = mat.pack();
    return true;
}

UnpackedBlockMaterial Chunk::getBlock(glm::ivec3 pos) {
    if (!validatePos(pos)) {
        UnpackedBlockMaterial m;
        m.id = 0;
        return m;
    }

    int index = getIndex(pos);
    BlockMaterial mat = data[index];
    return mat.unpack();
}

bool Chunk::validatePos(glm::ivec3 pos) {
    if (pos.x < 0 || pos.y < 0 || pos.z < 0) {
        return false;
    }
    if (pos.x > CHUNK_SIZE - 1 || pos.y > CHUNK_SIZE - 1 || pos.z > CHUNK_SIZE - 1) {
        return false;
    }
    return true;
}

int Chunk::getIndex(glm::ivec3 pos) {
    return (pos.x * CHUNK_SIZE * CHUNK_SIZE) + (pos.y * CHUNK_SIZE) + pos.z;
}