#include "solum_engine/voxel/Chunk.h"

#include <cstddef>

Chunk::Chunk(ChunkCoord coord)
    : coord_(coord) {}

ChunkCoord Chunk::coord() const {
    return coord_;
}

bool Chunk::setBlock(BlockCoord localPos, UnpackedBlockMaterial mat) {
    if (!validateLocalPos(localPos)) {
        return false;
    }

    BlockMaterial* data = getBlockData();
    if (data == nullptr) {
        return false;
    }

    data[localIndex(localPos)] = mat.pack();

    state_.bumpBlockDataVersion();
    state_.markDirty(ChunkDirtyFlags::NeedsLodScan | ChunkDirtyFlags::NeedsMeshL0);

    if (localPos.x() == 0) {
        state_.markEdgeDirty(Direction::MinusX);
    }
    if (localPos.x() == CHUNK_SIZE - 1) {
        state_.markEdgeDirty(Direction::PlusX);
    }
    if (localPos.y() == 0) {
        state_.markEdgeDirty(Direction::MinusY);
    }
    if (localPos.y() == CHUNK_SIZE - 1) {
        state_.markEdgeDirty(Direction::PlusY);
    }
    if (localPos.z() == 0) {
        state_.markEdgeDirty(Direction::MinusZ);
    }
    if (localPos.z() == CHUNK_SIZE - 1) {
        state_.markEdgeDirty(Direction::PlusZ);
    }

    return true;
}

UnpackedBlockMaterial Chunk::getBlock(BlockCoord localPos) const {
    if (!validateLocalPos(localPos)) {
        return {};
    }

    const BlockMaterial* data = getBlockData();
    if (data == nullptr) {
        return {};
    }

    return data[localIndex(localPos)].unpack();
}

BlockMaterial* Chunk::getBlockData() {
    return bootstrapData_.data();
}

const BlockMaterial* Chunk::getBlockData() const {
    return bootstrapData_.data();
}

ChunkState& Chunk::state() {
    return state_;
}

const ChunkState& Chunk::state() const {
    return state_;
}

LodStorage& Chunk::lodStorage() {
    return lodStorage_;
}

const LodStorage& Chunk::lodStorage() const {
    return lodStorage_;
}

bool Chunk::rebuildLodData() {
    const BlockMaterial* blocks = getBlockData();
    if (blocks == nullptr) {
        return false;
    }

    const uint32_t version = state_.blockDataVersion();
    lodStorage_.rebuild(blocks, version);
    state_.setLodDataVersion(version);
    state_.clearDirty(ChunkDirtyFlags::NeedsLodScan);
    return true;
}

MeshHandle Chunk::meshHandleL0() const {
    return meshHandleL0_;
}

void Chunk::setMeshHandleL0(MeshHandle handle, uint32_t derivedFromVersion) {
    meshHandleL0_ = handle;
    state_.setMeshDataVersion(derivedFromVersion);
    state_.clearDirty(ChunkDirtyFlags::NeedsMeshL0);
}

void Chunk::markBulkDataWrite() {
    state_.bumpBlockDataVersion();
    state_.markDirty(ChunkDirtyFlags::NeedsLodScan | ChunkDirtyFlags::NeedsMeshL0);
}

bool Chunk::validateLocalPos(BlockCoord pos) {
    return pos.x() >= 0 && pos.y() >= 0 && pos.z() >= 0 &&
        pos.x() < CHUNK_SIZE && pos.y() < CHUNK_SIZE && pos.z() < CHUNK_SIZE;
}

std::size_t Chunk::localIndex(BlockCoord pos) {
    return static_cast<std::size_t>((pos.x() * CHUNK_SIZE * CHUNK_SIZE) + (pos.y() * CHUNK_SIZE) + pos.z());
}
