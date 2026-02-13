#include "solum_engine/voxel/ChunkState.h"

#include <array>

uint32_t ChunkState::blockDataVersion() const {
    return blockDataVersion_.load(std::memory_order_acquire);
}

uint32_t ChunkState::lodDataVersion() const {
    return lodDataVersion_.load(std::memory_order_acquire);
}

uint32_t ChunkState::meshDataVersion() const {
    return meshL0Version_.load(std::memory_order_acquire);
}

uint32_t ChunkState::bumpBlockDataVersion() {
    return blockDataVersion_.fetch_add(1u, std::memory_order_acq_rel) + 1u;
}

void ChunkState::setLodDataVersion(uint32_t version) {
    lodDataVersion_.store(version, std::memory_order_release);
}

void ChunkState::setMeshDataVersion(uint32_t version) {
    meshL0Version_.store(version, std::memory_order_release);
}

void ChunkState::markDirty(ChunkDirtyFlags flags) {
    dirtyFlags_.fetch_or(static_cast<uint32_t>(flags), std::memory_order_acq_rel);
}

void ChunkState::clearDirty(ChunkDirtyFlags flags) {
    dirtyFlags_.fetch_and(~static_cast<uint32_t>(flags), std::memory_order_acq_rel);
}

bool ChunkState::isDirty(ChunkDirtyFlags flags) const {
    const uint32_t value = dirtyFlags_.load(std::memory_order_acquire);
    return (value & static_cast<uint32_t>(flags)) == static_cast<uint32_t>(flags);
}

bool ChunkState::needsLodScan() const {
    return isDirty(ChunkDirtyFlags::NeedsLodScan) || lodDataVersion() != blockDataVersion();
}

bool ChunkState::needsMeshL0() const {
    return isDirty(ChunkDirtyFlags::NeedsMeshL0) || meshDataVersion() != blockDataVersion();
}

void ChunkState::markEdgeDirty(Direction direction) {
    constexpr std::array<ChunkDirtyFlags, 6> kEdgeFlags = {
        ChunkDirtyFlags::EdgeDirtyPosX,
        ChunkDirtyFlags::EdgeDirtyNegX,
        ChunkDirtyFlags::EdgeDirtyPosY,
        ChunkDirtyFlags::EdgeDirtyNegY,
        ChunkDirtyFlags::EdgeDirtyPosZ,
        ChunkDirtyFlags::EdgeDirtyNegZ,
    };

    const int index = static_cast<int>(direction);
    if (index < 0 || index >= static_cast<int>(kEdgeFlags.size())) {
        return;
    }

    markDirty(kEdgeFlags[static_cast<std::size_t>(index)] | ChunkDirtyFlags::NeedsMeshL0);
}

uint32_t ChunkState::consumeEdgeDirtyMask() {
    constexpr uint32_t kEdgeMask =
        static_cast<uint32_t>(ChunkDirtyFlags::EdgeDirtyPosX) |
        static_cast<uint32_t>(ChunkDirtyFlags::EdgeDirtyNegX) |
        static_cast<uint32_t>(ChunkDirtyFlags::EdgeDirtyPosY) |
        static_cast<uint32_t>(ChunkDirtyFlags::EdgeDirtyNegY) |
        static_cast<uint32_t>(ChunkDirtyFlags::EdgeDirtyPosZ) |
        static_cast<uint32_t>(ChunkDirtyFlags::EdgeDirtyNegZ);

    const uint32_t current = dirtyFlags_.load(std::memory_order_acquire);
    const uint32_t edgeBits = current & kEdgeMask;
    if (edgeBits == 0u) {
        return 0u;
    }

    dirtyFlags_.fetch_and(~kEdgeMask, std::memory_order_acq_rel);
    return edgeBits;
}
