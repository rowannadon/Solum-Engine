#pragma once

#include "solum_engine/resources/Constants.h"

#include <atomic>
#include <cstdint>

enum class ChunkDirtyFlags : uint32_t {
    None = 0,
    NeedsLodScan = 1u << 0u,
    NeedsMeshL0 = 1u << 1u,
    NeedsCompression = 1u << 2u,
    PendingUncompress = 1u << 3u,
    EdgeDirtyPosX = 1u << 4u,
    EdgeDirtyNegX = 1u << 5u,
    EdgeDirtyPosY = 1u << 6u,
    EdgeDirtyNegY = 1u << 7u,
    EdgeDirtyPosZ = 1u << 8u,
    EdgeDirtyNegZ = 1u << 9u,
};

inline constexpr ChunkDirtyFlags operator|(ChunkDirtyFlags lhs, ChunkDirtyFlags rhs) {
    return static_cast<ChunkDirtyFlags>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

inline constexpr ChunkDirtyFlags operator&(ChunkDirtyFlags lhs, ChunkDirtyFlags rhs) {
    return static_cast<ChunkDirtyFlags>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}

class ChunkState {
public:
    uint32_t blockDataVersion() const;
    uint32_t lodDataVersion() const;
    uint32_t meshDataVersion() const;

    uint32_t bumpBlockDataVersion();
    void setLodDataVersion(uint32_t version);
    void setMeshDataVersion(uint32_t version);

    void markDirty(ChunkDirtyFlags flags);
    void clearDirty(ChunkDirtyFlags flags);
    bool isDirty(ChunkDirtyFlags flags) const;

    bool needsLodScan() const;
    bool needsMeshL0() const;

    void markEdgeDirty(Direction direction);
    uint32_t consumeEdgeDirtyMask();

private:
    std::atomic<uint32_t> blockDataVersion_{1};
    std::atomic<uint32_t> lodDataVersion_{0};
    std::atomic<uint32_t> meshL0Version_{0};
    std::atomic<uint32_t> dirtyFlags_{static_cast<uint32_t>(ChunkDirtyFlags::NeedsLodScan | ChunkDirtyFlags::NeedsMeshL0)};
};
