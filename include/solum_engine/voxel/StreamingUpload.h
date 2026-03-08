#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "solum_engine/render/MeshletPacking.h"
#include "solum_engine/voxel/MeshTileTypes.h"

struct MeshTileLodKey {
    MeshTileSliceCoord tile{};
    uint8_t lod = 0u;

    friend bool operator==(const MeshTileLodKey& a, const MeshTileLodKey& b) {
        return a.tile == b.tile && a.lod == b.lod;
    }

    friend bool operator<(const MeshTileLodKey& a, const MeshTileLodKey& b) {
        if (a.tile == b.tile) {
            return a.lod < b.lod;
        }
        return a.tile < b.tile;
    }
};

struct MeshTileLodUpload {
    MeshTileLodKey key{};
    std::shared_ptr<const PackedMeshletData> culledPacked;
    std::shared_ptr<const PackedMeshletData> doubleSidedPacked;
    uint64_t revision = 0u;
};

struct MeshTileSelectionEntry {
    MeshTileSliceCoord tile{};
    int8_t selectedLod = -1;
};

struct MeshStreamingDelta {
    std::vector<MeshTileLodUpload> upserts;
    std::vector<MeshTileLodKey> removals;
    std::vector<MeshTileSelectionEntry> selectionSnapshot;
    uint64_t revision = 0u;
};

namespace std {
template <>
struct hash<MeshTileLodKey> {
    size_t operator()(const MeshTileLodKey& key) const noexcept {
#if SIZE_MAX > UINT32_MAX
        constexpr size_t kGoldenRatio = 0x9e3779b97f4a7c15ull;
#else
        constexpr size_t kGoldenRatio = 0x9e3779b9u;
#endif
        size_t seed = hash<MeshTileSliceCoord>{}(key.tile);
        seed ^= hash<uint8_t>{}(key.lod) + kGoldenRatio + (seed << 6) + (seed >> 2);
        return seed;
    }
};
}  // namespace std
