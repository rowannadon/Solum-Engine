#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

struct MeshTileCoord {
    int32_t x = 0;
    int32_t y = 0;

    friend bool operator==(const MeshTileCoord& a, const MeshTileCoord& b) {
        return a.x == b.x && a.y == b.y;
    }

    friend bool operator<(const MeshTileCoord& a, const MeshTileCoord& b) {
        if (a.x != b.x) {
            return a.x < b.x;
        }
        return a.y < b.y;
    }
};

struct MeshTileSliceCoord {
    MeshTileCoord tile{};
    int32_t z = 0;

    friend bool operator==(const MeshTileSliceCoord& a, const MeshTileSliceCoord& b) {
        return a.tile == b.tile && a.z == b.z;
    }

    friend bool operator<(const MeshTileSliceCoord& a, const MeshTileSliceCoord& b) {
        if (a.tile == b.tile) {
            return a.z < b.z;
        }
        return a.tile < b.tile;
    }
};

struct TileLodCoord {
    MeshTileSliceCoord tile{};
    uint8_t lodLevel = 0;

    friend bool operator==(const TileLodCoord& a, const TileLodCoord& b) {
        return a.lodLevel == b.lodLevel && a.tile == b.tile;
    }

    friend bool operator<(const TileLodCoord& a, const TileLodCoord& b) {
        if (a.tile == b.tile) {
            return a.lodLevel < b.lodLevel;
        }
        return a.tile < b.tile;
    }
};

namespace std {
template <>
struct hash<MeshTileCoord> {
    size_t operator()(const MeshTileCoord& coord) const noexcept {
#if SIZE_MAX > UINT32_MAX
        constexpr size_t kGoldenRatio = 0x9e3779b97f4a7c15ull;
#else
        constexpr size_t kGoldenRatio = 0x9e3779b9u;
#endif
        size_t seed = hash<int32_t>{}(coord.x);
        seed ^= hash<int32_t>{}(coord.y) + kGoldenRatio + (seed << 6) + (seed >> 2);
        return seed;
    }
};

template <>
struct hash<MeshTileSliceCoord> {
    size_t operator()(const MeshTileSliceCoord& coord) const noexcept {
#if SIZE_MAX > UINT32_MAX
        constexpr size_t kGoldenRatio = 0x9e3779b97f4a7c15ull;
#else
        constexpr size_t kGoldenRatio = 0x9e3779b9u;
#endif
        size_t seed = hash<MeshTileCoord>{}(coord.tile);
        seed ^= hash<int32_t>{}(coord.z) + kGoldenRatio + (seed << 6) + (seed >> 2);
        return seed;
    }
};

template <>
struct hash<TileLodCoord> {
    size_t operator()(const TileLodCoord& coord) const noexcept {
#if SIZE_MAX > UINT32_MAX
        constexpr size_t kGoldenRatio = 0x9e3779b97f4a7c15ull;
#else
        constexpr size_t kGoldenRatio = 0x9e3779b9u;
#endif
        size_t seed = hash<MeshTileSliceCoord>{}(coord.tile);
        seed ^= hash<uint8_t>{}(coord.lodLevel) + kGoldenRatio + (seed << 6) + (seed >> 2);
        return seed;
    }
};
}  // namespace std
