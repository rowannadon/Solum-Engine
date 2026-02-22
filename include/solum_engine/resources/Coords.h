#pragma once
#include "Constants.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <ostream>
#include <cassert>
#include <glm/glm.hpp>

// --------------------------------------------
// Helpers: floor division/mod for negatives
// --------------------------------------------
// Requires b > 0.
inline constexpr int32_t floor_div(int32_t a, int32_t b) {
    int32_t q = a / b;
    int32_t r = a % b;
    if (r != 0 && r < 0) {
        --q;
    }
    return q;
}

// Requires b > 0.
inline constexpr int32_t floor_mod(int32_t a, int32_t b) {
    int32_t m = a % b;
    if (m < 0) m += b;
    return m;
}

// --------------------------------------------
// Tag types to prevent mixing coordinate spaces
// --------------------------------------------
struct RegionTag {};
struct ColumnTag {};
struct ChunkTag {};
struct BlockTag {};

// --------------------------------------------
// Generic integer coord wrappers
// --------------------------------------------
template <typename Tag>
struct GridCoord3 {
    glm::ivec3 v{0, 0, 0};

    constexpr GridCoord3() = default;
    constexpr explicit GridCoord3(glm::ivec3 vv) : v(vv) {}
    constexpr GridCoord3(int32_t x, int32_t y, int32_t z) : v(x, y, z) {}

    constexpr int32_t x() const { return v.x; }
    constexpr int32_t y() const { return v.y; }
    constexpr int32_t z() const { return v.z; }

    friend constexpr bool operator==(const GridCoord3& a, const GridCoord3& b) {
        return a.v.x == b.v.x && a.v.y == b.v.y && a.v.z == b.v.z;
    }

    friend constexpr bool operator<(const GridCoord3& a, const GridCoord3& b) {
        if (a.v.x != b.v.x) return a.v.x < b.v.x;
        if (a.v.y != b.v.y) return a.v.y < b.v.y;
        return a.v.z < b.v.z;
    }
};

template <typename Tag>
struct GridCoord2 {
    glm::ivec2 v{0, 0};

    constexpr GridCoord2() = default;
    constexpr explicit GridCoord2(glm::ivec2 vv) : v(vv) {}
    constexpr GridCoord2(int32_t x, int32_t y) : v(x, y) {}

    constexpr int32_t x() const { return v.x; }
    constexpr int32_t y() const { return v.y; }

    friend constexpr bool operator==(const GridCoord2& a, const GridCoord2& b) {
        return a.v.x == b.v.x && a.v.y == b.v.y;
    }

    friend constexpr bool operator<(const GridCoord2& a, const GridCoord2& b) {
        if (a.v.x != b.v.x) return a.v.x < b.v.x;
        return a.v.y < b.v.y;
    }
};

// Common aliases
using RegionCoord = GridCoord2<RegionTag>;
using ColumnCoord = GridCoord2<ColumnTag>;
using ChunkCoord  = GridCoord3<ChunkTag>;
using BlockCoord  = GridCoord3<BlockTag>;

// Axis convention used throughout the engine:
// X/Y are the horizontal plane, Z is vertical (z-up).

template <typename Tag>
inline std::ostream& operator<<(std::ostream& os, const GridCoord2<Tag>& coord) {
    os << "(" << coord.v.x << ", " << coord.v.y << ")";
    return os;
}

template <typename Tag>
inline std::ostream& operator<<(std::ostream& os, const GridCoord3<Tag>& coord) {
    os << "(" << coord.v.x << ", " << coord.v.y << ", " << coord.v.z << ")";
    return os;
}

// --------------------------------------------
// Conversions (world/grid relations)
// --------------------------------------------

// Block -> Chunk (each chunk covers CHUNK_SIZE^3 blocks)
inline constexpr ChunkCoord block_to_chunk(BlockCoord b) {
    return ChunkCoord{
        floor_div(b.v.x, cfg::CHUNK_SIZE),
        floor_div(b.v.y, cfg::CHUNK_SIZE),
        floor_div(b.v.z, cfg::CHUNK_SIZE)
    };
}

// Chunk -> Column (drop vertical chunk index; columns keyed by chunk x,y)
inline constexpr ColumnCoord chunk_to_column(ChunkCoord c) {
    return ColumnCoord{ c.v.x, c.v.y };
}

// Column -> Region (regions are REGION_COLS x REGION_COLS columns)
inline constexpr RegionCoord column_to_region(ColumnCoord col) {
    return RegionCoord{
        floor_div(col.v.x, cfg::REGION_SIZE),
        floor_div(col.v.y, cfg::REGION_SIZE)
    };
}

// Chunk -> Region (same as its column's region)
inline constexpr RegionCoord chunk_to_region(ChunkCoord c) {
    return column_to_region(chunk_to_column(c));
}

// Local column index within its region: [0..REGION_COLS-1] x [0..REGION_COLS-1]
inline constexpr glm::ivec2 column_local_in_region(ColumnCoord col) {
    return glm::ivec2{
        floor_mod(col.v.x, cfg::REGION_SIZE),
        floor_mod(col.v.y, cfg::REGION_SIZE)
    };
}

// Local block index within its chunk: [0..CHUNK_SIZE-1]^3
inline constexpr glm::ivec3 block_local_in_chunk(BlockCoord b) {
    return glm::ivec3{
        floor_mod(b.v.x, cfg::CHUNK_SIZE),
        floor_mod(b.v.y, cfg::CHUNK_SIZE),
        floor_mod(b.v.z, cfg::CHUNK_SIZE)
    };
}

inline constexpr BlockCoord chunk_to_block_origin(ChunkCoord c) {
    return BlockCoord{
        c.v.x * cfg::CHUNK_SIZE,
        c.v.y * cfg::CHUNK_SIZE,
        c.v.z * cfg::CHUNK_SIZE
    };
}

inline constexpr BlockCoord chunk_local_to_block(ChunkCoord c, glm::ivec3 local) {
    BlockCoord origin = chunk_to_block_origin(c);
    return BlockCoord{
        origin.v.x + local.x,
        origin.v.y + local.y,
        origin.v.z + local.z
    };
}

// --------------------------------------------
// Forward conversions (needed for eager construction)
// --------------------------------------------

// Region -> Column origin (minimum global column coord covered by that region).
inline constexpr ColumnCoord region_to_column_origin(RegionCoord r) {
    return ColumnCoord{
        r.v.x * cfg::REGION_SIZE,
        r.v.y * cfg::REGION_SIZE
    };
}

// Region + local (x,y) -> global ColumnCoord.
inline constexpr ColumnCoord region_local_to_column(RegionCoord r, int32_t local_x, int32_t local_y) {
    ColumnCoord base = region_to_column_origin(r);
    return ColumnCoord{
        base.v.x + local_x,
        base.v.y + local_y
    };
}

// Column + local z -> global ChunkCoord.
// local_z is [0..cfg::COLUMN_HEIGHT-1] and maps to the chunk's global z in a z-up world.
inline constexpr ChunkCoord column_local_to_chunk(ColumnCoord col, int32_t local_z) {
    return ChunkCoord{ col.v.x, col.v.y, local_z };
}

// Region + local (x,y,z) -> global ChunkCoord.
inline constexpr ChunkCoord region_local_to_chunk(RegionCoord r, int32_t local_x, int32_t local_y, int32_t local_z) {
    return column_local_to_chunk(region_local_to_column(r, local_x, local_y), local_z);
}

// Optional: local chunk index within its column (since global z == local z here).
inline constexpr int32_t chunk_local_in_column(ChunkCoord c) {
    return c.v.z;
}

// --------------------------------------------
// Hashing
// --------------------------------------------
namespace coord_hash_detail {
inline std::size_t hash_combine(std::size_t seed, std::size_t value) noexcept {
#if SIZE_MAX > UINT32_MAX
    constexpr std::size_t kGoldenRatio = 0x9e3779b97f4a7c15ull;
#else
    constexpr std::size_t kGoldenRatio = 0x9e3779b9u;
#endif
    return seed ^ (value + kGoldenRatio + (seed << 6) + (seed >> 2));
}
} // namespace coord_hash_detail

namespace std {
template <typename Tag>
struct hash<GridCoord2<Tag>> {
    std::size_t operator()(const GridCoord2<Tag>& coord) const noexcept {
        std::size_t seed = std::hash<int32_t>{}(coord.v.x);
        seed = coord_hash_detail::hash_combine(seed, std::hash<int32_t>{}(coord.v.y));
        return seed;
    }
};

template <typename Tag>
struct hash<GridCoord3<Tag>> {
    std::size_t operator()(const GridCoord3<Tag>& coord) const noexcept {
        std::size_t seed = std::hash<int32_t>{}(coord.v.x);
        seed = coord_hash_detail::hash_combine(seed, std::hash<int32_t>{}(coord.v.y));
        seed = coord_hash_detail::hash_combine(seed, std::hash<int32_t>{}(coord.v.z));
        return seed;
    }
};
} // namespace std
