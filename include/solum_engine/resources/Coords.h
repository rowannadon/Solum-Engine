#pragma once
#include "Constants.h"
#include <cstdint>
#include <glm/glm.hpp>

// --------------------------------------------
// Helpers: floor division/mod for negatives
// --------------------------------------------
// Requires b > 0.
inline constexpr int32_t floor_div(int32_t a, int32_t b) {
    int32_t q = a / b;
    int32_t r = a % b;
    if (r != 0 && r < 0) {
        // For positive b, adjust toward -infinity.
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

    // Lexicographic ordering for ordered containers.
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
using RegionCoord = GridCoord2<RegionTag>; // region indices in region-grid (2D)
using ColumnCoord = GridCoord2<ColumnTag>; // column indices in column-grid (2D: x,y)
using ChunkCoord  = GridCoord3<ChunkTag>;  // chunk indices in chunk-grid (3D: x,y,z)
using BlockCoord  = GridCoord3<BlockTag>;  // block indices in block-grid (3D: x,y,z)


// --------------------------------------------
// Conversions
// --------------------------------------------

// Block -> Chunk (each chunk covers 32^3 blocks)
inline constexpr ChunkCoord block_to_chunk(BlockCoord b) {
    return ChunkCoord{
        floor_div(b.v.x, CHUNK_SIZE),
        floor_div(b.v.y, CHUNK_SIZE),
        floor_div(b.v.z, CHUNK_SIZE)
    };
}

// Chunk -> Column (drop vertical chunk index; columns keyed by chunk x,y)
inline constexpr ColumnCoord chunk_to_column(ChunkCoord c) {
    return ColumnCoord{ c.v.x, c.v.y };
}

// Column -> Region (regions are 16x16 columns)
inline constexpr RegionCoord column_to_region(ColumnCoord col) {
    return RegionCoord{
        floor_div(col.v.x, REGION_COLS),
        floor_div(col.v.y, REGION_COLS)
    };
}

// Chunk -> Region (same as its column's region)
inline constexpr RegionCoord chunk_to_region(ChunkCoord c) {
    return column_to_region(chunk_to_column(c));
}

// Local column index within its region: [0..15] x [0..15]
inline constexpr glm::ivec2 column_local_in_region(ColumnCoord col) {
    return glm::ivec2{
        floor_mod(col.v.x, REGION_COLS),
        floor_mod(col.v.y, REGION_COLS)
    };
}

inline constexpr glm::ivec3 block_local_in_chunk(BlockCoord b) {
    return glm::ivec3{
        floor_mod(b.v.x, CHUNK_SIZE),
        floor_mod(b.v.y, CHUNK_SIZE),
        floor_mod(b.v.z, CHUNK_SIZE)
    };
}

inline constexpr BlockCoord chunk_to_block_origin(ChunkCoord c) {
    return BlockCoord{
        c.v.x * CHUNK_SIZE,
        c.v.y * CHUNK_SIZE,
        c.v.z * CHUNK_SIZE
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
