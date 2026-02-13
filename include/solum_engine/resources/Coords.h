#pragma once

#include "Constants.h"

#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>

// Requires b > 0.
inline constexpr int32_t floor_div(int32_t a, int32_t b) {
    int32_t q = a / b;
    const int32_t r = a % b;
    if (r != 0 && r < 0) {
        --q;
    }
    return q;
}

// Requires b > 0.
inline constexpr int32_t floor_mod(int32_t a, int32_t b) {
    int32_t m = a % b;
    if (m < 0) {
        m += b;
    }
    return m;
}

struct RegionTag {};
struct ColumnTag {};
struct ChunkTag {};
struct BlockTag {};

template <typename Tag>
struct GridCoord3 {
    glm::ivec3 v{0, 0, 0};

    constexpr GridCoord3() = default;
    constexpr explicit GridCoord3(glm::ivec3 value) : v(value) {}
    constexpr GridCoord3(int32_t x, int32_t y, int32_t z) : v(x, y, z) {}

    constexpr int32_t x() const { return v.x; }
    constexpr int32_t y() const { return v.y; }
    constexpr int32_t z() const { return v.z; }

    friend constexpr bool operator==(const GridCoord3& a, const GridCoord3& b) {
        return a.v.x == b.v.x && a.v.y == b.v.y && a.v.z == b.v.z;
    }

    friend constexpr bool operator!=(const GridCoord3& a, const GridCoord3& b) {
        return !(a == b);
    }

    friend constexpr bool operator<(const GridCoord3& a, const GridCoord3& b) {
        if (a.v.x != b.v.x) {
            return a.v.x < b.v.x;
        }
        if (a.v.y != b.v.y) {
            return a.v.y < b.v.y;
        }
        return a.v.z < b.v.z;
    }
};

template <typename Tag>
struct GridCoord2 {
    glm::ivec2 v{0, 0};

    constexpr GridCoord2() = default;
    constexpr explicit GridCoord2(glm::ivec2 value) : v(value) {}
    constexpr GridCoord2(int32_t x, int32_t y) : v(x, y) {}

    constexpr int32_t x() const { return v.x; }
    constexpr int32_t y() const { return v.y; }

    friend constexpr bool operator==(const GridCoord2& a, const GridCoord2& b) {
        return a.v.x == b.v.x && a.v.y == b.v.y;
    }

    friend constexpr bool operator!=(const GridCoord2& a, const GridCoord2& b) {
        return !(a == b);
    }

    friend constexpr bool operator<(const GridCoord2& a, const GridCoord2& b) {
        if (a.v.x != b.v.x) {
            return a.v.x < b.v.x;
        }
        return a.v.y < b.v.y;
    }
};

using RegionCoord = GridCoord2<RegionTag>;
using ColumnCoord = GridCoord2<ColumnTag>;
using ChunkCoord = GridCoord3<ChunkTag>;
using BlockCoord = GridCoord3<BlockTag>;

template <typename Tag>
struct GridCoord2Hash {
    std::size_t operator()(const GridCoord2<Tag>& coord) const noexcept {
        const uint64_t x = static_cast<uint32_t>(coord.x());
        const uint64_t y = static_cast<uint32_t>(coord.y());
        return static_cast<std::size_t>((x << 32u) ^ y);
    }
};

template <typename Tag>
struct GridCoord3Hash {
    std::size_t operator()(const GridCoord3<Tag>& coord) const noexcept {
        const uint64_t x = static_cast<uint32_t>(coord.x());
        const uint64_t y = static_cast<uint32_t>(coord.y());
        const uint64_t z = static_cast<uint32_t>(coord.z());
        std::size_t h = static_cast<std::size_t>((x * 0x9E3779B185EBCA87ull) ^ (y * 0xC2B2AE3D27D4EB4Full));
        h ^= static_cast<std::size_t>(z + 0x165667B19E3779F9ull + (h << 6u) + (h >> 2u));
        return h;
    }
};

using RegionCoordHash = GridCoord2Hash<RegionTag>;
using ColumnCoordHash = GridCoord2Hash<ColumnTag>;
using ChunkCoordHash = GridCoord3Hash<ChunkTag>;
using BlockCoordHash = GridCoord3Hash<BlockTag>;

inline constexpr ChunkCoord block_to_chunk(BlockCoord b) {
    return ChunkCoord{
        floor_div(b.v.x, CHUNK_SIZE),
        floor_div(b.v.y, CHUNK_SIZE),
        floor_div(b.v.z, CHUNK_SIZE)
    };
}

inline constexpr ColumnCoord chunk_to_column(ChunkCoord c) {
    return ColumnCoord{c.v.x, c.v.y};
}

inline constexpr RegionCoord column_to_region(ColumnCoord col) {
    return RegionCoord{
        floor_div(col.v.x, REGION_COLS),
        floor_div(col.v.y, REGION_COLS)
    };
}

inline constexpr RegionCoord chunk_to_region(ChunkCoord c) {
    return column_to_region(chunk_to_column(c));
}

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
    const BlockCoord origin = chunk_to_block_origin(c);
    return BlockCoord{
        origin.v.x + local.x,
        origin.v.y + local.y,
        origin.v.z + local.z
    };
}
