#pragma once
#include "Constants.h"
#include <cstdint>
#include <compare>
#include <glm/glm.hpp>

// --------------------------------------------
// Helpers: floor division/mod for negatives
// --------------------------------------------
inline constexpr int32_t floor_div(int32_t a, int32_t b) {
	// b > 0
	int32_t q = a / b;
	int32_t r = a % b;
	if (r != 0 && ((r < 0) != (b < 0))) --q;
	return q;
}

inline constexpr int32_t floor_mod(int32_t a, int32_t b) {
	// b > 0
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
// Generic 3D integer coord wrapper
// --------------------------------------------
template <typename Tag>
struct GridCoord3 {
	glm::ivec3 v{ 0 };

	constexpr GridCoord3() = default;
	constexpr explicit GridCoord3(glm::ivec3 vv) : v(vv) {}
	constexpr GridCoord3(int32_t x, int32_t y, int32_t z) : v(x, y, z) {}

	constexpr int32_t x() const { return v.x; }
	constexpr int32_t y() const { return v.y; }
	constexpr int32_t z() const { return v.z; }

	// Comparisons: enables use in ordered maps/sets if desired
	auto operator<=>(const GridCoord3&) const = default;
};

template <typename Tag>
struct GridCoord2 {
	glm::ivec2 v{ 0 };

	constexpr GridCoord2() = default;
	constexpr explicit GridCoord2(glm::ivec2 vv) : v(vv) {}
	constexpr GridCoord2(int32_t x, int32_t y) : v(x, y) {}

	constexpr int32_t x() const { return v.x; }
	constexpr int32_t y() const { return v.y; }

	// Comparisons: enables use in ordered maps/sets if desired
	auto operator<=>(const GridCoord2&) const = default;
};

// Common aliases
using RegionCoord = GridCoord2<RegionTag>; // region indices in region-grid
using ColumnCoord = GridCoord2<ColumnTag>; // column indices in column-grid (x,y, z=unused or 0)
using ChunkCoord = GridCoord3<ChunkTag>;  // chunk indices in chunk-grid
using BlockCoord = GridCoord3<BlockTag>;  // block indices in block-grid

// Block -> Chunk (each chunk covers 32^3 blocks)
inline ChunkCoord block_to_chunk(BlockCoord b) {
	return ChunkCoord{
	  floor_div(b.v.x, CHUNK_SIZE),
	  floor_div(b.v.y, CHUNK_SIZE),
	  floor_div(b.v.z, CHUNK_SIZE)
	};
}

// Chunk -> Column (column is chunk-grid in XY, z unused or 0)
inline ColumnCoord chunk_to_column(ChunkCoord c) {
	return ColumnCoord{ c.v.x, c.v.y };
}

// Column -> Region (region is 16x16 columns)
inline RegionCoord column_to_region(ColumnCoord col) {
	return RegionCoord{
	  floor_div(col.v.x, REGION_COLS),
	  floor_div(col.v.y, REGION_COLS)
	};
}

// Chunk -> Region (go through column)
inline RegionCoord chunk_to_region(ChunkCoord c) {
	return RegionCoord{
	  floor_div(c.v.x, REGION_COLS),
	  floor_div(c.v.y, REGION_COLS)
	};
}

// Chunk local coords within its region (0..15 in XY for columns, 0..31 for chunk z etc.)
inline glm::ivec2 column_local_in_region(ColumnCoord col) {
	return {
	  floor_mod(col.v.x, REGION_COLS),
	  floor_mod(col.v.y, REGION_COLS)
	};
}

// Block local coords within chunk: (0..31)
inline glm::ivec3 block_local_in_chunk(BlockCoord b) {
	return {
	  floor_mod(b.v.x, CHUNK_SIZE),
	  floor_mod(b.v.y, CHUNK_SIZE),
	  floor_mod(b.v.z, CHUNK_SIZE)
	};
}