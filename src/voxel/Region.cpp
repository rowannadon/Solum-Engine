#include "solum_engine/voxel/Region.h"

#include <algorithm>

namespace {
constexpr int kRegionColumnCount = REGION_COLS * REGION_COLS;
constexpr int kLodLevelCount = 4;
}

Region::Region(RegionCoord coord)
    : coord_(coord),
      columns_(kRegionColumnCount),
      lodLevels_(kLodLevelCount) {
    for (int level = 0; level < kLodLevelCount; ++level) {
        LodLevelGrid& grid = lodLevels_[static_cast<std::size_t>(level)];
        grid.shift = level + 1;
        grid.tilesPerAxis = REGION_COLS >> grid.shift;
        if (grid.tilesPerAxis < 1) {
            grid.tilesPerAxis = 1;
        }

        const std::size_t tileCount = static_cast<std::size_t>(grid.tilesPerAxis * grid.tilesPerAxis * COLUMN_CHUNKS_Z);
        grid.tiles.resize(tileCount);
    }
}

RegionCoord Region::coord() const {
    return coord_;
}

RegionState& Region::state() {
    return state_;
}

const RegionState& Region::state() const {
    return state_;
}

Column& Region::ensureColumn(int localX, int localY) {
    const std::size_t idx = columnIndex(localX, localY);
    if (!columns_[idx]) {
        const int worldX = coord_.x() * REGION_COLS + localX;
        const int worldY = coord_.y() * REGION_COLS + localY;
        columns_[idx] = std::make_unique<Column>(ColumnCoord{worldX, worldY});
    }

    return *columns_[idx];
}

Column* Region::tryGetColumn(int localX, int localY) {
    if (!validateColumnLocal(localX, localY)) {
        return nullptr;
    }

    return columns_[columnIndex(localX, localY)].get();
}

const Column* Region::tryGetColumn(int localX, int localY) const {
    if (!validateColumnLocal(localX, localY)) {
        return nullptr;
    }

    return columns_[columnIndex(localX, localY)].get();
}

Column* Region::tryGetColumn(ColumnCoord worldColumnCoord) {
    if (column_to_region(worldColumnCoord) != coord_) {
        return nullptr;
    }

    const glm::ivec2 local = column_local_in_region(worldColumnCoord);
    return tryGetColumn(local.x, local.y);
}

const Column* Region::tryGetColumn(ColumnCoord worldColumnCoord) const {
    if (column_to_region(worldColumnCoord) != coord_) {
        return nullptr;
    }

    const glm::ivec2 local = column_local_in_region(worldColumnCoord);
    return tryGetColumn(local.x, local.y);
}

Chunk* Region::tryGetChunk(ChunkCoord worldChunkCoord) {
    if (chunk_to_region(worldChunkCoord) != coord_) {
        return nullptr;
    }

    Column* column = tryGetColumn(chunk_to_column(worldChunkCoord));
    if (column == nullptr) {
        return nullptr;
    }

    return column->tryGetChunk(worldChunkCoord.z());
}

const Chunk* Region::tryGetChunk(ChunkCoord worldChunkCoord) const {
    if (chunk_to_region(worldChunkCoord) != coord_) {
        return nullptr;
    }

    const Column* column = tryGetColumn(chunk_to_column(worldChunkCoord));
    if (column == nullptr) {
        return nullptr;
    }

    return column->tryGetChunk(worldChunkCoord.z());
}

void Region::markLodTilesDirtyForChunk(ChunkCoord chunkCoord) {
    for (int level = 0; level < kLodLevelCount; ++level) {
        const std::optional<RegionLodTileCoord> tile = chunkToTileCoord(chunkCoord, level);
        if (!tile.has_value()) {
            continue;
        }

        RegionLodTileState* state = tryGetTileState(level, *tile);
        if (state == nullptr) {
            continue;
        }

        state->dirty = true;
    }
}

std::vector<RegionLodTileCoord> Region::collectDirtyTiles(int lodLevel, uint8_t zSlice) const {
    std::vector<RegionLodTileCoord> result;
    if (lodLevel < 0 || lodLevel >= kLodLevelCount || !validateZSlice(zSlice)) {
        return result;
    }

    const LodLevelGrid& grid = lodLevels_[static_cast<std::size_t>(lodLevel)];
    for (int y = 0; y < grid.tilesPerAxis; ++y) {
        for (int x = 0; x < grid.tilesPerAxis; ++x) {
            const RegionLodTileCoord coord{
                static_cast<uint8_t>(x),
                static_cast<uint8_t>(y),
                zSlice,
            };

            const RegionLodTileState* tile = tryGetTileState(lodLevel, coord);
            if (tile != nullptr && tile->dirty) {
                result.push_back(coord);
            }
        }
    }

    return result;
}

RegionLodTileState* Region::tryGetTileState(int lodLevel, RegionLodTileCoord coord) {
    if (lodLevel < 0 || lodLevel >= kLodLevelCount || !validateZSlice(coord.z)) {
        return nullptr;
    }

    LodLevelGrid& grid = lodLevels_[static_cast<std::size_t>(lodLevel)];
    if (coord.x >= grid.tilesPerAxis || coord.y >= grid.tilesPerAxis) {
        return nullptr;
    }

    return &grid.tiles[tileIndex(grid, coord)];
}

const RegionLodTileState* Region::tryGetTileState(int lodLevel, RegionLodTileCoord coord) const {
    if (lodLevel < 0 || lodLevel >= kLodLevelCount || !validateZSlice(coord.z)) {
        return nullptr;
    }

    const LodLevelGrid& grid = lodLevels_[static_cast<std::size_t>(lodLevel)];
    if (coord.x >= grid.tilesPerAxis || coord.y >= grid.tilesPerAxis) {
        return nullptr;
    }

    return &grid.tiles[tileIndex(grid, coord)];
}

void Region::markTileClean(int lodLevel, RegionLodTileCoord coord, MeshHandle handle, uint32_t sourceVersion) {
    RegionLodTileState* tile = tryGetTileState(lodLevel, coord);
    if (tile == nullptr) {
        return;
    }

    tile->dirty = false;
    tile->meshHandle = handle;
    tile->sourceVersion = sourceVersion;
}

std::size_t Region::columnIndex(int localX, int localY) {
    return static_cast<std::size_t>(localY * REGION_COLS + localX);
}

bool Region::validateColumnLocal(int localX, int localY) {
    return localX >= 0 && localY >= 0 && localX < REGION_COLS && localY < REGION_COLS;
}

bool Region::validateZSlice(int zSlice) {
    return zSlice >= 0 && zSlice < COLUMN_CHUNKS_Z;
}

std::optional<RegionLodTileCoord> Region::chunkToTileCoord(ChunkCoord chunkCoord, int lodLevel) const {
    if (lodLevel < 0 || lodLevel >= kLodLevelCount) {
        return std::nullopt;
    }
    if (chunk_to_region(chunkCoord) != coord_) {
        return std::nullopt;
    }

    const int shift = lodLevels_[static_cast<std::size_t>(lodLevel)].shift;
    const int localX = floor_mod(chunkCoord.x(), REGION_COLS);
    const int localY = floor_mod(chunkCoord.y(), REGION_COLS);
    const int localZ = chunkCoord.z();
    if (!validateZSlice(localZ)) {
        return std::nullopt;
    }

    const int tileX = localX >> shift;
    const int tileY = localY >> shift;

    const LodLevelGrid& grid = lodLevels_[static_cast<std::size_t>(lodLevel)];
    if (tileX < 0 || tileY < 0 || tileX >= grid.tilesPerAxis || tileY >= grid.tilesPerAxis) {
        return std::nullopt;
    }

    return RegionLodTileCoord{
        static_cast<uint8_t>(tileX),
        static_cast<uint8_t>(tileY),
        static_cast<uint8_t>(localZ),
    };
}

std::size_t Region::tileIndex(const LodLevelGrid& grid, RegionLodTileCoord coord) const {
    const std::size_t layerStride = static_cast<std::size_t>(grid.tilesPerAxis * grid.tilesPerAxis);
    const std::size_t layerBase = static_cast<std::size_t>(coord.z) * layerStride;
    const std::size_t localIndex = static_cast<std::size_t>(coord.y * grid.tilesPerAxis + coord.x);
    return layerBase + localIndex;
}
