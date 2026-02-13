#pragma once

#include "solum_engine/resources/Constants.h"
#include "solum_engine/resources/Coords.h"
#include "solum_engine/voxel/ChunkMeshes.h"
#include "solum_engine/voxel/Column.h"
#include "solum_engine/voxel/RegionState.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

class ChunkPool;

struct RegionLodTileCoord {
    uint8_t x = 0;
    uint8_t y = 0;
    uint8_t z = 0;
};

struct RegionLodTileState {
    bool dirty = true;
    uint32_t sourceVersion = 0;
    MeshHandle meshHandle = MeshHandle::invalid();
};

class Region {
public:
    explicit Region(RegionCoord coord, ChunkPool* pool = nullptr);

    RegionCoord coord() const;
    RegionState& state();
    const RegionState& state() const;

    Column& ensureColumn(int localX, int localY);
    Column* tryGetColumn(int localX, int localY);
    const Column* tryGetColumn(int localX, int localY) const;

    Column* tryGetColumn(ColumnCoord worldColumnCoord);
    const Column* tryGetColumn(ColumnCoord worldColumnCoord) const;

    Chunk* tryGetChunk(ChunkCoord worldChunkCoord);
    const Chunk* tryGetChunk(ChunkCoord worldChunkCoord) const;

    void markLodTilesDirtyForChunk(ChunkCoord chunkCoord);
    std::vector<RegionLodTileCoord> collectDirtyTiles(int lodLevel, uint8_t zSlice) const;

    RegionLodTileState* tryGetTileState(int lodLevel, RegionLodTileCoord coord);
    const RegionLodTileState* tryGetTileState(int lodLevel, RegionLodTileCoord coord) const;

    void markTileClean(int lodLevel, RegionLodTileCoord coord, MeshHandle handle, uint32_t sourceVersion);

private:
    struct LodLevelGrid {
        int shift = 1;
        int tilesPerAxis = 8;
        std::vector<RegionLodTileState> tiles;
    };

    static std::size_t columnIndex(int localX, int localY);
    static bool validateColumnLocal(int localX, int localY);
    static bool validateZSlice(int zSlice);

    std::optional<RegionLodTileCoord> chunkToTileCoord(ChunkCoord chunkCoord, int lodLevel) const;
    std::size_t tileIndex(const LodLevelGrid& grid, RegionLodTileCoord coord) const;

    RegionCoord coord_;
    ChunkPool* pool_ = nullptr;
    RegionState state_;

    std::vector<std::unique_ptr<Column>> columns_;
    std::vector<LodLevelGrid> lodLevels_;
};
