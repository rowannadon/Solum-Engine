#pragma once

#include "solum_engine/resources/Constants.h"
#include "solum_engine/resources/Coords.h"
#include "solum_engine/voxel/Chunk.h"
#include "solum_engine/voxel/ColumnState.h"

#include <array>

class Column {
public:
    explicit Column(ColumnCoord coord);

    ColumnCoord coord() const;

    Chunk& chunkAt(int localZ);
    const Chunk& chunkAt(int localZ) const;

    Chunk* tryGetChunk(int localZ);
    const Chunk* tryGetChunk(int localZ) const;

    Chunk* tryGetChunkByWorldZ(int worldChunkZ);
    const Chunk* tryGetChunkByWorldZ(int worldChunkZ) const;

    ColumnState& state();
    const ColumnState& state() const;

private:
    static std::array<Chunk, COLUMN_CHUNKS_Z> createChunks(ColumnCoord coord);

    ColumnCoord coord_;
    std::array<Chunk, COLUMN_CHUNKS_Z> chunks_;
    ColumnState state_;
};
