#pragma once

#include "solum_engine/resources/Constants.h"
#include "solum_engine/resources/Coords.h"
#include "solum_engine/voxel/Chunk.h"
#include "solum_engine/voxel/ColumnState.h"

#include <array>

class ChunkPool;

class Column {
public:
    explicit Column(ColumnCoord coord, ChunkPool* pool = nullptr);

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
    static std::array<Chunk, COLUMN_CHUNKS_Z> createChunks(ColumnCoord coord, ChunkPool* pool);

    ColumnCoord coord_;
    std::array<Chunk, COLUMN_CHUNKS_Z> chunks_;
    ColumnState state_;
};
