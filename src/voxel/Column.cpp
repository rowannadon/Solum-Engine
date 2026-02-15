#include "solum_engine/voxel/Column.h"

#include <utility>
#include <stdexcept>

namespace {
template <std::size_t... I>
std::array<Chunk, sizeof...(I)> makeChunks(ColumnCoord coord, std::index_sequence<I...>) {
    return {Chunk(ChunkCoord{coord.x(), coord.y(), static_cast<int>(I)})...};
}
}

Column::Column(ColumnCoord coord)
    : coord_(coord), chunks_(createChunks(coord)) {}

ColumnCoord Column::coord() const {
    return coord_;
}

Chunk& Column::chunkAt(int localZ) {
    Chunk* chunk = tryGetChunk(localZ);
    if (chunk == nullptr) {
        throw std::out_of_range("Column::chunkAt localZ out of range");
    }
    return *chunk;
}

const Chunk& Column::chunkAt(int localZ) const {
    const Chunk* chunk = tryGetChunk(localZ);
    if (chunk == nullptr) {
        throw std::out_of_range("Column::chunkAt localZ out of range");
    }
    return *chunk;
}

Chunk* Column::tryGetChunk(int localZ) {
    if (localZ < 0 || localZ >= COLUMN_CHUNKS_Z) {
        return nullptr;
    }

    return &chunks_[static_cast<std::size_t>(localZ)];
}

const Chunk* Column::tryGetChunk(int localZ) const {
    if (localZ < 0 || localZ >= COLUMN_CHUNKS_Z) {
        return nullptr;
    }

    return &chunks_[static_cast<std::size_t>(localZ)];
}

Chunk* Column::tryGetChunkByWorldZ(int worldChunkZ) {
    const int localZ = worldChunkZ;
    return tryGetChunk(localZ);
}

const Chunk* Column::tryGetChunkByWorldZ(int worldChunkZ) const {
    const int localZ = worldChunkZ;
    return tryGetChunk(localZ);
}

ColumnState& Column::state() {
    return state_;
}

const ColumnState& Column::state() const {
    return state_;
}

std::array<Chunk, COLUMN_CHUNKS_Z> Column::createChunks(ColumnCoord coord) {
    return makeChunks(coord, std::make_index_sequence<COLUMN_CHUNKS_Z>{});
}
