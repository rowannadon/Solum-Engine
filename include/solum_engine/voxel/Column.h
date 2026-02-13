#include "solum_engine/voxel/Chunk.h"
#include "solum_engine/voxel/ColumnState.h"
#include "solum_engine/resources/Constants.h"

class Column {
    std::array<Chunk, COLUMN_CHUNKS_Z> chunks;
    ColumnState state;

};