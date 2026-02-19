#include "solum_engine/voxel/Chunk.h"
#include "solum_engine/voxel/ColumnState.h"
#include "solum_engine/resources/Constants.h"
#include "solum_engine/resources/Util.h"

class Column {
private:
    ColumnCoord coord_;
    ColumnState state_;
    std::array<Chunk, COLUMN_HEIGHT_CHUNKS> chunks_;

public:
    explicit Column(ColumnCoord coord) : 
        coord_(coord), 
        chunks_(make_array<COLUMN_HEIGHT_CHUNKS>([&](auto I) {
            constexpr int z = static_cast<int>(I);
            return Chunk(column_local_to_chunk(coord_, z));
        })) {};

    Chunk* getChunk(int z) {
        if (z < COLUMN_HEIGHT_CHUNKS)
            return &chunks_[z];
        else
            return nullptr;
    }
};