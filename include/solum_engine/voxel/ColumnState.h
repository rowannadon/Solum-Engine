#include <atomic>
#include "solum_engine/voxel/ColumnStage.h"

class ColumnState {
    std::atomic<ColumnStage> stage;
    std::atomic<uint32_t> readyNeighborCount;
};