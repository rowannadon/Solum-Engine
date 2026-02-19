#include <atomic>

enum ColumnStage {
    Empty = 0,
    TerrainDone = 1,
    StructuresDone = 2
};

struct ColumnState {
    std::atomic<ColumnStage> stage;
    std::atomic<uint32_t> blockEpoch;
    std::atomic<uint32_t> readyNeighborCount;
};