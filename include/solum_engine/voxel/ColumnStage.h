#pragma once

#include <cstdint>

enum class ColumnStage : uint8_t {
    Empty = 0,
    TerrainReady = 1,
    StructureReady = 2,
    Completed = 3,
};
