#pragma once
#include <glm/glm.hpp>
#include <array>

namespace cfg {
    static constexpr int CHUNK_SIZE = 16;
    static constexpr int CHUNK_VOLUME_BLOCKS = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

    static constexpr int COLUMN_HEIGHT = 32;
    static constexpr int COLUMN_HEIGHT_BLOCKS = COLUMN_HEIGHT * CHUNK_SIZE;

    static constexpr int REGION_SIZE = 32;
    static constexpr int REGION_SIZE_BLOCKS = REGION_SIZE * CHUNK_SIZE;
    static constexpr int REGION_VOLUME_COLUMNS = REGION_SIZE * REGION_SIZE;
}

static constexpr float PI = 3.14159265359f;

enum Direction {
    PlusX,
    MinusX,
    PlusY,
    MinusY,
    // In z-up mode, +Z/-Z are top/bottom.
    PlusZ,
    MinusZ
};
