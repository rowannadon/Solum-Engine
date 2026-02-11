#pragma once
#include <glm/glm.hpp>
#include <array>

#define CHUNK_SIZE 32
#define CHUNK_SIZE_P 34

#define REGION_COLS 16
#define COLUMN_CHUNKS_Z 32

static constexpr int32_t REGION_BLOCKS_XY = REGION_COLS * CHUNK_SIZE; // 512 blocks

static constexpr float PI = 3.14159265359f;

enum Direction {
    PlusX,
    MinusX,
    PlusY,
    MinusY,
    PlusZ,
    MinusZ
};