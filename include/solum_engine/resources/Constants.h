#pragma once
#include <glm/glm.hpp>
#include <array>

#define CHUNK_SIZE 32
#define CHUNK_SIZE_P 34

#define REGION_COLS 16
#define COLUMN_HEIGHT_CHUNKS 32

static constexpr int REGION_BLOCKS_XY = REGION_COLS * CHUNK_SIZE; // 512 blocks
static constexpr int REGION_TOTAL_COLUMNS = REGION_COLS * REGION_COLS;
static constexpr int REGION_TOTAL_CHUNKS = REGION_COLS * REGION_COLS * COLUMN_HEIGHT_CHUNKS; // 512 blocks

static constexpr int CHUNK_TOTAL_BLOCKS = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

static constexpr int COLUMN_HEIGHT_BLOCKS = COLUMN_HEIGHT_CHUNKS * CHUNK_SIZE;

static constexpr float PI = 3.14159265359f;

enum Direction {
    PlusX,
    MinusX,
    PlusY,
    MinusY,
    PlusZ,
    MinusZ
};