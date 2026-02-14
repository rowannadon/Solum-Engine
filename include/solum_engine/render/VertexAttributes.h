#pragma once

#include <cstdint>

struct VertexAttributes {
    int32_t x;
    int32_t y;
    int32_t z;
    uint16_t u;
    uint16_t v;
    uint16_t material;
    uint8_t n_x;
    uint8_t n_y;
    uint8_t n_z;
    uint8_t lodLevel;
};
