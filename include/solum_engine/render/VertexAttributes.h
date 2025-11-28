#pragma once

#include <cstdint>

struct VertexAttributes {
    uint16_t x;
    uint16_t y;
    uint16_t z;
    uint16_t u;
    uint16_t v;
    uint16_t material;
    uint8_t n_x;
    uint8_t n_y;
    uint8_t n_z;
};