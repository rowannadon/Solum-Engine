#pragma once

#include "solum_engine/resources/Constants.h"

#include <array>
#include <cstdint>

class LightStorage {
public:
    uint8_t skyLight(int index) const;
    uint8_t blockLight(int index) const;

    void setSkyLight(int index, uint8_t value);
    void setBlockLight(int index, uint8_t value);

private:
    static uint8_t clampLight(uint8_t value);

    std::array<uint8_t, CHUNK_BLOCKS> skyLight_{};
    std::array<uint8_t, CHUNK_BLOCKS> blockLight_{};
};
