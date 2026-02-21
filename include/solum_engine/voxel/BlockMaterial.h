#pragma once

#include <cstdint>
#include <solum_engine/resources/Constants.h>

struct BlockMaterial;

struct UnpackedBlockMaterial {
    uint16_t id = 0;
    int waterLevel = 0;    // 0-15
    Direction dir = Direction::PlusY;
    uint8_t rotation = 0;  // 0-3 (4 rotation states)

    BlockMaterial pack() const;
};

struct BlockMaterial {
    uint32_t data = 0;

    bool operator==(const BlockMaterial& other) const { return data == other.data; }
    bool operator!=(const BlockMaterial& other) const { return !(*this == other); }

    UnpackedBlockMaterial unpack() const;
};
