#include <cstdint>
#include <solum_engine/resources/Constants.h>

struct BlockMaterial;

struct UnpackedBlockMaterial {
    uint16_t id;
    int waterLevel; // 0-15
    Direction dir;

    BlockMaterial pack() const;
};

struct BlockMaterial {
    uint32_t data;

    UnpackedBlockMaterial unpack() const;
};


