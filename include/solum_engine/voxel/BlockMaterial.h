#include <cstdint>

enum FacingDirection {
    PlusX,
    MinusX,
    PlusY,
    MinusY,
    PlusZ,
    MinusZ
};

struct BlockMaterial;

struct UnpackedBlockMaterial {
    uint16_t id;
    int waterLevel; // 0-15
    FacingDirection dir;

    BlockMaterial pack() const;
};

struct BlockMaterial {
    uint32_t data;

    UnpackedBlockMaterial unpack() const;
};


