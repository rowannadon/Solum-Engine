#include "solum_engine/voxel/BlockMaterial.h"

namespace {
    // Bit positions / masks
    constexpr uint32_t DIR_SHIFT   = 0;
    constexpr uint32_t DIR_MASK    = 0x7; // 3 bits

    constexpr uint32_t ROT_SHIFT   = 3;
    constexpr uint32_t ROT_MASK    = 0x3; // 2 bits

    constexpr uint32_t WATER_SHIFT = 5;
    constexpr uint32_t WATER_MASK  = 0xF; // 4 bits

    constexpr uint32_t ID_SHIFT    = 16;
    constexpr uint32_t ID_MASK     = 0xFFFF; // 16 bits
}


BlockMaterial UnpackedBlockMaterial::pack() const {
    BlockMaterial mat{0};

    // Pack facing direction in bits 0-2
    mat.data |= (static_cast<uint32_t>(dir) & DIR_MASK) << DIR_SHIFT;

    // Pack rotation in bits 3-4
    mat.data |= (static_cast<uint32_t>(rotation) & ROT_MASK) << ROT_SHIFT;

    // Pack water level in bits 5-8
    mat.data |= (static_cast<uint32_t>(waterLevel) & WATER_MASK) << WATER_SHIFT;

    // Pack material ID in bits 16-31
    mat.data |= (static_cast<uint32_t>(id) & ID_MASK) << ID_SHIFT;

    return mat;
}

UnpackedBlockMaterial BlockMaterial::unpack() const {
    UnpackedBlockMaterial mat{};

    // Extract ID from bits 16-31
    mat.id = static_cast<uint16_t>((data >> ID_SHIFT) & ID_MASK);

    // Extract water level from bits 5-8
    mat.waterLevel = static_cast<int>((data >> WATER_SHIFT) & WATER_MASK);

    // Extract rotation from bits 3-4
    mat.rotation = static_cast<uint8_t>((data >> ROT_SHIFT) & ROT_MASK);

    // Extract facing direction from bits 0-2
    mat.dir = static_cast<Direction>((data >> DIR_SHIFT) & DIR_MASK);

    return mat;
}
