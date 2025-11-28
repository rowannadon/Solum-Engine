#include "solum_engine/voxel/BlockMaterial.h"


BlockMaterial UnpackedBlockMaterial::pack() const {
    BlockMaterial mat{0};

    // Pack facing direction in bits 0–2
    mat.data |= (static_cast<uint32_t>(dir) & 0x7);

    // Pack water level in bits 3–6 (4 bits)
    mat.data |= ((waterLevel & 0xF) << 3);

    // Pack material ID in bits 16–31
    mat.data |= (static_cast<uint32_t>(id) << 16);

    return mat;
}

UnpackedBlockMaterial BlockMaterial::unpack() const {
    UnpackedBlockMaterial mat{};

    // Extract ID from bits 16–31
    mat.id = static_cast<uint16_t>((data >> 16) & 0xFFFF);

    // Extract water level from bits 3–6
    mat.waterLevel = (data >> 3) & 0xF;

    // Extract facing direction from bits 0–2
    mat.dir = static_cast<Direction>(data & 0x7);

    return mat;
}