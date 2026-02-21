#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>
#include "solum_engine/voxel/BlockMaterial.h"

class Chunk {
public:
    static constexpr size_t SIZE = 16;
    static constexpr size_t VOLUME = SIZE * SIZE * SIZE; // 4096 voxels

    Chunk();

    // High performance getters and setters
    BlockMaterial getBlock(uint8_t x, uint8_t y, uint8_t z) const;
    void setBlock(uint8_t x, uint8_t y, uint8_t z, const BlockMaterial blockID);

private:
    uint8_t bits_per_block_;
    std::vector<BlockMaterial> palette_;
    std::vector<uint64_t> data_; // Bitpacked indices

    // Flattens 3D coordinates into a 1D index using fast bitwise shifts
    // y * 256 + z * 16 + x
    inline uint16_t getVoxelIndex(uint8_t x, uint8_t y, uint8_t z) const {
        return (static_cast<uint16_t>(y) << 8) | (static_cast<uint16_t>(z) << 4) | x;
    }

    uint32_t getPaletteIndex(uint16_t voxel_index) const;
    void setPaletteIndex(uint16_t voxel_index, uint32_t palette_index);
    void resizeBitArray(uint8_t new_bits_per_block);
};