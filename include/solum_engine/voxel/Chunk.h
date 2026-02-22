#pragma once
#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "solum_engine/voxel/BlockMaterial.h"

class Chunk {
public:
    static constexpr size_t SIZE = 16;
    static constexpr size_t VOLUME = SIZE * SIZE * SIZE; // 4096 voxels
    static constexpr uint8_t MAX_MIP_LEVEL = 4;

    Chunk();

    // High performance getters and setters
    BlockMaterial getBlock(uint8_t x, uint8_t y, uint8_t z, uint8_t mipLevel = 0) const;
    void setBlock(uint8_t x, uint8_t y, uint8_t z, const BlockMaterial blockID);
    static constexpr uint8_t mipSize(uint8_t mipLevel) {
        return (mipLevel > MAX_MIP_LEVEL) ? 1u : static_cast<uint8_t>(SIZE >> mipLevel);
    }

private:
    struct MipStorage {
        uint8_t bitsPerBlock = 0;
        uint8_t size = 0;
        std::vector<BlockMaterial> palette;
        std::vector<uint64_t> data;
    };

    std::array<MipStorage, MAX_MIP_LEVEL + 1> mips_{};

    static uint16_t getVoxelIndex(uint8_t x, uint8_t y, uint8_t z, uint8_t size);
    static uint32_t getPaletteIndex(const MipStorage& storage, uint16_t voxelIndex);
    static void setPaletteIndex(MipStorage& storage, uint16_t voxelIndex, uint32_t paletteIndex);
    static void resizeBitArray(MipStorage& storage, uint8_t newBitsPerBlock);

    static bool isSolid(BlockMaterial block);
    static BlockMaterial airBlock();
    static BlockMaterial downsampleBlockFromChildren(const MipStorage& childLevel, uint8_t px, uint8_t py, uint8_t pz);

    static void setBlockInStorage(MipStorage& storage,
                                  uint8_t x,
                                  uint8_t y,
                                  uint8_t z,
                                  BlockMaterial blockID,
                                  bool* outChanged);
};
