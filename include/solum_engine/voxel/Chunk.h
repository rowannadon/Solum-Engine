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
    uint8_t setBlock(uint8_t x, uint8_t y, uint8_t z, const BlockMaterial blockID);
    uint8_t getPackedLight(uint8_t x, uint8_t y, uint8_t z, uint8_t mipLevel = 0) const;
    void setPackedLight(uint8_t x, uint8_t y, uint8_t z, uint8_t packedLight);
    void setPackedLightVolume(const std::array<uint8_t, VOLUME>& packedLights);
    uint8_t getSkyLight(uint8_t x, uint8_t y, uint8_t z) const;
    uint8_t getBlockLight(uint8_t x, uint8_t y, uint8_t z) const;
    bool isAllAir() const noexcept { return solidVoxelCount_ == 0; }
    static constexpr uint8_t mipSize(uint8_t mipLevel) {
        return (mipLevel > MAX_MIP_LEVEL) ? 1u : static_cast<uint8_t>(SIZE >> mipLevel);
    }
    static constexpr uint8_t packLight(uint8_t skyLight, uint8_t blockLight) {
        return static_cast<uint8_t>(((skyLight & 0x0Fu) << 4u) | (blockLight & 0x0Fu));
    }
    static constexpr uint8_t unpackSkyLight(uint8_t packedLight) {
        return static_cast<uint8_t>((packedLight >> 4u) & 0x0Fu);
    }
    static constexpr uint8_t unpackBlockLight(uint8_t packedLight) {
        return static_cast<uint8_t>(packedLight & 0x0Fu);
    }

private:
    struct MipStorage {
        uint8_t bitsPerBlock = 0;
        uint8_t size = 0;
        std::vector<BlockMaterial> palette;
        std::vector<uint64_t> data;
    };

    struct LightMipStorage {
        uint8_t size = 0;
        uint8_t uniformValue = 0;
        std::vector<uint8_t> denseData;
        std::vector<uint16_t> valueCounts;
        uint16_t distinctValueCount = 0;

        bool isUniform() const noexcept { return denseData.empty(); }
    };

    std::array<MipStorage, MAX_MIP_LEVEL + 1> mips_{};
    std::array<LightMipStorage, MAX_MIP_LEVEL + 1> lightMips_{};
    uint8_t defaultPackedLight_ = packLight(0u, 0u);
    uint16_t solidVoxelCount_ = 0;

    static uint16_t getVoxelIndex(uint8_t x, uint8_t y, uint8_t z, uint8_t size);
    static uint32_t getPaletteIndex(const MipStorage& storage, uint16_t voxelIndex);
    static void setPaletteIndex(MipStorage& storage, uint16_t voxelIndex, uint32_t paletteIndex);
    static void resizeBitArray(MipStorage& storage, uint8_t newBitsPerBlock);
    static size_t lightLevelVolume(const LightMipStorage& storage);
    static uint8_t getPackedLightFromStorage(const LightMipStorage& storage, uint16_t voxelIndex);
    static void compressLightToUniform(LightMipStorage& storage, uint8_t uniformPackedLight);
    static void rebuildLightValueCounts(LightMipStorage& storage);
    static bool setPackedLightInStorage(LightMipStorage& storage,
                                        uint16_t voxelIndex,
                                        uint8_t packedLight,
                                        bool trackDistinctCounts);
    static bool setPackedLightLevelFromArray(LightMipStorage& storage,
                                             const uint8_t* packedLights,
                                             size_t count,
                                             bool trackDistinctCounts);

    static bool isSolid(BlockMaterial block);
    static uint8_t downsamplePackedLightFromChildren(const LightMipStorage& childLevel,
                                                     uint8_t childSize,
                                                     uint8_t px,
                                                     uint8_t py,
                                                     uint8_t pz);
    static BlockMaterial airBlock();
    static BlockMaterial downsampleBlockFromChildren(const MipStorage& childLevel, uint8_t px, uint8_t py, uint8_t pz);

    static void setBlockInStorage(MipStorage& storage,
                                  uint8_t x,
                                  uint8_t y,
                                  uint8_t z,
                                  BlockMaterial blockID,
                                  bool* outChanged);
};
