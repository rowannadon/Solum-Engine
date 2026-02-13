#pragma once

#include "solum_engine/resources/Constants.h"
#include "solum_engine/voxel/BlockMaterial.h"

#include <array>
#include <cstddef>
#include <cstdint>

class LodStorage {
public:
    static constexpr int kLevelCount = 5;

    void clear();
    void rebuild(const BlockMaterial* blocks, uint32_t blockDataVersion);

    bool isUpToDate(uint32_t blockDataVersion) const;
    uint32_t sourceVersion() const;

    uint16_t dominantMaterial(int level, int x, int y, int z) const;
    void setDominantMaterial(int level, int x, int y, int z, uint16_t materialId);

    static int levelDimension(int level);

private:
    static constexpr std::array<int, kLevelCount> kDimensions = {16, 8, 4, 2, 1};
    static constexpr std::array<std::size_t, kLevelCount> kOffsets = {
        0u,       // 16^3
        4096u,    // 16^3
        4608u,    // 16^3 + 8^3
        4672u,    // 16^3 + 8^3 + 4^3
        4680u,    // 16^3 + 8^3 + 4^3 + 2^3
    };
    static constexpr std::size_t kTotalCellCount =
        (16u * 16u * 16u) + (8u * 8u * 8u) + (4u * 4u * 4u) + (2u * 2u * 2u) + 1u;

    static std::size_t levelIndex(int level, int x, int y, int z);

    std::array<uint16_t, kTotalCellCount> dominantMaterials_{};
    uint32_t sourceVersion_ = 0;
};
