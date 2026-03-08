#pragma once

#include <cstdint>
#include <string_view>

#include "solum_engine/voxel/BlockMaterial.h"

class MaterialRegistry {
public:
    static bool tryResolveMaterialId(std::string_view materialName, uint16_t& outMaterialId);
    static bool tryResolveBlock(std::string_view materialName, BlockMaterial& outBlock);
    static BlockMaterial resolveBlockOr(std::string_view materialName, const BlockMaterial& fallbackBlock);
};
