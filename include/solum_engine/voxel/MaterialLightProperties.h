#pragma once

#include <array>
#include <cstdint>
#include <limits>

class MaterialLightProperties {
public:
    static constexpr uint32_t kLookupEntryCount = 65536u;
    static constexpr uint8_t kOpaqueLightLoss = std::numeric_limits<uint8_t>::max();

    struct LookupTables {
        std::array<float, kLookupEntryCount> blockLightOpacity{};
        std::array<uint8_t, kLookupEntryCount> blockLightStepLoss{};
        std::array<uint8_t, kLookupEntryCount> skyLightVerticalLoss{};
        std::array<uint8_t, kLookupEntryCount> blocksLightMask{};
    };

    static float blockLightOpacity(uint16_t materialId);
    static uint8_t blockLightStepLoss(uint16_t materialId);
    static uint8_t skyLightVerticalLoss(uint16_t materialId);
    static bool blocksLight(uint16_t materialId);

private:
    static const LookupTables& lookup();
};
