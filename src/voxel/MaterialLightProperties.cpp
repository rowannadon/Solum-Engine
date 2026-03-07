#include "solum_engine/voxel/MaterialLightProperties.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "nlohmann_json/json.hpp"

using json = nlohmann::json;

namespace {
constexpr float kDefaultOpacity = 1.0f;
constexpr float kOpaqueThreshold = 0.999f;

std::filesystem::path materialConfigPath() {
    return std::filesystem::path(RESOURCE_DIR) / "materials.json";
}

uint8_t toStepLoss(float opacity) {
    if (opacity >= kOpaqueThreshold) {
        return MaterialLightProperties::kOpaqueLightLoss;
    }

    const float scaled = 1.0f + (opacity * 14.0f);
    const int rounded = static_cast<int>(std::lround(scaled));
    return static_cast<uint8_t>(std::clamp(rounded, 1, 15));
}

uint8_t toSkyVerticalLoss(float opacity) {
    const float scaled = opacity * 15.0f;
    const int rounded = static_cast<int>(std::lround(scaled));
    return static_cast<uint8_t>(std::clamp(rounded, 0, 15));
}

void applyOpacity(MaterialLightProperties::LookupTables& lookup, uint16_t materialId, float opacity) {
    const float clampedOpacity = std::clamp(opacity, 0.0f, 1.0f);
    lookup.blockLightOpacity[materialId] = clampedOpacity;
    lookup.blockLightStepLoss[materialId] = toStepLoss(clampedOpacity);
    lookup.skyLightVerticalLoss[materialId] = toSkyVerticalLoss(clampedOpacity);
    lookup.blocksLightMask[materialId] = (clampedOpacity >= kOpaqueThreshold) ? 1u : 0u;
}
}  // namespace

const MaterialLightProperties::LookupTables& MaterialLightProperties::lookup() {
    static const LookupTables kLookup = [] {
        LookupTables lookup{};
        lookup.blockLightOpacity.fill(kDefaultOpacity);
        lookup.blockLightStepLoss.fill(kOpaqueLightLoss);
        lookup.skyLightVerticalLoss.fill(15u);
        lookup.blocksLightMask.fill(1u);

        applyOpacity(lookup, 0u, 0.0f);  // Air is always fully transparent to light.

        const std::filesystem::path path = materialConfigPath();
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "MaterialLightProperties: unable to open '" << path.string()
                      << "'. Using default blockLightOpacity=1.0 for all materials." << std::endl;
            return lookup;
        }

        json root;
        try {
            file >> root;
        } catch (const std::exception& e) {
            std::cerr << "MaterialLightProperties: failed to parse '" << path.string()
                      << "': " << e.what()
                      << ". Using default blockLightOpacity=1.0 for all materials." << std::endl;
            return lookup;
        }

        const json* materialsJson = nullptr;
        if (root.is_array()) {
            materialsJson = &root;
        } else if (root.is_object() && root.contains("materials") && root["materials"].is_array()) {
            materialsJson = &root["materials"];
        } else {
            std::cerr << "MaterialLightProperties: '" << path.string()
                      << "' must be an array or contain a 'materials' array. "
                      << "Using default blockLightOpacity=1.0 for all materials." << std::endl;
            return lookup;
        }

        const size_t maxMaterials = static_cast<size_t>(kLookupEntryCount - 1u);
        const size_t materialCount = std::min(materialsJson->size(), maxMaterials);

        for (size_t i = 0; i < materialCount; ++i) {
            const json& entry = (*materialsJson)[i];
            if (!entry.is_object()) {
                continue;
            }

            if (!entry.contains("blockLightOpacity")) {
                continue;
            }

            if (!entry["blockLightOpacity"].is_number()) {
                std::cerr << "MaterialLightProperties: materials[" << i
                          << "] field 'blockLightOpacity' must be a number. "
                          << "Using default 1.0 for this material." << std::endl;
                continue;
            }

            float opacity = entry["blockLightOpacity"].get<float>();
            if (!std::isfinite(opacity)) {
                std::cerr << "MaterialLightProperties: materials[" << i
                          << "] field 'blockLightOpacity' must be finite. "
                          << "Using default 1.0 for this material." << std::endl;
                continue;
            }

            if (opacity < 0.0f || opacity > 1.0f) {
                std::cerr << "MaterialLightProperties: materials[" << i
                          << "] field 'blockLightOpacity' is clamped to [0.0, 1.0]." << std::endl;
            }

            const uint16_t materialId = static_cast<uint16_t>(i + 1u);
            applyOpacity(lookup, materialId, opacity);
        }

        return lookup;
    }();

    return kLookup;
}

float MaterialLightProperties::blockLightOpacity(uint16_t materialId) {
    return lookup().blockLightOpacity[materialId];
}

uint8_t MaterialLightProperties::blockLightStepLoss(uint16_t materialId) {
    return lookup().blockLightStepLoss[materialId];
}

uint8_t MaterialLightProperties::skyLightVerticalLoss(uint16_t materialId) {
    return lookup().skyLightVerticalLoss[materialId];
}

bool MaterialLightProperties::blocksLight(uint16_t materialId) {
    return lookup().blocksLightMask[materialId] != 0u;
}
