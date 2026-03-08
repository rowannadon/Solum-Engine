#include "solum_engine/voxel/MaterialRegistry.h"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>

#include "nlohmann_json/json.hpp"

using json = nlohmann::json;

namespace {

std::filesystem::path materialConfigPath() {
    return std::filesystem::path(RESOURCE_DIR) / "materials.json";
}

struct MaterialLookup {
    std::unordered_map<std::string, uint16_t> materialIdsByName;
};

MaterialLookup loadMaterialLookup() {
    MaterialLookup lookup;

    const std::filesystem::path path = materialConfigPath();
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "MaterialRegistry: unable to open '" << path.string() << "'." << std::endl;
        return lookup;
    }

    json root;
    try {
        file >> root;
    } catch (const std::exception& e) {
        std::cerr << "MaterialRegistry: failed to parse '" << path.string() << "': "
                  << e.what() << std::endl;
        return lookup;
    }

    const json* materialsJson = nullptr;
    if (root.is_array()) {
        materialsJson = &root;
    } else if (root.is_object() && root.contains("materials") && root["materials"].is_array()) {
        materialsJson = &root["materials"];
    } else {
        std::cerr << "MaterialRegistry: '" << path.string()
                  << "' must be an array or contain a 'materials' array." << std::endl;
        return lookup;
    }

    const size_t maxMaterialCount = static_cast<size_t>(std::numeric_limits<uint16_t>::max());
    const size_t materialCount = std::min(materialsJson->size(), maxMaterialCount);
    for (size_t i = 0; i < materialCount; ++i) {
        const json& material = (*materialsJson)[i];
        if (!material.is_object() || !material.contains("name") || !material["name"].is_string()) {
            continue;
        }

        const std::string name = material["name"].get<std::string>();
        const uint16_t materialId = static_cast<uint16_t>(i + 1u);
        const bool inserted = lookup.materialIdsByName.emplace(name, materialId).second;
        if (!inserted) {
            std::cerr << "MaterialRegistry: duplicate material name '" << name
                      << "' in '" << path.string() << "'." << std::endl;
        }
    }

    return lookup;
}

const MaterialLookup& materialLookup() {
    static const MaterialLookup kLookup = loadMaterialLookup();
    return kLookup;
}

}  // namespace

bool MaterialRegistry::tryResolveMaterialId(std::string_view materialName, uint16_t& outMaterialId) {
    outMaterialId = 0u;
    if (materialName.empty()) {
        return false;
    }

    const auto& lookup = materialLookup().materialIdsByName;
    const auto found = lookup.find(std::string(materialName));
    if (found == lookup.end()) {
        return false;
    }

    outMaterialId = found->second;
    return true;
}

bool MaterialRegistry::tryResolveBlock(std::string_view materialName, BlockMaterial& outBlock) {
    uint16_t materialId = 0u;
    if (!tryResolveMaterialId(materialName, materialId)) {
        return false;
    }

    outBlock = UnpackedBlockMaterial{materialId, 0, Direction::PlusZ, 0}.pack();
    return true;
}

BlockMaterial MaterialRegistry::resolveBlockOr(std::string_view materialName, const BlockMaterial& fallbackBlock) {
    BlockMaterial resolved{};
    if (tryResolveBlock(materialName, resolved)) {
        return resolved;
    }
    return fallbackBlock;
}
