#include "solum_engine/render/MaterialManager.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "nlohmann_json/json.hpp"

using json = nlohmann::json;

namespace {
bool parseDirectionMask(const std::string& value, uint8_t& outMask) {
    outMask = 0u;
    for (const char c : value) {
        if (std::isspace(static_cast<unsigned char>(c)) != 0) {
            continue;
        }
        switch (static_cast<char>(std::toupper(static_cast<unsigned char>(c)))) {
            case 'X':
                outMask |= 0x1u;
                break;
            case 'Y':
                outMask |= 0x2u;
                break;
            case 'Z':
                outMask |= 0x4u;
                break;
            default:
                return false;
        }
    }
    return true;
}
}  // namespace

bool MaterialManager::loadMaterialConfig(const std::filesystem::path& path,
                                         std::vector<MaterialConfigEntry>& outMaterials) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "MaterialManager: unable to open material config '" << path.string() << "'." << std::endl;
        return false;
    }

    json root;
    try {
        file >> root;
    } catch (const std::exception& e) {
        std::cerr << "MaterialManager: failed to parse '" << path.string() << "': " << e.what() << std::endl;
        return false;
    }

    const json* materialsJson = nullptr;
    if (root.is_array()) {
        materialsJson = &root;
    } else if (root.is_object() && root.contains("materials") && root["materials"].is_array()) {
        materialsJson = &root["materials"];
    } else {
        std::cerr << "MaterialManager: '" << path.string()
                  << "' must be an array or an object with a 'materials' array." << std::endl;
        return false;
    }

    outMaterials.clear();
    outMaterials.reserve(materialsJson->size());

    for (size_t i = 0; i < materialsJson->size(); ++i) {
        const json& entry = (*materialsJson)[i];
        if (!entry.is_object()) {
            std::cerr << "MaterialManager: materials[" << i << "] must be an object." << std::endl;
            return false;
        }
        if (!entry.contains("name") || !entry["name"].is_string()) {
            std::cerr << "MaterialManager: materials[" << i << "] is missing string field 'name'." << std::endl;
            return false;
        }
        if (!entry.contains("texture") || !entry["texture"].is_string()) {
            std::cerr << "MaterialManager: materials[" << i << "] is missing string field 'texture'." << std::endl;
            return false;
        }

        MaterialConfigEntry material{};
        material.name = entry["name"].get<std::string>();
        material.texture = entry["texture"].get<std::string>();
        if (entry.contains("model") && !entry["model"].is_string()) {
            std::cerr << "MaterialManager: materials[" << i << "] field 'model' must be a string when present." << std::endl;
            return false;
        }
        if (entry.contains("doubleSided") && !entry["doubleSided"].is_boolean()) {
            std::cerr << "MaterialManager: materials[" << i << "] field 'doubleSided' must be a boolean when present." << std::endl;
            return false;
        }
        if (entry.contains("randomTextureRotation") && !entry["randomTextureRotation"].is_boolean()) {
            std::cerr << "MaterialManager: materials[" << i << "] field 'randomTextureRotation' must be a boolean when present." << std::endl;
            return false;
        }
        if (entry.contains("randomRotation") && !entry["randomRotation"].is_boolean()) {
            std::cerr << "MaterialManager: materials[" << i << "] field 'randomRotation' must be a boolean when present." << std::endl;
            return false;
        }
        if (entry.contains("randomRotationDirections") && !entry["randomRotationDirections"].is_string()) {
            std::cerr << "MaterialManager: materials[" << i << "] field 'randomRotationDirections' must be a string when present." << std::endl;
            return false;
        }
        if (entry.contains("randomOffsetDirections") && !entry["randomOffsetDirections"].is_string()) {
            std::cerr << "MaterialManager: materials[" << i << "] field 'randomOffsetDirections' must be a string when present." << std::endl;
            return false;
        }
        if (entry.contains("randomOffsetAmount") && !entry["randomOffsetAmount"].is_number()) {
            std::cerr << "MaterialManager: materials[" << i << "] field 'randomOffsetAmount' must be a number when present." << std::endl;
            return false;
        }
        if (entry.contains("blockLightOpacity") && !entry["blockLightOpacity"].is_number()) {
            std::cerr << "MaterialManager: materials[" << i << "] field 'blockLightOpacity' must be a number when present." << std::endl;
            return false;
        }
        if (entry.contains("emissiveLight") && !entry["emissiveLight"].is_number_integer()) {
            std::cerr << "MaterialManager: materials[" << i << "] field 'emissiveLight' must be an integer when present." << std::endl;
            return false;
        }
        if (entry.contains("aoOccluder") && !entry["aoOccluder"].is_boolean()) {
            std::cerr << "MaterialManager: materials[" << i << "] field 'aoOccluder' must be a boolean when present." << std::endl;
            return false;
        }
        if (entry.contains("model")) {
            material.model = entry["model"].get<std::string>();
        }
        if (entry.contains("doubleSided")) {
            material.doubleSided = entry["doubleSided"].get<bool>();
        }
        if (entry.contains("randomTextureRotation")) {
            material.randomTextureRotation = entry["randomTextureRotation"].get<bool>();
        }
        if (entry.contains("randomRotation")) {
            material.randomRotation = entry["randomRotation"].get<bool>();
        }
        if (entry.contains("randomRotationDirections")) {
            const std::string directions = entry["randomRotationDirections"].get<std::string>();
            if (!parseDirectionMask(directions, material.randomRotationDirectionsMask)) {
                std::cerr << "MaterialManager: materials[" << i
                          << "] field 'randomRotationDirections' contains invalid characters. "
                          << "Use only combinations of X, Y, and Z." << std::endl;
                return false;
            }
        }
        if (entry.contains("randomOffsetDirections")) {
            const std::string directions = entry["randomOffsetDirections"].get<std::string>();
            if (!parseDirectionMask(directions, material.randomOffsetDirectionsMask)) {
                std::cerr << "MaterialManager: materials[" << i
                          << "] field 'randomOffsetDirections' contains invalid characters. "
                          << "Use only combinations of X, Y, and Z." << std::endl;
                return false;
            }
        }
        if (entry.contains("randomOffsetAmount")) {
            material.randomOffsetAmount = entry["randomOffsetAmount"].get<float>();
            if (material.randomOffsetAmount < 0.0f || material.randomOffsetAmount > 1.0f) {
                std::cerr << "MaterialManager: materials[" << i
                          << "] field 'randomOffsetAmount' must be within [0.0, 1.0]." << std::endl;
                return false;
            }
        }
        if (entry.contains("blockLightOpacity")) {
            material.blockLightOpacity = entry["blockLightOpacity"].get<float>();
            if (material.blockLightOpacity < 0.0f || material.blockLightOpacity > 1.0f) {
                std::cerr << "MaterialManager: materials[" << i
                          << "] field 'blockLightOpacity' must be within [0.0, 1.0]." << std::endl;
                return false;
            }
        }
        if (entry.contains("emissiveLight")) {
            const int emissive = entry["emissiveLight"].get<int>();
            if (emissive < 0 || emissive > 15) {
                std::cerr << "MaterialManager: materials[" << i
                          << "] field 'emissiveLight' must be within [0, 15]." << std::endl;
                return false;
            }
            material.emissiveLight = static_cast<uint8_t>(emissive);
        }
        if (entry.contains("aoOccluder")) {
            material.aoOccluder = entry["aoOccluder"].get<bool>();
        }
        outMaterials.push_back(std::move(material));
    }

    return true;
}
