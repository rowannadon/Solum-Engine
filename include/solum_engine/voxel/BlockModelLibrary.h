#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

struct BlockModelQuadRef {
    uint32_t gpuQuadIndex = 0u;
    uint8_t preferredFace = 0u;
    glm::vec3 minCorner{0.0f};
    glm::vec3 maxCorner{1.0f};
};

struct BlockModelDefinition {
    std::array<std::vector<uint32_t>, 6> cullableQuadRefs{};
    std::vector<uint32_t> nonCullableQuadRefs{};
};

struct BlockModelLibrary {
    static constexpr uint32_t kMaterialEntryCount = 65536u;

    std::array<uint16_t, kMaterialEntryCount> materialToModel{};
    std::array<uint8_t, kMaterialEntryCount> materialDoubleSided{};
    std::vector<BlockModelDefinition> models{};
    std::vector<BlockModelQuadRef> quadRefs{};
    uint16_t fallbackModelIndex = 0u;

    const BlockModelDefinition* modelByIndex(uint16_t modelIndex) const {
        if (modelIndex >= models.size()) {
            return nullptr;
        }
        return &models[modelIndex];
    }

    const BlockModelDefinition* modelForMaterial(uint16_t materialId) const {
        const uint16_t modelIndex = materialToModel[materialId];
        return modelByIndex(modelIndex);
    }

    bool isMaterialDoubleSided(uint16_t materialId) const {
        return materialDoubleSided[materialId] != 0u;
    }
};
