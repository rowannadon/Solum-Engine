#include "solum_engine/render/MaterialManager.h"

#include <iostream>

#include "solum_engine/resources/Constants.h"

namespace {
constexpr uint32_t kRandomTextureRotationFlagBit = 1u << 0u;
constexpr uint32_t kRandomOffsetXFlagBit = 1u << 1u;
constexpr uint32_t kRandomOffsetYFlagBit = 1u << 2u;
constexpr uint32_t kRandomOffsetZFlagBit = 1u << 3u;
constexpr uint32_t kRandomModelRotationEnabledFlagBit = 1u << 4u;
constexpr uint32_t kRandomModelRotationXFlagBit = 1u << 5u;
constexpr uint32_t kRandomModelRotationYFlagBit = 1u << 6u;
constexpr uint32_t kRandomModelRotationZFlagBit = 1u << 7u;
constexpr uint32_t kDoubleSidedMaterialFlagBit = 1u << 8u;
}  // namespace

bool MaterialManager::initialize(BufferManager& bufferManager, TextureManager& textureManager) {
    if (initialized_) {
        return true;
    }

    materialMetadata_.assign(kLookupEntryCount, MaterialMetadataGPU{});
    materials_.clear();
    blockModelLibrary_.reset();

    if (!buildDefaultMaterials(bufferManager, textureManager)) {
        modelManager_.terminate(bufferManager);
        blockModelLibrary_.reset();
        materials_.clear();
        materialMetadata_.assign(kLookupEntryCount, MaterialMetadataGPU{});
        return false;
    }

    initialized_ = true;
    return true;
}

void MaterialManager::terminate(BufferManager& bufferManager, TextureManager& textureManager) {
    if (!initialized_) {
        return;
    }

    modelManager_.terminate(bufferManager);
    bufferManager.deleteBuffer(kMaterialMetadataBufferName);
    textureManager.removeTextureView(kMaterialTextureArrayViewName);
    textureManager.removeTexture(kMaterialTextureArrayName);
    textureManager.removeSampler(kMaterialSamplerName);

    blockModelLibrary_.reset();
    materials_.clear();
    materialMetadata_.clear();
    initialized_ = false;
}

std::shared_ptr<const BlockModelLibrary> MaterialManager::blockModelLibrary() const {
    return blockModelLibrary_;
}

bool MaterialManager::buildDefaultMaterials(BufferManager& bufferManager, TextureManager& textureManager) {
    const std::filesystem::path materialConfigPath = std::filesystem::path(RESOURCE_DIR) / "materials.json";

    std::vector<MaterialConfigEntry> configMaterials;
    if (!loadMaterialConfig(materialConfigPath, configMaterials)) {
        return false;
    }
    if (configMaterials.empty()) {
        std::cerr << "MaterialManager: '" << materialConfigPath.string() << "' contains no materials." << std::endl;
        return false;
    }
    if (configMaterials.size() > static_cast<size_t>(kMaxMaterialId)) {
        std::cerr << "MaterialManager: material count exceeds max supported IDs (65535)." << std::endl;
        return false;
    }

    std::vector<LoadedMaterialAsset> loadedMaterials;
    loadedMaterials.reserve(configMaterials.size());

    if (!loadMaterialAssets(configMaterials, loadedMaterials)) {
        return false;
    }
    if (!buildMaterialModelLibrary(bufferManager, loadedMaterials)) {
        return false;
    }

    for (const LoadedMaterialAsset& material : loadedMaterials) {
        uint32_t flags = 0u;
        if (material.randomTextureRotation) {
            flags |= kRandomTextureRotationFlagBit;
        }
        if (material.randomRotation) {
            flags |= kRandomModelRotationEnabledFlagBit;
        }
        if ((material.randomRotationDirectionsMask & 0x1u) != 0u) {
            flags |= kRandomModelRotationXFlagBit;
        }
        if ((material.randomRotationDirectionsMask & 0x2u) != 0u) {
            flags |= kRandomModelRotationYFlagBit;
        }
        if ((material.randomRotationDirectionsMask & 0x4u) != 0u) {
            flags |= kRandomModelRotationZFlagBit;
        }
        if ((material.randomOffsetDirectionsMask & 0x1u) != 0u) {
            flags |= kRandomOffsetXFlagBit;
        }
        if ((material.randomOffsetDirectionsMask & 0x2u) != 0u) {
            flags |= kRandomOffsetYFlagBit;
        }
        if ((material.randomOffsetDirectionsMask & 0x4u) != 0u) {
            flags |= kRandomOffsetZFlagBit;
        }
        if (material.doubleSided) {
            flags |= kDoubleSidedMaterialFlagBit;
        }

        materialMetadata_[material.materialId].textureIndex = material.textureLayer;
        materialMetadata_[material.materialId].flags = flags;
        materialMetadata_[material.materialId].randomOffsetAmount = material.randomOffsetAmount;
        materialMetadata_[material.materialId].pad0 = 0.0f;

        materials_.emplace(
            material.materialId,
            MaterialDefinition{
                material.materialId,
                material.name,
                material.textureLayer,
                1.0f,
                0.0f,
                material.doubleSided,
                material.randomTextureRotation,
                material.randomRotation,
                material.randomRotationDirectionsMask,
                material.randomOffsetDirectionsMask,
                material.randomOffsetAmount,
                material.blockLightOpacity,
                material.emissiveLight,
                material.aoOccluder
            }
        );
    }

    if (!uploadMaterialGpuResources(bufferManager, textureManager, loadedMaterials)) {
        return false;
    }

    return true;
}
