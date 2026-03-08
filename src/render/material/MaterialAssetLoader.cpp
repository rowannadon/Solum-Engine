#include "solum_engine/render/MaterialManager.h"

#include <iostream>
#include <limits>
#include <unordered_map>
#include <unordered_set>

#include "lodepng/lodepng.h"
#include "solum_engine/resources/Constants.h"

namespace {
constexpr uint32_t kFirstMaterialId = 1u;
constexpr const char* kDefaultVoxelModelName = "__default_voxel_model__";
constexpr const char* kDefaultVoxelModelFile = "voxel_model.obj";
}  // namespace

bool MaterialManager::loadPngRgba8(const std::filesystem::path& path,
                                   std::vector<uint8_t>& outPixels,
                                   uint32_t& outWidth,
                                   uint32_t& outHeight) {
    std::vector<unsigned char> rgba;
    unsigned w = 0;
    unsigned h = 0;
    const unsigned decodeError = lodepng::decode(rgba, w, h, path.string());
    if (decodeError != 0 || w == 0 || h == 0) {
        if (decodeError != 0) {
            std::cerr << "MaterialManager: lodepng decode error " << decodeError
                      << " (" << lodepng_error_text(decodeError) << ") for "
                      << path.string() << std::endl;
        }
        return false;
    }

    outPixels.assign(rgba.begin(), rgba.end());
    outWidth = static_cast<uint32_t>(w);
    outHeight = static_cast<uint32_t>(h);
    return true;
}

bool MaterialManager::loadMaterialAssets(const std::vector<MaterialConfigEntry>& configMaterials,
                                         std::vector<LoadedMaterialAsset>& outLoadedMaterials) {
    outLoadedMaterials.clear();
    outLoadedMaterials.reserve(configMaterials.size());

    const std::filesystem::path texturesRoot = std::filesystem::path(RESOURCE_DIR) / "textures";
    for (size_t i = 0; i < configMaterials.size(); ++i) {
        const std::string& name = configMaterials[i].name;
        const std::string& textureRelativePath = configMaterials[i].texture;
        const std::filesystem::path texturePath = texturesRoot / textureRelativePath;

        LoadedMaterialAsset loaded{};
        loaded.name = name;
        loaded.textureRelativePath = textureRelativePath;
        loaded.modelRelativePath = configMaterials[i].model;
        loaded.doubleSided = configMaterials[i].doubleSided;
        loaded.randomTextureRotation = configMaterials[i].randomTextureRotation;
        loaded.randomRotation = configMaterials[i].randomRotation;
        loaded.randomRotationDirectionsMask = configMaterials[i].randomRotationDirectionsMask;
        loaded.randomOffsetDirectionsMask = configMaterials[i].randomOffsetDirectionsMask;
        loaded.randomOffsetAmount = configMaterials[i].randomOffsetAmount;
        loaded.blockLightOpacity = configMaterials[i].blockLightOpacity;
        loaded.emissiveLight = configMaterials[i].emissiveLight;
        loaded.aoOccluder = configMaterials[i].aoOccluder;
        loaded.materialId = static_cast<uint16_t>(kFirstMaterialId + static_cast<uint32_t>(i));
        loaded.textureLayer = static_cast<uint32_t>(i);

        if (!loadPngRgba8(texturePath, loaded.pixels, loaded.width, loaded.height)) {
            std::cerr << "MaterialManager: failed to load material texture '" << texturePath.string()
                      << "' for material '" << name << "'." << std::endl;
            return false;
        }

        outLoadedMaterials.push_back(std::move(loaded));
    }

    return true;
}

bool MaterialManager::buildMaterialModelLibrary(BufferManager& bufferManager,
                                                const std::vector<LoadedMaterialAsset>& loadedMaterials) {
    modelManager_.terminate(bufferManager);
    const std::filesystem::path modelsRoot = std::filesystem::path(RESOURCE_DIR) / "models";
    if (!modelManager_.loadModel(kDefaultVoxelModelName, modelsRoot / kDefaultVoxelModelFile)) {
        std::cerr << "MaterialManager: failed to load fallback model '" << kDefaultVoxelModelFile << "'." << std::endl;
        return false;
    }

    std::unordered_set<std::string> seenModelPaths;
    for (const LoadedMaterialAsset& loaded : loadedMaterials) {
        if (loaded.modelRelativePath.empty()) {
            continue;
        }
        if (!seenModelPaths.insert(loaded.modelRelativePath).second) {
            continue;
        }
        if (!modelManager_.loadModel(loaded.modelRelativePath, modelsRoot / loaded.modelRelativePath)) {
            std::cerr << "MaterialManager: failed to load model '" << loaded.modelRelativePath
                      << "' for material '" << loaded.name << "'." << std::endl;
            return false;
        }
    }

    if (!modelManager_.uploadModels(bufferManager)) {
        return false;
    }

    auto blockModels = std::make_shared<BlockModelLibrary>();
    blockModels->materialToModel.fill(0u);
    blockModels->materialDoubleSided.fill(0u);
    blockModels->materialAoOccluder.fill(1u);

    std::unordered_map<std::string, uint16_t> modelIndexByName;
    auto appendModelToLibrary = [&](const std::string& modelName, uint16_t& outIndex) -> bool {
        const auto existing = modelIndexByName.find(modelName);
        if (existing != modelIndexByName.end()) {
            outIndex = existing->second;
            return true;
        }

        const LoadedModel* loadedModel = modelManager_.getModel(modelName);
        if (loadedModel == nullptr) {
            std::cerr << "MaterialManager: model '" << modelName << "' is missing from ModelManager." << std::endl;
            return false;
        }

        BlockModelDefinition definition{};
        auto appendRefs = [&](const std::vector<uint32_t>& source, std::vector<uint32_t>& destination) {
            destination.reserve(source.size());
            for (uint32_t localQuadIndex : source) {
                if (localQuadIndex >= loadedModel->quadMetadata.size() ||
                    localQuadIndex >= loadedModel->localToGpuQuadIndex.size()) {
                    continue;
                }

                const uint32_t gpuQuadIndex = loadedModel->localToGpuQuadIndex[localQuadIndex];
                if (gpuQuadIndex == std::numeric_limits<uint32_t>::max()) {
                    continue;
                }

                const ModelQuadMetadata& metadata = loadedModel->quadMetadata[localQuadIndex];
                BlockModelQuadRef ref{};
                ref.gpuQuadIndex = gpuQuadIndex;
                ref.preferredFace = metadata.preferredFace;
                ref.minCorner = metadata.minCorner;
                ref.maxCorner = metadata.maxCorner;

                const uint32_t refIndex = static_cast<uint32_t>(blockModels->quadRefs.size());
                blockModels->quadRefs.push_back(ref);
                destination.push_back(refIndex);
            }
        };

        for (uint32_t face = 0u; face < 6u; ++face) {
            appendRefs(loadedModel->cullableQuadIndices[face], definition.cullableQuadRefs[face]);
        }
        appendRefs(loadedModel->nonCullableQuadIndices, definition.nonCullableQuadRefs);

        outIndex = static_cast<uint16_t>(blockModels->models.size());
        blockModels->models.push_back(std::move(definition));
        modelIndexByName.emplace(modelName, outIndex);
        return true;
    };

    uint16_t fallbackModelIndex = 0u;
    if (!appendModelToLibrary(kDefaultVoxelModelName, fallbackModelIndex)) {
        return false;
    }
    blockModels->fallbackModelIndex = fallbackModelIndex;
    blockModels->materialToModel[0] = fallbackModelIndex;
    blockModels->materialDoubleSided[0] = 0u;
    blockModels->materialAoOccluder[0] = 0u;

    for (const LoadedMaterialAsset& material : loadedMaterials) {
        const std::string& modelName = material.modelRelativePath.empty()
            ? std::string(kDefaultVoxelModelName)
            : material.modelRelativePath;
        uint16_t modelIndex = fallbackModelIndex;
        if (!appendModelToLibrary(modelName, modelIndex)) {
            return false;
        }
        blockModels->materialToModel[material.materialId] = modelIndex;
        blockModels->materialDoubleSided[material.materialId] = material.doubleSided ? 1u : 0u;
        blockModels->materialAoOccluder[material.materialId] = material.aoOccluder ? 1u : 0u;
    }

    blockModelLibrary_ = std::move(blockModels);
    return true;
}
