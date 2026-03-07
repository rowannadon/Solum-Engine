#include "solum_engine/render/MaterialManager.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <unordered_set>

#include "lodepng/lodepng.h"
#include "nlohmann_json/json.hpp"

using json = nlohmann::json;
using namespace wgpu;

namespace {
constexpr uint32_t kFirstMaterialId = 1u;
constexpr const char* kDefaultVoxelModelName = "__default_voxel_model__";
constexpr const char* kDefaultVoxelModelFile = "voxel_model.obj";

struct LoadedMaterialTexture {
    std::string name;
    std::string textureRelativePath;
    std::string modelRelativePath;
    bool doubleSided = false;
    bool randomRotation = false;
    uint8_t randomOffsetDirectionsMask = 0u;
    float randomOffsetAmount = 0.0f;
    float blockLightOpacity = 1.0f;
    std::vector<uint8_t> pixels;
    uint32_t width = 0u;
    uint32_t height = 0u;
    uint16_t materialId = 0u;
    uint32_t textureLayer = 0u;
};

constexpr uint32_t kRandomRotationFlagBit = 1u << 0u;
constexpr uint32_t kRandomOffsetXFlagBit = 1u << 1u;
constexpr uint32_t kRandomOffsetYFlagBit = 1u << 2u;
constexpr uint32_t kRandomOffsetZFlagBit = 1u << 3u;

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

std::optional<MaterialDefinition> MaterialManager::getMaterial(uint16_t materialId) const {
    const auto it = materials_.find(materialId);
    if (it == materials_.end()) {
        return std::nullopt;
    }
    return it->second;
}

uint32_t MaterialManager::textureIndexForMaterial(uint16_t materialId) const {
    if (materialId >= materialMetadata_.size()) {
        return 0u;
    }
    return materialMetadata_[materialId].textureIndex;
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

    std::vector<LoadedMaterialTexture> loadedMaterials;
    loadedMaterials.reserve(configMaterials.size());

    const std::filesystem::path texturesRoot = std::filesystem::path(RESOURCE_DIR) / "textures";
    for (size_t i = 0; i < configMaterials.size(); ++i) {
        const std::string& name = configMaterials[i].name;
        const std::string& textureRelativePath = configMaterials[i].texture;
        const std::filesystem::path texturePath = texturesRoot / textureRelativePath;

        LoadedMaterialTexture loaded{};
        loaded.name = name;
        loaded.textureRelativePath = textureRelativePath;
        loaded.modelRelativePath = configMaterials[i].model;
        loaded.doubleSided = configMaterials[i].doubleSided;
        loaded.randomRotation = configMaterials[i].randomRotation;
        loaded.randomOffsetDirectionsMask = configMaterials[i].randomOffsetDirectionsMask;
        loaded.randomOffsetAmount = configMaterials[i].randomOffsetAmount;
        loaded.blockLightOpacity = configMaterials[i].blockLightOpacity;
        loaded.materialId = static_cast<uint16_t>(kFirstMaterialId + static_cast<uint32_t>(i));
        loaded.textureLayer = static_cast<uint32_t>(i);

        if (!loadPngRgba8(texturePath, loaded.pixels, loaded.width, loaded.height)) {
            std::cerr << "MaterialManager: failed to load material texture '" << texturePath.string()
                      << "' for material '" << name << "'." << std::endl;
            return false;
        }

        loadedMaterials.push_back(std::move(loaded));
    }

    modelManager_.terminate(bufferManager);
    const std::filesystem::path modelsRoot = std::filesystem::path(RESOURCE_DIR) / "models";
    if (!modelManager_.loadModel(kDefaultVoxelModelName, modelsRoot / kDefaultVoxelModelFile)) {
        std::cerr << "MaterialManager: failed to load fallback model '" << kDefaultVoxelModelFile << "'." << std::endl;
        return false;
    }

    std::unordered_set<std::string> seenModelPaths;
    for (const LoadedMaterialTexture& loaded : loadedMaterials) {
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

    for (const LoadedMaterialTexture& material : loadedMaterials) {
        const std::string& modelName = material.modelRelativePath.empty()
            ? std::string(kDefaultVoxelModelName)
            : material.modelRelativePath;
        uint16_t modelIndex = fallbackModelIndex;
        if (!appendModelToLibrary(modelName, modelIndex)) {
            return false;
        }
        blockModels->materialToModel[material.materialId] = modelIndex;
        blockModels->materialDoubleSided[material.materialId] = material.doubleSided ? 1u : 0u;
    }
    blockModelLibrary_ = std::move(blockModels);

    const uint32_t baseWidth = loadedMaterials.front().width;
    const uint32_t baseHeight = loadedMaterials.front().height;
    for (const LoadedMaterialTexture& material : loadedMaterials) {
        if (material.width != baseWidth || material.height != baseHeight) {
            std::cerr << "MaterialManager: texture size mismatch for material '" << material.name
                      << "'. Expected " << baseWidth << "x" << baseHeight
                      << ", got " << material.width << "x" << material.height << "." << std::endl;
            return false;
        }
    }

    TextureDescriptor textureDesc = Default;
    textureDesc.label = StringView("material texture array");
    textureDesc.dimension = TextureDimension::_2D;
    textureDesc.format = TextureFormat::RGBA8Unorm;
    textureDesc.sampleCount = 1;
    textureDesc.size = {baseWidth, baseHeight, static_cast<uint32_t>(loadedMaterials.size())};
    textureDesc.mipLevelCount = mipLevelCount(baseWidth, baseHeight);
    textureDesc.usage = TextureUsage::TextureBinding | TextureUsage::CopyDst;

    Texture texture = textureManager.createTexture(kMaterialTextureArrayName, textureDesc);
    if (!texture) {
        return false;
    }

    for (const LoadedMaterialTexture& material : loadedMaterials) {
        writeMipMapsArrayLayer(
            textureManager,
            texture,
            textureDesc.size,
            textureDesc.mipLevelCount,
            material.textureLayer,
            material.pixels
        );
    }

    TextureViewDescriptor viewDesc = Default;
    viewDesc.label = StringView("material texture array view");
    viewDesc.format = textureDesc.format;
    viewDesc.dimension = TextureViewDimension::_2DArray;
    viewDesc.aspect = TextureAspect::All;
    viewDesc.baseMipLevel = 0;
    viewDesc.mipLevelCount = textureDesc.mipLevelCount;
    viewDesc.baseArrayLayer = 0;
    viewDesc.arrayLayerCount = static_cast<uint32_t>(loadedMaterials.size());

    TextureView textureView = textureManager.createTextureView(
        kMaterialTextureArrayName,
        kMaterialTextureArrayViewName,
        viewDesc
    );
    if (!textureView) {
        return false;
    }

    SamplerDescriptor samplerDesc = Default;
    samplerDesc.label = StringView("material sampler");
    samplerDesc.addressModeU = AddressMode::Repeat;
    samplerDesc.addressModeV = AddressMode::Repeat;
    samplerDesc.addressModeW = AddressMode::ClampToEdge;
    samplerDesc.magFilter = FilterMode::Nearest;
    samplerDesc.minFilter = FilterMode::Nearest;
    samplerDesc.mipmapFilter = MipmapFilterMode::Nearest;
    samplerDesc.maxAnisotropy = 1;
    Sampler sampler = textureManager.createSampler(kMaterialSamplerName, samplerDesc);
    if (!sampler) {
        return false;
    }

    for (const LoadedMaterialTexture& material : loadedMaterials) {
        uint32_t flags = 0u;
        if (material.randomRotation) {
            flags |= kRandomRotationFlagBit;
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
                material.randomRotation,
                material.randomOffsetDirectionsMask,
                material.randomOffsetAmount,
                material.blockLightOpacity
            }
        );
    }

    BufferDescriptor metadataBufferDesc = Default;
    metadataBufferDesc.label = StringView("material metadata buffer");
    metadataBufferDesc.size = static_cast<uint64_t>(materialMetadata_.size()) * sizeof(MaterialMetadataGPU);
    metadataBufferDesc.usage = BufferUsage::Storage | BufferUsage::CopyDst;
    metadataBufferDesc.mappedAtCreation = false;

    Buffer metadataBuffer = bufferManager.createBuffer(kMaterialMetadataBufferName, metadataBufferDesc);
    if (!metadataBuffer) {
        return false;
    }

    bufferManager.writeBuffer(
        kMaterialMetadataBufferName,
        0,
        materialMetadata_.data(),
        materialMetadata_.size() * sizeof(MaterialMetadataGPU)
    );

    return true;
}

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
        if (entry.contains("randomRotation") && !entry["randomRotation"].is_boolean()) {
            std::cerr << "MaterialManager: materials[" << i << "] field 'randomRotation' must be a boolean when present." << std::endl;
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
        if (entry.contains("model")) {
            material.model = entry["model"].get<std::string>();
        }
        if (entry.contains("doubleSided")) {
            material.doubleSided = entry["doubleSided"].get<bool>();
        }
        if (entry.contains("randomRotation")) {
            material.randomRotation = entry["randomRotation"].get<bool>();
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
        outMaterials.push_back(std::move(material));
    }

    return true;
}

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

uint32_t MaterialManager::mipLevelCount(uint32_t width, uint32_t height) {
    uint32_t levels = 1u;
    uint32_t size = std::max(width, height);
    while (size > 1u) {
        size = std::max(1u, size / 2u);
        levels += 1u;
    }
    return levels;
}

void MaterialManager::writeMipMapsArrayLayer(TextureManager& textureManager,
                                             Texture texture,
                                             Extent3D textureSize,
                                             uint32_t mipLevelCount,
                                             uint32_t arrayLayer,
                                             const std::vector<uint8_t>& pixels) {
    TexelCopyTextureInfo destination{};
    destination.texture = texture;
    destination.origin = {0u, 0u, arrayLayer};
    destination.aspect = TextureAspect::All;

    TexelCopyBufferLayout source{};
    source.offset = 0;

    Extent3D mipLevelSize = textureSize;
    std::vector<uint8_t> previousLevelPixels;
    Extent3D previousMipLevelSize{};

    for (uint32_t level = 0; level < mipLevelCount; ++level) {
        std::vector<uint8_t> mipPixels(
            static_cast<size_t>(4u) * static_cast<size_t>(mipLevelSize.width) * static_cast<size_t>(mipLevelSize.height)
        );
        if (level == 0u) {
            mipPixels = pixels;
        } else {
            for (uint32_t x = 0; x < mipLevelSize.width; ++x) {
                for (uint32_t y = 0; y < mipLevelSize.height; ++y) {
                    uint8_t* p = &mipPixels[4u * (static_cast<size_t>(y) * mipLevelSize.width + x)];
                    const uint32_t srcX0 = std::min(2u * x + 0u, previousMipLevelSize.width - 1u);
                    const uint32_t srcX1 = std::min(2u * x + 1u, previousMipLevelSize.width - 1u);
                    const uint32_t srcY0 = std::min(2u * y + 0u, previousMipLevelSize.height - 1u);
                    const uint32_t srcY1 = std::min(2u * y + 1u, previousMipLevelSize.height - 1u);

                    const uint8_t* p00 = &previousLevelPixels[4u * (static_cast<size_t>(srcY0) * previousMipLevelSize.width + srcX0)];
                    const uint8_t* p01 = &previousLevelPixels[4u * (static_cast<size_t>(srcY0) * previousMipLevelSize.width + srcX1)];
                    const uint8_t* p10 = &previousLevelPixels[4u * (static_cast<size_t>(srcY1) * previousMipLevelSize.width + srcX0)];
                    const uint8_t* p11 = &previousLevelPixels[4u * (static_cast<size_t>(srcY1) * previousMipLevelSize.width + srcX1)];

                    const float a00 = p00[3] / 255.0f;
                    const float a01 = p01[3] / 255.0f;
                    const float a10 = p10[3] / 255.0f;
                    const float a11 = p11[3] / 255.0f;
                    const float avgA = (a00 + a01 + a10 + a11) / 4.0f;
                    const uint8_t finalA = (avgA >= 0.5f) ? 255u : 0u;

                    if (finalA > 0u) {
                        float total = 0.0f;
                        float wr = 0.0f;
                        float wg = 0.0f;
                        float wb = 0.0f;
                        auto acc = [&](const uint8_t* s, float a) {
                            if (a >= 0.5f) {
                                wr += s[0] * a;
                                wg += s[1] * a;
                                wb += s[2] * a;
                                total += a;
                            }
                        };
                        acc(p00, a00);
                        acc(p01, a01);
                        acc(p10, a10);
                        acc(p11, a11);
                        if (total > 0.0f) {
                            p[0] = static_cast<uint8_t>(wr / total);
                            p[1] = static_cast<uint8_t>(wg / total);
                            p[2] = static_cast<uint8_t>(wb / total);
                        } else {
                            p[0] = static_cast<uint8_t>((p00[0] + p01[0] + p10[0] + p11[0]) / 4u);
                            p[1] = static_cast<uint8_t>((p00[1] + p01[1] + p10[1] + p11[1]) / 4u);
                            p[2] = static_cast<uint8_t>((p00[2] + p01[2] + p10[2] + p11[2]) / 4u);
                        }
                    } else {
                        p[0] = p[1] = p[2] = 0u;
                    }
                    p[3] = finalA;
                }
            }
        }

        destination.mipLevel = level;
        source.bytesPerRow = 4u * mipLevelSize.width;
        source.rowsPerImage = mipLevelSize.height;

        Extent3D writeSize = mipLevelSize;
        writeSize.depthOrArrayLayers = 1u;

        textureManager.writeTexture(destination, mipPixels.data(), mipPixels.size(), source, writeSize);

        previousLevelPixels = std::move(mipPixels);
        previousMipLevelSize = mipLevelSize;
        mipLevelSize.width = std::max(1u, mipLevelSize.width / 2u);
        mipLevelSize.height = std::max(1u, mipLevelSize.height / 2u);
    }
}
