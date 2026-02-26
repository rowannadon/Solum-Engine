#include "solum_engine/render/MaterialManager.h"

#include <algorithm>
#include <fstream>
#include <iostream>

#include "lodepng/lodepng.h"
#include "nlohmann_json/json.hpp"

using json = nlohmann::json;
using namespace wgpu;

namespace {
constexpr uint32_t kFirstMaterialId = 1u;

struct LoadedMaterialTexture {
    std::string name;
    std::string textureRelativePath;
    std::vector<uint8_t> pixels;
    uint32_t width = 0u;
    uint32_t height = 0u;
    uint16_t materialId = 0u;
    uint32_t textureLayer = 0u;
};
}  // namespace

bool MaterialManager::initialize(BufferManager& bufferManager, TextureManager& textureManager) {
    if (initialized_) {
        return true;
    }

    materialLookup_.assign(kLookupEntryCount, 0u);
    materials_.clear();

    if (!buildDefaultMaterials(bufferManager, textureManager)) {
        return false;
    }

    initialized_ = true;
    return true;
}

void MaterialManager::terminate(BufferManager& bufferManager, TextureManager& textureManager) {
    if (!initialized_) {
        return;
    }

    bufferManager.deleteBuffer(kMaterialLookupBufferName);
    textureManager.removeTextureView(kMaterialTextureArrayViewName);
    textureManager.removeTexture(kMaterialTextureArrayName);
    textureManager.removeSampler(kMaterialSamplerName);

    materials_.clear();
    materialLookup_.clear();
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
    if (materialId >= materialLookup_.size()) {
        return 0u;
    }
    return materialLookup_[materialId];
}

bool MaterialManager::buildDefaultMaterials(BufferManager& bufferManager, TextureManager& textureManager) {
    const std::filesystem::path materialConfigPath = std::filesystem::path(RESOURCE_DIR) / "materials.json";

    std::vector<std::pair<std::string, std::string>> configMaterials;
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
        const std::string& name = configMaterials[i].first;
        const std::string& textureRelativePath = configMaterials[i].second;
        const std::filesystem::path texturePath = texturesRoot / textureRelativePath;

        LoadedMaterialTexture loaded{};
        loaded.name = name;
        loaded.textureRelativePath = textureRelativePath;
        loaded.materialId = static_cast<uint16_t>(kFirstMaterialId + static_cast<uint32_t>(i));
        loaded.textureLayer = static_cast<uint32_t>(i);

        if (!loadPngRgba8(texturePath, loaded.pixels, loaded.width, loaded.height)) {
            std::cerr << "MaterialManager: failed to load material texture '" << texturePath.string()
                      << "' for material '" << name << "'." << std::endl;
            return false;
        }

        loadedMaterials.push_back(std::move(loaded));
    }

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
        materialLookup_[material.materialId] = material.textureLayer;
        materials_.emplace(
            material.materialId,
            MaterialDefinition{
                material.materialId,
                material.name,
                material.textureLayer,
                1.0f,
                0.0f
            }
        );
    }

    BufferDescriptor lookupBufferDesc = Default;
    lookupBufferDesc.label = StringView("material lookup buffer");
    lookupBufferDesc.size = static_cast<uint64_t>(materialLookup_.size()) * sizeof(uint32_t);
    lookupBufferDesc.usage = BufferUsage::Storage | BufferUsage::CopyDst;
    lookupBufferDesc.mappedAtCreation = false;

    Buffer lookupBuffer = bufferManager.createBuffer(kMaterialLookupBufferName, lookupBufferDesc);
    if (!lookupBuffer) {
        return false;
    }

    bufferManager.writeBuffer(
        kMaterialLookupBufferName,
        0,
        materialLookup_.data(),
        materialLookup_.size() * sizeof(uint32_t)
    );

    return true;
}

bool MaterialManager::loadMaterialConfig(const std::filesystem::path& path,
                                         std::vector<std::pair<std::string, std::string>>& outMaterials) {
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

        outMaterials.emplace_back(entry["name"].get<std::string>(), entry["texture"].get<std::string>());
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
