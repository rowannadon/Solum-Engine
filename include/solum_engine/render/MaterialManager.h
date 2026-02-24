#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "solum_engine/render/BufferManager.h"
#include "solum_engine/render/TextureManager.h"

struct MaterialDefinition {
    uint16_t materialId = 0;
    std::string name;
    uint32_t textureIndex = 0;
    float roughness = 1.0f;
    float metallic = 0.0f;
};

class MaterialManager {
public:
    static constexpr uint32_t kMaxMaterialId = 65535u;
    static constexpr uint32_t kLookupEntryCount = kMaxMaterialId + 1u;

    static constexpr const char* kMaterialLookupBufferName = "material_lookup_buffer";
    static constexpr const char* kMaterialTextureArrayName = "material_texture_array";
    static constexpr const char* kMaterialTextureArrayViewName = "material_texture_array_view";
    static constexpr const char* kMaterialSamplerName = "material_sampler";

    bool initialize(BufferManager& bufferManager, TextureManager& textureManager);
    void terminate(BufferManager& bufferManager, TextureManager& textureManager);

    std::optional<MaterialDefinition> getMaterial(uint16_t materialId) const;
    uint32_t textureIndexForMaterial(uint16_t materialId) const;

private:
    bool buildDefaultMaterials(BufferManager& bufferManager, TextureManager& textureManager);
    static bool loadPngRgba8(const std::filesystem::path& path,
                             std::vector<uint8_t>& outPixels,
                             uint32_t& outWidth,
                             uint32_t& outHeight);
    static uint32_t mipLevelCount(uint32_t width, uint32_t height);
    static void writeMipMapsArrayLayer(TextureManager& textureManager,
                                       Texture texture,
                                       Extent3D textureSize,
                                       uint32_t mipLevelCount,
                                       uint32_t arrayLayer,
                                       const std::vector<uint8_t>& pixels);

    std::unordered_map<uint16_t, MaterialDefinition> materials_;
    std::vector<uint32_t> materialLookup_;
    bool initialized_ = false;
};
