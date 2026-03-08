#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "solum_engine/render/BufferManager.h"
#include "solum_engine/render/ModelManager.h"
#include "solum_engine/render/TextureManager.h"
#include "solum_engine/voxel/BlockModelLibrary.h"

struct MaterialDefinition {
    uint16_t materialId = 0;
    std::string name;
    uint32_t textureIndex = 0;
    float roughness = 1.0f;
    float metallic = 0.0f;
    bool doubleSided = false;
    bool randomTextureRotation = false;
    bool randomRotation = false;
    uint8_t randomRotationDirectionsMask = 0x7u;
    uint8_t randomOffsetDirectionsMask = 0u;
    float randomOffsetAmount = 0.0f;
    float blockLightOpacity = 1.0f;
    uint8_t emissiveLight = 0u;
    bool aoOccluder = true;
};

class MaterialManager {
public:
    static constexpr uint32_t kMaxMaterialId = 65535u;
    static constexpr uint32_t kLookupEntryCount = kMaxMaterialId + 1u;

    struct MaterialMetadataGPU {
        uint32_t textureIndex = 0u;
        uint32_t flags = 0u;
        float randomOffsetAmount = 0.0f;
        float pad0 = 0.0f;
    };
    static_assert(sizeof(MaterialMetadataGPU) == 16, "Material metadata GPU layout must stay tightly packed.");

    static constexpr const char* kMaterialMetadataBufferName = "material_metadata_buffer";
    static constexpr const char* kMaterialTextureArrayName = "material_texture_array";
    static constexpr const char* kMaterialTextureArrayViewName = "material_texture_array_view";
    static constexpr const char* kMaterialSamplerName = "material_sampler";

    bool initialize(BufferManager& bufferManager, TextureManager& textureManager);
    void terminate(BufferManager& bufferManager, TextureManager& textureManager);

    std::shared_ptr<const BlockModelLibrary> blockModelLibrary() const;

private:
    struct MaterialConfigEntry {
        std::string name;
        std::string texture;
        std::string model;
        bool doubleSided = false;
        bool randomTextureRotation = false;
        bool randomRotation = false;
        uint8_t randomRotationDirectionsMask = 0x7u;
        uint8_t randomOffsetDirectionsMask = 0u;
        float randomOffsetAmount = 0.0f;
        float blockLightOpacity = 1.0f;
        uint8_t emissiveLight = 0u;
        bool aoOccluder = true;
    };

    bool buildDefaultMaterials(BufferManager& bufferManager, TextureManager& textureManager);
    static bool loadMaterialConfig(const std::filesystem::path& path,
                                   std::vector<MaterialConfigEntry>& outMaterials);
    static bool loadPngRgba8(const std::filesystem::path& path,
                             std::vector<uint8_t>& outPixels,
                             uint32_t& outWidth,
                             uint32_t& outHeight);
    static uint32_t mipLevelCount(uint32_t width, uint32_t height);
    static void writeMipMapsArrayLayer(TextureManager& textureManager,
                                       wgpu::Texture texture,
                                       wgpu::Extent3D textureSize,
                                       uint32_t mipLevelCount,
                                       uint32_t arrayLayer,
                                       const std::vector<uint8_t>& pixels);

    std::unordered_map<uint16_t, MaterialDefinition> materials_;
    std::vector<MaterialMetadataGPU> materialMetadata_;
    ModelManager modelManager_{};
    std::shared_ptr<BlockModelLibrary> blockModelLibrary_{};
    bool initialized_ = false;
};
