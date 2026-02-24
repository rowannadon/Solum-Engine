#ifndef TEXTURE_MANAGER
#define TEXTURE_MANAGER
#include <unordered_map>
#include <webgpu/webgpu.hpp>
#include <filesystem>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <set>
#include <memory>
#include <shared_mutex>
#include <tuple>
#include <optional>
#include <unordered_set>
#include "nlohmann_json/json.hpp" // nlohmann/json

using json = nlohmann::json;
using namespace wgpu;

class TextureManager {
    std::unordered_map<std::string, Texture> textures;
    std::unordered_map<std::string, TextureView> textureViews;
    std::unordered_map<std::string, Sampler> samplers;
    // New: store materials for the most recent/active array load (or keyed by name if desired)

    mutable std::shared_mutex textureMutex;

    Device device;
    Queue queue;
public:
    TextureManager(Device d, Queue q) : device(d), queue(q) {}
    Texture createTexture(const std::string& name, const TextureDescriptor& config);
    TextureView createTextureView(const std::string& textureName, const std::string& viewName, const TextureViewDescriptor& config);
    Sampler createSampler(const std::string& samplerName, const SamplerDescriptor& config);

    Texture loadTexture(const std::string name, const std::string textureViewName,
        const std::filesystem::path& path, TextureFormat format);

    Texture getTexture(const std::string textureName);
    TextureView getTextureView(const std::string viewName);
    Sampler getSampler(const std::string samplerName);

    void writeTexture(const TexelCopyTextureInfo& destination, const void* data, size_t size, const TexelCopyBufferLayout& source, const Extent3D& writeSize);
    void removeTextureView(const std::string& name);
    void removeTexture(const std::string& name);
    void removeSampler(const std::string& name);
    void terminate();

private:
    uint32_t bit_width(uint32_t m);
    void writeMipMaps(
        Texture texture,
        Extent3D textureSize,
        uint32_t mipLevelCount,
        const void* rawPixelData,
        TextureFormat format);
    void writeMipMapsArray(Texture texture, Extent3D textureSize, uint32_t mipLevelCount, uint32_t arrayLayer, const unsigned char* pixelData);

};
#endif
