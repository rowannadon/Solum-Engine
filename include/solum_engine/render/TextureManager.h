#ifndef TEXTURE_MANAGER
#define TEXTURE_MANAGER
#include <cstddef>
#include <string>
#include <unordered_map>
#include <webgpu/webgpu.hpp>
using namespace wgpu;

class TextureManager {
    std::unordered_map<std::string, Texture> textures;
    std::unordered_map<std::string, TextureView> textureViews;
    std::unordered_map<std::string, Sampler> samplers;

    Device device;
    Queue queue;
public:
    TextureManager(Device d, Queue q) : device(d), queue(q) {}
    Texture createTexture(const std::string& name, const TextureDescriptor& config);
    TextureView createTextureView(const std::string& textureName, const std::string& viewName, const TextureViewDescriptor& config);
    Sampler createSampler(const std::string& samplerName, const SamplerDescriptor& config);

    Texture getTexture(const std::string textureName);
    TextureView getTextureView(const std::string viewName);
    Sampler getSampler(const std::string samplerName);

    void writeTexture(const TexelCopyTextureInfo& destination, const void* data, size_t size, const TexelCopyBufferLayout& source, const Extent3D& writeSize);
    void removeTextureView(const std::string& name);
    void removeTexture(const std::string& name);
    void removeSampler(const std::string& name);
    void terminate();
};
#endif
