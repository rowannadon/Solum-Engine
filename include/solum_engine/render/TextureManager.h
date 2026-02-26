#ifndef TEXTURE_MANAGER
#define TEXTURE_MANAGER
#include <cstdint>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <webgpu/webgpu.hpp>

class TextureManager {
    std::unordered_map<std::string, wgpu::Texture> textures;
    std::unordered_map<std::string, wgpu::TextureView> textureViews;
    std::unordered_map<std::string, wgpu::Sampler> samplers;

    wgpu::Device device;
    wgpu::Queue queue;
public:
    TextureManager(wgpu::Device d, wgpu::Queue q) : device(d), queue(q) {}
    wgpu::Texture createTexture(const std::string& name, const wgpu::TextureDescriptor& config);
    wgpu::TextureView createTextureView(const std::string& textureName, const std::string& viewName, const wgpu::TextureViewDescriptor& config);
    wgpu::Sampler createSampler(const std::string& samplerName, const wgpu::SamplerDescriptor& config);

    wgpu::Texture getTexture(const std::string& textureName) const;
    wgpu::TextureView getTextureView(const std::string& viewName) const;
    wgpu::Sampler getSampler(const std::string& samplerName) const;

    void writeTexture(const wgpu::TexelCopyTextureInfo& destination,
                      const void* data,
                      size_t size,
                      const wgpu::TexelCopyBufferLayout& source,
                      const wgpu::Extent3D& writeSize);
    void removeTextureView(const std::string& name);
    void removeTexture(const std::string& name);
    void removeSampler(const std::string& name);
    void terminate();
};
#endif
