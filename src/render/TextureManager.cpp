#include "solum_engine/render/TextureManager.h"

// ---------- basic IO ----------
void TextureManager::writeTexture(const wgpu::TexelCopyTextureInfo& destination,
    const void* data, size_t size,
    const wgpu::TexelCopyBufferLayout& source,
    const wgpu::Extent3D& writeSize) {
    queue.writeTexture(destination, data, size, source, writeSize);
}

wgpu::Texture TextureManager::getTexture(const std::string& textureName) const {
    auto it = textures.find(textureName);
    return it != textures.end() ? it->second : nullptr;
}
wgpu::TextureView TextureManager::getTextureView(const std::string& viewName) const {
    auto it = textureViews.find(viewName);
    return it != textureViews.end() ? it->second : nullptr;
}
wgpu::Sampler TextureManager::getSampler(const std::string& samplerName) const {
    auto it = samplers.find(samplerName);
    return it != samplers.end() ? it->second : nullptr;
}
wgpu::Texture TextureManager::createTexture(const std::string& name, const wgpu::TextureDescriptor& config) {
    removeTexture(name);
    wgpu::Texture texture = device.createTexture(config);
    textures[name] = texture;
    return texture;
}
wgpu::TextureView TextureManager::createTextureView(const std::string& textureName,
                                                    const std::string& viewName,
                                                    const wgpu::TextureViewDescriptor& config) {
    removeTextureView(viewName);
    auto it = textures.find(textureName);
    if (it == textures.end()) return nullptr;
    wgpu::TextureView view = it->second.createView(config);
    textureViews[viewName] = view;
    return view;
}
wgpu::Sampler TextureManager::createSampler(const std::string& samplerName, const wgpu::SamplerDescriptor& config) {
    removeSampler(samplerName);
    wgpu::Sampler sampler = device.createSampler(config);
    samplers[samplerName] = sampler;
    return sampler;
}

void TextureManager::terminate() {
    for (auto& kv : textureViews) {
        if (kv.second) {
            kv.second.release();
        }
    }

    for (auto& kv : samplers) {
        if (kv.second) {
            kv.second.release();
        }
    }

    for (auto& kv : textures) {
        if (kv.second) {
            kv.second.destroy();
            kv.second.release();
        }
    }

    textureViews.clear();
    samplers.clear();
    textures.clear();
}

// ---------- cleanup ----------
void TextureManager::removeTextureView(const std::string& name) {
    auto it = textureViews.find(name);
    if (it != textureViews.end()) {
        it->second.release();
        textureViews.erase(it);
    }
}

void TextureManager::removeTexture(const std::string& name) {
    auto it = textures.find(name);
    if (it != textures.end()) {
        it->second.destroy();
        it->second.release();
        textures.erase(it);
    }
}

void TextureManager::removeSampler(const std::string& name) {
    auto it = samplers.find(name);
    if (it != samplers.end()) {
        it->second.release();
        samplers.erase(it);
    }
}
