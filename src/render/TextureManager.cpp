#include "solum_engine/render/TextureManager.h"

// ---------- basic IO ----------
void TextureManager::writeTexture(const TexelCopyTextureInfo& destination,
    const void* data, size_t size,
    const TexelCopyBufferLayout& source,
    const Extent3D& writeSize) {
    queue.writeTexture(destination, data, size, source, writeSize);
}

Texture TextureManager::getTexture(const std::string textureName) {
    auto it = textures.find(textureName);
    return it != textures.end() ? it->second : nullptr;
}
TextureView TextureManager::getTextureView(const std::string viewName) {
    auto it = textureViews.find(viewName);
    return it != textureViews.end() ? it->second : nullptr;
}
Sampler TextureManager::getSampler(const std::string samplerName) {
    auto it = samplers.find(samplerName);
    return it != samplers.end() ? it->second : nullptr;
}
Texture TextureManager::createTexture(const std::string& name, const TextureDescriptor& config) {
    Texture texture = device.createTexture(config);
    textures[name] = texture;
    return texture;
}
TextureView TextureManager::createTextureView(const std::string& textureName, const std::string& viewName, const TextureViewDescriptor& config) {
    auto it = textures.find(textureName);
    if (it == textures.end()) return nullptr;
    TextureView view = it->second.createView(config);
    textureViews[viewName] = view;
    return view;
}
Sampler TextureManager::createSampler(const std::string& samplerName, const SamplerDescriptor& config) {
    Sampler sampler = device.createSampler(config);
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
