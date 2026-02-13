#include "solum_engine/render/TextureManager.h"
#include <cstring>
#include <iostream>
#include "stb_image.h"
#include <glm/glm.hpp>

using std::uint32_t;

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
    for (auto& kv : textures) {
        if (kv.second) {
            kv.second.destroy();
            kv.second.release();
        }
    }
}

uint32_t TextureManager::bit_width(uint32_t m) {
    if (m == 0) return 0;
    uint32_t w = 0;
    while (m >>= 1) ++w;
    return w;
}

// ---------- single texture ----------
Texture TextureManager::loadTexture(const std::string name, const std::string textureViewName, const std::filesystem::path& path, TextureFormat format) {
    int width, height, channels;
    void* pixelData = nullptr;

    if (format == TextureFormat::RGBA16Unorm) {
        pixelData = stbi_load_16(path.string().c_str(), &width, &height, &channels, 4); // 4 channels
    }
    else {
        pixelData = stbi_load(path.string().c_str(), &width, &height, &channels, 4);
    }

    if (!pixelData) return nullptr;

    TextureDescriptor textureDesc{};
    textureDesc.dimension = TextureDimension::_2D;
    textureDesc.format = format;
    textureDesc.sampleCount = 1;
    textureDesc.size = { (unsigned int)width, (unsigned int)height, 1 };
    textureDesc.mipLevelCount = bit_width(std::max(textureDesc.size.width, textureDesc.size.height));
    textureDesc.usage = TextureUsage::TextureBinding | TextureUsage::CopyDst;

    Texture texture = createTexture(name, textureDesc);
    writeMipMaps(texture, textureDesc.size, textureDesc.mipLevelCount, pixelData, format);

    // Free memory based on format
    if (format == TextureFormat::RGBA16Unorm) {
        stbi_image_free((unsigned short*)pixelData);
    }
    else {
        stbi_image_free((unsigned char*)pixelData);
    }

    if (!textureViewName.empty()) {
        TextureViewDescriptor vd{};
        vd.aspect = TextureAspect::All;
        vd.baseArrayLayer = 0;
        vd.arrayLayerCount = 1;
        vd.baseMipLevel = 0;
        vd.mipLevelCount = textureDesc.mipLevelCount;
        vd.dimension = TextureViewDimension::_2D;
        vd.format = textureDesc.format;
        createTextureView(name, textureViewName, vd);
    }

    return texture;
}

void TextureManager::writeMipMaps(
    Texture texture,
    Extent3D textureSize,
    uint32_t mipLevelCount,
    const void* rawPixelData,
    TextureFormat format)
{
    TexelCopyTextureInfo destination{};
    destination.texture = texture;
    destination.origin = { 0, 0, 0 };
    destination.aspect = TextureAspect::All;

    TexelCopyBufferLayout source{};
    source.offset = 0;

    Extent3D mipLevelSize = textureSize;

    if (format == TextureFormat::RGBA8Unorm || format == TextureFormat::RGBA8UnormSrgb) {
        const unsigned char* pixelData = static_cast<const unsigned char*>(rawPixelData);
        std::vector<unsigned char> previousLevelPixels;
        Extent3D previousMipLevelSize;

        for (uint32_t level = 0; level < mipLevelCount; ++level) {
            std::vector<unsigned char> pixels(4 * mipLevelSize.width * mipLevelSize.height);
            if (level == 0) {
                memcpy(pixels.data(), pixelData, pixels.size());
            }
            else {
                for (uint32_t i = 0; i < mipLevelSize.width; ++i) {
                    for (uint32_t j = 0; j < mipLevelSize.height; ++j) {
                        unsigned char* p = &pixels[4 * (j * mipLevelSize.width + i)];
                        unsigned char* p00 = &previousLevelPixels[4 * ((2 * j + 0) * previousMipLevelSize.width + (2 * i + 0))];
                        unsigned char* p01 = &previousLevelPixels[4 * ((2 * j + 0) * previousMipLevelSize.width + (2 * i + 1))];
                        unsigned char* p10 = &previousLevelPixels[4 * ((2 * j + 1) * previousMipLevelSize.width + (2 * i + 0))];
                        unsigned char* p11 = &previousLevelPixels[4 * ((2 * j + 1) * previousMipLevelSize.width + (2 * i + 1))];

                        float a00 = p00[3] / 255.0f, a01 = p01[3] / 255.0f, a10 = p10[3] / 255.0f, a11 = p11[3] / 255.0f;
                        float avgA = (a00 + a01 + a10 + a11) / 4.0f;
                        unsigned char finalA = (avgA >= 0.5f) ? 255 : 0;

                        if (finalA > 0) {
                            float total = 0.f, wr = 0.f, wg = 0.f, wb = 0.f;
                            auto acc = [&](unsigned char* s, float a) {
                                if (a >= 0.5f) {
                                    wr += s[0] * a;
                                    wg += s[1] * a;
                                    wb += s[2] * a;
                                    total += a;
                                }
                                };
                            acc(p00, a00); acc(p01, a01); acc(p10, a10); acc(p11, a11);
                            if (total > 0) {
                                p[0] = (unsigned char)(wr / total);
                                p[1] = (unsigned char)(wg / total);
                                p[2] = (unsigned char)(wb / total);
                            }
                            else {
                                p[0] = (p00[0] + p01[0] + p10[0] + p11[0]) / 4;
                                p[1] = (p00[1] + p01[1] + p10[1] + p11[1]) / 4;
                                p[2] = (p00[2] + p01[2] + p10[2] + p11[2]) / 4;
                            }
                        }
                        else {
                            p[0] = p[1] = p[2] = 0;
                        }
                        p[3] = finalA;
                    }
                }
            }

            destination.mipLevel = level;
            source.bytesPerRow = 4 * mipLevelSize.width;
            source.rowsPerImage = mipLevelSize.height;
            queue.writeTexture(destination, pixels.data(), pixels.size(), source, mipLevelSize);

            previousLevelPixels = std::move(pixels);
            previousMipLevelSize = mipLevelSize;
            mipLevelSize.width = std::max(1u, mipLevelSize.width / 2);
            mipLevelSize.height = std::max(1u, mipLevelSize.height / 2);
        }

    }
    else if (format == TextureFormat::RGBA16Unorm) {
        const uint16_t* pixelData = static_cast<const uint16_t*>(rawPixelData);
        std::vector<uint16_t> previousLevelPixels;
        Extent3D previousMipLevelSize;

        for (uint32_t level = 0; level < mipLevelCount; ++level) {
            std::vector<uint16_t> pixels(4 * mipLevelSize.width * mipLevelSize.height);
            if (level == 0) {
                memcpy(pixels.data(), pixelData, pixels.size() * sizeof(uint16_t));
            }
            else {
                for (uint32_t i = 0; i < mipLevelSize.width; ++i) {
                    for (uint32_t j = 0; j < mipLevelSize.height; ++j) {
                        uint16_t* p = &pixels[4 * (j * mipLevelSize.width + i)];
                        uint16_t* p00 = &previousLevelPixels[4 * ((2 * j + 0) * previousMipLevelSize.width + (2 * i + 0))];
                        uint16_t* p01 = &previousLevelPixels[4 * ((2 * j + 0) * previousMipLevelSize.width + (2 * i + 1))];
                        uint16_t* p10 = &previousLevelPixels[4 * ((2 * j + 1) * previousMipLevelSize.width + (2 * i + 0))];
                        uint16_t* p11 = &previousLevelPixels[4 * ((2 * j + 1) * previousMipLevelSize.width + (2 * i + 1))];

                        float a00 = p00[3] / 65535.0f, a01 = p01[3] / 65535.0f, a10 = p10[3] / 65535.0f, a11 = p11[3] / 65535.0f;
                        float avgA = (a00 + a01 + a10 + a11) / 4.0f;
                        uint16_t finalA = (avgA >= 0.5f) ? 65535 : 0;

                        if (finalA > 0) {
                            float total = 0.f, wr = 0.f, wg = 0.f, wb = 0.f;
                            auto acc = [&](uint16_t* s, float a) {
                                if (a >= 0.5f) {
                                    wr += s[0] * a;
                                    wg += s[1] * a;
                                    wb += s[2] * a;
                                    total += a;
                                }
                                };
                            acc(p00, a00); acc(p01, a01); acc(p10, a10); acc(p11, a11);
                            if (total > 0) {
                                p[0] = uint16_t(wr / total);
                                p[1] = uint16_t(wg / total);
                                p[2] = uint16_t(wb / total);
                            }
                            else {
                                p[0] = (p00[0] + p01[0] + p10[0] + p11[0]) / 4;
                                p[1] = (p00[1] + p01[1] + p10[1] + p11[1]) / 4;
                                p[2] = (p00[2] + p01[2] + p10[2] + p11[2]) / 4;
                            }
                        }
                        else {
                            p[0] = p[1] = p[2] = 0;
                        }
                        p[3] = finalA;
                    }
                }
            }

            destination.mipLevel = level;
            source.bytesPerRow = 4 * mipLevelSize.width * sizeof(uint16_t);
            source.rowsPerImage = mipLevelSize.height;

            queue.writeTexture(destination, pixels.data(), pixels.size() * sizeof(uint16_t), source, mipLevelSize);

            previousLevelPixels = std::move(pixels);
            previousMipLevelSize = mipLevelSize;
            mipLevelSize.width = std::max(1u, mipLevelSize.width / 2);
            mipLevelSize.height = std::max(1u, mipLevelSize.height / 2);
        }

    }
    else {
        std::cerr << "[TextureManager] Unsupported texture format in writeMipMaps.\n";
    }
}

void TextureManager::writeMipMapsArray(
    Texture texture,
    Extent3D textureSize,
    uint32_t mipLevelCount,
    uint32_t arrayLayer,
    const unsigned char* pixelData)
{
    TexelCopyTextureInfo destination{};
    destination.texture = texture;
    destination.origin = { 0, 0, arrayLayer };
    destination.aspect = TextureAspect::All;

    TexelCopyBufferLayout source{};
    source.offset = 0;

    Extent3D mipLevelSize = textureSize;
    std::vector<unsigned char> previousLevelPixels;
    Extent3D previousMipLevelSize;

    for (uint32_t level = 0; level < mipLevelCount; ++level) {
        std::vector<unsigned char> pixels(4 * mipLevelSize.width * mipLevelSize.height);
        if (level == 0) {
            memcpy(pixels.data(), pixelData, pixels.size());
        }
        else {
            for (uint32_t i = 0; i < mipLevelSize.width; ++i) {
                for (uint32_t j = 0; j < mipLevelSize.height; ++j) {
                    unsigned char* p = &pixels[4 * (j * mipLevelSize.width + i)];
                    unsigned char* p00 = &previousLevelPixels[4 * ((2 * j + 0) * previousMipLevelSize.width + (2 * i + 0))];
                    unsigned char* p01 = &previousLevelPixels[4 * ((2 * j + 0) * previousMipLevelSize.width + (2 * i + 1))];
                    unsigned char* p10 = &previousLevelPixels[4 * ((2 * j + 1) * previousMipLevelSize.width + (2 * i + 0))];
                    unsigned char* p11 = &previousLevelPixels[4 * ((2 * j + 1) * previousMipLevelSize.width + (2 * i + 1))];

                    float a00 = p00[3] / 255.0f, a01 = p01[3] / 255.0f, a10 = p10[3] / 255.0f, a11 = p11[3] / 255.0f;
                    float avgA = (a00 + a01 + a10 + a11) / 4.0f;
                    unsigned char finalA = (avgA >= 0.5f) ? 255 : 0;

                    if (finalA > 0) {
                        float total = 0.f, wr = 0.f, wg = 0.f, wb = 0.f;
                        auto acc = [&](unsigned char* s, float a) { if (a >= 0.5f) { wr += s[0] * a; wg += s[1] * a; wb += s[2] * a; total += a; } };
                        acc(p00, a00); acc(p01, a01); acc(p10, a10); acc(p11, a11);
                        if (total > 0) { p[0] = (unsigned char)(wr / total); p[1] = (unsigned char)(wg / total); p[2] = (unsigned char)(wb / total); }
                        else { p[0] = (p00[0] + p01[0] + p10[0] + p11[0]) / 4; p[1] = (p00[1] + p01[1] + p10[1] + p11[1]) / 4; p[2] = (p00[2] + p01[2] + p10[2] + p11[2]) / 4; }
                    }
                    else { p[0] = p[1] = p[2] = 0; }
                    p[3] = finalA;
                }
            }
        }
        destination.mipLevel = level;
        source.bytesPerRow = 4 * mipLevelSize.width;
        source.rowsPerImage = mipLevelSize.height;
        Extent3D writeSize = mipLevelSize; writeSize.depthOrArrayLayers = 1;
        queue.writeTexture(destination, pixels.data(), pixels.size(), source, writeSize);

        previousLevelPixels = std::move(pixels);
        previousMipLevelSize = mipLevelSize;
        mipLevelSize.width = std::max(1u, mipLevelSize.width / 2);
        mipLevelSize.height = std::max(1u, mipLevelSize.height / 2);
    }
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
