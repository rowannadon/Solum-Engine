#include "solum_engine/render/MaterialManager.h"

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

using namespace wgpu;

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

bool MaterialManager::uploadMaterialGpuResources(BufferManager& bufferManager,
                                                 TextureManager& textureManager,
                                                 const std::vector<LoadedMaterialAsset>& loadedMaterials) {
    if (loadedMaterials.empty()) {
        return false;
    }

    const uint32_t baseWidth = loadedMaterials.front().width;
    const uint32_t baseHeight = loadedMaterials.front().height;
    for (const LoadedMaterialAsset& material : loadedMaterials) {
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

    for (const LoadedMaterialAsset& material : loadedMaterials) {
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
