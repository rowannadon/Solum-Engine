#include "solum_engine/render/WebGPURenderer.h"
#include "solum_engine/voxel/World.h"
#include "lodepng.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <unordered_set>
#include <vector>

namespace {
constexpr int kLodCount = kRegionLodCount;

constexpr std::array<uint64_t, 6> kSizeClasses = {
    64ull * 1024ull,
    256ull * 1024ull,
    1024ull * 1024ull,
    4ull * 1024ull * 1024ull,
    16ull * 1024ull * 1024ull,
    64ull * 1024ull * 1024ull,
};
constexpr std::array<uint32_t, 6> kSlotsPerPage = { 64u, 32u, 16u, 8u, 2u, 1u };

constexpr uint16_t kUvMin = 0;
constexpr uint16_t kUvMax = 65535;

float sampleFallbackHeight01(int blockX, int blockY) {
    const float x = static_cast<float>(blockX);
    const float y = static_cast<float>(blockY);

    const float waveA = std::sin(x * 0.0065f) * 140.0f;
    const float waveB = std::cos(y * 0.0081f) * 120.0f;
    const float waveC = std::sin((x + y) * 0.0037f) * 80.0f;
    const float waveD = std::cos((x - y) * 0.0049f) * 48.0f;

    const float normalized = 0.5f + (waveA + waveB + waveC + waveD) / 800.0f;
    return std::clamp(normalized, 0.0f, 1.0f);
}

uint8_t encodeNormalComponent(int component) {
    if (component > 0) {
        return 255;
    }
    if (component < 0) {
        return 0;
    }
    return 127;
}

void emitFace(
    std::vector<VertexAttributes>& vertices,
    std::vector<uint32_t>& indices,
    const std::array<glm::ivec3, 4>& corners,
    const glm::ivec3& normal,
    uint16_t materialId,
    uint8_t lodLevel
) {
    const uint32_t base = static_cast<uint32_t>(vertices.size());

    VertexAttributes v0{};
    v0.x = corners[0].x;
    v0.y = corners[0].y;
    v0.z = corners[0].z;
    v0.u = kUvMin;
    v0.v = kUvMin;
    v0.material = materialId;
    v0.n_x = encodeNormalComponent(normal.x);
    v0.n_y = encodeNormalComponent(normal.y);
    v0.n_z = encodeNormalComponent(normal.z);
    v0.lodLevel = lodLevel;

    VertexAttributes v1 = v0;
    v1.x = corners[1].x;
    v1.y = corners[1].y;
    v1.z = corners[1].z;
    v1.u = kUvMax;
    v1.v = kUvMin;

    VertexAttributes v2 = v0;
    v2.x = corners[2].x;
    v2.y = corners[2].y;
    v2.z = corners[2].z;
    v2.u = kUvMin;
    v2.v = kUvMax;

    VertexAttributes v3 = v0;
    v3.x = corners[3].x;
    v3.y = corners[3].y;
    v3.z = corners[3].z;
    v3.u = kUvMax;
    v3.v = kUvMax;

    vertices.push_back(v0);
    vertices.push_back(v1);
    vertices.push_back(v2);
    vertices.push_back(v3);

    indices.push_back(base + 0);
    indices.push_back(base + 1);
    indices.push_back(base + 2);
    indices.push_back(base + 1);
    indices.push_back(base + 3);
    indices.push_back(base + 2);
}
}

int WebGPURenderer::chooseSizeClass(uint64_t requiredBytes) const {
    for (int i = 0; i < static_cast<int>(kSizeClasses.size()); ++i) {
        if (requiredBytes <= kSizeClasses[static_cast<std::size_t>(i)]) {
            return i;
        }
    }
    return -1;
}

int WebGPURenderer::chooseLodByDistance(float distance) const {
    for (int i = 0; i < kLodCount - 1; ++i) {
        if (distance < lodSwitchDistances_[static_cast<std::size_t>(i)]) {
            return i;
        }
    }
    return kLodCount - 1;
}

bool WebGPURenderer::loadHeightmap(const std::string& path) {
    std::vector<unsigned char> pixels;
    unsigned width = 0;
    unsigned height = 0;
    const unsigned error = lodepng::decode(pixels, width, height, path);
    if (error != 0 || width == 0 || height == 0) {
        std::cerr << "Failed to load heightmap: " << path
                  << " (" << lodepng_error_text(error) << ")" << std::endl;
        heightmapRgba_.clear();
        heightmapWidth_ = 0;
        heightmapHeight_ = 0;
        return false;
    }

    heightmapRgba_ = std::move(pixels);
    heightmapWidth_ = static_cast<uint32_t>(width);
    heightmapHeight_ = static_cast<uint32_t>(height);
    return true;
}

float WebGPURenderer::sampleHeightmap01(int blockX, int blockY) const {
    if (heightmapRgba_.empty() || heightmapWidth_ == 0 || heightmapHeight_ == 0) {
        return sampleFallbackHeight01(blockX, blockY);
    }

    const int upscale = std::max(1, heightmapUpscaleFactor_);
    const int worldSpanX = static_cast<int>(heightmapWidth_) * upscale;
    const int worldSpanY = static_cast<int>(heightmapHeight_) * upscale;
    if (worldSpanX <= 0 || worldSpanY <= 0) {
        return sampleFallbackHeight01(blockX, blockY);
    }

    float xInMapBlocks = static_cast<float>(blockX);
    float yInMapBlocks = static_cast<float>(blockY);

    if (heightmapWrap_) {
        xInMapBlocks = std::fmod(xInMapBlocks, static_cast<float>(worldSpanX));
        yInMapBlocks = std::fmod(yInMapBlocks, static_cast<float>(worldSpanY));
        if (xInMapBlocks < 0.0f) {
            xInMapBlocks += static_cast<float>(worldSpanX);
        }
        if (yInMapBlocks < 0.0f) {
            yInMapBlocks += static_cast<float>(worldSpanY);
        }
    } else {
        xInMapBlocks = std::clamp(xInMapBlocks, 0.0f, static_cast<float>(worldSpanX - 1));
        yInMapBlocks = std::clamp(yInMapBlocks, 0.0f, static_cast<float>(worldSpanY - 1));
    }

    const float sampleX = xInMapBlocks / static_cast<float>(upscale);
    const float sampleY = yInMapBlocks / static_cast<float>(upscale);

    int x0 = static_cast<int>(std::floor(sampleX));
    int y0 = static_cast<int>(std::floor(sampleY));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    auto wrapOrClamp = [this](int value, uint32_t dimension) {
        if (dimension == 0) {
            return 0;
        }
        if (heightmapWrap_) {
            int v = value % static_cast<int>(dimension);
            if (v < 0) {
                v += static_cast<int>(dimension);
            }
            return v;
        }
        return std::clamp(value, 0, static_cast<int>(dimension) - 1);
    };

    x0 = wrapOrClamp(x0, heightmapWidth_);
    x1 = wrapOrClamp(x1, heightmapWidth_);
    y0 = wrapOrClamp(y0, heightmapHeight_);
    y1 = wrapOrClamp(y1, heightmapHeight_);

    auto samplePixel01 = [this](int px, int py) {
        const std::size_t index = (static_cast<std::size_t>(py) * static_cast<std::size_t>(heightmapWidth_) + static_cast<std::size_t>(px)) * 4ull;
        const float r = static_cast<float>(heightmapRgba_[index + 0]) / 255.0f;
        const float g = static_cast<float>(heightmapRgba_[index + 1]) / 255.0f;
        const float b = static_cast<float>(heightmapRgba_[index + 2]) / 255.0f;
        return 0.2126f * r + 0.7152f * g + 0.0722f * b;
    };

    const float tx = sampleX - std::floor(sampleX);
    const float ty = sampleY - std::floor(sampleY);

    const float h00 = samplePixel01(x0, y0);
    const float h10 = samplePixel01(x1, y0);
    const float h01 = samplePixel01(x0, y1);
    const float h11 = samplePixel01(x1, y1);

    const float hx0 = h00 + (h10 - h00) * tx;
    const float hx1 = h01 + (h11 - h01) * tx;
    const float h = hx0 + (hx1 - hx0) * ty;
    return std::clamp(h, 0.0f, 1.0f);
}

int WebGPURenderer::sampleHeightBlocks(int blockX, int blockY) const {
    const float h01 = sampleHeightmap01(blockX, blockY);
    const int minHeight = terrainHeightMinBlocks_;
    const int maxHeight = terrainHeightMaxBlocks_;
    const int range = std::max(1, maxHeight - minHeight);
    const int h = minHeight + static_cast<int>(std::round(h01 * static_cast<float>(range)));
    return std::clamp(h, minHeight, maxHeight);
}

WebGPURenderer::BufferSlot* WebGPURenderer::acquireSlot(std::vector<BufferSlot>& slots, bool isVertex, uint64_t requiredBytes) {
    const int classIndex = chooseSizeClass(requiredBytes);
    if (classIndex < 0) {
        return nullptr;
    }

    for (BufferSlot& slot : slots) {
        if (!slot.inUse && slot.classIndex == classIndex) {
            slot.inUse = true;
            slot.usedBytes = requiredBytes;
            return &slot;
        }
    }

    const std::string bufferName = std::string(isVertex ? "mesh_vp_" : "mesh_ip_") + std::to_string(nextBufferId_++);
    const uint64_t slotBytes = kSizeClasses[static_cast<std::size_t>(classIndex)];
    const uint32_t slotsPerPage = kSlotsPerPage[static_cast<std::size_t>(classIndex)];

    BufferDescriptor desc = Default;
    desc.label = StringView(isVertex ? "mesh vertex page" : "mesh index page");
    desc.size = slotBytes * static_cast<uint64_t>(slotsPerPage);
    desc.usage = BufferUsage::CopyDst | (isVertex ? BufferUsage::Vertex : BufferUsage::Index);
    desc.mappedAtCreation = false;

    Buffer buffer = bufferManager->createBuffer(bufferName, desc);
    if (!buffer) {
        return nullptr;
    }

    std::size_t firstNewSlot = slots.size();
    slots.reserve(slots.size() + static_cast<std::size_t>(slotsPerPage));
    for (uint32_t i = 0; i < slotsPerPage; ++i) {
        BufferSlot slot;
        slot.bufferName = bufferName;
        slot.offset = static_cast<uint64_t>(i) * slotBytes;
        slot.capacity = slotBytes;
        slot.usedBytes = 0;
        slot.classIndex = classIndex;
        slot.inUse = false;
        slots.push_back(std::move(slot));
    }

    BufferSlot& allocated = slots[firstNewSlot];
    allocated.inUse = true;
    allocated.usedBytes = requiredBytes;
    return &allocated;
}

void WebGPURenderer::releaseSlot(std::vector<BufferSlot>& slots, const std::string& bufferName, uint64_t offset) {
    for (BufferSlot& slot : slots) {
        if (slot.bufferName == bufferName && slot.offset == offset) {
            slot.inUse = false;
            slot.usedBytes = 0;
            return;
        }
    }
}

bool WebGPURenderer::uploadMesh(std::vector<VertexAttributes>&& vertices, std::vector<uint32_t>&& indices, MeshSlotRef& outMeshSlot) {
    outMeshSlot = {};

    if (vertices.empty() || indices.empty()) {
        return false;
    }

    const uint64_t vertexBytes = static_cast<uint64_t>(vertices.size()) * sizeof(VertexAttributes);
    const uint64_t indexBytes = static_cast<uint64_t>(indices.size()) * sizeof(uint32_t);

    BufferSlot* vSlot = acquireSlot(vertexSlots_, true, vertexBytes);
    if (vSlot == nullptr) {
        return false;
    }

    BufferSlot* iSlot = acquireSlot(indexSlots_, false, indexBytes);
    if (iSlot == nullptr) {
        vSlot->inUse = false;
        vSlot->usedBytes = 0;
        return false;
    }

    bufferManager->writeBuffer(vSlot->bufferName, vSlot->offset, vertices.data(), static_cast<size_t>(vertexBytes));
    bufferManager->writeBuffer(iSlot->bufferName, iSlot->offset, indices.data(), static_cast<size_t>(indexBytes));

    outMeshSlot.vertexBufferName = vSlot->bufferName;
    outMeshSlot.vertexOffset = vSlot->offset;
    outMeshSlot.vertexBytes = vertexBytes;
    outMeshSlot.indexBufferName = iSlot->bufferName;
    outMeshSlot.indexOffset = iSlot->offset;
    outMeshSlot.indexBytes = indexBytes;
    outMeshSlot.indexCount = static_cast<uint32_t>(indices.size());
    outMeshSlot.valid = true;
    return true;
}

void WebGPURenderer::releaseMesh(MeshSlotRef& meshSlot) {
    if (!meshSlot.valid) {
        return;
    }

    releaseSlot(vertexSlots_, meshSlot.vertexBufferName, meshSlot.vertexOffset);
    releaseSlot(indexSlots_, meshSlot.indexBufferName, meshSlot.indexOffset);
    meshSlot = {};
}

void WebGPURenderer::releaseAllRegionMeshes() {
    for (auto& regionPair : renderedRegions_) {
        RegionRenderEntry& region = regionPair.second;
        for (MeshSlotRef& mesh : region.lodMeshes) {
            releaseMesh(mesh);
        }
    }
    renderedRegions_.clear();
    drawOrder_.clear();
    pendingBuilds_.clear();
}

std::pair<std::vector<VertexAttributes>, std::vector<uint32_t>> WebGPURenderer::buildRegionLodMesh(RegionCoord regionCoord, int lodLevel) const {
    const int step = lodSteps_[static_cast<std::size_t>(std::clamp(lodLevel, 0, kLodCount - 1))];
    const int cells = REGION_BLOCKS_XY / step;

    const int originX = regionCoord.x() * REGION_BLOCKS_XY;
    const int originY = regionCoord.y() * REGION_BLOCKS_XY;

    const int globalCellOriginX = floor_div(originX, step);
    const int globalCellOriginY = floor_div(originY, step);

    auto sampleQuantizedCellHeight = [this, step](int globalCellX, int globalCellY) {
        const int sampleX = globalCellX * step + step / 2;
        const int sampleY = globalCellY * step + step / 2;
        const int h = sampleHeightBlocks(sampleX, sampleY);
        return (h / step) * step;
    };

    auto localIndex = [cells](int x, int y) -> std::size_t {
        return static_cast<std::size_t>(y * cells + x);
    };

    std::vector<int> heights(static_cast<std::size_t>(cells * cells), 0);
    for (int y = 0; y < cells; ++y) {
        for (int x = 0; x < cells; ++x) {
            const int gx = globalCellOriginX + x;
            const int gy = globalCellOriginY + y;
            heights[localIndex(x, y)] = sampleQuantizedCellHeight(gx, gy);
        }
    }

    std::vector<VertexAttributes> vertices;
    std::vector<uint32_t> indices;

    vertices.reserve(static_cast<std::size_t>(cells * cells * 12));
    indices.reserve(static_cast<std::size_t>(cells * cells * 18));

    for (int y = 0; y < cells; ++y) {
        for (int x = 0; x < cells; ++x) {
            const int h = heights[localIndex(x, y)];
            if (h <= 0) {
                continue;
            }

            const int x0 = originX + x * step;
            const int y0 = originY + y * step;
            const int z0 = h - step;
            const int x1 = x0 + step;
            const int y1 = y0 + step;
            const int z1 = h;

            const uint16_t topMaterial = 2;
            const uint16_t sideMaterial = 1;

            emitFace(vertices, indices,
                std::array<glm::ivec3, 4>{
                    glm::ivec3{x0, y0, z1}, glm::ivec3{x1, y0, z1}, glm::ivec3{x0, y1, z1}, glm::ivec3{x1, y1, z1}
                },
                glm::ivec3{0, 0, 1}, topMaterial, static_cast<uint8_t>(lodLevel));

            const int globalCellX = globalCellOriginX + x;
            const int globalCellY = globalCellOriginY + y;

            const int hPosX = (x + 1 < cells)
                ? heights[localIndex(x + 1, y)]
                : sampleQuantizedCellHeight(globalCellX + 1, globalCellY);
            const int hNegX = (x - 1 >= 0)
                ? heights[localIndex(x - 1, y)]
                : sampleQuantizedCellHeight(globalCellX - 1, globalCellY);
            const int hPosY = (y + 1 < cells)
                ? heights[localIndex(x, y + 1)]
                : sampleQuantizedCellHeight(globalCellX, globalCellY + 1);
            const int hNegY = (y - 1 >= 0)
                ? heights[localIndex(x, y - 1)]
                : sampleQuantizedCellHeight(globalCellX, globalCellY - 1);

            if (h > hPosX) {
                emitFace(vertices, indices,
                    std::array<glm::ivec3, 4>{
                        glm::ivec3{x1, y0, hPosX}, glm::ivec3{x1, y1, hPosX}, glm::ivec3{x1, y0, h}, glm::ivec3{x1, y1, h}
                    },
                    glm::ivec3{1, 0, 0}, sideMaterial, static_cast<uint8_t>(lodLevel));
            }
            if (h > hNegX) {
                emitFace(vertices, indices,
                    std::array<glm::ivec3, 4>{
                        glm::ivec3{x0, y0, hNegX}, glm::ivec3{x0, y0, h}, glm::ivec3{x0, y1, hNegX}, glm::ivec3{x0, y1, h}
                    },
                    glm::ivec3{-1, 0, 0}, sideMaterial, static_cast<uint8_t>(lodLevel));
            }
            if (h > hPosY) {
                emitFace(vertices, indices,
                    std::array<glm::ivec3, 4>{
                        glm::ivec3{x0, y1, hPosY}, glm::ivec3{x0, y1, h}, glm::ivec3{x1, y1, hPosY}, glm::ivec3{x1, y1, h}
                    },
                    glm::ivec3{0, 1, 0}, sideMaterial, static_cast<uint8_t>(lodLevel));
            }
            if (h > hNegY) {
                emitFace(vertices, indices,
                    std::array<glm::ivec3, 4>{
                        glm::ivec3{x0, y0, hNegY}, glm::ivec3{x1, y0, hNegY}, glm::ivec3{x0, y0, h}, glm::ivec3{x1, y0, h}
                    },
                    glm::ivec3{0, -1, 0}, sideMaterial, static_cast<uint8_t>(lodLevel));
            }

            if (lodLevel == 0 && z0 == 0) {
                emitFace(vertices, indices,
                    std::array<glm::ivec3, 4>{
                        glm::ivec3{x0, y0, 0}, glm::ivec3{x0, y1, 0}, glm::ivec3{x1, y0, 0}, glm::ivec3{x1, y1, 0}
                    },
                    glm::ivec3{0, 0, -1}, 7, static_cast<uint8_t>(lodLevel));
            }
        }
    }

    emitFace(vertices, indices,
        std::array<glm::ivec3, 4>{
            glm::ivec3{originX, originY, 0},
            glm::ivec3{originX, originY + REGION_BLOCKS_XY, 0},
            glm::ivec3{originX + REGION_BLOCKS_XY, originY, 0},
            glm::ivec3{originX + REGION_BLOCKS_XY, originY + REGION_BLOCKS_XY, 0}
        },
        glm::ivec3{0, 0, -1}, 7, static_cast<uint8_t>(lodLevel));

    return {std::move(vertices), std::move(indices)};
}

void WebGPURenderer::rebuildRegionsAroundPlayer(int centerRegionX, int centerRegionY) {
    if (centerRegionX == activeCenterRegionX && centerRegionY == activeCenterRegionY) {
        return;
    }

    std::vector<RegionCoord> desiredOrder;
    desiredOrder.reserve(static_cast<std::size_t>((regionRadius_ * 2 + 1) * (regionRadius_ * 2 + 1)));

    for (int ry = -regionRadius_; ry <= regionRadius_; ++ry) {
        for (int rx = -regionRadius_; rx <= regionRadius_; ++rx) {
            desiredOrder.emplace_back(centerRegionX + rx, centerRegionY + ry);
        }
    }

    std::unordered_set<RegionCoord, RegionCoordHash> desiredSet;
    desiredSet.reserve(desiredOrder.size() * 2);
    for (const RegionCoord& coord : desiredOrder) {
        desiredSet.insert(coord);
    }

    for (auto it = renderedRegions_.begin(); it != renderedRegions_.end();) {
        if (desiredSet.find(it->first) == desiredSet.end()) {
            RegionRenderEntry& region = it->second;
            for (MeshSlotRef& mesh : region.lodMeshes) {
                releaseMesh(mesh);
            }
            it = renderedRegions_.erase(it);
        } else {
            ++it;
        }
    }

    for (const RegionCoord& coord : desiredOrder) {
        if (renderedRegions_.find(coord) == renderedRegions_.end()) {
            RegionRenderEntry entry;
            entry.coord = coord;
            renderedRegions_.emplace(coord, std::move(entry));
        }
    }

    drawOrder_ = desiredOrder;
    pendingBuilds_.clear();

    for (int ring = 0; ring <= regionRadius_; ++ring) {
        for (const RegionCoord& coord : desiredOrder) {
            const int rx = coord.x() - centerRegionX;
            const int ry = coord.y() - centerRegionY;
            const int coordRing = std::max(std::abs(rx), std::abs(ry));
            if (coordRing != ring) {
                continue;
            }

            RegionRenderEntry& entry = renderedRegions_.find(coord)->second;

            const float approxDistance = std::sqrt(
                static_cast<float>(rx * rx + ry * ry)
            ) * static_cast<float>(REGION_BLOCKS_XY);
            const int minLod = chooseLodByDistance(approxDistance);
            for (int lod = kLodCount - 1; lod >= minLod; --lod) {
                if (!entry.lodMeshes[static_cast<std::size_t>(lod)].valid) {
                    pendingBuilds_.push_back({coord, lod});
                }
            }
        }
    }

    activeCenterRegionX = centerRegionX;
    activeCenterRegionY = centerRegionY;
}

void WebGPURenderer::processPendingRegionBuilds() {
    if (pendingBuilds_.empty()) {
        return;
    }

    const auto start = std::chrono::steady_clock::now();
    while (!pendingBuilds_.empty()) {
        const PendingRegionBuild build = pendingBuilds_.front();
        pendingBuilds_.pop_front();

        auto regionIt = renderedRegions_.find(build.coord);
        if (regionIt == renderedRegions_.end()) {
            continue;
        }

        RegionRenderEntry& entry = regionIt->second;
        if (entry.lodMeshes[static_cast<std::size_t>(build.lodLevel)].valid) {
            continue;
        }

        auto [vertices, indices] = buildRegionLodMesh(build.coord, build.lodLevel);
        uploadMesh(std::move(vertices), std::move(indices), entry.lodMeshes[static_cast<std::size_t>(build.lodLevel)]);

        const auto now = std::chrono::steady_clock::now();
        const double elapsedMs = std::chrono::duration<double, std::milli>(now - start).count();
        if (elapsedMs >= buildBudgetMs_) {
            break;
        }
    }
}

void WebGPURenderer::drawRegionSet(RenderPassEncoder& pass, const glm::vec3& cameraPos) {
    const glm::vec2 cameraXY{cameraPos.x, cameraPos.y};

    for (const RegionCoord& coord : drawOrder_) {
        const auto regionIt = renderedRegions_.find(coord);
        if (regionIt == renderedRegions_.end()) {
            continue;
        }

        const RegionRenderEntry& entry = regionIt->second;
        const glm::vec2 regionCenter{
            (static_cast<float>(entry.coord.x()) + 0.5f) * static_cast<float>(REGION_BLOCKS_XY),
            (static_cast<float>(entry.coord.y()) + 0.5f) * static_cast<float>(REGION_BLOCKS_XY),
        };
        const float distance = glm::distance(cameraXY, regionCenter);
        const int targetLod = chooseLodByDistance(distance);

        const MeshSlotRef* mesh = nullptr;
        for (int lod = targetLod; lod < kLodCount; ++lod) {
            const MeshSlotRef& candidate = entry.lodMeshes[static_cast<std::size_t>(lod)];
            if (candidate.valid) {
                mesh = &candidate;
                break;
            }
        }
        if (mesh == nullptr) {
            for (int lod = targetLod - 1; lod >= 0; --lod) {
                const MeshSlotRef& candidate = entry.lodMeshes[static_cast<std::size_t>(lod)];
                if (candidate.valid) {
                    mesh = &candidate;
                    break;
                }
            }
        }

        if (mesh == nullptr || mesh->indexCount == 0) {
            continue;
        }

        Buffer vertexBuffer = bufferManager->getBuffer(mesh->vertexBufferName);
        Buffer indexBuffer = bufferManager->getBuffer(mesh->indexBufferName);
        if (!vertexBuffer || !indexBuffer) {
            continue;
        }

        pass.setVertexBuffer(0, vertexBuffer, mesh->vertexOffset, mesh->vertexBytes);
        pass.setIndexBuffer(indexBuffer, IndexFormat::Uint32, mesh->indexOffset, mesh->indexBytes);
        pass.drawIndexed(mesh->indexCount, 1, 0, 0, 0);
    }
}

bool WebGPURenderer::initialize() {
    RenderConfig config;

    context = std::make_unique<WebGPUContext>();
    if (!context->initialize(config)) {
        return false;
    }

    pipelineManager = std::make_unique<PipelineManager>(context->getDevice(), context->getSurfaceFormat());
    bufferManager = std::make_unique<BufferManager>(context->getDevice(), context->getQueue());
    textureManager = std::make_unique<TextureManager>(context->getDevice(), context->getQueue());

    const WorldTuningParameters tuning = kDefaultWorldTuningParameters;

    const int configuredViewDistanceChunks = std::max(1, tuning.viewDistanceChunks);
    regionRadius_ = std::max(1, (configuredViewDistanceChunks + (REGION_COLS - 1)) / REGION_COLS);

    lodSteps_ = tuning.regionLodSteps;
    for (int i = 0; i < kLodCount; ++i) {
        if (lodSteps_[static_cast<std::size_t>(i)] < 1) {
            lodSteps_[static_cast<std::size_t>(i)] = 1;
        }
        if (i > 0) {
            const int prev = lodSteps_[static_cast<std::size_t>(i - 1)];
            if (lodSteps_[static_cast<std::size_t>(i)] < prev) {
                lodSteps_[static_cast<std::size_t>(i)] = prev;
            }
        }
    }

    lodSwitchDistances_ = tuning.regionLodSwitchDistances;
    for (int i = 0; i < kLodCount - 1; ++i) {
        const float minAllowed = (i == 0)
            ? 1.0f
            : lodSwitchDistances_[static_cast<std::size_t>(i - 1)] + 1.0f;
        lodSwitchDistances_[static_cast<std::size_t>(i)] = std::max(minAllowed, lodSwitchDistances_[static_cast<std::size_t>(i)]);
    }
    buildBudgetMs_ = std::max(0.1, tuning.regionBuildBudgetMs);

    heightmapUpscaleFactor_ = std::max(1, tuning.heightmapUpscaleFactor);
    heightmapWrap_ = tuning.heightmapWrap;
    const int worldHeightMaxBlock = CHUNK_SIZE * COLUMN_CHUNKS_Z - 1;
    terrainHeightMinBlocks_ = std::clamp(tuning.terrainMinHeightBlocks, 0, worldHeightMaxBlock);
    terrainHeightMaxBlocks_ = std::clamp(tuning.terrainMaxHeightBlocks, 0, worldHeightMaxBlock);
    if (terrainHeightMaxBlocks_ <= terrainHeightMinBlocks_) {
        if (terrainHeightMinBlocks_ > 0) {
            terrainHeightMinBlocks_ -= 1;
        } else if (terrainHeightMaxBlocks_ < worldHeightMaxBlock) {
            terrainHeightMaxBlocks_ += 1;
        }
    }

    const char* heightmapRelativePath = (tuning.heightmapRelativePath != nullptr) ? tuning.heightmapRelativePath : "heightmap.png";
    const std::filesystem::path heightmapPath = std::filesystem::path(RESOURCE_DIR) / heightmapRelativePath;
    loadHeightmap(heightmapPath.string());

    const std::size_t expectedRegionCount = static_cast<std::size_t>((regionRadius_ * 2 + 1) * (regionRadius_ * 2 + 1));
    renderedRegions_.reserve(expectedRegionCount * 2);
    drawOrder_.reserve(expectedRegionCount);

    {
        BufferDescriptor desc = Default;
        desc.label = StringView("uniform buffer");
        desc.size = sizeof(FrameUniforms);
        desc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
        desc.mappedAtCreation = false;
        Buffer ubo = bufferManager->createBuffer("uniform_buffer", desc);
        if (!ubo) {
            return false;
        }
    }

    voxelPipeline.init(bufferManager.get(), textureManager.get(), pipelineManager.get(), context.get());
    if (!voxelPipeline.createResources()) {
        std::cerr << "Failed to create voxel rendering resources." << std::endl;
        return false;
    }
    if (!voxelPipeline.createPipeline()) {
        std::cerr << "Failed to create voxel render pipeline." << std::endl;
        return false;
    }
    if (!voxelPipeline.createBindGroup()) {
        std::cerr << "Failed to create voxel bind group." << std::endl;
        return false;
    }

    return true;
}

void WebGPURenderer::createRenderingTextures() {
    if (!voxelPipeline.createResources()) {
        std::cerr << "Failed to recreate voxel rendering resources." << std::endl;
        return;
    }
    if (!voxelPipeline.createBindGroup()) {
        std::cerr << "Failed to recreate voxel bind group." << std::endl;
    }
}

void WebGPURenderer::removeRenderingTextures() {
    voxelPipeline.removeResources();
}

bool WebGPURenderer::resizeSurfaceAndAttachments() {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(context->getWindow(), &width, &height);
    if (width <= 0 || height <= 0) {
        return false;
    }

    removeRenderingTextures();
    context->unconfigureSurface();
    if (!context->configureSurface()) {
        return false;
    }
    createRenderingTextures();
    resizePending = false;
    return true;
}

void WebGPURenderer::requestResize() {
    resizePending = true;
}

WebGPUContext* WebGPURenderer::getContext() {
    return context.get();
}

PipelineManager* WebGPURenderer::getPipelineManager() {
    return pipelineManager.get();
}

TextureManager* WebGPURenderer::getTextureManager() {
    return textureManager.get();
}

BufferManager* WebGPURenderer::getBufferManager() {
    return bufferManager.get();
}

void WebGPURenderer::toggleRegionDebugColors() {
    debugRenderFlags_ ^= DebugRegionColors;
}

void WebGPURenderer::toggleLodDebugColors() {
    debugRenderFlags_ ^= DebugLodColors;
}

void WebGPURenderer::toggleChunkDebugColors() {
    debugRenderFlags_ ^= DebugChunkColors;
}

void WebGPURenderer::clearDebugColors() {
    debugRenderFlags_ = 0u;
}

uint32_t WebGPURenderer::debugRenderFlags() const {
    return debugRenderFlags_;
}

void WebGPURenderer::renderFrame(FrameUniforms& uniforms) {
    uniforms.debugParams = glm::uvec4{debugRenderFlags_, 0u, 0u, 0u};
    bufferManager->writeBuffer(
        "uniform_buffer",
        offsetof(FrameUniforms, debugParams),
        &uniforms.debugParams,
        sizeof(FrameUniforms::debugParams)
    );

    const glm::vec3 cameraPos = glm::vec3(uniforms.inverseViewMatrix * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    const int regionX = floor_div(static_cast<int>(std::floor(cameraPos.x)), REGION_BLOCKS_XY);
    const int regionY = floor_div(static_cast<int>(std::floor(cameraPos.y)), REGION_BLOCKS_XY);
    rebuildRegionsAroundPlayer(regionX, regionY);
    processPendingRegionBuilds();

    int fbWidth = 0;
    int fbHeight = 0;
    glfwGetFramebufferSize(context->getWindow(), &fbWidth, &fbHeight);
    if (fbWidth > 0 && fbHeight > 0 &&
        (fbWidth != context->width || fbHeight != context->height)) {
        requestResize();
    }

    if (resizePending) {
        if (!resizeSurfaceAndAttachments()) {
            return;
        }
    }

    auto [surfaceTexture, targetView] = GetNextSurfaceViewData();
    if (!targetView) {
        return;
    }

    CommandEncoderDescriptor encoderDesc = Default;
    encoderDesc.label = StringView("Frame command encoder");
    CommandEncoder encoder = context->getDevice().createCommandEncoder(encoderDesc);

    {
        voxelPipeline.render(targetView, encoder, [&](RenderPassEncoder& pass) {
            drawRegionSet(pass, cameraPos);
            ImGui::Render();
            ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), pass);
        });
    }

    CommandBufferDescriptor cmdBufferDescriptor = Default;
    cmdBufferDescriptor.label = StringView("Frame command buffer");
    CommandBuffer command = encoder.finish(cmdBufferDescriptor);
    encoder.release();

    context->getQueue().submit(1, &command);
    command.release();

#ifdef WEBGPU_BACKEND_DAWN
    context->getDevice().tick();
#endif

    targetView.release();
    context->getSurface().present();

#ifdef WEBGPU_BACKEND_DAWN
    context->getDevice().tick();
#endif
}

GLFWwindow* WebGPURenderer::getWindow() {
    return context->getWindow();
}

std::pair<SurfaceTexture, TextureView> WebGPURenderer::GetNextSurfaceViewData() {
    SurfaceTexture surfaceTexture;
    context->getSurface().getCurrentTexture(&surfaceTexture);
    Texture texture = surfaceTexture.texture;

    if (surfaceTexture.status == SurfaceGetCurrentTextureStatus::SuccessSuboptimal) {
        requestResize();
        if (texture) {
            texture.release();
        }
        return { surfaceTexture, nullptr };
    }

    if (surfaceTexture.status != SurfaceGetCurrentTextureStatus::SuccessOptimal) {
        if (surfaceTexture.status == SurfaceGetCurrentTextureStatus::Outdated ||
            surfaceTexture.status == SurfaceGetCurrentTextureStatus::Lost) {
            requestResize();
        }
        if (texture) {
            texture.release();
        }
        return { surfaceTexture, nullptr };
    }

    TextureViewDescriptor viewDescriptor;
    viewDescriptor.nextInChain = nullptr;
    viewDescriptor.label = StringView("Surface texture view");
    viewDescriptor.format = texture.getFormat();
    viewDescriptor.dimension = TextureViewDimension::_2D;
    viewDescriptor.baseMipLevel = 0;
    viewDescriptor.mipLevelCount = 1;
    viewDescriptor.baseArrayLayer = 0;
    viewDescriptor.arrayLayerCount = 1;
    viewDescriptor.aspect = TextureAspect::All;
    TextureView targetView = texture.createView(viewDescriptor);

#ifndef WEBGPU_BACKEND_WGPU
    texture.release();
#endif

    return { surfaceTexture, targetView };
}

void WebGPURenderer::terminate() {
    releaseAllRegionMeshes();
    textureManager->terminate();
    pipelineManager->terminate();
    bufferManager->terminate();
}
