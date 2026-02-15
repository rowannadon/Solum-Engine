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
#include <limits>
#include <unordered_set>
#include <vector>

namespace {
constexpr int kLodCount = kRegionLodCount;
constexpr uint32_t kMeshletsPerPage = 8192u;
constexpr uint32_t kInitialMeshletPageCount = 2u;
constexpr uint32_t kMaxMeshletPages = 16u;
constexpr uint32_t kInitialMeshletMetadataCapacity = 65536u;
constexpr uint32_t kMaxDrawMeshletsPerFrame = 16384u;

constexpr uint16_t kUvMin = 0;
constexpr uint16_t kUvMax = 65535;

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

uint16_t clampU16(int32_t value) {
    return static_cast<uint16_t>(std::clamp(value, 0, 65535));
}

uint8_t decodeNormalSign(uint8_t encoded) {
    if (encoded > 170u) {
        return 2u; // +1
    }
    if (encoded < 85u) {
        return 0u; // -1
    }
    return 1u; // 0
}

uint8_t encodeNormalAxisIndex(const VertexAttributes& vertex) {
    const uint8_t sx = decodeNormalSign(vertex.n_x);
    const uint8_t sy = decodeNormalSign(vertex.n_y);
    const uint8_t sz = decodeNormalSign(vertex.n_z);

    if (sx == 2u) return 0u; // +X
    if (sx == 0u) return 1u; // -X
    if (sy == 2u) return 2u; // +Y
    if (sy == 0u) return 3u; // -Y
    if (sz == 2u) return 4u; // +Z
    return 5u;               // -Z
}

PackedVertexAttributes packVertex(const VertexAttributes& vertex, const glm::ivec3& origin) {
    const uint16_t relX = clampU16(vertex.x - origin.x);
    const uint16_t relY = clampU16(vertex.y - origin.y);
    const uint16_t relZ = clampU16(vertex.z - origin.z);
    const uint32_t uBit = (vertex.u > 32767u) ? 1u : 0u;
    const uint32_t vBit = (vertex.v > 32767u) ? 1u : 0u;
    const uint32_t normalBits = static_cast<uint32_t>(encodeNormalAxisIndex(vertex)) & 0x7u;
    const uint32_t lodBits = static_cast<uint32_t>(vertex.lodLevel) & 0xFFu;

    PackedVertexAttributes packed{};
    packed.xy = static_cast<uint32_t>(relX) | (static_cast<uint32_t>(relY) << 16u);
    packed.zMaterial = static_cast<uint32_t>(relZ) | (static_cast<uint32_t>(vertex.material) << 16u);
    packed.packedFlags = uBit | (vBit << 1u) | (normalBits << 2u) | (lodBits << 5u);
    return packed;
}

MeshData buildMeshletsFromTriangles(const std::vector<VertexAttributes>& vertices, const std::vector<uint32_t>& indices) {
    MeshData meshData;
    if (vertices.empty() || indices.empty()) {
        return meshData;
    }

    const std::size_t quadCount = indices.size() / 6ull;
    if (quadCount == 0) {
        return meshData;
    }

    const std::size_t meshletCount = (quadCount + MeshData::kMeshletQuadCapacity - 1ull) / MeshData::kMeshletQuadCapacity;
    meshData.meshlets.reserve(meshletCount);
    meshData.packedVertices.reserve(meshletCount * MeshData::kMeshletVertexCapacity);
    meshData.packedIndices.reserve(meshletCount * MeshData::kMeshletIndexCapacity);

    for (std::size_t quadBase = 0; quadBase < quadCount; quadBase += MeshData::kMeshletQuadCapacity) {
        const uint32_t quadsInMeshlet = static_cast<uint32_t>(std::min<std::size_t>(MeshData::kMeshletQuadCapacity, quadCount - quadBase));
        const std::size_t globalVertexBase = quadBase * 4ull;
        if (globalVertexBase >= vertices.size()) {
            break;
        }

        const std::size_t requestedVertexCount = static_cast<std::size_t>(quadsInMeshlet) * 4ull;
        const std::size_t availableVertexCount = std::min<std::size_t>(requestedVertexCount, vertices.size() - globalVertexBase);
        if (availableVertexCount == 0) {
            continue;
        }

        glm::ivec3 origin{std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), std::numeric_limits<int>::max()};
        for (std::size_t i = 0; i < availableVertexCount; ++i) {
            const VertexAttributes& v = vertices[globalVertexBase + i];
            origin.x = std::min(origin.x, v.x);
            origin.y = std::min(origin.y, v.y);
            origin.z = std::min(origin.z, v.z);
        }

        MeshletInfo info{};
        info.origin = origin;
        info.lodLevel = static_cast<uint32_t>(vertices[globalVertexBase].lodLevel);
        info.quadCount = quadsInMeshlet;
        info.vertexOffset = static_cast<uint32_t>(meshData.packedVertices.size());
        info.indexOffset = static_cast<uint32_t>(meshData.packedIndices.size());

        for (std::size_t i = 0; i < availableVertexCount; ++i) {
            meshData.packedVertices.push_back(packVertex(vertices[globalVertexBase + i], origin));
        }
        while ((meshData.packedVertices.size() - info.vertexOffset) < MeshData::kMeshletVertexCapacity) {
            meshData.packedVertices.push_back(PackedVertexAttributes{});
        }

        const std::size_t globalIndexBase = quadBase * 6ull;
        const std::size_t indexCount = static_cast<std::size_t>(quadsInMeshlet) * 6ull;
        for (std::size_t i = 0; i < indexCount; ++i) {
            uint32_t localIndex = 0u;
            const std::size_t globalIndexPos = globalIndexBase + i;
            if (globalIndexPos < indices.size()) {
                const uint32_t globalIndex = indices[globalIndexPos];
                if (globalIndex >= globalVertexBase && globalIndex < (globalVertexBase + availableVertexCount)) {
                    localIndex = static_cast<uint32_t>(globalIndex - globalVertexBase);
                }
            }
            meshData.packedIndices.push_back(localIndex);
        }
        while ((meshData.packedIndices.size() - info.indexOffset) < MeshData::kMeshletIndexCapacity) {
            meshData.packedIndices.push_back(0u);
        }

        meshData.meshlets.push_back(info);
    }

    return meshData;
}
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
        return 0.0f;
    }

    const int upscale = std::max(1, heightmapUpscaleFactor_);
    const int worldSpanX = static_cast<int>(heightmapWidth_) * upscale;
    const int worldSpanY = static_cast<int>(heightmapHeight_) * upscale;
    if (worldSpanX <= 0 || worldSpanY <= 0) {
        return 0.0f;
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

bool WebGPURenderer::refreshVoxelBindGroup() {
    for (const MeshletPage& page : meshletPages_) {
        std::vector<BindGroupEntry> bindings(4);
        int i = 0;
        bindings[i].binding = i;
        bindings[i].buffer = bufferManager->getBuffer("uniform_buffer");
        bindings[i].offset = 0;
        bindings[i].size = sizeof(FrameUniforms);
        i++;

        bindings[i].binding = i;
        bindings[i].buffer = bufferManager->getBuffer(page.metadataBufferName);
        bindings[i].offset = 0;
        bindings[i].size = std::numeric_limits<uint64_t>::max();
        i++;

        bindings[i].binding = i;
        bindings[i].buffer = bufferManager->getBuffer(page.vertexBufferName);
        bindings[i].offset = 0;
        bindings[i].size = std::numeric_limits<uint64_t>::max();
        i++;

        bindings[i].binding = i;
        bindings[i].buffer = bufferManager->getBuffer(page.indexBufferName);
        bindings[i].offset = 0;
        bindings[i].size = std::numeric_limits<uint64_t>::max();
        i++;

        if (!pipelineManager->createBindGroup(page.bindGroupName, "global_uniforms", bindings)) {
            return false;
        }
    }
    return true;
}

bool WebGPURenderer::createMeshletPage(uint32_t pageIndex) {
    MeshletPage page;
    if (pageIndex == 0u) {
        page.metadataBufferName = "meshlet_metadata_storage";
        page.vertexBufferName = "meshlet_vertex_storage";
        page.indexBufferName = "meshlet_index_storage";
        page.bindGroupName = "global_uniforms_bg";
    } else {
        page.metadataBufferName = "meshlet_metadata_storage_" + std::to_string(pageIndex);
        page.vertexBufferName = "meshlet_vertex_storage_" + std::to_string(pageIndex);
        page.indexBufferName = "meshlet_index_storage_" + std::to_string(pageIndex);
        page.bindGroupName = "global_uniforms_bg_page_" + std::to_string(pageIndex);
    }

    BufferDescriptor metaDesc = Default;
    metaDesc.label = StringView("meshlet metadata storage page");
    metaDesc.size = static_cast<uint64_t>(meshletMetadataCapacity_) * sizeof(GpuMeshletMetadata);
    metaDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    metaDesc.mappedAtCreation = false;
    bufferManager->deleteBuffer(page.metadataBufferName);
    if (!bufferManager->createBuffer(page.metadataBufferName, metaDesc)) {
        return false;
    }

    BufferDescriptor vertexDesc = Default;
    vertexDesc.label = StringView("meshlet vertex storage page");
    vertexDesc.size =
        static_cast<uint64_t>(meshletSlotsPerPage_) *
        static_cast<uint64_t>(MeshData::kMeshletVertexCapacity) *
        sizeof(PackedVertexAttributes);
    vertexDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    vertexDesc.mappedAtCreation = false;
    bufferManager->deleteBuffer(page.vertexBufferName);
    if (!bufferManager->createBuffer(page.vertexBufferName, vertexDesc)) {
        return false;
    }

    BufferDescriptor indexDesc = Default;
    indexDesc.label = StringView("meshlet index storage page");
    indexDesc.size =
        static_cast<uint64_t>(meshletSlotsPerPage_) *
        static_cast<uint64_t>(MeshData::kMeshletIndexCapacity) *
        sizeof(uint32_t);
    indexDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    indexDesc.mappedAtCreation = false;
    bufferManager->deleteBuffer(page.indexBufferName);
    if (!bufferManager->createBuffer(page.indexBufferName, indexDesc)) {
        return false;
    }

    page.freeSlots.reserve(meshletSlotsPerPage_);
    for (uint32_t i = 0; i < meshletSlotsPerPage_; ++i) {
        page.freeSlots.push_back(meshletSlotsPerPage_ - 1u - i);
    }
    meshletPages_.push_back(std::move(page));
    return true;
}

bool WebGPURenderer::initializeMeshletBuffers(uint32_t meshletCapacity, uint32_t metadataCapacity) {
    meshletSlotsPerPage_ = std::max(1u, meshletCapacity);
    meshletMetadataCapacity_ = std::max(1u, metadataCapacity);

    for (const MeshletPage& page : meshletPages_) {
        bufferManager->deleteBuffer(page.metadataBufferName);
        bufferManager->deleteBuffer(page.vertexBufferName);
        bufferManager->deleteBuffer(page.indexBufferName);
        pipelineManager->deleteBindGroup(page.bindGroupName);
    }
    meshletPages_.clear();

    meshletCapacityWarningEmitted_ = false;
    metadataCapacityWarningEmitted_ = false;

    for (uint32_t pageIndex = 0; pageIndex < kInitialMeshletPageCount; ++pageIndex) {
        if (!createMeshletPage(pageIndex)) {
            return false;
        }
    }

    return refreshVoxelBindGroup();
}

uint32_t WebGPURenderer::acquireMeshletSlot(uint32_t& outPageIndex) {
    for (uint32_t pageIndex = 0; pageIndex < meshletPages_.size(); ++pageIndex) {
        MeshletPage& page = meshletPages_[pageIndex];
        if (!page.freeSlots.empty()) {
            const uint32_t slot = page.freeSlots.back();
            page.freeSlots.pop_back();
            outPageIndex = pageIndex;
            return slot;
        }
    }

    if (meshletPages_.size() >= kMaxMeshletPages) {
        if (!meshletCapacityWarningEmitted_) {
            const uint64_t totalSlots = static_cast<uint64_t>(kMaxMeshletPages) * static_cast<uint64_t>(meshletSlotsPerPage_);
            std::cerr << "Meshlet page capacity reached (" << totalSlots
                      << " meshlets across " << kMaxMeshletPages
                      << " pages). Skipping additional mesh uploads." << std::endl;
            meshletCapacityWarningEmitted_ = true;
        }
        return std::numeric_limits<uint32_t>::max();
    }

    const uint32_t newPageIndex = static_cast<uint32_t>(meshletPages_.size());
    if (!createMeshletPage(newPageIndex) || !refreshVoxelBindGroup()) {
        return std::numeric_limits<uint32_t>::max();
    }

    MeshletPage& newPage = meshletPages_.back();
    const uint32_t slot = newPage.freeSlots.back();
    newPage.freeSlots.pop_back();
    outPageIndex = newPageIndex;
    return slot;
}

void WebGPURenderer::releaseMeshletSlot(uint32_t pageIndex, uint32_t slotInPage) {
    if (pageIndex >= meshletPages_.size() || slotInPage >= meshletSlotsPerPage_) {
        return;
    }
    meshletPages_[pageIndex].freeSlots.push_back(slotInPage);
}

bool WebGPURenderer::uploadMesh(MeshData&& meshData, MeshSlotRef& outMeshSlot) {
    outMeshSlot = {};

    if (meshData.empty()) {
        return false;
    }

    outMeshSlot.meshlets.reserve(meshData.meshlets.size());

    for (const MeshletInfo& meshlet : meshData.meshlets) {
        uint32_t pageIndex = 0;
        const uint32_t slot = acquireMeshletSlot(pageIndex);
        if (slot == std::numeric_limits<uint32_t>::max()) {
            for (const UploadedMeshletRef& uploaded : outMeshSlot.meshlets) {
                releaseMeshletSlot(uploaded.pageIndex, uploaded.slotInPage);
            }
            outMeshSlot = {};
            return false;
        }

        const std::size_t globalVertexOffset = static_cast<std::size_t>(slot) * MeshData::kMeshletVertexCapacity;
        const std::size_t globalIndexOffset = static_cast<std::size_t>(slot) * MeshData::kMeshletIndexCapacity;
        const MeshletPage& page = meshletPages_[pageIndex];

        bufferManager->writeBuffer(
            page.vertexBufferName,
            static_cast<uint64_t>(globalVertexOffset) * sizeof(PackedVertexAttributes),
            meshData.packedVertices.data() + meshlet.vertexOffset,
            static_cast<size_t>(MeshData::kMeshletVertexCapacity) * sizeof(PackedVertexAttributes)
        );
        bufferManager->writeBuffer(
            page.indexBufferName,
            static_cast<uint64_t>(globalIndexOffset) * sizeof(uint32_t),
            meshData.packedIndices.data() + meshlet.indexOffset,
            static_cast<size_t>(MeshData::kMeshletIndexCapacity) * sizeof(uint32_t)
        );

        outMeshSlot.meshlets.push_back(UploadedMeshletRef{
            pageIndex,
            slot,
            meshlet.origin,
            meshlet.lodLevel,
            meshlet.quadCount
        });
    }

    outMeshSlot.valid = true;
    return true;
}

void WebGPURenderer::releaseMesh(MeshSlotRef& meshSlot) {
    if (!meshSlot.valid) {
        return;
    }

    for (const UploadedMeshletRef& meshlet : meshSlot.meshlets) {
        releaseMeshletSlot(meshlet.pageIndex, meshlet.slotInPage);
    }
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

MeshData WebGPURenderer::buildRegionLodMesh(RegionCoord regionCoord, int lodLevel) const {
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

    MeshData meshData = buildMeshletsFromTriangles(vertices, indices);
    meshData.minBounds = glm::vec3(
        static_cast<float>(originX),
        static_cast<float>(originY),
        0.0f
    );
    meshData.maxBounds = glm::vec3(
        static_cast<float>(originX + REGION_BLOCKS_XY),
        static_cast<float>(originY + REGION_BLOCKS_XY),
        static_cast<float>(terrainHeightMaxBlocks_)
    );
    return meshData;
}

bool WebGPURenderer::isRegionLodBuildQueued(const RegionCoord& coord, int lodLevel) const {
    if (buildInFlight_ && activeBuildLodLevel_ == lodLevel && activeBuildCoord_ == coord) {
        return true;
    }

    for (const PendingRegionBuild& pending : pendingBuilds_) {
        if (pending.lodLevel == lodLevel && pending.coord == coord) {
            return true;
        }
    }
    return false;
}

void WebGPURenderer::enqueueMissingRegionLodBuilds(int centerRegionX, int centerRegionY, const glm::vec2& cameraXY) {
    for (int ring = 0; ring <= regionRadius_; ++ring) {
        for (const RegionCoord& coord : drawOrder_) {
            const int rx = coord.x() - centerRegionX;
            const int ry = coord.y() - centerRegionY;
            const int coordRing = std::max(std::abs(rx), std::abs(ry));
            if (coordRing != ring) {
                continue;
            }

            auto regionIt = renderedRegions_.find(coord);
            if (regionIt == renderedRegions_.end()) {
                continue;
            }
            RegionRenderEntry& entry = regionIt->second;

            const glm::vec2 regionCenter{
                (static_cast<float>(coord.x()) + 0.5f) * static_cast<float>(REGION_BLOCKS_XY),
                (static_cast<float>(coord.y()) + 0.5f) * static_cast<float>(REGION_BLOCKS_XY),
            };
            const float distance = glm::distance(cameraXY, regionCenter);
            const int targetLod = chooseLodByDistance(distance);
            for (int lod = kLodCount - 1; lod >= targetLod; --lod) {
                if (entry.lodMeshes[static_cast<std::size_t>(lod)].valid) {
                    continue;
                }
                if (isRegionLodBuildQueued(coord, lod)) {
                    continue;
                }
                pendingBuilds_.push_back({coord, lod});
            }
        }
    }
}

void WebGPURenderer::rebuildRegionsAroundPlayer(int centerRegionX, int centerRegionY, const glm::vec2& cameraXY) {
    if (centerRegionX == activeCenterRegionX && centerRegionY == activeCenterRegionY) {
        enqueueMissingRegionLodBuilds(centerRegionX, centerRegionY, cameraXY);
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
    enqueueMissingRegionLodBuilds(centerRegionX, centerRegionY, cameraXY);

    activeCenterRegionX = centerRegionX;
    activeCenterRegionY = centerRegionY;
}

void WebGPURenderer::processPendingRegionBuilds() {
    if (buildInFlight_ && activeBuildFuture_.valid()) {
        if (activeBuildFuture_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            try {
                CompletedRegionBuild completed = activeBuildFuture_.get();
                buildInFlight_ = false;

                const int lodLevel = completed.lodLevel;
                if (lodLevel >= 0 && lodLevel < kLodCount) {
                    auto regionIt = renderedRegions_.find(completed.coord);
                    if (regionIt != renderedRegions_.end()) {
                        RegionRenderEntry& entry = regionIt->second;
                        MeshSlotRef& mesh = entry.lodMeshes[static_cast<std::size_t>(lodLevel)];
                        if (!mesh.valid) {
                            uploadMesh(std::move(completed.meshData), mesh);
                        }
                    }
                }
                activeBuildLodLevel_ = -1;
            } catch (const std::exception& ex) {
                std::cerr << "Background region build failed: " << ex.what() << std::endl;
                buildInFlight_ = false;
                activeBuildLodLevel_ = -1;
            } catch (...) {
                std::cerr << "Background region build failed with unknown error." << std::endl;
                buildInFlight_ = false;
                activeBuildLodLevel_ = -1;
            }
        }
    }

    if (buildInFlight_) {
        return;
    }

    while (!pendingBuilds_.empty()) {
        const PendingRegionBuild build = pendingBuilds_.front();
        pendingBuilds_.pop_front();

        if (build.lodLevel < 0 || build.lodLevel >= kLodCount) {
            continue;
        }

        auto regionIt = renderedRegions_.find(build.coord);
        if (regionIt == renderedRegions_.end()) {
            continue;
        }

        RegionRenderEntry& entry = regionIt->second;
        if (entry.lodMeshes[static_cast<std::size_t>(build.lodLevel)].valid) {
            continue;
        }

        const RegionCoord coord = build.coord;
        const int lodLevel = build.lodLevel;
        activeBuildCoord_ = coord;
        activeBuildLodLevel_ = lodLevel;
        activeBuildFuture_ = std::async(std::launch::async, [this, coord, lodLevel]() {
            CompletedRegionBuild completed;
            completed.coord = coord;
            completed.lodLevel = lodLevel;
            completed.meshData = buildRegionLodMesh(coord, lodLevel);
            return completed;
        });
        buildInFlight_ = true;
        break;
    }
}

void WebGPURenderer::drawRegionSet(RenderPassEncoder& pass, const glm::vec3& cameraPos) {
    const glm::vec2 cameraXY{cameraPos.x, cameraPos.y};
    lastFrameRequestedMeshlets_ = 0u;
    lastFrameDrawnMeshlets_ = 0u;
    lastFrameMetadataCulledMeshlets_ = 0u;
    lastFrameBudgetCulledMeshlets_ = 0u;
    lastFramePagesDrawn_ = 0u;

    if (meshletPages_.empty()) {
        return;
    }
    std::vector<std::vector<GpuMeshletMetadata>> meshletMetadataByPage(meshletPages_.size());

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

        if (mesh == nullptr || mesh->meshlets.empty()) {
            continue;
        }

        lastFrameRequestedMeshlets_ += static_cast<uint32_t>(mesh->meshlets.size());
        for (const UploadedMeshletRef& meshlet : mesh->meshlets) {
            if (meshlet.pageIndex >= meshletPages_.size()) {
                continue;
            }
            const uint32_t slotInPage = meshlet.slotInPage;
            meshletMetadataByPage[meshlet.pageIndex].push_back(GpuMeshletMetadata{
                meshlet.origin.x,
                meshlet.origin.y,
                meshlet.origin.z,
                static_cast<int32_t>(meshlet.lodLevel),
                slotInPage * MeshData::kMeshletVertexCapacity,
                slotInPage * MeshData::kMeshletIndexCapacity,
                meshlet.quadCount,
                0u
            });
        }
    }

    uint32_t remainingGlobalBudget = kMaxDrawMeshletsPerFrame;
    for (uint32_t pageIndex = 0; pageIndex < meshletPages_.size(); ++pageIndex) {
        if (remainingGlobalBudget == 0) {
            break;
        }

        std::vector<GpuMeshletMetadata>& pageMetadata = meshletMetadataByPage[pageIndex];
        if (pageMetadata.empty()) {
            continue;
        }

        uint32_t drawInstanceCount = static_cast<uint32_t>(pageMetadata.size());
        const uint32_t originalInstanceCount = drawInstanceCount;
        if (drawInstanceCount > meshletMetadataCapacity_) {
            if (!metadataCapacityWarningEmitted_) {
                std::cerr << "Meshlet metadata capacity reached (" << meshletMetadataCapacity_
                          << " instances/page). Culling excess meshlets." << std::endl;
                metadataCapacityWarningEmitted_ = true;
            }
            drawInstanceCount = meshletMetadataCapacity_;
        }
        if (drawInstanceCount < originalInstanceCount) {
            lastFrameMetadataCulledMeshlets_ += (originalInstanceCount - drawInstanceCount);
        }
        const uint32_t beforeGlobalBudgetClamp = drawInstanceCount;
        drawInstanceCount = std::min(drawInstanceCount, remainingGlobalBudget);
        if (drawInstanceCount < beforeGlobalBudgetClamp) {
            lastFrameBudgetCulledMeshlets_ += (beforeGlobalBudgetClamp - drawInstanceCount);
        }
        if (drawInstanceCount == 0) {
            continue;
        }

        bufferManager->writeBuffer(
            meshletPages_[pageIndex].metadataBufferName,
            0,
            pageMetadata.data(),
            static_cast<std::size_t>(drawInstanceCount) * sizeof(GpuMeshletMetadata)
        );

        const BindGroup pageBindGroup = pipelineManager->getBindGroup(meshletPages_[pageIndex].bindGroupName);
        if (!pageBindGroup) {
            continue;
        }

        pass.setBindGroup(0, pageBindGroup, 0, nullptr);
        pass.draw(MeshData::kMeshletIndexCapacity, drawInstanceCount, 0, 0);
        remainingGlobalBudget -= drawInstanceCount;
        lastFrameDrawnMeshlets_ += drawInstanceCount;
        ++lastFramePagesDrawn_;
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

    const int configuredViewDistanceChunks = std::clamp(tuning.viewDistanceChunks, 1, 500);
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
    if (!initializeMeshletBuffers(kMeshletsPerPage, kInitialMeshletMetadataCapacity)) {
        std::cerr << "Failed to create meshlet storage buffers." << std::endl;
        return false;
    }

    return true;
}

void WebGPURenderer::createRenderingTextures() {
    if (!voxelPipeline.createResources()) {
        std::cerr << "Failed to recreate voxel rendering resources." << std::endl;
        return;
    }
    if (!refreshVoxelBindGroup()) {
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

void WebGPURenderer::toggleMeshletDebugColors() {
    debugRenderFlags_ ^= DebugMeshletColors;
}

void WebGPURenderer::clearDebugColors() {
    debugRenderFlags_ = 0u;
}

uint32_t WebGPURenderer::debugRenderFlags() const {
    return debugRenderFlags_;
}

RendererDebugSnapshot WebGPURenderer::debugSnapshot() const {
    RendererDebugSnapshot snapshot;
    snapshot.renderedRegionCount = renderedRegions_.size();
    snapshot.drawOrderRegionCount = drawOrder_.size();
    snapshot.pendingRegionBuildCount = pendingBuilds_.size();
    snapshot.buildInFlight = buildInFlight_;
    snapshot.lastFrameRequestedMeshlets = lastFrameRequestedMeshlets_;
    snapshot.lastFrameDrawnMeshlets = lastFrameDrawnMeshlets_;
    snapshot.lastFrameMetadataCulledMeshlets = lastFrameMetadataCulledMeshlets_;
    snapshot.lastFrameBudgetCulledMeshlets = lastFrameBudgetCulledMeshlets_;
    snapshot.lastFramePagesDrawn = lastFramePagesDrawn_;

    snapshot.meshletPages.activePageCount = static_cast<uint32_t>(meshletPages_.size());
    snapshot.meshletPages.maxPageCount = kMaxMeshletPages;
    snapshot.meshletPages.slotsPerPage = meshletSlotsPerPage_;
    snapshot.meshletPages.metadataCapacityPerPage = meshletMetadataCapacity_;
    snapshot.meshletPages.totalSlotCapacity =
        static_cast<uint64_t>(meshletPages_.size()) * static_cast<uint64_t>(meshletSlotsPerPage_);

    uint64_t totalFreeSlots = 0;
    for (const MeshletPage& page : meshletPages_) {
        totalFreeSlots += static_cast<uint64_t>(page.freeSlots.size());
    }
    snapshot.meshletPages.freeSlots = totalFreeSlots;
    snapshot.meshletPages.usedSlots = snapshot.meshletPages.totalSlotCapacity - totalFreeSlots;
    return snapshot;
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
    rebuildRegionsAroundPlayer(regionX, regionY, glm::vec2{cameraPos.x, cameraPos.y});
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
    if (activeBuildFuture_.valid()) {
        try {
            activeBuildFuture_.wait();
            (void)activeBuildFuture_.get();
        } catch (...) {
        }
    }
    buildInFlight_ = false;
    releaseAllRegionMeshes();
    textureManager->terminate();
    pipelineManager->terminate();
    bufferManager->terminate();
}
