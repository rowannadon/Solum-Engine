#include "solum_engine/render/WebGPURenderer.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <unordered_map>

namespace {
constexpr int kDemoColumnsX = 4;
constexpr int kDemoColumnsY = 4;
constexpr int kDemoChunksZ = 2;

constexpr std::array<glm::ivec3, 6> kNeighborOffsets = {
    glm::ivec3{1, 0, 0},
    glm::ivec3{-1, 0, 0},
    glm::ivec3{0, 1, 0},
    glm::ivec3{0, -1, 0},
    glm::ivec3{0, 0, 1},
    glm::ivec3{0, 0, -1},
};

int sampleHeight(int worldX, int worldY) {
    const int h = (worldX * 31 + worldY * 17 + worldX * worldY) & 15;
    return 12 + h;
}
}

std::pair<std::vector<VertexAttributes>, std::vector<uint32_t>> WebGPURenderer::buildDemoWorldMesh() {
    std::vector<std::unique_ptr<Chunk>> chunks;
    chunks.reserve(static_cast<std::size_t>(kDemoColumnsX * kDemoColumnsY * kDemoChunksZ));

    std::unordered_map<ChunkCoord, Chunk*, ChunkCoordHash> coordToChunk;
    coordToChunk.reserve(chunks.capacity());

    for (int cy = 0; cy < kDemoColumnsY; ++cy) {
        for (int cx = 0; cx < kDemoColumnsX; ++cx) {
            for (int cz = 0; cz < kDemoChunksZ; ++cz) {
                const ChunkCoord coord{cx, cy, cz};
                auto chunk = std::make_unique<Chunk>(coord);
                Chunk* chunkPtr = chunk.get();
                chunks.push_back(std::move(chunk));
                coordToChunk.emplace(coord, chunkPtr);
            }
        }
    }

    UnpackedBlockMaterial air{};
    air.id = 0;

    for (const std::unique_ptr<Chunk>& chunkPtr : chunks) {
        Chunk& chunk = *chunkPtr;
        const int chunkWorldX = chunk.coord().x() * CHUNK_SIZE;
        const int chunkWorldY = chunk.coord().y() * CHUNK_SIZE;
        const int chunkWorldZ = chunk.coord().z() * CHUNK_SIZE;
        BlockMaterial* blockData = chunk.getBlockData();
        if (blockData == nullptr) {
            continue;
        }

        for (int x = 0; x < CHUNK_SIZE; ++x) {
            for (int y = 0; y < CHUNK_SIZE; ++y) {
                const int worldX = chunkWorldX + x;
                const int worldY = chunkWorldY + y;
                const int height = sampleHeight(worldX, worldY);

                for (int z = 0; z < CHUNK_SIZE; ++z) {
                    const int worldZ = chunkWorldZ + z;

                    UnpackedBlockMaterial material{};
                    if (worldZ <= height) {
                        if (worldZ == height) {
                            material.id = 2;
                        } else if (worldZ > height - 3) {
                            material.id = 1;
                        } else {
                            material.id = 7;
                        }
                    } else {
                        material = air;
                    }

                    const std::size_t index = static_cast<std::size_t>((x * CHUNK_SIZE * CHUNK_SIZE) + (y * CHUNK_SIZE) + z);
                    blockData[index] = material.pack();
                }
            }
        }

        chunk.markBulkDataWrite();
    }

    std::vector<VertexAttributes> mergedVertices;
    std::vector<uint32_t> mergedIndices;

    ChunkMesher mesher;
    for (const std::unique_ptr<Chunk>& chunkPtr : chunks) {
        Chunk& chunk = *chunkPtr;
        std::vector<Chunk*> neighbors(6, nullptr);
        for (int i = 0; i < 6; ++i) {
            const ChunkCoord neighborCoord{
                chunk.coord().x() + kNeighborOffsets[static_cast<std::size_t>(i)].x,
                chunk.coord().y() + kNeighborOffsets[static_cast<std::size_t>(i)].y,
                chunk.coord().z() + kNeighborOffsets[static_cast<std::size_t>(i)].z,
            };

            const auto it = coordToChunk.find(neighborCoord);
            if (it != coordToChunk.end()) {
                neighbors[static_cast<std::size_t>(i)] = it->second;
            }
        }

        auto [vertices, indices] = mesher.mesh(chunk, neighbors);
        if (vertices.empty() || indices.empty()) {
            continue;
        }

        const uint32_t baseVertex = static_cast<uint32_t>(mergedVertices.size());
        mergedVertices.insert(mergedVertices.end(), vertices.begin(), vertices.end());

        mergedIndices.reserve(mergedIndices.size() + indices.size());
        for (const uint16_t idx : indices) {
            mergedIndices.push_back(baseVertex + idx);
        }
    }

    return {std::move(mergedVertices), std::move(mergedIndices)};
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

    auto [vertexData, indexData] = buildDemoWorldMesh();
    worldVertexCount = static_cast<uint32_t>(vertexData.size());
    worldIndexCount = static_cast<uint32_t>(indexData.size());

    if (worldVertexCount == 0 || worldIndexCount == 0) {
        std::cerr << "World mesh generation produced no geometry." << std::endl;
        return false;
    }

    {
        BufferDescriptor desc = Default;
        desc.label = StringView("world vertex buffer");
        desc.size = std::max<uint64_t>(sizeof(VertexAttributes), static_cast<uint64_t>(worldVertexCount) * sizeof(VertexAttributes));
        desc.usage = BufferUsage::CopyDst | BufferUsage::Vertex;
        desc.mappedAtCreation = false;
        Buffer vbo = bufferManager->createBuffer("world_vertex_buffer", desc);
        if (!vbo) {
            return false;
        }
    }

    {
        BufferDescriptor desc = Default;
        desc.label = StringView("world index buffer");
        desc.size = std::max<uint64_t>(sizeof(uint32_t), static_cast<uint64_t>(worldIndexCount) * sizeof(uint32_t));
        desc.usage = BufferUsage::CopyDst | BufferUsage::Index;
        desc.mappedAtCreation = false;
        Buffer ibo = bufferManager->createBuffer("world_index_buffer", desc);
        if (!ibo) {
            return false;
        }
    }

    bufferManager->writeBuffer("world_vertex_buffer", 0, vertexData.data(), static_cast<size_t>(worldVertexCount) * sizeof(VertexAttributes));
    bufferManager->writeBuffer("world_index_buffer", 0, indexData.data(), static_cast<size_t>(worldIndexCount) * sizeof(uint32_t));

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


void WebGPURenderer::renderFrame(FrameUniforms& uniforms) {
    (void)uniforms;

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
    if (!targetView) return;

    CommandEncoderDescriptor encoderDesc = Default;
    encoderDesc.label = StringView("Frame command encoder");
    CommandEncoder encoder = context->getDevice().createCommandEncoder(encoderDesc);

    {
        voxelPipeline.render(targetView, encoder, [&](RenderPassEncoder& pass) {
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
    textureManager->terminate();
    pipelineManager->terminate();
    bufferManager->terminate();
}
