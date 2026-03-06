#pragma once

#include "solum_engine/render/pipelines/AbstractRenderPipeline.h"

#include <cstdint>
#include <functional>
#include <string>

class VoxelPipeline : public AbstractRenderPipeline {
public:
    struct Config {
        std::string namePrefix;
        wgpu::CullMode cullMode = wgpu::CullMode::Back;
        bool manageRenderTargets = true;
    };

    struct RenderOptions {
        bool clearColor = true;
        bool clearDepth = true;
    };

    explicit VoxelPipeline(RenderServices& r);
    VoxelPipeline(RenderServices& r, Config config);

    void setDrawConfig(uint32_t meshletVertexCount, uint32_t meshletCount);
    void setIndirectDrawBuffer(const std::string& bufferName, uint64_t offset = 0u);
    void clearIndirectDrawBuffer();

    bool createResources() override;

    void removeResources() override;

    bool createPipeline() override;

    bool createBindGroup() override;
    bool createBindGroupForMeshBuffers(const std::string& meshDataBufferName,
                                       const std::string& metadataBufferName,
                                       const std::string& visibleIndicesBufferName);
    
    bool build() override;

    bool render(
        wgpu::TextureView targetView,
        wgpu::CommandEncoder encoder,
        const RenderOptions& options,
        const std::function<void(wgpu::RenderPassEncoder&)>& overlayCallback = {}
    );
private:
    static std::string prefixedName(const std::string& prefix, const char* baseName);

    std::string pipelineName_;
    std::string bindGroupLayoutName_;
    std::string bindGroupName_;
    wgpu::CullMode cullMode_ = wgpu::CullMode::Back;
    bool manageRenderTargets_ = true;

    uint32_t meshletVertexCount = 0;
    uint32_t meshletCount = 0;
    bool useIndirectDraw_ = false;
    std::string indirectDrawBufferName_;
    uint64_t indirectDrawOffset_ = 0u;
};
