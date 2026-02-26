#pragma once

#include "solum_engine/render/pipelines/AbstractRenderPipeline.h"

#include <cstdint>
#include <string>

class VoxelPipeline : public AbstractRenderPipeline {
public:
    explicit VoxelPipeline(RenderServices& r) : AbstractRenderPipeline(r) {}

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
        TextureView targetView,
        CommandEncoder encoder,
        const std::function<void(RenderPassEncoder&)>& overlayCallback = {}
    ) override;
private:
    uint32_t meshletVertexCount = 0;
    uint32_t meshletCount = 0;
    bool useIndirectDraw_ = false;
    std::string indirectDrawBufferName_;
    uint64_t indirectDrawOffset_ = 0u;
};
