#pragma once

#include "solum_engine/render/pipelines/AbstractRenderPipeline.h"

#include <cstdint>

class VoxelPipeline : public AbstractRenderPipeline {
public:
    explicit VoxelPipeline(RenderServices& r) : AbstractRenderPipeline(r) {}

    void setDrawConfig(uint32_t meshletVertexCount, uint32_t meshletCount);

    bool createResources() override;

    void removeResources() override;

    bool createPipeline() override;

    bool createBindGroup() override;
    
    bool build() override;

    bool render(
        TextureView targetView,
        CommandEncoder encoder,
        const std::function<void(RenderPassEncoder&)>& overlayCallback = {}
    ) override;
private:
    uint32_t meshletVertexCount = 0;
    uint32_t meshletCount = 0;
};
