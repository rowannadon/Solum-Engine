#pragma once

#include "solum_engine/render/MeshletBufferController.h"
#include "solum_engine/render/pipelines/AbstractRenderPipeline.h"

#include <cstdint>
#include <string>

class MeshletCullingPipeline : public AbstractRenderPipeline {
public:
    static constexpr const char* kCullParamsBufferName = "meshlet_cull_params_buffer";
    static constexpr const char* kIndirectArgsBufferName = "meshlet_cull_indirect_args_buffer";
    static constexpr const char* kIndirectResetBufferName = "meshlet_cull_indirect_reset_buffer";

    explicit MeshletCullingPipeline(RenderServices& r) : AbstractRenderPipeline(r) {}

    bool build() override;
    bool build(const MeshletBufferController& meshletBuffers,
               uint32_t occlusionHiZMipCount,
               const char* occlusionHiZViewName);
    bool refreshBindGroup(const MeshletBufferController& meshletBuffers,
                          const char* occlusionHiZViewName);

    void updateCullParams(uint32_t meshletCount, uint32_t occlusionHiZMipCount);
    void encode(wgpu::CommandEncoder encoder, const MeshletBufferController& meshletBuffers);

    bool createResources() override;
    void removeResources() override;
    bool createPipeline() override;
    bool createBindGroup() override;
    bool render(
        wgpu::TextureView targetView,
        wgpu::CommandEncoder encoder,
        const std::function<void(wgpu::RenderPassEncoder&)>& overlayCallback = {}
    ) override;

private:
    bool createBindGroupForMeshBuffers(const std::string& meshletAabbBufferName,
                                       const std::string& visibleIndicesBufferName,
                                       const char* occlusionHiZViewName);

    static constexpr const char* kCullBglName = "meshlet_cull_bgl";
    static constexpr const char* kCullBgName = "meshlet_cull_bg";
    static constexpr const char* kCullPipelineName = "meshlet_cull_pipeline";
    static constexpr const char* kDefaultHiZViewName = "meshlet_occlusion_hiz_view";

    static constexpr uint32_t kMeshletCullWorkgroupSize = 128u;

    std::string activeHiZViewName_ = kDefaultHiZViewName;
};
