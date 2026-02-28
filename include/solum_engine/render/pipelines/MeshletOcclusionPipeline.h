#pragma once

#include "solum_engine/render/MeshletBufferController.h"
#include "solum_engine/render/pipelines/AbstractRenderPipeline.h"

#include <cstdint>
#include <string>

class MeshletOcclusionPipeline : public AbstractRenderPipeline {
public:
    static constexpr uint32_t kOcclusionDepthDownsample = 2u;
    static constexpr const char* kOcclusionDepthTextureName = "meshlet_occlusion_depth_texture";
    static constexpr const char* kOcclusionDepthViewName = "meshlet_occlusion_depth_view";
    static constexpr const char* kOcclusionHiZTextureName = "meshlet_occlusion_hiz_texture";
    static constexpr const char* kOcclusionHiZViewName = "meshlet_occlusion_hiz_view";

    explicit MeshletOcclusionPipeline(RenderServices& r) : AbstractRenderPipeline(r) {}

    bool build() override;
    bool build(const MeshletBufferController& meshletBuffers);
    bool recreateResources(const MeshletBufferController& meshletBuffers);
    bool refreshMeshBindGroup(const MeshletBufferController& meshletBuffers);

    void encodeDepthPrepass(wgpu::CommandEncoder encoder,
                            const MeshletBufferController& meshletBuffers);
    void encodeHierarchyPass(wgpu::CommandEncoder encoder);

    uint32_t hizMipCount() const noexcept { return occlusionHiZMipCount_; }

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
    bool createBindGroupForMeshBuffers(const std::string& meshDataBufferName,
                                       const std::string& metadataBufferName);
    static uint32_t computeMipCount(uint32_t width, uint32_t height);

    uint32_t occlusionHiZMipCount_ = 1u;
    uint32_t occlusionDepthWidth_ = 1u;
    uint32_t occlusionDepthHeight_ = 1u;

    static constexpr const char* kDepthPrepassBglName = "meshlet_depth_prepass_bgl";
    static constexpr const char* kDepthPrepassBgName = "meshlet_depth_prepass_bg";
    static constexpr const char* kDepthPrepassPipelineName = "meshlet_depth_prepass_pipeline";

    static constexpr const char* kHiZSeedBglName = "meshlet_hiz_seed_bgl";
    static constexpr const char* kHiZDownsampleBglName = "meshlet_hiz_downsample_bgl";
    static constexpr const char* kHiZSeedPipelineName = "meshlet_hiz_seed_pipeline";
    static constexpr const char* kHiZDownsamplePipelineName = "meshlet_hiz_downsample_pipeline";

    static constexpr uint32_t kOcclusionHiZWorkgroupSize = 8u;
};
