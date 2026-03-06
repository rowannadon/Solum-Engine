#pragma once

#include "solum_engine/render/MeshletBufferController.h"
#include "solum_engine/render/pipelines/AbstractRenderPipeline.h"

#include <cstdint>
#include <string>

class MeshletCullingPipeline : public AbstractRenderPipeline {
public:
    struct Config {
        std::string namePrefix;
    };

    explicit MeshletCullingPipeline(RenderServices& r, Config config = {});

    bool build() override;
    bool build(const MeshletBufferController& meshletBuffers,
               uint32_t occlusionHiZMipCount,
               const char* occlusionHiZViewName);
    bool refreshBindGroup(const MeshletBufferController& meshletBuffers,
                          const char* occlusionHiZViewName);
    const std::string& indirectArgsBufferName() const noexcept;

    void updateCullParams(uint32_t meshletCount, uint32_t occlusionHiZMipCount, uint32_t activeRangeCount);
    void encode(wgpu::CommandEncoder encoder, const MeshletBufferController& meshletBuffers);

    bool createResources() override;
    void removeResources() override;
    bool createPipeline() override;
    bool createBindGroup() override;

private:
    bool createBindGroupForMeshBuffers(const std::string& meshletAabbBufferName,
                                       const std::string& visibleIndicesBufferName,
                                       const std::string& activeRangeBufferName,
                                       const char* occlusionHiZViewName);

    static constexpr const char* kDefaultHiZViewName = "meshlet_occlusion_hiz_view";

    static constexpr uint32_t kMeshletCullWorkgroupSize = 128u;

    static std::string prefixedName(const std::string& prefix, const char* baseName);

    std::string cullParamsBufferName_;
    std::string indirectArgsBufferName_;
    std::string indirectResetBufferName_;
    std::string cullBglName_;
    std::string cullBgName_;
    std::string cullPipelineName_;
    std::string activeHiZViewName_ = kDefaultHiZViewName;
};
