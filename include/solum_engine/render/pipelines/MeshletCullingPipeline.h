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
    bool createBindGroupForSceneBuffers(const std::string& visibleTileIdBufferName,
                                        const std::string& tileSlotBufferName,
                                        const std::string& residentTileLodBufferName,
                                        const std::string& visibleIndicesBufferName,
                                        const std::string& tileSceneParamsBufferName);

    static constexpr uint32_t kTileExpansionWorkgroupSize = 64u;

    static std::string prefixedName(const std::string& prefix, const char* baseName);

    std::string indirectArgsBufferName_;
    std::string indirectResetBufferName_;
    std::string cullBglName_;
    std::string cullBgName_;
    std::string cullPipelineName_;
};
