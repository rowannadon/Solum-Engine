#pragma once

#include "solum_engine/render/pipelines/AbstractRenderPipeline.h"

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

struct DebugLineVertex {
    glm::vec3 position{0.0f};
    glm::vec4 color{1.0f};
};

class BoundsDebugPipeline : public AbstractRenderPipeline {
public:
    explicit BoundsDebugPipeline(RenderServices& r) : AbstractRenderPipeline(r) {}

    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }

    bool updateVertices(const std::vector<DebugLineVertex>& vertices);
    void draw(wgpu::RenderPassEncoder& renderPass);

    bool createResources() override;
    void removeResources() override;
    bool createPipeline() override;
    bool createBindGroup() override;
    bool build() override;

    bool render(
        wgpu::TextureView targetView,
        wgpu::CommandEncoder encoder,
        const std::function<void(wgpu::RenderPassEncoder&)>& overlayCallback = {}
    ) override;

private:
    bool ensureVertexBufferCapacity(uint64_t requiredBytes);

    bool enabled_ = false;
    uint32_t vertexCount_ = 0;
    uint64_t vertexCapacityBytes_ = 0;
};
