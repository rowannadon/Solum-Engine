#pragma once

#include "solum_engine/render/pipelines/AbstractRenderPipeline.h"
#include "solum_engine/resources/Coords.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include <glm/glm.hpp>

struct SelectionOutlineVertex {
    glm::vec4 lineStartAndAlong{0.0f};
    glm::vec4 lineEndAndSide{0.0f};
    glm::vec4 color{1.0f};
};

class SelectionOutlinePipeline : public AbstractRenderPipeline {
public:
    explicit SelectionOutlinePipeline(RenderServices& r) : AbstractRenderPipeline(r) {}

    bool setSelectedBlock(const std::optional<BlockCoord>& selectedBlock);

    bool createResources() override;
    void removeResources() override;
    bool createPipeline() override;
    bool createBindGroup() override;
    bool build() override;

    bool render(
        wgpu::TextureView targetView,
        wgpu::CommandEncoder encoder,
        const std::function<void(wgpu::RenderPassEncoder&)>& overlayCallback = {}
    );

private:
    bool ensureVertexBufferCapacity(uint64_t requiredBytes);
    bool updateVertices(const std::vector<SelectionOutlineVertex>& vertices);

    std::optional<BlockCoord> selectedBlock_;
    uint32_t vertexCount_ = 0;
    uint64_t vertexCapacityBytes_ = 0;
};
