#include "solum_engine/render/BufferManager.h"
#include "solum_engine/render/TextureManager.h"
#include "solum_engine/render/PipelineManager.h"
#include "solum_engine/platform/WebGPUContext.h"
#include <functional>

struct RenderServices {
    BufferManager& buf;
    TextureManager& tex;
    PipelineManager& pip;
    WebGPUContext& ctx;
};

class AbstractRenderPipeline {
protected:
    RenderServices& r_;

public:
    explicit AbstractRenderPipeline(RenderServices& r) : r_(r) {}
    virtual ~AbstractRenderPipeline() = default;

    AbstractRenderPipeline(const AbstractRenderPipeline&) = delete;
    AbstractRenderPipeline& operator=(const AbstractRenderPipeline&) = delete;
    AbstractRenderPipeline(AbstractRenderPipeline&&) = delete;
    AbstractRenderPipeline& operator=(AbstractRenderPipeline&&) = delete;

    virtual bool createResources() = 0;
    virtual void removeResources() = 0;
    virtual bool createPipeline() = 0;
    virtual bool createBindGroup() = 0;
    virtual bool build() = 0;

    virtual bool render(
        TextureView targetView,
        CommandEncoder encoder,
        const std::function<void(RenderPassEncoder&)>& overlayCallback = {}
    ) = 0;
};