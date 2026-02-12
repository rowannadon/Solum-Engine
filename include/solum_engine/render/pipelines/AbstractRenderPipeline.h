#include "solum_engine/render/BufferManager.h"
#include "solum_engine/render/TextureManager.h"
#include "solum_engine/render/PipelineManager.h"
#include "solum_engine/platform/WebGPUContext.h"
#include <functional>

class AbstractRenderPipeline {
public:
    BufferManager* buf;
	TextureManager* tex;
	PipelineManager* pip;
	WebGPUContext* context;

    void init(BufferManager* b, TextureManager* t, PipelineManager* p, WebGPUContext *con) {
        buf = b;
        tex = t;
        pip = p;
        context = con;
    }

    virtual bool createResources() = 0;

    virtual void removeResources() = 0;

    virtual bool createPipeline() = 0;

    virtual bool createBindGroup() = 0;

    virtual bool render(
        TextureView targetView,
        CommandEncoder encoder,
        const std::function<void(RenderPassEncoder&)>& overlayCallback = {}
    ) = 0;

};
