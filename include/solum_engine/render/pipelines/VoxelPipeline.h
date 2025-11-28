#include "solum_engine/render/pipelines/AbstractRenderPipeline.h"

class VoxelPipeline : public AbstractRenderPipeline {
public:
    bool createResources() override;

    void removeResources() override;

    bool createPipeline() override;

    bool createBindGroup() override;

    bool render(TextureView targetView, CommandEncoder encoder) override;

};
