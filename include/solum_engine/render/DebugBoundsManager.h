#pragma once

#include <cstdint>

#include "solum_engine/render/Uniforms.h"
#include "solum_engine/render/MeshletBufferController.h"
#include "solum_engine/render/pipelines/BoundsDebugPipeline.h"

class World;

class DebugBoundsManager {
public:
    void setWorld(const World* world);
    void reset();

    void update(const FrameUniforms& frameUniforms,
                BoundsDebugPipeline& boundsPipeline,
                const MeshletBufferController& meshletBuffers);

private:
    void rebuild(uint32_t layerMask,
                 BoundsDebugPipeline& boundsPipeline,
                 const MeshletBufferController& meshletBuffers) const;

    const World* world_ = nullptr;
    uint64_t uploadedWorldRevision_ = 0;
    uint64_t uploadedMeshRevision_ = 0;
    uint32_t uploadedLayerMask_ = 0u;
};
