#pragma once

#include <webgpu/webgpu.hpp>
#include <GLFW/glfw3.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "solum_engine/platform/WebGPUContext.h"
#include "solum_engine/render/BufferManager.h"
#include "solum_engine/render/DebugBoundsManager.h"
#include "solum_engine/render/MaterialManager.h"
#include "solum_engine/render/MeshletBufferController.h"
#include "solum_engine/render/PipelineManager.h"
#include "solum_engine/render/RuntimeTiming.h"
#include "solum_engine/render/RuntimeTimingTracker.h"
#include "solum_engine/render/TextureManager.h"
#include "solum_engine/render/Uniforms.h"
#include "solum_engine/render/pipelines/BoundsDebugPipeline.h"
#include "solum_engine/render/pipelines/MeshletCullingPipeline.h"
#include "solum_engine/render/pipelines/MeshletOcclusionPipeline.h"
#include "solum_engine/render/pipelines/VoxelPipeline.h"
#include "solum_engine/voxel/StreamingUpload.h"

class World;

class WebGPURenderer {
public:
    struct Config {};

private:
    std::unique_ptr<WebGPUContext> context;
    std::unique_ptr<PipelineManager> pipelineManager;
    std::unique_ptr<BufferManager> bufferManager;
    std::unique_ptr<TextureManager> textureManager;
    std::unique_ptr<MaterialManager> materialManager;

    MeshletBufferController meshletBuffers_;

    std::optional<RenderServices> services_;
    std::optional<VoxelPipeline> voxelPipeline_;
    std::optional<MeshletOcclusionPipeline> meshletOcclusionPipeline_;
    std::optional<MeshletCullingPipeline> meshletCullingPipeline_;
    std::optional<BoundsDebugPipeline> boundsDebugPipeline_;

    DebugBoundsManager debugBoundsManager_;
    RuntimeTimingTracker timingTracker_;

    bool resizePending = false;
    std::optional<MeshStreamingDelta> pendingMeshDelta_;

    static constexpr uint32_t kMaxFramesInFlight = 2u;
    std::atomic<uint32_t> framesInFlight_{0};

    bool refreshMeshBindings(bool uploadApplied, bool rebuildDrawConfig);
    void processPendingMeshUploads();

public:
    WebGPURenderer() = default;
    ~WebGPURenderer() = default;

    bool initialize();
    bool initialize(const Config& config);

    PipelineManager* getPipelineManager();
    BufferManager* getBufferManager();
    TextureManager* getTextureManager();
    WebGPUContext* getContext();
    GLFWwindow* getWindow();
    std::shared_ptr<const BlockModelLibrary> getBlockModelLibrary() const;

    void removeRenderingTextures();
    void createRenderingTextures();
    bool resizeSurfaceAndAttachments();
    void requestResize();

    std::pair<wgpu::SurfaceTexture, wgpu::TextureView> GetNextSurfaceViewData();
    RuntimeTimingSnapshot getRuntimeTimingSnapshot();

    void setDebugWorld(const World* world);
    void queueMeshDelta(MeshStreamingDelta&& delta);
    bool isMeshUploadInProgress() const noexcept;
    uint64_t uploadedMeshRevision() const noexcept;

    void renderFrame(FrameUniforms& uniforms);

    void terminate();
};
