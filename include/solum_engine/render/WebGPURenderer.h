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

    MeshletBufferController culledMeshletBuffers_{
        MeshletBufferController::Config{"", MeshletGeometryVariant::Culled}
    };
    MeshletBufferController doubleSidedMeshletBuffers_{
        MeshletBufferController::Config{"double_sided", MeshletGeometryVariant::DoubleSided}
    };

    std::optional<RenderServices> services_;
    std::optional<VoxelPipeline> culledVoxelPipeline_;
    std::optional<VoxelPipeline> doubleSidedVoxelPipeline_;
    std::optional<MeshletOcclusionPipeline> meshletOcclusionPipeline_;
    std::optional<MeshletCullingPipeline> culledMeshletCullingPipeline_;
    std::optional<MeshletCullingPipeline> doubleSidedMeshletCullingPipeline_;
    std::optional<BoundsDebugPipeline> boundsDebugPipeline_;

    DebugBoundsManager debugBoundsManager_;
    RuntimeTimingTracker timingTracker_;

    bool resizePending = false;
    std::optional<MeshStreamingDelta> pendingMeshDelta_;

#ifdef __APPLE__
    static constexpr uint32_t kMaxFramesInFlight = 1u;
#else
    static constexpr uint32_t kMaxFramesInFlight = 2u;
#endif
    std::atomic<uint32_t> framesInFlight_{0};

    bool gpuStallDetected_{false};
    uint32_t consecutiveSlowFrames_{0};
    static constexpr uint32_t kMaxConsecutiveSlowFrames = 15u;

    bool checkGpuStall(const char* stage, std::chrono::steady_clock::time_point start,
                       std::chrono::milliseconds threshold);

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
