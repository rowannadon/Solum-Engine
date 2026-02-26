#pragma once

#include <webgpu/webgpu.hpp>
#include <GLFW/glfw3.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "solum_engine/render/PipelineManager.h"
#include "solum_engine/render/BufferManager.h"
#include "solum_engine/render/TextureManager.h"
#include "solum_engine/render/MaterialManager.h"
#include "solum_engine/platform/WebGPUContext.h"
#include "solum_engine/render/Uniforms.h"
#include "solum_engine/render/RuntimeTiming.h"
#include "solum_engine/render/pipelines/BoundsDebugPipeline.h"
#include "solum_engine/render/pipelines/VoxelPipeline.h"
#include "solum_engine/render/MeshletManager.h"
#include "solum_engine/voxel/StreamingUpload.h"

class World;

class WebGPURenderer {
private:
    enum class TimingStage : std::size_t {
        MainUploadMeshlets = 0,
        MainUpdateDebugBounds,
        MainRenderFrameCpu,
        MainAcquireSurface,
        MainEncodeCommands,
        MainQueueSubmit,
        MainPresent,
        MainDeviceTick,
        Count
    };

    struct TimingAccumulator {
        std::atomic<uint64_t> totalNs{0};
        std::atomic<uint64_t> callCount{0};
        std::atomic<uint64_t> maxNs{0};
    };

    struct TimingRawTotals {
        std::array<uint64_t, static_cast<std::size_t>(TimingStage::Count)> totalNs{};
        std::array<uint64_t, static_cast<std::size_t>(TimingStage::Count)> callCount{};
        std::array<uint64_t, static_cast<std::size_t>(TimingStage::Count)> maxNs{};
        uint64_t mainUploadsApplied = 0;
    };

    struct ChunkedMeshUploadState {
        StreamingMeshUpload upload;
        uint32_t targetBufferIndex = 0;
        size_t metadataUploadedBytes = 0;
        size_t quadUploadedBytes = 0;
        size_t aabbUploadedBytes = 0;
    };

    std::unique_ptr<WebGPUContext> context;
    std::unique_ptr<PipelineManager> pipelineManager;
    std::unique_ptr<BufferManager> bufferManager;
    std::unique_ptr<TextureManager> textureManager;
    std::unique_ptr<MaterialManager> materialManager;

    std::unique_ptr<MeshletManager> meshletManager;

    std::optional<RenderServices> services_;
    std::optional<VoxelPipeline> voxelPipeline_;
    std::optional<BoundsDebugPipeline> boundsDebugPipeline_;
    const World* debugWorld_ = nullptr;

    uint32_t meshletCapacity_ = 0;
    uint32_t quadCapacity_ = 0;
    uint64_t uploadedMeshRevision_ = 0;
    std::vector<MeshletAabb> activeMeshletBounds_;
    uint64_t uploadedDebugBoundsRevision_ = 0;
    uint64_t uploadedDebugBoundsMeshRevision_ = 0;
    uint32_t uploadedDebugBoundsLayerMask_ = 0u;
    ColumnCoord uploadedCenterColumn_{0, 0};
    bool hasUploadedCenterColumn_ = false;
    double lastMeshUploadTimeSeconds_ = -1.0;

    std::optional<StreamingMeshUpload> pendingMeshUpload_;
    std::optional<ChunkedMeshUploadState> chunkedMeshUpload_;
    std::atomic<bool> mainMeshUploadInProgress_{false};

    std::array<TimingAccumulator, static_cast<std::size_t>(TimingStage::Count)> timingAccumulators_{};
    std::atomic<uint64_t> mainUploadsApplied_{0};
    std::mutex timingSnapshotMutex_;
    TimingRawTotals lastTimingRawTotals_{};
    std::optional<std::chrono::steady_clock::time_point> lastTimingSampleTime_;

    bool resizePending = false;
    static constexpr size_t kMeshUploadBudgetBytesPerFrame = 2u * 1024u * 1024u;
    static constexpr uint32_t kOcclusionDepthDownsample = 2u;
    static constexpr const char* kOcclusionDepthTextureName = "meshlet_occlusion_depth_texture";
    static constexpr const char* kOcclusionDepthViewName = "meshlet_occlusion_depth_view";
    static constexpr const char* kOcclusionHiZTextureName = "meshlet_occlusion_hiz_texture";
    static constexpr const char* kOcclusionHiZViewName = "meshlet_occlusion_hiz_view";
    static constexpr const char* kMeshletCullParamsBufferName = "meshlet_cull_params_buffer";
    static constexpr const char* kMeshletCullIndirectArgsBufferName = "meshlet_cull_indirect_args_buffer";
    static constexpr const char* kMeshletCullIndirectResetBufferName = "meshlet_cull_indirect_reset_buffer";
    static constexpr uint32_t kMeshletCullWorkgroupSize = 128u;
    static constexpr uint32_t kOcclusionHiZWorkgroupSize = 8u;

    BindGroupLayout meshletDepthPrepassBindGroupLayout_ = nullptr;
    BindGroup meshletDepthPrepassBindGroup_ = nullptr;
    RenderPipeline meshletDepthPrepassPipeline_ = nullptr;
    BindGroupLayout meshletHiZSeedBindGroupLayout_ = nullptr;
    BindGroupLayout meshletHiZDownsampleBindGroupLayout_ = nullptr;
    ComputePipeline meshletHiZSeedPipeline_ = nullptr;
    ComputePipeline meshletHiZDownsamplePipeline_ = nullptr;
    uint32_t occlusionHiZMipCount_ = 1u;
    uint32_t occlusionDepthWidth_ = 1u;
    uint32_t occlusionDepthHeight_ = 1u;

    BindGroupLayout meshletCullBindGroupLayout_ = nullptr;
    BindGroup meshletCullBindGroup_ = nullptr;
    ComputePipeline meshletCullPipeline_ = nullptr;

    bool uploadMeshlets(StreamingMeshUpload&& upload);
    bool initializeMeshletOcclusionResources();
    bool createOcclusionDepthResources();
    void removeOcclusionDepthResources();
    bool refreshMeshletOcclusionBindGroup();
    void encodeMeshletOcclusionDepthPass(CommandEncoder encoder);
    void encodeMeshletOcclusionHierarchyPass(CommandEncoder encoder);
    bool initializeMeshletCullingResources();
    bool refreshMeshletCullingBindGroup();
    void updateMeshletCullParams(uint32_t meshletCount);
    void encodeMeshletCullingPass(CommandEncoder encoder);
    void processPendingMeshUploads();
    void updateDebugBounds(const FrameUniforms& frameUniforms);
    void rebuildDebugBounds(uint32_t layerMask);
    void recordTimingNs(TimingStage stage, uint64_t ns) noexcept;
    TimingRawTotals captureTimingRawTotals() const;
    static TimingStageSnapshot makeStageSnapshot(const TimingRawTotals& current,
                                                 const TimingRawTotals& previous,
                                                 TimingStage stage,
                                                 double sampleWindowSeconds);

public:
    WebGPURenderer() = default;
    ~WebGPURenderer() = default;

    bool initialize();

    PipelineManager* getPipelineManager();
    BufferManager* getBufferManager();
    TextureManager* getTextureManager();
    WebGPUContext* getContext();
    GLFWwindow* getWindow();

    void removeRenderingTextures();
    void createRenderingTextures();
    bool resizeSurfaceAndAttachments();
    void requestResize();

    std::pair<SurfaceTexture, TextureView> GetNextSurfaceViewData();
    RuntimeTimingSnapshot getRuntimeTimingSnapshot();

    void setDebugWorld(const World* world);
    void queueMeshUpload(StreamingMeshUpload&& upload);
    bool isMeshUploadInProgress() const noexcept;
    uint64_t uploadedMeshRevision() const noexcept;

    void renderFrame(FrameUniforms& uniforms);

    void terminate();
};
