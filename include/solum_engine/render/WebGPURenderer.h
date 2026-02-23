#pragma once

#include <webgpu/webgpu.hpp>
#include <GLFW/glfw3.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

#include "solum_engine/render/PipelineManager.h"
#include "solum_engine/render/BufferManager.h"
#include "solum_engine/render/TextureManager.h"
#include "solum_engine/platform/WebGPUContext.h"
#include "solum_engine/render/Uniforms.h"
#include "solum_engine/render/pipelines/BoundsDebugPipeline.h"
#include "solum_engine/render/pipelines/VoxelPipeline.h"
#include "solum_engine/render/MeshletManager.h"
#include "solum_engine/voxel/MeshManager.h"
#include "solum_engine/voxel/World.h"

class WebGPURenderer {
private:
    struct PendingMeshUpload {
        std::vector<Meshlet> meshlets;
        uint64_t meshRevision = 0;
        ColumnCoord centerColumn{0, 0};
    };

    std::unique_ptr<WebGPUContext> context;
    std::unique_ptr<PipelineManager> pipelineManager;
    std::unique_ptr<BufferManager> bufferManager;
    std::unique_ptr<TextureManager> textureManager;

    std::unique_ptr<MeshletManager> meshletManager;
    std::unique_ptr<World> world_;
    std::unique_ptr<MeshManager> voxelMeshManager_;

    std::optional<RenderServices> services_;
    std::optional<VoxelPipeline> voxelPipeline_;
    std::optional<BoundsDebugPipeline> boundsDebugPipeline_;

    uint32_t meshletCapacity_ = 0;
    uint32_t quadCapacity_ = 0;
    uint64_t uploadedMeshRevision_ = 0;
    uint64_t uploadedDebugBoundsRevision_ = 0;
    uint32_t uploadedDebugBoundsLayerMask_ = 0u;
    ColumnCoord uploadedCenterColumn_{0, 0};
    bool hasUploadedCenterColumn_ = false;
    int32_t uploadColumnRadius_ = 1;
    double lastMeshUploadTimeSeconds_ = -1.0;

    std::thread streamingThread_;
    std::mutex streamingMutex_;
    std::condition_variable streamingCv_;
    bool streamingStopRequested_ = false;
    bool hasLatestStreamingCamera_ = false;
    glm::vec3 latestStreamingCamera_{0.0f, 0.0f, 0.0f};
    std::optional<PendingMeshUpload> pendingMeshUpload_;
    uint64_t streamerLastPreparedRevision_ = 0;
    ColumnCoord streamerLastPreparedCenter_{0, 0};
    bool streamerHasLastPreparedCenter_ = false;
    std::optional<std::chrono::steady_clock::time_point> streamerLastSnapshotTime_;

    bool resizePending = false;

    bool uploadMeshlets(const std::vector<Meshlet>& meshlets);
    void updateWorldStreaming(const FrameUniforms& frameUniforms);
    void updateDebugBounds(const FrameUniforms& frameUniforms);
    void rebuildDebugBounds(uint32_t layerMask);
    void startStreamingThread(const glm::vec3& initialCameraPosition, const ColumnCoord& initialCenterColumn);
    void stopStreamingThread();
    void streamingThreadMain();
    glm::vec3 extractCameraPosition(const FrameUniforms& frameUniforms) const;
    ColumnCoord extractCameraColumn(const FrameUniforms& frameUniforms) const;
    int32_t cameraColumnChebyshevDistance(const ColumnCoord& a, const ColumnCoord& b) const;

public:
    WebGPURenderer() = default;
    ~WebGPURenderer();

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

    void renderFrame(FrameUniforms& uniforms);

    void terminate();
};
