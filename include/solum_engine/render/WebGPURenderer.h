#pragma once

#include <webgpu/webgpu.hpp>
#include <GLFW/glfw3.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

#include "solum_engine/render/PipelineManager.h"
#include "solum_engine/render/BufferManager.h"
#include "solum_engine/render/TextureManager.h"
#include "solum_engine/platform/WebGPUContext.h"
#include "solum_engine/render/Uniforms.h"
#include "solum_engine/render/pipelines/VoxelPipeline.h"
#include "solum_engine/render/MeshletManager.h"
#include "solum_engine/voxel/World.h"

class WebGPURenderer {
private:
    std::unique_ptr<WebGPUContext> context;
    std::unique_ptr<PipelineManager> pipelineManager;
    std::unique_ptr<BufferManager> bufferManager;
    std::unique_ptr<TextureManager> textureManager;

    std::unique_ptr<MeshletManager> meshletManager;
    std::unique_ptr<World> world_;

    std::optional<RenderServices> services_;
    std::optional<VoxelPipeline> voxelPipeline_;

    uint32_t meshletCapacity_ = 0;
    uint32_t quadCapacity_ = 0;
    uint64_t uploadedMeshRevision_ = 0;
    ColumnCoord uploadedCenterColumn_{0, 0};
    bool hasUploadedCenterColumn_ = false;
    int32_t uploadColumnRadius_ = 1;
    double lastMeshUploadTimeSeconds_ = -1.0;

    bool resizePending = false;

    bool uploadMeshlets(const std::vector<Meshlet>& meshlets);
    void updateWorldStreaming(const FrameUniforms& frameUniforms);
    glm::vec3 extractCameraPosition(const FrameUniforms& frameUniforms) const;
    ColumnCoord extractCameraColumn(const FrameUniforms& frameUniforms) const;
    int32_t cameraColumnChebyshevDistance(const ColumnCoord& a, const ColumnCoord& b) const;

public:
    WebGPURenderer() = default;

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
