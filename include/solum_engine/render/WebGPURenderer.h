#pragma once

#include <webgpu/webgpu.hpp>
#include <GLFW/glfw3.h>
#include <unordered_map>
#include <future>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include "solum_engine/render/PipelineManager.h"
#include "solum_engine/render/BufferManager.h"
#include "solum_engine/render/TextureManager.h"
#include "solum_engine/platform/WebGPUContext.h"
#include "solum_engine/resources/Constants.h"
#include "solum_engine/resources/Coords.h"
#include "solum_engine/render/Uniforms.h"
#include "solum_engine/render/pipelines/VoxelPipeline.h"
#include "solum_engine/render/VertexAttributes.h"

#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_wgpu.h>
#include <array>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

class WebGPURenderer {
private:
    struct MeshSlotRef {
        std::string vertexBufferName;
        uint64_t vertexOffset = 0;
        uint64_t vertexBytes = 0;
        std::string indexBufferName;
        uint64_t indexOffset = 0;
        uint64_t indexBytes = 0;
        uint32_t indexCount = 0;
        bool valid = false;
    };

    struct RegionRenderEntry {
        RegionCoord coord;
        std::array<MeshSlotRef, 3> lodMeshes;
    };

    struct BufferSlot {
        std::string bufferName;
        uint64_t offset = 0;
        uint64_t capacity = 0;
        uint64_t usedBytes = 0;
        int classIndex = 0;
        bool inUse = false;
    };

    struct PendingRegionBuild {
        RegionCoord coord;
        int lodLevel = 0;
    };

    std::unique_ptr<WebGPUContext> context;
    std::unique_ptr<PipelineManager> pipelineManager;
    std::unique_ptr<BufferManager> bufferManager;
    std::unique_ptr<TextureManager> textureManager;

    VoxelPipeline voxelPipeline;
    bool resizePending = false;

    // Add this for ImGUI support
    RenderPassEncoder currentCommandEncoder = nullptr;

    int activeCenterRegionX = std::numeric_limits<int>::min();
    int activeCenterRegionY = std::numeric_limits<int>::min();
    int regionRadius_ = 1;
    std::array<int, 3> lodSteps_{};
    float lodDistance0_ = 0.0f;
    float lodDistance1_ = 0.0f;
    double buildBudgetMs_ = 0.0;
    std::vector<unsigned char> heightmapRgba_;
    uint32_t heightmapWidth_ = 0;
    uint32_t heightmapHeight_ = 0;
    int heightmapUpscaleFactor_ = 8;
    bool heightmapWrap_ = true;
    int terrainHeightMinBlocks_ = 0;
    int terrainHeightMaxBlocks_ = CHUNK_SIZE * COLUMN_CHUNKS_Z - 1;
    std::unordered_map<RegionCoord, RegionRenderEntry, RegionCoordHash> renderedRegions_;
    std::vector<RegionCoord> drawOrder_;
    std::deque<PendingRegionBuild> pendingBuilds_;
    std::vector<BufferSlot> vertexSlots_;
    std::vector<BufferSlot> indexSlots_;
    uint64_t nextBufferId_ = 0;

    std::pair<std::vector<VertexAttributes>, std::vector<uint32_t>> buildRegionLodMesh(RegionCoord regionCoord, int lodLevel) const;
    bool uploadMesh(std::vector<VertexAttributes>&& vertices, std::vector<uint32_t>&& indices, MeshSlotRef& outMeshSlot);
    void releaseMesh(MeshSlotRef& meshSlot);
    void releaseAllRegionMeshes();
    void rebuildRegionsAroundPlayer(int centerRegionX, int centerRegionY);
    void processPendingRegionBuilds();
    int chooseLodByDistance(float distance) const;
    bool loadHeightmap(const std::string& path);
    float sampleHeightmap01(int blockX, int blockY) const;
    int sampleHeightBlocks(int blockX, int blockY) const;
    void drawRegionSet(RenderPassEncoder& pass, const glm::vec3& cameraPos);

    BufferSlot* acquireSlot(std::vector<BufferSlot>& slots, bool isVertex, uint64_t requiredBytes);
    void releaseSlot(std::vector<BufferSlot>& slots, const std::string& bufferName, uint64_t offset);
    int chooseSizeClass(uint64_t requiredBytes) const;

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
