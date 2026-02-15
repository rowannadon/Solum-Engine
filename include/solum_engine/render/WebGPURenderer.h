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
#include "solum_engine/voxel/World.h"
#include "solum_engine/voxel/ChunkMeshes.h"

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

struct MeshletPageDebugSnapshot {
    uint32_t activePageCount = 0;
    uint32_t maxPageCount = 0;
    uint32_t slotsPerPage = 0;
    uint32_t metadataCapacityPerPage = 0;
    uint64_t totalSlotCapacity = 0;
    uint64_t usedSlots = 0;
    uint64_t freeSlots = 0;
};

struct RendererDebugSnapshot {
    MeshletPageDebugSnapshot meshletPages;
    std::size_t renderedRegionCount = 0;
    std::size_t drawOrderRegionCount = 0;
    std::size_t pendingRegionBuildCount = 0;
    bool buildInFlight = false;
    uint32_t lastFrameRequestedMeshlets = 0;
    uint32_t lastFrameDrawnMeshlets = 0;
    uint32_t lastFrameMetadataCulledMeshlets = 0;
    uint32_t lastFrameBudgetCulledMeshlets = 0;
    uint32_t lastFramePagesDrawn = 0;
};

class WebGPURenderer {
private:
    struct GpuMeshletMetadata {
        int32_t originX = 0;
        int32_t originY = 0;
        int32_t originZ = 0;
        int32_t lodLevel = 0;
        uint32_t vertexBase = 0;
        uint32_t indexBase = 0;
        uint32_t quadCount = 0;
        uint32_t _padding = 0;
    };

    struct UploadedMeshletRef {
        uint32_t pageIndex = 0;
        uint32_t slotInPage = 0;
        glm::ivec3 origin{0};
        uint32_t lodLevel = 0;
        uint32_t quadCount = 0;
    };

    enum DebugRenderFlags : uint32_t {
        DebugRegionColors = 1u << 0u,
        DebugLodColors = 1u << 1u,
        DebugChunkColors = 1u << 2u,
        DebugMeshletColors = 1u << 3u,
    };

    struct MeshletPage {
        std::string metadataBufferName;
        std::string vertexBufferName;
        std::string indexBufferName;
        std::string bindGroupName;
        std::vector<uint32_t> freeSlots;
    };

    struct MeshSlotRef {
        std::vector<UploadedMeshletRef> meshlets;
        bool valid = false;
    };

    struct RegionRenderEntry {
        RegionCoord coord;
        std::array<MeshSlotRef, kRegionLodCount> lodMeshes;
    };

    struct PendingRegionBuild {
        RegionCoord coord;
        int lodLevel = 0;
    };

    struct CompletedRegionBuild {
        RegionCoord coord;
        int lodLevel = 0;
        MeshData meshData;
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
    std::array<int, kRegionLodCount> lodSteps_{};
    std::array<float, kRegionLodCount - 1> lodSwitchDistances_{};
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
    std::future<CompletedRegionBuild> activeBuildFuture_;
    bool buildInFlight_ = false;
    RegionCoord activeBuildCoord_{0, 0};
    int activeBuildLodLevel_ = -1;
    std::vector<MeshletPage> meshletPages_;
    uint32_t meshletSlotsPerPage_ = 0;
    uint32_t meshletMetadataCapacity_ = 0;
    bool meshletCapacityWarningEmitted_ = false;
    bool metadataCapacityWarningEmitted_ = false;
    uint32_t debugRenderFlags_ = 0;
    uint32_t lastFrameRequestedMeshlets_ = 0;
    uint32_t lastFrameDrawnMeshlets_ = 0;
    uint32_t lastFrameMetadataCulledMeshlets_ = 0;
    uint32_t lastFrameBudgetCulledMeshlets_ = 0;
    uint32_t lastFramePagesDrawn_ = 0;

    MeshData buildRegionLodMesh(RegionCoord regionCoord, int lodLevel) const;
    bool uploadMesh(MeshData&& meshData, MeshSlotRef& outMeshSlot);
    void releaseMesh(MeshSlotRef& meshSlot);
    void releaseAllRegionMeshes();
    void rebuildRegionsAroundPlayer(int centerRegionX, int centerRegionY, const glm::vec2& cameraXY);
    bool isRegionLodBuildQueued(const RegionCoord& coord, int lodLevel) const;
    void enqueueMissingRegionLodBuilds(int centerRegionX, int centerRegionY, const glm::vec2& cameraXY);
    void processPendingRegionBuilds();
    int chooseLodByDistance(float distance) const;
    bool loadHeightmap(const std::string& path);
    float sampleHeightmap01(int blockX, int blockY) const;
    int sampleHeightBlocks(int blockX, int blockY) const;
    void drawRegionSet(RenderPassEncoder& pass, const glm::vec3& cameraPos);

    bool createMeshletPage(uint32_t pageIndex);
    bool initializeMeshletBuffers(uint32_t meshletCapacity, uint32_t metadataCapacity);
    uint32_t acquireMeshletSlot(uint32_t& outPageIndex);
    void releaseMeshletSlot(uint32_t pageIndex, uint32_t slotInPage);
    bool refreshVoxelBindGroup();

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
    void toggleRegionDebugColors();
    void toggleLodDebugColors();
    void toggleChunkDebugColors();
    void toggleMeshletDebugColors();
    void clearDebugColors();
    uint32_t debugRenderFlags() const;
    RendererDebugSnapshot debugSnapshot() const;

    void terminate();
};
