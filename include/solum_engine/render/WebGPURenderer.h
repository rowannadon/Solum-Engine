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
#include "solum_engine/render/Uniforms.h"
#include "solum_engine/render/pipelines/VoxelPipeline.h"
#include "solum_engine/render/VertexAttributes.h"
#include "solum_engine/voxel/Chunk.h"
#include "solum_engine/voxel/ChunkMesher.h"

#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_wgpu.h>

class WebGPURenderer {
private:
    std::unique_ptr<WebGPUContext> context;
    std::unique_ptr<PipelineManager> pipelineManager;
    std::unique_ptr<BufferManager> bufferManager;
    std::unique_ptr<TextureManager> textureManager;

    FrameUniforms uniforms;

    Chunk chunk;
    Chunk chunk2;
    ChunkMesher mesher;

    VoxelPipeline voxelPipeline;

    // Add this for ImGUI support
    RenderPassEncoder currentCommandEncoder = nullptr;

    uint16_t vertexCount;
    uint16_t indexCount;

public:
    bool initialize();

    PipelineManager* getPipelineManager();
    BufferManager* getBufferManager();
    TextureManager* getTextureManager();
    WebGPUContext* getContext();
    GLFWwindow* getWindow();

    void removeRenderingTextures();
    void createRenderingTextures();

    std::pair<SurfaceTexture, TextureView> GetNextSurfaceViewData();

    void renderFrame(FrameUniforms& uniforms);

    void terminate();
};

