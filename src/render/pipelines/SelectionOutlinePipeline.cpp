#include "solum_engine/render/pipelines/SelectionOutlinePipeline.h"

#include <algorithm>
#include <array>
#include <cstddef>

#include "solum_engine/render/Uniforms.h"

using namespace wgpu;

namespace {
constexpr const char* kPipelineName = "selection_outline_pipeline";
constexpr const char* kBindGroupLayoutName = "selection_outline_uniforms";
constexpr const char* kBindGroupName = "selection_outline_uniforms_bg";
constexpr const char* kVertexBufferName = "selection_outline_vertex_buffer";

constexpr glm::vec4 kOutlineColor{0.05f, 0.05f, 0.05f, 0.95f};
constexpr float kOutlineExpansion = 0.002f;
constexpr float kOutlineHalfWidthPixels = 1.25f;

void appendEdgeQuad(std::vector<SelectionOutlineVertex>& vertices,
                    const glm::vec3& start,
                    const glm::vec3& end,
                    const glm::vec4& color) {
    const std::array<glm::vec2, 6> quadCorners{{
        glm::vec2{0.0f, -kOutlineHalfWidthPixels},
        glm::vec2{1.0f, -kOutlineHalfWidthPixels},
        glm::vec2{1.0f,  kOutlineHalfWidthPixels},
        glm::vec2{0.0f, -kOutlineHalfWidthPixels},
        glm::vec2{1.0f,  kOutlineHalfWidthPixels},
        glm::vec2{0.0f,  kOutlineHalfWidthPixels},
    }};

    for (const glm::vec2& corner : quadCorners) {
        vertices.push_back(SelectionOutlineVertex{
            glm::vec4(start, corner.x),
            glm::vec4(end, corner.y),
            color
        });
    }
}

void appendWireBoxQuads(std::vector<SelectionOutlineVertex>& vertices,
                        const glm::vec3& minCorner,
                        const glm::vec3& maxCorner,
                        const glm::vec4& color) {
    const std::array<glm::vec3, 8> corners{
        glm::vec3{minCorner.x, minCorner.y, minCorner.z},
        glm::vec3{maxCorner.x, minCorner.y, minCorner.z},
        glm::vec3{maxCorner.x, maxCorner.y, minCorner.z},
        glm::vec3{minCorner.x, maxCorner.y, minCorner.z},
        glm::vec3{minCorner.x, minCorner.y, maxCorner.z},
        glm::vec3{maxCorner.x, minCorner.y, maxCorner.z},
        glm::vec3{maxCorner.x, maxCorner.y, maxCorner.z},
        glm::vec3{minCorner.x, maxCorner.y, maxCorner.z},
    };

    constexpr std::array<std::array<uint8_t, 2>, 12> edgeIndices{{
        {{0, 1}}, {{1, 2}}, {{2, 3}}, {{3, 0}},
        {{4, 5}}, {{5, 6}}, {{6, 7}}, {{7, 4}},
        {{0, 4}}, {{1, 5}}, {{2, 6}}, {{3, 7}},
    }};

    for (const auto& edge : edgeIndices) {
        appendEdgeQuad(vertices, corners[edge[0]], corners[edge[1]], color);
    }
}
}  // namespace

bool SelectionOutlinePipeline::build() {
    return createResources() && createPipeline() && createBindGroup();
}

bool SelectionOutlinePipeline::createResources() {
    return true;
}

void SelectionOutlinePipeline::removeResources() {
    r_.pip.deleteBindGroup(kBindGroupName);
    r_.buf.deleteBuffer(kVertexBufferName);
    selectedBlock_.reset();
    vertexCount_ = 0;
    vertexCapacityBytes_ = 0;
}

bool SelectionOutlinePipeline::createPipeline() {
    PipelineConfig config;
    config.shaderPath = SHADER_DIR "/selection_outline.wgsl";
    config.colorFormat = r_.ctx.getSurfaceFormat();
    config.depthFormat = TextureFormat::Depth32Float;
    config.sampleCount = 4;
    config.topology = PrimitiveTopology::TriangleList;
    config.cullMode = CullMode::None;
    config.depthWriteEnabled = false;
    config.depthCompare = CompareFunction::LessEqual;
    config.fragmentShaderName = "fs_main";
    config.vertexShaderName = "vs_main";
    config.useVertexBuffers = true;
    config.vertexBufferStride = sizeof(SelectionOutlineVertex);
    config.useCustomBlending = true;
    config.blendState.color.srcFactor = BlendFactor::SrcAlpha;
    config.blendState.color.dstFactor = BlendFactor::OneMinusSrcAlpha;
    config.blendState.color.operation = BlendOperation::Add;
    config.blendState.alpha.srcFactor = BlendFactor::One;
    config.blendState.alpha.dstFactor = BlendFactor::OneMinusSrcAlpha;
    config.blendState.alpha.operation = BlendOperation::Add;

    std::vector<VertexAttribute> attributes(3, Default);
    attributes[0].shaderLocation = 0;
    attributes[0].format = VertexFormat::Float32x4;
    attributes[0].offset = offsetof(SelectionOutlineVertex, lineStartAndAlong);

    attributes[1].shaderLocation = 1;
    attributes[1].format = VertexFormat::Float32x4;
    attributes[1].offset = offsetof(SelectionOutlineVertex, lineEndAndSide);

    attributes[2].shaderLocation = 2;
    attributes[2].format = VertexFormat::Float32x4;
    attributes[2].offset = offsetof(SelectionOutlineVertex, color);
    config.vertexAttributes = attributes;

    std::vector<BindGroupLayoutEntry> uniformsLayout(1, Default);
    uniformsLayout[0].binding = 0;
    uniformsLayout[0].visibility = ShaderStage::Vertex;
    uniformsLayout[0].buffer.type = BufferBindingType::Uniform;
    uniformsLayout[0].buffer.minBindingSize = sizeof(FrameUniforms);

    config.bindGroupLayouts.push_back(
        r_.pip.createBindGroupLayout(kBindGroupLayoutName, uniformsLayout)
    );

    RenderPipeline pipeline = r_.pip.createRenderPipeline(kPipelineName, config);
    return pipeline != nullptr;
}

bool SelectionOutlinePipeline::createBindGroup() {
    Buffer uniformBuffer = r_.buf.getBuffer("uniform_buffer");
    if (!uniformBuffer) {
        return false;
    }

    std::vector<BindGroupEntry> bindings(1, Default);
    bindings[0].binding = 0;
    bindings[0].buffer = uniformBuffer;
    bindings[0].offset = 0;
    bindings[0].size = sizeof(FrameUniforms);

    BindGroup bindGroup = r_.pip.createBindGroup(kBindGroupName, kBindGroupLayoutName, bindings);
    return bindGroup != nullptr;
}

bool SelectionOutlinePipeline::ensureVertexBufferCapacity(uint64_t requiredBytes) {
    Buffer existingBuffer = r_.buf.getBuffer(kVertexBufferName);
    if (existingBuffer && requiredBytes <= vertexCapacityBytes_) {
        return true;
    }

    r_.buf.deleteBuffer(kVertexBufferName);

    BufferDescriptor desc = Default;
    desc.label = StringView("selection outline vertex buffer");
    desc.size = std::max<uint64_t>(requiredBytes, sizeof(SelectionOutlineVertex) * 2ull);
    desc.usage = BufferUsage::Vertex | BufferUsage::CopyDst;
    desc.mappedAtCreation = false;

    Buffer buffer = r_.buf.createBuffer(kVertexBufferName, desc);
    if (!buffer) {
        vertexCapacityBytes_ = 0;
        return false;
    }

    vertexCapacityBytes_ = desc.size;
    return true;
}

bool SelectionOutlinePipeline::updateVertices(const std::vector<SelectionOutlineVertex>& vertices) {
    const uint32_t nextVertexCount = static_cast<uint32_t>(vertices.size());
    if (nextVertexCount == 0) {
        vertexCount_ = 0;
        return true;
    }

    const uint64_t requiredBytes = static_cast<uint64_t>(nextVertexCount) * sizeof(SelectionOutlineVertex);
    if (!ensureVertexBufferCapacity(requiredBytes)) {
        return false;
    }

    r_.buf.writeBuffer(kVertexBufferName, 0, vertices.data(), static_cast<size_t>(requiredBytes));
    vertexCount_ = nextVertexCount;
    return true;
}

bool SelectionOutlinePipeline::setSelectedBlock(const std::optional<BlockCoord>& selectedBlock) {
    if (selectedBlock_ == selectedBlock) {
        return true;
    }

    if (!selectedBlock.has_value()) {
        selectedBlock_.reset();
        return updateVertices({});
    }

    const glm::vec3 minCorner{
        static_cast<float>(selectedBlock->v.x) - kOutlineExpansion,
        static_cast<float>(selectedBlock->v.y) - kOutlineExpansion,
        static_cast<float>(selectedBlock->v.z) - kOutlineExpansion
    };
    const glm::vec3 maxCorner{
        static_cast<float>(selectedBlock->v.x + 1) + kOutlineExpansion,
        static_cast<float>(selectedBlock->v.y + 1) + kOutlineExpansion,
        static_cast<float>(selectedBlock->v.z + 1) + kOutlineExpansion
    };

    std::vector<SelectionOutlineVertex> vertices;
    vertices.reserve(12 * 6);
    appendWireBoxQuads(vertices, minCorner, maxCorner, kOutlineColor);
    if (!updateVertices(vertices)) {
        return false;
    }

    selectedBlock_ = selectedBlock;
    return true;
}

bool SelectionOutlinePipeline::render(
    TextureView targetView,
    CommandEncoder encoder,
    const std::function<void(RenderPassEncoder&)>& overlayCallback
) {
    TextureView multisampleView = r_.tex.getTextureView("multisample_view");
    TextureView depthView = r_.tex.getTextureView("depth_view");
    if (!multisampleView || !depthView) {
        return false;
    }

    const bool shouldDrawSelection = vertexCount_ > 0;
    if (!shouldDrawSelection && !overlayCallback) {
        return true;
    }

    RenderPassDescriptor renderPassDesc = Default;
    RenderPassColorAttachment renderPassColorAttachment = {};
    renderPassColorAttachment.view = multisampleView;
    renderPassColorAttachment.resolveTarget = targetView;
    renderPassColorAttachment.loadOp = LoadOp::Load;
    renderPassColorAttachment.storeOp = StoreOp::Store;
    renderPassColorAttachment.clearValue = Color{0.0, 0.0, 0.0, 0.0};
#ifndef WEBGPU_BACKEND_WGPU
    renderPassColorAttachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
#endif

    renderPassDesc.colorAttachmentCount = 1;
    renderPassDesc.colorAttachments = &renderPassColorAttachment;

    RenderPassDepthStencilAttachment depthStencilAttachment = Default;
    depthStencilAttachment.view = depthView;
    depthStencilAttachment.depthClearValue = 1.0f;
    depthStencilAttachment.depthLoadOp = LoadOp::Load;
    depthStencilAttachment.depthStoreOp = StoreOp::Store;
    depthStencilAttachment.depthReadOnly = false;
    depthStencilAttachment.stencilClearValue = 0;
    depthStencilAttachment.stencilLoadOp = LoadOp::Undefined;
    depthStencilAttachment.stencilStoreOp = StoreOp::Undefined;
    depthStencilAttachment.stencilReadOnly = true;
    renderPassDesc.depthStencilAttachment = &depthStencilAttachment;
    renderPassDesc.timestampWrites = nullptr;

    RenderPassEncoder pass = encoder.beginRenderPass(renderPassDesc);

    if (shouldDrawSelection) {
        Buffer vertexBuffer = r_.buf.getBuffer(kVertexBufferName);
        RenderPipeline pipeline = r_.pip.getPipeline(kPipelineName);
        BindGroup bindGroup = r_.pip.getBindGroup(kBindGroupName);
        if (vertexBuffer && pipeline && bindGroup) {
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup, 0, nullptr);
            pass.setVertexBuffer(
                0,
                vertexBuffer,
                0,
                static_cast<uint64_t>(vertexCount_) * sizeof(SelectionOutlineVertex)
            );
            pass.draw(vertexCount_, 1, 0, 0);
        }
    }

    if (overlayCallback) {
        overlayCallback(pass);
    }

    pass.end();
    pass.release();
    return true;
}
