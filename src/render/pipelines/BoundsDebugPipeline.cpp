#include "solum_engine/render/pipelines/BoundsDebugPipeline.h"

#include <algorithm>
#include <cstddef>

#include "solum_engine/render/Uniforms.h"

using namespace wgpu;

namespace {
constexpr const char* kPipelineName = "debug_bounds_pipeline";
constexpr const char* kBindGroupLayoutName = "debug_bounds_uniforms";
constexpr const char* kBindGroupName = "debug_bounds_uniforms_bg";
constexpr const char* kVertexBufferName = "debug_bounds_vertex_buffer";
}  // namespace

bool BoundsDebugPipeline::build() {
    return createResources() && createPipeline() && createBindGroup();
}

bool BoundsDebugPipeline::createResources() {
    return true;
}

void BoundsDebugPipeline::removeResources() {
    r_.pip.deleteBindGroup(kBindGroupName);
    r_.buf.deleteBuffer(kVertexBufferName);
    vertexCount_ = 0;
    vertexCapacityBytes_ = 0;
}

bool BoundsDebugPipeline::createPipeline() {
    PipelineConfig config;
    config.shaderPath = SHADER_DIR "/debug_bounds.wgsl";
    config.colorFormat = r_.ctx.getSurfaceFormat();
    config.depthFormat = TextureFormat::Depth32Float;
    config.sampleCount = 4;
    config.topology = PrimitiveTopology::LineList;
    config.cullMode = CullMode::None;
    config.depthWriteEnabled = false;
    config.depthCompare = CompareFunction::Always;
    config.fragmentShaderName = "fs_main";
    config.vertexShaderName = "vs_main";
    config.useVertexBuffers = true;
    config.vertexBufferStride = sizeof(DebugLineVertex);
    config.useCustomBlending = true;
    config.blendState.color.srcFactor = BlendFactor::SrcAlpha;
    config.blendState.color.dstFactor = BlendFactor::OneMinusSrcAlpha;
    config.blendState.color.operation = BlendOperation::Add;
    config.blendState.alpha.srcFactor = BlendFactor::One;
    config.blendState.alpha.dstFactor = BlendFactor::OneMinusSrcAlpha;
    config.blendState.alpha.operation = BlendOperation::Add;

    std::vector<VertexAttribute> attributes(2, Default);
    attributes[0].shaderLocation = 0;
    attributes[0].format = VertexFormat::Float32x3;
    attributes[0].offset = offsetof(DebugLineVertex, position);

    attributes[1].shaderLocation = 1;
    attributes[1].format = VertexFormat::Float32x4;
    attributes[1].offset = offsetof(DebugLineVertex, color);
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

bool BoundsDebugPipeline::createBindGroup() {
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

bool BoundsDebugPipeline::ensureVertexBufferCapacity(uint64_t requiredBytes) {
    Buffer existingBuffer = r_.buf.getBuffer(kVertexBufferName);
    if (existingBuffer && requiredBytes <= vertexCapacityBytes_) {
        return true;
    }

    r_.buf.deleteBuffer(kVertexBufferName);

    BufferDescriptor desc = Default;
    desc.label = StringView("debug bounds vertex buffer");
    desc.size = std::max<uint64_t>(requiredBytes, sizeof(DebugLineVertex) * 2ull);
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

bool BoundsDebugPipeline::updateVertices(const std::vector<DebugLineVertex>& vertices) {
    vertexCount_ = static_cast<uint32_t>(vertices.size());
    if (vertexCount_ == 0) {
        return true;
    }

    const uint64_t requiredBytes = static_cast<uint64_t>(vertexCount_) * sizeof(DebugLineVertex);
    if (!ensureVertexBufferCapacity(requiredBytes)) {
        return false;
    }

    r_.buf.writeBuffer(kVertexBufferName, 0, vertices.data(), static_cast<size_t>(requiredBytes));
    return true;
}

void BoundsDebugPipeline::draw(RenderPassEncoder& renderPass) {
    if (!enabled_ || vertexCount_ == 0) {
        return;
    }

    Buffer vertexBuffer = r_.buf.getBuffer(kVertexBufferName);
    if (!vertexBuffer) {
        return;
    }

    RenderPipeline pipeline = r_.pip.getPipeline(kPipelineName);
    BindGroup bindGroup = r_.pip.getBindGroup(kBindGroupName);
    if (!pipeline || !bindGroup) {
        return;
    }

    renderPass.setPipeline(pipeline);
    renderPass.setBindGroup(0, bindGroup, 0, nullptr);
    renderPass.setVertexBuffer(0, vertexBuffer, 0, static_cast<uint64_t>(vertexCount_) * sizeof(DebugLineVertex));
    renderPass.draw(vertexCount_, 1, 0, 0);
}

bool BoundsDebugPipeline::render(
    TextureView /* targetView */,
    CommandEncoder /* encoder */,
    const std::function<void(RenderPassEncoder&)>& /* overlayCallback */
) {
    return false;
}
