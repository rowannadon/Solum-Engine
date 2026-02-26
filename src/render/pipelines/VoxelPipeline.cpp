#include "solum_engine/render/pipelines/VoxelPipeline.h"

#include "solum_engine/render/MaterialManager.h"
#include "solum_engine/render/MeshletManager.h"
#include "solum_engine/render/Uniforms.h"

using namespace wgpu;

bool VoxelPipeline::build() {
    return createResources() && createPipeline() && createBindGroup();
}

void VoxelPipeline::setDrawConfig(uint32_t meshletVertices, uint32_t totalMeshletCount) {
    meshletVertexCount = meshletVertices;
    meshletCount = totalMeshletCount;
}

void VoxelPipeline::setIndirectDrawBuffer(const std::string& bufferName, uint64_t offset) {
    indirectDrawBufferName_ = bufferName;
    indirectDrawOffset_ = offset;
    useIndirectDraw_ = !indirectDrawBufferName_.empty();
}

void VoxelPipeline::clearIndirectDrawBuffer() {
    useIndirectDraw_ = false;
    indirectDrawBufferName_.clear();
    indirectDrawOffset_ = 0u;
}

bool VoxelPipeline::createResources() {
    int width, height;
    glfwGetFramebufferSize(r_.ctx.getWindow(), &width, &height);
    if (width <= 0 || height <= 0) {
        return false;
    }

    TextureFormat depthTextureFormat = TextureFormat::Depth32Float;
    TextureDescriptor depthTextureDesc = Default;
    depthTextureDesc.dimension = TextureDimension::_2D;
    depthTextureDesc.format = depthTextureFormat;
    depthTextureDesc.mipLevelCount = 1;
    depthTextureDesc.sampleCount = 4;
    depthTextureDesc.size = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 };
    depthTextureDesc.usage = TextureUsage::RenderAttachment | TextureUsage::TextureBinding;
    depthTextureDesc.viewFormatCount = 0;
    depthTextureDesc.viewFormats = nullptr;
    if (!r_.tex.createTexture("depth_texture", depthTextureDesc)) {
        return false;
    }

    TextureViewDescriptor depthTextureViewDesc = Default;
    depthTextureViewDesc.aspect = TextureAspect::DepthOnly;
    depthTextureViewDesc.baseArrayLayer = 0;
    depthTextureViewDesc.arrayLayerCount = 1;
    depthTextureViewDesc.baseMipLevel = 0;
    depthTextureViewDesc.mipLevelCount = 1;
    depthTextureViewDesc.dimension = TextureViewDimension::_2D;
    depthTextureViewDesc.format = depthTextureFormat;
    TextureView depthTextureView = r_.tex.createTextureView("depth_texture", "depth_view", depthTextureViewDesc);

    TextureFormat multiSampleTextureFormat = r_.ctx.getSurfaceFormat();

    TextureDescriptor multiSampleTextureDesc = Default;
    multiSampleTextureDesc.dimension = TextureDimension::_2D;
    multiSampleTextureDesc.format = multiSampleTextureFormat;
    multiSampleTextureDesc.mipLevelCount = 1;
    multiSampleTextureDesc.sampleCount = 4;
    multiSampleTextureDesc.size = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 };
    multiSampleTextureDesc.usage = TextureUsage::RenderAttachment;
    multiSampleTextureDesc.viewFormatCount = 0;
    multiSampleTextureDesc.viewFormats = nullptr;
    if (!r_.tex.createTexture("multisample_texture", multiSampleTextureDesc)) {
        return false;
    }

    TextureViewDescriptor multiSampleTextureViewDesc = Default;
    multiSampleTextureViewDesc.aspect = TextureAspect::All;
    multiSampleTextureViewDesc.baseArrayLayer = 0;
    multiSampleTextureViewDesc.arrayLayerCount = 1;
    multiSampleTextureViewDesc.baseMipLevel = 0;
    multiSampleTextureViewDesc.mipLevelCount = 1;
    multiSampleTextureViewDesc.dimension = TextureViewDimension::_2D;
    multiSampleTextureViewDesc.format = multiSampleTextureFormat;
    TextureView multiSampleTextureView = r_.tex.createTextureView("multisample_texture", "multisample_view", multiSampleTextureViewDesc);

    return multiSampleTextureView != nullptr && depthTextureView != nullptr;
}

void VoxelPipeline::removeResources() {
    r_.tex.removeTextureView("multisample_view");
    r_.tex.removeTexture("multisample_texture");

    r_.tex.removeTextureView("depth_view");
    r_.tex.removeTexture("depth_texture");

    r_.pip.deleteBindGroup("global_uniforms_bg");
}

bool VoxelPipeline::createPipeline() {
    PipelineConfig config;

    config.shaderPath = SHADER_DIR "/voxel.wgsl";
    config.colorFormat = r_.ctx.getSurfaceFormat();
    config.depthFormat = TextureFormat::Depth32Float;
    config.sampleCount = 4;
    config.cullMode = CullMode::Back;
    config.depthWriteEnabled = true;
    config.depthCompare = CompareFunction::Less;
    config.fragmentShaderName = "fs_main";
    config.vertexShaderName = "vs_main";
    config.useVertexBuffers = false;
    config.useCustomBlending = false;
    config.alphaToCoverageEnabled = false;

    std::vector<BindGroupLayoutEntry> globalUniforms(7, Default);

    int i = 0;
    globalUniforms[i].binding = i;
    globalUniforms[i].visibility = ShaderStage::Vertex | ShaderStage::Fragment;
    globalUniforms[i].buffer.type = BufferBindingType::Uniform;
    globalUniforms[i].buffer.minBindingSize = sizeof(FrameUniforms);
    i++;

    globalUniforms[i].binding = i;
    globalUniforms[i].visibility = ShaderStage::Vertex;
    globalUniforms[i].buffer.type = BufferBindingType::ReadOnlyStorage;
    i++;

    globalUniforms[i].binding = i;
    globalUniforms[i].visibility = ShaderStage::Vertex;
    globalUniforms[i].buffer.type = BufferBindingType::ReadOnlyStorage;
    i++;

    globalUniforms[i].binding = i;
    globalUniforms[i].visibility = ShaderStage::Vertex | ShaderStage::Fragment;
    globalUniforms[i].buffer.type = BufferBindingType::ReadOnlyStorage;
    i++;

    globalUniforms[i].binding = i;
    globalUniforms[i].visibility = ShaderStage::Vertex;
    globalUniforms[i].buffer.type = BufferBindingType::ReadOnlyStorage;
    i++;

    globalUniforms[i].binding = i;
    globalUniforms[i].visibility = ShaderStage::Fragment;
    globalUniforms[i].texture.sampleType = TextureSampleType::Float;
    globalUniforms[i].texture.viewDimension = TextureViewDimension::_2DArray;
    i++;

    globalUniforms[i].binding = i;
    globalUniforms[i].visibility = ShaderStage::Fragment;
    globalUniforms[i].sampler.type = SamplerBindingType::Filtering;

    config.bindGroupLayouts.push_back(
        r_.pip.createBindGroupLayout("global_uniforms", globalUniforms)
    );

    RenderPipeline pipeline = r_.pip.createRenderPipeline("voxel_pipeline", config);

    return pipeline != nullptr;
}

bool VoxelPipeline::createBindGroup() {
    return createBindGroupForMeshBuffers(
        MeshletManager::meshDataBufferName(0),
        MeshletManager::meshMetadataBufferName(0),
        MeshletManager::visibleMeshletIndexBufferName(0)
    );
}

bool VoxelPipeline::createBindGroupForMeshBuffers(const std::string& meshDataBufferName,
                                                  const std::string& metadataBufferName,
                                                  const std::string& visibleIndicesBufferName) {
    Buffer uniformBuffer = r_.buf.getBuffer("uniform_buffer");
    Buffer meshDataBuffer = r_.buf.getBuffer(meshDataBufferName);
    Buffer metadataBuffer = r_.buf.getBuffer(metadataBufferName);
    Buffer visibleIndicesBuffer = r_.buf.getBuffer(visibleIndicesBufferName);
    Buffer materialLookupBuffer = r_.buf.getBuffer(MaterialManager::kMaterialLookupBufferName);
    TextureView materialTextureArrayView = r_.tex.getTextureView(MaterialManager::kMaterialTextureArrayViewName);
    Sampler materialSampler = r_.tex.getSampler(MaterialManager::kMaterialSamplerName);

    if (!uniformBuffer || !meshDataBuffer || !metadataBuffer || !visibleIndicesBuffer || !materialLookupBuffer || !materialTextureArrayView || !materialSampler) {
        return false;
    }

    std::vector<BindGroupEntry> bindings(7, Default);

    int i = 0;
    bindings[i].binding = i;
    bindings[i].buffer = uniformBuffer;
    bindings[i].offset = 0;
    bindings[i].size = sizeof(FrameUniforms);
    i++;

    bindings[i].binding = i;
    bindings[i].buffer = meshDataBuffer;
    bindings[i].offset = 0;
    bindings[i].size = meshDataBuffer.getSize();
    i++;

    bindings[i].binding = i;
    bindings[i].buffer = metadataBuffer;
    bindings[i].offset = 0;
    bindings[i].size = metadataBuffer.getSize();
    i++;

    bindings[i].binding = i;
    bindings[i].buffer = materialLookupBuffer;
    bindings[i].offset = 0;
    bindings[i].size = materialLookupBuffer.getSize();
    i++;

    bindings[i].binding = i;
    bindings[i].buffer = visibleIndicesBuffer;
    bindings[i].offset = 0;
    bindings[i].size = visibleIndicesBuffer.getSize();
    i++;

    bindings[i].binding = i;
    bindings[i].textureView = materialTextureArrayView;
    i++;

    bindings[i].binding = i;
    bindings[i].sampler = materialSampler;

    r_.pip.deleteBindGroup("global_uniforms_bg");
    BindGroup bindGroup = r_.pip.createBindGroup("global_uniforms_bg", "global_uniforms", bindings);

    return bindGroup != nullptr;
}

bool VoxelPipeline::render(
    TextureView targetView,
    CommandEncoder encoder,
    const std::function<void(RenderPassEncoder&)>& overlayCallback
) {
    RenderPassDescriptor renderPassDesc = Default;
    RenderPassColorAttachment renderPassColorAttachment = {};
    renderPassColorAttachment.view = r_.tex.getTextureView("multisample_view");
    renderPassColorAttachment.resolveTarget = targetView;
    renderPassColorAttachment.loadOp = LoadOp::Clear;
    renderPassColorAttachment.storeOp = StoreOp::Store;
    renderPassColorAttachment.clearValue = Color{ 0.2, 0.2, 0.3, 1.0 };
#ifndef WEBGPU_BACKEND_WGPU
    renderPassColorAttachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
#endif

    renderPassDesc.colorAttachmentCount = 1;
    renderPassDesc.colorAttachments = &renderPassColorAttachment;

    RenderPassDepthStencilAttachment depthStencilAttachment = Default;
    depthStencilAttachment.view = r_.tex.getTextureView("depth_view");
    depthStencilAttachment.depthClearValue = 1.0f;
    depthStencilAttachment.depthLoadOp = LoadOp::Clear;
    depthStencilAttachment.depthStoreOp = StoreOp::Store;
    depthStencilAttachment.depthReadOnly = false;
    depthStencilAttachment.stencilClearValue = 0;
    depthStencilAttachment.stencilLoadOp = LoadOp::Undefined;
    depthStencilAttachment.stencilStoreOp = StoreOp::Undefined;
    depthStencilAttachment.stencilReadOnly = true;

    renderPassDesc.depthStencilAttachment = &depthStencilAttachment;
    renderPassDesc.timestampWrites = nullptr;

    RenderPassEncoder voxelRenderPass = encoder.beginRenderPass(renderPassDesc);
    voxelRenderPass.setPipeline(r_.pip.getPipeline("voxel_pipeline"));

    voxelRenderPass.setBindGroup(0, r_.pip.getBindGroup("global_uniforms_bg"), 0, nullptr);

    if (useIndirectDraw_) {
        Buffer indirectBuffer = r_.buf.getBuffer(indirectDrawBufferName_);
        if (indirectBuffer) {
            voxelRenderPass.drawIndirect(indirectBuffer, indirectDrawOffset_);
        } else if (meshletVertexCount > 0 && meshletCount > 0) {
            voxelRenderPass.draw(meshletVertexCount, meshletCount, 0, 0);
        }
    } else if (meshletVertexCount > 0 && meshletCount > 0) {
        voxelRenderPass.draw(meshletVertexCount, meshletCount, 0, 0);
    }

    if (overlayCallback) {
        overlayCallback(voxelRenderPass);
    }

    voxelRenderPass.end();
    voxelRenderPass.release();
    return true;
}
