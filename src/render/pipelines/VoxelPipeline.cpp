#include "solum_engine/render/pipelines/VoxelPipeline.h"

#include <cstdint>
#include <limits>

bool VoxelPipeline::createResources() {
    int width, height;
    glfwGetFramebufferSize(context->getWindow(), &width, &height);
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
    Texture depthTexture = tex->createTexture("depth_texture", depthTextureDesc);

    TextureViewDescriptor depthTextureViewDesc = Default;
    depthTextureViewDesc.aspect = TextureAspect::DepthOnly;
    depthTextureViewDesc.baseArrayLayer = 0;
    depthTextureViewDesc.arrayLayerCount = 1;
    depthTextureViewDesc.baseMipLevel = 0;
    depthTextureViewDesc.mipLevelCount = 1;
    depthTextureViewDesc.dimension = TextureViewDimension::_2D;
    depthTextureViewDesc.format = depthTextureFormat;
    TextureView depthTextureView = tex->createTextureView("depth_texture", "depth_view", depthTextureViewDesc);

        TextureFormat multiSampleTextureFormat = context->getSurfaceFormat();

        TextureDescriptor multiSampleTextureDesc = Default;
        multiSampleTextureDesc.dimension = TextureDimension::_2D;
        multiSampleTextureDesc.format = multiSampleTextureFormat;
        multiSampleTextureDesc.mipLevelCount = 1;
    multiSampleTextureDesc.sampleCount = 4;
        multiSampleTextureDesc.size = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 };
        multiSampleTextureDesc.usage = TextureUsage::RenderAttachment;
        multiSampleTextureDesc.viewFormatCount = 0;
        multiSampleTextureDesc.viewFormats = nullptr;
    Texture multiSampleTexture = tex->createTexture("multisample_texture", multiSampleTextureDesc);

        TextureViewDescriptor multiSampleTextureViewDesc = Default;
        multiSampleTextureViewDesc.aspect = TextureAspect::All;
        multiSampleTextureViewDesc.baseArrayLayer = 0;
        multiSampleTextureViewDesc.arrayLayerCount = 1;
        multiSampleTextureViewDesc.baseMipLevel = 0;
        multiSampleTextureViewDesc.mipLevelCount = 1;
        multiSampleTextureViewDesc.dimension = TextureViewDimension::_2D;
        multiSampleTextureViewDesc.format = multiSampleTextureFormat;
        TextureView multiSampleTextureView = tex->createTextureView("multisample_texture", "multisample_view", multiSampleTextureViewDesc);

    return multiSampleTextureView != nullptr && depthTextureView != nullptr;
};

void VoxelPipeline::removeResources() {
    tex->removeTextureView("multisample_view");
    tex->removeTexture("multisample_texture");

    tex->removeTextureView("depth_view");
    tex->removeTexture("depth_texture");

    pip->deleteBindGroup("global_uniforms_bg");
};

bool VoxelPipeline::createPipeline() {
    PipelineConfig config;
    config.shaderPath = SHADER_DIR "/voxel.wgsl";
    config.colorFormat = context->getSurfaceFormat();
    config.depthFormat = TextureFormat::Depth32Float;
    config.sampleCount = 4;
    config.cullMode = CullMode::Back;
    config.depthWriteEnabled = true;
    config.depthCompare = CompareFunction::Less;
    config.fragmentShaderName = "fs_main";  // Fragment shader entry point
    config.vertexShaderName = "vs_main";  // Vertex shader entry point
    config.useVertexBuffers = false;
    config.useCustomBlending = false;
    config.alphaToCoverageEnabled = false;

    int i = 0;
    std::vector<BindGroupLayoutEntry> globalUniforms(4, Default);
    globalUniforms[i].binding = i;
    globalUniforms[i].visibility = ShaderStage::Vertex | ShaderStage::Fragment;
    globalUniforms[i].buffer.type = BufferBindingType::Uniform;
    globalUniforms[i].buffer.minBindingSize = sizeof(FrameUniforms);
    i++;

    globalUniforms[i].binding = i;
    globalUniforms[i].visibility = ShaderStage::Vertex;
    globalUniforms[i].buffer.type = BufferBindingType::ReadOnlyStorage;
    globalUniforms[i].buffer.minBindingSize = 0;
    i++;

    globalUniforms[i].binding = i;
    globalUniforms[i].visibility = ShaderStage::Vertex;
    globalUniforms[i].buffer.type = BufferBindingType::ReadOnlyStorage;
    globalUniforms[i].buffer.minBindingSize = 0;
    i++;

    globalUniforms[i].binding = i;
    globalUniforms[i].visibility = ShaderStage::Vertex;
    globalUniforms[i].buffer.type = BufferBindingType::ReadOnlyStorage;
    globalUniforms[i].buffer.minBindingSize = 0;
    i++;

    config.bindGroupLayouts.push_back(
        pip->createBindGroupLayout("global_uniforms", globalUniforms)
    );

    RenderPipeline pipeline = pip->createRenderPipeline("voxel_pipeline", config);

    return pipeline != nullptr;
}

bool VoxelPipeline::createBindGroup() {
    std::vector<BindGroupEntry> bindings(4);

    int i = 0;
    bindings[i].binding = i;
    bindings[i].buffer = buf->getBuffer("uniform_buffer");
    bindings[i].offset = 0;
    bindings[i].size = sizeof(FrameUniforms);
    i++;

    bindings[i].binding = i;
    bindings[i].buffer = buf->getBuffer("meshlet_metadata_storage");
    bindings[i].offset = 0;
    bindings[i].size = std::numeric_limits<uint64_t>::max();
    i++;

    bindings[i].binding = i;
    bindings[i].buffer = buf->getBuffer("meshlet_vertex_storage");
    bindings[i].offset = 0;
    bindings[i].size = std::numeric_limits<uint64_t>::max();
    i++;

    bindings[i].binding = i;
    bindings[i].buffer = buf->getBuffer("meshlet_index_storage");
    bindings[i].offset = 0;
    bindings[i].size = std::numeric_limits<uint64_t>::max();
    i++;

    BindGroup bindGroup = pip->createBindGroup("global_uniforms_bg", "global_uniforms", bindings);

    return bindGroup != nullptr;
}

bool VoxelPipeline::render(
    TextureView targetView,
    CommandEncoder encoder,
    const std::function<void(RenderPassEncoder&)>& overlayCallback
) {
    RenderPassDescriptor renderPassDesc = Default;
    RenderPassColorAttachment renderPassColorAttachment = {};
    renderPassColorAttachment.view = tex->getTextureView("multisample_view");
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
    depthStencilAttachment.view = tex->getTextureView("depth_view");
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
    voxelRenderPass.setPipeline(pip->getPipeline("voxel_pipeline"));

    voxelRenderPass.setBindGroup(0, pip->getBindGroup("global_uniforms_bg"), 0, nullptr);

    if (overlayCallback) {
        overlayCallback(voxelRenderPass);
    }

    voxelRenderPass.end();
    voxelRenderPass.release();
    return true;
}
