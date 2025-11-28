#include "solum_engine/render/pipelines/VoxelPipeline.h"

bool VoxelPipeline::createResources() {
    int width, height;
    glfwGetFramebufferSize(context->getWindow(), &width, &height);

    TextureFormat depthTextureFormat = TextureFormat::Depth24Plus;
    TextureDescriptor depthTextureDesc;
    depthTextureDesc.dimension = TextureDimension::_2D;
    depthTextureDesc.format = depthTextureFormat;
    depthTextureDesc.mipLevelCount = 1;
    depthTextureDesc.sampleCount = 4;
    depthTextureDesc.size = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 };
    depthTextureDesc.usage = TextureUsage::RenderAttachment | TextureUsage::TextureBinding;
    depthTextureDesc.viewFormatCount = 0;
    depthTextureDesc.viewFormats = nullptr;
    Texture depthTexture = tex->createTexture("depth_texture", depthTextureDesc);

    TextureViewDescriptor depthTextureViewDesc;
    depthTextureViewDesc.aspect = TextureAspect::DepthOnly;
    depthTextureViewDesc.baseArrayLayer = 0;
    depthTextureViewDesc.arrayLayerCount = 1;
    depthTextureViewDesc.baseMipLevel = 0;
    depthTextureViewDesc.mipLevelCount = 1;
    depthTextureViewDesc.dimension = TextureViewDimension::_2D;
    depthTextureViewDesc.format = depthTextureFormat;
    TextureView depthTextureView = tex->createTextureView("depth_texture", "depth_view", depthTextureViewDesc);

    TextureFormat multiSampleTextureFormat = context->getSurfaceFormat();

    TextureDescriptor multiSampleTextureDesc;
    multiSampleTextureDesc.dimension = TextureDimension::_2D;
    multiSampleTextureDesc.format = multiSampleTextureFormat;
    multiSampleTextureDesc.mipLevelCount = 1;
    multiSampleTextureDesc.sampleCount = 4;
    multiSampleTextureDesc.size = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 };
    multiSampleTextureDesc.usage = TextureUsage::RenderAttachment;
    multiSampleTextureDesc.viewFormatCount = 0;
    multiSampleTextureDesc.viewFormats = nullptr;
    Texture multiSampleTexture = tex->createTexture("multisample_texture", multiSampleTextureDesc);

    TextureViewDescriptor multiSampleTextureViewDesc;
    multiSampleTextureViewDesc.aspect = TextureAspect::All;
    multiSampleTextureViewDesc.baseArrayLayer = 0;
    multiSampleTextureViewDesc.arrayLayerCount = 1;
    multiSampleTextureViewDesc.baseMipLevel = 0;
    multiSampleTextureViewDesc.mipLevelCount = 1;
    multiSampleTextureViewDesc.dimension = TextureViewDimension::_2D;
    multiSampleTextureViewDesc.format = multiSampleTextureFormat;
    TextureView multiSampleTextureView = tex->createTextureView("multisample_texture", "multisample_view", multiSampleTextureViewDesc);

    return multiSampleTextureView != nullptr;
};

void VoxelPipeline::removeResources() {
    tex->removeTexture("multisample_texture");
    tex->removeTextureView("multisample_view");

    tex->removeTexture("depth_texture");
    tex->removeTextureView("depth_view");

    pip->deleteBindGroup("global_uniforms_bg");
};

bool VoxelPipeline::createPipeline() {
    PipelineConfig config;

    // Allocate space for the 9 vertex attributes we actually use
    config.vertexAttributes.resize(9);
    config.vertexAttributes[0].shaderLocation = 0;
    config.vertexAttributes[0].format = VertexFormat::Uint16;
    config.vertexAttributes[0].offset = offsetof(VertexAttributes, x);

    config.vertexAttributes[1].shaderLocation = 1;
    config.vertexAttributes[1].format = VertexFormat::Uint16;
    config.vertexAttributes[1].offset = offsetof(VertexAttributes, y);

    config.vertexAttributes[2].shaderLocation = 2;
    config.vertexAttributes[2].format = VertexFormat::Uint16;
    config.vertexAttributes[2].offset = offsetof(VertexAttributes, z);

    config.vertexAttributes[3].shaderLocation = 3;
    config.vertexAttributes[3].format = VertexFormat::Uint16;
    config.vertexAttributes[3].offset = offsetof(VertexAttributes, u);

    config.vertexAttributes[4].shaderLocation = 4;
    config.vertexAttributes[4].format = VertexFormat::Uint16;
    config.vertexAttributes[4].offset = offsetof(VertexAttributes, v);

    config.vertexAttributes[5].shaderLocation = 5;
    config.vertexAttributes[5].format = VertexFormat::Uint16;
    config.vertexAttributes[5].offset = offsetof(VertexAttributes, material);

    config.vertexAttributes[6].shaderLocation = 6;
    config.vertexAttributes[6].format = VertexFormat::Uint8;
    config.vertexAttributes[6].offset = offsetof(VertexAttributes, n_x);

    config.vertexAttributes[7].shaderLocation = 7;
    config.vertexAttributes[7].format = VertexFormat::Uint8;
    config.vertexAttributes[7].offset = offsetof(VertexAttributes, n_y);

    config.vertexAttributes[8].shaderLocation = 8;
    config.vertexAttributes[8].format = VertexFormat::Uint8;
    config.vertexAttributes[8].offset = offsetof(VertexAttributes, n_z);


    config.shaderPath = SHADER_DIR "/voxel.wgsl";
    config.colorFormat = TextureFormat::BGRA8Unorm;
    config.depthFormat = TextureFormat::Depth24Plus;
    config.sampleCount = 4;
    config.cullMode = CullMode::None;
    config.depthWriteEnabled = true;
    config.depthCompare = CompareFunction::Less;
    config.fragmentShaderName = "fs_main";  // Fragment shader entry point
    config.vertexShaderName = "vs_main";  // Vertex shader entry point
    config.useVertexBuffers = true;
    config.useCustomBlending = false;
    config.alphaToCoverageEnabled = false;

    // uniforms binding
    int i = 0;
    std::vector<BindGroupLayoutEntry> globalUniforms(1, Default);
    globalUniforms[i].binding = i;
    globalUniforms[i].visibility = ShaderStage::Vertex | ShaderStage::Fragment;
    globalUniforms[i].buffer.type = BufferBindingType::Uniform;
    globalUniforms[i].buffer.minBindingSize = sizeof(FrameUniforms);
    i++;

    config.bindGroupLayouts.push_back(
        pip->createBindGroupLayout("global_uniforms", globalUniforms)
    );

    RenderPipeline pipeline = pip->createRenderPipeline("voxel_pipeline", config);

    return pipeline != nullptr;
}

bool VoxelPipeline::createBindGroup() {
    std::vector<BindGroupEntry> bindings(1);

    int i = 0;
    bindings[i].binding = i;
    bindings[i].buffer = buf->getBuffer("uniform_buffer");
    bindings[i].offset = 0;
    bindings[i].size = sizeof(FrameUniforms);
    i++;

    BindGroup bindGroup = pip->createBindGroup("global_uniforms_bg", "global_uniforms", bindings);

    return bindGroup != nullptr;
}

bool VoxelPipeline::render(TextureView targetView, CommandEncoder encoder) {
    RenderPassDescriptor renderPassDesc = {};
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

    RenderPassDepthStencilAttachment depthStencilAttachment;
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

    Buffer vertexBuffer = buf->getBuffer("vertex_buffer");
    Buffer indexBuffer = buf->getBuffer("index_buffer");
    voxelRenderPass.setVertexBuffer(0, vertexBuffer, 0, vertexBuffer.getSize());
    voxelRenderPass.setIndexBuffer(indexBuffer, IndexFormat::Uint16, 0, indexBuffer.getSize());

    voxelRenderPass.setBindGroup(0, pip->getBindGroup("global_uniforms_bg"), 0, nullptr);

    voxelRenderPass.drawIndexed(indexBuffer.getSize() / sizeof(uint16_t), 1, 0, 0, 0);

    voxelRenderPass.end();
    voxelRenderPass.release();
    return true;
}