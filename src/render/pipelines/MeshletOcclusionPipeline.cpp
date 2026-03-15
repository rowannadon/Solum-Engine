#include "solum_engine/render/pipelines/MeshletOcclusionPipeline.h"

#include <algorithm>
#include <array>
#include <vector>

#include "solum_engine/render/Uniforms.h"

using namespace wgpu;

uint32_t MeshletOcclusionPipeline::computeMipCount(uint32_t width, uint32_t height) {
    uint32_t mipCount = 1u;
    uint32_t w = std::max(width, 1u);
    uint32_t h = std::max(height, 1u);
    while (w > 1u || h > 1u) {
        w = std::max(1u, w / 2u);
        h = std::max(1u, h / 2u);
        ++mipCount;
    }
    return mipCount;
}

bool MeshletOcclusionPipeline::build() {
    return createResources() && createPipeline() && createBindGroup();
}

bool MeshletOcclusionPipeline::build(const MeshletBufferController& meshletBuffers) {
    if (!createResources() || !createPipeline()) {
        return false;
    }
    return refreshMeshBindGroup(meshletBuffers);
}

bool MeshletOcclusionPipeline::recreateResources(const MeshletBufferController& meshletBuffers) {
    if (!createResources()) {
        return false;
    }
    if (!rebuildHierarchyBindings()) {
        return false;
    }
    return refreshMeshBindGroup(meshletBuffers);
}

bool MeshletOcclusionPipeline::refreshMeshBindGroup(const MeshletBufferController& meshletBuffers) {
    if (!meshletBuffers.hasMeshletManager()) {
        return createBindGroup();
    }

    return createBindGroupForSceneBuffers(
        meshletBuffers.visibleTileIdBufferName(),
        meshletBuffers.tileSlotBufferName(),
        meshletBuffers.tileSceneParamsBufferName()
    );
}

bool MeshletOcclusionPipeline::createResources() {
    if (hizSeedBindGroup_) {
        hizSeedBindGroup_.release();
        hizSeedBindGroup_ = nullptr;
    }
    for (BindGroup& bindGroup : hizDownsampleBindGroups_) {
        if (bindGroup) {
            bindGroup.release();
        }
    }
    hizDownsampleBindGroups_.clear();
    for (TextureView& view : occlusionHiZMipViews_) {
        if (view) {
            view.release();
        }
    }
    occlusionHiZMipViews_.clear();

    r_.tex.removeTextureView(kOcclusionHiZViewName);
    r_.tex.removeTexture(kOcclusionHiZTextureName);
    r_.tex.removeTextureView(kOcclusionDepthViewName);
    r_.tex.removeTexture(kOcclusionDepthTextureName);

    const uint32_t width = std::max(1, r_.ctx.width / static_cast<int>(kOcclusionDepthDownsample));
    const uint32_t height = std::max(1, r_.ctx.height / static_cast<int>(kOcclusionDepthDownsample));
    occlusionDepthWidth_ = width;
    occlusionDepthHeight_ = height;
    occlusionHiZMipCount_ = computeMipCount(width, height);

    TextureDescriptor depthDesc = Default;
    depthDesc.label = StringView("meshlet occlusion depth texture");
    depthDesc.dimension = TextureDimension::_2D;
    depthDesc.format = TextureFormat::Depth32Float;
    depthDesc.mipLevelCount = 1;
    depthDesc.sampleCount = 1;
    depthDesc.size = {width, height, 1};
    depthDesc.usage = TextureUsage::RenderAttachment | TextureUsage::TextureBinding;
    if (!r_.tex.createTexture(kOcclusionDepthTextureName, depthDesc)) {
        return false;
    }

    TextureViewDescriptor depthViewDesc = Default;
    depthViewDesc.aspect = TextureAspect::DepthOnly;
    depthViewDesc.baseMipLevel = 0;
    depthViewDesc.mipLevelCount = 1;
    depthViewDesc.dimension = TextureViewDimension::_2D;
    depthViewDesc.format = TextureFormat::Depth32Float;
    if (!r_.tex.createTextureView(kOcclusionDepthTextureName, kOcclusionDepthViewName, depthViewDesc)) {
        return false;
    }

    TextureDescriptor hizDesc = Default;
    hizDesc.label = StringView("meshlet occlusion hiz texture");
    hizDesc.dimension = TextureDimension::_2D;
    hizDesc.format = TextureFormat::R32Float;
    hizDesc.mipLevelCount = occlusionHiZMipCount_;
    hizDesc.sampleCount = 1;
    hizDesc.size = {width, height, 1};
    hizDesc.usage = TextureUsage::TextureBinding | TextureUsage::StorageBinding;
    if (!r_.tex.createTexture(kOcclusionHiZTextureName, hizDesc)) {
        return false;
    }

    TextureViewDescriptor hizViewDesc = Default;
    hizViewDesc.aspect = TextureAspect::All;
    hizViewDesc.baseMipLevel = 0;
    hizViewDesc.mipLevelCount = occlusionHiZMipCount_;
    hizViewDesc.dimension = TextureViewDimension::_2D;
    hizViewDesc.format = TextureFormat::R32Float;
    return r_.tex.createTextureView(kOcclusionHiZTextureName, kOcclusionHiZViewName, hizViewDesc) != nullptr;
}

void MeshletOcclusionPipeline::removeResources() {
    r_.pip.deleteBindGroup(kDepthPrepassBgName);

    r_.tex.removeTextureView(kOcclusionHiZViewName);
    r_.tex.removeTexture(kOcclusionHiZTextureName);
    r_.tex.removeTextureView(kOcclusionDepthViewName);
    r_.tex.removeTexture(kOcclusionDepthTextureName);

    if (hizSeedBindGroup_) {
        hizSeedBindGroup_.release();
        hizSeedBindGroup_ = nullptr;
    }
    for (BindGroup& bindGroup : hizDownsampleBindGroups_) {
        if (bindGroup) {
            bindGroup.release();
        }
    }
    hizDownsampleBindGroups_.clear();
    for (TextureView& view : occlusionHiZMipViews_) {
        if (view) {
            view.release();
        }
    }
    occlusionHiZMipViews_.clear();

    occlusionHiZMipCount_ = 1u;
    occlusionDepthWidth_ = 1u;
    occlusionDepthHeight_ = 1u;
}

bool MeshletOcclusionPipeline::createPipeline() {
    std::vector<BindGroupLayoutEntry> prepassLayoutEntries(4, Default);
    prepassLayoutEntries[0].binding = 0;
    prepassLayoutEntries[0].visibility = ShaderStage::Vertex;
    prepassLayoutEntries[0].buffer.type = BufferBindingType::Uniform;
    prepassLayoutEntries[0].buffer.minBindingSize = sizeof(FrameUniforms);

    prepassLayoutEntries[1].binding = 1;
    prepassLayoutEntries[1].visibility = ShaderStage::Vertex;
    prepassLayoutEntries[1].buffer.type = BufferBindingType::ReadOnlyStorage;

    prepassLayoutEntries[2].binding = 2;
    prepassLayoutEntries[2].visibility = ShaderStage::Vertex;
    prepassLayoutEntries[2].buffer.type = BufferBindingType::ReadOnlyStorage;

    prepassLayoutEntries[3].binding = 3;
    prepassLayoutEntries[3].visibility = ShaderStage::Vertex;
    prepassLayoutEntries[3].buffer.type = BufferBindingType::Uniform;
    prepassLayoutEntries[3].buffer.minBindingSize = sizeof(TileSceneParamsGPU);

    BindGroupLayout prepassBgl = r_.pip.createBindGroupLayout(kDepthPrepassBglName, prepassLayoutEntries);
    if (!prepassBgl) {
        return false;
    }

    PipelineConfig prepassConfig;
    prepassConfig.shaderPath = SHADER_DIR "/meshlet_depth_prepass.wgsl";
    prepassConfig.vertexShaderName = "vs_main";
    prepassConfig.useVertexBuffers = false;
    prepassConfig.useColorTarget = false;
    prepassConfig.useFragmentStage = false;
    prepassConfig.useDepthStencil = true;
    prepassConfig.depthFormat = TextureFormat::Depth32Float;
    prepassConfig.depthWriteEnabled = true;
    prepassConfig.depthCompare = CompareFunction::Less;
    prepassConfig.sampleCount = 1;
    prepassConfig.cullMode = CullMode::Back;
    prepassConfig.bindGroupLayouts.push_back(prepassBgl);
    if (!r_.pip.createRenderPipeline(kDepthPrepassPipelineName, prepassConfig)) {
        return false;
    }

    std::vector<BindGroupLayoutEntry> hizSeedLayoutEntries(2, Default);
    hizSeedLayoutEntries[0].binding = 0;
    hizSeedLayoutEntries[0].visibility = ShaderStage::Compute;
    hizSeedLayoutEntries[0].texture.sampleType = TextureSampleType::Depth;
    hizSeedLayoutEntries[0].texture.viewDimension = TextureViewDimension::_2D;
    hizSeedLayoutEntries[1].binding = 1;
    hizSeedLayoutEntries[1].visibility = ShaderStage::Compute;
    hizSeedLayoutEntries[1].storageTexture.access = StorageTextureAccess::WriteOnly;
    hizSeedLayoutEntries[1].storageTexture.format = TextureFormat::R32Float;
    hizSeedLayoutEntries[1].storageTexture.viewDimension = TextureViewDimension::_2D;

    BindGroupLayout hizSeedBgl = r_.pip.createBindGroupLayout(kHiZSeedBglName, hizSeedLayoutEntries);
    if (!hizSeedBgl) {
        return false;
    }

    std::vector<BindGroupLayoutEntry> hizDownsampleLayoutEntries(2, Default);
    hizDownsampleLayoutEntries[0].binding = 0;
    hizDownsampleLayoutEntries[0].visibility = ShaderStage::Compute;
    hizDownsampleLayoutEntries[0].texture.sampleType = TextureSampleType::UnfilterableFloat;
    hizDownsampleLayoutEntries[0].texture.viewDimension = TextureViewDimension::_2D;
    hizDownsampleLayoutEntries[1].binding = 1;
    hizDownsampleLayoutEntries[1].visibility = ShaderStage::Compute;
    hizDownsampleLayoutEntries[1].storageTexture.access = StorageTextureAccess::WriteOnly;
    hizDownsampleLayoutEntries[1].storageTexture.format = TextureFormat::R32Float;
    hizDownsampleLayoutEntries[1].storageTexture.viewDimension = TextureViewDimension::_2D;

    BindGroupLayout hizDownsampleBgl = r_.pip.createBindGroupLayout(kHiZDownsampleBglName, hizDownsampleLayoutEntries);
    if (!hizDownsampleBgl) {
        return false;
    }

    ComputePipelineConfig seedConfig;
    seedConfig.shaderPath = SHADER_DIR "/meshlet_hiz_seed.wgsl";
    seedConfig.entryPoint = "cs_main";
    seedConfig.bindGroupLayouts.push_back(hizSeedBgl);
    if (!r_.pip.createComputePipeline(kHiZSeedPipelineName, seedConfig)) {
        return false;
    }

    ComputePipelineConfig downsampleConfig;
    downsampleConfig.shaderPath = SHADER_DIR "/meshlet_hiz_downsample.wgsl";
    downsampleConfig.entryPoint = "cs_main";
    downsampleConfig.bindGroupLayouts.push_back(hizDownsampleBgl);
    if (!r_.pip.createComputePipeline(kHiZDownsamplePipelineName, downsampleConfig)) {
        return false;
    }

    return rebuildHierarchyBindings();
}

bool MeshletOcclusionPipeline::createBindGroup() {
    return createBindGroupForSceneBuffers(
        "visible_tile_ids_buffer",
        "tile_slots_buffer",
        "tile_scene_params_buffer"
    );
}

bool MeshletOcclusionPipeline::createBindGroupForSceneBuffers(const std::string& visibleTileIdBufferName,
                                                              const std::string& tileSlotBufferName,
                                                              const std::string& tileSceneParamsBufferName) {
    Buffer uniformBuffer = r_.buf.getBuffer("uniform_buffer");
    Buffer visibleTileIds = r_.buf.getBuffer(visibleTileIdBufferName);
    Buffer tileSlots = r_.buf.getBuffer(tileSlotBufferName);
    Buffer sceneParams = r_.buf.getBuffer(tileSceneParamsBufferName);
    if (!uniformBuffer || !visibleTileIds || !tileSlots || !sceneParams) {
        return false;
    }

    std::vector<BindGroupEntry> entries(4, Default);
    entries[0].binding = 0;
    entries[0].buffer = uniformBuffer;
    entries[0].offset = 0;
    entries[0].size = sizeof(FrameUniforms);

    entries[1].binding = 1;
    entries[1].buffer = visibleTileIds;
    entries[1].offset = 0;
    entries[1].size = visibleTileIds.getSize();

    entries[2].binding = 2;
    entries[2].buffer = tileSlots;
    entries[2].offset = 0;
    entries[2].size = tileSlots.getSize();

    entries[3].binding = 3;
    entries[3].buffer = sceneParams;
    entries[3].offset = 0;
    entries[3].size = sizeof(TileSceneParamsGPU);

    r_.pip.deleteBindGroup(kDepthPrepassBgName);
    return r_.pip.createBindGroup(kDepthPrepassBgName, kDepthPrepassBglName, entries) != nullptr;
}

bool MeshletOcclusionPipeline::rebuildHierarchyBindings() {
    TextureView depthView = r_.tex.getTextureView(kOcclusionDepthViewName);
    Texture hizTexture = r_.tex.getTexture(kOcclusionHiZTextureName);
    BindGroupLayout seedBgl = r_.pip.getBindGroupLayout(kHiZSeedBglName);
    BindGroupLayout downsampleBgl = r_.pip.getBindGroupLayout(kHiZDownsampleBglName);
    if (!depthView || !hizTexture || !seedBgl || !downsampleBgl) {
        return false;
    }

    if (hizSeedBindGroup_) {
        hizSeedBindGroup_.release();
        hizSeedBindGroup_ = nullptr;
    }
    for (BindGroup& bindGroup : hizDownsampleBindGroups_) {
        if (bindGroup) {
            bindGroup.release();
        }
    }
    hizDownsampleBindGroups_.clear();
    for (TextureView& view : occlusionHiZMipViews_) {
        if (view) {
            view.release();
        }
    }
    occlusionHiZMipViews_.clear();

    const uint32_t mipCount = std::max(occlusionHiZMipCount_, 1u);
    occlusionHiZMipViews_.reserve(mipCount);

    for (uint32_t mipLevel = 0u; mipLevel < mipCount; ++mipLevel) {
        TextureViewDescriptor viewDesc = Default;
        viewDesc.aspect = TextureAspect::All;
        viewDesc.baseMipLevel = mipLevel;
        viewDesc.mipLevelCount = 1;
        viewDesc.dimension = TextureViewDimension::_2D;
        viewDesc.format = TextureFormat::R32Float;
        TextureView view = hizTexture.createView(viewDesc);
        if (!view) {
            return false;
        }
        occlusionHiZMipViews_.push_back(view);
    }

    std::array<BindGroupEntry, 2> seedEntries{};
    seedEntries[0] = Default;
    seedEntries[0].binding = 0;
    seedEntries[0].textureView = depthView;
    seedEntries[1] = Default;
    seedEntries[1].binding = 1;
    seedEntries[1].textureView = occlusionHiZMipViews_[0];

    BindGroupDescriptor seedBgDesc = Default;
    seedBgDesc.label = StringView("meshlet hiz seed bg");
    seedBgDesc.layout = seedBgl;
    seedBgDesc.entryCount = static_cast<uint32_t>(seedEntries.size());
    seedBgDesc.entries = seedEntries.data();
    hizSeedBindGroup_ = r_.ctx.getDevice().createBindGroup(seedBgDesc);
    if (!hizSeedBindGroup_) {
        return false;
    }

    if (mipCount <= 1u) {
        return true;
    }

    hizDownsampleBindGroups_.reserve(mipCount - 1u);
    for (uint32_t mip = 1u; mip < mipCount; ++mip) {
        std::array<BindGroupEntry, 2> entries{};
        entries[0] = Default;
        entries[0].binding = 0;
        entries[0].textureView = occlusionHiZMipViews_[mip - 1u];
        entries[1] = Default;
        entries[1].binding = 1;
        entries[1].textureView = occlusionHiZMipViews_[mip];

        BindGroupDescriptor bgDesc = Default;
        bgDesc.label = StringView("meshlet hiz downsample bg");
        bgDesc.layout = downsampleBgl;
        bgDesc.entryCount = static_cast<uint32_t>(entries.size());
        bgDesc.entries = entries.data();
        BindGroup bindGroup = r_.ctx.getDevice().createBindGroup(bgDesc);
        if (!bindGroup) {
            return false;
        }
        hizDownsampleBindGroups_.push_back(bindGroup);
    }

    return true;
}

void MeshletOcclusionPipeline::encodeDepthPrepass(CommandEncoder encoder,
                                                  const MeshletBufferController& meshletBuffers) {
    RenderPipeline prepassPipeline = r_.pip.getPipeline(kDepthPrepassPipelineName);
    BindGroup prepassBindGroup = r_.pip.getBindGroup(kDepthPrepassBgName);
    if (!prepassPipeline || !prepassBindGroup) {
        return;
    }

    const uint32_t visibleTileCount = meshletBuffers.visibleTileCount();
    if (visibleTileCount == 0u) {
        return;
    }

    TextureView occlusionDepthView = r_.tex.getTextureView(kOcclusionDepthViewName);
    if (!occlusionDepthView) {
        return;
    }

    RenderPassDepthStencilAttachment depthAttachment = Default;
    depthAttachment.view = occlusionDepthView;
    depthAttachment.depthClearValue = 1.0f;
    depthAttachment.depthLoadOp = LoadOp::Clear;
    depthAttachment.depthStoreOp = StoreOp::Store;
    depthAttachment.depthReadOnly = false;
    depthAttachment.stencilClearValue = 0;
    depthAttachment.stencilLoadOp = LoadOp::Undefined;
    depthAttachment.stencilStoreOp = StoreOp::Undefined;
    depthAttachment.stencilReadOnly = true;

    RenderPassDescriptor passDesc = Default;
    passDesc.colorAttachmentCount = 0;
    passDesc.colorAttachments = nullptr;
    passDesc.depthStencilAttachment = &depthAttachment;

    RenderPassEncoder pass = encoder.beginRenderPass(passDesc);
    pass.setPipeline(prepassPipeline);
    pass.setBindGroup(0, prepassBindGroup, 0, nullptr);
    pass.draw(36u, visibleTileCount, 0, 0);
    pass.end();
    pass.release();
}

void MeshletOcclusionPipeline::encodeHierarchyPass(CommandEncoder encoder) {
    ComputePipeline seedPipeline = r_.pip.getComputePipeline(kHiZSeedPipelineName);
    ComputePipeline downsamplePipeline = r_.pip.getComputePipeline(kHiZDownsamplePipelineName);
    if (!seedPipeline || !downsamplePipeline) {
        return;
    }
    if ((hizSeedBindGroup_ == nullptr || occlusionHiZMipViews_.empty()) && !rebuildHierarchyBindings()) {
        return;
    }

    const uint32_t mipCount = std::max(occlusionHiZMipCount_, 1u);
    ComputePassDescriptor passDesc = Default;
    ComputePassEncoder pass = encoder.beginComputePass(passDesc);

    pass.setPipeline(seedPipeline);
    pass.setBindGroup(0, hizSeedBindGroup_, 0, nullptr);
    const uint32_t gx =
        (std::max(occlusionDepthWidth_, 1u) + kOcclusionHiZWorkgroupSize - 1u) /
        kOcclusionHiZWorkgroupSize;
    const uint32_t gy =
        (std::max(occlusionDepthHeight_, 1u) + kOcclusionHiZWorkgroupSize - 1u) /
        kOcclusionHiZWorkgroupSize;
    pass.dispatchWorkgroups(gx, gy, 1u);

    for (uint32_t mip = 1u; mip < mipCount; ++mip) {
        const size_t bindGroupIndex = static_cast<size_t>(mip - 1u);
        if (bindGroupIndex >= hizDownsampleBindGroups_.size() || !hizDownsampleBindGroups_[bindGroupIndex]) {
            continue;
        }
        const uint32_t mipWidth = std::max(1u, occlusionDepthWidth_ >> mip);
        const uint32_t mipHeight = std::max(1u, occlusionDepthHeight_ >> mip);
        const uint32_t dispatchX = (mipWidth + kOcclusionHiZWorkgroupSize - 1u) / kOcclusionHiZWorkgroupSize;
        const uint32_t dispatchY = (mipHeight + kOcclusionHiZWorkgroupSize - 1u) / kOcclusionHiZWorkgroupSize;
        pass.setPipeline(downsamplePipeline);
        pass.setBindGroup(0, hizDownsampleBindGroups_[bindGroupIndex], 0, nullptr);
        pass.dispatchWorkgroups(dispatchX, dispatchY, 1u);
    }

    pass.end();
    pass.release();
}
