#include "solum_engine/render/pipelines/MeshletOcclusionPipeline.h"

#include <algorithm>
#include <array>
#include <vector>

#include "solum_engine/render/MeshletManager.h"
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
    return refreshMeshBindGroup(meshletBuffers);
}

bool MeshletOcclusionPipeline::refreshMeshBindGroup(const MeshletBufferController& meshletBuffers) {
    if (!meshletBuffers.hasMeshletManager()) {
        return createBindGroup();
    }

    return createBindGroupForMeshBuffers(
        meshletBuffers.activeMeshDataBufferName(),
        meshletBuffers.activeMeshMetadataBufferName()
    );
}

bool MeshletOcclusionPipeline::createResources() {
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
    depthDesc.viewFormatCount = 0;
    depthDesc.viewFormats = nullptr;
    if (!r_.tex.createTexture(kOcclusionDepthTextureName, depthDesc)) {
        return false;
    }

    TextureViewDescriptor depthViewDesc = Default;
    depthViewDesc.aspect = TextureAspect::DepthOnly;
    depthViewDesc.baseArrayLayer = 0;
    depthViewDesc.arrayLayerCount = 1;
    depthViewDesc.baseMipLevel = 0;
    depthViewDesc.mipLevelCount = 1;
    depthViewDesc.dimension = TextureViewDimension::_2D;
    depthViewDesc.format = TextureFormat::Depth32Float;
    if (!r_.tex.createTextureView(
            kOcclusionDepthTextureName,
            kOcclusionDepthViewName,
            depthViewDesc)) {
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
    hizDesc.viewFormatCount = 0;
    hizDesc.viewFormats = nullptr;
    if (!r_.tex.createTexture(kOcclusionHiZTextureName, hizDesc)) {
        return false;
    }

    TextureViewDescriptor hizViewDesc = Default;
    hizViewDesc.aspect = TextureAspect::All;
    hizViewDesc.baseArrayLayer = 0;
    hizViewDesc.arrayLayerCount = 1;
    hizViewDesc.baseMipLevel = 0;
    hizViewDesc.mipLevelCount = occlusionHiZMipCount_;
    hizViewDesc.dimension = TextureViewDimension::_2D;
    hizViewDesc.format = TextureFormat::R32Float;
    return r_.tex.createTextureView(
               kOcclusionHiZTextureName,
               kOcclusionHiZViewName,
               hizViewDesc
           ) != nullptr;
}

void MeshletOcclusionPipeline::removeResources() {
    r_.pip.deleteBindGroup(kDepthPrepassBgName);

    r_.tex.removeTextureView(kOcclusionHiZViewName);
    r_.tex.removeTexture(kOcclusionHiZTextureName);
    r_.tex.removeTextureView(kOcclusionDepthViewName);
    r_.tex.removeTexture(kOcclusionDepthTextureName);

    occlusionHiZMipCount_ = 1u;
    occlusionDepthWidth_ = 1u;
    occlusionDepthHeight_ = 1u;
}

bool MeshletOcclusionPipeline::createPipeline() {
    std::vector<BindGroupLayoutEntry> prepassLayoutEntries(3, Default);
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

    BindGroupLayout hizDownsampleBgl =
        r_.pip.createBindGroupLayout(kHiZDownsampleBglName, hizDownsampleLayoutEntries);
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

    return true;
}

bool MeshletOcclusionPipeline::createBindGroup() {
    return createBindGroupForMeshBuffers(
        MeshletManager::meshDataBufferName(0),
        MeshletManager::meshMetadataBufferName(0)
    );
}

bool MeshletOcclusionPipeline::createBindGroupForMeshBuffers(const std::string& meshDataBufferName,
                                                             const std::string& metadataBufferName) {
    Buffer uniformBuffer = r_.buf.getBuffer("uniform_buffer");
    Buffer meshDataBuffer = r_.buf.getBuffer(meshDataBufferName);
    Buffer metadataBuffer = r_.buf.getBuffer(metadataBufferName);
    if (!uniformBuffer || !meshDataBuffer || !metadataBuffer) {
        return false;
    }

    std::vector<BindGroupEntry> entries(3, Default);
    entries[0].binding = 0;
    entries[0].buffer = uniformBuffer;
    entries[0].offset = 0;
    entries[0].size = sizeof(FrameUniforms);

    entries[1].binding = 1;
    entries[1].buffer = meshDataBuffer;
    entries[1].offset = 0;
    entries[1].size = meshDataBuffer.getSize();

    entries[2].binding = 2;
    entries[2].buffer = metadataBuffer;
    entries[2].offset = 0;
    entries[2].size = metadataBuffer.getSize();

    r_.pip.deleteBindGroup(kDepthPrepassBgName);
    return r_.pip.createBindGroup(kDepthPrepassBgName, kDepthPrepassBglName, entries) != nullptr;
}

void MeshletOcclusionPipeline::encodeDepthPrepass(CommandEncoder encoder,
                                                  const MeshletBufferController& meshletBuffers) {
    RenderPipeline prepassPipeline = r_.pip.getPipeline(kDepthPrepassPipelineName);
    BindGroup prepassBindGroup = r_.pip.getBindGroup(kDepthPrepassBgName);
    if (!prepassPipeline || !prepassBindGroup) {
        return;
    }

    const uint32_t meshletCount = meshletBuffers.effectiveMeshletCountForPasses();
    if (meshletCount == 0u) {
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
    passDesc.timestampWrites = nullptr;

    RenderPassEncoder pass = encoder.beginRenderPass(passDesc);
    pass.setPipeline(prepassPipeline);
    pass.setBindGroup(0, prepassBindGroup, 0, nullptr);
    pass.draw(MESHLET_VERTEX_CAPACITY, meshletCount, 0, 0);
    pass.end();
    pass.release();
}

void MeshletOcclusionPipeline::encodeHierarchyPass(CommandEncoder encoder) {
    ComputePipeline seedPipeline = r_.pip.getComputePipeline(kHiZSeedPipelineName);
    ComputePipeline downsamplePipeline = r_.pip.getComputePipeline(kHiZDownsamplePipelineName);
    BindGroupLayout seedBgl = r_.pip.getBindGroupLayout(kHiZSeedBglName);
    BindGroupLayout downsampleBgl = r_.pip.getBindGroupLayout(kHiZDownsampleBglName);
    if (!seedPipeline || !downsamplePipeline || !seedBgl || !downsampleBgl) {
        return;
    }

    TextureView depthView = r_.tex.getTextureView(kOcclusionDepthViewName);
    Texture hizTexture = r_.tex.getTexture(kOcclusionHiZTextureName);
    if (!depthView || !hizTexture) {
        return;
    }

    const uint32_t mipCount = std::max(occlusionHiZMipCount_, 1u);
    const Device device = r_.ctx.getDevice();

    auto createHizMipView = [hizTexture](uint32_t mipLevel) -> TextureView {
        TextureViewDescriptor viewDesc = Default;
        viewDesc.aspect = TextureAspect::All;
        viewDesc.baseArrayLayer = 0;
        viewDesc.arrayLayerCount = 1;
        viewDesc.baseMipLevel = mipLevel;
        viewDesc.mipLevelCount = 1;
        viewDesc.dimension = TextureViewDimension::_2D;
        viewDesc.format = TextureFormat::R32Float;
        return hizTexture.createView(viewDesc);
    };

    ComputePassDescriptor passDesc = Default;
    ComputePassEncoder pass = encoder.beginComputePass(passDesc);

    {
        TextureView dstMip0View = createHizMipView(0u);
        if (dstMip0View) {
            std::array<BindGroupEntry, 2> entries{};
            entries[0] = Default;
            entries[0].binding = 0;
            entries[0].textureView = depthView;
            entries[1] = Default;
            entries[1].binding = 1;
            entries[1].textureView = dstMip0View;

            BindGroupDescriptor bgDesc = Default;
            bgDesc.label = StringView("meshlet hiz seed bg");
            bgDesc.layout = seedBgl;
            bgDesc.entryCount = static_cast<uint32_t>(entries.size());
            bgDesc.entries = entries.data();
            BindGroup bg = device.createBindGroup(bgDesc);

            if (bg) {
                pass.setPipeline(seedPipeline);
                pass.setBindGroup(0, bg, 0, nullptr);
                const uint32_t gx =
                    (std::max(occlusionDepthWidth_, 1u) + kOcclusionHiZWorkgroupSize - 1u) /
                    kOcclusionHiZWorkgroupSize;
                const uint32_t gy =
                    (std::max(occlusionDepthHeight_, 1u) + kOcclusionHiZWorkgroupSize - 1u) /
                    kOcclusionHiZWorkgroupSize;
                pass.dispatchWorkgroups(gx, gy, 1u);
                bg.release();
            }

            dstMip0View.release();
        }
    }

    for (uint32_t mip = 1u; mip < mipCount; ++mip) {
        TextureView srcView = createHizMipView(mip - 1u);
        TextureView dstView = createHizMipView(mip);
        if (!srcView || !dstView) {
            if (srcView) {
                srcView.release();
            }
            if (dstView) {
                dstView.release();
            }
            continue;
        }

        std::array<BindGroupEntry, 2> entries{};
        entries[0] = Default;
        entries[0].binding = 0;
        entries[0].textureView = srcView;
        entries[1] = Default;
        entries[1].binding = 1;
        entries[1].textureView = dstView;

        BindGroupDescriptor bgDesc = Default;
        bgDesc.label = StringView("meshlet hiz downsample bg");
        bgDesc.layout = downsampleBgl;
        bgDesc.entryCount = static_cast<uint32_t>(entries.size());
        bgDesc.entries = entries.data();
        BindGroup bg = device.createBindGroup(bgDesc);

        if (bg) {
            const uint32_t mipWidth = std::max(1u, occlusionDepthWidth_ >> mip);
            const uint32_t mipHeight = std::max(1u, occlusionDepthHeight_ >> mip);
            const uint32_t gx = (mipWidth + kOcclusionHiZWorkgroupSize - 1u) / kOcclusionHiZWorkgroupSize;
            const uint32_t gy = (mipHeight + kOcclusionHiZWorkgroupSize - 1u) / kOcclusionHiZWorkgroupSize;
            pass.setPipeline(downsamplePipeline);
            pass.setBindGroup(0, bg, 0, nullptr);
            pass.dispatchWorkgroups(gx, gy, 1u);
            bg.release();
        }

        srcView.release();
        dstView.release();
    }

    pass.end();
    pass.release();
}

bool MeshletOcclusionPipeline::render(
    TextureView /* targetView */,
    CommandEncoder /* encoder */,
    const std::function<void(RenderPassEncoder&)>& /* overlayCallback */
) {
    return false;
}
