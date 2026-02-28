#include "solum_engine/render/pipelines/MeshletCullingPipeline.h"

#include <algorithm>
#include <vector>

#include "solum_engine/render/MeshletManager.h"
#include "solum_engine/render/MeshletTypes.h"
#include "solum_engine/render/Uniforms.h"

using namespace wgpu;

bool MeshletCullingPipeline::build() {
    return createResources() && createPipeline() && createBindGroup();
}

bool MeshletCullingPipeline::build(const MeshletBufferController& meshletBuffers,
                                   uint32_t occlusionHiZMipCount,
                                   const char* occlusionHiZViewName) {
    if (!createResources() || !createPipeline()) {
        return false;
    }

    updateCullParams(meshletBuffers.meshletCount(), occlusionHiZMipCount);
    return refreshBindGroup(meshletBuffers, occlusionHiZViewName);
}

bool MeshletCullingPipeline::refreshBindGroup(const MeshletBufferController& meshletBuffers,
                                              const char* occlusionHiZViewName) {
    activeHiZViewName_ = (occlusionHiZViewName != nullptr) ? occlusionHiZViewName : kDefaultHiZViewName;

    if (!meshletBuffers.hasMeshletManager()) {
        return createBindGroup();
    }

    return createBindGroupForMeshBuffers(
        meshletBuffers.activeMeshAabbBufferName(),
        meshletBuffers.activeVisibleMeshletIndexBufferName(),
        activeHiZViewName_.c_str()
    );
}

bool MeshletCullingPipeline::createResources() {
    {
        BufferDescriptor paramsDesc = Default;
        paramsDesc.label = StringView("meshlet cull params buffer");
        paramsDesc.size = 16u;
        paramsDesc.usage = BufferUsage::Uniform | BufferUsage::CopyDst;
        paramsDesc.mappedAtCreation = false;
        if (!r_.buf.createBuffer(kCullParamsBufferName, paramsDesc)) {
            return false;
        }
    }

    {
        BufferDescriptor indirectDesc = Default;
        indirectDesc.label = StringView("meshlet cull indirect args buffer");
        indirectDesc.size = sizeof(uint32_t) * 4u;
        indirectDesc.usage = BufferUsage::Storage | BufferUsage::Indirect | BufferUsage::CopyDst;
        indirectDesc.mappedAtCreation = false;
        if (!r_.buf.createBuffer(kIndirectArgsBufferName, indirectDesc)) {
            return false;
        }

        const uint32_t safeDrawArgs[4] = {MESHLET_VERTEX_CAPACITY, 0u, 0u, 0u};
        r_.buf.writeBuffer(kIndirectArgsBufferName, 0u, safeDrawArgs, sizeof(safeDrawArgs));
    }

    {
        BufferDescriptor resetDesc = Default;
        resetDesc.label = StringView("meshlet cull indirect reset buffer");
        resetDesc.size = sizeof(uint32_t) * 4u;
        resetDesc.usage = BufferUsage::CopySrc | BufferUsage::CopyDst;
        resetDesc.mappedAtCreation = false;
        if (!r_.buf.createBuffer(kIndirectResetBufferName, resetDesc)) {
            return false;
        }

        const uint32_t drawArgsReset[4] = {MESHLET_VERTEX_CAPACITY, 0u, 0u, 0u};
        r_.buf.writeBuffer(
            kIndirectResetBufferName,
            0u,
            drawArgsReset,
            sizeof(drawArgsReset)
        );
    }

    return true;
}

void MeshletCullingPipeline::removeResources() {
    r_.pip.deleteBindGroup(kCullBgName);
    r_.buf.deleteBuffer(kCullParamsBufferName);
    r_.buf.deleteBuffer(kIndirectArgsBufferName);
    r_.buf.deleteBuffer(kIndirectResetBufferName);
}

bool MeshletCullingPipeline::createPipeline() {
    std::vector<BindGroupLayoutEntry> cullLayoutEntries(6, Default);
    cullLayoutEntries[0].binding = 0;
    cullLayoutEntries[0].visibility = ShaderStage::Compute;
    cullLayoutEntries[0].buffer.type = BufferBindingType::Uniform;
    cullLayoutEntries[0].buffer.minBindingSize = sizeof(FrameUniforms);

    cullLayoutEntries[1].binding = 1;
    cullLayoutEntries[1].visibility = ShaderStage::Compute;
    cullLayoutEntries[1].buffer.type = BufferBindingType::ReadOnlyStorage;

    cullLayoutEntries[2].binding = 2;
    cullLayoutEntries[2].visibility = ShaderStage::Compute;
    cullLayoutEntries[2].buffer.type = BufferBindingType::Storage;

    cullLayoutEntries[3].binding = 3;
    cullLayoutEntries[3].visibility = ShaderStage::Compute;
    cullLayoutEntries[3].buffer.type = BufferBindingType::Storage;

    cullLayoutEntries[4].binding = 4;
    cullLayoutEntries[4].visibility = ShaderStage::Compute;
    cullLayoutEntries[4].buffer.type = BufferBindingType::Uniform;
    cullLayoutEntries[4].buffer.minBindingSize = 16u;

    cullLayoutEntries[5].binding = 5;
    cullLayoutEntries[5].visibility = ShaderStage::Compute;
    cullLayoutEntries[5].texture.sampleType = TextureSampleType::UnfilterableFloat;
    cullLayoutEntries[5].texture.viewDimension = TextureViewDimension::_2D;

    BindGroupLayout cullBgl = r_.pip.createBindGroupLayout(kCullBglName, cullLayoutEntries);
    if (!cullBgl) {
        return false;
    }

    ComputePipelineConfig pipelineConfig;
    pipelineConfig.shaderPath = SHADER_DIR "/meshlet_cull.wgsl";
    pipelineConfig.entryPoint = "cs_main";
    pipelineConfig.bindGroupLayouts.push_back(cullBgl);

    return r_.pip.createComputePipeline(kCullPipelineName, pipelineConfig) != nullptr;
}

bool MeshletCullingPipeline::createBindGroup() {
    return createBindGroupForMeshBuffers(
        MeshletManager::meshAabbBufferName(0),
        MeshletManager::visibleMeshletIndexBufferName(0),
        activeHiZViewName_.c_str()
    );
}

bool MeshletCullingPipeline::createBindGroupForMeshBuffers(const std::string& meshletAabbBufferName,
                                                           const std::string& visibleIndicesBufferName,
                                                           const char* occlusionHiZViewName) {
    BindGroupLayout cullBgl = r_.pip.getBindGroupLayout(kCullBglName);
    if (!cullBgl) {
        return false;
    }

    Buffer uniformBuffer = r_.buf.getBuffer("uniform_buffer");
    Buffer meshletAabbBuffer = r_.buf.getBuffer(meshletAabbBufferName);
    Buffer visibleIndicesBuffer = r_.buf.getBuffer(visibleIndicesBufferName);
    Buffer drawArgsBuffer = r_.buf.getBuffer(kIndirectArgsBufferName);
    Buffer cullParamsBuffer = r_.buf.getBuffer(kCullParamsBufferName);
    TextureView occlusionHiZView = r_.tex.getTextureView(
        (occlusionHiZViewName != nullptr) ? occlusionHiZViewName : kDefaultHiZViewName
    );

    if (!uniformBuffer || !meshletAabbBuffer || !visibleIndicesBuffer ||
        !drawArgsBuffer || !cullParamsBuffer || !occlusionHiZView) {
        return false;
    }

    std::vector<BindGroupEntry> entries(6, Default);
    entries[0].binding = 0;
    entries[0].buffer = uniformBuffer;
    entries[0].offset = 0;
    entries[0].size = sizeof(FrameUniforms);

    entries[1].binding = 1;
    entries[1].buffer = meshletAabbBuffer;
    entries[1].offset = 0;
    entries[1].size = meshletAabbBuffer.getSize();

    entries[2].binding = 2;
    entries[2].buffer = visibleIndicesBuffer;
    entries[2].offset = 0;
    entries[2].size = visibleIndicesBuffer.getSize();

    entries[3].binding = 3;
    entries[3].buffer = drawArgsBuffer;
    entries[3].offset = 0;
    entries[3].size = drawArgsBuffer.getSize();

    entries[4].binding = 4;
    entries[4].buffer = cullParamsBuffer;
    entries[4].offset = 0;
    entries[4].size = 16u;

    entries[5].binding = 5;
    entries[5].textureView = occlusionHiZView;

    r_.pip.deleteBindGroup(kCullBgName);
    return r_.pip.createBindGroup(kCullBgName, kCullBglName, entries) != nullptr;
}

void MeshletCullingPipeline::updateCullParams(uint32_t meshletCount, uint32_t occlusionHiZMipCount) {
    const uint32_t params[4] = {meshletCount, std::max(occlusionHiZMipCount, 1u), 0u, 0u};
    r_.buf.writeBuffer(kCullParamsBufferName, 0u, params, sizeof(params));
}

void MeshletCullingPipeline::encode(CommandEncoder encoder,
                                    const MeshletBufferController& meshletBuffers) {
    ComputePipeline cullPipeline = r_.pip.getComputePipeline(kCullPipelineName);
    BindGroup cullBindGroup = r_.pip.getBindGroup(kCullBgName);
    if (!cullPipeline || !cullBindGroup) {
        return;
    }

    Buffer resetBuffer = r_.buf.getBuffer(kIndirectResetBufferName);
    Buffer indirectArgsBuffer = r_.buf.getBuffer(kIndirectArgsBufferName);
    if (!resetBuffer || !indirectArgsBuffer) {
        return;
    }

    encoder.copyBufferToBuffer(
        resetBuffer,
        0u,
        indirectArgsBuffer,
        0u,
        sizeof(uint32_t) * 4u
    );

    const uint32_t meshletCount = meshletBuffers.effectiveMeshletCountForPasses();
    if (meshletCount == 0u) {
        return;
    }

    ComputePassDescriptor passDesc = Default;
    ComputePassEncoder pass = encoder.beginComputePass(passDesc);
    pass.setPipeline(cullPipeline);
    pass.setBindGroup(0, cullBindGroup, 0, nullptr);
    const uint32_t workgroupCount =
        (meshletCount + kMeshletCullWorkgroupSize - 1u) / kMeshletCullWorkgroupSize;
    pass.dispatchWorkgroups(workgroupCount, 1u, 1u);
    pass.end();
    pass.release();
}

bool MeshletCullingPipeline::render(
    TextureView /* targetView */,
    CommandEncoder /* encoder */,
    const std::function<void(RenderPassEncoder&)>& /* overlayCallback */
) {
    return false;
}
