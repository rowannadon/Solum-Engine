#include "solum_engine/render/pipelines/MeshletCullingPipeline.h"

#include <vector>

#include "solum_engine/render/MeshletTypes.h"

using namespace wgpu;

MeshletCullingPipeline::MeshletCullingPipeline(RenderServices& r, Config config)
    : AbstractRenderPipeline(r),
      indirectArgsBufferName_(prefixedName(config.namePrefix, "meshlet_cull_indirect_args_buffer")),
      indirectResetBufferName_(prefixedName(config.namePrefix, "meshlet_cull_indirect_reset_buffer")),
      cullBglName_(prefixedName(config.namePrefix, "meshlet_cull_bgl")),
      cullBgName_(prefixedName(config.namePrefix, "meshlet_cull_bg")),
      cullPipelineName_(prefixedName(config.namePrefix, "meshlet_cull_pipeline")) {}

std::string MeshletCullingPipeline::prefixedName(const std::string& prefix, const char* baseName) {
    if (prefix.empty()) {
        return std::string(baseName);
    }
    return prefix + "_" + baseName;
}

bool MeshletCullingPipeline::build() {
    return createResources() && createPipeline() && createBindGroup();
}

bool MeshletCullingPipeline::build(const MeshletBufferController& meshletBuffers,
                                   uint32_t occlusionHiZMipCount,
                                   const char* occlusionHiZViewName) {
    (void)occlusionHiZMipCount;
    (void)occlusionHiZViewName;
    if (!createResources() || !createPipeline()) {
        return false;
    }
    return refreshBindGroup(meshletBuffers, nullptr);
}

bool MeshletCullingPipeline::refreshBindGroup(const MeshletBufferController& meshletBuffers,
                                              const char* occlusionHiZViewName) {
    (void)occlusionHiZViewName;
    if (!meshletBuffers.hasMeshletManager()) {
        return createBindGroup();
    }

    return createBindGroupForSceneBuffers(
        meshletBuffers.visibleTileIdBufferName(),
        meshletBuffers.tileSlotBufferName(),
        meshletBuffers.residentTileLodBufferName(),
        meshletBuffers.activeVisibleMeshletIndexBufferName(),
        meshletBuffers.tileSceneParamsBufferName()
    );
}

bool MeshletCullingPipeline::createResources() {
    BufferDescriptor indirectDesc = Default;
    indirectDesc.label = StringView("meshlet tile expansion indirect args buffer");
    indirectDesc.size = sizeof(uint32_t) * 4u;
    indirectDesc.usage = BufferUsage::Storage | BufferUsage::Indirect | BufferUsage::CopyDst;
    if (!r_.buf.createBuffer(indirectArgsBufferName_, indirectDesc)) {
        return false;
    }

    const uint32_t safeDrawArgs[4] = {MESHLET_VERTEX_CAPACITY, 0u, 0u, 0u};
    r_.buf.writeBuffer(indirectArgsBufferName_, 0u, safeDrawArgs, sizeof(safeDrawArgs));

    BufferDescriptor resetDesc = Default;
    resetDesc.label = StringView("meshlet tile expansion indirect reset buffer");
    resetDesc.size = sizeof(uint32_t) * 4u;
    resetDesc.usage = BufferUsage::CopySrc | BufferUsage::CopyDst;
    if (!r_.buf.createBuffer(indirectResetBufferName_, resetDesc)) {
        return false;
    }

    r_.buf.writeBuffer(indirectResetBufferName_, 0u, safeDrawArgs, sizeof(safeDrawArgs));
    return true;
}

void MeshletCullingPipeline::removeResources() {
    r_.pip.deleteBindGroup(cullBgName_);
    r_.buf.deleteBuffer(indirectArgsBufferName_);
    r_.buf.deleteBuffer(indirectResetBufferName_);
}

bool MeshletCullingPipeline::createPipeline() {
    std::vector<BindGroupLayoutEntry> cullLayoutEntries(6, Default);
    cullLayoutEntries[0].binding = 0;
    cullLayoutEntries[0].visibility = ShaderStage::Compute;
    cullLayoutEntries[0].buffer.type = BufferBindingType::ReadOnlyStorage;

    cullLayoutEntries[1].binding = 1;
    cullLayoutEntries[1].visibility = ShaderStage::Compute;
    cullLayoutEntries[1].buffer.type = BufferBindingType::ReadOnlyStorage;

    cullLayoutEntries[2].binding = 2;
    cullLayoutEntries[2].visibility = ShaderStage::Compute;
    cullLayoutEntries[2].buffer.type = BufferBindingType::ReadOnlyStorage;

    cullLayoutEntries[3].binding = 3;
    cullLayoutEntries[3].visibility = ShaderStage::Compute;
    cullLayoutEntries[3].buffer.type = BufferBindingType::Storage;

    cullLayoutEntries[4].binding = 4;
    cullLayoutEntries[4].visibility = ShaderStage::Compute;
    cullLayoutEntries[4].buffer.type = BufferBindingType::Storage;

    cullLayoutEntries[5].binding = 5;
    cullLayoutEntries[5].visibility = ShaderStage::Compute;
    cullLayoutEntries[5].buffer.type = BufferBindingType::Uniform;
    cullLayoutEntries[5].buffer.minBindingSize = sizeof(TileSceneParamsGPU);

    BindGroupLayout cullBgl = r_.pip.createBindGroupLayout(cullBglName_, cullLayoutEntries);
    if (!cullBgl) {
        return false;
    }

    ComputePipelineConfig pipelineConfig;
    pipelineConfig.shaderPath = SHADER_DIR "/meshlet_cull.wgsl";
    pipelineConfig.entryPoint = "cs_main";
    pipelineConfig.bindGroupLayouts.push_back(cullBgl);
    return r_.pip.createComputePipeline(cullPipelineName_, pipelineConfig) != nullptr;
}

bool MeshletCullingPipeline::createBindGroup() {
    return createBindGroupForSceneBuffers(
        "visible_tile_ids_buffer",
        "tile_slots_buffer",
        "resident_tile_lods_buffer",
        "visible_meshlet_indices_buffer",
        "tile_scene_params_buffer"
    );
}

bool MeshletCullingPipeline::createBindGroupForSceneBuffers(const std::string& visibleTileIdBufferName,
                                                            const std::string& tileSlotBufferName,
                                                            const std::string& residentTileLodBufferName,
                                                            const std::string& visibleIndicesBufferName,
                                                            const std::string& tileSceneParamsBufferName) {
    BindGroupLayout cullBgl = r_.pip.getBindGroupLayout(cullBglName_);
    if (!cullBgl) {
        return false;
    }

    Buffer visibleTileIds = r_.buf.getBuffer(visibleTileIdBufferName);
    Buffer tileSlots = r_.buf.getBuffer(tileSlotBufferName);
    Buffer residentTileLods = r_.buf.getBuffer(residentTileLodBufferName);
    Buffer visibleIndicesBuffer = r_.buf.getBuffer(visibleIndicesBufferName);
    Buffer drawArgsBuffer = r_.buf.getBuffer(indirectArgsBufferName_);
    Buffer sceneParams = r_.buf.getBuffer(tileSceneParamsBufferName);

    if (!visibleTileIds || !tileSlots || !residentTileLods || !visibleIndicesBuffer || !drawArgsBuffer || !sceneParams) {
        return false;
    }

    std::vector<BindGroupEntry> entries(6, Default);
    entries[0].binding = 0;
    entries[0].buffer = visibleTileIds;
    entries[0].offset = 0;
    entries[0].size = visibleTileIds.getSize();

    entries[1].binding = 1;
    entries[1].buffer = tileSlots;
    entries[1].offset = 0;
    entries[1].size = tileSlots.getSize();

    entries[2].binding = 2;
    entries[2].buffer = residentTileLods;
    entries[2].offset = 0;
    entries[2].size = residentTileLods.getSize();

    entries[3].binding = 3;
    entries[3].buffer = visibleIndicesBuffer;
    entries[3].offset = 0;
    entries[3].size = visibleIndicesBuffer.getSize();

    entries[4].binding = 4;
    entries[4].buffer = drawArgsBuffer;
    entries[4].offset = 0;
    entries[4].size = drawArgsBuffer.getSize();

    entries[5].binding = 5;
    entries[5].buffer = sceneParams;
    entries[5].offset = 0;
    entries[5].size = sizeof(TileSceneParamsGPU);

    r_.pip.deleteBindGroup(cullBgName_);
    return r_.pip.createBindGroup(cullBgName_, cullBglName_, entries) != nullptr;
}

void MeshletCullingPipeline::updateCullParams(uint32_t meshletCount,
                                              uint32_t occlusionHiZMipCount,
                                              uint32_t activeRangeCount) {
    (void)meshletCount;
    (void)occlusionHiZMipCount;
    (void)activeRangeCount;
}

void MeshletCullingPipeline::encode(CommandEncoder encoder,
                                    const MeshletBufferController& meshletBuffers) {
    ComputePipeline cullPipeline = r_.pip.getComputePipeline(cullPipelineName_);
    BindGroup cullBindGroup = r_.pip.getBindGroup(cullBgName_);
    if (!cullPipeline || !cullBindGroup) {
        return;
    }

    Buffer resetBuffer = r_.buf.getBuffer(indirectResetBufferName_);
    Buffer indirectArgsBuffer = r_.buf.getBuffer(indirectArgsBufferName_);
    if (!resetBuffer || !indirectArgsBuffer) {
        return;
    }

    encoder.copyBufferToBuffer(resetBuffer, 0u, indirectArgsBuffer, 0u, sizeof(uint32_t) * 4u);

    const uint32_t visibleTileCount = meshletBuffers.visibleTileCount();
    if (visibleTileCount == 0u) {
        return;
    }

    ComputePassDescriptor passDesc = Default;
    ComputePassEncoder pass = encoder.beginComputePass(passDesc);
    pass.setPipeline(cullPipeline);
    pass.setBindGroup(0, cullBindGroup, 0, nullptr);
    const uint32_t workgroupCount =
        (visibleTileCount + kTileExpansionWorkgroupSize - 1u) / kTileExpansionWorkgroupSize;
    pass.dispatchWorkgroups(workgroupCount, 1u, 1u);
    pass.end();
    pass.release();
}

const std::string& MeshletCullingPipeline::indirectArgsBufferName() const noexcept {
    return indirectArgsBufferName_;
}
