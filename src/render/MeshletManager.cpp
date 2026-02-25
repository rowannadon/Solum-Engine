#include "solum_engine/render/MeshletManager.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iostream>

bool MeshletManager::initialize(BufferManager* manager, uint32_t maxMeshlets, uint32_t maxQuads) {
    if (manager == nullptr || maxMeshlets == 0 || maxQuads == 0) {
        return false;
    }

    bufferManager = manager;
    meshletCapacity = maxMeshlets;
    quadCapacity = maxQuads;

    metadataCpu.clear();
    quadDataCpu.clear();

    metadataCpu.reserve(meshletCapacity);
    quadDataCpu.reserve(quadCapacity);

    BufferDescriptor metadataDesc = Default;
    metadataDesc.label = StringView("meshlet metadata buffer");
    metadataDesc.size = static_cast<uint64_t>(meshletCapacity) * sizeof(MeshletMetadataGPU);
    metadataDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    metadataDesc.mappedAtCreation = false;

    Buffer metadataBuffer = bufferManager->createBuffer(kMeshMetadataBufferName, metadataDesc);
    if (!metadataBuffer) {
        return false;
    }

    BufferDescriptor meshDataDesc = Default;
    meshDataDesc.label = StringView("meshlet data buffer");
    meshDataDesc.size = static_cast<uint64_t>(quadCapacity) * sizeof(uint32_t);
    meshDataDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    meshDataDesc.mappedAtCreation = false;

    Buffer meshDataBuffer = bufferManager->createBuffer(kMeshDataBufferName, meshDataDesc);
    if (!meshDataBuffer) {
        return false;
    }

    return true;
}

void MeshletManager::clear() {
    metadataCpu.clear();
    quadDataCpu.clear();
}

MeshletGroupHandle MeshletManager::registerMeshletGroup(const std::vector<Meshlet>& meshlets) {
    MeshletGroupHandle handle;
    handle.firstMeshlet = static_cast<uint32_t>(metadataCpu.size());
    handle.firstQuad = static_cast<uint32_t>(quadDataCpu.size() / MESHLET_QUAD_DATA_WORD_STRIDE);

    for (const Meshlet& meshlet : meshlets) {
        if (meshlet.quadCount == 0) {
            continue;
        }

        const uint32_t quadWordCount = meshlet.quadCount * MESHLET_QUAD_DATA_WORD_STRIDE;
        if (metadataCpu.size() >= meshletCapacity || quadDataCpu.size() + quadWordCount > quadCapacity) {
            std::cerr << "MeshletManager capacity exceeded while registering meshlet group." << std::endl;
            break;
        }

        MeshletMetadataGPU metadata{};
        metadata.originX = meshlet.origin.x;
        metadata.originY = meshlet.origin.y;
        metadata.originZ = meshlet.origin.z;
        metadata.quadCount = meshlet.quadCount;
        metadata.faceDirection = meshlet.faceDirection;
        metadata.dataOffset = static_cast<uint32_t>(quadDataCpu.size());
        metadata.voxelScale = std::max(meshlet.voxelScale, 1u);

        metadataCpu.push_back(metadata);

        for (uint32_t i = 0; i < meshlet.quadCount; ++i) {
            quadDataCpu.push_back(packMeshletQuadData(
                meshlet.packedQuadLocalOffsets[i],
                meshlet.quadMaterialIds[i]
            ));
            quadDataCpu.push_back(static_cast<uint32_t>(meshlet.quadAoData[i]));
        }

        handle.meshletCount += 1;
        handle.quadCount += meshlet.quadCount;
    }

    return handle;
}

void MeshletManager::adoptPreparedData(std::vector<MeshletMetadataGPU>&& metadata,
                                       std::vector<uint32_t>&& quadData) {
    metadataCpu = std::move(metadata);
    quadDataCpu = std::move(quadData);
}

bool MeshletManager::upload() {
    if (bufferManager == nullptr) {
        return false;
    }

    if (!metadataCpu.empty()) {
        bufferManager->writeBuffer(
            kMeshMetadataBufferName,
            0,
            metadataCpu.data(),
            metadataCpu.size() * sizeof(MeshletMetadataGPU)
        );
    }

    if (!quadDataCpu.empty()) {
        const size_t dataBytes = quadDataCpu.size() * sizeof(uint32_t);
        bufferManager->writeBuffer(
            kMeshDataBufferName,
            0,
            quadDataCpu.data(),
            dataBytes
        );
    }

    return true;
}

uint32_t MeshletManager::getMeshletCount() const {
    return static_cast<uint32_t>(metadataCpu.size());
}

uint32_t MeshletManager::getQuadCount() const {
    return static_cast<uint32_t>(quadDataCpu.size() / MESHLET_QUAD_DATA_WORD_STRIDE);
}

uint32_t MeshletManager::getVerticesPerMeshlet() const {
    return MESHLET_VERTEX_CAPACITY;
}
