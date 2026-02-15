#include "solum_engine/render/MeshletManager.h"

#include <cstddef>
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
    meshDataDesc.size = (static_cast<uint64_t>(quadCapacity) * sizeof(uint16_t) + 3ull) & ~3ull;
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
    handle.firstQuad = static_cast<uint32_t>(quadDataCpu.size());

    for (const Meshlet& meshlet : meshlets) {
        if (meshlet.quadCount == 0) {
            continue;
        }

        if (metadataCpu.size() >= meshletCapacity || quadDataCpu.size() + meshlet.quadCount > quadCapacity) {
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

        metadataCpu.push_back(metadata);

        quadDataCpu.insert(
            quadDataCpu.end(),
            meshlet.packedQuadLocalOffsets.begin(),
            meshlet.packedQuadLocalOffsets.begin() + static_cast<std::ptrdiff_t>(meshlet.quadCount)
        );

        handle.meshletCount += 1;
        handle.quadCount += meshlet.quadCount;
    }

    return handle;
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
        bufferManager->writeBuffer(
            kMeshDataBufferName,
            0,
            quadDataCpu.data(),
            quadDataCpu.size() * sizeof(uint16_t)
        );
    }

    return true;
}

uint32_t MeshletManager::getMeshletCount() const {
    return static_cast<uint32_t>(metadataCpu.size());
}

uint32_t MeshletManager::getQuadCount() const {
    return static_cast<uint32_t>(quadDataCpu.size());
}

uint32_t MeshletManager::getVerticesPerMeshlet() const {
    return MESHLET_VERTEX_CAPACITY;
}
