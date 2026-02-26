#include "solum_engine/render/MeshletManager.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <numeric>

const char* MeshletManager::meshDataBufferName(uint32_t bufferIndex) noexcept {
    return (bufferIndex % kBufferSetCount == 0u) ? kMeshDataBufferName0 : kMeshDataBufferName1;
}

const char* MeshletManager::meshMetadataBufferName(uint32_t bufferIndex) noexcept {
    return (bufferIndex % kBufferSetCount == 0u) ? kMeshMetadataBufferName0 : kMeshMetadataBufferName1;
}

const char* MeshletManager::meshAabbBufferName(uint32_t bufferIndex) noexcept {
    return (bufferIndex % kBufferSetCount == 0u) ? kMeshAabbBufferName0 : kMeshAabbBufferName1;
}

const char* MeshletManager::visibleMeshletIndexBufferName(uint32_t bufferIndex) noexcept {
    return (bufferIndex % kBufferSetCount == 0u) ? kVisibleMeshletIndexBufferName0 : kVisibleMeshletIndexBufferName1;
}

bool MeshletManager::initialize(BufferManager* manager, uint32_t maxMeshlets, uint32_t maxQuads) {
    if (manager == nullptr || maxMeshlets == 0 || maxQuads == 0) {
        return false;
    }

    bufferManager = manager;
    meshletCapacity = maxMeshlets;
    quadCapacity = maxQuads;

    metadataCpu.clear();
    quadDataCpu.clear();
    aabbCpu.clear();
    activeBufferIndex_ = 0;
    activeMeshletCount_ = 0;
    activeVisibleMeshletCount_ = 0;
    activeQuadWordCount_ = 0;

    metadataCpu.reserve(meshletCapacity);
    quadDataCpu.reserve(quadCapacity);
    aabbCpu.reserve(meshletCapacity);
    sequentialVisibleIndicesCpu.reserve(meshletCapacity);

    BufferDescriptor metadataDesc = Default;
    metadataDesc.size = static_cast<uint64_t>(meshletCapacity) * sizeof(MeshletMetadataGPU);
    metadataDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    metadataDesc.mappedAtCreation = false;

    BufferDescriptor meshDataDesc = Default;
    meshDataDesc.size = static_cast<uint64_t>(quadCapacity) * sizeof(uint32_t);
    meshDataDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    meshDataDesc.mappedAtCreation = false;

    BufferDescriptor visibleIndicesDesc = Default;
    visibleIndicesDesc.size = static_cast<uint64_t>(meshletCapacity) * sizeof(uint32_t);
    visibleIndicesDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    visibleIndicesDesc.mappedAtCreation = false;

    BufferDescriptor aabbDesc = Default;
    aabbDesc.size = static_cast<uint64_t>(meshletCapacity) * sizeof(MeshletAabbGPU);
    aabbDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    aabbDesc.mappedAtCreation = false;

    for (uint32_t i = 0; i < kBufferSetCount; ++i) {
        metadataDesc.label = StringView("meshlet metadata buffer");
        Buffer metadataBuffer = bufferManager->createBuffer(meshMetadataBufferName(i), metadataDesc);
        if (!metadataBuffer) {
            return false;
        }

        meshDataDesc.label = StringView("meshlet data buffer");
        Buffer meshDataBuffer = bufferManager->createBuffer(meshDataBufferName(i), meshDataDesc);
        if (!meshDataBuffer) {
            return false;
        }

        visibleIndicesDesc.label = StringView("visible meshlet indices buffer");
        Buffer visibleIndicesBuffer = bufferManager->createBuffer(visibleMeshletIndexBufferName(i), visibleIndicesDesc);
        if (!visibleIndicesBuffer) {
            return false;
        }

        aabbDesc.label = StringView("meshlet aabb buffer");
        Buffer aabbBuffer = bufferManager->createBuffer(meshAabbBufferName(i), aabbDesc);
        if (!aabbBuffer) {
            return false;
        }
    }

    return true;
}

void MeshletManager::clear() {
    metadataCpu.clear();
    quadDataCpu.clear();
    aabbCpu.clear();
    activeMeshletCount_ = 0;
    activeVisibleMeshletCount_ = 0;
    activeQuadWordCount_ = 0;
}

void MeshletManager::adoptPreparedData(std::vector<MeshletMetadataGPU>&& metadata,
                                       std::vector<uint32_t>&& quadData,
                                       std::vector<MeshletAabbGPU>&& aabbs) {
    metadataCpu = std::move(metadata);
    quadDataCpu = std::move(quadData);
    aabbCpu = std::move(aabbs);
}

bool MeshletManager::upload() {
    if (bufferManager == nullptr) {
        return false;
    }

    const uint32_t targetBufferIndex = getInactiveBufferIndex();
    bool wroteAny = false;

    if (!metadataCpu.empty()) {
        wroteAny = true;
        if (!writeMetadataChunk(
                targetBufferIndex,
                0,
                metadataCpu.data(),
                metadataCpu.size() * sizeof(MeshletMetadataGPU))) {
            return false;
        }
    }

    if (!quadDataCpu.empty()) {
        wroteAny = true;
        if (!writeQuadChunk(
                targetBufferIndex,
                0,
                quadDataCpu.data(),
                quadDataCpu.size() * sizeof(uint32_t))) {
            return false;
        }
    }

    if (!aabbCpu.empty()) {
        wroteAny = true;
        if (!writeAabbChunk(
                targetBufferIndex,
                0,
                aabbCpu.data(),
                aabbCpu.size() * sizeof(MeshletAabbGPU))) {
            return false;
        }
    }

    if (wroteAny) {
        activateBuffer(
            targetBufferIndex,
            static_cast<uint32_t>(metadataCpu.size()),
            static_cast<uint32_t>(quadDataCpu.size())
        );
    }

    return true;
}

bool MeshletManager::writeMetadataChunk(uint32_t bufferIndex,
                                        uint64_t byteOffset,
                                        const void* data,
                                        size_t sizeBytes) {
    if (bufferManager == nullptr || data == nullptr || sizeBytes == 0) {
        return false;
    }

    bufferManager->writeBuffer(meshMetadataBufferName(bufferIndex), byteOffset, data, sizeBytes);
    return true;
}

bool MeshletManager::writeQuadChunk(uint32_t bufferIndex,
                                    uint64_t byteOffset,
                                    const void* data,
                                    size_t sizeBytes) {
    if (bufferManager == nullptr || data == nullptr || sizeBytes == 0) {
        return false;
    }

    bufferManager->writeBuffer(meshDataBufferName(bufferIndex), byteOffset, data, sizeBytes);
    return true;
}

bool MeshletManager::writeAabbChunk(uint32_t bufferIndex,
                                    uint64_t byteOffset,
                                    const void* data,
                                    size_t sizeBytes) {
    if (bufferManager == nullptr || data == nullptr || sizeBytes == 0) {
        return false;
    }

    bufferManager->writeBuffer(meshAabbBufferName(bufferIndex), byteOffset, data, sizeBytes);
    return true;
}

void MeshletManager::activateBuffer(uint32_t bufferIndex, uint32_t meshletCount, uint32_t quadWordCount) {
    activeBufferIndex_ = bufferIndex % kBufferSetCount;
    activeMeshletCount_ = meshletCount;
    activeQuadWordCount_ = quadWordCount;
    activeVisibleMeshletCount_ = meshletCount;

    if (bufferManager == nullptr || meshletCount == 0u) {
        return;
    }

    if (sequentialVisibleIndicesCpu.size() < meshletCount) {
        sequentialVisibleIndicesCpu.resize(meshletCount);
        std::iota(sequentialVisibleIndicesCpu.begin(), sequentialVisibleIndicesCpu.end(), 0u);
    } else if (!sequentialVisibleIndicesCpu.empty()) {
        std::iota(sequentialVisibleIndicesCpu.begin(), sequentialVisibleIndicesCpu.begin() + meshletCount, 0u);
    }

    bufferManager->writeBuffer(
        visibleMeshletIndexBufferName(activeBufferIndex_),
        0u,
        sequentialVisibleIndicesCpu.data(),
        static_cast<size_t>(meshletCount) * sizeof(uint32_t)
    );
}

uint32_t MeshletManager::getActiveBufferIndex() const noexcept {
    return activeBufferIndex_;
}

uint32_t MeshletManager::getInactiveBufferIndex() const noexcept {
    return (activeBufferIndex_ + 1u) % kBufferSetCount;
}

const char* MeshletManager::getActiveMeshDataBufferName() const noexcept {
    return meshDataBufferName(activeBufferIndex_);
}

const char* MeshletManager::getActiveMeshMetadataBufferName() const noexcept {
    return meshMetadataBufferName(activeBufferIndex_);
}

const char* MeshletManager::getActiveMeshAabbBufferName() const noexcept {
    return meshAabbBufferName(activeBufferIndex_);
}

const char* MeshletManager::getActiveVisibleMeshletIndexBufferName() const noexcept {
    return visibleMeshletIndexBufferName(activeBufferIndex_);
}

bool MeshletManager::updateVisibleMeshletIndices(const uint32_t* indices, uint32_t visibleCount) {
    if (bufferManager == nullptr) {
        return false;
    }

    const uint32_t clampedVisibleCount = std::min(visibleCount, activeMeshletCount_);
    if (clampedVisibleCount > 0u && indices == nullptr) {
        return false;
    }

    if (clampedVisibleCount > 0u) {
        bufferManager->writeBuffer(
            visibleMeshletIndexBufferName(activeBufferIndex_),
            0u,
            indices,
            static_cast<size_t>(clampedVisibleCount) * sizeof(uint32_t)
        );
    }

    activeVisibleMeshletCount_ = clampedVisibleCount;
    return true;
}

uint32_t MeshletManager::getMeshletCount() const {
    return activeMeshletCount_;
}

uint32_t MeshletManager::getVisibleMeshletCount() const {
    return activeVisibleMeshletCount_;
}

uint32_t MeshletManager::getQuadCount() const {
    return activeQuadWordCount_ / MESHLET_QUAD_DATA_WORD_STRIDE;
}

uint32_t MeshletManager::getVerticesPerMeshlet() const {
    return MESHLET_VERTEX_CAPACITY;
}
