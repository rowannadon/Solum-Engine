#include "solum_engine/render/MeshletManager.h"

#include <algorithm>
#include <cstddef>
#include <iostream>

const char* MeshletManager::meshDataBufferName(uint32_t bufferIndex) noexcept {
    return (bufferIndex % kBufferSetCount == 0u) ? kMeshDataBufferName0 : kMeshDataBufferName1;
}

const char* MeshletManager::meshMetadataBufferName(uint32_t bufferIndex) noexcept {
    return (bufferIndex % kBufferSetCount == 0u) ? kMeshMetadataBufferName0 : kMeshMetadataBufferName1;
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
    activeBufferIndex_ = 0;
    activeMeshletCount_ = 0;
    activeQuadWordCount_ = 0;

    metadataCpu.reserve(meshletCapacity);
    quadDataCpu.reserve(quadCapacity);

    BufferDescriptor metadataDesc = Default;
    metadataDesc.size = static_cast<uint64_t>(meshletCapacity) * sizeof(MeshletMetadataGPU);
    metadataDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    metadataDesc.mappedAtCreation = false;

    BufferDescriptor meshDataDesc = Default;
    meshDataDesc.size = static_cast<uint64_t>(quadCapacity) * sizeof(uint32_t);
    meshDataDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
    meshDataDesc.mappedAtCreation = false;

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
    }

    return true;
}

void MeshletManager::clear() {
    metadataCpu.clear();
    quadDataCpu.clear();
    activeMeshletCount_ = 0;
    activeQuadWordCount_ = 0;
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

void MeshletManager::activateBuffer(uint32_t bufferIndex, uint32_t meshletCount, uint32_t quadWordCount) {
    activeBufferIndex_ = bufferIndex % kBufferSetCount;
    activeMeshletCount_ = meshletCount;
    activeQuadWordCount_ = quadWordCount;
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

uint32_t MeshletManager::getMeshletCount() const {
    return activeMeshletCount_;
}

uint32_t MeshletManager::getQuadCount() const {
    return activeQuadWordCount_ / MESHLET_QUAD_DATA_WORD_STRIDE;
}

uint32_t MeshletManager::getVerticesPerMeshlet() const {
    return MESHLET_VERTEX_CAPACITY;
}
