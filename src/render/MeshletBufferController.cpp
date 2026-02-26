#include "solum_engine/render/MeshletBufferController.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>

uint32_t MeshletBufferController::computeRequiredMeshletCapacity(const StreamingMeshUpload& upload) noexcept {
    return std::max(
        upload.requiredMeshletCapacity,
        std::max(upload.totalMeshletCount + 16u, 64u)
    );
}

uint32_t MeshletBufferController::computeRequiredQuadCapacity(const StreamingMeshUpload& upload,
                                                              uint32_t requiredMeshletCapacity) noexcept {
    return std::max(
        upload.requiredQuadCapacity,
        std::max(
            upload.totalQuadCount + (1024u * MESHLET_QUAD_DATA_WORD_STRIDE),
            requiredMeshletCapacity * MESHLET_QUAD_CAPACITY * MESHLET_QUAD_DATA_WORD_STRIDE
        )
    );
}

bool MeshletBufferController::initialize(BufferManager* bufferManager) {
    bufferManager_ = bufferManager;
    meshletManager_.reset();
    meshletCapacity_ = 0;
    quadCapacity_ = 0;
    uploadedMeshRevision_ = 0;
    activeMeshletBounds_.clear();
    pendingMeshUpload_.reset();
    chunkedMeshUpload_.reset();
    meshUploadInProgress_.store(false, std::memory_order_relaxed);

    if (bufferManager_ == nullptr) {
        return false;
    }

    return uploadImmediate(StreamingMeshUpload{});
}

bool MeshletBufferController::ensureCapacity(const StreamingMeshUpload& upload, bool* buffersRecreated) {
    if (buffersRecreated != nullptr) {
        *buffersRecreated = false;
    }

    if (bufferManager_ == nullptr) {
        return false;
    }

    const uint32_t requiredMeshletCapacity = computeRequiredMeshletCapacity(upload);
    const uint32_t requiredQuadCapacity = computeRequiredQuadCapacity(upload, requiredMeshletCapacity);

    const bool requiresRecreate =
        !meshletManager_ ||
        meshletCapacity_ < requiredMeshletCapacity ||
        quadCapacity_ < requiredQuadCapacity;

    if (!requiresRecreate) {
        return true;
    }

    auto replacement = std::make_unique<MeshletManager>();
    if (!replacement->initialize(bufferManager_, requiredMeshletCapacity, requiredQuadCapacity)) {
        std::cerr << "Failed to create meshlet buffers." << std::endl;
        return false;
    }

    meshletManager_ = std::move(replacement);
    meshletCapacity_ = requiredMeshletCapacity;
    quadCapacity_ = requiredQuadCapacity;
    if (buffersRecreated != nullptr) {
        *buffersRecreated = true;
    }
    return true;
}

bool MeshletBufferController::uploadImmediate(StreamingMeshUpload&& upload) {
    if (!ensureCapacity(upload)) {
        return false;
    }
    if (!meshletManager_) {
        return false;
    }

    meshletManager_->adoptPreparedData(
        std::move(upload.metadata),
        std::move(upload.quadData),
        std::move(upload.meshletAabbsGpu)
    );
    if (!meshletManager_->upload()) {
        std::cerr << "Failed to upload meshlet buffers." << std::endl;
        return false;
    }

    activeMeshletBounds_ = std::move(upload.meshletBounds);
    uploadedMeshRevision_ = upload.meshRevision;
    return true;
}

void MeshletBufferController::queueUpload(StreamingMeshUpload&& upload) {
    pendingMeshUpload_ = std::move(upload);
}

bool MeshletBufferController::streamChunkedUploadBytes(ChunkedMeshUploadState& uploadState, size_t budgetBytes) {
    if (!meshletManager_) {
        return false;
    }

    size_t remainingBudgetBytes = budgetBytes;

    const size_t metadataTotalBytes = uploadState.upload.metadata.size() * sizeof(MeshletMetadataGPU);
    if (uploadState.metadataUploadedBytes < metadataTotalBytes && remainingBudgetBytes > 0u) {
        const size_t remainingMetadata = metadataTotalBytes - uploadState.metadataUploadedBytes;
        const size_t metadataChunkBytes = std::min(remainingBudgetBytes, remainingMetadata);
        const auto* metadataBytes = reinterpret_cast<const uint8_t*>(uploadState.upload.metadata.data());
        if (!meshletManager_->writeMetadataChunk(
                uploadState.targetBufferIndex,
                static_cast<uint64_t>(uploadState.metadataUploadedBytes),
                metadataBytes + uploadState.metadataUploadedBytes,
                metadataChunkBytes)) {
            std::cerr << "Failed to stream meshlet metadata chunk." << std::endl;
            return false;
        }
        uploadState.metadataUploadedBytes += metadataChunkBytes;
        remainingBudgetBytes -= metadataChunkBytes;
    }

    const size_t quadTotalBytes = uploadState.upload.quadData.size() * sizeof(uint32_t);
    if (uploadState.quadUploadedBytes < quadTotalBytes && remainingBudgetBytes > 0u) {
        const size_t remainingQuad = quadTotalBytes - uploadState.quadUploadedBytes;
        const size_t quadChunkBytes = std::min(remainingBudgetBytes, remainingQuad);
        const auto* quadBytes = reinterpret_cast<const uint8_t*>(uploadState.upload.quadData.data());
        if (!meshletManager_->writeQuadChunk(
                uploadState.targetBufferIndex,
                static_cast<uint64_t>(uploadState.quadUploadedBytes),
                quadBytes + uploadState.quadUploadedBytes,
                quadChunkBytes)) {
            std::cerr << "Failed to stream meshlet quad-data chunk." << std::endl;
            return false;
        }
        uploadState.quadUploadedBytes += quadChunkBytes;
        remainingBudgetBytes -= quadChunkBytes;
    }

    const size_t aabbTotalBytes = uploadState.upload.meshletAabbsGpu.size() * sizeof(MeshletAabbGPU);
    if (uploadState.aabbUploadedBytes < aabbTotalBytes && remainingBudgetBytes > 0u) {
        const size_t remainingAabbs = aabbTotalBytes - uploadState.aabbUploadedBytes;
        const size_t aabbChunkBytes = std::min(remainingBudgetBytes, remainingAabbs);
        const auto* aabbBytes = reinterpret_cast<const uint8_t*>(uploadState.upload.meshletAabbsGpu.data());
        if (!meshletManager_->writeAabbChunk(
                uploadState.targetBufferIndex,
                static_cast<uint64_t>(uploadState.aabbUploadedBytes),
                aabbBytes + uploadState.aabbUploadedBytes,
                aabbChunkBytes)) {
            std::cerr << "Failed to stream meshlet aabb chunk." << std::endl;
            return false;
        }
        uploadState.aabbUploadedBytes += aabbChunkBytes;
    }

    return true;
}

MeshletBufferController::ProcessResult MeshletBufferController::processPendingUpload() {
    ProcessResult result;

    if (!chunkedMeshUpload_.has_value() && !pendingMeshUpload_.has_value()) {
        meshUploadInProgress_.store(false, std::memory_order_relaxed);
        return result;
    }

    if (!chunkedMeshUpload_.has_value() && pendingMeshUpload_.has_value()) {
        meshUploadInProgress_.store(true, std::memory_order_relaxed);
        StreamingMeshUpload pendingUpload = std::move(*pendingMeshUpload_);
        pendingMeshUpload_.reset();

        bool buffersRecreated = false;
        if (!ensureCapacity(pendingUpload, &buffersRecreated) || !meshletManager_) {
            meshUploadInProgress_.store(false, std::memory_order_relaxed);
            return result;
        }

        result.buffersRecreated = buffersRecreated;
        if (buffersRecreated) {
            // A freshly recreated meshlet manager has no valid active dataset.
            // Apply the upload atomically to avoid rendering a transient empty frame.
            meshletManager_->adoptPreparedData(
                std::move(pendingUpload.metadata),
                std::move(pendingUpload.quadData),
                std::move(pendingUpload.meshletAabbsGpu)
            );
            if (!meshletManager_->upload()) {
                std::cerr << "Failed to upload meshlet buffers after recreation." << std::endl;
                meshUploadInProgress_.store(false, std::memory_order_relaxed);
                return result;
            }

            activeMeshletBounds_ = std::move(pendingUpload.meshletBounds);
            uploadedMeshRevision_ = pendingUpload.meshRevision;
            meshUploadInProgress_.store(false, std::memory_order_relaxed);
            result.uploadApplied = true;
            return result;
        }

        chunkedMeshUpload_ = ChunkedMeshUploadState{
            std::move(pendingUpload),
            meshletManager_->getInactiveBufferIndex(),
            0,
            0,
            0
        };
    }

    if (!chunkedMeshUpload_.has_value()) {
        return result;
    }

    ChunkedMeshUploadState& uploadState = *chunkedMeshUpload_;
    if (!streamChunkedUploadBytes(uploadState, kMeshUploadBudgetBytesPerFrame)) {
        return result;
    }

    const size_t metadataTotalBytes = uploadState.upload.metadata.size() * sizeof(MeshletMetadataGPU);
    const size_t quadTotalBytes = uploadState.upload.quadData.size() * sizeof(uint32_t);
    const size_t aabbTotalBytes = uploadState.upload.meshletAabbsGpu.size() * sizeof(MeshletAabbGPU);
    const bool uploadComplete =
        uploadState.metadataUploadedBytes >= metadataTotalBytes &&
        uploadState.quadUploadedBytes >= quadTotalBytes &&
        uploadState.aabbUploadedBytes >= aabbTotalBytes;

    if (!uploadComplete || !meshletManager_) {
        return result;
    }

    meshletManager_->activateBuffer(
        uploadState.targetBufferIndex,
        uploadState.upload.totalMeshletCount,
        static_cast<uint32_t>(uploadState.upload.quadData.size())
    );

    activeMeshletBounds_ = std::move(uploadState.upload.meshletBounds);
    uploadedMeshRevision_ = uploadState.upload.meshRevision;
    chunkedMeshUpload_.reset();
    meshUploadInProgress_.store(false, std::memory_order_relaxed);
    result.uploadApplied = true;
    return result;
}

bool MeshletBufferController::hasMeshletManager() const noexcept {
    return meshletManager_ != nullptr;
}

const char* MeshletBufferController::activeMeshDataBufferName() const noexcept {
    if (!meshletManager_) {
        return MeshletManager::meshDataBufferName(0u);
    }
    return meshletManager_->getActiveMeshDataBufferName();
}

const char* MeshletBufferController::activeMeshMetadataBufferName() const noexcept {
    if (!meshletManager_) {
        return MeshletManager::meshMetadataBufferName(0u);
    }
    return meshletManager_->getActiveMeshMetadataBufferName();
}

const char* MeshletBufferController::activeMeshAabbBufferName() const noexcept {
    if (!meshletManager_) {
        return MeshletManager::meshAabbBufferName(0u);
    }
    return meshletManager_->getActiveMeshAabbBufferName();
}

const char* MeshletBufferController::activeVisibleMeshletIndexBufferName() const noexcept {
    if (!meshletManager_) {
        return MeshletManager::visibleMeshletIndexBufferName(0u);
    }
    return meshletManager_->getActiveVisibleMeshletIndexBufferName();
}

uint32_t MeshletBufferController::meshletCount() const noexcept {
    if (!meshletManager_) {
        return 0u;
    }
    return meshletManager_->getMeshletCount();
}

uint32_t MeshletBufferController::verticesPerMeshlet() const noexcept {
    if (!meshletManager_) {
        return MESHLET_VERTEX_CAPACITY;
    }
    return meshletManager_->getVerticesPerMeshlet();
}

uint32_t MeshletBufferController::effectiveMeshletCountForPasses() const noexcept {
    const uint32_t activeCount = meshletCount();
    if (activeCount == 0u && chunkedMeshUpload_.has_value() && !activeMeshletBounds_.empty()) {
        return static_cast<uint32_t>(activeMeshletBounds_.size());
    }
    return activeCount;
}

uint64_t MeshletBufferController::uploadedMeshRevision() const noexcept {
    return uploadedMeshRevision_;
}

const std::vector<MeshletAabb>& MeshletBufferController::activeMeshletBounds() const noexcept {
    return activeMeshletBounds_;
}

bool MeshletBufferController::isUploadInProgress() const noexcept {
    return meshUploadInProgress_.load(std::memory_order_relaxed);
}

bool MeshletBufferController::hasPendingOrActiveUpload() const noexcept {
    return pendingMeshUpload_.has_value() ||
           chunkedMeshUpload_.has_value() ||
           meshUploadInProgress_.load(std::memory_order_relaxed);
}

bool MeshletBufferController::hasChunkedUploadInProgress() const noexcept {
    return chunkedMeshUpload_.has_value();
}

void MeshletBufferController::resetPendingUploads() {
    pendingMeshUpload_.reset();
    chunkedMeshUpload_.reset();
    meshUploadInProgress_.store(false, std::memory_order_relaxed);
}
