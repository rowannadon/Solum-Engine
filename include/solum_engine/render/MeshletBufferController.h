#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "solum_engine/render/BufferManager.h"
#include "solum_engine/render/MeshletManager.h"
#include "solum_engine/voxel/StreamingUpload.h"

class MeshletBufferController {
public:
    struct ProcessResult {
        bool buffersRecreated = false;
        bool uploadApplied = false;
    };

    bool initialize(BufferManager* bufferManager);
    bool uploadImmediate(StreamingMeshUpload&& upload);
    void queueUpload(StreamingMeshUpload&& upload);
    ProcessResult processPendingUpload();

    bool hasMeshletManager() const noexcept;
    const char* activeMeshDataBufferName() const noexcept;
    const char* activeMeshMetadataBufferName() const noexcept;
    const char* activeMeshAabbBufferName() const noexcept;
    const char* activeVisibleMeshletIndexBufferName() const noexcept;

    uint32_t meshletCount() const noexcept;
    uint32_t verticesPerMeshlet() const noexcept;
    uint32_t effectiveMeshletCountForPasses() const noexcept;

    uint64_t uploadedMeshRevision() const noexcept;
    const std::vector<MeshletAabb>& activeMeshletBounds() const noexcept;

    bool isUploadInProgress() const noexcept;
    bool hasPendingOrActiveUpload() const noexcept;
    bool hasChunkedUploadInProgress() const noexcept;
    void resetPendingUploads();

private:
    struct ChunkedMeshUploadState {
        StreamingMeshUpload upload;
        uint32_t targetBufferIndex = 0;
        size_t metadataUploadedBytes = 0;
        size_t quadUploadedBytes = 0;
        size_t aabbUploadedBytes = 0;
    };

    static uint32_t computeRequiredMeshletCapacity(const StreamingMeshUpload& upload) noexcept;
    static uint32_t computeRequiredQuadCapacity(const StreamingMeshUpload& upload,
                                                uint32_t requiredMeshletCapacity) noexcept;
    bool ensureCapacity(const StreamingMeshUpload& upload, bool* buffersRecreated = nullptr);
    bool streamChunkedUploadBytes(ChunkedMeshUploadState& uploadState, size_t budgetBytes);

    BufferManager* bufferManager_ = nullptr;
    std::unique_ptr<MeshletManager> meshletManager_;
    uint32_t meshletCapacity_ = 0;
    uint32_t quadCapacity_ = 0;
    uint64_t uploadedMeshRevision_ = 0;
    std::vector<MeshletAabb> activeMeshletBounds_;

    std::optional<StreamingMeshUpload> pendingMeshUpload_;
    std::optional<ChunkedMeshUploadState> chunkedMeshUpload_;
    std::atomic<bool> meshUploadInProgress_{false};

    static constexpr size_t kMeshUploadBudgetBytesPerFrame = 512u * 1024u;
};
