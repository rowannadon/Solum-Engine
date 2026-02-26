#pragma once

#include "solum_engine/render/BufferManager.h"
#include "solum_engine/render/MeshletTypes.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class MeshletManager {
public:
    static constexpr uint32_t kBufferSetCount = 2;
    static constexpr const char* kMeshDataBufferName0 = "meshlet_data_buffer_0";
    static constexpr const char* kMeshDataBufferName1 = "meshlet_data_buffer_1";
    static constexpr const char* kMeshMetadataBufferName0 = "meshlet_metadata_buffer_0";
    static constexpr const char* kMeshMetadataBufferName1 = "meshlet_metadata_buffer_1";
    static constexpr const char* kMeshAabbBufferName0 = "meshlet_aabb_buffer_0";
    static constexpr const char* kMeshAabbBufferName1 = "meshlet_aabb_buffer_1";
    static constexpr const char* kVisibleMeshletIndexBufferName0 = "visible_meshlet_indices_buffer_0";
    static constexpr const char* kVisibleMeshletIndexBufferName1 = "visible_meshlet_indices_buffer_1";

    static const char* meshDataBufferName(uint32_t bufferIndex) noexcept;
    static const char* meshMetadataBufferName(uint32_t bufferIndex) noexcept;
    static const char* meshAabbBufferName(uint32_t bufferIndex) noexcept;
    static const char* visibleMeshletIndexBufferName(uint32_t bufferIndex) noexcept;

    bool initialize(BufferManager* bufferManager, uint32_t maxMeshlets, uint32_t maxQuads);

    void clear();

    void adoptPreparedData(std::vector<MeshletMetadataGPU>&& metadata,
                           std::vector<uint32_t>&& quadData,
                           std::vector<MeshletAabbGPU>&& aabbs);

    bool upload();
    bool writeMetadataChunk(uint32_t bufferIndex, uint64_t byteOffset, const void* data, size_t sizeBytes);
    bool writeQuadChunk(uint32_t bufferIndex, uint64_t byteOffset, const void* data, size_t sizeBytes);
    bool writeAabbChunk(uint32_t bufferIndex, uint64_t byteOffset, const void* data, size_t sizeBytes);
    void activateBuffer(uint32_t bufferIndex, uint32_t meshletCount, uint32_t quadWordCount);

    uint32_t getActiveBufferIndex() const noexcept;
    uint32_t getInactiveBufferIndex() const noexcept;
    const char* getActiveMeshDataBufferName() const noexcept;
    const char* getActiveMeshMetadataBufferName() const noexcept;
    const char* getActiveMeshAabbBufferName() const noexcept;
    const char* getActiveVisibleMeshletIndexBufferName() const noexcept;

    bool updateVisibleMeshletIndices(const uint32_t* indices, uint32_t visibleCount);

    uint32_t getMeshletCount() const;
    uint32_t getVisibleMeshletCount() const;
    uint32_t getQuadCount() const;
    uint32_t getVerticesPerMeshlet() const;

private:
    BufferManager* bufferManager = nullptr;
    uint32_t meshletCapacity = 0;
    uint32_t quadCapacity = 0;
    uint32_t activeBufferIndex_ = 0;
    uint32_t activeMeshletCount_ = 0;
    uint32_t activeVisibleMeshletCount_ = 0;
    uint32_t activeQuadWordCount_ = 0;

    std::vector<MeshletMetadataGPU> metadataCpu;
    std::vector<uint32_t> quadDataCpu;
    std::vector<MeshletAabbGPU> aabbCpu;
    std::vector<uint32_t> sequentialVisibleIndicesCpu;
};
