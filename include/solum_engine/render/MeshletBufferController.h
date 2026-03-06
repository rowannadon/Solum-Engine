#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "solum_engine/render/BufferManager.h"
#include "solum_engine/render/MeshletTypes.h"
#include "solum_engine/voxel/StreamingUpload.h"

struct ActiveMeshletRangeGPU {
    uint32_t meshletOffset = 0u;
    uint32_t meshletCount = 0u;
    uint32_t prefixEnd = 0u;
    uint32_t pad = 0u;
};

enum class MeshletGeometryVariant {
    Culled,
    DoubleSided
};

class MeshletBufferController {
public:
    struct Config {
        std::string namePrefix;
        MeshletGeometryVariant geometryVariant = MeshletGeometryVariant::Culled;
    };

    MeshletBufferController();
    explicit MeshletBufferController(Config config);

    struct ActiveBindings {
        const char* meshDataBufferName = nullptr;
        const char* meshMetadataBufferName = nullptr;
        const char* meshAabbBufferName = nullptr;
        const char* visibleMeshletIndexBufferName = nullptr;
        const char* activeMeshletRangeBufferName = nullptr;
        const char* activeMeshletRangeParamsBufferName = nullptr;
        uint32_t meshletCount = 0u;
        uint32_t activeRangeCount = 0u;
        uint32_t verticesPerMeshlet = MESHLET_VERTEX_CAPACITY;
    };

    struct ApplyResult {
        bool buffersRecreated = false;
        bool deltaApplied = false;
    };

    bool initialize(BufferManager* bufferManager);

    ApplyResult applyDelta(const MeshStreamingDelta& delta);
    bool buildActiveRanges();

    bool hasMeshletManager() const noexcept;
    const char* activeMeshDataBufferName() const noexcept;
    const char* activeMeshMetadataBufferName() const noexcept;
    const char* activeMeshAabbBufferName() const noexcept;
    const char* activeVisibleMeshletIndexBufferName() const noexcept;
    const char* activeMeshletRangeBufferName() const noexcept;
    const char* activeMeshletRangeParamsBufferName() const noexcept;

    uint32_t meshletCount() const noexcept;
    uint32_t activeSelectionMeshletCount() const noexcept;
    uint32_t activeRangeCount() const noexcept;
    uint32_t verticesPerMeshlet() const noexcept;

    uint64_t uploadedMeshRevision() const noexcept;
    const std::vector<MeshletAabb>& activeMeshletBounds() const noexcept;

    ActiveBindings activeBindings() const noexcept;

private:
    struct PackedTileLodData {
        std::vector<MeshletMetadataGPU> metadata;
        std::vector<uint32_t> quadData;
        std::vector<MeshletAabbGPU> aabbGpu;
        std::vector<MeshletAabb> bounds;
    };

    struct AllocationRecord {
        MeshTileLodKey key{};
        PackedTileLodData packed;
        uint32_t meshletOffset = 0u;
        uint32_t quadOffset = 0u;
    };

    struct FreeRange {
        uint32_t offset = 0u;
        uint32_t length = 0u;
    };

    struct RangeParamsGPU {
        uint32_t rangeCount = 0u;
        uint32_t totalActiveMeshlets = 0u;
        uint32_t pad0 = 0u;
        uint32_t pad1 = 0u;
    };

    static constexpr uint32_t kInitialMeshletCapacity = 64u;
    static constexpr uint32_t kInitialQuadWordCapacity =
        kInitialMeshletCapacity * MESHLET_QUAD_CAPACITY * MESHLET_QUAD_DATA_WORD_STRIDE;
    static constexpr uint32_t kInitialRangeCapacity = 256u;

    static MeshletAabb computeMeshletAabb(const Meshlet& meshlet);
    static MeshletAabbGPU toGpuAabb(const MeshletAabb& aabb);
    static PackedTileLodData packTileLodMeshlets(const std::vector<Meshlet>& meshlets);

    bool ensureBuffers(uint32_t requiredMeshlets, uint32_t requiredQuadWords, uint32_t requiredRanges, bool* recreated);
    bool recreateBuffers(uint32_t meshletCapacity, uint32_t quadWordCapacity, uint32_t rangeCapacity);
    bool repackExistingAllocations();

    static bool allocateRange(std::vector<FreeRange>& freeList, uint32_t length, uint32_t& outOffset);
    static void freeRange(std::vector<FreeRange>& freeList, uint32_t offset, uint32_t length);

    bool writeAllocation(const AllocationRecord& record);
    void releaseAllocation(const MeshTileLodKey& key);

    const std::vector<Meshlet>& selectMeshletsForVariant(const MeshTileLodUpload& upload) const;
    static std::string prefixedName(const std::string& prefix, const char* baseName);

    BufferManager* bufferManager_ = nullptr;
    std::string namePrefix_;
    MeshletGeometryVariant geometryVariant_ = MeshletGeometryVariant::Culled;
    std::string meshDataBufferName_;
    std::string meshMetadataBufferName_;
    std::string meshAabbBufferName_;
    std::string visibleMeshletIndexBufferName_;
    std::string activeMeshletRangeBufferName_;
    std::string activeMeshletRangeParamsBufferName_;

    uint32_t meshletCapacity_ = 0u;
    uint32_t quadWordCapacity_ = 0u;
    uint32_t rangeCapacity_ = 0u;

    std::vector<FreeRange> meshletFreeRanges_;
    std::vector<FreeRange> quadFreeRanges_;

    std::unordered_map<MeshTileLodKey, AllocationRecord> allocations_;
    std::unordered_map<MeshTileCoord, int8_t> tileSelection_;

    std::vector<ActiveMeshletRangeGPU> activeRanges_;
    std::vector<MeshletAabb> activeMeshletBounds_;
    uint32_t activeSelectionMeshletCount_ = 0u;

    uint64_t uploadedMeshRevision_ = 0u;
};
