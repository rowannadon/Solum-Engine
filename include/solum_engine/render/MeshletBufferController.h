#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "solum_engine/render/BufferManager.h"
#include "solum_engine/render/MeshletPacking.h"
#include "solum_engine/render/MeshletRangeAllocator.h"
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
    static constexpr uint32_t kMaxLods = 8u;
    static constexpr uint32_t kInitialTileSlotCapacity = 4096u;
    static constexpr uint32_t kInitialMeshletCapacity = 4096u;
    static constexpr uint32_t kInitialQuadWordCapacity =
        kInitialMeshletCapacity * MESHLET_QUAD_CAPACITY * MESHLET_QUAD_DATA_WORD_STRIDE;
    static constexpr uint32_t kInitialRangeCapacity = 2048u;

    // Per-LOD allocation info stored in a tile slot.
    struct LodAllocation {
        uint32_t meshletOffset = 0u;
        uint32_t meshletCount = 0u;           // Actual meshlet count (for rendering)
        uint32_t reservedMeshletCount = 0u;   // Reserved space (for allocation/free)
        uint32_t quadDataOffset = 0u;
        uint32_t quadDataWordCount = 0u;      // Actual quad data size
        uint32_t reservedQuadWordCount = 0u;  // Reserved space
        std::shared_ptr<const PackedMeshletData> packed;
        bool resident = false;
    };

    // Stable slot for one MeshTileSliceCoord. Each slot can hold multiple LODs
    // simultaneously, with one LOD selected as active for rendering.
    struct TileSlot {
        MeshTileSliceCoord coord{};
        std::array<LodAllocation, kMaxLods> lods{};
        int8_t activeLod = -1;
        bool inActiveList = false;
    };

    struct RangeParamsGPU {
        uint32_t rangeCount = 0u;
        uint32_t totalActiveMeshlets = 0u;
        uint32_t pad0 = 0u;
        uint32_t pad1 = 0u;
    };

    // Buffer management
    bool ensureBuffers(uint32_t requiredMeshlets, uint32_t requiredQuadWords,
                       uint32_t requiredRanges, bool* recreated);
    bool recreateBuffers(uint32_t meshletCapacity, uint32_t quadWordCapacity,
                         uint32_t rangeCapacity);
    bool repackExistingAllocations();

    // GPU writes
    bool writeLodAllocation(uint32_t meshletOffset, uint32_t quadDataOffset,
                            const PackedMeshletData& packed);

    // Tile slot management
    uint32_t acquireTileSlot(const MeshTileSliceCoord& coord);
    void releaseTileSlot(uint32_t slotIndex);
    void ensureTileSlotCapacity();

    // Active list management (O(1) show/hide)
    void showTileSlot(uint32_t slotIndex);
    void hideTileSlot(uint32_t slotIndex);
    void switchTileLod(uint32_t slotIndex, int8_t newLod);

    // Helpers
    const std::shared_ptr<const PackedMeshletData>& selectPackedForVariant(
        const MeshTileLodUpload& upload) const;
    static std::string prefixedName(const std::string& prefix, const char* baseName);

    BufferManager* bufferManager_ = nullptr;
    std::string namePrefix_;
    MeshletGeometryVariant geometryVariant_ = MeshletGeometryVariant::Culled;

    // GPU buffer names
    std::string meshDataBufferName_;
    std::string meshMetadataBufferName_;
    std::string meshAabbBufferName_;
    std::string visibleMeshletIndexBufferName_;
    std::string activeMeshletRangeBufferName_;
    std::string activeMeshletRangeParamsBufferName_;

    // Buffer capacities
    uint32_t meshletCapacity_ = 0u;
    uint32_t quadWordCapacity_ = 0u;
    uint32_t rangeCapacity_ = 0u;

    // O(log n) allocators (replacing O(n) free-list vectors)
    MeshletRangeAllocator meshletAllocator_;
    MeshletRangeAllocator quadDataAllocator_;

    // Tile slot table
    std::vector<TileSlot> tileSlots_;
    std::vector<uint32_t> freeSlotIndices_;
    std::unordered_map<MeshTileSliceCoord, uint32_t> coordToSlot_;

    // Active rendering state
    std::vector<uint32_t> activeSlotIndices_;
    std::unordered_map<uint32_t, uint32_t> slotToActivePos_;
    bool activeRangesDirty_ = true;
    uint32_t totalActiveMeshlets_ = 0u;

    // GPU active ranges (rebuilt from activeSlotIndices_ when dirty)
    std::vector<ActiveMeshletRangeGPU> activeRanges_;
    mutable std::vector<MeshletAabb> activeMeshletBoundsCache_;
    mutable bool activeMeshletBoundsDirty_ = true;

    uint64_t uploadedMeshRevision_ = 0u;

#ifdef __APPLE__
    static constexpr uint32_t kDefaultMaxActiveMeshlets = 300'000u;
#else
    static constexpr uint32_t kDefaultMaxActiveMeshlets = UINT32_MAX;
#endif
    uint32_t maxActiveMeshlets_ = kDefaultMaxActiveMeshlets;
};
