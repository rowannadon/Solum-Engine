#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>

#include "solum_engine/render/BufferManager.h"
#include "solum_engine/render/MeshletPacking.h"
#include "solum_engine/render/MeshletTypes.h"
#include "solum_engine/voxel/StreamingUpload.h"

enum class MeshletGeometryVariant {
    Culled,
    DoubleSided
};

struct ResidentTileLodHandle {
    static constexpr uint32_t kInvalidSlot = UINT32_MAX;

    uint32_t slot = kInvalidSlot;
    uint32_t generation = 0u;

    bool valid() const noexcept {
        return slot != kInvalidSlot;
    }
};

struct ResidentTileLodGPU {
    uint32_t meshletStart = 0u;
    uint32_t meshletCount = 0u;
    uint32_t quadWordStart = 0u;
    uint32_t flags = 0u;
    glm::vec4 minCorner{0.0f};
    glm::vec4 maxCorner{0.0f};
};

struct TileSlotGPU {
    uint32_t selectedResidentSlot = ResidentTileLodHandle::kInvalidSlot;
    uint32_t visible = 0u;
    uint32_t flags = 0u;
    uint32_t pad0 = 0u;
    glm::vec4 minCorner{0.0f};
    glm::vec4 maxCorner{0.0f};
};

struct TileSceneParamsGPU {
    uint32_t visibleTileCount = 0u;
    uint32_t tileSlotCount = 0u;
    uint32_t residentTileCount = 0u;
    uint32_t totalVisibleMeshlets = 0u;
};

using VisibleTileId = uint32_t;

static_assert(sizeof(ResidentTileLodGPU) == 48, "ResidentTileLodGPU must stay tightly packed for WGSL");
static_assert(sizeof(TileSlotGPU) == 48, "TileSlotGPU must stay tightly packed for WGSL");
static_assert(sizeof(TileSceneParamsGPU) == 16, "TileSceneParamsGPU must stay tightly packed for WGSL");

class MeshletBufferController {
public:
    struct Config {
        std::string namePrefix;
        MeshletGeometryVariant geometryVariant = MeshletGeometryVariant::Culled;
        uint32_t meshletCapacity = 131072u;
        uint32_t quadWordCapacity = 16u * 1024u * 1024u;
        uint32_t residentTileCapacity = 131072u;
        uint32_t tileSlotCapacity = 65536u;
        uint32_t meshletPageSize = 64u;
        uint32_t quadWordPageSize = 2048u;
        uint32_t maxResidentLodsPerTile = 16u;
    };

    struct ActiveBindings {
        const char* meshDataBufferName = nullptr;
        const char* meshMetadataBufferName = nullptr;
        const char* meshAabbBufferName = nullptr;
        const char* visibleMeshletIndexBufferName = nullptr;
        uint32_t meshletCount = 0u;
        uint32_t activeRangeCount = 0u;
        uint32_t verticesPerMeshlet = MESHLET_VERTEX_CAPACITY;
    };

    struct ApplyResult {
        bool buffersRecreated = false;
        bool deltaApplied = false;
    };

    MeshletBufferController();
    explicit MeshletBufferController(Config config);

    bool initialize(BufferManager* bufferManager);
    ApplyResult applyDelta(const MeshStreamingDelta& delta);
    bool flushDirtyState();

    bool hasMeshletManager() const noexcept;
    const char* activeMeshDataBufferName() const noexcept;
    const char* activeMeshMetadataBufferName() const noexcept;
    const char* activeMeshAabbBufferName() const noexcept;
    const char* activeVisibleMeshletIndexBufferName() const noexcept;
    const char* activeMeshletRangeBufferName() const noexcept;
    const char* activeMeshletRangeParamsBufferName() const noexcept;
    const char* residentTileLodBufferName() const noexcept;
    const char* tileSlotBufferName() const noexcept;
    const char* visibleTileIdBufferName() const noexcept;
    const char* tileSceneParamsBufferName() const noexcept;

    uint32_t meshletCount() const noexcept;
    uint32_t activeSelectionMeshletCount() const noexcept;
    uint32_t activeRangeCount() const noexcept;
    uint32_t verticesPerMeshlet() const noexcept;
    uint32_t visibleTileCount() const noexcept;
    uint32_t residentTileCount() const noexcept;

    uint64_t uploadedMeshRevision() const noexcept;
    const std::vector<MeshletAabb>& activeMeshletBounds() const noexcept;

    ActiveBindings activeBindings() const noexcept;

private:
    static constexpr uint32_t kMaxResidentLods = 16u;
    static constexpr uint32_t kResidentFlagActive = 1u << 0;
    static constexpr uint32_t kTileSlotFlagSelectedResidentValid = 1u << 0;
    static constexpr uint32_t kTileSlotFlagAnyResident = 1u << 1;

    struct TileAabb {
        glm::vec3 minCorner{0.0f};
        glm::vec3 maxCorner{0.0f};
    };

    struct PageRun {
        uint32_t startPage = 0u;
        uint32_t pageCount = 0u;

        bool valid() const noexcept {
            return pageCount > 0u;
        }
    };

    struct PageAllocator {
        uint32_t pageSize = 1u;
        uint32_t pageCount = 0u;
        uint32_t maxOrder = 0u;
        std::vector<std::unordered_set<uint32_t>> freeLists;
        std::unordered_map<uint32_t, uint32_t> freeBlockOrders;

        void reset(uint32_t capacityUnits, uint32_t pageSizeUnits);
        uint32_t pagesForUnits(uint32_t unitCount) const noexcept;
        bool allocate(uint32_t unitCount, PageRun& outRun);
        void release(const PageRun& run);
        uint32_t unitOffset(const PageRun& run) const noexcept;
        void addFreeBlock(uint32_t startPage, uint32_t order);
        void removeFreeBlock(uint32_t startPage, uint32_t order);
    };

    struct ResidentRecord {
        MeshTileLodKey key{};
        std::shared_ptr<const PackedMeshletData> packed;
        TileAabb tileAabb{};
        ResidentTileLodGPU gpu{};
        PageRun meshletPages{};
        PageRun quadPages{};
        uint32_t generation = 1u;
        uint32_t lastTouchedRevision = 0u;
        bool active = false;
    };

    struct TileSlotState {
        std::array<ResidentTileLodHandle, kMaxResidentLods> lodHandles{};
        MeshTileSliceCoord tile{};
        ResidentTileLodHandle selectedResident{};
        int8_t selectedLod = -1;
        uint32_t visibleListIndex = ResidentTileLodHandle::kInvalidSlot;
        bool visible = false;
        bool allocated = false;
    };

    bool createBuffers();
    bool writeAllocation(uint32_t meshletStart,
                         uint32_t quadWordStart,
                         const PackedMeshletData& packed) const;
    bool evictForAllocation(uint32_t requiredMeshlets,
                            uint32_t requiredQuadWords,
                            const MeshTileLodKey* protectedKey);
    int32_t chooseEvictionCandidate(const MeshTileLodKey* protectedKey) const;
    bool removeResidentTileLod(const MeshTileLodKey& key);
    bool upsertResidentTileLod(const MeshTileLodUpload& upload);

    void setTileSelectedLod(const MeshTileSliceCoord& tile, int8_t selectedLod);
    void setTileVisible(const MeshTileSliceCoord& tile, bool visible);
    void setResidentLod(const MeshTileSliceCoord& tile, uint8_t lod, ResidentTileLodHandle handle);
    void clearResidentLod(const MeshTileSliceCoord& tile, uint8_t lod, ResidentTileLodHandle handle);

    uint32_t ensureTileSlot(const MeshTileSliceCoord& tile);
    int32_t allocateResidentSlot();
    ResidentTileLodHandle handleForSlot(uint32_t slot) const noexcept;
    bool isHandleValid(ResidentTileLodHandle handle) const noexcept;
    ResidentRecord* residentForHandle(ResidentTileLodHandle handle) noexcept;
    const ResidentRecord* residentForHandle(ResidentTileLodHandle handle) const noexcept;
    ResidentTileLodHandle chooseTileResidentHandle(const TileSlotState& tileSlot) const noexcept;

    void updateTileSelectedResident(uint32_t tileSlotIndex);
    void updateTileResidentFlags(uint32_t tileSlotIndex);
    void updateTileAabb(uint32_t tileSlotIndex);
    void updateVisibleMeshletContribution(uint32_t tileSlotIndex, int64_t delta) noexcept;
    uint32_t visibleMeshletContribution(uint32_t tileSlotIndex) const noexcept;

    void markResidentDirty(uint32_t residentSlot);
    void markTileDirty(uint32_t tileSlot);
    void markVisibleIndexDirty(uint32_t visibleIndex);

    static std::string prefixedName(const std::string& prefix, const char* baseName);
    const std::shared_ptr<const PackedMeshletData>& selectPackedForVariant(const MeshTileLodUpload& upload) const;
    TileAabb computeTileAabb(const PackedMeshletData& packed) const;

    BufferManager* bufferManager_ = nullptr;
    Config config_{};
    std::string namePrefix_;
    MeshletGeometryVariant geometryVariant_ = MeshletGeometryVariant::Culled;

    std::string meshDataBufferName_;
    std::string meshMetadataBufferName_;
    std::string meshAabbBufferName_;
    std::string visibleMeshletIndexBufferName_;
    std::string residentTileLodBufferName_;
    std::string tileSlotBufferName_;
    std::string visibleTileIdBufferName_;
    std::string tileSceneParamsBufferName_;

    PageAllocator meshletPages_{};
    PageAllocator quadPages_{};

    std::vector<ResidentRecord> residentRecords_;
    std::vector<uint32_t> freeResidentSlots_;
    std::unordered_map<MeshTileLodKey, uint32_t> residentSlotByKey_;

    std::vector<TileSlotState> tileSlots_;
    std::vector<TileSlotGPU> tileSlotsGpu_;
    std::vector<VisibleTileId> visibleTileIds_;
    std::unordered_map<MeshTileSliceCoord, uint32_t> tileSlotIndexByCoord_;

    std::vector<uint32_t> dirtyResidentSlots_;
    std::vector<uint8_t> residentDirtyMask_;
    std::vector<uint32_t> dirtyTileSlots_;
    std::vector<uint8_t> tileDirtyMask_;
    std::vector<uint32_t> dirtyVisibleIndices_;
    std::vector<uint8_t> visibleIndexDirtyMask_;
    bool sceneParamsDirty_ = true;
    bool warnedTileSlotCapacity_ = false;
    bool warnedResidentSlotCapacity_ = false;
    bool warnedArenaCapacity_ = false;

    uint32_t activeSelectionMeshletCount_ = 0u;
    uint64_t uploadedMeshRevision_ = 0u;

    mutable std::vector<MeshletAabb> activeMeshletBoundsCache_;
    mutable bool activeMeshletBoundsDirty_ = true;
};
