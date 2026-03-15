#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <queue>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>

#include "solum_engine/jobsystem/job_system.hpp"
#include "solum_engine/render/MeshletTypes.h"
#include "solum_engine/resources/Coords.h"
#include "solum_engine/voxel/BlockModelLibrary.h"
#include "solum_engine/voxel/ChunkMesher.h"
#include "solum_engine/voxel/MeshTileTypes.h"
#include "solum_engine/voxel/StreamingUpload.h"
#include "solum_engine/voxel/World.h"

class MeshManager {
public:
    struct Config {
        int32_t lodLevelCount = 7;
        int32_t meshTileSizeChunks = 4;
        int32_t meshTileHeightChunks = 32;
        int32_t activeChunkRadius = 128;
        float lodSseTargetPixels = 8.0f;
        float lodSseHysteresisPixels = 0.25f;
        float lodSseMinDepthBlocks = 4.0f;
        float lodSseFallbackProjectionScale = 390.0f;
        jobsystem::JobSystem::Config jobConfig{};
    };

    explicit MeshManager(const World& world, std::shared_ptr<const BlockModelLibrary> blockModelLibrary = {});
    MeshManager(const World& world, Config config, std::shared_ptr<const BlockModelLibrary> blockModelLibrary = {});
    ~MeshManager();

    MeshManager(const MeshManager&) = delete;
    MeshManager& operator=(const MeshManager&) = delete;
    MeshManager(MeshManager&&) = delete;
    MeshManager& operator=(MeshManager&&) = delete;

    void updatePlayerPosition(const glm::vec3& playerWorldPosition, float sseProjectionScale);

    std::vector<MeshTileLodUpload> consumePendingTileLodUploads(std::size_t maxCount);
    std::vector<MeshTileLodKey> consumePendingTileLodRemovals(std::size_t maxCount);
    bool consumeSelectionSnapshot(uint64_t& outRevision,
                                  std::vector<MeshTileSelectionEntry>& outSelection);

    uint64_t meshRevision() const noexcept;
    bool hasPendingJobs() const;

private:
    struct PlanningUpdate {
        ChunkCoord centerChunk{};
        ColumnCoord centerColumn{};
        float safeSseProjectionScale = 390.0f;
        int32_t centerShiftChunks = 0;
        bool planningResetRequired = false;
        bool shouldPumpPendingWork = false;
    };

    struct WorldChangeFlags {
        bool playerEdits = false;
        bool lighting = false;
        bool generation = false;
    };

    struct PendingMeshDispatch {
        TileLodCoord coord{};
        jobsystem::Priority priority = jobsystem::Priority::Low;
        bool forceRemesh = false;
        bool usePriorityQueue = false;
    };

    struct MeshTileLodState {
        std::shared_ptr<const PackedMeshletData> culledPacked;
        std::shared_ptr<const PackedMeshletData> doubleSidedPacked;
        bool resident = false;
        bool uploadQueued = false;
        uint64_t revision = 0u;
    };

    struct MeshTileState {
        std::unordered_map<uint8_t, std::unordered_map<int32_t, MeshTileLodState>> lodStates;
        int8_t desiredLod = -1;
        int8_t selectedLod = -1;
        bool preferLod0DuringRemesh = false;
    };

    struct MeshGenerationResult {
        TileLodCoord coord{};
        ChunkMeshOutput meshOutput;
        bool meshed = false;
    };

    struct QueuedVisibleTileEntry {
        MeshTileCoord tile{};
        int32_t ring = 0;
        int32_t distanceSq = 0;
        uint64_t centerVersion = 0u;
        uint64_t sequence = 0u;
    };

    struct QueuedVisibleTileEntryCompare {
        bool operator()(const QueuedVisibleTileEntry& a, const QueuedVisibleTileEntry& b) const noexcept {
            if (a.ring != b.ring) {
                return a.ring > b.ring;
            }
            if (a.distanceSq != b.distanceSq) {
                return a.distanceSq > b.distanceSq;
            }
            return a.sequence > b.sequence;
        }
    };

    void scheduleTilesAround(const ChunkCoord& centerChunk,
                             const glm::vec3& playerWorldPosition,
                             float sseProjectionScale,
                             int32_t centerShiftChunks);
    void scheduleRemeshForChangedChunks(const ColumnCoord& centerColumn,
                                        const std::vector<WorldChunkEdit>& changedChunks,
                                        bool preferFastLod0Visibility = false);
    void scheduleRemeshForPlayerEditedChunks(const ColumnCoord& centerColumn);
    void scheduleRemeshForLightingChangedChunks(const ColumnCoord& centerColumn);
    void scheduleRemeshForNewColumns(const ColumnCoord& centerColumn);
    void wakeVisibleTilesForGeneratedColumns(const std::vector<ColumnCoord>& generatedColumns);
    PlanningUpdate updatePlanningInputs(const glm::vec3& playerWorldPosition, float sseProjectionScale);
    WorldChangeFlags drainWorldChangeFeeds(const ColumnCoord& centerColumn);
    void maybeResetTilePlanning(const PlanningUpdate& update, const glm::vec3& playerWorldPosition);
    void pumpPendingVisibleWork(const PlanningUpdate& update, const WorldChangeFlags& changeFlags);
    bool scheduleTileLodMeshing(const TileLodCoord& coord,
                                jobsystem::Priority priority,
                                bool forceRemesh,
                                int32_t activeWindowExtraChunks,
                                bool usePriorityQueue = false);
    void queueTileLodUploadLocked(const MeshTileLodKey& key, bool highPriority);
    void queueTileLodRemovalLocked(const MeshTileLodKey& key);
    void resetTileQueuesLocked(const ChunkCoord& centerChunk,
                               const glm::vec3& playerWorldPosition,
                               float sseProjectionScale,
                               int32_t centerShiftChunks);
    void pruneMeshTilesOutsideWindowLocked();
    void ensureVisibleFrontierLocked();
    bool initializeVisibleRingLocked(int32_t ring);
    void enqueueVisibleTileLocked(const MeshTileCoord& tile, int32_t ring, int32_t distanceSq);
    void markVisibleTileReadyLocked(const MeshTileCoord& tile);
    void noteVisibleTileAttemptFinishedLocked(const MeshTileCoord& tile);
    void waitVisibleTileForFootprintLocked(const MeshTileCoord& tile);
    void advanceVisibleTileLocked(const MeshTileCoord& tile,
                                  bool dispatchNow,
                                  std::vector<PendingMeshDispatch>* dispatches,
                                  std::vector<MeshTileCoord>* dispatchedTiles,
                                  bool* repump);
    void pumpTileQueues();

    int8_t desiredLodForTile(const MeshTileCoord& tileCoord,
                             const ChunkCoord& centerChunk,
                             const glm::vec3& playerWorldPosition,
                             float sseProjectionScale,
                             int32_t extraChunks) const;
    float tileDepthEstimateBlocks(const MeshTileCoord& tileCoord,
                                  const glm::vec3& playerWorldPosition,
                                  int32_t extraChunks) const;
    float projectedSsePixels(uint8_t lodLevel, float depthBlocks, float sseProjectionScale) const;
    int8_t applyLodHysteresis(const MeshTileCoord& tileCoord,
                              int8_t candidateLod,
                              int8_t previousLod,
                              const glm::vec3& playerWorldPosition,
                              float sseProjectionScale) const;

    bool isTileWithinActiveWindowLocked(const MeshTileCoord& tileCoord, int32_t extraChunks) const;
    bool isTileFootprintGenerated(const MeshTileCoord& tileCoord) const;
    bool isLodCellAllAir(const ChunkCoord& cellCoord,
                         uint8_t lodLevel,
                         std::unordered_map<ColumnCoord, uint32_t>& emptyMaskCache) const;
    int8_t chooseRenderableLodForTileLocked(const MeshTileCoord& tileCoord, const MeshTileState& state) const;
    bool refreshSelectedLodLocked(const MeshTileCoord& tileCoord, MeshTileState& state) const;
    bool lodFullyResidentLocked(const MeshTileState& state, int32_t lodLevel) const;
    bool allTileLodsResidentLocked(const MeshTileState& state) const;
    uint8_t pendingSliceCountForLodLocked(const MeshTileCoord& tileCoord, uint8_t lodLevel) const;
    bool canDisplayLod0DuringRemeshLocked(const MeshTileCoord& tileCoord, const MeshTileState& state) const;
    bool isTileDisplayReadyLocked(const MeshTileCoord& tileCoord, const MeshTileState& state) const;
    bool hasVisibleQueueWorkLocked() const;
    int32_t visibleRingForTileLocked(const MeshTileCoord& tileCoord) const;
    int32_t visibleDistanceSqForTileLocked(const MeshTileCoord& tileCoord) const;

    ChunkMeshOutput meshTileLod(const TileLodCoord& coord) const;
    ChunkMeshOutput meshLodCell(const ChunkCoord& cellCoord, uint8_t lodLevel) const;

    int32_t cellCountPerAxisForLod(uint8_t lodLevel) const;
    int32_t cellCountPerZForLod(uint8_t lodLevel) const;
    int32_t maxConfiguredRadius() const;

    static bool tileInBounds(const MeshTileCoord& tileCoord,
                             int32_t minTileX,
                             int32_t maxTileX,
                             int32_t minTileY,
                             int32_t maxTileY);
    static int32_t chunkSpanForLod(uint8_t lodLevel);
    static int32_t chunkZCountForLod(uint8_t lodLevel);
    static jobsystem::Priority priorityFromLodLevel(uint8_t lodLevel);
    static void sanitizeConfig(Config& config);

    const World& world_;
    std::shared_ptr<const BlockModelLibrary> blockModelLibrary_;
    Config config_;
    jobsystem::JobSystem jobs_;
    jobsystem::JobSystem priorityJobs_;
    int32_t meshTileSizeChunks_ = 4;
    int32_t meshTileHeightChunks_ = 32;
    int32_t meshTileSliceCount_ = 1;

    mutable std::shared_mutex meshMutex_;
    std::unordered_set<TileLodCoord> pendingTileLodJobs_;
    std::unordered_set<TileLodCoord> pendingPriorityTileLodJobs_;
    std::unordered_set<TileLodCoord> deferredRemeshTileLods_;
    std::unordered_map<MeshTileCoord, MeshTileState> meshTiles_;
    std::unordered_set<MeshTileCoord> currentVisibleRingOutstandingTiles_;
    std::unordered_set<MeshTileCoord> queuedVisibleTiles_;
    std::unordered_set<MeshTileCoord> waitingVisibleTiles_;
    std::priority_queue<
        QueuedVisibleTileEntry,
        std::vector<QueuedVisibleTileEntry>,
        QueuedVisibleTileEntryCompare
    > queuedVisibleTileHeap_;

    std::deque<MeshTileLodKey> pendingUploadOrder_;
    std::unordered_set<MeshTileLodKey> pendingUploadSet_;
    std::deque<MeshTileLodKey> pendingPriorityUploadOrder_;
    std::unordered_set<MeshTileLodKey> pendingPriorityUploadSet_;
    std::deque<MeshTileLodKey> pendingRemovalOrder_;
    std::unordered_set<MeshTileLodKey> pendingRemovalSet_;

    std::atomic<uint64_t> meshRevision_{0};
    std::atomic<uint64_t> processedWorldPlayerEditRevision_{0};
    std::atomic<uint64_t> processedWorldLightingRevision_{0};
    std::atomic<uint64_t> processedWorldGenerationRevision_{0};
    std::atomic<bool> shuttingDown_{false};

    ChunkCoord lastScheduledCenterChunk_{0, 0, 0};
    bool hasLastScheduledCenter_ = false;
    glm::vec3 lastPlayerWorldPosition_{0.0f, 0.0f, 0.0f};
    bool hasLastPlayerWorldPosition_ = false;
    glm::vec3 lastPlanningPlayerWorldPosition_{0.0f, 0.0f, 0.0f};
    bool hasLastPlanningPlayerWorldPosition_ = false;
    float lastSseProjectionScale_ = 390.0f;
    bool hasLastSseProjectionScale_ = false;
    uint64_t tileQueueCenterVersion_ = 0u;
    uint64_t tileQueueSequence_ = 0u;
    MeshTileCoord currentCenterTile_{0, 0};
    int32_t planningMinTileX_ = 0;
    int32_t planningMaxTileX_ = -1;
    int32_t planningMinTileY_ = 0;
    int32_t planningMaxTileY_ = -1;
    int32_t planningPrefetchChunks_ = 0;
    int32_t visibleFrontierRing_ = 0;
    bool currentVisibleRingInitialized_ = false;
    int32_t maxVisibleFrontierRing_ = 0;

    uint64_t selectionRevision_ = 0u;
    bool selectionSnapshotDirty_ = true;
};
