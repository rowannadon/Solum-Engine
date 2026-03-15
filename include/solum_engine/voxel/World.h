#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <limits>
#include <queue>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>

#include "solum_engine/jobsystem/job_system.hpp"
#include "solum_engine/resources/Coords.h"
#include "solum_engine/voxel/BlockMaterial.h"
#include "solum_engine/voxel/ChunkMesher.h"

class Column;
class Region;

struct WorldChunkEdit {
    ChunkCoord coord{};
    uint8_t changedMipMask = 0u;
};

class World : public IBlockSource {
public:
    struct Config {
        int32_t columnLoadRadius = 1;
        std::size_t maxInFlightColumnJobs = 0;
        jobsystem::JobSystem::Config jobConfig{};
    };

    World();
    explicit World(Config config);
    ~World();

    World(const World&) = delete;
    World& operator=(const World&) = delete;
    World(World&&) = delete;
    World& operator=(World&&) = delete;

    void updatePlayerPosition(const glm::vec3& playerWorldPosition);

    BlockMaterial getBlock(const BlockCoord& coord) const override;
    uint8_t getPackedLight(const BlockCoord& coord) const override;
    BlockMaterial getBlock(const BlockCoord& coord, uint8_t mipLevel) const;
    bool tryGetBlock(const BlockCoord& coord, BlockMaterial& outBlock) const;
    bool tryGetBlock(const BlockCoord& coord, BlockMaterial& outBlock, uint8_t mipLevel) const;
    bool tryGetBlockAndPackedLight(const BlockCoord& coord,
                                   BlockMaterial& outBlock,
                                   uint8_t& outPackedLight,
                                   uint8_t mipLevel) const;
    void sampleBlockAndLightVolume(const BlockCoord& origin,
                                   const glm::ivec3& extent,
                                   const glm::ivec3& stride,
                                   uint8_t mipLevel,
                                   BlockMaterial* outBlocks,
                                   uint8_t* outPackedLights,
                                   uint8_t* outKnownMask) const;
    bool tryGetPackedLight(const BlockCoord& coord, uint8_t& outPackedLight) const;
    bool tryGetPackedLight(const BlockCoord& coord, uint8_t& outPackedLight, uint8_t mipLevel) const;
    bool breakBlock(const BlockCoord& coord);
    bool placeBlock(const BlockCoord& coord, const BlockMaterial& block);
    bool isColumnGenerated(const ColumnCoord& coord) const;
    bool tryGetColumnEmptyChunkMask(const ColumnCoord& coord, uint32_t& outMask) const;
    uint64_t generationRevision() const;
    uint64_t playerEditChunkRevision() const;
    uint64_t lightingChunkRevision() const;
    uint64_t copyGeneratedColumnsSince(uint64_t afterRevision,
                                       std::vector<ColumnCoord>& outColumns,
                                       std::size_t maxCount = std::numeric_limits<std::size_t>::max()) const;
    uint64_t copyPlayerEditedChunksSince(uint64_t afterRevision,
                                         std::vector<WorldChunkEdit>& outChunks,
                                         std::size_t maxCount = std::numeric_limits<std::size_t>::max()) const;
    uint64_t copyLightingChangedChunksSince(uint64_t afterRevision,
                                            std::vector<ChunkCoord>& outChunks,
                                            std::size_t maxCount = std::numeric_limits<std::size_t>::max()) const;
    void copyGeneratedColumns(std::vector<ColumnCoord>& outColumns) const;

    bool hasPendingJobs() const;

private:
    struct ColumnGenerationResult;
    struct ChunkPropagationResult;
    struct ChunkPropagationTask {
        ChunkCoord coord{};
        uint64_t targetEpoch = 0u;
        bool highPriority = false;
    };
    struct LightingChunkState {
        uint64_t topologyEpoch = 1u;
        uint64_t lightingEpoch = 0u;
        uint64_t queuedEpoch = 0u;
        uint64_t inFlightEpoch = 0u;
        uint8_t dirtyFlags = 0u;
        uint64_t lastSolveSignature = 0u;
    };
    struct ScheduledColumnJob {
        ColumnCoord coord{};
        jobsystem::Priority priority = jobsystem::Priority::Low;
    };

    void enqueueColumnGenerationLocked(const ColumnCoord& coord);
    void refillQueuedColumnsLocked();
    std::size_t desiredQueuedColumnCountLocked() const;
    void pruneQueuedColumnsOutsideActiveWindowLocked();
    void collectColumnJobsToScheduleLocked(std::vector<ScheduledColumnJob>& outJobs);
    void dispatchScheduledColumnJobs(std::vector<ScheduledColumnJob>&& jobsToSchedule);
    void pumpColumnGenerationQueue();
    void enqueueChunkPropagationCandidatesLocked(const ColumnCoord& coord);
    void enqueueChunkPropagationIfReadyLocked(const ChunkCoord& coord, bool highPriority = false);
    void collectChunkPropagationJobsLocked(std::vector<ChunkPropagationTask>& outChunks);
    void dispatchChunkPropagationJobs(std::vector<ChunkPropagationTask>&& chunksToSchedule);
    void pumpChunkPropagationQueue();
    void bumpChunkTopologyEpochLocked(const ChunkCoord& coord, bool highPriority, uint8_t dirtyFlags);
    LightingChunkState* tryGetLightingChunkStateLocked(const ChunkCoord& coord);
    const LightingChunkState* tryGetLightingChunkStateLocked(const ChunkCoord& coord) const;
    bool isChunkKnownLocked(const ChunkCoord& coord) const;
    uint64_t computeChunkSolveSignatureLocked(const ChunkCoord& coord) const;
    bool tryApplyImmediateLightingAround(const ChunkCoord& centerChunk);

    void onColumnGenerated(const ColumnCoord& coord, Column&& column);
    bool propagateChunkLighting(const ChunkCoord& coord,
                                uint64_t targetEpoch,
                                bool* outLightChanged = nullptr,
                                uint64_t* outSolveSignature = nullptr);
    bool applyBlockEditLocked(const BlockCoord& coord,
                              const BlockMaterial& newBlock,
                              bool requireCurrentAir,
                              bool requireCurrentSolid);
    bool canPropagateChunkLocked(const ChunkCoord& coord) const;
    bool isColumnSkycastCompleteLocked(const ColumnCoord& coord) const;
    Column* tryGetSkycastColumnLocked(const ColumnCoord& coord);
    const Column* tryGetSkycastColumnLocked(const ColumnCoord& coord) const;

    bool tryGetBlockLocked(const BlockCoord& coord, BlockMaterial& outBlock, uint8_t mipLevel) const;
    bool tryGetBlockAndPackedLightLocked(const BlockCoord& coord,
                                         BlockMaterial& outBlock,
                                         uint8_t& outPackedLight,
                                         uint8_t mipLevel) const;
    bool tryGetPackedLightLocked(const BlockCoord& coord, uint8_t& outPackedLight, uint8_t mipLevel) const;
    bool isColumnGeneratedLocked(const ColumnCoord& coord) const;
    bool isWithinActiveWindowLocked(const ColumnCoord& coord, int32_t extraRadius) const;
    Region* getOrCreateRegionLocked(const RegionCoord& coord);

    static jobsystem::Priority priorityFromDistanceSq(int32_t distanceSq);

    struct QueuedColumnEntry {
        ColumnCoord coord{};
        int32_t distanceSq = 0;
        uint64_t centerVersion = 0;
        uint64_t sequence = 0;
    };

    struct QueuedColumnEntryCompare {
        bool operator()(const QueuedColumnEntry& a, const QueuedColumnEntry& b) const noexcept {
            if (a.distanceSq != b.distanceSq) {
                return a.distanceSq > b.distanceSq;
            }
            return a.sequence > b.sequence;
        }
    };

    Config config_;
    jobsystem::JobSystem jobs_;
    jobsystem::JobSystem chunkPropagationJobs_;

    mutable std::shared_mutex worldMutex_;
    std::unordered_map<RegionCoord, std::unique_ptr<Region>> regions_;
    std::unordered_set<ColumnCoord> skycastColumns_;
    std::unordered_map<ChunkCoord, LightingChunkState> lightingChunkStates_;
    std::unordered_set<ColumnCoord> generatedColumns_;
    std::vector<ColumnCoord> generatedColumnHistory_;
    std::vector<WorldChunkEdit> playerEditedChunkHistory_;
    std::vector<ChunkCoord> lightingChangedChunkHistory_;
    std::unordered_set<ColumnCoord> pendingColumnJobs_;
    std::deque<ChunkPropagationTask> queuedChunkPropagationJobs_;
    std::unordered_map<ChunkCoord, uint64_t> pendingChunkPropagationJobs_;
    std::unordered_set<ColumnCoord> queuedColumnJobs_;
    std::priority_queue<
        QueuedColumnEntry,
        std::vector<QueuedColumnEntry>,
        QueuedColumnEntryCompare
    > queuedColumnHeap_;
    std::atomic<uint64_t> generationRevision_{0};
    std::atomic<uint64_t> playerEditChunkRevision_{0};
    std::atomic<uint64_t> lightingChunkRevision_{0};
    std::atomic<bool> shuttingDown_{false};
    std::size_t maxInFlightColumnJobs_ = 1;
    std::size_t maxInFlightChunkPropagationJobs_ = 1;
    uint64_t queueSequence_ = 0;
    uint64_t queueCenterVersion_ = 0;

    ColumnCoord lastScheduledCenter_{0, 0};
    bool hasLastScheduledCenter_ = false;
};
