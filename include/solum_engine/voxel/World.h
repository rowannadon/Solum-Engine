#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
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

class World;

class WorldSection : public IBlockSource {
public:
    struct Sample {
        BlockMaterial block{};
        bool known = false;
    };

    WorldSection(const World& world, const BlockCoord& origin, const glm::ivec3& extent, uint8_t mipLevel = 0);

    const BlockCoord& origin() const { return origin_; }
    const glm::ivec3& extent() const { return extent_; }
    uint8_t mipLevel() const { return mipLevel_; }

    BlockMaterial getBlock(const BlockCoord& coord) const override;
    bool tryGetBlock(const BlockCoord& coord, BlockMaterial& outBlock) const;
    BlockMaterial getLocalBlock(int32_t x, int32_t y, int32_t z) const;
    bool tryGetLocalBlock(int32_t x, int32_t y, int32_t z, BlockMaterial& outBlock) const;
    void copySamples(std::vector<Sample>& outSamples) const;

private:
    const World& world_;
    BlockCoord origin_;
    glm::ivec3 extent_;
    uint8_t mipLevel_ = 0;
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
    void waitForIdle();

    BlockMaterial getBlock(const BlockCoord& coord) const override;
    BlockMaterial getBlock(const BlockCoord& coord, uint8_t mipLevel) const;
    bool tryGetBlock(const BlockCoord& coord, BlockMaterial& outBlock) const;
    bool tryGetBlock(const BlockCoord& coord, BlockMaterial& outBlock, uint8_t mipLevel) const;
    bool isColumnGenerated(const ColumnCoord& coord) const;
    bool isChunkEmpty(const ChunkCoord& coord) const;
    bool tryGetColumnEmptyChunkMask(const ColumnCoord& coord, uint32_t& outMask) const;
    uint64_t generationRevision() const;
    void copyGeneratedColumns(std::vector<ColumnCoord>& outColumns) const;
    void copyGeneratedColumnsAround(const ColumnCoord& centerColumn, int32_t radius, std::vector<ColumnCoord>& outColumns) const;

    WorldSection createSection(const BlockCoord& origin, const glm::ivec3& extent) const;
    WorldSection createSection(const BlockCoord& origin, const glm::ivec3& extent, uint8_t mipLevel) const;

    bool hasPendingJobs() const;

private:
    struct ColumnGenerationResult;
    struct ScheduledColumnJob {
        ColumnCoord coord{};
        jobsystem::Priority priority = jobsystem::Priority::Low;
    };
    friend class WorldSection;

    void scheduleColumnsAround(const ColumnCoord& centerColumn);
    void scheduleColumnsDelta(const ColumnCoord& previousCenter, const ColumnCoord& newCenter);
    void enqueueColumnGenerationLocked(const ColumnCoord& coord);
    void enqueueColumnGenerationBatch(const std::vector<ColumnCoord>& coords);
    void pruneQueuedColumnsOutsideActiveWindowLocked();
    void collectColumnJobsToScheduleLocked(std::vector<ScheduledColumnJob>& outJobs);
    void dispatchScheduledColumnJobs(std::vector<ScheduledColumnJob>&& jobsToSchedule);
    void pumpColumnGenerationQueue();

    void onColumnGenerated(const ColumnCoord& coord, Column&& column);

    bool tryGetBlockLocked(const BlockCoord& coord, BlockMaterial& outBlock, uint8_t mipLevel) const;
    bool isColumnGeneratedLocked(const ColumnCoord& coord) const;
    bool isWithinActiveWindowLocked(const ColumnCoord& coord, int32_t extraRadius) const;
    Region* getOrCreateRegionLocked(const RegionCoord& coord);

    static jobsystem::Priority priorityFromDistanceSq(int32_t distanceSq);

    Config config_;
    jobsystem::JobSystem jobs_;

    mutable std::shared_mutex worldMutex_;
    std::unordered_map<RegionCoord, std::unique_ptr<Region>> regions_;
    std::unordered_set<ColumnCoord> generatedColumns_;
    std::unordered_set<ColumnCoord> pendingColumnJobs_;
    std::unordered_set<ColumnCoord> queuedColumnJobs_;
    std::atomic<uint64_t> generationRevision_{0};
    std::atomic<bool> shuttingDown_{false};
    std::size_t maxInFlightColumnJobs_ = 1;

    ColumnCoord lastScheduledCenter_{0, 0};
    bool hasLastScheduledCenter_ = false;
};
