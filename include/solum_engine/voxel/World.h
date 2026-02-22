#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>

#include "solum_engine/jobsystem/job_system.hpp"
#include "solum_engine/render/MeshletTypes.h"
#include "solum_engine/resources/Coords.h"
#include "solum_engine/voxel/BlockMaterial.h"
#include "solum_engine/voxel/ChunkMesher.h"

class Column;
class Region;

class World;

class WorldSection : public IBlockSource {
public:
    WorldSection(const World& world, const BlockCoord& origin, const glm::ivec3& extent);

    const BlockCoord& origin() const { return origin_; }
    const glm::ivec3& extent() const { return extent_; }

    BlockMaterial getBlock(const BlockCoord& coord) const override;
    BlockMaterial getLocalBlock(int32_t x, int32_t y, int32_t z) const;

private:
    const World& world_;
    BlockCoord origin_;
    glm::ivec3 extent_;
};

class World : public IBlockSource {
public:
    struct Config {
        int32_t columnLoadRadius = 1;
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
    bool tryGetBlock(const BlockCoord& coord, BlockMaterial& outBlock) const;

    WorldSection createSection(const BlockCoord& origin, const glm::ivec3& extent) const;

    std::vector<Meshlet> copyMeshlets() const;
    std::vector<Meshlet> copyMeshletsAround(const ColumnCoord& centerColumn, int32_t columnRadius) const;
    uint64_t meshRevision() const noexcept;
    bool hasPendingJobs() const;

private:
    struct ColumnGenerationResult;
    struct MeshGenerationResult;

    void scheduleColumnsAround(const ColumnCoord& centerColumn);
    void scheduleColumnsDelta(const ColumnCoord& previousCenter, const ColumnCoord& newCenter);
    void scheduleColumnGeneration(const ColumnCoord& coord, jobsystem::Priority priority);
    void scheduleChunkMeshing(const ChunkCoord& coord, jobsystem::Priority priority, bool forceRemesh);

    void onColumnGenerated(const ColumnCoord& coord, Column&& column);
    void onChunkMeshed(const ChunkCoord& coord, std::vector<Meshlet>&& meshlets);

    bool tryGetBlockLocked(const BlockCoord& coord, BlockMaterial& outBlock) const;
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
    std::unordered_set<ChunkCoord> pendingMeshJobs_;
    std::unordered_set<ChunkCoord> deferredRemeshChunks_;
    std::unordered_map<ChunkCoord, std::vector<Meshlet>> chunkMeshes_;

    std::atomic<uint64_t> meshRevision_{0};
    std::atomic<bool> shuttingDown_{false};

    ColumnCoord lastScheduledCenter_{0, 0};
    bool hasLastScheduledCenter_ = false;
};
