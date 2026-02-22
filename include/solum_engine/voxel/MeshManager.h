#pragma once

#include <atomic>
#include <cstdint>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>

#include "solum_engine/jobsystem/job_system.hpp"
#include "solum_engine/render/MeshletTypes.h"
#include "solum_engine/resources/Coords.h"

class World;

class MeshManager {
public:
    struct Config {
        int32_t columnMeshRadius = 1;
        jobsystem::JobSystem::Config jobConfig{};
    };

    explicit MeshManager(const World& world);
    MeshManager(const World& world, Config config);
    ~MeshManager();

    MeshManager(const MeshManager&) = delete;
    MeshManager& operator=(const MeshManager&) = delete;
    MeshManager(MeshManager&&) = delete;
    MeshManager& operator=(MeshManager&&) = delete;

    void updatePlayerPosition(const glm::vec3& playerWorldPosition);
    void waitForIdle();

    std::vector<Meshlet> copyMeshlets() const;
    std::vector<Meshlet> copyMeshletsAround(const ColumnCoord& centerColumn, int32_t columnRadius) const;
    uint64_t meshRevision() const noexcept;
    bool hasPendingJobs() const;

private:
    struct MeshGenerationResult;

    void scheduleChunksAround(const ColumnCoord& centerColumn);
    void scheduleChunksDelta(const ColumnCoord& previousCenter, const ColumnCoord& newCenter);
    void scheduleRemeshForNewColumns(const ColumnCoord& centerColumn);
    void scheduleChunkMeshing(const ChunkCoord& coord, jobsystem::Priority priority, bool forceRemesh);

    void onChunkMeshed(const ChunkCoord& coord, std::vector<Meshlet>&& meshlets);

    bool isWithinActiveWindowLocked(const ColumnCoord& coord, int32_t extraRadius) const;

    static jobsystem::Priority priorityFromDistanceSq(int32_t distanceSq);

    const World& world_;
    Config config_;
    jobsystem::JobSystem jobs_;

    mutable std::shared_mutex meshMutex_;
    std::unordered_set<ColumnCoord> knownGeneratedColumns_;
    std::unordered_set<ChunkCoord> pendingMeshJobs_;
    std::unordered_set<ChunkCoord> deferredRemeshChunks_;
    std::unordered_map<ChunkCoord, std::vector<Meshlet>> chunkMeshes_;

    std::atomic<uint64_t> meshRevision_{0};
    std::atomic<bool> shuttingDown_{false};

    ColumnCoord lastScheduledCenter_{0, 0};
    bool hasLastScheduledCenter_ = false;
};
