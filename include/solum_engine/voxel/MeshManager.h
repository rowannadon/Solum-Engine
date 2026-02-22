#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>

#include "solum_engine/jobsystem/job_system.hpp"
#include "solum_engine/render/MeshletTypes.h"
#include "solum_engine/resources/Coords.h"
#include "solum_engine/voxel/Chunk.h"

class World;

struct LODChunkCoord {
    ChunkCoord coord{};
    uint8_t lodLevel = 0;

    friend bool operator==(const LODChunkCoord& a, const LODChunkCoord& b) {
        return a.lodLevel == b.lodLevel && a.coord == b.coord;
    }

    friend bool operator<(const LODChunkCoord& a, const LODChunkCoord& b) {
        if (a.lodLevel != b.lodLevel) {
            return a.lodLevel < b.lodLevel;
        }
        return a.coord < b.coord;
    }
};

namespace std {
template <>
struct hash<LODChunkCoord> {
    size_t operator()(const LODChunkCoord& coord) const noexcept {
#if SIZE_MAX > UINT32_MAX
        constexpr size_t kGoldenRatio = 0x9e3779b97f4a7c15ull;
#else
        constexpr size_t kGoldenRatio = 0x9e3779b9u;
#endif
        size_t seed = hash<ChunkCoord>{}(coord.coord);
        seed ^= hash<uint8_t>{}(coord.lodLevel) + kGoldenRatio + (seed << 6) + (seed >> 2);
        return seed;
    }
};
}  // namespace std

class MeshManager {
public:
    struct Config {
        std::vector<int32_t> lodChunkRadii{4, 8, 16};
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

    void scheduleLodRingsAround(const ChunkCoord& centerChunk, int32_t centerShiftChunks);
    void scheduleRemeshForNewColumns(const ColumnCoord& centerColumn);
    void scheduleChunkMeshing(const LODChunkCoord& coord,
                              jobsystem::Priority priority,
                              bool forceRemesh,
                              const ChunkCoord& centerChunkForSeams,
                              int32_t activeWindowExtraChunks);

    void onChunkMeshed(const LODChunkCoord& coord, std::vector<Meshlet>&& meshlets);

    bool isInDesiredRingForCenter(const LODChunkCoord& coord,
                                  const ChunkCoord& centerChunk,
                                  int32_t extraChunks) const;
    bool isWithinActiveWindowLocked(const LODChunkCoord& coord, int32_t extraChunks) const;
    bool isFootprintGenerated(const LODChunkCoord& coord) const;

    int32_t maxConfiguredRadius() const;

    static uint8_t chunkSpanForLod(uint8_t lodLevel);
    static int32_t chunkZCountForLod(uint8_t lodLevel);
    static jobsystem::Priority priorityFromDistanceSq(int32_t distanceSq);
    static void sanitizeConfig(Config& config);

    const World& world_;
    Config config_;
    jobsystem::JobSystem jobs_;

    mutable std::shared_mutex meshMutex_;
    std::unordered_set<ColumnCoord> knownGeneratedColumns_;
    std::unordered_set<LODChunkCoord> pendingMeshJobs_;
    std::unordered_set<LODChunkCoord> deferredRemeshChunks_;
    std::unordered_map<LODChunkCoord, std::vector<Meshlet>> chunkMeshes_;

    std::atomic<uint64_t> meshRevision_{0};
    std::atomic<bool> shuttingDown_{false};

    ChunkCoord lastScheduledCenterChunk_{0, 0, 0};
    bool hasLastScheduledCenter_ = false;
};
