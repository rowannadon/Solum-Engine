#pragma once

#include "solum_engine/voxel/ChunkMeshes.h"
#include "solum_engine/voxel/ChunkPool.h"
#include "solum_engine/voxel/RegionManager.h"

#include <glm/glm.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_set>

class ChunkResidencyManager;
class CompressedStore;
class JobScheduler;
struct VoxelJob;
struct JobResult;

struct PlayerStreamingContext {
    glm::vec3 playerPosition{0.0f};
    int viewDistanceChunks = 10;
    int verticalChunkMin = 0;
    int verticalChunkMax = COLUMN_CHUNKS_Z - 1;

    float lod0Distance = 10.0f;
    float lod1Distance = 20.0f;
    float lod2Distance = 35.0f;
    float lod3Distance = 50.0f;

    std::array<glm::vec4, 6> frustumPlanes{};
};

struct WorldInterestSet {
    std::unordered_set<RegionCoord, GridCoord2Hash<RegionTag>> regions;
    std::unordered_set<ColumnCoord, GridCoord2Hash<ColumnTag>> columns;
    std::unordered_set<ChunkCoord, GridCoord3Hash<ChunkTag>> chunks;

    void clear();
};

class World {
public:
    explicit World(std::size_t chunkPoolCapacity = 2048, std::size_t workerThreads = 0);
    ~World();

    void update(const PlayerStreamingContext& context);

    bool setBlock(BlockCoord worldBlockCoord, UnpackedBlockMaterial material);
    UnpackedBlockMaterial getBlock(BlockCoord worldBlockCoord) const;

    RegionManager& regions();
    const RegionManager& regions() const;

    ChunkPool& chunkPool();
    const ChunkPool& chunkPool() const;

private:
    WorldInterestSet buildInterestSet(const PlayerStreamingContext& context) const;
    void scheduleStreamingWork(const WorldInterestSet& interestSet);
    void applyCompletedJobs(std::size_t budget);

    uint64_t hashForColumnSeed(ColumnCoord coord) const;

    JobResult executeJob(const VoxelJob& job);

    void applyTerrainCompletion(ColumnCoord columnCoord, bool success);
    void applyStructureCompletion(ColumnCoord columnCoord, bool success);
    void applyLodScanCompletion(ChunkCoord chunkCoord, uint32_t derivedVersion, bool success);
    void applyMeshCompletion(ChunkCoord chunkCoord, MeshData meshData, uint32_t derivedVersion, bool success);
    void applyLodTileCompletion(RegionCoord regionCoord, int lodLevel, RegionLodTileCoord tileCoord, MeshData meshData, uint32_t derivedVersion, bool success);

    mutable std::mutex worldMutex_;

    ChunkPool chunkPool_;
    std::unique_ptr<CompressedStore> compressedStore_;
    RegionManager regionManager_;
    std::unique_ptr<ChunkResidencyManager> residencyManager_;
    std::unique_ptr<JobScheduler> scheduler_;

    MeshHandleTable meshHandles_;

    std::unordered_set<ColumnCoord, GridCoord2Hash<ColumnTag>> terrainScheduled_;
    std::unordered_set<ColumnCoord, GridCoord2Hash<ColumnTag>> structureScheduled_;
    std::unordered_set<ChunkCoord, GridCoord3Hash<ChunkTag>> lodScanScheduled_;
    std::unordered_set<ChunkCoord, GridCoord3Hash<ChunkTag>> meshScheduled_;
    std::unordered_set<uint64_t> lodTileScheduled_;

    uint64_t worldSeed_ = 0xC0FFEEULL;
};
