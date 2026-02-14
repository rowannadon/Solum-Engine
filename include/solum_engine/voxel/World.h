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

// Shared startup tuning for both world streaming defaults and renderer region/LOD behavior.
// This is the single place to tweak for now (GUI controls can later drive these values at runtime).
inline constexpr int kRegionLodCount = 5;

struct WorldTuningParameters {
    int viewDistanceChunks = 255;
    int verticalChunkMin = 0;
    int verticalChunkMax = COLUMN_CHUNKS_Z - 1;

    // Region renderer LOD mesh decimation in blocks per cell.
    std::array<int, kRegionLodCount> regionLodSteps{2, 4, 8, 16, 32};
    // Distance thresholds where LOD switches from i to i+1.
    std::array<float, kRegionLodCount - 1> regionLodSwitchDistances{
        REGION_BLOCKS_XY * 4.0f,
        REGION_BLOCKS_XY * 8.0f,
        REGION_BLOCKS_XY * 16.0f,
        REGION_BLOCKS_XY * 32.0f,
    };
    double regionBuildBudgetMs = 4.0f;

    // Heightmap terrain settings.
    const char* heightmapRelativePath = "heightmap.png";
    int heightmapUpscaleFactor = 4;
    bool heightmapWrap = true;
    int terrainMinHeightBlocks = CHUNK_SIZE * 2;
    int terrainMaxHeightBlocks = CHUNK_SIZE * (COLUMN_CHUNKS_Z - 2);
};

inline constexpr WorldTuningParameters kDefaultWorldTuningParameters{};

struct PlayerStreamingContext {
    glm::vec3 playerPosition{0.0f};
    int viewDistanceChunks = kDefaultWorldTuningParameters.viewDistanceChunks;
    int verticalChunkMin = kDefaultWorldTuningParameters.verticalChunkMin;
    int verticalChunkMax = kDefaultWorldTuningParameters.verticalChunkMax;

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
    explicit World(std::size_t chunkPoolCapacity = 16384, std::size_t workerThreads = 0);
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
