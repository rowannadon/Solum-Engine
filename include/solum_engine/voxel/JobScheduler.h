#pragma once

#include "solum_engine/resources/Coords.h"
#include "solum_engine/voxel/ChunkMeshes.h"
#include "solum_engine/voxel/Region.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <variant>
#include <vector>

enum class VoxelJobType : uint8_t {
    TerrainGeneration,
    StructureGeneration,
    LodScan,
    MeshL0,
    LodTile,
    CompressChunk,
    UncompressChunk,
};

enum class JobPriority : uint8_t {
    High = 0,
    Medium = 1,
    Low = 2,
};

struct TerrainGenerationJobInput {
    ColumnCoord columnCoord;
    uint64_t seed = 0;
};

struct StructureGenerationJobInput {
    ColumnCoord columnCoord;
};

struct LodScanJobInput {
    ChunkCoord chunkCoord;
    uint32_t expectedBlockVersion = 0;
};

struct MeshJobInput {
    ChunkCoord chunkCoord;
    uint32_t expectedBlockVersion = 0;
};

struct LodTileJobInput {
    RegionCoord regionCoord;
    int lodLevel = 0;
    RegionLodTileCoord tileCoord;
    uint32_t expectedVersion = 0;
};

struct CompressChunkJobInput {
    ChunkCoord chunkCoord;
};

struct UncompressChunkJobInput {
    ChunkCoord chunkCoord;
};

using JobPayload = std::variant<
    TerrainGenerationJobInput,
    StructureGenerationJobInput,
    LodScanJobInput,
    MeshJobInput,
    LodTileJobInput,
    CompressChunkJobInput,
    UncompressChunkJobInput>;

struct VoxelJob {
    VoxelJobType type = VoxelJobType::TerrainGeneration;
    JobPriority priority = JobPriority::Medium;
    uint64_t ticket = 0;
    JobPayload payload;
};

struct TerrainJobResult {
    ColumnCoord columnCoord;
    bool success = false;
};

struct StructureJobResult {
    ColumnCoord columnCoord;
    bool success = false;
};

struct LodScanJobResult {
    ChunkCoord chunkCoord;
    uint32_t derivedVersion = 0;
    bool success = false;
};

struct MeshJobResult {
    ChunkCoord chunkCoord;
    MeshData meshData;
    uint32_t derivedVersion = 0;
    bool success = false;
};

struct LodTileJobResult {
    RegionCoord regionCoord;
    int lodLevel = 0;
    RegionLodTileCoord tileCoord;
    MeshData meshData;
    uint32_t derivedVersion = 0;
    bool success = false;
};

struct CompressChunkJobResult {
    ChunkCoord chunkCoord;
    bool success = false;
};

struct UncompressChunkJobResult {
    ChunkCoord chunkCoord;
    bool success = false;
};

using JobResultPayload = std::variant<
    TerrainJobResult,
    StructureJobResult,
    LodScanJobResult,
    MeshJobResult,
    LodTileJobResult,
    CompressChunkJobResult,
    UncompressChunkJobResult>;

struct JobResult {
    VoxelJobType type = VoxelJobType::TerrainGeneration;
    uint64_t ticket = 0;
    JobResultPayload payload;
};

class JobScheduler {
public:
    using Executor = std::function<JobResult(const VoxelJob&)>;

    explicit JobScheduler(std::size_t workerThreads = 0);
    ~JobScheduler();

    void setExecutor(Executor executor);

    uint64_t enqueue(VoxelJob job);
    bool tryPopResult(JobResult& outResult);

private:
    void workerMain();
    bool tryPopNextJobLocked(VoxelJob& job);

    std::atomic<bool> stopping_{false};
    std::atomic<uint64_t> nextTicket_{1};

    mutable std::mutex queueMutex_;
    std::condition_variable queueCv_;
    std::deque<VoxelJob> highQueue_;
    std::deque<VoxelJob> mediumQueue_;
    std::deque<VoxelJob> lowQueue_;

    mutable std::mutex resultMutex_;
    std::deque<JobResult> completedResults_;

    mutable std::mutex executorMutex_;
    Executor executor_;

    std::vector<std::thread> workers_;
};
