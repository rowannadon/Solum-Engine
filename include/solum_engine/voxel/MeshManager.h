#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
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

struct MeshTileCoord {
    int32_t x = 0;
    int32_t y = 0;

    friend bool operator==(const MeshTileCoord& a, const MeshTileCoord& b) {
        return a.x == b.x && a.y == b.y;
    }

    friend bool operator<(const MeshTileCoord& a, const MeshTileCoord& b) {
        if (a.x != b.x) {
            return a.x < b.x;
        }
        return a.y < b.y;
    }
};

struct TileLodCoord {
    MeshTileCoord tile{};
    uint8_t lodLevel = 0;

    friend bool operator==(const TileLodCoord& a, const TileLodCoord& b) {
        return a.lodLevel == b.lodLevel && a.tile == b.tile;
    }

    friend bool operator<(const TileLodCoord& a, const TileLodCoord& b) {
        if (a.lodLevel != b.lodLevel) {
            return a.lodLevel < b.lodLevel;
        }
        return a.tile < b.tile;
    }
};

struct TileLodCellCoord {
    TileLodCoord tileLod{};
    uint16_t cellX = 0;
    uint16_t cellY = 0;

    friend bool operator==(const TileLodCellCoord& a, const TileLodCellCoord& b) {
        return a.tileLod == b.tileLod &&
               a.cellX == b.cellX &&
               a.cellY == b.cellY;
    }

    friend bool operator<(const TileLodCellCoord& a, const TileLodCellCoord& b) {
        if (!(a.tileLod == b.tileLod)) {
            return a.tileLod < b.tileLod;
        }
        if (a.cellY != b.cellY) {
            return a.cellY < b.cellY;
        }
        return a.cellX < b.cellX;
    }
};

namespace std {
template <>
struct hash<MeshTileCoord> {
    size_t operator()(const MeshTileCoord& coord) const noexcept {
#if SIZE_MAX > UINT32_MAX
        constexpr size_t kGoldenRatio = 0x9e3779b97f4a7c15ull;
#else
        constexpr size_t kGoldenRatio = 0x9e3779b9u;
#endif
        size_t seed = hash<int32_t>{}(coord.x);
        seed ^= hash<int32_t>{}(coord.y) + kGoldenRatio + (seed << 6) + (seed >> 2);
        return seed;
    }
};

template <>
struct hash<TileLodCoord> {
    size_t operator()(const TileLodCoord& coord) const noexcept {
#if SIZE_MAX > UINT32_MAX
        constexpr size_t kGoldenRatio = 0x9e3779b97f4a7c15ull;
#else
        constexpr size_t kGoldenRatio = 0x9e3779b9u;
#endif
        size_t seed = hash<MeshTileCoord>{}(coord.tile);
        seed ^= hash<uint8_t>{}(coord.lodLevel) + kGoldenRatio + (seed << 6) + (seed >> 2);
        return seed;
    }
};

template <>
struct hash<TileLodCellCoord> {
    size_t operator()(const TileLodCellCoord& coord) const noexcept {
#if SIZE_MAX > UINT32_MAX
        constexpr size_t kGoldenRatio = 0x9e3779b97f4a7c15ull;
#else
        constexpr size_t kGoldenRatio = 0x9e3779b9u;
#endif
        size_t seed = hash<TileLodCoord>{}(coord.tileLod);
        seed ^= hash<uint16_t>{}(coord.cellX) + kGoldenRatio + (seed << 6) + (seed >> 2);
        seed ^= hash<uint16_t>{}(coord.cellY) + kGoldenRatio + (seed << 6) + (seed >> 2);
        return seed;
    }
};
}  // namespace std

class MeshManager {
public:
    struct Config {
        std::vector<int32_t> lodChunkRadii{4, 8, 16};
        float lodSseTargetPixels = 1.0f;
        float lodSseHysteresisPixels = 0.25f;
        float lodSseMinDepthBlocks = 4.0f;
        float lodSseFallbackProjectionScale = 390.0f;
        jobsystem::JobSystem::Config jobConfig{};
    };

    explicit MeshManager(const World& world);
    MeshManager(const World& world, Config config);
    ~MeshManager();

    MeshManager(const MeshManager&) = delete;
    MeshManager& operator=(const MeshManager&) = delete;
    MeshManager(MeshManager&&) = delete;
    MeshManager& operator=(MeshManager&&) = delete;

    void updatePlayerPosition(const glm::vec3& playerWorldPosition, float sseProjectionScale);

    std::vector<Meshlet> copyMeshlets() const;
    std::vector<Meshlet> copyMeshletsAround(const ColumnCoord& centerColumn, int32_t columnRadius) const;
    uint64_t meshRevision() const noexcept;
    bool hasPendingJobs() const;

private:
    struct CompletedTileCellResult {
        TileLodCellCoord coord;
        std::vector<Meshlet> meshlets;
    };

    struct MeshTileLodState {
        std::unordered_map<uint32_t, std::vector<Meshlet>> cellMeshes;
        int32_t expectedCellCount = 0;
    };

    struct MeshTileState {
        std::unordered_map<uint8_t, MeshTileLodState> lodStates;
        int8_t desiredLod = -1;
        int8_t renderedLod = -1;
    };

    struct MeshGenerationResult;

    void scheduleTilesAround(const ChunkCoord& centerChunk,
                             const glm::vec3& playerWorldPosition,
                             float sseProjectionScale,
                             const ChunkCoord* previousCenterChunk,
                             int32_t centerShiftChunks);
    void scheduleRemeshForNewColumns(const ColumnCoord& centerColumn);
    void scheduleTileLodMeshing(const TileLodCoord& coord,
                                jobsystem::Priority priority,
                                bool forceRemesh,
                                int32_t activeWindowExtraChunks);
    void scheduleTileLodCellMeshing(const TileLodCellCoord& coord,
                                    jobsystem::Priority priority,
                                    bool forceRemesh,
                                    int32_t activeWindowExtraChunks);
    void applyCompletedTileResultsBudgeted();

    void onTileLodCellMeshed(const TileLodCellCoord& coord, std::vector<Meshlet>&& meshlets);

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
    int8_t chooseRenderableLodForTileLocked(const MeshTileState& state) const;
    void refreshRenderedLodsLocked();

    std::vector<Meshlet> meshLodCell(const ChunkCoord& cellCoord, uint8_t lodLevel) const;

    int32_t cellSpanChunksForLod(uint8_t lodLevel) const;
    int32_t cellSpanLodCellsForLod(uint8_t lodLevel) const;
    int32_t cellCountPerAxisForLod(uint8_t lodLevel) const;
    uint32_t packCellKey(uint16_t cellX, uint16_t cellY) const;
    static bool tileInBounds(const MeshTileCoord& tileCoord,
                             int32_t minTileX,
                             int32_t maxTileX,
                             int32_t minTileY,
                             int32_t maxTileY);

    int32_t maxConfiguredRadius() const;

    static uint8_t chunkSpanForLod(uint8_t lodLevel);
    static int32_t chunkZCountForLod(uint8_t lodLevel);
    static jobsystem::Priority priorityFromLodLevel(uint8_t lodLevel);
    static void sanitizeConfig(Config& config);

    const World& world_;
    Config config_;
    jobsystem::JobSystem jobs_;
    int32_t meshTileSizeChunks_ = 1;

    mutable std::shared_mutex meshMutex_;
    std::unordered_set<ColumnCoord> knownGeneratedColumns_;
    std::unordered_set<TileLodCellCoord> pendingTileJobs_;
    std::unordered_set<TileLodCellCoord> deferredRemeshTileLods_;
    std::unordered_map<MeshTileCoord, std::vector<CompletedTileCellResult>> completedTileResultsByTile_;
    std::deque<MeshTileCoord> completedTileResultOrder_;
    std::unordered_set<MeshTileCoord> completedTileResultQueued_;
    std::unordered_map<MeshTileCoord, MeshTileState> meshTiles_;

    std::atomic<uint64_t> meshRevision_{0};
    std::atomic<uint64_t> processedWorldGenerationRevision_{0};
    std::atomic<bool> shuttingDown_{false};

    ChunkCoord lastScheduledCenterChunk_{0, 0, 0};
    bool hasLastScheduledCenter_ = false;

    ChunkCoord lodRefreshScanCenterChunk_{0, 0, 0};
    bool hasLodRefreshScanCenter_ = false;
    int32_t lodRefreshScanNextIndex_ = 0;
    glm::vec3 lastPlayerWorldPosition_{0.0f, 0.0f, 0.0f};
    float lastSseProjectionScale_ = 390.0f;
    bool hasLastSseProjectionScale_ = false;
};
