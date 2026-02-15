#pragma once

#include "solum_engine/resources/Coords.h"
#include "solum_engine/voxel/ChunkMeshes.h"

#include <deque>
#include <functional>
#include <future>

class RegionBuildQueue {
public:
    struct PendingBuild {
        RegionCoord coord;
        int lodLevel = 0;
    };

    struct CompletedBuild {
        RegionCoord coord;
        int lodLevel = 0;
        MeshData meshData;
    };

    bool isBuildQueued(const RegionCoord& coord, int lodLevel) const;
    void enqueue(PendingBuild build);
    void clearPending();

    void process(
        const std::function<bool(const PendingBuild&)>& shouldStartBuild,
        const std::function<MeshData(RegionCoord, int)>& buildMesh,
        const std::function<void(CompletedBuild&&)>& applyCompleted
    );

    void waitForInFlight();

private:
    std::deque<PendingBuild> pendingBuilds_;
    std::future<CompletedBuild> activeBuildFuture_;
    bool buildInFlight_ = false;
    RegionCoord activeBuildCoord_{0, 0};
    int activeBuildLodLevel_ = -1;
};
