#include "solum_engine/render/RegionBuildQueue.h"

#include <chrono>
#include <exception>
#include <iostream>
#include <utility>

bool RegionBuildQueue::isBuildQueued(const RegionCoord& coord, int lodLevel) const {
    if (buildInFlight_ && activeBuildLodLevel_ == lodLevel && activeBuildCoord_ == coord) {
        return true;
    }

    for (const PendingBuild& pending : pendingBuilds_) {
        if (pending.lodLevel == lodLevel && pending.coord == coord) {
            return true;
        }
    }
    return false;
}

void RegionBuildQueue::enqueue(PendingBuild build) {
    pendingBuilds_.push_back(build);
}

void RegionBuildQueue::clearPending() {
    pendingBuilds_.clear();
}

void RegionBuildQueue::process(
    const std::function<bool(const PendingBuild&)>& shouldStartBuild,
    const std::function<MeshData(RegionCoord, int)>& buildMesh,
    const std::function<void(CompletedBuild&&)>& applyCompleted
) {
    if (buildInFlight_ && activeBuildFuture_.valid()) {
        if (activeBuildFuture_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            try {
                CompletedBuild completed = activeBuildFuture_.get();
                buildInFlight_ = false;
                applyCompleted(std::move(completed));
                activeBuildLodLevel_ = -1;
            } catch (const std::exception& ex) {
                std::cerr << "Background region build failed: " << ex.what() << std::endl;
                buildInFlight_ = false;
                activeBuildLodLevel_ = -1;
            } catch (...) {
                std::cerr << "Background region build failed with unknown error." << std::endl;
                buildInFlight_ = false;
                activeBuildLodLevel_ = -1;
            }
        }
    }

    if (buildInFlight_) {
        return;
    }

    while (!pendingBuilds_.empty()) {
        const PendingBuild build = pendingBuilds_.front();
        pendingBuilds_.pop_front();

        if (!shouldStartBuild(build)) {
            continue;
        }

        const RegionCoord coord = build.coord;
        const int lodLevel = build.lodLevel;
        activeBuildCoord_ = coord;
        activeBuildLodLevel_ = lodLevel;
        const auto buildMeshCopy = buildMesh;
        activeBuildFuture_ = std::async(std::launch::async, [buildMeshCopy, coord, lodLevel]() {
            CompletedBuild completed;
            completed.coord = coord;
            completed.lodLevel = lodLevel;
            completed.meshData = buildMeshCopy(coord, lodLevel);
            return completed;
        });
        buildInFlight_ = true;
        break;
    }
}

void RegionBuildQueue::waitForInFlight() {
    if (activeBuildFuture_.valid()) {
        try {
            activeBuildFuture_.wait();
            (void)activeBuildFuture_.get();
        } catch (...) {
        }
    }
    buildInFlight_ = false;
    activeBuildLodLevel_ = -1;
}
