#include "solum_engine/voxel/VoxelStreamingSystem.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "solum_engine/render/MeshUploadAssembler.h"
#include "solum_engine/voxel/MeshManager.h"
#include "solum_engine/voxel/World.h"

VoxelStreamingSystem::VoxelStreamingSystem() = default;

VoxelStreamingSystem::~VoxelStreamingSystem() {
    stop();
}

bool VoxelStreamingSystem::initialize() {
    World::Config worldConfig;
    worldConfig.columnLoadRadius = 512;
    worldConfig.jobConfig.worker_threads = 4;

    MeshManager::Config meshConfig;
    meshConfig.lodChunkRadii = {16, 48, 96, 128};
    meshConfig.jobConfig.worker_threads = worldConfig.jobConfig.worker_threads;
    const int32_t clampedWorldRadius = std::max(1, worldConfig.columnLoadRadius);
    for (int32_t& lodRadius : meshConfig.lodChunkRadii) {
        lodRadius = std::min(lodRadius, clampedWorldRadius);
    }
    std::sort(meshConfig.lodChunkRadii.begin(), meshConfig.lodChunkRadii.end());
    meshConfig.lodChunkRadii.erase(
        std::unique(meshConfig.lodChunkRadii.begin(), meshConfig.lodChunkRadii.end()),
        meshConfig.lodChunkRadii.end()
    );
    if (meshConfig.lodChunkRadii.empty()) {
        meshConfig.lodChunkRadii.push_back(clampedWorldRadius);
    }

    world_ = std::make_unique<World>(worldConfig);
    meshManager_ = std::make_unique<MeshManager>(*world_, meshConfig);
    uploadColumnRadius_ = std::min(
        clampedWorldRadius,
        std::max(1, meshConfig.lodChunkRadii.back() + 1)
    );

    return world_ && meshManager_;
}

void VoxelStreamingSystem::start(const glm::vec3& initialCameraPosition, uint64_t initialUploadedMeshRevision) {
    stop();

    const BlockCoord initialBlock{
        static_cast<int32_t>(std::floor(initialCameraPosition.x)),
        static_cast<int32_t>(std::floor(initialCameraPosition.y)),
        static_cast<int32_t>(std::floor(initialCameraPosition.z))
    };
    const ColumnCoord initialCenterColumn = chunk_to_column(block_to_chunk(initialBlock));

    {
        std::lock_guard<std::mutex> lock(streamingMutex_);
        streamingStopRequested_ = false;
        hasLatestStreamingCamera_ = true;
        latestStreamingCamera_ = initialCameraPosition;
        latestStreamingSseProjectionScale_ = 390.0f;
        uploadMailbox_.clear();
        streamerLastPreparedRevision_ = initialUploadedMeshRevision;
        streamerLastPreparedCenter_ = initialCenterColumn;
        streamerHasLastPreparedCenter_ = true;
        streamerLastSnapshotTime_.reset();
        mainUploadInProgress_.store(false, std::memory_order_relaxed);
    }

    streamingThread_ = std::thread([this] {
        streamingThreadMain();
    });
}

void VoxelStreamingSystem::stop() {
    {
        std::lock_guard<std::mutex> lock(streamingMutex_);
        streamingStopRequested_ = true;
        hasLatestStreamingCamera_ = false;
    }
    streamingCv_.notify_all();

    if (streamingThread_.joinable()) {
        streamingThread_.join();
    }

    {
        std::lock_guard<std::mutex> lock(streamingMutex_);
        streamingStopRequested_ = false;
        uploadMailbox_.clear();
        streamerLastSnapshotTime_.reset();
    }
    mainUploadInProgress_.store(false, std::memory_order_relaxed);
}

void VoxelStreamingSystem::setMainUploadInProgress(bool inProgress) noexcept {
    mainUploadInProgress_.store(inProgress, std::memory_order_relaxed);
}

void VoxelStreamingSystem::updateCamera(const glm::vec3& cameraPosition, float sseProjectionScale) {
    {
        std::lock_guard<std::mutex> lock(streamingMutex_);
        hasLatestStreamingCamera_ = true;
        latestStreamingCamera_ = cameraPosition;
        latestStreamingSseProjectionScale_ = sseProjectionScale;
    }
    streamingCv_.notify_one();
}

std::optional<StreamingMeshUpload> VoxelStreamingSystem::tryConsumePreparedUpload() {
    return uploadMailbox_.tryConsume();
}

void VoxelStreamingSystem::recordMainUpdateDurationNs(uint64_t ns) noexcept {
    recordTimingNs(TimingStage::MainUpdateWorldStreaming, ns);
}

const World* VoxelStreamingSystem::world() const noexcept {
    return world_.get();
}

int32_t VoxelStreamingSystem::cameraColumnChebyshevDistance(const ColumnCoord& a, const ColumnCoord& b) {
    const int32_t dx = std::abs(a.v.x - b.v.x);
    const int32_t dy = std::abs(a.v.y - b.v.y);
    return std::max(dx, dy);
}

void VoxelStreamingSystem::streamingThreadMain() {
    glm::vec3 cameraPosition{0.0f, 0.0f, 0.0f};
    float cameraSseProjectionScale = 390.0f;
    bool hasCameraPosition = false;

    while (true) {
        {
            const auto waitStart = std::chrono::steady_clock::now();
            std::unique_lock<std::mutex> lock(streamingMutex_);
            streamingCv_.wait_for(lock, std::chrono::milliseconds(16), [this] {
                return streamingStopRequested_ || hasLatestStreamingCamera_;
            });
            const uint64_t waitNs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - waitStart
            ).count());
            recordTimingNs(TimingStage::StreamWait, waitNs);
            if (streamingStopRequested_) {
                return;
            }
            if (hasLatestStreamingCamera_) {
                cameraPosition = latestStreamingCamera_;
                cameraSseProjectionScale = latestStreamingSseProjectionScale_;
                hasLatestStreamingCamera_ = false;
                hasCameraPosition = true;
            }
        }

        if (!hasCameraPosition || !world_ || !meshManager_) {
            streamSkipNoCamera_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        const auto worldUpdateStart = std::chrono::steady_clock::now();
        world_->updatePlayerPosition(cameraPosition);
        recordTimingNs(
            TimingStage::StreamWorldUpdate,
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - worldUpdateStart
            ).count())
        );

        const auto meshUpdateStart = std::chrono::steady_clock::now();
        meshManager_->updatePlayerPosition(cameraPosition, cameraSseProjectionScale);
        recordTimingNs(
            TimingStage::StreamMeshUpdate,
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - meshUpdateStart
            ).count())
        );

        const BlockCoord cameraBlock{
            static_cast<int32_t>(std::floor(cameraPosition.x)),
            static_cast<int32_t>(std::floor(cameraPosition.y)),
            static_cast<int32_t>(std::floor(cameraPosition.z))
        };
        const ColumnCoord centerColumn = chunk_to_column(block_to_chunk(cameraBlock));
        const bool centerChanged = !streamerHasLastPreparedCenter_ || !(centerColumn == streamerLastPreparedCenter_);
        const int32_t centerShift = streamerHasLastPreparedCenter_
            ? cameraColumnChebyshevDistance(centerColumn, streamerLastPreparedCenter_)
            : 0;
        const int32_t centerUploadStrideChunks = std::max(2, uploadColumnRadius_ / 8);

        const uint64_t currentRevision = meshManager_->meshRevision();
        if (currentRevision == streamerLastPreparedRevision_ &&
            (!centerChanged || centerShift < centerUploadStrideChunks)) {
            streamSkipUnchanged_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        if (mainUploadInProgress_.load(std::memory_order_relaxed)) {
            streamSkipThrottle_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        if (uploadMailbox_.hasPending()) {
            streamSkipThrottle_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        const bool pendingJobs = world_->hasPendingJobs() || meshManager_->hasPendingJobs();
        const double minSnapshotIntervalSeconds =
            pendingJobs ? 0.0 :
            (uploadColumnRadius_ >= 8) ? 0.35 :
            (uploadColumnRadius_ >= 4) ? 0.25 :
            0.15;
        const auto now = std::chrono::steady_clock::now();
        const bool intervalElapsed =
            !streamerLastSnapshotTime_.has_value() ||
            std::chrono::duration<double>(now - *streamerLastSnapshotTime_).count() >= minSnapshotIntervalSeconds;
        const bool forceForCenterChange = centerChanged && centerShift >= centerUploadStrideChunks;

        if (pendingJobs && !intervalElapsed && !forceForCenterChange) {
            streamSkipThrottle_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        const auto copyStart = std::chrono::steady_clock::now();
        std::vector<Meshlet> meshlets = meshManager_->copyMeshletsAround(centerColumn, uploadColumnRadius_);
        recordTimingNs(
            TimingStage::StreamCopyMeshlets,
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - copyStart
            ).count())
        );

        const auto prepareStart = std::chrono::steady_clock::now();
        StreamingMeshUpload preparedUpload = MeshUploadAssembler::assemble(
            meshlets,
            currentRevision,
            centerColumn
        );
        recordTimingNs(
            TimingStage::StreamPrepareUpload,
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - prepareStart
            ).count())
        );

        {
            std::lock_guard<std::mutex> lock(streamingMutex_);
            if (streamingStopRequested_) {
                return;
            }
        }
        uploadMailbox_.pushLatest(std::move(preparedUpload));

        streamerLastPreparedRevision_ = currentRevision;
        streamerLastPreparedCenter_ = centerColumn;
        streamerHasLastPreparedCenter_ = true;
        streamerLastSnapshotTime_ = now;
        streamSnapshotsPrepared_.fetch_add(1, std::memory_order_relaxed);
    }
}

void VoxelStreamingSystem::recordTimingNs(TimingStage stage, uint64_t ns) noexcept {
    const std::size_t stageIndex = static_cast<std::size_t>(stage);
    recordTimingAccumulator(timingAccumulators_[stageIndex], ns);
}

VoxelStreamingSystem::TimingRawTotals VoxelStreamingSystem::captureTimingRawTotals() const {
    TimingRawTotals totals;
    captureTimingAccumulatorArrays(timingAccumulators_, totals.totalNs, totals.callCount, totals.maxNs);

    totals.streamSkipNoCamera = streamSkipNoCamera_.load(std::memory_order_relaxed);
    totals.streamSkipUnchanged = streamSkipUnchanged_.load(std::memory_order_relaxed);
    totals.streamSkipThrottle = streamSkipThrottle_.load(std::memory_order_relaxed);
    totals.streamSnapshotsPrepared = streamSnapshotsPrepared_.load(std::memory_order_relaxed);
    return totals;
}

TimingStageSnapshot VoxelStreamingSystem::makeStageSnapshot(const TimingRawTotals& current,
                                                            const TimingRawTotals& previous,
                                                            TimingStage stage,
                                                            double sampleWindowSeconds) {
    const std::size_t i = static_cast<std::size_t>(stage);
    return makeTimingStageSnapshotFromRaw(
        current.totalNs[i],
        current.callCount[i],
        current.maxNs[i],
        previous.totalNs[i],
        previous.callCount[i],
        sampleWindowSeconds
    );
}

RuntimeTimingSnapshot VoxelStreamingSystem::getRuntimeTimingSnapshot() {
    RuntimeTimingSnapshot snapshot;
    const TimingRawTotals currentTotals = captureTimingRawTotals();
    const auto now = std::chrono::steady_clock::now();

    {
        std::lock_guard<std::mutex> lock(timingSnapshotMutex_);
        if (!lastTimingSampleTime_.has_value()) {
            lastTimingSampleTime_ = now;
            lastTimingRawTotals_ = currentTotals;
        } else {
            const double sampleWindowSeconds = std::chrono::duration<double>(now - *lastTimingSampleTime_).count();
            snapshot.sampleWindowSeconds = sampleWindowSeconds;
            snapshot.mainUpdateWorldStreaming = makeStageSnapshot(
                currentTotals,
                lastTimingRawTotals_,
                TimingStage::MainUpdateWorldStreaming,
                sampleWindowSeconds
            );
            snapshot.streamWait = makeStageSnapshot(
                currentTotals,
                lastTimingRawTotals_,
                TimingStage::StreamWait,
                sampleWindowSeconds
            );
            snapshot.streamWorldUpdate = makeStageSnapshot(
                currentTotals,
                lastTimingRawTotals_,
                TimingStage::StreamWorldUpdate,
                sampleWindowSeconds
            );
            snapshot.streamMeshUpdate = makeStageSnapshot(
                currentTotals,
                lastTimingRawTotals_,
                TimingStage::StreamMeshUpdate,
                sampleWindowSeconds
            );
            snapshot.streamCopyMeshlets = makeStageSnapshot(
                currentTotals,
                lastTimingRawTotals_,
                TimingStage::StreamCopyMeshlets,
                sampleWindowSeconds
            );
            snapshot.streamPrepareUpload = makeStageSnapshot(
                currentTotals,
                lastTimingRawTotals_,
                TimingStage::StreamPrepareUpload,
                sampleWindowSeconds
            );

            snapshot.streamSkipNoCamera =
                currentTotals.streamSkipNoCamera - lastTimingRawTotals_.streamSkipNoCamera;
            snapshot.streamSkipUnchanged =
                currentTotals.streamSkipUnchanged - lastTimingRawTotals_.streamSkipUnchanged;
            snapshot.streamSkipThrottle =
                currentTotals.streamSkipThrottle - lastTimingRawTotals_.streamSkipThrottle;
            snapshot.streamSnapshotsPrepared =
                currentTotals.streamSnapshotsPrepared - lastTimingRawTotals_.streamSnapshotsPrepared;

            lastTimingSampleTime_ = now;
            lastTimingRawTotals_ = currentTotals;
        }
    }

    snapshot.worldHasPendingJobs = world_ && world_->hasPendingJobs();
    snapshot.meshHasPendingJobs = meshManager_ && meshManager_->hasPendingJobs();
    {
        std::lock_guard<std::mutex> lock(streamingMutex_);
        snapshot.pendingUploadQueued =
            uploadMailbox_.hasPending() ||
            mainUploadInProgress_.load(std::memory_order_relaxed);
    }
    return snapshot;
}
