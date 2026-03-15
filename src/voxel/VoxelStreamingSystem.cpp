#include "solum_engine/voxel/VoxelStreamingSystem.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "solum_engine/voxel/MeshManager.h"
#include "solum_engine/voxel/World.h"

VoxelStreamingSystem::VoxelStreamingSystem() = default;

VoxelStreamingSystem::~VoxelStreamingSystem() {
    stop();
}

bool VoxelStreamingSystem::initialize(std::shared_ptr<const BlockModelLibrary> blockModelLibrary) {
    blockModelLibrary_ = std::move(blockModelLibrary);

    World::Config worldConfig;
    worldConfig.columnLoadRadius = 32;
    worldConfig.jobConfig.worker_threads = 4;

    MeshManager::Config meshConfig;
    meshConfig.meshTileSizeChunks = 2;
    meshConfig.meshTileHeightChunks = 2;
    meshConfig.lodLevelCount = 4;
    meshConfig.activeChunkRadius = 32;
    meshConfig.lodSseTargetPixels = 8.0f;
    meshConfig.jobConfig.worker_threads = worldConfig.jobConfig.worker_threads;
    const int32_t clampedWorldRadius = std::max(1, worldConfig.columnLoadRadius);
    meshConfig.activeChunkRadius = std::min(meshConfig.activeChunkRadius, clampedWorldRadius);

    world_ = std::make_unique<World>(worldConfig);
    meshManager_ = std::make_unique<MeshManager>(*world_, meshConfig, blockModelLibrary_);

    return world_ && meshManager_;
}

void VoxelStreamingSystem::start(const glm::vec3& initialCameraPosition, uint64_t initialUploadedMeshRevision) {
    stop();

    {
        std::lock_guard<std::mutex> lock(streamingMutex_);
        streamingStopRequested_ = false;
        uploadMailbox_.clear();
        streamerLastDeltaRevision_ = initialUploadedMeshRevision;
        requestStreamingWorkLocked(true);
        latestStreamingCamera_ = initialCameraPosition;
        latestStreamingSseProjectionScale_ = 390.0f;
    }

    streamingThread_ = std::thread([this] {
        streamingThreadMain();
    });
    streamingCv_.notify_one();
}

void VoxelStreamingSystem::stop() {
    {
        std::lock_guard<std::mutex> lock(streamingMutex_);
        streamingStopRequested_ = true;
        streamingWorkRequested_ = false;
        hasLatestStreamingCamera_ = false;
    }
    streamingCv_.notify_all();

    if (streamingThread_.joinable()) {
        streamingThread_.join();
    }

    {
        std::lock_guard<std::mutex> lock(streamingMutex_);
        streamingStopRequested_ = false;
        streamingWorkRequested_ = false;
        uploadMailbox_.clear();
    }
}

void VoxelStreamingSystem::updateCamera(const glm::vec3& cameraPosition, float sseProjectionScale) {
    {
        std::lock_guard<std::mutex> lock(streamingMutex_);
        latestStreamingCamera_ = cameraPosition;
        latestStreamingSseProjectionScale_ = sseProjectionScale;
        requestStreamingWorkLocked(true);
    }
    streamingCv_.notify_one();
}

bool VoxelStreamingSystem::breakBlock(const BlockCoord& coord) {
    if (!world_) {
        return false;
    }

    const bool changed = world_->breakBlock(coord);
    if (changed) {
        {
            std::lock_guard<std::mutex> lock(streamingMutex_);
            requestStreamingWorkLocked(false);
        }
        streamingCv_.notify_one();
    }
    return changed;
}

bool VoxelStreamingSystem::placeBlock(const BlockCoord& coord, const BlockMaterial& block) {
    if (!world_) {
        return false;
    }

    const bool changed = world_->placeBlock(coord, block);
    if (changed) {
        {
            std::lock_guard<std::mutex> lock(streamingMutex_);
            requestStreamingWorkLocked(false);
        }
        streamingCv_.notify_one();
    }
    return changed;
}

std::optional<MeshStreamingDelta> VoxelStreamingSystem::tryConsumePreparedDelta() {
    return uploadMailbox_.tryConsume();
}

void VoxelStreamingSystem::recordMainUpdateDurationNs(uint64_t ns) noexcept {
    recordTimingNs(TimingStage::MainUpdateWorldStreaming, ns);
}

const World* VoxelStreamingSystem::world() const noexcept {
    return world_.get();
}

void VoxelStreamingSystem::requestStreamingWorkLocked(bool cameraUpdated) {
    streamingWorkRequested_ = true;
    if (cameraUpdated) {
        hasLatestStreamingCamera_ = true;
    }
}

bool VoxelStreamingSystem::waitForWork(StreamingLoopState& state) {
    const auto waitStart = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(streamingMutex_);
    streamingCv_.wait_for(lock, std::chrono::milliseconds(16), [this] {
        return streamingStopRequested_ || hasLatestStreamingCamera_ || streamingWorkRequested_;
    });
    const uint64_t waitNs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - waitStart
    ).count());
    recordTimingNs(TimingStage::StreamWait, waitNs);

    if (streamingStopRequested_) {
        return false;
    }

    if (hasLatestStreamingCamera_) {
        state.cameraPosition = latestStreamingCamera_;
        state.cameraSseProjectionScale = latestStreamingSseProjectionScale_;
        state.hasCameraPosition = true;
        hasLatestStreamingCamera_ = false;
    }

    streamingWorkRequested_ = false;
    return true;
}

void VoxelStreamingSystem::runWorldStep(const StreamingLoopState& state) {
    const auto worldUpdateStart = std::chrono::steady_clock::now();
    world_->updatePlayerPosition(state.cameraPosition);
    recordTimingNs(
        TimingStage::StreamWorldUpdate,
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - worldUpdateStart
        ).count())
    );
}

void VoxelStreamingSystem::runMeshStep(const StreamingLoopState& state) {
    const auto meshUpdateStart = std::chrono::steady_clock::now();
    meshManager_->updatePlayerPosition(state.cameraPosition, state.cameraSseProjectionScale);
    recordTimingNs(
        TimingStage::StreamMeshUpdate,
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - meshUpdateStart
        ).count())
    );
}

std::optional<MeshStreamingDelta> VoxelStreamingSystem::buildDelta() {
    const auto copyStart = std::chrono::steady_clock::now();
    constexpr std::size_t kMaxDeltaEntriesPerTick = 64u;
    std::vector<MeshTileLodUpload> upserts = meshManager_->consumePendingTileLodUploads(kMaxDeltaEntriesPerTick);
    std::vector<MeshTileLodKey> removals = meshManager_->consumePendingTileLodRemovals(kMaxDeltaEntriesPerTick);
    recordTimingNs(
        TimingStage::StreamCopyMeshlets,
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - copyStart
        ).count())
    );

    const auto prepareStart = std::chrono::steady_clock::now();
    std::vector<MeshTileSelectionEntry> selectionChanges;
    const bool hasSelectionChanges = meshManager_->consumeSelectionChanges(selectionChanges);

    if (upserts.empty() && removals.empty() && !hasSelectionChanges) {
        streamSkipUnchanged_.fetch_add(1, std::memory_order_relaxed);
        recordTimingNs(
            TimingStage::StreamPrepareUpload,
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - prepareStart
            ).count())
        );
        return std::nullopt;
    }

    MeshStreamingDelta delta{};
    delta.upserts = std::move(upserts);
    delta.removals = std::move(removals);
    if (hasSelectionChanges) {
        delta.selectionChanges = std::move(selectionChanges);
    }
    delta.revision = streamerLastDeltaRevision_ + 1u;

    recordTimingNs(
        TimingStage::StreamPrepareUpload,
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - prepareStart
        ).count())
    );
    return delta;
}

void VoxelStreamingSystem::streamingThreadMain() {
    StreamingLoopState state;
    while (true) {
        if (!waitForWork(state)) {
            return;
        }

        if (!state.hasCameraPosition || !world_ || !meshManager_) {
            streamSkipNoCamera_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        runWorldStep(state);
        runMeshStep(state);

        std::optional<MeshStreamingDelta> delta = buildDelta();
        if (!delta.has_value()) {
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(streamingMutex_);
            if (streamingStopRequested_) {
                return;
            }
        }

        streamerLastDeltaRevision_ = delta->revision;
        uploadMailbox_.pushLatest(std::move(*delta));
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
    totals.streamSkipThrottle = 0u;
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
        snapshot.pendingUploadQueued = uploadMailbox_.hasPending();
    }
    return snapshot;
}
