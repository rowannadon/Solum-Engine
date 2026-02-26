#include "solum_engine/voxel/VoxelStreamingSystem.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "solum_engine/voxel/MeshManager.h"
#include "solum_engine/voxel/World.h"

namespace {
struct PreparedMeshUploadData {
    std::vector<MeshletMetadataGPU> metadata;
    std::vector<uint32_t> quadData;
    std::vector<MeshletAabbGPU> meshletAabbsGpu;
    std::vector<MeshletAabb> meshletBounds;
    uint32_t totalMeshletCount = 0;
    uint32_t totalQuadCount = 0;
    uint32_t requiredMeshletCapacity = 64u;
    uint32_t requiredQuadCapacity = 64u * MESHLET_QUAD_CAPACITY * MESHLET_QUAD_DATA_WORD_STRIDE;
};

MeshletAabb computeMeshletAabb(const Meshlet& meshlet) {
    if (meshlet.quadCount == 0u) {
        const glm::vec3 origin = glm::vec3(meshlet.origin);
        return MeshletAabb{origin, origin};
    }

    static const std::array<std::array<glm::vec3, 4>, 6> kFaceCornerOffsets{{
        {{glm::vec3{1.0f, 0.0f, 0.0f}, glm::vec3{1.0f, 1.0f, 0.0f}, glm::vec3{1.0f, 0.0f, 1.0f}, glm::vec3{1.0f, 1.0f, 1.0f}}},
        {{glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{0.0f, 0.0f, 1.0f}, glm::vec3{0.0f, 1.0f, 0.0f}, glm::vec3{0.0f, 1.0f, 1.0f}}},
        {{glm::vec3{0.0f, 1.0f, 0.0f}, glm::vec3{0.0f, 1.0f, 1.0f}, glm::vec3{1.0f, 1.0f, 0.0f}, glm::vec3{1.0f, 1.0f, 1.0f}}},
        {{glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{1.0f, 0.0f, 0.0f}, glm::vec3{0.0f, 0.0f, 1.0f}, glm::vec3{1.0f, 0.0f, 1.0f}}},
        {{glm::vec3{0.0f, 0.0f, 1.0f}, glm::vec3{1.0f, 0.0f, 1.0f}, glm::vec3{0.0f, 1.0f, 1.0f}, glm::vec3{1.0f, 1.0f, 1.0f}}},
        {{glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{0.0f, 1.0f, 0.0f}, glm::vec3{1.0f, 0.0f, 0.0f}, glm::vec3{1.0f, 1.0f, 0.0f}}},
    }};

    const uint32_t safeFaceDirection = std::min(meshlet.faceDirection, 5u);
    const float voxelScale = static_cast<float>(std::max(meshlet.voxelScale, 1u));

    bool firstVertex = true;
    glm::vec3 minCorner{0.0f};
    glm::vec3 maxCorner{0.0f};

    for (uint32_t quadIndex = 0; quadIndex < meshlet.quadCount; ++quadIndex) {
        const glm::uvec3 local = unpackMeshletLocalOffset(meshlet.packedQuadLocalOffsets[quadIndex]);
        const glm::vec3 quadBase = glm::vec3(meshlet.origin) + (glm::vec3(local) * voxelScale);
        for (const glm::vec3& cornerOffset : kFaceCornerOffsets[safeFaceDirection]) {
            const glm::vec3 vertex = quadBase + (cornerOffset * voxelScale);
            if (firstVertex) {
                minCorner = vertex;
                maxCorner = vertex;
                firstVertex = false;
                continue;
            }
            minCorner = glm::min(minCorner, vertex);
            maxCorner = glm::max(maxCorner, vertex);
        }
    }

    return MeshletAabb{minCorner, maxCorner};
}

MeshletAabbGPU toGpuAabb(const MeshletAabb& aabb) {
    return MeshletAabbGPU{
        glm::vec4(aabb.minCorner, 0.0f),
        glm::vec4(aabb.maxCorner, 0.0f)
    };
}

PreparedMeshUploadData prepareMeshUploadData(const std::vector<Meshlet>& meshlets) {
    PreparedMeshUploadData prepared;

    for (const Meshlet& meshlet : meshlets) {
        if (meshlet.quadCount == 0) {
            continue;
        }
        ++prepared.totalMeshletCount;
        prepared.totalQuadCount += meshlet.quadCount * MESHLET_QUAD_DATA_WORD_STRIDE;
    }

    prepared.metadata.reserve(prepared.totalMeshletCount);
    prepared.quadData.reserve(prepared.totalQuadCount);
    prepared.meshletAabbsGpu.reserve(prepared.totalMeshletCount);
    prepared.meshletBounds.reserve(prepared.totalMeshletCount);

    for (const Meshlet& meshlet : meshlets) {
        if (meshlet.quadCount == 0) {
            continue;
        }

        MeshletMetadataGPU metadata{};
        metadata.originX = meshlet.origin.x;
        metadata.originY = meshlet.origin.y;
        metadata.originZ = meshlet.origin.z;
        metadata.quadCount = meshlet.quadCount;
        metadata.faceDirection = meshlet.faceDirection;
        metadata.dataOffset = static_cast<uint32_t>(prepared.quadData.size());
        metadata.voxelScale = std::max(meshlet.voxelScale, 1u);
        prepared.metadata.push_back(metadata);
        const MeshletAabb meshletBounds = computeMeshletAabb(meshlet);
        prepared.meshletAabbsGpu.push_back(toGpuAabb(meshletBounds));
        prepared.meshletBounds.push_back(meshletBounds);

        for (uint32_t i = 0; i < meshlet.quadCount; ++i) {
            prepared.quadData.push_back(packMeshletQuadData(
                meshlet.packedQuadLocalOffsets[i],
                meshlet.quadMaterialIds[i]
            ));
            prepared.quadData.push_back(static_cast<uint32_t>(meshlet.quadAoData[i]));
        }
    }

    prepared.requiredMeshletCapacity = std::max(prepared.totalMeshletCount + 16u, 64u);
    prepared.requiredQuadCapacity = std::max(
        prepared.totalQuadCount + (1024u * MESHLET_QUAD_DATA_WORD_STRIDE),
        prepared.requiredMeshletCapacity * MESHLET_QUAD_CAPACITY * MESHLET_QUAD_DATA_WORD_STRIDE
    );

    return prepared;
}
}  // namespace

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
        pendingMeshUpload_.reset();
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
        pendingMeshUpload_.reset();
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

std::optional<StreamingMeshUpload> VoxelStreamingSystem::consumePendingMeshUpload() {
    std::lock_guard<std::mutex> lock(streamingMutex_);
    if (!pendingMeshUpload_.has_value()) {
        return std::nullopt;
    }
    std::optional<StreamingMeshUpload> upload = std::move(pendingMeshUpload_);
    pendingMeshUpload_.reset();
    return upload;
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
        {
            std::lock_guard<std::mutex> lock(streamingMutex_);
            if (pendingMeshUpload_.has_value()) {
                streamSkipThrottle_.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
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
        PreparedMeshUploadData prepared = prepareMeshUploadData(meshlets);
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
            pendingMeshUpload_ = StreamingMeshUpload{
                std::move(prepared.metadata),
                std::move(prepared.quadData),
                std::move(prepared.meshletAabbsGpu),
                std::move(prepared.meshletBounds),
                prepared.totalMeshletCount,
                prepared.totalQuadCount,
                prepared.requiredMeshletCapacity,
                prepared.requiredQuadCapacity,
                currentRevision,
                centerColumn
            };
        }

        streamerLastPreparedRevision_ = currentRevision;
        streamerLastPreparedCenter_ = centerColumn;
        streamerHasLastPreparedCenter_ = true;
        streamerLastSnapshotTime_ = now;
        streamSnapshotsPrepared_.fetch_add(1, std::memory_order_relaxed);
    }
}

void VoxelStreamingSystem::recordTimingNs(TimingStage stage, uint64_t ns) noexcept {
    const std::size_t stageIndex = static_cast<std::size_t>(stage);
    TimingAccumulator& accumulator = timingAccumulators_[stageIndex];
    accumulator.totalNs.fetch_add(ns, std::memory_order_relaxed);
    accumulator.callCount.fetch_add(1, std::memory_order_relaxed);

    uint64_t observedMax = accumulator.maxNs.load(std::memory_order_relaxed);
    while (ns > observedMax &&
           !accumulator.maxNs.compare_exchange_weak(
               observedMax,
               ns,
               std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
}

VoxelStreamingSystem::TimingRawTotals VoxelStreamingSystem::captureTimingRawTotals() const {
    TimingRawTotals totals;
    for (std::size_t i = 0; i < static_cast<std::size_t>(TimingStage::Count); ++i) {
        const TimingAccumulator& accumulator = timingAccumulators_[i];
        totals.totalNs[i] = accumulator.totalNs.load(std::memory_order_relaxed);
        totals.callCount[i] = accumulator.callCount.load(std::memory_order_relaxed);
        totals.maxNs[i] = accumulator.maxNs.load(std::memory_order_relaxed);
    }

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
    const uint64_t deltaNs = current.totalNs[i] - previous.totalNs[i];
    const uint64_t deltaCalls = current.callCount[i] - previous.callCount[i];
    const double deltaMs = static_cast<double>(deltaNs) / 1'000'000.0;
    const double window = std::max(sampleWindowSeconds, 1e-6);

    TimingStageSnapshot snapshot;
    snapshot.averageMs = (deltaCalls > 0) ? (deltaMs / static_cast<double>(deltaCalls)) : 0.0;
    snapshot.peakMs = static_cast<double>(current.maxNs[i]) / 1'000'000.0;
    snapshot.totalMsPerSecond = deltaMs / window;
    snapshot.callsPerSecond = static_cast<double>(deltaCalls) / window;
    snapshot.totalCalls = current.callCount[i];
    return snapshot;
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
            pendingMeshUpload_.has_value() ||
            mainUploadInProgress_.load(std::memory_order_relaxed);
    }
    return snapshot;
}
