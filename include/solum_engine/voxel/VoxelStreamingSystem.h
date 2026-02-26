#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

#include <glm/glm.hpp>

#include "solum_engine/render/RuntimeTiming.h"
#include "solum_engine/resources/Coords.h"
#include "solum_engine/voxel/StreamingUpload.h"

class MeshManager;
class World;

class VoxelStreamingSystem {
private:
    enum class TimingStage : std::size_t {
        MainUpdateWorldStreaming = 0,
        StreamWait,
        StreamWorldUpdate,
        StreamMeshUpdate,
        StreamCopyMeshlets,
        StreamPrepareUpload,
        Count
    };

    struct TimingAccumulator {
        std::atomic<uint64_t> totalNs{0};
        std::atomic<uint64_t> callCount{0};
        std::atomic<uint64_t> maxNs{0};
    };

    struct TimingRawTotals {
        std::array<uint64_t, static_cast<std::size_t>(TimingStage::Count)> totalNs{};
        std::array<uint64_t, static_cast<std::size_t>(TimingStage::Count)> callCount{};
        std::array<uint64_t, static_cast<std::size_t>(TimingStage::Count)> maxNs{};
        uint64_t streamSkipNoCamera = 0;
        uint64_t streamSkipUnchanged = 0;
        uint64_t streamSkipThrottle = 0;
        uint64_t streamSnapshotsPrepared = 0;
    };

    std::unique_ptr<World> world_;
    std::unique_ptr<MeshManager> meshManager_;
    int32_t uploadColumnRadius_ = 1;

    std::thread streamingThread_;
    mutable std::mutex streamingMutex_;
    std::condition_variable streamingCv_;
    bool streamingStopRequested_ = false;
    bool hasLatestStreamingCamera_ = false;
    glm::vec3 latestStreamingCamera_{0.0f, 0.0f, 0.0f};
    float latestStreamingSseProjectionScale_ = 390.0f;
    std::optional<StreamingMeshUpload> pendingMeshUpload_;
    uint64_t streamerLastPreparedRevision_ = 0;
    ColumnCoord streamerLastPreparedCenter_{0, 0};
    bool streamerHasLastPreparedCenter_ = false;
    std::optional<std::chrono::steady_clock::time_point> streamerLastSnapshotTime_;
    std::atomic<bool> mainUploadInProgress_{false};

    std::array<TimingAccumulator, static_cast<std::size_t>(TimingStage::Count)> timingAccumulators_{};
    std::atomic<uint64_t> streamSkipNoCamera_{0};
    std::atomic<uint64_t> streamSkipUnchanged_{0};
    std::atomic<uint64_t> streamSkipThrottle_{0};
    std::atomic<uint64_t> streamSnapshotsPrepared_{0};
    std::mutex timingSnapshotMutex_;
    TimingRawTotals lastTimingRawTotals_{};
    std::optional<std::chrono::steady_clock::time_point> lastTimingSampleTime_;

    void streamingThreadMain();
    static int32_t cameraColumnChebyshevDistance(const ColumnCoord& a, const ColumnCoord& b);

    void recordTimingNs(TimingStage stage, uint64_t ns) noexcept;
    TimingRawTotals captureTimingRawTotals() const;
    static TimingStageSnapshot makeStageSnapshot(const TimingRawTotals& current,
                                                 const TimingRawTotals& previous,
                                                 TimingStage stage,
                                                 double sampleWindowSeconds);

public:
    VoxelStreamingSystem();
    ~VoxelStreamingSystem();

    bool initialize();
    void start(const glm::vec3& initialCameraPosition, uint64_t initialUploadedMeshRevision);
    void stop();

    void setMainUploadInProgress(bool inProgress) noexcept;
    void updateCamera(const glm::vec3& cameraPosition, float sseProjectionScale);
    std::optional<StreamingMeshUpload> consumePendingMeshUpload();
    void recordMainUpdateDurationNs(uint64_t ns) noexcept;

    RuntimeTimingSnapshot getRuntimeTimingSnapshot();
    const World* world() const noexcept;
};
