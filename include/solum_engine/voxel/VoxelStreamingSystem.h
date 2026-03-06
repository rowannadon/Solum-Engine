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
#include "solum_engine/render/TimingAccumulator.h"
#include "solum_engine/resources/Coords.h"
#include "solum_engine/voxel/BlockModelLibrary.h"
#include "solum_engine/voxel/StreamingUpload.h"
#include "solum_engine/voxel/mesh_stream/UploadMailbox.h"

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
    std::shared_ptr<const BlockModelLibrary> blockModelLibrary_;

    std::thread streamingThread_;
    mutable std::mutex streamingMutex_;
    std::condition_variable streamingCv_;
    bool streamingStopRequested_ = false;
    bool hasLatestStreamingCamera_ = false;
    glm::vec3 latestStreamingCamera_{0.0f, 0.0f, 0.0f};
    float latestStreamingSseProjectionScale_ = 390.0f;
    UploadMailbox uploadMailbox_{};
    uint64_t streamerLastDeltaRevision_ = 0;

    std::array<TimingAccumulator, static_cast<std::size_t>(TimingStage::Count)> timingAccumulators_{};
    std::atomic<uint64_t> streamSkipNoCamera_{0};
    std::atomic<uint64_t> streamSkipUnchanged_{0};
    std::atomic<uint64_t> streamSnapshotsPrepared_{0};
    std::mutex timingSnapshotMutex_;
    TimingRawTotals lastTimingRawTotals_{};
    std::optional<std::chrono::steady_clock::time_point> lastTimingSampleTime_;

    void streamingThreadMain();

    void recordTimingNs(TimingStage stage, uint64_t ns) noexcept;
    TimingRawTotals captureTimingRawTotals() const;
    static TimingStageSnapshot makeStageSnapshot(const TimingRawTotals& current,
                                                 const TimingRawTotals& previous,
                                                 TimingStage stage,
                                                 double sampleWindowSeconds);

public:
    VoxelStreamingSystem();
    ~VoxelStreamingSystem();

    bool initialize(std::shared_ptr<const BlockModelLibrary> blockModelLibrary);
    void start(const glm::vec3& initialCameraPosition, uint64_t initialUploadedMeshRevision);
    void stop();

    void updateCamera(const glm::vec3& cameraPosition, float sseProjectionScale);
    std::optional<MeshStreamingDelta> tryConsumePreparedDelta();
    void recordMainUpdateDurationNs(uint64_t ns) noexcept;

    RuntimeTimingSnapshot getRuntimeTimingSnapshot();
    const World* world() const noexcept;
};
