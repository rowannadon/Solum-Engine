#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>

#include "solum_engine/render/RuntimeTiming.h"

enum class MainTimingStage : std::size_t {
    UploadMeshlets = 0,
    UpdateDebugBounds,
    RenderFrameCpu,
    AcquireSurface,
    EncodeCommands,
    QueueSubmit,
    Present,
    DeviceTick,
    Count
};

class RuntimeTimingTracker {
public:
    void record(MainTimingStage stage, uint64_t ns) noexcept;
    void incrementMainUploadsApplied() noexcept;
    RuntimeTimingSnapshot snapshot(bool pendingUploadQueued);

private:
    struct TimingAccumulator {
        std::atomic<uint64_t> totalNs{0};
        std::atomic<uint64_t> callCount{0};
        std::atomic<uint64_t> maxNs{0};
    };

    struct TimingRawTotals {
        std::array<uint64_t, static_cast<std::size_t>(MainTimingStage::Count)> totalNs{};
        std::array<uint64_t, static_cast<std::size_t>(MainTimingStage::Count)> callCount{};
        std::array<uint64_t, static_cast<std::size_t>(MainTimingStage::Count)> maxNs{};
        uint64_t mainUploadsApplied = 0;
    };

    TimingRawTotals captureRawTotals() const;
    static TimingStageSnapshot makeStageSnapshot(const TimingRawTotals& current,
                                                 const TimingRawTotals& previous,
                                                 MainTimingStage stage,
                                                 double sampleWindowSeconds);

    std::array<TimingAccumulator, static_cast<std::size_t>(MainTimingStage::Count)> accumulators_{};
    std::atomic<uint64_t> mainUploadsApplied_{0};

    std::mutex snapshotMutex_;
    TimingRawTotals lastRawTotals_{};
    std::optional<std::chrono::steady_clock::time_point> lastSampleTime_;
};
