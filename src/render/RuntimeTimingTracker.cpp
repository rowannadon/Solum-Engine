#include "solum_engine/render/RuntimeTimingTracker.h"

#include <algorithm>

void RuntimeTimingTracker::record(MainTimingStage stage, uint64_t ns) noexcept {
    const std::size_t stageIndex = static_cast<std::size_t>(stage);
    TimingAccumulator& accumulator = accumulators_[stageIndex];
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

void RuntimeTimingTracker::incrementMainUploadsApplied() noexcept {
    mainUploadsApplied_.fetch_add(1, std::memory_order_relaxed);
}

RuntimeTimingTracker::TimingRawTotals RuntimeTimingTracker::captureRawTotals() const {
    TimingRawTotals totals;
    for (std::size_t i = 0; i < static_cast<std::size_t>(MainTimingStage::Count); ++i) {
        const TimingAccumulator& accumulator = accumulators_[i];
        totals.totalNs[i] = accumulator.totalNs.load(std::memory_order_relaxed);
        totals.callCount[i] = accumulator.callCount.load(std::memory_order_relaxed);
        totals.maxNs[i] = accumulator.maxNs.load(std::memory_order_relaxed);
    }

    totals.mainUploadsApplied = mainUploadsApplied_.load(std::memory_order_relaxed);
    return totals;
}

TimingStageSnapshot RuntimeTimingTracker::makeStageSnapshot(const TimingRawTotals& current,
                                                            const TimingRawTotals& previous,
                                                            MainTimingStage stage,
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

RuntimeTimingSnapshot RuntimeTimingTracker::snapshot(bool pendingUploadQueued) {
    RuntimeTimingSnapshot out;
    const TimingRawTotals currentTotals = captureRawTotals();
    const auto now = std::chrono::steady_clock::now();

    {
        std::lock_guard<std::mutex> lock(snapshotMutex_);
        if (!lastSampleTime_.has_value()) {
            lastSampleTime_ = now;
            lastRawTotals_ = currentTotals;
        } else {
            const double sampleWindowSeconds =
                std::chrono::duration<double>(now - *lastSampleTime_).count();
            out.sampleWindowSeconds = sampleWindowSeconds;
            out.mainUploadMeshlets = makeStageSnapshot(
                currentTotals,
                lastRawTotals_,
                MainTimingStage::UploadMeshlets,
                sampleWindowSeconds
            );
            out.mainUpdateDebugBounds = makeStageSnapshot(
                currentTotals,
                lastRawTotals_,
                MainTimingStage::UpdateDebugBounds,
                sampleWindowSeconds
            );
            out.mainRenderFrameCpu = makeStageSnapshot(
                currentTotals,
                lastRawTotals_,
                MainTimingStage::RenderFrameCpu,
                sampleWindowSeconds
            );
            out.mainAcquireSurface = makeStageSnapshot(
                currentTotals,
                lastRawTotals_,
                MainTimingStage::AcquireSurface,
                sampleWindowSeconds
            );
            out.mainEncodeCommands = makeStageSnapshot(
                currentTotals,
                lastRawTotals_,
                MainTimingStage::EncodeCommands,
                sampleWindowSeconds
            );
            out.mainQueueSubmit = makeStageSnapshot(
                currentTotals,
                lastRawTotals_,
                MainTimingStage::QueueSubmit,
                sampleWindowSeconds
            );
            out.mainPresent = makeStageSnapshot(
                currentTotals,
                lastRawTotals_,
                MainTimingStage::Present,
                sampleWindowSeconds
            );
            out.mainDeviceTick = makeStageSnapshot(
                currentTotals,
                lastRawTotals_,
                MainTimingStage::DeviceTick,
                sampleWindowSeconds
            );
            out.mainUploadsApplied = currentTotals.mainUploadsApplied - lastRawTotals_.mainUploadsApplied;

            lastSampleTime_ = now;
            lastRawTotals_ = currentTotals;
        }
    }

    out.pendingUploadQueued = pendingUploadQueued;
    return out;
}
