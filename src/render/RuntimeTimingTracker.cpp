#include "solum_engine/render/RuntimeTimingTracker.h"

void RuntimeTimingTracker::record(MainTimingStage stage, uint64_t ns) noexcept {
    const std::size_t stageIndex = static_cast<std::size_t>(stage);
    recordTimingAccumulator(accumulators_[stageIndex], ns);
}

void RuntimeTimingTracker::incrementMainUploadsApplied() noexcept {
    mainUploadsApplied_.fetch_add(1, std::memory_order_relaxed);
}

RuntimeTimingTracker::TimingRawTotals RuntimeTimingTracker::captureRawTotals() const {
    TimingRawTotals totals;
    captureTimingAccumulatorArrays(accumulators_, totals.totalNs, totals.callCount, totals.maxNs);

    totals.mainUploadsApplied = mainUploadsApplied_.load(std::memory_order_relaxed);
    return totals;
}

TimingStageSnapshot RuntimeTimingTracker::makeStageSnapshot(const TimingRawTotals& current,
                                                            const TimingRawTotals& previous,
                                                            MainTimingStage stage,
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
