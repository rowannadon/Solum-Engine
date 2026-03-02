#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

#include "solum_engine/render/RuntimeTiming.h"

struct TimingAccumulator {
    std::atomic<uint64_t> totalNs{0};
    std::atomic<uint64_t> callCount{0};
    std::atomic<uint64_t> maxNs{0};
};

inline void recordTimingAccumulator(TimingAccumulator& accumulator, uint64_t ns) noexcept {
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

template <size_t N>
inline void captureTimingAccumulatorArrays(
    const std::array<TimingAccumulator, N>& accumulators,
    std::array<uint64_t, N>& totalNs,
    std::array<uint64_t, N>& callCount,
    std::array<uint64_t, N>& maxNs
) {
    for (size_t i = 0; i < N; ++i) {
        totalNs[i] = accumulators[i].totalNs.load(std::memory_order_relaxed);
        callCount[i] = accumulators[i].callCount.load(std::memory_order_relaxed);
        maxNs[i] = accumulators[i].maxNs.load(std::memory_order_relaxed);
    }
}

inline TimingStageSnapshot makeTimingStageSnapshotFromRaw(uint64_t currentTotalNs,
                                                          uint64_t currentCallCount,
                                                          uint64_t currentMaxNs,
                                                          uint64_t previousTotalNs,
                                                          uint64_t previousCallCount,
                                                          double sampleWindowSeconds) {
    const uint64_t deltaNs = currentTotalNs - previousTotalNs;
    const uint64_t deltaCalls = currentCallCount - previousCallCount;
    const double deltaMs = static_cast<double>(deltaNs) / 1'000'000.0;
    const double window = std::max(sampleWindowSeconds, 1e-6);

    TimingStageSnapshot snapshot;
    snapshot.averageMs = (deltaCalls > 0) ? (deltaMs / static_cast<double>(deltaCalls)) : 0.0;
    snapshot.peakMs = static_cast<double>(currentMaxNs) / 1'000'000.0;
    snapshot.totalMsPerSecond = deltaMs / window;
    snapshot.callsPerSecond = static_cast<double>(deltaCalls) / window;
    snapshot.totalCalls = currentCallCount;
    return snapshot;
}
