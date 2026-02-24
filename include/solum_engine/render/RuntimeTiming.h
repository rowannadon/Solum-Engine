#pragma once

#include <cstdint>

struct TimingStageSnapshot {
    double averageMs = 0.0;
    double peakMs = 0.0;
    double totalMsPerSecond = 0.0;
    double callsPerSecond = 0.0;
    uint64_t totalCalls = 0;
};

struct RuntimeTimingSnapshot {
    double sampleWindowSeconds = 0.0;

    TimingStageSnapshot mainUpdateWorldStreaming;
    TimingStageSnapshot mainUploadMeshlets;
    TimingStageSnapshot mainUpdateDebugBounds;
    TimingStageSnapshot mainRenderFrameCpu;
    TimingStageSnapshot mainAcquireSurface;
    TimingStageSnapshot mainEncodeCommands;
    TimingStageSnapshot mainQueueSubmit;
    TimingStageSnapshot mainPresent;
    TimingStageSnapshot mainDeviceTick;

    TimingStageSnapshot streamWait;
    TimingStageSnapshot streamWorldUpdate;
    TimingStageSnapshot streamMeshUpdate;
    TimingStageSnapshot streamCopyMeshlets;
    TimingStageSnapshot streamPrepareUpload;

    uint64_t streamSkipNoCamera = 0;
    uint64_t streamSkipUnchanged = 0;
    uint64_t streamSkipThrottle = 0;
    uint64_t streamSnapshotsPrepared = 0;
    uint64_t mainUploadsApplied = 0;

    bool worldHasPendingJobs = false;
    bool meshHasPendingJobs = false;
    bool pendingUploadQueued = false;
};
