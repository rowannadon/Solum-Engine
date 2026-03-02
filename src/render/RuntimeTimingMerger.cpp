#include "solum_engine/render/RuntimeTimingMerger.h"

#include <algorithm>

RuntimeTimingSnapshot mergeRuntimeTimingSnapshots(const RuntimeTimingSnapshot& gpuTiming,
                                                  const RuntimeTimingSnapshot& streamingTiming) {
    RuntimeTimingSnapshot merged = gpuTiming;
    merged.sampleWindowSeconds = std::max(merged.sampleWindowSeconds, streamingTiming.sampleWindowSeconds);
    merged.mainUpdateWorldStreaming = streamingTiming.mainUpdateWorldStreaming;
    merged.streamWait = streamingTiming.streamWait;
    merged.streamWorldUpdate = streamingTiming.streamWorldUpdate;
    merged.streamMeshUpdate = streamingTiming.streamMeshUpdate;
    merged.streamCopyMeshlets = streamingTiming.streamCopyMeshlets;
    merged.streamPrepareUpload = streamingTiming.streamPrepareUpload;
    merged.streamSkipNoCamera = streamingTiming.streamSkipNoCamera;
    merged.streamSkipUnchanged = streamingTiming.streamSkipUnchanged;
    merged.streamSkipThrottle = streamingTiming.streamSkipThrottle;
    merged.streamSnapshotsPrepared = streamingTiming.streamSnapshotsPrepared;
    merged.worldHasPendingJobs = streamingTiming.worldHasPendingJobs;
    merged.meshHasPendingJobs = streamingTiming.meshHasPendingJobs;
    merged.pendingUploadQueued = streamingTiming.pendingUploadQueued || gpuTiming.pendingUploadQueued;
    return merged;
}
