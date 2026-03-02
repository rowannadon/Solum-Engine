#pragma once

#include <vector>

struct FramePacingResult {
    float frameTime = 0.0f;
    float averageFrameTime = 0.0f;
    float averageFps = 0.0f;
};

class FramePacer {
public:
    explicit FramePacer(float targetFps);

    FramePacingResult finalizeFrameAndPace(float currentFrame,
                                           float frameStartTime,
                                           std::vector<float>& frameTimes);

private:
    float targetFrameTime_ = 1.0f / 60.0f;
    float lastDebugTime_ = 0.0f;
};
