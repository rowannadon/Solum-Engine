#include "solum_engine/core/FramePacer.h"

#include <chrono>
#include <iostream>
#include <numeric>
#include <thread>

#include <GLFW/glfw3.h>

FramePacer::FramePacer(float targetFps) {
    const float safeFps = (targetFps > 1.0f) ? targetFps : 60.0f;
    targetFrameTime_ = 1.0f / safeFps;
}

FramePacingResult FramePacer::finalizeFrameAndPace(float currentFrame,
                                                   float frameStartTime,
                                                   std::vector<float>& frameTimes) {
    const float frameEndTime = static_cast<float>(glfwGetTime());
    const float frameTime = frameEndTime - frameStartTime;

    frameTimes.push_back(frameTime);
    if (frameTimes.size() > 100u) {
        frameTimes.erase(frameTimes.begin());
    }

    const float averageFrameTime =
        std::accumulate(frameTimes.begin(), frameTimes.end(), 0.0f) /
        static_cast<float>(frameTimes.size());
    const float averageFps = (averageFrameTime > 0.0f) ? (1.0f / averageFrameTime) : 0.0f;

    if (currentFrame - lastDebugTime_ >= 1.0f) {
        const float frameBudgetMs = targetFrameTime_ * 1000.0f;
        const float currentFrameMs = frameTime * 1000.0f;
        const float averageFrameMs = averageFrameTime * 1000.0f;
        const float frameBudgetUtilization = (averageFrameTime / targetFrameTime_) * 100.0f;

        std::cout << "=== Frame Timing Debug ===" << std::endl;
        std::cout << "Target FPS: " << (1.0f / targetFrameTime_) << " (Budget: " << frameBudgetMs << "ms)" << std::endl;
        std::cout << "Current Frame: " << currentFrameMs << "ms" << std::endl;
        std::cout << "Average Frame: " << averageFrameMs << "ms (" << averageFps << " FPS)" << std::endl;
        std::cout << "Frame Budget Utilization: " << frameBudgetUtilization << "%" << std::endl;
        std::cout << "=========================" << std::endl;
        lastDebugTime_ = currentFrame;
    }

    const float timeAfterWork = static_cast<float>(glfwGetTime());
    const float workTime = timeAfterWork - frameStartTime;
    if (workTime < targetFrameTime_) {
        const float remainingTime = targetFrameTime_ - workTime;
        constexpr float kSleepBuffer = 0.0005f;
        if (remainingTime > kSleepBuffer) {
            const float sleepTime = remainingTime - kSleepBuffer;
            std::this_thread::sleep_for(std::chrono::duration<float>(sleepTime));
        }

        while (static_cast<float>(glfwGetTime()) - frameStartTime < targetFrameTime_) {
            std::this_thread::yield();
        }
    }

    return FramePacingResult{
        frameTime,
        averageFrameTime,
        averageFps
    };
}
