#pragma once

#include <glm/glm.hpp>

#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_wgpu.h>

#include <webgpu/webgpu.hpp>
#include <vector>

#include "solum_engine/core/Camera.h"
#include "solum_engine/render/RuntimeTiming.h"
#include "solum_engine/render/Uniforms.h"

class GuiManager {
public:
    struct Config {
        float dayDurationSeconds = 300.0f;
        float timeMultiplier = 0.5f;
        bool pauseTime = false;
        float manualTimeHours = 12.0f;
        bool useManualTime = false;
        float initialTimeHours = 12.0f;
        bool freezeCullingCamera = false;
        glm::vec3 cameraResetPosition{0.0f, 0.0f, 175.0f};
        float cameraResetYaw = 180.0f;
        float cameraResetPitch = 0.0f;
        float cameraResetMovementSpeed = 80.0f;
        float cameraResetMouseSensitivity = 0.1f;
        float cameraResetFieldOfView = 85.0f;
        bool occlusionEnabled = true;
        float occlusionBias = 0.01f;
        float occlusionNearSkipDistance = 20.0f;
        float occlusionMinProjectedSpanPixels = 1.0f;
    };

private:
    // ImGUI state
    struct ImGUIState {
        bool showMainWindow = true;
        float timeMultiplier = 0.5f;
        bool pauseTime = false;
        float manualTime = 12.0f;
        bool useManualTime = false;
        float currentTimeHours = 12.0f;

        bool showCameraControls = true;
        bool showPerformanceMetrics = true;
        bool showDebugControls = true;
        bool freezeCullingCamera = false;
    };

    ImGUIState imguiState;
    Config config_{};
    bool enabled_ = false;

public:
    bool initImGUI(GLFWwindow* window, wgpu::Device device, wgpu::TextureFormat format);
    bool initImGUI(GLFWwindow* window, wgpu::Device device, wgpu::TextureFormat format, const Config& config);
    void renderImGUI(FrameUniforms& uniforms,
                     float deltaTime,
                     const std::vector<float>& frameTimes,
                     FirstPersonCamera& camera,
                     float frameTime,
                     const RuntimeTimingSnapshot& runtimeTiming);
    bool isEnabled() const noexcept { return enabled_; }
    bool isCullingCameraFrozen() const noexcept { return enabled_ && imguiState.freezeCullingCamera; }
    void terminateImGUI();
    void updateImGUIFrame();
};
