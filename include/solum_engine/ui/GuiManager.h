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
        // ImGUI state
    struct ImGUIState {
        bool showMainWindow = true;
        float timeMultiplier = 0.5f;  // Multiplier for time (originally hardcoded as 0.5)
        bool pauseTime = false;       // Allow pausing time
        float manualTime = 0.0f;      // Manual time override
        bool useManualTime = false;   // Use manual time instead of automatic

        // Camera controls
        bool showCameraControls = true;

        // Performance metrics
        bool showPerformanceMetrics = true;

        // Lighting controls
        bool showLightingControls = true;
        glm::vec3 lightDirection = glm::vec3(0.3f, 0.3f, -0.7f);
        glm::vec3 lightColor = glm::vec3(1.0f, 1.0f, 0.9f);
        float lightIntensity = 1.0f;

        // Debug controls
        bool showDebugControls = true;
    };

    ImGUIState imguiState;  // Add ImGUI state

public:
    bool initImGUI(GLFWwindow* window, wgpu::Device device, wgpu::TextureFormat format);
    void renderImGUI(FrameUniforms& uniforms,
                     const std::vector<float>& frameTimes,
                     FirstPersonCamera& camera,
                     float frameTime,
                     const RuntimeTimingSnapshot& runtimeTiming);
    void terminateImGUI();
    void updateImGUIFrame();
};
