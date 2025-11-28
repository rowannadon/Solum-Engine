#include "solum_engine/ui/GuiManager.h"

#include <iostream>
#include <numeric>

bool GuiManager:: initImGUI(GLFWwindow* window, Device device, TextureFormat format) {
// Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    if (!ImGui_ImplGlfw_InitForOther(window, false)) {
        std::cerr << "Failed to initialize ImGui GLFW backend" << std::endl;
        return false;
    }

    ImGui_ImplWGPU_InitInfo webgpu_init_info = {};
    webgpu_init_info.Device = device;
    webgpu_init_info.NumFramesInFlight = 3;
    webgpu_init_info.RenderTargetFormat = format;
    webgpu_init_info.DepthStencilFormat = TextureFormat::Undefined;

    webgpu_init_info.PipelineMultisampleState.count = 1;
    webgpu_init_info.PipelineMultisampleState.alphaToCoverageEnabled = false;

    if (!ImGui_ImplWGPU_Init(&webgpu_init_info)) {
        std::cerr << "Failed to initialize ImGui WebGPU backend" << std::endl;
        return false;
    }

    return true;    
}

void GuiManager::renderImGUI(const FrameUniforms& uniforms, const std::vector<float>& frameTimes, FirstPersonCamera& camera, float frameTime) {
    // Main control window
    if (imguiState.showMainWindow) {
        ImGui::Begin("Engine Controls", &imguiState.showMainWindow);

        // Time Controls
        if (ImGui::CollapsingHeader("Time Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Pause Time", &imguiState.pauseTime);

            if (!imguiState.pauseTime) {
                ImGui::SliderFloat("Time Multiplier", &imguiState.timeMultiplier, 0.0f, 5.0f, "%.2f");
                ImGui::SameLine();
                if (ImGui::Button("Reset##time")) {
                    imguiState.timeMultiplier = 0.5f;
                }
            }

            ImGui::Checkbox("Use Manual Time", &imguiState.useManualTime);
            if (imguiState.useManualTime) {
                ImGui::SliderFloat("Manual Time", &imguiState.manualTime, 0.0f, 100.0f, "%.2f");
            }

            //ImGui::Text("Current Time: %.2f", uniforms.time);
        }

        // Camera Controls
        if (imguiState.showCameraControls && ImGui::CollapsingHeader("Camera Controls")) {
            ImGui::SliderFloat("Movement Speed", &camera.movementSpeed, 5.0f, 500.0f, "%.1f");
            ImGui::SliderFloat("Mouse Sensitivity", &camera.mouseSensitivity, 0.01f, 1.0f, "%.3f");
            ImGui::SliderFloat("FOV", &camera.zoom, 10.0f, 180.0f, "%.1f");

            if (ImGui::Button("Reset Camera")) {
                camera.position = glm::vec3(5.0f, 0.0f, 200.0f);
                camera.yaw = 180.0f;
                camera.pitch = 0.0f;
                camera.zoom = 85.0f;
                camera.updateCameraVectors();
                //updateViewMatrix();
                //updateProjectionMatrix(camera.zoom);
            }

            ImGui::Text("Position: %.1f, %.1f, %.1f", camera.position.x, camera.position.y, camera.position.z);
            ImGui::Text("Yaw: %.1f, Pitch: %.1f", camera.yaw, camera.pitch);
        }

        // Performance Metrics
        if (imguiState.showPerformanceMetrics && ImGui::CollapsingHeader("Performance")) {
            float averageFrameTime = frameTimes.empty() ? 0.0f :
                std::accumulate(frameTimes.begin(), frameTimes.end(), 0.0f) / frameTimes.size();
            float averageFPS = averageFrameTime > 0 ? 1.0f / averageFrameTime : 0.0f;

            ImGui::Text("Average FPS: %.1f", averageFPS);
            ImGui::Text("Frame Time: %.2f ms", averageFrameTime * 1000.0f);
            ImGui::Text("Current Frame: %.2f ms", frameTime * 1000.0f);

            // Frame time graph
            if (frameTimes.size() > 10) {
                std::vector<float> frameTimeMs;
                for (float ft : frameTimes) {
                    frameTimeMs.push_back(ft * 1000.0f);
                }
                ImGui::PlotLines("Frame Time (ms)", frameTimeMs.data(), frameTimeMs.size(), 0, nullptr, 0.0f, 50.0f, ImVec2(0, 80));
            }
        }

        ImGui::End();
    }
}

void GuiManager::terminateImGUI() {
    ImGui_ImplWGPU_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void GuiManager::updateImGUIFrame() {
    ImGui_ImplWGPU_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}
