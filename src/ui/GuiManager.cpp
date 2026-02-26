#include "solum_engine/ui/GuiManager.h"

#include <iostream>
#include <numeric>

namespace { constexpr bool kEnableImGuiGamepadNav = false; }

bool GuiManager:: initImGUI(GLFWwindow* window, Device device, TextureFormat format) {
// Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    if (kEnableImGuiGamepadNav) {
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    }

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
    // Match the main voxel pass so ImGui can be drawn in the same render pass.
    webgpu_init_info.DepthStencilFormat = TextureFormat::Depth32Float;

    // Must match voxel pass sample count.
    webgpu_init_info.PipelineMultisampleState.count = 4;
    webgpu_init_info.PipelineMultisampleState.alphaToCoverageEnabled = false;

    if (!ImGui_ImplWGPU_Init(&webgpu_init_info)) {
        std::cerr << "Failed to initialize ImGui WebGPU backend" << std::endl;
        return false;
    }

    return true;    
}

void GuiManager::renderImGUI(FrameUniforms& uniforms,
                             const std::vector<float>& frameTimes,
                             FirstPersonCamera& camera,
                             float frameTime,
                             const RuntimeTimingSnapshot& runtimeTiming) {
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

            ImGui::Separator();
            ImGui::Text("Runtime Timing Window: %.2f s", runtimeTiming.sampleWindowSeconds);
            ImGui::Text("Main Render CPU: avg %.3f ms, load %.2f ms/s", runtimeTiming.mainRenderFrameCpu.averageMs, runtimeTiming.mainRenderFrameCpu.totalMsPerSecond);
            ImGui::Text("Main World Streaming: avg %.3f ms, load %.2f ms/s", runtimeTiming.mainUpdateWorldStreaming.averageMs, runtimeTiming.mainUpdateWorldStreaming.totalMsPerSecond);
            ImGui::Text("Main Mesh Upload: avg %.3f ms, load %.2f ms/s", runtimeTiming.mainUploadMeshlets.averageMs, runtimeTiming.mainUploadMeshlets.totalMsPerSecond);
            ImGui::Text("Main Debug Bounds: avg %.3f ms, load %.2f ms/s", runtimeTiming.mainUpdateDebugBounds.averageMs, runtimeTiming.mainUpdateDebugBounds.totalMsPerSecond);
            ImGui::Text("Main Acquire Surface: avg %.3f ms, load %.2f ms/s", runtimeTiming.mainAcquireSurface.averageMs, runtimeTiming.mainAcquireSurface.totalMsPerSecond);
            ImGui::Text("Main Encode Commands: avg %.3f ms, load %.2f ms/s", runtimeTiming.mainEncodeCommands.averageMs, runtimeTiming.mainEncodeCommands.totalMsPerSecond);
            ImGui::Text("Main Queue Submit: avg %.3f ms, load %.2f ms/s", runtimeTiming.mainQueueSubmit.averageMs, runtimeTiming.mainQueueSubmit.totalMsPerSecond);
            ImGui::Text("Main Present: avg %.3f ms, load %.2f ms/s", runtimeTiming.mainPresent.averageMs, runtimeTiming.mainPresent.totalMsPerSecond);
            ImGui::Text("Main Device Tick: avg %.3f ms, load %.2f ms/s", runtimeTiming.mainDeviceTick.averageMs, runtimeTiming.mainDeviceTick.totalMsPerSecond);

            ImGui::Separator();
            ImGui::Text("Stream Wait: avg %.3f ms, load %.2f ms/s", runtimeTiming.streamWait.averageMs, runtimeTiming.streamWait.totalMsPerSecond);
            ImGui::Text("Stream World Update: avg %.3f ms, load %.2f ms/s", runtimeTiming.streamWorldUpdate.averageMs, runtimeTiming.streamWorldUpdate.totalMsPerSecond);
            ImGui::Text("Stream Mesh Update: avg %.3f ms, load %.2f ms/s", runtimeTiming.streamMeshUpdate.averageMs, runtimeTiming.streamMeshUpdate.totalMsPerSecond);
            ImGui::Text("Stream Copy Meshlets: avg %.3f ms, load %.2f ms/s", runtimeTiming.streamCopyMeshlets.averageMs, runtimeTiming.streamCopyMeshlets.totalMsPerSecond);
            ImGui::Text("Stream Prepare Upload: avg %.3f ms, load %.2f ms/s", runtimeTiming.streamPrepareUpload.averageMs, runtimeTiming.streamPrepareUpload.totalMsPerSecond);

            ImGui::Separator();
            ImGui::Text("Stream skips (window): no camera %llu, unchanged %llu, throttle %llu",
                        static_cast<unsigned long long>(runtimeTiming.streamSkipNoCamera),
                        static_cast<unsigned long long>(runtimeTiming.streamSkipUnchanged),
                        static_cast<unsigned long long>(runtimeTiming.streamSkipThrottle));
            ImGui::Text("Stream snapshots (window): %llu",
                        static_cast<unsigned long long>(runtimeTiming.streamSnapshotsPrepared));
            ImGui::Text("Main uploads (window): %llu",
                        static_cast<unsigned long long>(runtimeTiming.mainUploadsApplied));
            ImGui::Text("Pending jobs: world %s, mesh %s, upload queued %s",
                        runtimeTiming.worldHasPendingJobs ? "yes" : "no",
                        runtimeTiming.meshHasPendingJobs ? "yes" : "no",
                        runtimeTiming.pendingUploadQueued ? "yes" : "no");
        }

        if (imguiState.showDebugControls && ImGui::CollapsingHeader("Debug", ImGuiTreeNodeFlags_DefaultOpen)) {
            bool meshletDebugEnabled = (uniforms.renderFlags[0] & kRenderFlagMeshletDebug) != 0u;
            if (ImGui::Checkbox("Meshlet Debug", &meshletDebugEnabled)) {
                if (meshletDebugEnabled) {
                    uniforms.renderFlags[0] |= kRenderFlagMeshletDebug;
                } else {
                    uniforms.renderFlags[0] &= ~kRenderFlagMeshletDebug;
                }
            }

            bool boundsMasterEnabled = (uniforms.renderFlags[0] & kRenderFlagBoundsDebug) != 0u;
            if (ImGui::Checkbox("Bounds Master", &boundsMasterEnabled)) {
                if (boundsMasterEnabled) {
                    uniforms.renderFlags[0] |= kRenderFlagBoundsDebug;
                } else {
                    uniforms.renderFlags[0] &= ~kRenderFlagBoundsDebug;
                }
            }
            ImGui::Separator();

            bool showChunks = (uniforms.renderFlags[0] & kRenderFlagBoundsChunks) != 0u;
            bool showColumns = (uniforms.renderFlags[0] & kRenderFlagBoundsColumns) != 0u;
            bool showRegions = (uniforms.renderFlags[0] & kRenderFlagBoundsRegions) != 0u;
            bool showMeshlets = (uniforms.renderFlags[0] & kRenderFlagBoundsMeshlets) != 0u;

            if (ImGui::Checkbox("Chunks", &showChunks)) {
                if (showChunks) {
                    uniforms.renderFlags[0] |= kRenderFlagBoundsChunks;
                } else {
                    uniforms.renderFlags[0] &= ~kRenderFlagBoundsChunks;
                }
            }

            if (ImGui::Checkbox("Columns", &showColumns)) {
                if (showColumns) {
                    uniforms.renderFlags[0] |= kRenderFlagBoundsColumns;
                } else {
                    uniforms.renderFlags[0] &= ~kRenderFlagBoundsColumns;
                }
            }

            if (ImGui::Checkbox("Regions", &showRegions)) {
                if (showRegions) {
                    uniforms.renderFlags[0] |= kRenderFlagBoundsRegions;
                } else {
                    uniforms.renderFlags[0] &= ~kRenderFlagBoundsRegions;
                }
            }

            if (ImGui::Checkbox("Meshlets", &showMeshlets)) {
                if (showMeshlets) {
                    uniforms.renderFlags[0] |= kRenderFlagBoundsMeshlets;
                } else {
                    uniforms.renderFlags[0] &= ~kRenderFlagBoundsMeshlets;
                }
            }

            ImGui::Separator();
            ImGui::Text("Occlusion Culling");
            bool occlusionEnabled = uniforms.occlusionParams[0] >= 0.5f;
            if (ImGui::Checkbox("Enable Occlusion", &occlusionEnabled)) {
                uniforms.occlusionParams[0] = occlusionEnabled ? 1.0f : 0.0f;
            }
            ImGui::SliderFloat("Occlusion Bias", &uniforms.occlusionParams[1], 0.0f, 0.05f, "%.4f");
            ImGui::SliderFloat("Near Skip Distance", &uniforms.occlusionParams[2], 0.0f, 128.0f, "%.1f");
            ImGui::SliderFloat("Min Projected Span (px)", &uniforms.occlusionParams[3], 0.0f, 8.0f, "%.2f");
            if (ImGui::Button("Reset Occlusion")) {
                uniforms.occlusionParams[0] = 1.0f;
                uniforms.occlusionParams[1] = 0.01f;
                uniforms.occlusionParams[2] = 20.0f;
                uniforms.occlusionParams[3] = 1.0f;
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
