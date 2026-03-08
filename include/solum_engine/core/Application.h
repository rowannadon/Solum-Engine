#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include <glm/mat4x4.hpp>

#include "solum_engine/render/Uniforms.h"
#include "solum_engine/render/RuntimeTiming.h"
#include "solum_engine/render/WebGPURenderer.h"
#include "solum_engine/core/FramePacer.h"
#include "solum_engine/ui/GuiManager.h"
#include "solum_engine/core/Camera.h"
#include "solum_engine/voxel/VoxelStreamingSystem.h"

struct GLFWwindow;
class BufferManager;

class Application {
public:
    Application() = default;

    bool Initialize();
    void Terminate();
    void MainLoop();
    bool IsRunning();

private:
    // Event handlers
    void registerMovementCallbacks();
    void onResize();
    void onMouseMove(double xpos, double ypos);
    void onMouseButton(int button, int action, int mods);
    void onScroll(double xoffset, double yoffset);
    void onKey(int key, int scancode, int action, int mods);

    void updateProjectionMatrix(int zoom);
    void updateViewMatrix();
    void processInput();
    void processBlockInteractions();
    void updateCullingCameraMatrices(const glm::mat4& renderViewMatrix, bool freezeCullingCamera);
    struct VoxelRaycastHit {
        bool hit = false;
        BlockCoord breakCoord{};
        BlockCoord placeCoord{};
    };
    bool raycastTargetBlock(float maxDistance, VoxelRaycastHit& outHit) const;

private:
    // Mouse state for first person look
    struct MouseState {
        bool firstMouse = true;
        bool leftClickRequested = false;
        bool rightClickRequested = false;
        float lastX = 640.0f;  // Half of initial window width
        float lastY = 360.0f;  // Half of initial window height
    };

    // Key states for WASD movement
    struct KeyStates {
        bool W = false;
        bool A = false;
        bool S = false;
        bool D = false;
        bool Space = false;   // Move up
        bool Shift = false;   // Move down
    };

    GLFWwindow* window;
    GuiManager gui;

    WebGPURenderer gpu;
    VoxelStreamingSystem voxelStreaming_;
    BufferManager *buf;

    FirstPersonCamera camera;
    std::mutex cameraMutex;

    MouseState mouseState;
    KeyStates keyStates;
    bool cursorCaptured = false;

    float deltaTime = 0.0f;
    float lastFrame = 0.0f;
    float frameTime = 0.0f;

    int refreshRate = 60;

    FrameUniforms uniforms;
    RuntimeTimingSnapshot runtimeTimingSnapshot_;
    glm::mat4 cullingViewMatrix_ = glm::mat4x4(1.0f);
    glm::mat4 inverseCullingViewMatrix_ = glm::mat4x4(1.0f);

    std::vector<float> frameTimes;
    std::unique_ptr<FramePacer> framePacer_;
};
