#pragma once

// Application.h
#include <GLFW/glfw3.h>
#include <vector>

#include "solum_engine/render/Uniforms.h"
#include "solum_engine/render/WebGPURenderer.h"
#include "solum_engine/ui/GuiManager.h"
#include "solum_engine/core/Camera.h"

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

private:
    // Mouse state for first person look
    struct MouseState {
        bool firstMouse = true;
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
    BufferManager* buf = nullptr;

    FirstPersonCamera camera;

    MouseState mouseState;
    KeyStates keyStates;
    bool cursorCaptured = false;

    float deltaTime = 0.0f;
    float lastFrame = 0.0f;
    float frameTime = 0.0f;

    int refreshRate = 60;

    FrameUniforms uniforms;

    std::vector<float> frameTimes;
};
