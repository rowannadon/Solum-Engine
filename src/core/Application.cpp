// Application.cpp

#include "solum_engine/core/Application.h"

#include <cfloat>

bool Application::Initialize() {
    if (!gpu.initialize()) return false;
    pip = gpu.getPipelineManager();
    buf = gpu.getBufferManager();
    tex = gpu.getTextureManager();

    window = gpu.getWindow();

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    if (mode) {
        std::cout << "Monitor refresh rate: " << mode->refreshRate << " Hz" << std::endl;
        refreshRate = mode->refreshRate;
    } else {
        std::cout << "Monitor refresh rate unavailable, using default: " << refreshRate << " Hz" << std::endl;
    }

    // initialize uniforms
    uniforms.modelMatrix = glm::mat4x4(1.0);
    uniforms.projectionMatrix = glm::mat4x4(1.0);
    uniforms.inverseProjectionMatrix = glm::mat4x4(1.0);
    uniforms.viewMatrix = glm::mat4x4(1.0);
    uniforms.inverseViewMatrix = glm::mat4x4(1.0);
    uniforms.renderFlags[0] = kRenderFlagBoundsLayerMask;
    uniforms.renderFlags[1] = 0u;
    uniforms.renderFlags[2] = 0u;
    uniforms.renderFlags[3] = 0u;

    camera.position = glm::vec3(10.0, 10.0, 0.0);
    camera.updateCameraVectors();
    updateProjectionMatrix(camera.zoom);
    updateViewMatrix();

    buf->writeBuffer("uniform_buffer", 0, &uniforms, sizeof(FrameUniforms));

    if (!gui.initImGUI(window, gpu.getContext()->getDevice(), gpu.getContext()->getSurfaceFormat())) {
        std::cerr << "Failed to initialize ImGUI" << std::endl;
        return false;
    }

    registerMovementCallbacks();
    // Install ImGui's full GLFW callback set and chain to the app callbacks above.
    ImGui_ImplGlfw_InstallCallbacks(window);

    return true;
}


void Application::Terminate() {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    gui.terminateImGUI();
}

void Application::MainLoop() {
    float TARGET_FPS = static_cast<float>(refreshRate);
    float TARGET_FRAME_TIME = 1.0f / TARGET_FPS;

    float currentFrame = static_cast<float>(glfwGetTime());
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    // Poll events first to minimize input lag
    glfwPollEvents();

    // Update ImGUI frame
    gui.updateImGUIFrame();

    // Process input (only if ImGUI doesn't want input)
    ImGuiIO& io = ImGui::GetIO();
    if (cursorCaptured) {
        io.MousePos = ImVec2(-FLT_MAX, -FLT_MAX);
        io.MouseDelta = ImVec2(0.0f, 0.0f);
        for (bool& down : io.MouseDown) {
            down = false;
        }
        io.MouseWheel = io.MouseWheelH = 0.0f;
    }

    if (!io.WantCaptureKeyboard && !io.WantCaptureMouse) {
        processInput();
    }

    // Early exit if frame budget is already exceeded
    float frameStartTime = currentFrame;

    const glm::mat4 viewGPU = glm::lookAt(camera.position, camera.position + camera.front, camera.up);
    uniforms.viewMatrix = viewGPU;
    uniforms.inverseViewMatrix = glm::inverse(viewGPU);

    gui.renderImGUI(uniforms, frameTimes, camera, frameTime);
    buf->writeBuffer("uniform_buffer", 0, &uniforms, sizeof(FrameUniforms));
    
    gpu.renderFrame(uniforms);

    // After rendering, perform frame timing

    float frameEndTime = static_cast<float>(glfwGetTime());
    frameTime = frameEndTime - frameStartTime;

    frameTimes.push_back(frameTime);
    if (frameTimes.size() > 100) {
        frameTimes.erase(frameTimes.begin());
    }

    float averageFrameTime = std::accumulate(frameTimes.begin(), frameTimes.end(), 0.0f) / frameTimes.size();
    float averageFPS = 1.0f / averageFrameTime;

    static float lastDebugTime = 0.0f;
    if (currentFrame - lastDebugTime >= 1.0f) {
        float frameBudgetMs = TARGET_FRAME_TIME * 1000.0f;
        float currentFrameMs = frameTime * 1000.0f;
        float averageFrameMs = averageFrameTime * 1000.0f;
        float frameBudgetUtilization = (averageFrameTime / TARGET_FRAME_TIME) * 100.0f;

        std::cout << "=== Frame Timing Debug ===" << std::endl;
        std::cout << "Target FPS: " << TARGET_FPS << " (Budget: " << frameBudgetMs << "ms)" << std::endl;
        std::cout << "Current Frame: " << currentFrameMs << "ms" << std::endl;
        std::cout << "Average Frame: " << averageFrameMs << "ms (" << averageFPS << " FPS)" << std::endl;
        std::cout << "Frame Budget Utilization: " << frameBudgetUtilization << "%" << std::endl;
        std::cout << "=========================" << std::endl;

        lastDebugTime = currentFrame;
    }

    float timeAfterWork = static_cast<float>(glfwGetTime());
    float workTime = timeAfterWork - frameStartTime;

    if (workTime < TARGET_FRAME_TIME) {
        float remainingTime = TARGET_FRAME_TIME - workTime;

        const float SLEEP_BUFFER = 0.0005f;

        if (remainingTime > SLEEP_BUFFER) {
            float sleepTime = remainingTime - SLEEP_BUFFER;
            std::this_thread::sleep_for(std::chrono::duration<float>(sleepTime));
        }

        while (static_cast<float>(glfwGetTime()) - frameStartTime < TARGET_FRAME_TIME) {
            std::this_thread::yield();
        }
    }
}

void Application::registerMovementCallbacks() {
    // Set the user pointer to be "this"
    glfwSetWindowUserPointer(window, this);
    // Use a non-capturing lambda as resize callback
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int, int) {
        auto that = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
        if (that != nullptr) that->onResize();
        });
    glfwSetCursorPosCallback(window, [](GLFWwindow* window, double xpos, double ypos) {
        auto that = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
        if (that != nullptr) that->onMouseMove(xpos, ypos);
        });
    glfwSetMouseButtonCallback(window, [](GLFWwindow* window, int button, int action, int mods) {
        auto that = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
        if (that != nullptr) that->onMouseButton(button, action, mods);
        });
    glfwSetScrollCallback(window, [](GLFWwindow* window, double xoffset, double yoffset) {
        auto that = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
        if (that != nullptr) that->onScroll(xoffset, yoffset);
        });
    glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
        auto that = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
        if (that != nullptr) that->onKey(key, scancode, action, mods);
        });
}


void Application::onResize() {
    gpu.requestResize();

    // Update projection matrix
    updateProjectionMatrix(camera.zoom);
}

void Application::processInput() {
    std::unique_lock<std::mutex> lock(cameraMutex);

    float velocity = camera.movementSpeed * deltaTime;

    // WASD movement
    if (keyStates.W)
        camera.position += camera.front * velocity;
    if (keyStates.S)
        camera.position -= camera.front * velocity;
    if (keyStates.A)
        camera.position -= camera.right * velocity;
    if (keyStates.D)
        camera.position += camera.right * velocity;

    // Vertical movement
    if (keyStates.Space)
        camera.position += camera.worldUp * velocity;
    if (keyStates.Shift)
        camera.position -= camera.worldUp * velocity;

    // Update view matrix if camera position changed
    updateViewMatrix();
}

void Application::updateProjectionMatrix(int zoom) {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    if (width <= 0 || height <= 0) {
        return;
    }
    float ratio = width / (float)height;
    uniforms.projectionMatrix = glm::perspective(zoom * PI / 180, ratio, 0.1f, 2500.0f);
    uniforms.inverseProjectionMatrix = glm::inverse(uniforms.projectionMatrix);

    buf->writeBuffer("uniform_buffer", offsetof(FrameUniforms, projectionMatrix), &uniforms.projectionMatrix, sizeof(FrameUniforms::projectionMatrix));
}

void Application::updateViewMatrix() {
    uniforms.viewMatrix = glm::lookAt(camera.position, camera.position + camera.front, camera.up);
    uniforms.inverseViewMatrix = glm::inverse(uniforms.viewMatrix);
    buf->writeBuffer("uniform_buffer", offsetof(FrameUniforms, viewMatrix), &uniforms.viewMatrix, sizeof(FrameUniforms::viewMatrix));
}

void Application::onMouseMove(double xpos, double ypos) {
    // Only handle mouse movement if window is focused (cursor is disabled)
    if (glfwGetInputMode(window, GLFW_CURSOR) != GLFW_CURSOR_DISABLED) return;

    if (mouseState.firstMouse) {
        mouseState.lastX = static_cast<float>(xpos);
        mouseState.lastY = static_cast<float>(ypos);
        mouseState.firstMouse = false;
    }

    float xoffset = static_cast<float>(xpos) - mouseState.lastX;
    float yoffset = mouseState.lastY - static_cast<float>(ypos); // Reversed since y-coordinates go from bottom to top

    mouseState.lastX = static_cast<float>(xpos);
    mouseState.lastY = static_cast<float>(ypos);

    xoffset *= camera.mouseSensitivity;
    yoffset *= camera.mouseSensitivity;

    camera.yaw += xoffset;
    camera.pitch += yoffset;

    // Constrain pitch to avoid screen flipping
    if (camera.pitch > 89.0f)
        camera.pitch = 89.0f;
    if (camera.pitch < -89.0f)
        camera.pitch = -89.0f;

    camera.updateCameraVectors();
    updateViewMatrix();
}

void Application::onMouseButton(int button, int action, int /* modifiers */) {
    if (ImGui::GetCurrentContext() != nullptr) {
        ImGuiIO& io = ImGui::GetIO();
        if (io.WantCaptureMouse) {
            return; // ImGUI is handling this input
        }
    }

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            // Left click focuses the window and enables camera control
            mouseState.firstMouse = true;
            cursorCaptured = true;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            glfwSetCursorPos(window, mouseState.lastX, mouseState.lastY);
        }
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            mouseState.rightMousePressed = true;
        }
        else if (action == GLFW_RELEASE) {
            mouseState.rightMousePressed = false;
        }
    }
}

void Application::onScroll(double /* xoffset */, double yoffset) {
    if (ImGui::GetCurrentContext() != nullptr) {
        ImGuiIO& io = ImGui::GetIO();
        if (io.WantCaptureMouse) {
            return; // ImGUI is handling this input
        }
    }

    camera.zoom -= 10 * static_cast<float>(yoffset);
    if (camera.zoom < 1.0f)
        camera.zoom = 1.0f;
    if (camera.zoom > 120.0f)
        camera.zoom = 120.0f;
    updateProjectionMatrix(camera.zoom);
}

void Application::onKey(int key, int scancode, int action, int mods) {
    //ImGuiIO& io = ImGui::GetIO();
    //if (io.WantCaptureKeyboard) {
    //    return; // ImGUI is handling this input
    //}

    bool keyPressed = (action == GLFW_PRESS || action == GLFW_REPEAT);
    bool keyReleased = (action == GLFW_RELEASE);

    switch (key) {
    case GLFW_KEY_W:
        if (keyPressed) keyStates.W = true;
        if (keyReleased) keyStates.W = false;
        break;
    case GLFW_KEY_S:
        if (keyPressed) keyStates.S = true;
        if (keyReleased) keyStates.S = false;
        break;
    case GLFW_KEY_A:
        if (keyPressed) keyStates.A = true;
        if (keyReleased) keyStates.A = false;
        break;
    case GLFW_KEY_D:
        if (keyPressed) keyStates.D = true;
        if (keyReleased) keyStates.D = false;
        break;
    case GLFW_KEY_SPACE:
        if (keyPressed) keyStates.Space = true;
        if (keyReleased) keyStates.Space = false;
        break;
    case GLFW_KEY_LEFT_SHIFT:
        if (keyPressed) keyStates.Shift = true;
        if (keyReleased) keyStates.Shift = false;
        break;
    case GLFW_KEY_ESCAPE:
        if (keyPressed) {
            cursorCaptured = false;
            mouseState.firstMouse = true;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
        break;
    }
}

bool Application::IsRunning() {
    return !glfwWindowShouldClose(window);
}
