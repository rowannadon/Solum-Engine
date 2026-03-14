// Application.cpp

#include "solum_engine/core/Application.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <chrono>
#include <iostream>
#include <limits>
#include <thread>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "solum_engine/render/RuntimeTimingMerger.h"
#include "solum_engine/voxel/MaterialRegistry.h"
#include "solum_engine/voxel/World.h"

bool Application::Initialize() {
    WebGPURenderer::Config rendererConfig{};
    if (!gpu.initialize(rendererConfig)) return false;
    if (!voxelStreaming_.initialize(gpu.getBlockModelLibrary())) return false;
    buf = gpu.getBufferManager();

    window = gpu.getWindow();

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    if (mode) {
        std::cout << "Monitor refresh rate: " << mode->refreshRate << " Hz" << std::endl;
        refreshRate = mode->refreshRate;
    } else {
        std::cout << "Monitor refresh rate unavailable, using default: " << refreshRate << " Hz" << std::endl;
    }
    framePacer_ = std::make_unique<FramePacer>(static_cast<float>(refreshRate));

    // initialize uniforms
    uniforms.modelMatrix = glm::mat4x4(1.0);
    uniforms.projectionMatrix = glm::mat4x4(1.0);
    uniforms.inverseProjectionMatrix = glm::mat4x4(1.0);
    uniforms.viewMatrix = glm::mat4x4(1.0);
    uniforms.inverseViewMatrix = glm::mat4x4(1.0);
    uniforms.cullingViewMatrix = glm::mat4x4(1.0);
    uniforms.inverseCullingViewMatrix = glm::mat4x4(1.0);
    uniforms.renderFlags[0] =
        kRenderFlagBoundsChunks |
        kRenderFlagBoundsColumns |
        kRenderFlagBoundsRegions;
    uniforms.renderFlags[1] = 0u;
    uniforms.renderFlags[2] = 0u;
    uniforms.renderFlags[3] = 0u;
    uniforms.occlusionParams[0] = 1.0f;
    uniforms.occlusionParams[1] = 0.01f;
    uniforms.occlusionParams[2] = 20.0f;
    uniforms.occlusionParams[3] = 1.0f;
    uniforms.timeParams[0] = 0.25f;
    uniforms.timeParams[1] = 12.0f;
    uniforms.timeParams[2] = 0.0f;
    uniforms.timeParams[3] = 0.0f;
    uniforms.viewportParams[0] = 1.0f;
    uniforms.viewportParams[1] = 1.0f;
    uniforms.viewportParams[2] = 2.0f;
    uniforms.viewportParams[3] = 2.0f;

    camera.position = glm::vec3(0.0, 0.0, 175.0);
    camera.updateCameraVectors();
    updateViewportUniforms();
    updateProjectionMatrix(camera.zoom);
    updateViewMatrix();
    cullingViewMatrix_ = uniforms.viewMatrix;
    inverseCullingViewMatrix_ = uniforms.inverseViewMatrix;
    uniforms.cullingViewMatrix = cullingViewMatrix_;
    uniforms.inverseCullingViewMatrix = inverseCullingViewMatrix_;
    gpu.setDebugWorld(voxelStreaming_.world());
    voxelStreaming_.start(camera.position, gpu.uploadedMeshRevision());

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
    voxelStreaming_.stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    gui.terminateImGUI();
}

void Application::MainLoop() {
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
    updateTargetedBlockSelection(cursorCaptured && !io.WantCaptureMouse);
    processBlockInteractions();

    // Early exit if frame budget is already exceeded
    float frameStartTime = currentFrame;

    const glm::mat4 viewGPU = glm::lookAt(camera.position, camera.position + camera.front, camera.up);
    uniforms.viewMatrix = viewGPU;
    uniforms.inverseViewMatrix = glm::inverse(viewGPU);
    const bool freezeCullingBeforeUi = gui.isCullingCameraFrozen();
    updateCullingCameraMatrices(viewGPU, freezeCullingBeforeUi);

    const auto streamUpdateStart = std::chrono::steady_clock::now();
    const float projectionYScale = std::abs(uniforms.projectionMatrix[1][1]);
    const int32_t framebufferHeight = std::max(1, gpu.getContext()->height);
    float sseProjectionScale = 0.5f * static_cast<float>(framebufferHeight) * projectionYScale;
    if (!std::isfinite(sseProjectionScale) || sseProjectionScale <= 0.0f) {
        sseProjectionScale = 390.0f;
    }

    voxelStreaming_.updateCamera(camera.position, sseProjectionScale);
    if (auto delta = voxelStreaming_.tryConsumePreparedDelta()) {
        gpu.queueMeshDelta(std::move(*delta));
    }
    voxelStreaming_.recordMainUpdateDurationNs(
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - streamUpdateStart
        ).count())
    );

    const RuntimeTimingSnapshot gpuTiming = gpu.getRuntimeTimingSnapshot();
    const RuntimeTimingSnapshot streamingTiming = voxelStreaming_.getRuntimeTimingSnapshot();
    runtimeTimingSnapshot_ = mergeRuntimeTimingSnapshots(gpuTiming, streamingTiming);

    gui.renderImGUI(uniforms, deltaTime, frameTimes, camera, frameTime, runtimeTimingSnapshot_);
    const bool freezeCullingAfterUi = gui.isCullingCameraFrozen();
    updateCullingCameraMatrices(viewGPU, freezeCullingAfterUi);
    updateViewportUniforms();
    buf->writeBuffer("uniform_buffer", 0, &uniforms, sizeof(FrameUniforms));
    
    gpu.renderFrame(uniforms);

    if (framePacer_) {
        const FramePacingResult pacing = framePacer_->finalizeFrameAndPace(
            currentFrame,
            frameStartTime,
            frameTimes
        );
        frameTime = pacing.frameTime;
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

void Application::updateViewportUniforms() {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    width = std::max(width, 1);
    height = std::max(height, 1);

    uniforms.viewportParams[0] = static_cast<float>(width);
    uniforms.viewportParams[1] = static_cast<float>(height);
    uniforms.viewportParams[2] = 2.0f / uniforms.viewportParams[0];
    uniforms.viewportParams[3] = 2.0f / uniforms.viewportParams[1];
}

void Application::updateViewMatrix() {
    uniforms.viewMatrix = glm::lookAt(camera.position, camera.position + camera.front, camera.up);
    uniforms.inverseViewMatrix = glm::inverse(uniforms.viewMatrix);
    buf->writeBuffer("uniform_buffer", offsetof(FrameUniforms, viewMatrix), &uniforms.viewMatrix, sizeof(FrameUniforms::viewMatrix));
}

void Application::updateCullingCameraMatrices(const glm::mat4& renderViewMatrix, bool freezeCullingCamera) {
    if (!freezeCullingCamera) {
        cullingViewMatrix_ = renderViewMatrix;
        inverseCullingViewMatrix_ = glm::inverse(renderViewMatrix);
    }

    uniforms.cullingViewMatrix = cullingViewMatrix_;
    uniforms.inverseCullingViewMatrix = inverseCullingViewMatrix_;
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
            if (!cursorCaptured) {
                // First left click focuses the window and enables camera control.
                mouseState.firstMouse = true;
                cursorCaptured = true;
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                glfwSetCursorPos(window, mouseState.lastX, mouseState.lastY);
            } else {
                mouseState.leftClickRequested = true;
            }
        }
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            if (cursorCaptured) {
                mouseState.rightClickRequested = true;
            }
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

void Application::onKey(int key, int /* scancode */, int action, int /* mods */) {
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
            mouseState.leftClickRequested = false;
            mouseState.rightClickRequested = false;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
        break;
    }
}

bool Application::IsRunning() {
    return !glfwWindowShouldClose(window);
}

bool Application::raycastTargetBlock(float maxDistance, VoxelRaycastHit& outHit) const {
    outHit = VoxelRaycastHit{};
    const World* world = voxelStreaming_.world();
    if (world == nullptr) {
        return false;
    }

    glm::vec3 rayDir = camera.front;
    const float rayLengthSq = glm::dot(rayDir, rayDir);
    if (rayLengthSq <= 0.000001f) {
        return false;
    }
    rayDir /= std::sqrt(rayLengthSq);

    const glm::vec3 rayOrigin = camera.position;
    BlockCoord current{
        static_cast<int32_t>(std::floor(rayOrigin.x)),
        static_cast<int32_t>(std::floor(rayOrigin.y)),
        static_cast<int32_t>(std::floor(rayOrigin.z))
    };
    BlockCoord previous = current;

    auto initialAxisStep = [](float originComponent, float directionComponent, int32_t& step, float& tMax, float& tDelta) {
        if (directionComponent > 0.0f) {
            step = 1;
            const float nextBoundary = std::floor(originComponent) + 1.0f;
            tMax = (nextBoundary - originComponent) / directionComponent;
            tDelta = 1.0f / directionComponent;
        } else if (directionComponent < 0.0f) {
            step = -1;
            const float previousBoundary = std::floor(originComponent);
            tMax = (originComponent - previousBoundary) / -directionComponent;
            tDelta = 1.0f / -directionComponent;
        } else {
            step = 0;
            tMax = std::numeric_limits<float>::infinity();
            tDelta = std::numeric_limits<float>::infinity();
        }
    };

    int32_t stepX = 0;
    int32_t stepY = 0;
    int32_t stepZ = 0;
    float tMaxX = 0.0f;
    float tMaxY = 0.0f;
    float tMaxZ = 0.0f;
    float tDeltaX = 0.0f;
    float tDeltaY = 0.0f;
    float tDeltaZ = 0.0f;
    initialAxisStep(rayOrigin.x, rayDir.x, stepX, tMaxX, tDeltaX);
    initialAxisStep(rayOrigin.y, rayDir.y, stepY, tMaxY, tDeltaY);
    initialAxisStep(rayOrigin.z, rayDir.z, stepZ, tMaxZ, tDeltaZ);

    float traveled = 0.0f;
    while (traveled <= maxDistance) {
        BlockMaterial block = UnpackedBlockMaterial{}.pack();
        if (world->tryGetBlock(current, block) && block.unpack().id != 0u) {
            outHit.hit = true;
            outHit.breakCoord = current;
            outHit.placeCoord = previous;
            return true;
        }

        previous = current;
        if (tMaxX <= tMaxY && tMaxX <= tMaxZ) {
            current.v.x += stepX;
            traveled = tMaxX;
            tMaxX += tDeltaX;
        } else if (tMaxY <= tMaxZ) {
            current.v.y += stepY;
            traveled = tMaxY;
            tMaxY += tDeltaY;
        } else {
            current.v.z += stepZ;
            traveled = tMaxZ;
            tMaxZ += tDeltaZ;
        }
    }

    return false;
}

void Application::updateTargetedBlockSelection(bool enabled) {
    if (!enabled) {
        currentTargetBlockHit_.reset();
        gpu.setSelectionOutlineBlock(std::nullopt);
        return;
    }

    VoxelRaycastHit rayHit{};
    if (raycastTargetBlock(8.0f, rayHit) && rayHit.hit) {
        currentTargetBlockHit_ = rayHit;
        gpu.setSelectionOutlineBlock(rayHit.breakCoord);
        return;
    }

    currentTargetBlockHit_.reset();
    gpu.setSelectionOutlineBlock(std::nullopt);
}

void Application::processBlockInteractions() {
    if (!cursorCaptured || !currentTargetBlockHit_.has_value()) {
        mouseState.leftClickRequested = false;
        mouseState.rightClickRequested = false;
        return;
    }

    const VoxelRaycastHit& rayHit = *currentTargetBlockHit_;

    if (mouseState.leftClickRequested) {
        voxelStreaming_.breakBlock(rayHit.breakCoord);
    }

    if (mouseState.rightClickRequested) {
        static const BlockMaterial kPlacementBlock = MaterialRegistry::resolveBlockOr(
            "glowstone",
            UnpackedBlockMaterial{6u, 0, Direction::PlusZ, 0}.pack()
        );
        const BlockMaterial placementBlock = kPlacementBlock;
        voxelStreaming_.placeBlock(rayHit.placeCoord, placementBlock);
    }

    mouseState.leftClickRequested = false;
    mouseState.rightClickRequested = false;
}
