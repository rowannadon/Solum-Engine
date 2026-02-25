#pragma once

#include <GLFW/glfw3.h>
#include <glfw3webgpu/glfw3webgpu.h>

#include <webgpu/webgpu.hpp>

using namespace wgpu;

struct RenderConfig {
    int width = 1280;
    int height = 720;
    const char* title = "Voxel Engine";
};

class WebGPUContext {
public:
    Instance instance;
    Device device;
    Queue queue;
    Surface surface;
    Adapter adapter;
    GLFWwindow* window;
    int width;
    int height;
    TextureFormat surfaceFormat = TextureFormat::Undefined;

    Device getDevice() { return device; }
    Queue getQueue() { return queue; }
    GLFWwindow* getWindow() { return window; }
    Surface getSurface() { return surface; }
    TextureFormat getSurfaceFormat() { return surfaceFormat; }

    bool initialize(const RenderConfig& config);
    bool configureSurface();
    void unconfigureSurface();
    Limits GetRequiredLimits(Adapter adapter) const;
    void terminate();
};
