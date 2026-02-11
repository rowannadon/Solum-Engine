#pragma once

#include <GLFW/glfw3.h>
#include <glfw3webgpu.h>

#include <webgpu/webgpu.hpp>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include "solum_engine/render/Uniforms.h"

using namespace wgpu;
using glm::mat4x4;
using glm::vec4;
using glm::vec3;
using glm::ivec3;

struct RenderConfig {
    int width = 1280;
    int height = 720;
    const char* title = "Voxel Engine";
    int samples = 4;
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
    uint32_t uniformStride = 0;

    Device getDevice() { return device; }
    Queue getQueue() { return queue; }
    GLFWwindow* getWindow() { return window; }
    Surface getSurface() { return surface; }
    TextureFormat getSurfaceFormat() { return surfaceFormat; }

    bool initialize(const RenderConfig& config);
    bool configureSurface();
    void unconfigureSurface();
    Limits GetRequiredLimits(Adapter adapter) const;
    uint32_t ceilToNextMultiple(uint32_t value, uint32_t step) const;
    void terminate();
};