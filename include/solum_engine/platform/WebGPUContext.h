#pragma once

#include <GLFW/glfw3.h>
#include <glfw3webgpu/glfw3webgpu.h>

#include <atomic>
#include <string>
#include <webgpu/webgpu.hpp>

struct RenderConfig {
    int width = 1280;
    int height = 720;
    std::string title = "Voxel Engine";
    std::string presentMode = "fifo";
};

class WebGPUContext {
public:
    wgpu::Instance instance;
    wgpu::Device device;
    wgpu::Queue queue;
    wgpu::Surface surface;
    wgpu::Adapter adapter;
    GLFWwindow* window = nullptr;
    int width = 0;
    int height = 0;
    wgpu::TextureFormat surfaceFormat = wgpu::TextureFormat::Undefined;

    wgpu::Device getDevice() { return device; }
    wgpu::Queue getQueue() { return queue; }
    GLFWwindow* getWindow() { return window; }
    wgpu::Surface getSurface() { return surface; }
    wgpu::TextureFormat getSurfaceFormat() { return surfaceFormat; }

    bool isDeviceLost() const { return deviceLost_.load(std::memory_order_acquire); }

    bool initialize(const RenderConfig& config);
    bool configureSurface();
    void unconfigureSurface();
    wgpu::Limits GetRequiredLimits(wgpu::Adapter adapter) const;
    void terminate();

private:
    std::atomic<bool> deviceLost_{false};
    std::string preferredPresentMode_ = "fifo";
};
