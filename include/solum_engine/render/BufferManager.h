// Simplified BufferManager.h - removes redundant variable size class functionality
#ifndef BUFFER_MANAGER
#define BUFFER_MANAGER
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <webgpu/webgpu.hpp>
#include <vector>

class BufferManager {
private:
    std::unordered_map<std::string, wgpu::Buffer> buffers;

    wgpu::Device device;
    wgpu::Queue queue;

public:
    BufferManager(wgpu::Device d, wgpu::Queue q) : device(d), queue(q) {}

    // Existing methods
    wgpu::Buffer createBuffer(const std::string& bufferName, const wgpu::BufferDescriptor& config);
    wgpu::Buffer getBuffer(const std::string& bufferName) const;
    void writeBuffer(const std::string& bufferName, uint64_t bufferOffset, const void* data, size_t size);

    void deleteBuffer(const std::string& bufferName);
    void terminate();

};

#endif
