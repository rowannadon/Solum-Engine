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

    size_t frameBytesWritten_{0};

public:
    BufferManager(wgpu::Device d, wgpu::Queue q) : device(d), queue(q) {}

    // Existing methods
    wgpu::Buffer createBuffer(const std::string& bufferName, const wgpu::BufferDescriptor& config);
    wgpu::Buffer getBuffer(const std::string& bufferName) const;
    void writeBuffer(const std::string& bufferName, uint64_t bufferOffset, const void* data, size_t size);

    void deleteBuffer(const std::string& bufferName);
    void terminate();

    void resetFrameBudget() { frameBytesWritten_ = 0; }
    size_t frameBytesWritten() const { return frameBytesWritten_; }

#ifdef __APPLE__
    static constexpr size_t kFrameUploadBudget = 2u * 1024u * 1024u;
#else
    static constexpr size_t kFrameUploadBudget = 128u * 1024u * 1024u;
#endif
    bool isOverBudget() const { return frameBytesWritten_ >= kFrameUploadBudget; }
};

#endif
