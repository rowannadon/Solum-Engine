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
    size_t frameUploadBudgetBytes_{0};

public:
#ifdef __APPLE__
    static constexpr size_t kDefaultFrameUploadBudgetBytes = 16u * 1024u * 1024u;
#else
    static constexpr size_t kDefaultFrameUploadBudgetBytes = 128u * 1024u * 1024u;
#endif

    struct Config {
        size_t frameUploadBudgetBytes = kDefaultFrameUploadBudgetBytes;
    };

    BufferManager(wgpu::Device d, wgpu::Queue q)
        : BufferManager(d, q, Config{}) {}

    BufferManager(wgpu::Device d, wgpu::Queue q, Config config)
        : device(d),
          queue(q),
          frameUploadBudgetBytes_(config.frameUploadBudgetBytes > 0 ? config.frameUploadBudgetBytes
                                                                   : kDefaultFrameUploadBudgetBytes) {}

    // Existing methods
    wgpu::Buffer createBuffer(const std::string& bufferName, const wgpu::BufferDescriptor& config);
    wgpu::Buffer getBuffer(const std::string& bufferName) const;
    void writeBuffer(const std::string& bufferName, uint64_t bufferOffset, const void* data, size_t size);

    /// Replace a named buffer with a new, larger one, copying old contents via
    /// GPU-side blit. Returns the new buffer (also stored under bufferName).
    /// The old buffer is released after the copy command is submitted.
    wgpu::Buffer growBuffer(const std::string& bufferName,
                            const wgpu::BufferDescriptor& newDesc);

    void deleteBuffer(const std::string& bufferName);
    void terminate();

    void resetFrameBudget() { frameBytesWritten_ = 0; }
    size_t frameBytesWritten() const { return frameBytesWritten_; }
    size_t frameUploadBudgetBytes() const { return frameUploadBudgetBytes_; }
    bool isOverBudget() const { return frameBytesWritten_ >= frameUploadBudgetBytes_; }
};

#endif
