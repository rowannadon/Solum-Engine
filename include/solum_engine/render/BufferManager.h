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
    uint32_t frameWriteCount_{0};
    uint32_t maxWriteCallsPerFrame_{0};

    // Grow-batch state: when active, growBuffer records copies into this
    // encoder instead of submitting immediately.
    wgpu::CommandEncoder growEncoder_;
    std::vector<wgpu::Buffer> growOldBuffers_;

public:
#ifdef __APPLE__
    static constexpr size_t kDefaultFrameUploadBudgetBytes = 4u * 1024u * 1024u;
    static constexpr uint32_t kDefaultMaxWriteCallsPerFrame = 16u;
#else
    static constexpr size_t kDefaultFrameUploadBudgetBytes = 128u * 1024u * 1024u;
    static constexpr uint32_t kDefaultMaxWriteCallsPerFrame = 0u;  // 0 = unlimited
#endif

    struct Config {
        size_t frameUploadBudgetBytes = kDefaultFrameUploadBudgetBytes;
        uint32_t maxWriteCallsPerFrame = kDefaultMaxWriteCallsPerFrame;
    };

    BufferManager(wgpu::Device d, wgpu::Queue q)
        : BufferManager(d, q, Config{}) {}

    BufferManager(wgpu::Device d, wgpu::Queue q, Config config)
        : device(d),
          queue(q),
          frameUploadBudgetBytes_(config.frameUploadBudgetBytes > 0 ? config.frameUploadBudgetBytes
                                                                   : kDefaultFrameUploadBudgetBytes),
          maxWriteCallsPerFrame_(config.maxWriteCallsPerFrame) {}

    // Existing methods
    wgpu::Buffer createBuffer(const std::string& bufferName, const wgpu::BufferDescriptor& config);
    wgpu::Buffer getBuffer(const std::string& bufferName) const;
    void writeBuffer(const std::string& bufferName, uint64_t bufferOffset, const void* data, size_t size);

    /// Begin a batched grow operation. All subsequent growBuffer() calls will
    /// record their copy commands into a shared encoder instead of submitting
    /// individually. Call endGrowBatch() to submit all copies in one command buffer.
    void beginGrowBatch();

    /// Replace a named buffer with a new, larger one, copying old contents via
    /// GPU-side blit. If a grow batch is active (beginGrowBatch was called),
    /// the copy is deferred until endGrowBatch(). Otherwise submits immediately.
    wgpu::Buffer growBuffer(const std::string& bufferName,
                            const wgpu::BufferDescriptor& newDesc);

    /// Submit all batched grow copies in a single command buffer and release
    /// old buffers. No-op if no grow batch is active.
    void endGrowBatch();

    void deleteBuffer(const std::string& bufferName);
    void terminate();

    void resetFrameBudget() { frameBytesWritten_ = 0; frameWriteCount_ = 0; }
    size_t frameBytesWritten() const { return frameBytesWritten_; }
    size_t frameUploadBudgetBytes() const { return frameUploadBudgetBytes_; }
    uint32_t frameWriteCount() const { return frameWriteCount_; }
    bool isOverBudget() const {
        if (frameBytesWritten_ >= frameUploadBudgetBytes_) return true;
        if (maxWriteCallsPerFrame_ > 0u && frameWriteCount_ >= maxWriteCallsPerFrame_) return true;
        return false;
    }
};

#endif
