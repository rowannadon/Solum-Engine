// Simplified BufferManager.h - removes redundant variable size class functionality
#ifndef BUFFER_MANAGER
#define BUFFER_MANAGER
#include <unordered_map>
#include <webgpu/webgpu.hpp>
#include <vector>

using namespace wgpu;

class BufferManager {
private:
    std::unordered_map<std::string, Buffer> buffers;

    Device device;
    Queue queue;

public:
    BufferManager(Device d, Queue q) : device(d), queue(q) {}

    // Existing methods
    Buffer createBuffer(std::string bufferName, BufferDescriptor config);
    Buffer getBuffer(std::string bufferName);
    void writeBuffer(const std::string bufferName, uint64_t bufferOffset, void* data, size_t size);

    void deleteBuffer(std::string bufferName);
    void terminate();

};

#endif
