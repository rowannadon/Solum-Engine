// Simplified BufferManager.cpp - removes redundant functionality
#include "solum_engine/render/BufferManager.h"
#include <iostream>

void BufferManager::deleteBuffer(std::string bufferName) {
    Buffer buffer = getBuffer(bufferName);
    if (buffer) {
        buffer.destroy();
        buffer.release();
        buffers.erase(bufferName);
    }
}

void BufferManager::writeBuffer(const std::string bufferName, uint64_t bufferOffset, const void* data, size_t size) {
    Buffer buffer = getBuffer(bufferName);
    if (buffer) {
        queue.writeBuffer(buffer, bufferOffset, data, size);
    }
}

Buffer BufferManager::createBuffer(std::string bufferName, BufferDescriptor config) {
    auto existing = buffers.find(bufferName);
    if (existing != buffers.end() && existing->second) {
        existing->second.release();
        buffers.erase(existing);
    }

    Buffer buffer = device.createBuffer(config);
    buffers[bufferName] = buffer;
    return buffer;
}

Buffer BufferManager::getBuffer(std::string bufferName) {
    auto buffer = buffers.find(bufferName);
    if (buffer != buffers.end()) {
        return buffer->second;
    }
    return nullptr;
}

void BufferManager::terminate() {
    // Clean up regular buffers
    for (auto pair : buffers) {
        if (pair.second) {
            pair.second.destroy();
            pair.second.release();
        }
    }

    // Clear containers
    buffers.clear();
}
