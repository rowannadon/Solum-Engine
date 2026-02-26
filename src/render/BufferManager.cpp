// Simplified BufferManager.cpp - removes redundant functionality
#include "solum_engine/render/BufferManager.h"

void BufferManager::deleteBuffer(const std::string& bufferName) {
    const auto it = buffers.find(bufferName);
    if (it == buffers.end()) {
        return;
    }

    if (it->second) {
        it->second.destroy();
        it->second.release();
    }
    buffers.erase(it);
}

void BufferManager::writeBuffer(const std::string& bufferName, uint64_t bufferOffset, const void* data, size_t size) {
    const wgpu::Buffer buffer = getBuffer(bufferName);
    if (buffer) {
        queue.writeBuffer(buffer, bufferOffset, data, size);
    }
}

wgpu::Buffer BufferManager::createBuffer(const std::string& bufferName, const wgpu::BufferDescriptor& config) {
    auto existing = buffers.find(bufferName);
    if (existing != buffers.end() && existing->second) {
        existing->second.destroy();
        existing->second.release();
        buffers.erase(existing);
    }

    wgpu::Buffer buffer = device.createBuffer(config);
    buffers[bufferName] = buffer;
    return buffer;
}

wgpu::Buffer BufferManager::getBuffer(const std::string& bufferName) const {
    auto buffer = buffers.find(bufferName);
    if (buffer != buffers.end()) {
        return buffer->second;
    }
    return nullptr;
}

void BufferManager::terminate() {
    // Clean up regular buffers
    for (auto& pair : buffers) {
        if (pair.second) {
            pair.second.destroy();
            pair.second.release();
        }
    }

    // Clear containers
    buffers.clear();
}
