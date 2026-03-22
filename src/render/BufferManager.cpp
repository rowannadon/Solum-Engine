// Simplified BufferManager.cpp - removes redundant functionality
#include "solum_engine/render/BufferManager.h"

#include <algorithm>

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
        frameBytesWritten_ += size;
    }
}

wgpu::Buffer BufferManager::createBuffer(const std::string& bufferName, const wgpu::BufferDescriptor& config) {
    auto existing = buffers.find(bufferName);
    if (existing != buffers.end() && existing->second) {
        existing->second.release();
        buffers.erase(existing);
    }

    wgpu::Buffer buffer = device.createBuffer(config);
    buffers[bufferName] = buffer;
    return buffer;
}

void BufferManager::beginGrowBatch() {
    if (growEncoder_) return;  // Already in a batch.

    wgpu::CommandEncoderDescriptor encDesc = wgpu::Default;
    encDesc.label = wgpu::StringView("Buffer grow batch");
    growEncoder_ = device.createCommandEncoder(encDesc);
}

wgpu::Buffer BufferManager::growBuffer(const std::string& bufferName,
                                       const wgpu::BufferDescriptor& newDesc) {
    wgpu::Buffer oldBuffer = getBuffer(bufferName);
    const uint64_t oldSize = oldBuffer ? oldBuffer.getSize() : 0u;

    // Ensure the new buffer can also be a copy destination so we can blit into it.
    wgpu::BufferDescriptor descWithCopy = newDesc;
    descWithCopy.usage = newDesc.usage | wgpu::BufferUsage::CopyDst;

    // Detach old buffer from the map before creating the new one.
    if (oldBuffer) {
        buffers.erase(bufferName);
    }

    wgpu::Buffer newBuffer = device.createBuffer(descWithCopy);
    buffers[bufferName] = newBuffer;

    // GPU-side copy of old contents into the new buffer.
    // The old buffer must have been created with CopySrc usage.
    if (oldBuffer && oldSize > 0u && newBuffer) {
        const uint64_t copySize = std::min(oldSize, newDesc.size);

        if (growEncoder_) {
            // Batched: record into the shared encoder; defer old buffer release.
            growEncoder_.copyBufferToBuffer(oldBuffer, 0u, newBuffer, 0u, copySize);
            growOldBuffers_.push_back(oldBuffer);
        } else {
            // Unbatched: submit immediately.
            wgpu::CommandEncoderDescriptor encDesc = wgpu::Default;
            encDesc.label = wgpu::StringView("Buffer grow copy");
            wgpu::CommandEncoder encoder = device.createCommandEncoder(encDesc);

            encoder.copyBufferToBuffer(oldBuffer, 0u, newBuffer, 0u, copySize);

            wgpu::CommandBufferDescriptor cbDesc = wgpu::Default;
            cbDesc.label = wgpu::StringView("Buffer grow copy cmd");
            wgpu::CommandBuffer cmd = encoder.finish(cbDesc);
            encoder.release();

            queue.submit(1, &cmd);
            cmd.release();
            oldBuffer.release();
        }
    } else if (oldBuffer) {
        oldBuffer.release();
    }

    return newBuffer;
}

void BufferManager::endGrowBatch() {
    if (!growEncoder_) return;

    wgpu::CommandBufferDescriptor cbDesc = wgpu::Default;
    cbDesc.label = wgpu::StringView("Buffer grow batch cmd");
    wgpu::CommandBuffer cmd = growEncoder_.finish(cbDesc);
    growEncoder_.release();
    growEncoder_ = nullptr;

    queue.submit(1, &cmd);
    cmd.release();

    // Release old buffers now that the copy has been submitted.
    for (wgpu::Buffer& buf : growOldBuffers_) {
        buf.release();
    }
    growOldBuffers_.clear();
}

wgpu::Buffer BufferManager::getBuffer(const std::string& bufferName) const {
    auto buffer = buffers.find(bufferName);
    if (buffer != buffers.end()) {
        return buffer->second;
    }
    return nullptr;
}

void BufferManager::terminate() {
    // Clean up any pending grow batch.
    if (growEncoder_) {
        growEncoder_.release();
        growEncoder_ = nullptr;
    }
    for (wgpu::Buffer& buf : growOldBuffers_) {
        buf.release();
    }
    growOldBuffers_.clear();

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
