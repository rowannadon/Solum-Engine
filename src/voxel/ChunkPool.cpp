#include "solum_engine/voxel/ChunkPool.h"

ChunkPool::ChunkPool(std::size_t capacity)
    : capacity_(capacity),
      data_(capacity * CHUNK_BLOCKS),
      slots_(capacity),
      freeList_(capacity) {
    for (std::size_t index = 0; index < capacity; ++index) {
        freeList_[index] = static_cast<uint32_t>(capacity - index - 1u);
    }
}

UncompressedChunkHandle ChunkPool::allocate() {
    std::scoped_lock lock(mutex_);
    if (freeList_.empty()) {
        return UncompressedChunkHandle::invalid();
    }

    const uint32_t slotIndex = freeList_.back();
    freeList_.pop_back();

    SlotMeta& slot = slots_[slotIndex];
    slot.allocated = true;
    slot.pinCount = 0;

    BlockMaterial* begin = data_.data() + static_cast<std::size_t>(slotIndex) * CHUNK_BLOCKS;
    for (std::size_t i = 0; i < CHUNK_BLOCKS; ++i) {
        begin[i] = BlockMaterial{};
    }

    return UncompressedChunkHandle{slotIndex, slot.generation};
}

bool ChunkPool::release(UncompressedChunkHandle handle) {
    std::scoped_lock lock(mutex_);
    if (!validateLocked(handle)) {
        return false;
    }

    SlotMeta& slot = slots_[handle.index];
    if (slot.pinCount > 0u) {
        return false;
    }

    slot.allocated = false;
    slot.generation += 1u;
    freeList_.push_back(handle.index);
    return true;
}

bool ChunkPool::pin(UncompressedChunkHandle handle) {
    std::scoped_lock lock(mutex_);
    if (!validateLocked(handle)) {
        return false;
    }

    slots_[handle.index].pinCount += 1u;
    return true;
}

bool ChunkPool::unpin(UncompressedChunkHandle handle) {
    std::scoped_lock lock(mutex_);
    if (!validateLocked(handle)) {
        return false;
    }

    SlotMeta& slot = slots_[handle.index];
    if (slot.pinCount == 0u) {
        return false;
    }

    slot.pinCount -= 1u;
    return true;
}

BlockMaterial* ChunkPool::data(UncompressedChunkHandle handle) {
    std::scoped_lock lock(mutex_);
    if (!validateLocked(handle)) {
        return nullptr;
    }

    return data_.data() + static_cast<std::size_t>(handle.index) * CHUNK_BLOCKS;
}

const BlockMaterial* ChunkPool::data(UncompressedChunkHandle handle) const {
    std::scoped_lock lock(mutex_);
    if (!validateLocked(handle)) {
        return nullptr;
    }

    return data_.data() + static_cast<std::size_t>(handle.index) * CHUNK_BLOCKS;
}

bool ChunkPool::isAllocated(UncompressedChunkHandle handle) const {
    std::scoped_lock lock(mutex_);
    return validateLocked(handle);
}

uint32_t ChunkPool::pinCount(UncompressedChunkHandle handle) const {
    std::scoped_lock lock(mutex_);
    if (!validateLocked(handle)) {
        return 0;
    }

    return slots_[handle.index].pinCount;
}

std::size_t ChunkPool::capacity() const {
    return capacity_;
}

std::size_t ChunkPool::freeSlots() const {
    std::scoped_lock lock(mutex_);
    return freeList_.size();
}

bool ChunkPool::validateLocked(UncompressedChunkHandle handle) const {
    if (!handle.isValid()) {
        return false;
    }
    if (handle.index >= slots_.size()) {
        return false;
    }

    const SlotMeta& slot = slots_[handle.index];
    return slot.allocated && slot.generation == handle.generation;
}
