#include "solum_engine/voxel/CompressedStore.h"

#include <utility>

CompressedStore::CompressedStore(std::size_t initialCapacity) {
    entries_.reserve(initialCapacity);
    freeList_.reserve(initialCapacity);
}

CompressedChunkHandle CompressedStore::store(std::vector<uint8_t> bytes, uint8_t codecId) {
    std::scoped_lock lock(mutex_);

    uint32_t slot = 0;
    if (!freeList_.empty()) {
        slot = freeList_.back();
        freeList_.pop_back();
    } else {
        slot = static_cast<uint32_t>(entries_.size());
        entries_.push_back({});
    }

    Entry& entry = entries_[slot];
    entry.allocated = true;
    entry.codecId = codecId;
    entry.data = std::move(bytes);

    return CompressedChunkHandle{slot, entry.generation};
}

bool CompressedStore::release(CompressedChunkHandle handle) {
    std::scoped_lock lock(mutex_);
    if (!validateLocked(handle)) {
        return false;
    }

    Entry& entry = entries_[handle.index];
    entry.allocated = false;
    entry.generation += 1u;
    entry.codecId = 0;
    entry.data.clear();

    freeList_.push_back(handle.index);
    return true;
}

CompressedChunkView CompressedStore::view(CompressedChunkHandle handle) const {
    std::scoped_lock lock(mutex_);
    if (!validateLocked(handle)) {
        return {};
    }

    const Entry& entry = entries_[handle.index];
    return CompressedChunkView{entry.data.data(), entry.data.size(), entry.codecId};
}

std::vector<uint8_t> CompressedStore::copy(CompressedChunkHandle handle) const {
    std::scoped_lock lock(mutex_);
    if (!validateLocked(handle)) {
        return {};
    }

    return entries_[handle.index].data;
}

std::size_t CompressedStore::liveEntries() const {
    std::scoped_lock lock(mutex_);
    return entries_.size() - freeList_.size();
}

bool CompressedStore::validateLocked(CompressedChunkHandle handle) const {
    if (!handle.isValid()) {
        return false;
    }
    if (handle.index >= entries_.size()) {
        return false;
    }

    const Entry& entry = entries_[handle.index];
    return entry.allocated && entry.generation == handle.generation;
}
