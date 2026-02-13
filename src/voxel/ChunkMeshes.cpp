#include "solum_engine/voxel/ChunkMeshes.h"

#include <utility>

MeshHandle MeshHandleTable::create(MeshData meshData) {
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
    entry.mesh = std::move(meshData);

    return MeshHandle{slot, entry.generation};
}

MeshHandle MeshHandleTable::updateOrCreate(MeshHandle handle, MeshData meshData) {
    std::scoped_lock lock(mutex_);

    if (isValidLocked(handle)) {
        entries_[handle.index].mesh = std::move(meshData);
        return handle;
    }

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
    entry.mesh = std::move(meshData);

    return MeshHandle{slot, entry.generation};
}

bool MeshHandleTable::release(MeshHandle handle) {
    std::scoped_lock lock(mutex_);
    if (!isValidLocked(handle)) {
        return false;
    }

    Entry& entry = entries_[handle.index];
    entry.allocated = false;
    entry.generation += 1u;
    entry.mesh = MeshData{};
    freeList_.push_back(handle.index);
    return true;
}

std::optional<MeshData> MeshHandleTable::copy(MeshHandle handle) const {
    std::scoped_lock lock(mutex_);
    if (!isValidLocked(handle)) {
        return std::nullopt;
    }

    return entries_[handle.index].mesh;
}

bool MeshHandleTable::isValidLocked(MeshHandle handle) const {
    if (!handle.isValid()) {
        return false;
    }
    if (handle.index >= entries_.size()) {
        return false;
    }

    const Entry& entry = entries_[handle.index];
    return entry.allocated && entry.generation == handle.generation;
}
