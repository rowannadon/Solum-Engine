#include "solum_engine/render/MeshletRangeAllocator.h"

MeshletRangeAllocator::MeshletRangeAllocator(uint32_t capacity)
    : capacity_(capacity), totalFree_(capacity) {
    if (capacity > 0) {
        freeBySize_.emplace(capacity, 0u);
        freeByOffset_.emplace(0u, capacity);
    }
}

bool MeshletRangeAllocator::allocate(uint32_t size, uint32_t& outOffset) {
    if (size == 0u) {
        outOffset = 0u;
        return true;
    }

    // Best-fit: find the smallest free range that can satisfy the request.
    auto it = freeBySize_.lower_bound(size);
    if (it == freeBySize_.end()) {
        return false;
    }

    const uint32_t foundSize = it->first;
    const uint32_t foundOffset = it->second;

    // Remove from both indices.
    freeBySize_.erase(it);
    freeByOffset_.erase(foundOffset);
    totalFree_ -= foundSize;

    outOffset = foundOffset;

    // If the range was larger than needed, return the remainder.
    if (foundSize > size) {
        const uint32_t remOffset = foundOffset + size;
        const uint32_t remSize = foundSize - size;
        freeBySize_.emplace(remSize, remOffset);
        freeByOffset_.emplace(remOffset, remSize);
        totalFree_ += remSize;
    }

    return true;
}

void MeshletRangeAllocator::free(uint32_t offset, uint32_t size) {
    if (size == 0u) {
        return;
    }

    uint32_t mergedOffset = offset;
    uint32_t mergedSize = size;

    // Try to merge with the predecessor (range ending exactly at our offset).
    auto nextIt = freeByOffset_.lower_bound(offset);
    if (nextIt != freeByOffset_.begin()) {
        auto prevIt = std::prev(nextIt);
        if (prevIt->first + prevIt->second == offset) {
            mergedOffset = prevIt->first;
            mergedSize += prevIt->second;
            eraseSizeEntry(prevIt->second, prevIt->first);
            freeByOffset_.erase(prevIt);
        }
    }

    // Try to merge with the successor (range starting exactly at our end).
    // Re-lookup since iterators may have been invalidated by the erase above.
    nextIt = freeByOffset_.lower_bound(mergedOffset + mergedSize);
    if (nextIt != freeByOffset_.end() && nextIt->first == mergedOffset + mergedSize) {
        mergedSize += nextIt->second;
        eraseSizeEntry(nextIt->second, nextIt->first);
        freeByOffset_.erase(nextIt);
    }

    // Insert the (possibly merged) free range.
    freeByOffset_.emplace(mergedOffset, mergedSize);
    freeBySize_.emplace(mergedSize, mergedOffset);
    totalFree_ += size;
}

void MeshletRangeAllocator::reset(uint32_t capacity) {
    freeBySize_.clear();
    freeByOffset_.clear();
    capacity_ = capacity;
    totalFree_ = capacity;
    if (capacity > 0) {
        freeBySize_.emplace(capacity, 0u);
        freeByOffset_.emplace(0u, capacity);
    }
}

void MeshletRangeAllocator::grow(uint32_t newCapacity) {
    if (newCapacity <= capacity_) {
        return;
    }
    // Free the range [oldCapacity, newCapacity). The free() method
    // handles coalescing with any adjacent free range at the old end.
    const uint32_t oldCapacity = capacity_;
    const uint32_t added = newCapacity - oldCapacity;
    capacity_ = newCapacity;
    free(oldCapacity, added);
}

void MeshletRangeAllocator::eraseSizeEntry(uint32_t size, uint32_t offset) {
    auto [begin, end] = freeBySize_.equal_range(size);
    for (auto it = begin; it != end; ++it) {
        if (it->second == offset) {
            freeBySize_.erase(it);
            return;
        }
    }
}
