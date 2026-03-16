#pragma once

#include <cstdint>
#include <map>

/// O(log n) best-fit range allocator with automatic coalescing on free.
/// Uses dual-indexed ordered sets: one by size (for allocation) and one by
/// offset (for coalescing). Replaces the previous O(n) linear-scan free-list.
class MeshletRangeAllocator {
public:
    MeshletRangeAllocator() = default;
    explicit MeshletRangeAllocator(uint32_t capacity);

    /// Allocate a contiguous range of `size` units. Returns true on success
    /// and writes the starting offset to `outOffset`. Uses best-fit strategy.
    /// Complexity: O(log n) where n = number of free ranges.
    bool allocate(uint32_t size, uint32_t& outOffset);

    /// Free a previously allocated range. Automatically coalesces with
    /// adjacent free ranges. Complexity: O(log n).
    void free(uint32_t offset, uint32_t size);

    /// Reset allocator to a single free range spanning [0, capacity).
    void reset(uint32_t capacity);

    uint32_t capacity() const noexcept { return capacity_; }
    uint32_t totalFree() const noexcept { return totalFree_; }
    size_t fragmentCount() const noexcept { return freeByOffset_.size(); }
    bool empty() const noexcept { return freeByOffset_.empty(); }

private:
    void eraseSizeEntry(uint32_t size, uint32_t offset);

    // Free ranges indexed by size for best-fit lookup.
    // Multiple ranges can have the same size, so we use multimap.
    std::multimap<uint32_t, uint32_t> freeBySize_;  // size -> offset

    // Free ranges indexed by offset for coalescing on free.
    std::map<uint32_t, uint32_t> freeByOffset_;      // offset -> size

    uint32_t capacity_ = 0;
    uint32_t totalFree_ = 0;
};
