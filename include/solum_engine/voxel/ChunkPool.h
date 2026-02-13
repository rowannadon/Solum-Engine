#pragma once

#include "solum_engine/resources/Constants.h"
#include "solum_engine/voxel/BlockMaterial.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

struct UncompressedChunkHandle {
    uint32_t index = std::numeric_limits<uint32_t>::max();
    uint32_t generation = 0;

    bool isValid() const {
        return index != std::numeric_limits<uint32_t>::max();
    }

    static UncompressedChunkHandle invalid() {
        return {};
    }
};

class ChunkPool {
public:
    explicit ChunkPool(std::size_t capacity = 1024);

    UncompressedChunkHandle allocate();
    bool release(UncompressedChunkHandle handle);

    bool pin(UncompressedChunkHandle handle);
    bool unpin(UncompressedChunkHandle handle);

    BlockMaterial* data(UncompressedChunkHandle handle);
    const BlockMaterial* data(UncompressedChunkHandle handle) const;

    bool isAllocated(UncompressedChunkHandle handle) const;
    uint32_t pinCount(UncompressedChunkHandle handle) const;

    std::size_t capacity() const;
    std::size_t freeSlots() const;

private:
    struct SlotMeta {
        uint32_t generation = 1;
        uint32_t pinCount = 0;
        bool allocated = false;
    };

    bool validateLocked(UncompressedChunkHandle handle) const;

    std::size_t capacity_;
    mutable std::mutex mutex_;
    std::vector<BlockMaterial> data_;
    std::vector<SlotMeta> slots_;
    std::vector<uint32_t> freeList_;
};
