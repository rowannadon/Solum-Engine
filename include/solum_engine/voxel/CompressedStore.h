#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

struct CompressedChunkHandle {
    uint32_t index = std::numeric_limits<uint32_t>::max();
    uint32_t generation = 0;

    bool isValid() const {
        return index != std::numeric_limits<uint32_t>::max();
    }

    static CompressedChunkHandle invalid() {
        return {};
    }
};

struct CompressedChunkView {
    const uint8_t* data = nullptr;
    std::size_t size = 0;
    uint8_t codecId = 0;

    bool isValid() const {
        return data != nullptr && size > 0;
    }
};

class CompressedStore {
public:
    explicit CompressedStore(std::size_t initialCapacity = 1024);

    CompressedChunkHandle store(std::vector<uint8_t> bytes, uint8_t codecId);
    bool release(CompressedChunkHandle handle);

    CompressedChunkView view(CompressedChunkHandle handle) const;
    std::vector<uint8_t> copy(CompressedChunkHandle handle) const;

    std::size_t liveEntries() const;

private:
    struct Entry {
        uint32_t generation = 1;
        bool allocated = false;
        uint8_t codecId = 0;
        std::vector<uint8_t> data;
    };

    bool validateLocked(CompressedChunkHandle handle) const;

    mutable std::mutex mutex_;
    std::vector<Entry> entries_;
    std::vector<uint32_t> freeList_;
};
