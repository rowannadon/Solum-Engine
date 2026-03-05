#pragma once

#include <array>
#include <cstddef>
#include <mutex>
#include <optional>

#include "solum_engine/voxel/StreamingUpload.h"

class UploadMailbox {
public:
    static constexpr size_t kCapacity = 2u;

    void pushLatest(MeshStreamingDelta&& upload) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (count_ == kCapacity) {
            head_ = (head_ + 1u) % kCapacity;
            --count_;
        }

        const size_t tail = (head_ + count_) % kCapacity;
        slots_[tail] = std::move(upload);
        ++count_;
    }

    std::optional<MeshStreamingDelta> tryConsume() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (count_ == 0u) {
            return std::nullopt;
        }

        std::optional<MeshStreamingDelta> out = std::move(slots_[head_]);
        slots_[head_].reset();
        head_ = (head_ + 1u) % kCapacity;
        --count_;
        return out;
    }

    bool hasPending() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_ > 0u;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::optional<MeshStreamingDelta>& slot : slots_) {
            slot.reset();
        }
        head_ = 0u;
        count_ = 0u;
    }

private:
    mutable std::mutex mutex_;
    std::array<std::optional<MeshStreamingDelta>, kCapacity> slots_{};
    size_t head_ = 0u;
    size_t count_ = 0u;
};
