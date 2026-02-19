#include <atomic>

class ChunkState {
    std::atomic<uint32_t> blockEpoch;
};