#include <atomic>

struct RegionState {
    std::atomic<uint32_t> regionEpoch;
};