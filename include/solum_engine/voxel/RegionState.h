#pragma once

#include <atomic>
#include <cstdint>

enum class RegionGenerationState : uint8_t {
    Empty = 0,
    Partial = 1,
    Complete = 2,
};

class RegionState {
public:
    RegionGenerationState generationState() const;
    void setGenerationState(RegionGenerationState state);

    uint32_t contentVersion() const;
    uint32_t bumpContentVersion();

private:
    std::atomic<RegionGenerationState> generationState_{RegionGenerationState::Empty};
    std::atomic<uint32_t> contentVersion_{1};
};
