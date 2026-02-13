#include "solum_engine/voxel/RegionState.h"

RegionGenerationState RegionState::generationState() const {
    return generationState_.load(std::memory_order_acquire);
}

void RegionState::setGenerationState(RegionGenerationState state) {
    generationState_.store(state, std::memory_order_release);
}

uint32_t RegionState::contentVersion() const {
    return contentVersion_.load(std::memory_order_acquire);
}

uint32_t RegionState::bumpContentVersion() {
    return contentVersion_.fetch_add(1u, std::memory_order_acq_rel) + 1u;
}
