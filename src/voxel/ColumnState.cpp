#include "solum_engine/voxel/ColumnState.h"

ColumnStage ColumnState::stage() const {
    return stage_.load(std::memory_order_acquire);
}

void ColumnState::setStage(ColumnStage stage) {
    stage_.store(stage, std::memory_order_release);
}

uint32_t ColumnState::neighborTerrainReadyCount() const {
    return neighborsTerrainReadyCount_.load(std::memory_order_acquire);
}

uint32_t ColumnState::incrementNeighborTerrainReadyCount() {
    return neighborsTerrainReadyCount_.fetch_add(1u, std::memory_order_acq_rel) + 1u;
}

void ColumnState::resetNeighborTerrainReadyCount() {
    neighborsTerrainReadyCount_.store(0u, std::memory_order_release);
}

uint32_t ColumnState::contentVersion() const {
    return contentVersion_.load(std::memory_order_acquire);
}

uint32_t ColumnState::bumpContentVersion() {
    return contentVersion_.fetch_add(1u, std::memory_order_acq_rel) + 1u;
}

uint64_t ColumnState::structureSeed() const {
    return structureSeed_.load(std::memory_order_acquire);
}

void ColumnState::setStructureSeed(uint64_t seed) {
    structureSeed_.store(seed, std::memory_order_release);
}

bool ColumnState::canRunStructureGeneration() const {
    return stage() == ColumnStage::TerrainReady && neighborTerrainReadyCount() >= 9u;
}
