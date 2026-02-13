#pragma once

#include "solum_engine/voxel/ColumnStage.h"

#include <atomic>
#include <cstdint>

class ColumnState {
public:
    ColumnStage stage() const;
    void setStage(ColumnStage stage);

    uint32_t neighborTerrainReadyCount() const;
    uint32_t incrementNeighborTerrainReadyCount();
    void resetNeighborTerrainReadyCount();

    uint32_t contentVersion() const;
    uint32_t bumpContentVersion();

    uint64_t structureSeed() const;
    void setStructureSeed(uint64_t seed);

    bool canRunStructureGeneration() const;

private:
    std::atomic<ColumnStage> stage_{ColumnStage::Empty};
    std::atomic<uint32_t> neighborsTerrainReadyCount_{0};
    std::atomic<uint32_t> contentVersion_{1};
    std::atomic<uint64_t> structureSeed_{0};
};
