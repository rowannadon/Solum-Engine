#include "solum_engine/voxel/RegionManager.h"

bool RegionManager::addRegion(const RegionCoord& coord) {
    auto it = regions_.find(coord);
    if (it != regions_.end()) {
        return false;
    }

    regions_.emplace(coord, std::make_unique<Region>(coord));
    return true;
}

bool RegionManager::removeRegion(const RegionCoord& coord) {
    // TODO change to two phase, mark evicted and delete once no jobs running
    auto it = regions_.find(coord);
    if (it == regions_.end()) {
        return false;
    }

    regions_.erase(it);
    return true;
}

Region* RegionManager::getRegion(const RegionCoord& coord) {
    auto it = regions_.find(coord);

    if (it == regions_.end()) {
        return nullptr;
    }

    return it->second.get();
}