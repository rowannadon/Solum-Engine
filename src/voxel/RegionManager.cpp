#include "solum_engine/voxel/RegionManager.h"

#include "solum_engine/voxel/ChunkPool.h"

#include <utility>

RegionManager::RegionManager(ChunkPool* pool)
    : pool_(pool) {}

Region& RegionManager::ensureRegion(RegionCoord coord) {
    auto it = regions_.find(coord);
    if (it != regions_.end()) {
        return *(it->second);
    }

    auto region = std::make_unique<Region>(coord, pool_);
    Region& ref = *region;
    regions_.emplace(coord, std::move(region));
    return ref;
}

Region* RegionManager::tryGetRegion(RegionCoord coord) {
    auto it = regions_.find(coord);
    if (it == regions_.end()) {
        return nullptr;
    }

    return it->second.get();
}

const Region* RegionManager::tryGetRegion(RegionCoord coord) const {
    auto it = regions_.find(coord);
    if (it == regions_.end()) {
        return nullptr;
    }

    return it->second.get();
}

Region& RegionManager::ensureRegionForColumn(ColumnCoord coord) {
    return ensureRegion(column_to_region(coord));
}

Region& RegionManager::ensureRegionForChunk(ChunkCoord coord) {
    return ensureRegion(chunk_to_region(coord));
}

Column& RegionManager::ensureColumn(ColumnCoord coord) {
    Region& region = ensureRegionForColumn(coord);
    const glm::ivec2 local = column_local_in_region(coord);
    return region.ensureColumn(local.x, local.y);
}

Column* RegionManager::tryGetColumn(ColumnCoord coord) {
    Region* region = tryGetRegion(column_to_region(coord));
    if (region == nullptr) {
        return nullptr;
    }

    const glm::ivec2 local = column_local_in_region(coord);
    return region->tryGetColumn(local.x, local.y);
}

const Column* RegionManager::tryGetColumn(ColumnCoord coord) const {
    const Region* region = tryGetRegion(column_to_region(coord));
    if (region == nullptr) {
        return nullptr;
    }

    const glm::ivec2 local = column_local_in_region(coord);
    return region->tryGetColumn(local.x, local.y);
}

Chunk* RegionManager::tryGetChunk(ChunkCoord coord) {
    Region* region = tryGetRegion(chunk_to_region(coord));
    if (region == nullptr) {
        return nullptr;
    }

    return region->tryGetChunk(coord);
}

const Chunk* RegionManager::tryGetChunk(ChunkCoord coord) const {
    const Region* region = tryGetRegion(chunk_to_region(coord));
    if (region == nullptr) {
        return nullptr;
    }

    return region->tryGetChunk(coord);
}

std::vector<RegionCoord> RegionManager::regionCoords() const {
    std::vector<RegionCoord> result;
    result.reserve(regions_.size());

    for (const auto& [coord, region] : regions_) {
        (void)region;
        result.push_back(coord);
    }

    return result;
}
