#pragma once

#include "solum_engine/resources/Coords.h"
#include "solum_engine/voxel/Region.h"

#include <memory>
#include <unordered_map>
#include <vector>

class ChunkPool;

class RegionManager {
public:
    explicit RegionManager(ChunkPool* pool = nullptr);

    Region& ensureRegion(RegionCoord coord);
    Region* tryGetRegion(RegionCoord coord);
    const Region* tryGetRegion(RegionCoord coord) const;

    Region& ensureRegionForColumn(ColumnCoord coord);
    Region& ensureRegionForChunk(ChunkCoord coord);

    Column& ensureColumn(ColumnCoord coord);
    Column* tryGetColumn(ColumnCoord coord);
    const Column* tryGetColumn(ColumnCoord coord) const;

    Chunk* tryGetChunk(ChunkCoord coord);
    const Chunk* tryGetChunk(ChunkCoord coord) const;

    std::vector<RegionCoord> regionCoords() const;

private:
    ChunkPool* pool_ = nullptr;
    std::unordered_map<RegionCoord, std::unique_ptr<Region>, GridCoord2Hash<RegionTag>> regions_;
};
