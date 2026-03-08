#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "solum_engine/resources/Coords.h"

namespace mesh_manager::tile_footprint {

struct FootprintDistanceRange {
    int32_t minDistanceChunks = 0;
    int32_t maxDistanceChunks = 0;
};

inline int32_t minDistanceToInterval(int32_t value, int32_t minValue, int32_t maxValue) {
    if (value < minValue) {
        return minValue - value;
    }
    if (value > maxValue) {
        return value - maxValue;
    }
    return 0;
}

inline float minDistanceToInterval(float value, float minValue, float maxValue) {
    if (value < minValue) {
        return minValue - value;
    }
    if (value > maxValue) {
        return value - maxValue;
    }
    return 0.0f;
}

inline int32_t maxDistanceToInterval(int32_t value, int32_t minValue, int32_t maxValue) {
    const int64_t distanceToMin = std::llabs(static_cast<int64_t>(value) - static_cast<int64_t>(minValue));
    const int64_t distanceToMax = std::llabs(static_cast<int64_t>(value) - static_cast<int64_t>(maxValue));
    return static_cast<int32_t>(std::max(distanceToMin, distanceToMax));
}

inline FootprintDistanceRange footprintDistanceRangeForCell(int32_t cellX,
                                                            int32_t cellY,
                                                            int32_t spanChunks,
                                                            const ChunkCoord& centerChunk) {
    const int32_t minChunkX = cellX * spanChunks;
    const int32_t maxChunkX = minChunkX + spanChunks - 1;
    const int32_t minChunkY = cellY * spanChunks;
    const int32_t maxChunkY = minChunkY + spanChunks - 1;

    const int32_t minDx = minDistanceToInterval(centerChunk.v.x, minChunkX, maxChunkX);
    const int32_t minDy = minDistanceToInterval(centerChunk.v.y, minChunkY, maxChunkY);
    const int32_t maxDx = maxDistanceToInterval(centerChunk.v.x, minChunkX, maxChunkX);
    const int32_t maxDy = maxDistanceToInterval(centerChunk.v.y, minChunkY, maxChunkY);

    return FootprintDistanceRange{
        std::max(minDx, minDy),
        std::max(maxDx, maxDy)
    };
}

}  // namespace mesh_manager::tile_footprint
