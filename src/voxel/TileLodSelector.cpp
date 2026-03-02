#include "solum_engine/voxel/MeshManager.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "solum_engine/resources/Constants.h"

namespace {
struct FootprintDistanceRange {
    int32_t minDistanceChunks = 0;
    int32_t maxDistanceChunks = 0;
};

int32_t minDistanceToInterval(int32_t value, int32_t minValue, int32_t maxValue) {
    if (value < minValue) {
        return minValue - value;
    }
    if (value > maxValue) {
        return value - maxValue;
    }
    return 0;
}

float minDistanceToInterval(float value, float minValue, float maxValue) {
    if (value < minValue) {
        return minValue - value;
    }
    if (value > maxValue) {
        return value - maxValue;
    }
    return 0.0f;
}

int32_t maxDistanceToInterval(int32_t value, int32_t minValue, int32_t maxValue) {
    const int64_t distanceToMin = std::llabs(static_cast<int64_t>(value) - static_cast<int64_t>(minValue));
    const int64_t distanceToMax = std::llabs(static_cast<int64_t>(value) - static_cast<int64_t>(maxValue));
    return static_cast<int32_t>(std::max(distanceToMin, distanceToMax));
}

FootprintDistanceRange footprintDistanceRangeForCell(int32_t cellX,
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
}  // namespace

int8_t MeshManager::desiredLodForTile(const MeshTileCoord& tileCoord,
                                      const ChunkCoord& centerChunk,
                                      const glm::vec3& playerWorldPosition,
                                      float sseProjectionScale,
                                      int32_t extraChunks) const {
    const int32_t radiusChunks = std::max(0, maxConfiguredRadius() + extraChunks);
    const FootprintDistanceRange distances = footprintDistanceRangeForCell(
        tileCoord.x,
        tileCoord.y,
        meshTileSizeChunks_,
        centerChunk
    );
    if (distances.minDistanceChunks > radiusChunks) {
        return -1;
    }

    const float depthBlocks = tileDepthEstimateBlocks(tileCoord, playerWorldPosition, extraChunks);
    for (int32_t lodIndex = config_.lodLevelCount - 1; lodIndex >= 0; --lodIndex) {
        const float ssePixels = projectedSsePixels(
            static_cast<uint8_t>(lodIndex),
            depthBlocks,
            sseProjectionScale
        );
        if (ssePixels <= config_.lodSseTargetPixels) {
            return static_cast<int8_t>(lodIndex);
        }
    }

    return 0;
}

float MeshManager::tileDepthEstimateBlocks(const MeshTileCoord& tileCoord,
                                           const glm::vec3& playerWorldPosition,
                                           int32_t extraChunks) const {
    const float tileWorldSpanBlocks = static_cast<float>(meshTileSizeChunks_ * cfg::CHUNK_SIZE);
    const float tileMinX = static_cast<float>(tileCoord.x) * tileWorldSpanBlocks;
    const float tileMaxX = tileMinX + tileWorldSpanBlocks;
    const float tileMinY = static_cast<float>(tileCoord.y) * tileWorldSpanBlocks;
    const float tileMaxY = tileMinY + tileWorldSpanBlocks;

    const float dx = minDistanceToInterval(playerWorldPosition.x, tileMinX, tileMaxX);
    const float dy = minDistanceToInterval(playerWorldPosition.y, tileMinY, tileMaxY);
    const float horizontalDistanceBlocks = std::sqrt((dx * dx) + (dy * dy));
    const float prefetchBiasBlocks = static_cast<float>(std::max(0, extraChunks) * cfg::CHUNK_SIZE);

    return std::max(config_.lodSseMinDepthBlocks, horizontalDistanceBlocks + prefetchBiasBlocks);
}

float MeshManager::projectedSsePixels(uint8_t lodLevel, float depthBlocks, float sseProjectionScale) const {
    const float clampedDepthBlocks = std::max(config_.lodSseMinDepthBlocks, depthBlocks);
    const float clampedProjectionScale = std::max(1.0e-4f, sseProjectionScale);
    const float geometricErrorBlocks = (lodLevel == 0u)
        ? 0.0f
        : (0.5f * static_cast<float>(chunkSpanForLod(lodLevel)));
    return (geometricErrorBlocks * clampedProjectionScale) / clampedDepthBlocks;
}

int8_t MeshManager::applyLodHysteresis(const MeshTileCoord& tileCoord,
                                       int8_t candidateLod,
                                       int8_t previousLod,
                                       const glm::vec3& playerWorldPosition,
                                       float sseProjectionScale) const {
    if (candidateLod < 0) {
        return -1;
    }
    if (previousLod < 0 || previousLod == candidateLod || config_.lodSseHysteresisPixels <= 0.0f) {
        return candidateLod;
    }

    const int32_t maxLod = config_.lodLevelCount - 1;
    if (previousLod > maxLod) {
        return candidateLod;
    }

    const float depthBlocks = tileDepthEstimateBlocks(tileCoord, playerWorldPosition, 0);
    const float previousSsePixels = projectedSsePixels(
        static_cast<uint8_t>(previousLod),
        depthBlocks,
        sseProjectionScale
    );

    if (candidateLod > previousLod) {
        const float coarseSwitchThreshold = std::max(
            0.0f,
            config_.lodSseTargetPixels - config_.lodSseHysteresisPixels
        );
        if (previousSsePixels > coarseSwitchThreshold) {
            return previousLod;
        }
        return candidateLod;
    }

    const float fineSwitchThreshold = config_.lodSseTargetPixels + config_.lodSseHysteresisPixels;
    if (previousSsePixels < fineSwitchThreshold) {
        return previousLod;
    }
    return candidateLod;
}
