#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

#include "FastNoise/FastNoise.h"
#include "solum_engine/voxel/Column.h"

namespace terrain_internal {

inline constexpr int kHeightmapUpscaleFactor = 2;
inline constexpr int kFallbackTerrainHeight = 100;
inline constexpr int kNoiseSeed = 1337;
inline constexpr float kNoiseHorizontalFrequency = 0.015f;
inline constexpr float kNoiseVerticalFrequency = 0.04f;
inline constexpr float kNoiseMaxStrengthBlocks = 64.0f;
inline constexpr float kNoiseFalloffBlocks = 55.0f;
inline constexpr float kGrassFlatnessThreshold = 0.75f;
inline constexpr float kDefaultTallGrassChance = 0.25f;
inline constexpr uint16_t kFallbackTallGrassMaterialId = 5u;

struct HeightmapData {
    int width = 0;
    int height = 0;
    std::vector<float> heights;
    bool valid = false;
};

struct TerrainDecorationConfig {
    float tallGrassChance = kDefaultTallGrassChance;
    uint16_t tallGrassMaterialId = kFallbackTallGrassMaterialId;
};

const HeightmapData& heightmapData();
TerrainDecorationConfig decorationConfig();
int sampleTerrainHeight(const HeightmapData& heightmap, int worldX, int worldY);
float sampleDensity(const FastNoise::SmartNode<>& fnGenerator,
                    int worldX,
                    int worldY,
                    int worldZ,
                    int terrainHeight);
void generateTerrainColumn(const glm::ivec3& origin,
                           Column& col,
                           const FastNoise::SmartNode<>& fnGenerator,
                           const HeightmapData& heightmap,
                           const TerrainDecorationConfig& config);
void placeColumnStructures(const glm::ivec3& origin,
                           Column& col,
                           const FastNoise::SmartNode<>& fnGenerator,
                           const HeightmapData& heightmap);
void bootstrapColumnLighting(Column& col);

}  // namespace terrain_internal
