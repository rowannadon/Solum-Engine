#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "FastNoise/FastNoise.h"
#include "solum_engine/voxel/Column.h"

namespace terrain_internal {

inline constexpr int kHeightmapUpscaleFactor = 8;
inline constexpr int kFallbackTerrainHeight = 100;
inline constexpr const char* kActiveBiomeName = "forest";

inline constexpr int kDefaultNoiseSeed = 1337;
inline constexpr float kDefaultNoiseHorizontalFrequency = 0.015f;
inline constexpr float kDefaultNoiseVerticalFrequency = 0.04f;
inline constexpr float kDefaultNoiseMaxStrengthBlocks = 64.0f;
inline constexpr float kDefaultNoiseFalloffBlocks = 55.0f;
inline constexpr const char* kDefaultNoiseEncodedNodeTree = "";
inline constexpr float kDefaultFlatnessThreshold = 0.75f;
inline constexpr float kDefaultDecorationChance = 0.25f;
inline constexpr const char* kDefaultAbovegroundMaterial = "grass";
inline constexpr const char* kDefaultUndergroundMaterial = "stone";
inline constexpr int32_t kDefaultStructureCellSize = 14;
inline constexpr int32_t kDefaultStructureMinDistance = 8;
inline constexpr float kDefaultStructureCellOccupancy = 0.45f;
inline constexpr uint32_t kDefaultStructureSeed = 0x51F15EEDu;

struct HeightmapData {
    int width = 0;
    int height = 0;
    std::vector<float> heights;
    bool valid = false;
};

struct TerrainDecorationDefinition {
    std::string name;
    BlockMaterial material{};
    uint32_t selectionWeight = 1u;
};

struct BiomeWeightedSelection {
    std::string name;
    uint32_t selectionWeight = 1u;
};

struct BiomeConfig {
    std::string name = kActiveBiomeName;
    std::vector<BiomeWeightedSelection> structures;
    std::vector<BiomeWeightedSelection> decorations;
    int noiseSeed = kDefaultNoiseSeed;
    float noiseHorizontalFrequency = kDefaultNoiseHorizontalFrequency;
    float noiseVerticalFrequency = kDefaultNoiseVerticalFrequency;
    float noiseMaxStrengthBlocks = kDefaultNoiseMaxStrengthBlocks;
    float noiseFalloffBlocks = kDefaultNoiseFalloffBlocks;
    std::string noiseEncodedNodeTree = kDefaultNoiseEncodedNodeTree;
    BlockMaterial abovegroundMaterial = UnpackedBlockMaterial{2, 0, Direction::PlusZ, 0}.pack();
    BlockMaterial undergroundMaterial = UnpackedBlockMaterial{1, 0, Direction::PlusZ, 0}.pack();
    float flatnessThreshold = kDefaultFlatnessThreshold;
    float decorationChance = kDefaultDecorationChance;
    int32_t structureCellSize = kDefaultStructureCellSize;
    int32_t structureMinDistance = kDefaultStructureMinDistance;
    float structureCellOccupancy = kDefaultStructureCellOccupancy;
    uint32_t structureSeed = kDefaultStructureSeed;
};

struct TerrainDecorationConfig {
    float placementChance = kDefaultDecorationChance;
    std::vector<TerrainDecorationDefinition> definitions;
    uint64_t totalSelectionWeight = 0u;
};

const HeightmapData& heightmapData();
const BiomeConfig& biomeConfig();
FastNoise::SmartNode<> createTerrainNoiseGenerator();
TerrainDecorationConfig decorationConfig();
int sampleTerrainHeight(const HeightmapData& heightmap, int worldX, int worldY);
float sampleDensity(const FastNoise::SmartNode<>& fnGenerator,
                    int worldX,
                    int worldY,
                    int worldZ,
                    const BiomeConfig& biome,
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
