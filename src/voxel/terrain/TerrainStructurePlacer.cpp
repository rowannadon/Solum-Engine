#include "TerrainGeneratorInternal.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "nlohmann_json/json.hpp"
#include "solum_engine/resources/Constants.h"
#include "solum_engine/voxel/MaterialRegistry.h"
#include "solum_engine/voxel/StructureManager.h"

namespace {
using json = nlohmann::json;

std::filesystem::path structureConfigPath() {
    return std::filesystem::path(RESOURCE_DIR) / "structures.json";
}

bool parseOrigin(const json& originJson, glm::ivec3& outOrigin) {
    if (!originJson.is_array() || originJson.size() != 3) {
        return false;
    }
    for (size_t i = 0; i < 3; ++i) {
        if (!originJson[i].is_number_integer()) {
            return false;
        }
    }
    outOrigin = glm::ivec3{
        originJson[0].get<int32_t>(),
        originJson[1].get<int32_t>(),
        originJson[2].get<int32_t>()
    };
    return true;
}

bool parseColorRgba(const json& rgbaJson, uint8_t& outR, uint8_t& outG, uint8_t& outB, uint8_t& outA) {
    if (!rgbaJson.is_array() || rgbaJson.size() != 4) {
        return false;
    }

    std::array<uint8_t, 4> channels{};
    for (size_t i = 0; i < 4; ++i) {
        if (!rgbaJson[i].is_number_integer()) {
            return false;
        }
        const int value = rgbaJson[i].get<int>();
        if (value < 0 || value > 255) {
            return false;
        }
        channels[i] = static_cast<uint8_t>(value);
    }

    outR = channels[0];
    outG = channels[1];
    outB = channels[2];
    outA = channels[3];
    return true;
}

std::vector<StructureManager::StructureDefinition> loadStructureDefinitions() {
    std::vector<StructureManager::StructureDefinition> definitions;

    const std::filesystem::path path = structureConfigPath();
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "TerrainGenerator: unable to open structure config '" << path.string()
                  << "'. No structures will be placed." << std::endl;
        return definitions;
    }

    json root;
    try {
        file >> root;
    } catch (const std::exception& e) {
        std::cerr << "TerrainGenerator: failed to parse structure config '" << path.string()
                  << "': " << e.what() << ". No structures will be placed." << std::endl;
        return definitions;
    }

    if (!root.is_array()) {
        std::cerr << "TerrainGenerator: structure config '" << path.string()
                  << "' must be an array. No structures will be placed." << std::endl;
        return definitions;
    }

    definitions.reserve(root.size());
    for (size_t i = 0; i < root.size(); ++i) {
        const json& entry = root[i];
        if (!entry.is_object()) {
            std::cerr << "TerrainGenerator: structures[" << i << "] must be an object." << std::endl;
            continue;
        }
        if (!entry.contains("name") || !entry["name"].is_string()) {
            std::cerr << "TerrainGenerator: structures[" << i << "] is missing string field 'name'." << std::endl;
            continue;
        }
        if (!entry.contains("vox") || !entry["vox"].is_string()) {
            std::cerr << "TerrainGenerator: structures[" << i << "] is missing string field 'vox'." << std::endl;
            continue;
        }
        if (!entry.contains("origin")) {
            std::cerr << "TerrainGenerator: structures[" << i << "] is missing field 'origin'." << std::endl;
            continue;
        }
        if (!entry.contains("colorMappings") || !entry["colorMappings"].is_array()) {
            std::cerr << "TerrainGenerator: structures[" << i
                      << "] is missing array field 'colorMappings'." << std::endl;
            continue;
        }

        StructureManager::StructureDefinition definition{};
        definition.name = entry["name"].get<std::string>();

        glm::ivec3 origin{};
        if (!parseOrigin(entry["origin"], origin)) {
            std::cerr << "TerrainGenerator: structures[" << i
                      << "] field 'origin' must be an integer array [x, y, z]." << std::endl;
            continue;
        }
        definition.generationOrigin = origin;

        const std::filesystem::path relativeVoxPath = entry["vox"].get<std::string>();
        const std::filesystem::path resolvedVoxPath =
            relativeVoxPath.is_absolute()
                ? relativeVoxPath
                : (std::filesystem::path(RESOURCE_DIR) / "structures" / relativeVoxPath);
        definition.voxFilePath = resolvedVoxPath.string();

        if (entry.contains("selectionWeight")) {
            if (!entry["selectionWeight"].is_number_integer()) {
                std::cerr << "TerrainGenerator: structures[" << i
                          << "] field 'selectionWeight' must be an integer." << std::endl;
                continue;
            }
            const int64_t selectionWeight = entry["selectionWeight"].get<int64_t>();
            if (selectionWeight <= 0) {
                std::cerr << "TerrainGenerator: structures[" << i
                          << "] field 'selectionWeight' must be greater than zero." << std::endl;
                continue;
            }
            definition.selectionWeight =
                static_cast<uint32_t>(std::min<int64_t>(selectionWeight, std::numeric_limits<uint32_t>::max()));
        }

        const json& colorMappings = entry["colorMappings"];
        definition.colorMappings.reserve(colorMappings.size());
        for (size_t mappingIndex = 0; mappingIndex < colorMappings.size(); ++mappingIndex) {
            const json& mappingEntry = colorMappings[mappingIndex];
            if (!mappingEntry.is_array() || mappingEntry.size() != 2) {
                std::cerr << "TerrainGenerator: structures[" << i << "].colorMappings[" << mappingIndex
                          << "] must be [[r,g,b,a], \"material_name\"]." << std::endl;
                continue;
            }
            if (!mappingEntry[1].is_string()) {
                std::cerr << "TerrainGenerator: structures[" << i << "].colorMappings[" << mappingIndex
                          << "] material name must be a string." << std::endl;
                continue;
            }

            uint8_t r = 0u;
            uint8_t g = 0u;
            uint8_t b = 0u;
            uint8_t a = 255u;
            if (!parseColorRgba(mappingEntry[0], r, g, b, a)) {
                std::cerr << "TerrainGenerator: structures[" << i << "].colorMappings[" << mappingIndex
                          << "] color must be an integer array [r, g, b, a]." << std::endl;
                continue;
            }

            const std::string materialName = mappingEntry[1].get<std::string>();
            uint16_t materialId = 0u;
            if (!MaterialRegistry::tryResolveMaterialId(materialName, materialId)) {
                std::cerr << "TerrainGenerator: structures[" << i << "].colorMappings[" << mappingIndex
                          << "] references unknown material '" << materialName << "'." << std::endl;
                continue;
            }

            definition.colorMappings.push_back(StructureManager::ColorMaterialMapping{
                r,
                g,
                b,
                a,
                UnpackedBlockMaterial{materialId, 0, Direction::PlusZ, 0}.pack()
            });
        }

        if (definition.colorMappings.empty()) {
            std::cerr << "TerrainGenerator: structures[" << i
                      << "] has no valid color mappings and will be skipped." << std::endl;
            continue;
        }

        definitions.push_back(std::move(definition));
    }

    if (definitions.empty()) {
        std::cerr << "TerrainGenerator: structure config '" << path.string()
                  << "' contains no valid structures." << std::endl;
    }

    return definitions;
}

const StructureManager& structureManager() {
    static const StructureManager kManager = [] {
        StructureManager::SamplerConfig sampler;
        sampler.cellSize = 14;
        sampler.minDistance = 8;
        sampler.cellOccupancy = 0.45f;
        sampler.seed = 0x51F15EEDu;

        StructureManager manager(sampler);
        const std::vector<StructureManager::StructureDefinition> definitions = loadStructureDefinitions();
        for (const StructureManager::StructureDefinition& definition : definitions) {
            manager.addStructure(definition);
        }
        return manager;
    }();

    return kManager;
}

template <typename DensityFn, typename HeightFn>
int findSurfaceForStructure(int worldX, int worldY, const DensityFn& densityAtWorld, const HeightFn& heightAtWorld) {
    constexpr int kColumnHeight = cfg::COLUMN_HEIGHT_BLOCKS;
    constexpr int kSearchPadding = static_cast<int>(terrain_internal::kNoiseMaxStrengthBlocks) + 4;

    const int estimated = std::clamp(heightAtWorld(worldX, worldY), 0, kColumnHeight - 1);
    const int searchTop = std::clamp(estimated + kSearchPadding, 0, kColumnHeight - 2);
    const int searchBottom = std::clamp(estimated - kSearchPadding, 0, kColumnHeight - 2);

    for (int z = searchTop; z >= searchBottom; --z) {
        const bool solid = densityAtWorld(worldX, worldY, z) >= 0.0f;
        const bool airAbove = densityAtWorld(worldX, worldY, z + 1) < 0.0f;
        if (solid && airAbove) {
            return z;
        }
    }

    for (int z = kColumnHeight - 2; z >= 0; --z) {
        const bool solid = densityAtWorld(worldX, worldY, z) >= 0.0f;
        const bool airAbove = densityAtWorld(worldX, worldY, z + 1) < 0.0f;
        if (solid && airAbove) {
            return z;
        }
    }

    return -1;
}

}  // namespace

namespace terrain_internal {

void placeColumnStructures(const glm::ivec3& origin,
                           Column& col,
                           const FastNoise::SmartNode<>& fnGenerator,
                           const HeightmapData& heightmap) {
    const StructureManager& manager = structureManager();
    if (!manager.hasStructures()) {
        return;
    }

    constexpr int kChunkSize = cfg::CHUNK_SIZE;
    constexpr int kColumnHeight = cfg::COLUMN_HEIGHT_BLOCKS;
    constexpr int kHeightCacheExtent = kChunkSize + 2;

    std::array<int, static_cast<size_t>(kHeightCacheExtent) * static_cast<size_t>(kHeightCacheExtent)> heightCache{};
    for (int localY = -1; localY <= kChunkSize; ++localY) {
        for (int localX = -1; localX <= kChunkSize; ++localX) {
            const int worldX = origin.x + localX;
            const int worldY = origin.y + localY;
            const size_t cacheIndex =
                static_cast<size_t>(localY + 1) * static_cast<size_t>(kHeightCacheExtent) +
                static_cast<size_t>(localX + 1);
            heightCache[cacheIndex] = sampleTerrainHeight(heightmap, worldX, worldY);
        }
    }

    auto cachedHeightAtWorld = [&](int worldX, int worldY) -> int {
        const int localX = worldX - origin.x;
        const int localY = worldY - origin.y;
        if (localX >= -1 && localX <= kChunkSize && localY >= -1 && localY <= kChunkSize) {
            const size_t cacheIndex =
                static_cast<size_t>(localY + 1) * static_cast<size_t>(kHeightCacheExtent) +
                static_cast<size_t>(localX + 1);
            return heightCache[cacheIndex];
        }
        return sampleTerrainHeight(heightmap, worldX, worldY);
    };

    auto densityAtWorld = [&](int worldX, int worldY, int worldZ) -> float {
        if (worldZ < 0 || worldZ >= kColumnHeight) {
            return -1.0f;
        }
        const int terrainHeight = cachedHeightAtWorld(worldX, worldY);
        return sampleDensity(fnGenerator, worldX, worldY, worldZ, terrainHeight);
    };

    const int32_t placementPadding = std::max(0, manager.maxHorizontalReach());
    const glm::ivec2 placementMin{
        origin.x - placementPadding,
        origin.y - placementPadding
    };
    const glm::ivec2 placementMax{
        origin.x + kChunkSize + placementPadding,
        origin.y + kChunkSize + placementPadding
    };

    std::vector<StructureManager::PlacementPoint> placementPoints;
    manager.collectPointsForBounds(placementMin, placementMax, placementPoints);

    const glm::ivec3 clipMin{origin.x, origin.y, 0};
    const glm::ivec3 clipMax{origin.x + kChunkSize, origin.y + kChunkSize, kColumnHeight};

    for (const StructureManager::PlacementPoint& point : placementPoints) {
        const int32_t surfaceZ = findSurfaceForStructure(
            point.worldXY.x,
            point.worldXY.y,
            densityAtWorld,
            cachedHeightAtWorld
        );
        if (surfaceZ < 0 || (surfaceZ + 1) >= kColumnHeight) {
            continue;
        }

        const glm::ivec3 anchorWorld{
            point.worldXY.x,
            point.worldXY.y,
            surfaceZ + 1
        };
        manager.placeStructureForPoint(point, anchorWorld, clipMin, clipMax, col);
    }
}

}  // namespace terrain_internal
