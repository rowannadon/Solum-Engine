#include "TerrainGeneratorInternal.h"

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "solum_engine/resources/Constants.h"
#include "solum_engine/voxel/StructureManager.h"

namespace {

std::vector<StructureManager::StructureDefinition> makeStructureDefinitions() {
    std::vector<StructureManager::StructureDefinition> definitions;

    StructureManager::StructureDefinition tree;
    tree.name = "aspen1";
    tree.voxFilePath = std::string(RESOURCE_DIR) + "/structures/aspen_1.vox";
    tree.generationOrigin = glm::ivec3{5, 5, 3};
    tree.selectionWeight = 1;
    tree.colorMappings = {
        {102, 51, 0, 255, UnpackedBlockMaterial{3, 0, Direction::PlusZ, 0}.pack()},
        {0, 68, 0, 255, UnpackedBlockMaterial{4, 0, Direction::PlusZ, 0}.pack()},
    };
    definitions.push_back(tree);

    StructureManager::StructureDefinition tree2;
    tree2.name = "aspen2";
    tree2.voxFilePath = std::string(RESOURCE_DIR) + "/structures/aspen_2.vox";
    tree2.generationOrigin = glm::ivec3{4, 5, 3};
    tree2.selectionWeight = 1;
    tree2.colorMappings = {
        {102, 51, 0, 255, UnpackedBlockMaterial{3, 0, Direction::PlusZ, 0}.pack()},
        {0, 68, 0, 255, UnpackedBlockMaterial{4, 0, Direction::PlusZ, 0}.pack()},
    };
    definitions.push_back(tree2);

    StructureManager::StructureDefinition tree3;
    tree3.name = "aspen3";
    tree3.voxFilePath = std::string(RESOURCE_DIR) + "/structures/aspen_3.vox";
    tree3.generationOrigin = glm::ivec3{4, 3, 3};
    tree3.selectionWeight = 1;
    tree3.colorMappings = {
        {102, 51, 0, 255, UnpackedBlockMaterial{3, 0, Direction::PlusZ, 0}.pack()},
        {0, 68, 0, 255, UnpackedBlockMaterial{4, 0, Direction::PlusZ, 0}.pack()},
    };
    definitions.push_back(tree3);

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
        const std::vector<StructureManager::StructureDefinition> definitions = makeStructureDefinitions();
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
