#include "solum_engine/voxel/TerrainGenerator.h"
#include "solum_engine/resources/Constants.h"
#include "solum_engine/voxel/StructureManager.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "lodepng/lodepng.h"

namespace {

constexpr int kHeightmapUpscaleFactor = 2;
constexpr int kFallbackTerrainHeight = 100;
constexpr int kNoiseSeed = 1337;
constexpr float kNoiseHorizontalFrequency = 0.045f;
constexpr float kNoiseVerticalFrequency = 0.08f;
constexpr float kNoiseMaxStrengthBlocks = 12.0f;
constexpr float kNoiseFalloffBlocks = 20.0f;
constexpr float kGrassFlatnessThreshold = 0.75f;

struct HeightmapData {
    int width = 0;
    int height = 0;
    std::vector<float> heights;
    bool valid = false;
};

int wrapIndex(int value, int size) {
    if (size <= 0) {
        return 0;
    }

    int wrapped = value % size;
    if (wrapped < 0) {
        wrapped += size;
    }
    return wrapped;
}

std::string resolveHeightmapPath() {
    const char* envPath = std::getenv("SOLUM_HEIGHTMAP_PATH");
    if (envPath != nullptr && envPath[0] != '\0') {
        return std::string(envPath);
    }
    return std::string(RESOURCE_DIR) + "/height/heightmap6.png";
}

HeightmapData loadHeightmap() {
    HeightmapData out;
    const std::string heightmapPath = resolveHeightmapPath();

    std::vector<unsigned char> rgba;
    unsigned srcWidth = 0;
    unsigned srcHeight = 0;
    const unsigned decodeError = lodepng::decode(rgba, srcWidth, srcHeight, heightmapPath);
    if (decodeError != 0 || srcWidth == 0 || srcHeight == 0) {
        std::cerr << "TerrainGenerator: failed to load heightmap '" << heightmapPath
                  << "'. Falling back to flat terrain at z=" << kFallbackTerrainHeight << std::endl;
        if (decodeError != 0) {
            std::cerr << "TerrainGenerator: lodepng error " << decodeError << " ("
                      << lodepng_error_text(decodeError) << ")" << std::endl;
        }
        return out;
    }

    const int srcW = static_cast<int>(srcWidth);
    const int srcH = static_cast<int>(srcHeight);
    std::vector<float> sourceHeights(static_cast<size_t>(srcW) * static_cast<size_t>(srcH), 0.0f);

    for (int y = 0; y < srcH; ++y) {
        for (int x = 0; x < srcW; ++x) {
            const size_t pixelIndex = (static_cast<size_t>(y) * static_cast<size_t>(srcW) + static_cast<size_t>(x)) * 4u;
            const float r = static_cast<float>(rgba[pixelIndex + 0]) / 255.0f;
            const float g = static_cast<float>(rgba[pixelIndex + 1]) / 255.0f;
            const float b = static_cast<float>(rgba[pixelIndex + 2]) / 255.0f;
            const float a = static_cast<float>(rgba[pixelIndex + 3]) / 255.0f;
            const float luminance = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            sourceHeights[static_cast<size_t>(y) * static_cast<size_t>(srcW) + static_cast<size_t>(x)] = luminance * a;
        }
    }

    out.width = srcW * kHeightmapUpscaleFactor;
    out.height = srcH * kHeightmapUpscaleFactor;
    out.heights.resize(static_cast<size_t>(out.width) * static_cast<size_t>(out.height), 0.0f);

    const auto sourceAt = [&sourceHeights, srcW](int x, int y) -> float {
        return sourceHeights[static_cast<size_t>(y) * static_cast<size_t>(srcW) + static_cast<size_t>(x)];
    };

    for (int y = 0; y < out.height; ++y) {
        const float sourceY = static_cast<float>(y) / static_cast<float>(kHeightmapUpscaleFactor);
        const int y0 = static_cast<int>(std::floor(sourceY));
        const int y1 = std::min(y0 + 1, srcH - 1);
        const float ty = sourceY - static_cast<float>(y0);

        for (int x = 0; x < out.width; ++x) {
            const float sourceX = static_cast<float>(x) / static_cast<float>(kHeightmapUpscaleFactor);
            const int x0 = static_cast<int>(std::floor(sourceX));
            const int x1 = std::min(x0 + 1, srcW - 1);
            const float tx = sourceX - static_cast<float>(x0);

            const float v00 = sourceAt(x0, y0);
            const float v10 = sourceAt(x1, y0);
            const float v01 = sourceAt(x0, y1);
            const float v11 = sourceAt(x1, y1);

            const float top = v00 + (v10 - v00) * tx;
            const float bottom = v01 + (v11 - v01) * tx;
            out.heights[static_cast<size_t>(y) * static_cast<size_t>(out.width) + static_cast<size_t>(x)] =
                top + (bottom - top) * ty;
        }
    }

    out.valid = true;
    std::cout << "TerrainGenerator: loaded heightmap '" << heightmapPath << "' (" << srcW << "x" << srcH
              << "), upscaled to " << out.width << "x" << out.height << std::endl;
    return out;
}

const HeightmapData& getHeightmapData() {
    static const HeightmapData kHeightmapData = loadHeightmap();
    return kHeightmapData;
}

std::vector<StructureManager::StructureDefinition> makeStructureDefinitions() {
    std::vector<StructureManager::StructureDefinition> definitions;

    // Fill this list with your structures.
    // Each structure needs:
    //  - path to the .vox file
    //  - generationOrigin at the "base" voxel of that model
    //  - colorMappings from .vox palette RGBA to engine BlockMaterial
    //
    // Example:
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

const StructureManager& getStructureManager() {
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

int sampleTerrainHeight(const HeightmapData& heightmap, int worldX, int worldY) {
    if (!heightmap.valid || heightmap.width <= 0 || heightmap.height <= 0 || heightmap.heights.empty()) {
        return std::clamp(kFallbackTerrainHeight, 0, cfg::COLUMN_HEIGHT_BLOCKS - 1);
    }

    // Shift world-space sampling so the upscaled map center lands at world origin.
    const int centerOffsetX = heightmap.width / 2;
    const int centerOffsetY = heightmap.height / 2;
    const int sx = wrapIndex(worldX + centerOffsetX, heightmap.width);
    const int sy = wrapIndex(worldY + centerOffsetY, heightmap.height);
    const float normalized = heightmap.heights[static_cast<size_t>(sy) * static_cast<size_t>(heightmap.width) +
                                               static_cast<size_t>(sx)];
    const int maxTerrainHeight = cfg::COLUMN_HEIGHT_BLOCKS - 1;
    const int sampledHeight = static_cast<int>(std::lround(normalized * static_cast<float>(maxTerrainHeight)));
    return std::clamp(sampledHeight, 0, maxTerrainHeight);
}

float smoothstep01(float t) {
    const float clamped = std::clamp(t, 0.0f, 1.0f);
    return clamped * clamped * (3.0f - 2.0f * clamped);
}

float sampleDensity(const FastNoise::SmartNode<>& fnGenerator,
                    int worldX,
                    int worldY,
                    int worldZ,
                    int terrainHeight) {
    const float baseDensity = static_cast<float>(terrainHeight - worldZ);
    if (!fnGenerator) {
        return baseDensity;
    }

    const float distanceFromSurface = std::abs(baseDensity);
    if (distanceFromSurface >= kNoiseFalloffBlocks) {
        // Outside the influence band: noise strength is zero, so skip sampling.
        return baseDensity;
    }

    const float strengthT = 1.0f - (distanceFromSurface / kNoiseFalloffBlocks);
    const float noiseStrength = kNoiseMaxStrengthBlocks * smoothstep01(strengthT);
    if (noiseStrength <= 0.0f) {
        return baseDensity;
    }

    const float nx = static_cast<float>(worldX) * kNoiseHorizontalFrequency;
    const float ny = static_cast<float>(worldY) * kNoiseHorizontalFrequency;
    const float nz = static_cast<float>(worldZ) * kNoiseVerticalFrequency;
    const float noise = fnGenerator->GenSingle3D(nx, ny, nz, kNoiseSeed);

    return baseDensity + (noise * noiseStrength);
}

inline bool localInBounds(int x, int y, int z) {
    return x >= 0 && y >= 0 && z >= 0 &&
           x < cfg::CHUNK_SIZE &&
           y < cfg::CHUNK_SIZE &&
           z < cfg::COLUMN_HEIGHT_BLOCKS;
}

template <typename DensityFn, typename HeightFn>
int findSurfaceForStructure(int worldX, int worldY, const DensityFn& densityAtWorld, const HeightFn& heightAtWorld) {
    constexpr int kColumnHeight = cfg::COLUMN_HEIGHT_BLOCKS;
    constexpr int kSearchPadding = static_cast<int>(kNoiseMaxStrengthBlocks) + 4;

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

} // namespace

void TerrainGenerator::generateColumn(const glm::ivec3& origin, Column& col) {
    const HeightmapData& heightmap = getHeightmapData();

    UnpackedBlockMaterial stone{1, 0, Direction::PlusZ, 0};
    UnpackedBlockMaterial grass{2, 0, Direction::PlusZ, 0};
    UnpackedBlockMaterial air{0, 0, Direction::PlusZ, 0};

    const BlockMaterial stonePacked = stone.pack();
    const BlockMaterial grassPacked = grass.pack();
    const BlockMaterial airPacked = air.pack();

    constexpr int kChunkSize = cfg::CHUNK_SIZE;
    constexpr int kColumnHeight = cfg::COLUMN_HEIGHT_BLOCKS;
    constexpr size_t kColumnVoxelCount = static_cast<size_t>(kChunkSize) *
                                         static_cast<size_t>(kChunkSize) *
                                         static_cast<size_t>(kColumnHeight);
    constexpr int kHeightCacheExtent = kChunkSize + 2; // local x/y in [-1, chunkSize]

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

    auto columnVoxelIndex = [](int x, int y, int z) -> size_t {
        return (static_cast<size_t>(z) * static_cast<size_t>(kChunkSize) + static_cast<size_t>(y)) *
                   static_cast<size_t>(kChunkSize) +
               static_cast<size_t>(x);
    };

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

    std::vector<float> densityField(kColumnVoxelCount, 0.0f);
    std::vector<uint8_t> solidField(kColumnVoxelCount, 0u);

    for (int z = 0; z < kColumnHeight; ++z) {
        for (int y = 0; y < kChunkSize; ++y) {
            for (int x = 0; x < kChunkSize; ++x) {
                const int worldX = origin.x + x;
                const int worldY = origin.y + y;
                const int worldZ = origin.z + z;
                const float density = densityAtWorld(worldX, worldY, worldZ);
                const bool solidVoxel = density >= 0.0f;

                const size_t idx = columnVoxelIndex(x, y, z);
                densityField[idx] = density;
                solidField[idx] = solidVoxel ? 1u : 0u;

                col.setBlock(x, y, z, solidVoxel ? stonePacked : airPacked);
            }
        }
    }

    const std::array<glm::ivec3, 6> neighborOffsets = {
        glm::ivec3{+1, 0, 0},
        glm::ivec3{-1, 0, 0},
        glm::ivec3{0, +1, 0},
        glm::ivec3{0, -1, 0},
        glm::ivec3{0, 0, +1},
        glm::ivec3{0, 0, -1},
    };

    auto densityAtLocalOrWorld = [&](int localX, int localY, int localZ) -> float {
        if (localInBounds(localX, localY, localZ)) {
            return densityField[columnVoxelIndex(localX, localY, localZ)];
        }

        const int worldX = origin.x + localX;
        const int worldY = origin.y + localY;
        const int worldZ = origin.z + localZ;
        return densityAtWorld(worldX, worldY, worldZ);
    };

    for (int z = 0; z < kColumnHeight; ++z) {
        for (int y = 0; y < kChunkSize; ++y) {
            for (int x = 0; x < kChunkSize; ++x) {
                const size_t idx = columnVoxelIndex(x, y, z);
                if (solidField[idx] == 0u) {
                    continue;
                }

                bool hasExposedFace = false;
                for (const glm::ivec3& offset : neighborOffsets) {
                    const int nx = x + offset.x;
                    const int ny = y + offset.y;
                    const int nz = z + offset.z;

                    bool neighborSolid = false;
                    if (localInBounds(nx, ny, nz)) {
                        neighborSolid = solidField[columnVoxelIndex(nx, ny, nz)] != 0u;
                    } else {
                        neighborSolid = densityAtLocalOrWorld(nx, ny, nz) >= 0.0f;
                    }

                    if (!neighborSolid) {
                        hasExposedFace = true;
                        break;
                    }
                }

                if (!hasExposedFace) {
                    continue;
                }

                const float dx = densityAtLocalOrWorld(x + 1, y, z) - densityAtLocalOrWorld(x - 1, y, z);
                const float dy = densityAtLocalOrWorld(x, y + 1, z) - densityAtLocalOrWorld(x, y - 1, z);
                const float dz = densityAtLocalOrWorld(x, y, z + 1) - densityAtLocalOrWorld(x, y, z - 1);
                const float gradLenSq = (dx * dx) + (dy * dy) + (dz * dz);
                const float flatness = (gradLenSq > 1e-6f) ? (std::abs(dz) / std::sqrt(gradLenSq)) : 1.0f;

                col.setBlock(x, y, z, (flatness >= kGrassFlatnessThreshold) ? grassPacked : stonePacked);
            }
        }
    }

    const StructureManager& structureManager = getStructureManager();
    if (!structureManager.hasStructures()) {
        return;
    }

    const int32_t placementPadding = std::max(0, structureManager.maxHorizontalReach());
    const glm::ivec2 placementMin{
        origin.x - placementPadding,
        origin.y - placementPadding
    };
    const glm::ivec2 placementMax{
        origin.x + kChunkSize + placementPadding,
        origin.y + kChunkSize + placementPadding
    };

    std::vector<StructureManager::PlacementPoint> placementPoints;
    structureManager.collectPointsForBounds(placementMin, placementMax, placementPoints);

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
        structureManager.placeStructureForPoint(point, anchorWorld, clipMin, clipMax, col);
    }
}
