#include "TerrainGeneratorInternal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "lodepng/lodepng.h"
#include "nlohmann_json/json.hpp"
#include "solum_engine/resources/Constants.h"

namespace {

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

uint16_t resolveTallGrassMaterialId() {
    const std::filesystem::path materialPath = std::filesystem::path(RESOURCE_DIR) / "materials.json";
    std::ifstream file(materialPath);
    if (!file.is_open()) {
        std::cerr << "TerrainGenerator: unable to open '" << materialPath.string()
                  << "', defaulting tall grass material ID to "
                  << terrain_internal::kFallbackTallGrassMaterialId << std::endl;
        return terrain_internal::kFallbackTallGrassMaterialId;
    }

    nlohmann::json root;
    try {
        file >> root;
    } catch (const std::exception& e) {
        std::cerr << "TerrainGenerator: failed to parse '" << materialPath.string() << "': "
                  << e.what() << ". Defaulting tall grass material ID to "
                  << terrain_internal::kFallbackTallGrassMaterialId << std::endl;
        return terrain_internal::kFallbackTallGrassMaterialId;
    }

    const nlohmann::json* materialsJson = nullptr;
    if (root.is_array()) {
        materialsJson = &root;
    } else if (root.is_object() && root.contains("materials") && root["materials"].is_array()) {
        materialsJson = &root["materials"];
    }
    if (materialsJson == nullptr) {
        std::cerr << "TerrainGenerator: '" << materialPath.string()
                  << "' does not contain a materials array. Defaulting tall grass material ID to "
                  << terrain_internal::kFallbackTallGrassMaterialId << std::endl;
        return terrain_internal::kFallbackTallGrassMaterialId;
    }

    for (size_t i = 0; i < materialsJson->size(); ++i) {
        const nlohmann::json& entry = (*materialsJson)[i];
        if (!entry.is_object() || !entry.contains("name") || !entry["name"].is_string()) {
            continue;
        }
        if (entry["name"].get<std::string>() == "tall_grass") {
            return static_cast<uint16_t>(i + 1u);
        }
    }

    std::cerr << "TerrainGenerator: material 'tall_grass' not found in '" << materialPath.string()
              << "', defaulting tall grass material ID to "
              << terrain_internal::kFallbackTallGrassMaterialId << std::endl;
    return terrain_internal::kFallbackTallGrassMaterialId;
}

float resolveTallGrassChance() {
    const char* envChance = std::getenv("SOLUM_TALL_GRASS_CHANCE");
    if (envChance == nullptr || envChance[0] == '\0') {
        return terrain_internal::kDefaultTallGrassChance;
    }

    char* endPtr = nullptr;
    const double parsed = std::strtod(envChance, &endPtr);
    if (endPtr == envChance || *endPtr != '\0' || !std::isfinite(parsed)) {
        std::cerr << "TerrainGenerator: invalid SOLUM_TALL_GRASS_CHANCE='" << envChance
                  << "', using default " << terrain_internal::kDefaultTallGrassChance << std::endl;
        return terrain_internal::kDefaultTallGrassChance;
    }

    if (parsed < 0.0 || parsed > 1.0) {
        std::cerr << "TerrainGenerator: clamping SOLUM_TALL_GRASS_CHANCE='" << envChance
                  << "' to [0, 1]" << std::endl;
    }

    return std::clamp(static_cast<float>(parsed), 0.0f, 1.0f);
}

float hashedUnitFloatForWorldPosition(int worldX, int worldY, int worldZ) {
    uint64_t value = 0x9E3779B97F4A7C15ull;
    value ^= static_cast<uint64_t>(static_cast<uint32_t>(worldX)) + 0x9E3779B9u + (value << 6u) + (value >> 2u);
    value ^= static_cast<uint64_t>(static_cast<uint32_t>(worldY)) + 0x85EBCA6Bu + (value << 6u) + (value >> 2u);
    value ^= static_cast<uint64_t>(static_cast<uint32_t>(worldZ)) + 0xC2B2AE35u + (value << 6u) + (value >> 2u);
    value ^= (value >> 33u);
    value *= 0xff51afd7ed558ccdull;
    value ^= (value >> 33u);
    value *= 0xc4ceb9fe1a85ec53ull;
    value ^= (value >> 33u);

    const uint32_t topBits = static_cast<uint32_t>(value >> 40u);
    return static_cast<float>(topBits) / static_cast<float>(0xFFFFFFu);
}

terrain_internal::HeightmapData loadHeightmap() {
    terrain_internal::HeightmapData out;
    const std::string heightmapPath = resolveHeightmapPath();

    std::vector<unsigned char> rgba;
    unsigned srcWidth = 0;
    unsigned srcHeight = 0;
    const unsigned decodeError = lodepng::decode(rgba, srcWidth, srcHeight, heightmapPath);
    if (decodeError != 0 || srcWidth == 0 || srcHeight == 0) {
        std::cerr << "TerrainGenerator: failed to load heightmap '" << heightmapPath
                  << "'. Falling back to flat terrain at z=" << terrain_internal::kFallbackTerrainHeight << std::endl;
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
            const size_t pixelIndex =
                (static_cast<size_t>(y) * static_cast<size_t>(srcW) + static_cast<size_t>(x)) * 4u;
            const float r = static_cast<float>(rgba[pixelIndex + 0]) / 255.0f;
            const float g = static_cast<float>(rgba[pixelIndex + 1]) / 255.0f;
            const float b = static_cast<float>(rgba[pixelIndex + 2]) / 255.0f;
            const float a = static_cast<float>(rgba[pixelIndex + 3]) / 255.0f;
            const float luminance = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            sourceHeights[static_cast<size_t>(y) * static_cast<size_t>(srcW) + static_cast<size_t>(x)] = luminance * a;
        }
    }

    out.width = srcW * terrain_internal::kHeightmapUpscaleFactor;
    out.height = srcH * terrain_internal::kHeightmapUpscaleFactor;
    out.heights.resize(static_cast<size_t>(out.width) * static_cast<size_t>(out.height), 0.0f);

    const auto sourceAt = [&sourceHeights, srcW](int x, int y) -> float {
        return sourceHeights[static_cast<size_t>(y) * static_cast<size_t>(srcW) + static_cast<size_t>(x)];
    };

    for (int y = 0; y < out.height; ++y) {
        const float sourceY = static_cast<float>(y) / static_cast<float>(terrain_internal::kHeightmapUpscaleFactor);
        const int y0 = static_cast<int>(std::floor(sourceY));
        const int y1 = std::min(y0 + 1, srcH - 1);
        const float ty = sourceY - static_cast<float>(y0);

        for (int x = 0; x < out.width; ++x) {
            const float sourceX = static_cast<float>(x) / static_cast<float>(terrain_internal::kHeightmapUpscaleFactor);
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

float smoothstep01(float t) {
    const float clamped = std::clamp(t, 0.0f, 1.0f);
    return clamped * clamped * (3.0f - 2.0f * clamped);
}

bool localInBounds(int x, int y, int z) {
    return x >= 0 && y >= 0 && z >= 0 &&
           x < cfg::CHUNK_SIZE &&
           y < cfg::CHUNK_SIZE &&
           z < cfg::COLUMN_HEIGHT_BLOCKS;
}

}  // namespace

namespace terrain_internal {

const HeightmapData& heightmapData() {
    static const HeightmapData kHeightmapData = loadHeightmap();
    return kHeightmapData;
}

TerrainDecorationConfig decorationConfig() {
    static const TerrainDecorationConfig kConfig{
        resolveTallGrassChance(),
        resolveTallGrassMaterialId()
    };
    return kConfig;
}

int sampleTerrainHeight(const HeightmapData& heightmap, int worldX, int worldY) {
    if (!heightmap.valid || heightmap.width <= 0 || heightmap.height <= 0 || heightmap.heights.empty()) {
        return std::clamp(kFallbackTerrainHeight, 0, cfg::COLUMN_HEIGHT_BLOCKS - 1);
    }

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

void generateTerrainColumn(const glm::ivec3& origin,
                           Column& col,
                           const FastNoise::SmartNode<>& fnGenerator,
                           const HeightmapData& heightmap,
                           const TerrainDecorationConfig& config) {
    UnpackedBlockMaterial stone{1, 0, Direction::PlusZ, 0};
    UnpackedBlockMaterial grass{2, 0, Direction::PlusZ, 0};
    UnpackedBlockMaterial tallGrass{config.tallGrassMaterialId, 0, Direction::PlusZ, 0};
    UnpackedBlockMaterial air{0, 0, Direction::PlusZ, 0};

    const BlockMaterial stonePacked = stone.pack();
    const BlockMaterial grassPacked = grass.pack();
    const BlockMaterial tallGrassPacked = tallGrass.pack();
    const BlockMaterial airPacked = air.pack();

    constexpr int kChunkSize = cfg::CHUNK_SIZE;
    constexpr int kColumnHeight = cfg::COLUMN_HEIGHT_BLOCKS;
    constexpr size_t kColumnVoxelCount = static_cast<size_t>(kChunkSize) *
                                         static_cast<size_t>(kChunkSize) *
                                         static_cast<size_t>(kColumnHeight);
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

    for (int z = 0; z < (kColumnHeight - 1); ++z) {
        for (int y = 0; y < kChunkSize; ++y) {
            for (int x = 0; x < kChunkSize; ++x) {
                if (col.getBlock(x, y, z).unpack().id != grass.id) {
                    continue;
                }
                if (col.getBlock(x, y, z + 1).unpack().id != air.id) {
                    continue;
                }

                const int worldX = origin.x + x;
                const int worldY = origin.y + y;
                const int worldZ = origin.z + z + 1;
                if (hashedUnitFloatForWorldPosition(worldX, worldY, worldZ) >= config.tallGrassChance) {
                    continue;
                }

                col.setBlock(x, y, z + 1, tallGrassPacked);
            }
        }
    }
}

}  // namespace terrain_internal
