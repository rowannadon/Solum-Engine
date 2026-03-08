#include "TerrainGeneratorInternal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "lodepng/lodepng.h"
#include "nlohmann_json/json.hpp"
#include "solum_engine/resources/Constants.h"
#include "solum_engine/voxel/MaterialRegistry.h"

namespace {
using json = nlohmann::json;

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

std::filesystem::path decorationConfigPath() {
    return std::filesystem::path(RESOURCE_DIR) / "decorations.json";
}

uint64_t hashWorldPosition(int worldX, int worldY, int worldZ, uint64_t salt) {
    uint64_t value = 0x9E3779B97F4A7C15ull;
    value ^= salt + 0x517CC1B727220A95ull + (value << 6u) + (value >> 2u);
    value ^= static_cast<uint64_t>(static_cast<uint32_t>(worldX)) + 0x9E3779B9u + (value << 6u) + (value >> 2u);
    value ^= static_cast<uint64_t>(static_cast<uint32_t>(worldY)) + 0x85EBCA6Bu + (value << 6u) + (value >> 2u);
    value ^= static_cast<uint64_t>(static_cast<uint32_t>(worldZ)) + 0xC2B2AE35u + (value << 6u) + (value >> 2u);
    value ^= (value >> 33u);
    value *= 0xff51afd7ed558ccdull;
    value ^= (value >> 33u);
    value *= 0xc4ceb9fe1a85ec53ull;
    value ^= (value >> 33u);
    return value;
}

float hashedUnitFloatForWorldPosition(int worldX, int worldY, int worldZ, uint64_t salt) {
    const uint64_t hashed = hashWorldPosition(worldX, worldY, worldZ, salt);
    const uint32_t topBits = static_cast<uint32_t>(hashed >> 40u);
    return static_cast<float>(topBits) / static_cast<float>(0xFFFFFFu);
}

terrain_internal::TerrainDecorationConfig loadDecorationConfig() {
    terrain_internal::TerrainDecorationConfig config{};
    config.placementChance = terrain_internal::kDefaultDecorationChance;

    const std::filesystem::path path = decorationConfigPath();
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "TerrainGenerator: unable to open decoration config '" << path.string()
                  << "'. No decorations will be placed." << std::endl;
        return config;
    }

    json root;
    try {
        file >> root;
    } catch (const std::exception& e) {
        std::cerr << "TerrainGenerator: failed to parse decoration config '" << path.string()
                  << "': " << e.what() << ". No decorations will be placed." << std::endl;
        return config;
    }

    if (!root.is_array()) {
        std::cerr << "TerrainGenerator: decoration config '" << path.string()
                  << "' must be an array. No decorations will be placed." << std::endl;
        return config;
    }

    config.definitions.reserve(root.size());
    for (size_t i = 0; i < root.size(); ++i) {
        const json& entry = root[i];
        if (!entry.is_object()) {
            std::cerr << "TerrainGenerator: decorations[" << i << "] must be an object." << std::endl;
            continue;
        }
        if (!entry.contains("material") || !entry["material"].is_string()) {
            std::cerr << "TerrainGenerator: decorations[" << i << "] is missing string field 'material'." << std::endl;
            continue;
        }

        const std::string materialName = entry["material"].get<std::string>();
        BlockMaterial material{};
        if (!MaterialRegistry::tryResolveBlock(materialName, material)) {
            std::cerr << "TerrainGenerator: decorations[" << i << "] references unknown material '"
                      << materialName << "'." << std::endl;
            continue;
        }

        uint32_t selectionWeight = 1u;
        if (entry.contains("selectionWeight")) {
            if (!entry["selectionWeight"].is_number_integer()) {
                std::cerr << "TerrainGenerator: decorations[" << i
                          << "] field 'selectionWeight' must be an integer." << std::endl;
                continue;
            }
            const int64_t weight = entry["selectionWeight"].get<int64_t>();
            if (weight <= 0) {
                std::cerr << "TerrainGenerator: decorations[" << i
                          << "] field 'selectionWeight' must be greater than zero." << std::endl;
                continue;
            }
            selectionWeight = static_cast<uint32_t>(std::min<int64_t>(weight, std::numeric_limits<uint32_t>::max()));
        }

        config.definitions.push_back(terrain_internal::TerrainDecorationDefinition{
            material,
            selectionWeight
        });
        config.totalSelectionWeight += static_cast<uint64_t>(selectionWeight);
    }

    if (config.definitions.empty()) {
        std::cerr << "TerrainGenerator: decoration config '" << path.string()
                  << "' contains no valid decoration definitions." << std::endl;
    }

    return config;
}

const terrain_internal::TerrainDecorationDefinition* pickDecorationForWorldPosition(
    const terrain_internal::TerrainDecorationConfig& config,
    int worldX,
    int worldY,
    int worldZ
) {
    if (config.definitions.empty() || config.totalSelectionWeight == 0u) {
        return nullptr;
    }

    const uint64_t selector =
        hashWorldPosition(worldX, worldY, worldZ, 0xC4B8A0D4F6E51923ull) % config.totalSelectionWeight;
    uint64_t runningWeight = 0u;
    for (const terrain_internal::TerrainDecorationDefinition& definition : config.definitions) {
        runningWeight += static_cast<uint64_t>(std::max<uint32_t>(definition.selectionWeight, 1u));
        if (selector < runningWeight) {
            return &definition;
        }
    }

    return &config.definitions.back();
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
    static const TerrainDecorationConfig kConfig = loadDecorationConfig();
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
    UnpackedBlockMaterial air{0, 0, Direction::PlusZ, 0};

    const BlockMaterial stonePacked = stone.pack();
    const BlockMaterial grassPacked = grass.pack();
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
                if (hashedUnitFloatForWorldPosition(worldX, worldY, worldZ, 0xAB4F12D8E91CC327ull) >=
                    config.placementChance) {
                    continue;
                }

                const TerrainDecorationDefinition* decoration =
                    pickDecorationForWorldPosition(config, worldX, worldY, worldZ);
                if (decoration == nullptr) {
                    continue;
                }

                col.setBlock(x, y, z + 1, decoration->material);
            }
        }
    }
}

}  // namespace terrain_internal
