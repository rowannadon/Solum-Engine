#include "TerrainGeneratorInternal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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
    return std::string(RESOURCE_DIR) + "/height/heightmap6.png";
}

std::filesystem::path biomeConfigPath() {
    return std::filesystem::path(RESOURCE_DIR) / "biomes.json";
}

bool parseWeightedSelectionField(const json& object,
                                 const char* fieldName,
                                 std::vector<terrain_internal::BiomeWeightedSelection>& outValues,
                                 size_t entryIndex,
                                 const char* contextLabel) {
    if (!object.contains(fieldName)) {
        outValues.clear();
        return true;
    }
    if (!object[fieldName].is_array()) {
        std::cerr << "TerrainGenerator: " << contextLabel << "[" << entryIndex << "] field '"
                  << fieldName << "' must be an array of [name, selectionWeight] pairs." << std::endl;
        return false;
    }

    outValues.clear();
    const json& values = object[fieldName];
    outValues.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        if (!values[i].is_array() || values[i].size() != 2 || !values[i][0].is_string() ||
            !values[i][1].is_number_integer()) {
            std::cerr << "TerrainGenerator: " << contextLabel << "[" << entryIndex << "]." << fieldName
                      << "[" << i << "] must be [\"name\", integer_weight]." << std::endl;
            return false;
        }

        const int64_t rawWeight = values[i][1].get<int64_t>();
        if (rawWeight <= 0) {
            std::cerr << "TerrainGenerator: " << contextLabel << "[" << entryIndex << "]." << fieldName
                      << "[" << i << "] weight must be greater than zero." << std::endl;
            return false;
        }

        outValues.push_back(terrain_internal::BiomeWeightedSelection{
            values[i][0].get<std::string>(),
            static_cast<uint32_t>(std::min<int64_t>(rawWeight, std::numeric_limits<uint32_t>::max()))
        });
    }

    return true;
}

terrain_internal::BiomeConfig loadBiomeConfig() {
    terrain_internal::BiomeConfig config{};
    config.name = terrain_internal::kActiveBiomeName;
    config.abovegroundMaterial = MaterialRegistry::resolveBlockOr(
        terrain_internal::kDefaultAbovegroundMaterial,
        UnpackedBlockMaterial{2u, 0, Direction::PlusZ, 0}.pack()
    );
    config.undergroundMaterial = MaterialRegistry::resolveBlockOr(
        terrain_internal::kDefaultUndergroundMaterial,
        UnpackedBlockMaterial{1u, 0, Direction::PlusZ, 0}.pack()
    );

    const std::filesystem::path path = biomeConfigPath();
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "TerrainGenerator: unable to open biome config '" << path.string()
                  << "'. Using defaults for biome '" << config.name << "'." << std::endl;
        return config;
    }

    json root;
    try {
        file >> root;
    } catch (const std::exception& e) {
        std::cerr << "TerrainGenerator: failed to parse biome config '" << path.string()
                  << "': " << e.what() << ". Using defaults for biome '" << config.name << "'." << std::endl;
        return config;
    }

    if (!root.is_array() || root.empty()) {
        std::cerr << "TerrainGenerator: biome config '" << path.string()
                  << "' must be a non-empty array. Using defaults for biome '" << config.name << "'." << std::endl;
        return config;
    }

    const json* selectedBiome = nullptr;
    size_t selectedIndex = 0;
    for (size_t i = 0; i < root.size(); ++i) {
        const json& candidate = root[i];
        if (!candidate.is_object() || !candidate.contains("name") || !candidate["name"].is_string()) {
            continue;
        }
        if (candidate["name"].get<std::string>() == terrain_internal::kActiveBiomeName) {
            selectedBiome = &candidate;
            selectedIndex = i;
            break;
        }
    }
    if (selectedBiome == nullptr) {
        for (size_t i = 0; i < root.size(); ++i) {
            if (root[i].is_object() && root[i].contains("name") && root[i]["name"].is_string()) {
                selectedBiome = &root[i];
                selectedIndex = i;
                std::cerr << "TerrainGenerator: biome '" << terrain_internal::kActiveBiomeName
                          << "' not found in '" << path.string() << "'. Falling back to '"
                          << (*selectedBiome)["name"].get<std::string>() << "'." << std::endl;
                break;
            }
        }
    }
    if (selectedBiome == nullptr) {
        std::cerr << "TerrainGenerator: biome config '" << path.string()
                  << "' has no valid biome entries. Using defaults for biome '" << config.name << "'." << std::endl;
        return config;
    }

    config.name = (*selectedBiome)["name"].get<std::string>();

    if (!parseWeightedSelectionField(*selectedBiome, "structures", config.structures, selectedIndex, "biomes")) {
        config.structures.clear();
    }
    if (!parseWeightedSelectionField(*selectedBiome, "decorations", config.decorations, selectedIndex, "biomes")) {
        config.decorations.clear();
    }

    if (selectedBiome->contains("noise")) {
        const json& noise = (*selectedBiome)["noise"];
        if (!noise.is_object()) {
            std::cerr << "TerrainGenerator: biomes[" << selectedIndex << "] field 'noise' must be an object."
                      << std::endl;
        } else {
            if (noise.contains("seed") && noise["seed"].is_number_integer()) {
                config.noiseSeed = noise["seed"].get<int>();
            }
            if (noise.contains("horizontalFrequency") && noise["horizontalFrequency"].is_number()) {
                config.noiseHorizontalFrequency = std::max(0.0f, noise["horizontalFrequency"].get<float>());
            }
            if (noise.contains("verticalFrequency") && noise["verticalFrequency"].is_number()) {
                config.noiseVerticalFrequency = std::max(0.0f, noise["verticalFrequency"].get<float>());
            }
            if (noise.contains("maxStrengthBlocks") && noise["maxStrengthBlocks"].is_number()) {
                config.noiseMaxStrengthBlocks = std::max(0.0f, noise["maxStrengthBlocks"].get<float>());
            }
            if (noise.contains("falloffBlocks") && noise["falloffBlocks"].is_number()) {
                config.noiseFalloffBlocks = std::max(0.001f, noise["falloffBlocks"].get<float>());
            }
        }
    }

    if (selectedBiome->contains("materials")) {
        const json& materials = (*selectedBiome)["materials"];
        if (!materials.is_object()) {
            std::cerr << "TerrainGenerator: biomes[" << selectedIndex << "] field 'materials' must be an object."
                      << std::endl;
        } else {
            if (materials.contains("aboveground") && materials["aboveground"].is_string()) {
                const std::string materialName = materials["aboveground"].get<std::string>();
                BlockMaterial material{};
                if (MaterialRegistry::tryResolveBlock(materialName, material)) {
                    config.abovegroundMaterial = material;
                } else {
                    std::cerr << "TerrainGenerator: biomes[" << selectedIndex
                              << "] references unknown aboveground material '" << materialName << "'."
                              << std::endl;
                }
            }
            if (materials.contains("underground") && materials["underground"].is_string()) {
                const std::string materialName = materials["underground"].get<std::string>();
                BlockMaterial material{};
                if (MaterialRegistry::tryResolveBlock(materialName, material)) {
                    config.undergroundMaterial = material;
                } else {
                    std::cerr << "TerrainGenerator: biomes[" << selectedIndex
                              << "] references unknown underground material '" << materialName << "'."
                              << std::endl;
                }
            }
        }
    }

    if (selectedBiome->contains("flatnessThreshold") && (*selectedBiome)["flatnessThreshold"].is_number()) {
        config.flatnessThreshold = std::clamp((*selectedBiome)["flatnessThreshold"].get<float>(), 0.0f, 1.0f);
    }
    if (selectedBiome->contains("decorationChance") && (*selectedBiome)["decorationChance"].is_number()) {
        config.decorationChance = std::clamp((*selectedBiome)["decorationChance"].get<float>(), 0.0f, 1.0f);
    }

    if (selectedBiome->contains("structurePlacement")) {
        const json& structurePlacement = (*selectedBiome)["structurePlacement"];
        if (!structurePlacement.is_object()) {
            std::cerr << "TerrainGenerator: biomes[" << selectedIndex
                      << "] field 'structurePlacement' must be an object." << std::endl;
        } else {
            if (structurePlacement.contains("cellSize") && structurePlacement["cellSize"].is_number_integer()) {
                config.structureCellSize = std::max<int32_t>(1, structurePlacement["cellSize"].get<int32_t>());
            }
            if (structurePlacement.contains("minDistance") && structurePlacement["minDistance"].is_number_integer()) {
                config.structureMinDistance = std::max<int32_t>(1, structurePlacement["minDistance"].get<int32_t>());
            }
            if (structurePlacement.contains("cellOccupancy") && structurePlacement["cellOccupancy"].is_number()) {
                config.structureCellOccupancy = std::clamp(
                    structurePlacement["cellOccupancy"].get<float>(),
                    0.0f,
                    1.0f
                );
            }
            if (structurePlacement.contains("seed") && structurePlacement["seed"].is_number_integer()) {
                config.structureSeed = structurePlacement["seed"].get<uint32_t>();
            }
        }
    }

    return config;
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
    const terrain_internal::BiomeConfig& biome = terrain_internal::biomeConfig();
    config.placementChance = biome.decorationChance;

    config.definitions.reserve(biome.decorations.size());
    for (const terrain_internal::BiomeWeightedSelection& selection : biome.decorations) {
        BlockMaterial material{};
        if (!MaterialRegistry::tryResolveBlock(selection.name, material)) {
            std::cerr << "TerrainGenerator: biome '" << biome.name
                      << "' references unknown decoration material '" << selection.name << "'." << std::endl;
            continue;
        }

        config.definitions.push_back(terrain_internal::TerrainDecorationDefinition{
            selection.name,
            material,
            selection.selectionWeight
        });
        config.totalSelectionWeight += static_cast<uint64_t>(selection.selectionWeight);
    }

    if (config.definitions.empty()) {
        std::cerr << "TerrainGenerator: biome '" << biome.name
                  << "' has no valid decoration material definitions." << std::endl;
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

const BiomeConfig& biomeConfig() {
    static const BiomeConfig kConfig = loadBiomeConfig();
    return kConfig;
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
                    const BiomeConfig& biome,
                    int terrainHeight) {
    const float baseDensity = static_cast<float>(terrainHeight - worldZ);
    if (!fnGenerator) {
        return baseDensity;
    }

    const float distanceFromSurface = std::abs(baseDensity);
    const float noiseFalloff = std::max(0.001f, biome.noiseFalloffBlocks);
    if (distanceFromSurface >= noiseFalloff) {
        return baseDensity;
    }

    const float strengthT = 1.0f - (distanceFromSurface / noiseFalloff);
    const float noiseStrength = std::max(0.0f, biome.noiseMaxStrengthBlocks) * smoothstep01(strengthT);
    if (noiseStrength <= 0.0f) {
        return baseDensity;
    }

    const float nx = static_cast<float>(worldX) * std::max(0.0f, biome.noiseHorizontalFrequency);
    const float ny = static_cast<float>(worldY) * std::max(0.0f, biome.noiseHorizontalFrequency);
    const float nz = static_cast<float>(worldZ) * std::max(0.0f, biome.noiseVerticalFrequency);
    const float noise = fnGenerator->GenSingle3D(nx, ny, nz, biome.noiseSeed);

    return baseDensity + (noise * noiseStrength);
}

void generateTerrainColumn(const glm::ivec3& origin,
                           Column& col,
                           const FastNoise::SmartNode<>& fnGenerator,
                           const HeightmapData& heightmap,
                           const TerrainDecorationConfig& config) {
    const BiomeConfig& biome = biomeConfig();
    const BlockMaterial stonePacked = biome.undergroundMaterial;
    const BlockMaterial grassPacked = biome.abovegroundMaterial;
    const BlockMaterial airPacked = UnpackedBlockMaterial{0, 0, Direction::PlusZ, 0}.pack();
    const uint16_t grassMaterialId = grassPacked.unpack().id;

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
        return sampleDensity(fnGenerator, worldX, worldY, worldZ, biome, terrainHeight);
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

                col.setBlock(x, y, z, (flatness >= biome.flatnessThreshold) ? grassPacked : stonePacked);
            }
        }
    }

    for (int z = 0; z < (kColumnHeight - 1); ++z) {
        for (int y = 0; y < kChunkSize; ++y) {
            for (int x = 0; x < kChunkSize; ++x) {
                if (col.getBlock(x, y, z).unpack().id != grassMaterialId) {
                    continue;
                }
                if (col.getBlock(x, y, z + 1).unpack().id != 0u) {
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
