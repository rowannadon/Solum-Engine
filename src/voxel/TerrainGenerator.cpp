#include "solum_engine/voxel/TerrainGenerator.h"
#include "solum_engine/resources/Constants.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "lodepng/lodepng.h"

namespace {

constexpr int kHeightmapUpscaleFactor = 2;
constexpr int kFallbackTerrainHeight = 100;

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

int sampleTerrainHeight(const HeightmapData& heightmap, int worldX, int worldY) {
    if (!heightmap.valid || heightmap.width <= 0 || heightmap.height <= 0 || heightmap.heights.empty()) {
        return std::clamp(kFallbackTerrainHeight, 0, cfg::COLUMN_HEIGHT_BLOCKS - 1);
    }

    const int sx = wrapIndex(worldX, heightmap.width);
    const int sy = wrapIndex(worldY, heightmap.height);
    const float normalized = heightmap.heights[static_cast<size_t>(sy) * static_cast<size_t>(heightmap.width) +
                                               static_cast<size_t>(sx)];
    const int maxTerrainHeight = cfg::COLUMN_HEIGHT_BLOCKS - 1;
    const int sampledHeight = static_cast<int>(std::lround(normalized * static_cast<float>(maxTerrainHeight)));
    return std::clamp(sampledHeight, 0, maxTerrainHeight);
}

} // namespace

void TerrainGenerator::generateColumn(const glm::ivec3& origin, Column& col) {
    const HeightmapData& heightmap = getHeightmapData();

    UnpackedBlockMaterial solid{1, 0, Direction::PlusZ, 0};
    UnpackedBlockMaterial air{0, 0, Direction::PlusZ, 0};

    const BlockMaterial solidPacked = solid.pack();
    const BlockMaterial airPacked = air.pack();

    std::array<int, static_cast<size_t>(cfg::CHUNK_SIZE) * static_cast<size_t>(cfg::CHUNK_SIZE)> columnHeights{};

    for (int y = 0; y < cfg::CHUNK_SIZE; ++y) {
        for (int x = 0; x < cfg::CHUNK_SIZE; ++x) {
            const int worldX = origin.x + x;
            const int worldY = origin.y + y;
            columnHeights[static_cast<size_t>(y) * static_cast<size_t>(cfg::CHUNK_SIZE) + static_cast<size_t>(x)] =
                sampleTerrainHeight(heightmap, worldX, worldY);
        }
    }

    for (int z = 0; z < cfg::COLUMN_HEIGHT_BLOCKS; z++)
    {
        for (int y = 0; y < cfg::CHUNK_SIZE; y++)
        {
            for (int x = 0; x < cfg::CHUNK_SIZE; x++)
            {
                const int terrainHeight = columnHeights[static_cast<size_t>(y) *
                                                        static_cast<size_t>(cfg::CHUNK_SIZE) +
                                                        static_cast<size_t>(x)];
                if (z <= terrainHeight) {
                    col.setBlock(x, y, z, solidPacked);
                } else {
                    col.setBlock(x, y, z, airPacked);
                }
            }
        }
    }
}
