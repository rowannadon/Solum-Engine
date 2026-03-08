#include "TerrainGeneratorInternal.h"

#include <array>
#include <vector>

#include "solum_engine/resources/Constants.h"
#include "solum_engine/voxel/MaterialLightProperties.h"

namespace {

int levelIndex(int x, int y) {
    return (y * cfg::CHUNK_SIZE) + x;
}

uint8_t attenuateLight(uint8_t light, uint8_t loss) {
    if (light == 0u || loss == MaterialLightProperties::kOpaqueLightLoss || loss >= light) {
        return 0u;
    }
    return static_cast<uint8_t>(light - loss);
}

void rebuildSimpleColumnLighting(Column& col) {
    constexpr uint8_t kMaxSkyLight = 15u;
    constexpr int kLevelArea = cfg::CHUNK_SIZE * cfg::CHUNK_SIZE;
    constexpr std::array<glm::ivec2, 4> kCardinalOffsets = {
        glm::ivec2{1, 0},
        glm::ivec2{-1, 0},
        glm::ivec2{0, 1},
        glm::ivec2{0, -1},
    };

    std::array<uint8_t, kLevelArea> skyFromAbove{};
    std::array<uint8_t, kLevelArea> levelSky{};
    std::array<uint8_t, kLevelArea> blockLightLoss{};
    std::array<uint8_t, kLevelArea> blocksLightMask{};
    skyFromAbove.fill(kMaxSkyLight);

    std::vector<int> floodQueue;
    floodQueue.reserve(kLevelArea);

    for (int z = cfg::COLUMN_HEIGHT_BLOCKS - 1; z >= 0; --z) {
        floodQueue.clear();
        size_t queueHead = 0u;

        for (int y = 0; y < cfg::CHUNK_SIZE; ++y) {
            for (int x = 0; x < cfg::CHUNK_SIZE; ++x) {
                const int idx = levelIndex(x, y);
                const BlockMaterial block = col.getBlock(
                    static_cast<uint8_t>(x),
                    static_cast<uint8_t>(y),
                    static_cast<uint16_t>(z)
                );
                const uint16_t materialId = block.unpack().id;
                const bool blocksLight = MaterialLightProperties::blocksLight(materialId);
                const uint8_t verticalLoss = MaterialLightProperties::skyLightVerticalLoss(materialId);
                blocksLightMask[static_cast<size_t>(idx)] = blocksLight ? 1u : 0u;
                blockLightLoss[static_cast<size_t>(idx)] = MaterialLightProperties::blockLightStepLoss(materialId);

                if (blocksLight) {
                    levelSky[static_cast<size_t>(idx)] = 0u;
                    continue;
                }

                const uint8_t directSky = attenuateLight(skyFromAbove[static_cast<size_t>(idx)], verticalLoss);
                levelSky[static_cast<size_t>(idx)] = directSky;
                if (directSky > 0u) {
                    floodQueue.push_back(idx);
                }
            }
        }

        while (queueHead < floodQueue.size()) {
            const int idx = floodQueue[queueHead++];
            const uint8_t current = levelSky[static_cast<size_t>(idx)];
            if (current == 0u) {
                continue;
            }

            const int x = idx % cfg::CHUNK_SIZE;
            const int y = idx / cfg::CHUNK_SIZE;

            for (const glm::ivec2& offset : kCardinalOffsets) {
                const int nx = x + offset.x;
                const int ny = y + offset.y;
                if (nx < 0 || ny < 0 || nx >= cfg::CHUNK_SIZE || ny >= cfg::CHUNK_SIZE) {
                    continue;
                }

                const int neighborIdx = levelIndex(nx, ny);
                if (blocksLightMask[static_cast<size_t>(neighborIdx)] != 0u) {
                    continue;
                }

                const uint8_t propagated = attenuateLight(
                    current,
                    blockLightLoss[static_cast<size_t>(neighborIdx)]
                );
                if (propagated == 0u) {
                    continue;
                }

                uint8_t& neighborSky = levelSky[static_cast<size_t>(neighborIdx)];
                if (propagated <= neighborSky) {
                    continue;
                }

                neighborSky = propagated;
                floodQueue.push_back(neighborIdx);
            }
        }

        for (int y = 0; y < cfg::CHUNK_SIZE; ++y) {
            for (int x = 0; x < cfg::CHUNK_SIZE; ++x) {
                const int idx = levelIndex(x, y);
                const uint8_t sky = (blocksLightMask[static_cast<size_t>(idx)] != 0u)
                    ? 0u
                    : levelSky[static_cast<size_t>(idx)];
                const BlockMaterial block = col.getBlock(
                    static_cast<uint8_t>(x),
                    static_cast<uint8_t>(y),
                    static_cast<uint16_t>(z)
                );
                const uint8_t emissive = MaterialLightProperties::emissiveLight(block.unpack().id);

                col.setPackedLight(
                    static_cast<uint8_t>(x),
                    static_cast<uint8_t>(y),
                    static_cast<uint16_t>(z),
                    Chunk::packLight(sky, emissive)
                );
                skyFromAbove[static_cast<size_t>(idx)] = sky;
            }
        }
    }
}

}  // namespace

namespace terrain_internal {

void bootstrapColumnLighting(Column& col) {
    rebuildSimpleColumnLighting(col);
}

}  // namespace terrain_internal
