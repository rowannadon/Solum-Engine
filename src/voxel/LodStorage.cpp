#include "solum_engine/voxel/LodStorage.h"

#include <unordered_map>

namespace {
constexpr int kChunkSize = CHUNK_SIZE;
constexpr int kChunkPlaneArea = CHUNK_SIZE * CHUNK_SIZE;

std::size_t blockIndex(int x, int y, int z) {
    return static_cast<std::size_t>((x * kChunkPlaneArea) + (y * kChunkSize) + z);
}
}

void LodStorage::clear() {
    dominantMaterials_.fill(0);
    sourceVersion_ = 0;
}

void LodStorage::rebuild(const BlockMaterial* blocks, uint32_t blockDataVersion) {
    if (blocks == nullptr) {
        clear();
        return;
    }

    for (int level = 0; level < kLevelCount; ++level) {
        const int dimension = levelDimension(level);
        const int step = 1 << (level + 1);

        for (int x = 0; x < dimension; ++x) {
            for (int y = 0; y < dimension; ++y) {
                for (int z = 0; z < dimension; ++z) {
                    std::unordered_map<uint16_t, uint16_t> counts;
                    counts.reserve(static_cast<std::size_t>(step * step * step));

                    for (int lx = 0; lx < step; ++lx) {
                        for (int ly = 0; ly < step; ++ly) {
                            for (int lz = 0; lz < step; ++lz) {
                                const int blockX = x * step + lx;
                                const int blockY = y * step + ly;
                                const int blockZ = z * step + lz;

                                const BlockMaterial block = blocks[blockIndex(blockX, blockY, blockZ)];
                                const uint16_t material = static_cast<uint16_t>((block.data >> 16u) & 0xFFFFu);
                                counts[material] += 1;
                            }
                        }
                    }

                    uint16_t bestMaterial = 0;
                    uint16_t bestCount = 0;
                    for (const auto& [material, count] : counts) {
                        if (count > bestCount || (count == bestCount && material < bestMaterial)) {
                            bestMaterial = material;
                            bestCount = count;
                        }
                    }

                    setDominantMaterial(level, x, y, z, bestMaterial);
                }
            }
        }
    }

    sourceVersion_ = blockDataVersion;
}

bool LodStorage::isUpToDate(uint32_t blockDataVersion) const {
    return sourceVersion_ == blockDataVersion;
}

uint32_t LodStorage::sourceVersion() const {
    return sourceVersion_;
}

uint16_t LodStorage::dominantMaterial(int level, int x, int y, int z) const {
    return dominantMaterials_[levelIndex(level, x, y, z)];
}

void LodStorage::setDominantMaterial(int level, int x, int y, int z, uint16_t materialId) {
    dominantMaterials_[levelIndex(level, x, y, z)] = materialId;
}

int LodStorage::levelDimension(int level) {
    if (level < 0 || level >= kLevelCount) {
        return 0;
    }

    return kDimensions[static_cast<std::size_t>(level)];
}

std::size_t LodStorage::levelIndex(int level, int x, int y, int z) {
    const int dimension = levelDimension(level);
    if (dimension <= 0 || x < 0 || y < 0 || z < 0 || x >= dimension || y >= dimension || z >= dimension) {
        return 0;
    }

    const std::size_t offset = kOffsets[static_cast<std::size_t>(level)];
    return offset + static_cast<std::size_t>(((x * dimension) + y) * dimension + z);
}
