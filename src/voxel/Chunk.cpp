#include "solum_engine/voxel/Chunk.h"

#include <array>
#include <utility>

namespace {
BlockMaterial makeAirBlock() {
    static const BlockMaterial kAir = UnpackedBlockMaterial{}.pack();
    return kAir;
}
}  // namespace

Chunk::Chunk() {
    for (uint8_t level = 0; level <= MAX_MIP_LEVEL; ++level) {
        MipStorage& storage = mips_[level];
        storage.bitsPerBlock = 0;
        storage.size = mipSize(level);
        storage.palette.assign(1, airBlock());
        storage.data.clear();
    }
    solidVoxelCount_ = 0;
}

BlockMaterial Chunk::getBlock(uint8_t x, uint8_t y, uint8_t z, uint8_t mipLevel) const {
    const uint8_t level = std::min<uint8_t>(mipLevel, MAX_MIP_LEVEL);
    const MipStorage& storage = mips_[level];
    if (x >= storage.size || y >= storage.size || z >= storage.size) {
        return airBlock();
    }

    if (storage.bitsPerBlock == 0) {
        return storage.palette.empty() ? airBlock() : storage.palette[0];
    }

    const uint16_t voxelIndex = getVoxelIndex(x, y, z, storage.size);
    const uint32_t paletteIndex = getPaletteIndex(storage, voxelIndex);
    if (paletteIndex >= storage.palette.size()) {
        return storage.palette.empty() ? airBlock() : storage.palette[0];
    }
    return storage.palette[paletteIndex];
}

void Chunk::setBlock(uint8_t x, uint8_t y, uint8_t z, const BlockMaterial blockID) {
    if (x >= SIZE || y >= SIZE || z >= SIZE) {
        return;
    }

    const BlockMaterial previousBlock = getBlock(x, y, z, 0);
    const bool previousSolid = isSolid(previousBlock);
    const bool newSolid = isSolid(blockID);

    bool levelChanged = false;
    setBlockInStorage(mips_[0], x, y, z, blockID, &levelChanged);
    if (!levelChanged) {
        return;
    }

    if (previousSolid != newSolid) {
        if (newSolid) {
            ++solidVoxelCount_;
        } else if (solidVoxelCount_ > 0) {
            --solidVoxelCount_;
        }
    }

    uint8_t px = x;
    uint8_t py = y;
    uint8_t pz = z;

    for (uint8_t level = 1; level <= MAX_MIP_LEVEL; ++level) {
        px >>= 1;
        py >>= 1;
        pz >>= 1;

        BlockMaterial parentBlock = downsampleBlockFromChildren(mips_[level - 1], px, py, pz);

        bool parentChanged = false;
        setBlockInStorage(mips_[level], px, py, pz, parentBlock, &parentChanged);
        if (!parentChanged) {
            break;
        }
    }
}

uint16_t Chunk::getVoxelIndex(uint8_t x, uint8_t y, uint8_t z, uint8_t size) {
    const uint16_t stride = static_cast<uint16_t>(size);
    return static_cast<uint16_t>((static_cast<uint16_t>(z) * stride * stride) +
                                 (static_cast<uint16_t>(y) * stride) +
                                 static_cast<uint16_t>(x));
}

uint32_t Chunk::getPaletteIndex(const MipStorage& storage, uint16_t voxelIndex) {
    const size_t bitsPerBlock = storage.bitsPerBlock;
    const size_t bitIndex = static_cast<size_t>(voxelIndex) * bitsPerBlock;
    const size_t wordIndex = bitIndex / 64;
    const size_t bitOffset = bitIndex % 64;

    if (bitOffset + bitsPerBlock <= 64) {
        const uint64_t mask = (1ULL << bitsPerBlock) - 1ULL;
        return static_cast<uint32_t>((storage.data[wordIndex] >> bitOffset) & mask);
    }

    const size_t bitsInFirst = 64 - bitOffset;
    const size_t bitsInNext = bitsPerBlock - bitsInFirst;

    const uint32_t val1 = static_cast<uint32_t>(
        (storage.data[wordIndex] >> bitOffset) & ((1ULL << bitsInFirst) - 1ULL));
    const uint32_t val2 = static_cast<uint32_t>(
        storage.data[wordIndex + 1] & ((1ULL << bitsInNext) - 1ULL));
    return val1 | (val2 << bitsInFirst);
}

void Chunk::setPaletteIndex(MipStorage& storage, uint16_t voxelIndex, uint32_t paletteIndex) {
    const size_t bitsPerBlock = storage.bitsPerBlock;
    const size_t bitIndex = static_cast<size_t>(voxelIndex) * bitsPerBlock;
    const size_t wordIndex = bitIndex / 64;
    const size_t bitOffset = bitIndex % 64;

    if (bitOffset + bitsPerBlock <= 64) {
        const uint64_t mask = ((1ULL << bitsPerBlock) - 1ULL) << bitOffset;
        storage.data[wordIndex] =
            (storage.data[wordIndex] & ~mask) | (static_cast<uint64_t>(paletteIndex) << bitOffset);
        return;
    }

    const size_t bitsInFirst = 64 - bitOffset;
    const size_t bitsInNext = bitsPerBlock - bitsInFirst;

    const uint64_t mask1 = ((1ULL << bitsInFirst) - 1ULL) << bitOffset;
    storage.data[wordIndex] =
        (storage.data[wordIndex] & ~mask1) |
        ((static_cast<uint64_t>(paletteIndex) & ((1ULL << bitsInFirst) - 1ULL)) << bitOffset);

    const uint64_t mask2 = (1ULL << bitsInNext) - 1ULL;
    storage.data[wordIndex + 1] =
        (storage.data[wordIndex + 1] & ~mask2) |
        (static_cast<uint64_t>(paletteIndex) >> bitsInFirst);
}

void Chunk::resizeBitArray(MipStorage& storage, uint8_t newBitsPerBlock) {
    const size_t oldBitsPerBlock = storage.bitsPerBlock;
    const uint8_t size = storage.size;
    const size_t volume = static_cast<size_t>(size) * static_cast<size_t>(size) * static_cast<size_t>(size);
    const size_t newDataWords = (volume * newBitsPerBlock + 63) / 64;

    std::vector<uint64_t> oldData = std::move(storage.data);
    storage.data.assign(newDataWords, 0ULL);
    storage.bitsPerBlock = newBitsPerBlock;

    if (oldBitsPerBlock == 0) {
        return;
    }

    for (size_t i = 0; i < volume; ++i) {
        const size_t bitIndex = i * oldBitsPerBlock;
        const size_t wordIndex = bitIndex / 64;
        const size_t bitOffset = bitIndex % 64;

        uint32_t paletteIndex = 0;
        if (bitOffset + oldBitsPerBlock <= 64) {
            paletteIndex = static_cast<uint32_t>(
                (oldData[wordIndex] >> bitOffset) & ((1ULL << oldBitsPerBlock) - 1ULL));
        } else {
            const size_t bitsInFirst = 64 - bitOffset;
            const size_t bitsInNext = oldBitsPerBlock - bitsInFirst;
            const uint32_t v1 = static_cast<uint32_t>(
                (oldData[wordIndex] >> bitOffset) & ((1ULL << bitsInFirst) - 1ULL));
            const uint32_t v2 = static_cast<uint32_t>(
                oldData[wordIndex + 1] & ((1ULL << bitsInNext) - 1ULL));
            paletteIndex = v1 | (v2 << bitsInFirst);
        }

        setPaletteIndex(storage, static_cast<uint16_t>(i), paletteIndex);
    }
}

bool Chunk::isSolid(BlockMaterial block) {
    return block.unpack().id != 0u;
}

BlockMaterial Chunk::airBlock() {
    return makeAirBlock();
}

BlockMaterial Chunk::downsampleBlockFromChildren(const MipStorage& childLevel,
                                                 uint8_t px,
                                                 uint8_t py,
                                                 uint8_t pz) {
    const uint8_t cx = static_cast<uint8_t>(px << 1);
    const uint8_t cy = static_cast<uint8_t>(py << 1);
    const uint8_t cz = static_cast<uint8_t>(pz << 1);

    std::array<BlockMaterial, 8> candidates{};
    std::array<uint8_t, 8> candidateCounts{};
    std::array<uint8_t, 8> candidateExposedCounts{};
    size_t candidateCount = 0;
    uint8_t solidChildCount = 0;
    bool hasExposedCandidate = false;

    auto sampleChildBlock = [&childLevel](uint8_t x, uint8_t y, uint8_t z) -> BlockMaterial {
        if (childLevel.bitsPerBlock == 0) {
            return childLevel.palette.empty() ? airBlock() : childLevel.palette[0];
        }

        const uint32_t childIndex = getPaletteIndex(
            childLevel,
            getVoxelIndex(x, y, z, childLevel.size));
        if (childIndex >= childLevel.palette.size()) {
            return airBlock();
        }
        return childLevel.palette[childIndex];
    };

    auto isAirNeighbor = [&sampleChildBlock, &childLevel](int32_t x, int32_t y, int32_t z) -> bool {
        const int32_t size = static_cast<int32_t>(childLevel.size);
        if (x < 0 || y < 0 || z < 0 || x >= size || y >= size || z >= size) {
            // Neighbor chunks are unavailable here, so out-of-bounds is treated as unknown-solid.
            return false;
        }

        return !isSolid(sampleChildBlock(
            static_cast<uint8_t>(x),
            static_cast<uint8_t>(y),
            static_cast<uint8_t>(z)));
    };

    static constexpr std::array<std::array<int8_t, 3>, 6> kNeighborOffsets{{
        {{1, 0, 0}},
        {{-1, 0, 0}},
        {{0, 1, 0}},
        {{0, -1, 0}},
        {{0, 0, 1}},
        {{0, 0, -1}},
    }};

    for (uint8_t dz = 0; dz < 2; ++dz) {
        for (uint8_t dy = 0; dy < 2; ++dy) {
            for (uint8_t dx = 0; dx < 2; ++dx) {
                const uint8_t childX = static_cast<uint8_t>(cx + dx);
                const uint8_t childY = static_cast<uint8_t>(cy + dy);
                const uint8_t childZ = static_cast<uint8_t>(cz + dz);
                const BlockMaterial child = sampleChildBlock(childX, childY, childZ);

                if (!isSolid(child)) {
                    continue;
                }

                ++solidChildCount;

                bool exposedToAir = false;
                for (const auto& offset : kNeighborOffsets) {
                    if (isAirNeighbor(
                            static_cast<int32_t>(childX) + offset[0],
                            static_cast<int32_t>(childY) + offset[1],
                            static_cast<int32_t>(childZ) + offset[2])) {
                        exposedToAir = true;
                        break;
                    }
                }

                bool merged = false;
                for (size_t i = 0; i < candidateCount; ++i) {
                    if (candidates[i] == child) {
                        ++candidateCounts[i];
                        if (exposedToAir) {
                            ++candidateExposedCounts[i];
                            hasExposedCandidate = true;
                        }
                        merged = true;
                        break;
                    }
                }

                if (!merged && candidateCount < candidates.size()) {
                    candidates[candidateCount] = child;
                    candidateCounts[candidateCount] = 1;
                    if (exposedToAir) {
                        candidateExposedCounts[candidateCount] = 1;
                        hasExposedCandidate = true;
                    }
                    ++candidateCount;
                }
            }
        }
    }

    if (solidChildCount < 4 || candidateCount == 0) {
        return airBlock();
    }

    size_t bestIndex = 0;
    for (size_t i = 1; i < candidateCount; ++i) {
        if (hasExposedCandidate) {
            if (candidateExposedCounts[i] > candidateExposedCounts[bestIndex] ||
                (candidateExposedCounts[i] == candidateExposedCounts[bestIndex] &&
                 candidateCounts[i] > candidateCounts[bestIndex])) {
                bestIndex = i;
            }
            continue;
        }

        if (candidateCounts[i] > candidateCounts[bestIndex]) {
            bestIndex = i;
        }
    }

    return candidates[bestIndex];
}

void Chunk::setBlockInStorage(MipStorage& storage,
                              uint8_t x,
                              uint8_t y,
                              uint8_t z,
                              BlockMaterial blockID,
                              bool* outChanged) {
    if (outChanged != nullptr) {
        *outChanged = false;
    }

    if (x >= storage.size || y >= storage.size || z >= storage.size) {
        return;
    }

    const uint16_t voxelIndex = getVoxelIndex(x, y, z, storage.size);

    auto it = std::find(storage.palette.begin(), storage.palette.end(), blockID);
    uint32_t paletteIndex = 0;

    if (it != storage.palette.end()) {
        paletteIndex = static_cast<uint32_t>(std::distance(storage.palette.begin(), it));
    } else {
        paletteIndex = static_cast<uint32_t>(storage.palette.size());
        storage.palette.push_back(blockID);

        if (storage.palette.size() > (1ULL << storage.bitsPerBlock)) {
            const uint8_t newBitsPerBlock = storage.bitsPerBlock == 0 ? 1 : storage.bitsPerBlock + 1;
            resizeBitArray(storage, newBitsPerBlock);
        }
    }

    const uint32_t previousIndex = (storage.bitsPerBlock == 0)
        ? 0u
        : getPaletteIndex(storage, voxelIndex);

    if (previousIndex == paletteIndex) {
        return;
    }

    if (storage.bitsPerBlock > 0) {
        setPaletteIndex(storage, voxelIndex, paletteIndex);
    }

    if (outChanged != nullptr) {
        *outChanged = true;
    }
}
