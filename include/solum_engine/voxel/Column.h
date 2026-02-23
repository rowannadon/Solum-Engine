#pragma once
#include "solum_engine/voxel/Chunk.h"
#include "solum_engine/voxel/BlockMaterial.h"
#include <array>
#include <cstdint>

class Column {
public:
    static constexpr size_t HEIGHT = 32;
    static constexpr uint32_t allChunksEmptyMask() {
        const uint32_t clampedBits = (HEIGHT >= 32) ? 32u : static_cast<uint32_t>(HEIGHT);
        return (clampedBits == 0u) ? 0u : (0xFFFFFFFFu >> (32u - clampedBits));
    }

    Column() = default;

    // Z is the vertical axis (z-up). X/Y address the horizontal plane.
    inline BlockMaterial getBlock(uint8_t x, uint8_t y, uint16_t z, uint8_t mipLevel = 0) const {
        const uint8_t chunkSizeAtMip = Chunk::mipSize(mipLevel);
        if (x >= chunkSizeAtMip || y >= chunkSizeAtMip) {
            return UnpackedBlockMaterial{}.pack();
        }
        const uint8_t chunk_z = static_cast<uint8_t>(z / chunkSizeAtMip);
        if (chunk_z >= HEIGHT) {
            return UnpackedBlockMaterial{}.pack();
        }
        const uint8_t local_z = static_cast<uint8_t>(z % chunkSizeAtMip);
        return chunks_[chunk_z].getBlock(x, y, local_z, mipLevel);
    }

    inline void setBlock(uint8_t x, uint8_t y, uint16_t z, const BlockMaterial blockID) {
        const uint8_t chunk_z = static_cast<uint8_t>(z / Chunk::SIZE);
        if (chunk_z >= HEIGHT) {
            return;
        }

        const uint8_t local_z = static_cast<uint8_t>(z % Chunk::SIZE);
        Chunk& chunk = chunks_[chunk_z];
        const bool wasEmpty = chunk.isAllAir();
        chunk.setBlock(x, y, local_z, blockID);
        const bool isEmpty = chunk.isAllAir();

        if (wasEmpty == isEmpty) {
            return;
        }

        const uint32_t bit = (1u << chunk_z);
        if (isEmpty) {
            emptyChunkMask_ |= bit;
        } else {
            emptyChunkMask_ &= ~bit;
        }
    }

    Chunk& getChunk(uint8_t chunk_z) { return chunks_[chunk_z]; }
    const Chunk& getChunk(uint8_t chunk_z) const { return chunks_[chunk_z]; }
    uint32_t getEmptyChunkMask() const noexcept { return emptyChunkMask_; }
    bool isChunkEmpty(uint8_t chunk_z) const noexcept {
        if (chunk_z >= HEIGHT) {
            return true;
        }
        return (emptyChunkMask_ & (1u << chunk_z)) != 0u;
    }

    void rebuildEmptyChunkMask() noexcept {
        emptyChunkMask_ = 0u;
        for (uint8_t chunk_z = 0; chunk_z < HEIGHT; ++chunk_z) {
            if (chunks_[chunk_z].isAllAir()) {
                emptyChunkMask_ |= (1u << chunk_z);
            }
        }
    }

private:
    std::array<Chunk, HEIGHT> chunks_;
    uint32_t emptyChunkMask_ = allChunksEmptyMask();
};
