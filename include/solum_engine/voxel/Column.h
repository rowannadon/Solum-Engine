#pragma once
#include "solum_engine/voxel/Chunk.h"
#include "solum_engine/voxel/BlockMaterial.h"
#include <array>

class Column {
public:
    static constexpr size_t HEIGHT = 32;

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
        uint8_t chunk_z = z / Chunk::SIZE;
        uint8_t local_z = z % Chunk::SIZE;
        chunks_[chunk_z].setBlock(x, y, local_z, blockID);
    }

    Chunk& getChunk(uint8_t chunk_z) { return chunks_[chunk_z]; }
    const Chunk& getChunk(uint8_t chunk_z) const { return chunks_[chunk_z]; }

private:
    std::array<Chunk, HEIGHT> chunks_;
};
