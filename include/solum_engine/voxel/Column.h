#pragma once
#include "solum_engine/voxel/Chunk.h"
#include "solum_engine/voxel/BlockMaterial.h"
#include <array>

class Column {
public:
    static constexpr size_t HEIGHT = 32;

    Column() = default;

    // Z is the vertical axis (z-up). X/Y address the horizontal plane.
    inline BlockMaterial getBlock(uint8_t x, uint8_t y, uint16_t z) const {
        uint8_t chunk_z = z / Chunk::SIZE;
        uint8_t local_z = z % Chunk::SIZE;
        return chunks_[chunk_z].getBlock(x, y, local_z);
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
