#pragma once
#include "solum_engine/voxel/Chunk.h"
#include "solum_engine/voxel/BlockMaterial.h"
#include <array>

class Column {
public:
    static constexpr size_t HEIGHT = 32;

    Column() = default;

    inline BlockMaterial getBlock(uint8_t x, uint16_t y, uint8_t z) const {
        uint8_t chunk_y = y / Chunk::SIZE;
        uint8_t local_y = y % Chunk::SIZE;
        return chunks_[chunk_y].getBlock(x, local_y, z);
    }

    inline void setBlock(uint8_t x, uint16_t y, uint8_t z, const BlockMaterial blockID) {
        uint8_t chunk_y = y / Chunk::SIZE;
        uint8_t local_y = y % Chunk::SIZE;
        chunks_[chunk_y].setBlock(x, local_y, z, blockID);
    }

    Chunk& getChunk(uint8_t chunk_y) { return chunks_[chunk_y]; }
    const Chunk& getChunk(uint8_t chunk_y) const { return chunks_[chunk_y]; }

private:
    std::array<Chunk, HEIGHT> chunks_;
};