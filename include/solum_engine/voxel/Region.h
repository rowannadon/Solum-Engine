#pragma once
#include "solum_engine/voxel/Column.h"
#include "solum_engine/voxel/BlockMaterial.h"
#include "solum_engine/resources/Coords.h"
#include <array>

class Region {
public:
    static constexpr size_t SIZE = 32;

    explicit Region(const RegionCoord& coord) : coord_(coord) {}

    BlockMaterial getBlock(uint16_t x, uint16_t y, uint16_t z) const {
        uint8_t col_x = x / Chunk::SIZE;
        uint8_t col_y = y / Chunk::SIZE;
        uint8_t local_x = x % Chunk::SIZE;
        uint8_t local_y = y % Chunk::SIZE;
        
        return getColumn(col_x, col_y).getBlock(local_x, local_y, z);
    }

    void setBlock(uint16_t x, uint16_t y, uint16_t z, BlockMaterial blockID) {
        uint8_t col_x = x / Chunk::SIZE;
        uint8_t col_y = y / Chunk::SIZE;
        uint8_t local_x = x % Chunk::SIZE;
        uint8_t local_y = y % Chunk::SIZE;
        
        getColumn(col_x, col_y).setBlock(local_x, local_y, z, blockID);
    }

    Column& getColumn(uint8_t x, uint8_t y) {
        return columns_[y * SIZE + x];
    }

    const Column& getColumn(uint8_t x, uint8_t y) const {
        return columns_[y * SIZE + x];
    }

    RegionCoord getCoord() const { return coord_; }

private:
    RegionCoord coord_;

    std::array<Column, SIZE * SIZE> columns_;
};
