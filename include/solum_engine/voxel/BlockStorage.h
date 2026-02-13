#pragma once

#include <array>
#include <cstddef>

#include "solum_engine/resources/Constants.h"
#include "solum_engine/resources/Coords.h"
#include "solum_engine/voxel/BlockMaterial.h"

class BlockStorage {
public:
    bool setBlock(BlockCoord pos, UnpackedBlockMaterial mat);
    UnpackedBlockMaterial getBlock(BlockCoord pos) const;

    BlockMaterial* getData() {
        return data_.data();
    }

    const BlockMaterial* getData() const {
        return data_.data();
    }

private:
    bool validatePos(BlockCoord pos) const;
    std::size_t getIndex(BlockCoord pos) const;

    std::array<BlockMaterial, CHUNK_BLOCKS> data_{};
};
