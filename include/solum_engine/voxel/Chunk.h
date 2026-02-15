#pragma once

#include "solum_engine/resources/Constants.h"
#include "solum_engine/resources/Coords.h"
#include "solum_engine/voxel/BlockMaterial.h"
#include "solum_engine/voxel/ChunkMeshes.h"
#include "solum_engine/voxel/ChunkState.h"
#include "solum_engine/voxel/LodStorage.h"

#include <array>
#include <cstddef>
#include <cstdint>

class Chunk {
public:
    explicit Chunk(ChunkCoord coord);

    ChunkCoord coord() const;

    bool setBlock(BlockCoord localPos, UnpackedBlockMaterial mat);
    UnpackedBlockMaterial getBlock(BlockCoord localPos) const;

    BlockMaterial* getBlockData();
    const BlockMaterial* getBlockData() const;

    ChunkState& state();
    const ChunkState& state() const;

    LodStorage& lodStorage();
    const LodStorage& lodStorage() const;

    bool rebuildLodData();

    MeshHandle meshHandleL0() const;
    void setMeshHandleL0(MeshHandle handle, uint32_t derivedFromVersion);

    void markBulkDataWrite();

private:
    static bool validateLocalPos(BlockCoord pos);
    static std::size_t localIndex(BlockCoord pos);

    ChunkCoord coord_;

    std::array<BlockMaterial, CHUNK_BLOCKS> bootstrapData_{};

    ChunkState state_;
    LodStorage lodStorage_;
    MeshHandle meshHandleL0_ = MeshHandle::invalid();

};
