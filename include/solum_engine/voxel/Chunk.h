#pragma once

#include "solum_engine/resources/Constants.h"
#include "solum_engine/resources/Coords.h"
#include "solum_engine/voxel/BlockMaterial.h"
#include "solum_engine/voxel/ChunkMeshes.h"
#include "solum_engine/voxel/ChunkPool.h"
#include "solum_engine/voxel/ChunkState.h"
#include "solum_engine/voxel/CompressedStore.h"
#include "solum_engine/voxel/LodStorage.h"

#include <array>
#include <cstddef>
#include <cstdint>

class ChunkPool;

class ChunkMesher;

class Chunk {
public:
    explicit Chunk(ChunkCoord coord, ChunkPool* pool = nullptr);

    ChunkCoord coord() const;

    bool setBlock(BlockCoord localPos, UnpackedBlockMaterial mat);
    UnpackedBlockMaterial getBlock(BlockCoord localPos) const;

    BlockMaterial* getBlockData();
    const BlockMaterial* getBlockData() const;

    bool attachUncompressedStorage(ChunkPool& pool, UncompressedChunkHandle handle);
    void clearUncompressedStorage();

    bool isPoolResident() const;
    bool isResident() const;

    UncompressedChunkHandle uncompressedHandle() const;
    void setCompressedHandle(CompressedChunkHandle handle);
    CompressedChunkHandle compressedHandle() const;
    void clearCompressedHandle();

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
    ChunkPool* pool_ = nullptr;
    bool bootstrapActive_ = true;

    UncompressedChunkHandle uncompressedHandle_ = UncompressedChunkHandle::invalid();
    CompressedChunkHandle compressedHandle_ = CompressedChunkHandle::invalid();

    std::array<BlockMaterial, CHUNK_BLOCKS> bootstrapData_{};

    ChunkState state_;
    LodStorage lodStorage_;
    MeshHandle meshHandleL0_ = MeshHandle::invalid();

    friend class ChunkMesher;
};
