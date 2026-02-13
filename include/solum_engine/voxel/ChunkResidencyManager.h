#pragma once

#include "solum_engine/voxel/Chunk.h"

#include <cstddef>
#include <vector>

class ChunkPool;
class CompressedStore;

class ChunkResidencyManager {
public:
    ChunkResidencyManager(ChunkPool& pool, CompressedStore& compressedStore);

    bool ensureResident(Chunk& chunk);
    bool compressChunk(Chunk& chunk);
    bool uncompressChunk(Chunk& chunk);

    bool pinChunk(Chunk& chunk);
    bool unpinChunk(Chunk& chunk);

    std::size_t evictChunks(std::vector<Chunk*>& candidates, std::size_t targetFreeSlots);

    std::size_t freeSlots() const;

private:
    std::vector<uint8_t> encodeRle(const BlockMaterial* blocks) const;
    bool decodeRle(const std::vector<uint8_t>& bytes, BlockMaterial* outBlocks) const;

    ChunkPool& pool_;
    CompressedStore& compressedStore_;
};
