#include "solum_engine/voxel/ChunkResidencyManager.h"

#include "solum_engine/voxel/ChunkPool.h"
#include "solum_engine/voxel/CompressedStore.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace {
constexpr uint8_t kCodecRleV1 = 1;

void appendU32(std::vector<uint8_t>& bytes, uint32_t value) {
    bytes.push_back(static_cast<uint8_t>(value & 0xFFu));
    bytes.push_back(static_cast<uint8_t>((value >> 8u) & 0xFFu));
    bytes.push_back(static_cast<uint8_t>((value >> 16u) & 0xFFu));
    bytes.push_back(static_cast<uint8_t>((value >> 24u) & 0xFFu));
}

bool readU32(const std::vector<uint8_t>& bytes, std::size_t offset, uint32_t& outValue) {
    if (offset + 4u > bytes.size()) {
        return false;
    }

    outValue = static_cast<uint32_t>(bytes[offset])
        | (static_cast<uint32_t>(bytes[offset + 1u]) << 8u)
        | (static_cast<uint32_t>(bytes[offset + 2u]) << 16u)
        | (static_cast<uint32_t>(bytes[offset + 3u]) << 24u);
    return true;
}
}

ChunkResidencyManager::ChunkResidencyManager(ChunkPool& pool, CompressedStore& compressedStore)
    : pool_(pool), compressedStore_(compressedStore) {}

bool ChunkResidencyManager::ensureResident(Chunk& chunk) {
    if (chunk.isPoolResident()) {
        return true;
    }

    if (chunk.compressedHandle().isValid()) {
        return uncompressChunk(chunk);
    }

    UncompressedChunkHandle handle = pool_.allocate();
    if (!handle.isValid()) {
        return false;
    }

    if (!chunk.attachUncompressedStorage(pool_, handle)) {
        pool_.release(handle);
        return false;
    }

    return true;
}

bool ChunkResidencyManager::compressChunk(Chunk& chunk) {
    if (!chunk.isPoolResident()) {
        return true;
    }

    const UncompressedChunkHandle handle = chunk.uncompressedHandle();
    if (!handle.isValid()) {
        return false;
    }

    if (pool_.pinCount(handle) > 0u) {
        return false;
    }

    const BlockMaterial* data = pool_.data(handle);
    if (data == nullptr) {
        return false;
    }

    std::vector<uint8_t> encoded = encodeRle(data);
    const CompressedChunkHandle compressedHandle = compressedStore_.store(std::move(encoded), kCodecRleV1);

    if (!pool_.release(handle)) {
        compressedStore_.release(compressedHandle);
        return false;
    }

    chunk.setCompressedHandle(compressedHandle);
    chunk.clearUncompressedStorage();
    return true;
}

bool ChunkResidencyManager::uncompressChunk(Chunk& chunk) {
    if (chunk.isPoolResident()) {
        return true;
    }

    const CompressedChunkHandle compressedHandle = chunk.compressedHandle();
    if (!compressedHandle.isValid()) {
        return ensureResident(chunk);
    }

    const std::vector<uint8_t> encoded = compressedStore_.copy(compressedHandle);
    if (encoded.empty()) {
        return false;
    }

    UncompressedChunkHandle handle = pool_.allocate();
    if (!handle.isValid()) {
        return false;
    }

    if (!chunk.attachUncompressedStorage(pool_, handle)) {
        pool_.release(handle);
        return false;
    }

    BlockMaterial* dst = pool_.data(handle);
    if (dst == nullptr || !decodeRle(encoded, dst)) {
        chunk.clearUncompressedStorage();
        pool_.release(handle);
        return false;
    }

    compressedStore_.release(compressedHandle);
    chunk.clearCompressedHandle();
    return true;
}

bool ChunkResidencyManager::pinChunk(Chunk& chunk) {
    if (!chunk.isPoolResident()) {
        return false;
    }

    return pool_.pin(chunk.uncompressedHandle());
}

bool ChunkResidencyManager::unpinChunk(Chunk& chunk) {
    if (!chunk.isPoolResident()) {
        return false;
    }

    return pool_.unpin(chunk.uncompressedHandle());
}

std::size_t ChunkResidencyManager::evictChunks(std::vector<Chunk*>& candidates, std::size_t targetFreeSlots) {
    std::size_t evicted = 0;
    for (Chunk* chunk : candidates) {
        if (chunk == nullptr) {
            continue;
        }

        if (freeSlots() >= targetFreeSlots) {
            break;
        }

        if (compressChunk(*chunk)) {
            ++evicted;
        }
    }

    return evicted;
}

std::size_t ChunkResidencyManager::freeSlots() const {
    return pool_.freeSlots();
}

std::vector<uint8_t> ChunkResidencyManager::encodeRle(const BlockMaterial* blocks) const {
    std::vector<uint8_t> bytes;
    if (blocks == nullptr) {
        return bytes;
    }

    bytes.reserve(CHUNK_BLOCKS * 2);

    std::size_t index = 0;
    while (index < CHUNK_BLOCKS) {
        const uint32_t value = blocks[index].data;
        uint32_t runLength = 1;

        while (index + runLength < CHUNK_BLOCKS && runLength < 0xFFFFFFFFu && blocks[index + runLength].data == value) {
            ++runLength;
        }

        appendU32(bytes, runLength);
        appendU32(bytes, value);
        index += runLength;
    }

    return bytes;
}

bool ChunkResidencyManager::decodeRle(const std::vector<uint8_t>& bytes, BlockMaterial* outBlocks) const {
    if (outBlocks == nullptr) {
        return false;
    }

    std::size_t outputIndex = 0;
    std::size_t inputOffset = 0;

    while (inputOffset + 8u <= bytes.size() && outputIndex < CHUNK_BLOCKS) {
        uint32_t runLength = 0;
        uint32_t value = 0;

        if (!readU32(bytes, inputOffset, runLength)) {
            return false;
        }
        if (!readU32(bytes, inputOffset + 4u, value)) {
            return false;
        }
        inputOffset += 8u;

        if (runLength == 0u) {
            return false;
        }

        const std::size_t remaining = CHUNK_BLOCKS - outputIndex;
        const std::size_t writeCount = std::min<std::size_t>(remaining, runLength);
        for (std::size_t i = 0; i < writeCount; ++i) {
            outBlocks[outputIndex + i].data = value;
        }

        outputIndex += writeCount;
    }

    return outputIndex == CHUNK_BLOCKS;
}
