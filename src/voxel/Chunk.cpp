#include "solum_engine/voxel/Chunk.h"

Chunk::Chunk() : bits_per_block_(0), palette_(1, BlockMaterial{}) {
    // Initializes perfectly empty: 1 palette entry (Block ID 0 = Air), 0 bits, empty data array.
}

BlockMaterial Chunk::getBlock(uint8_t x, uint8_t y, uint8_t z) const {
    if (bits_per_block_ == 0) return palette_[0];
    return palette_[getPaletteIndex(getVoxelIndex(x, y, z))];
}

void Chunk::setBlock(uint8_t x, uint8_t y, uint8_t z, const BlockMaterial blockID) {
    uint16_t voxel_index = getVoxelIndex(x, y, z);
    
    // Find blockID in palette
    auto it = std::find(palette_.begin(), palette_.end(), blockID);
    uint32_t palette_index = 0;
    
    if (it != palette_.end()) {
        palette_index = static_cast<uint32_t>(std::distance(palette_.begin(), it));
    } else {
        // Not in palette, add it
        palette_index = static_cast<uint32_t>(palette_.size());
        palette_.push_back(blockID);
        
        // Resize bit array if the palette size exceeds what the current bit width can store
        if (palette_.size() > (1ULL << bits_per_block_)) {
            uint8_t new_bpi = bits_per_block_ == 0 ? 1 : bits_per_block_ + 1;
            resizeBitArray(new_bpi);
        }
    }
    
    if (bits_per_block_ > 0) {
        setPaletteIndex(voxel_index, palette_index);
    }
}

uint32_t Chunk::getPaletteIndex(uint16_t voxel_index) const {
    size_t bit_index = voxel_index * bits_per_block_;
    size_t word_index = bit_index / 64;
    size_t bit_offset = bit_index % 64;

    if (bit_offset + bits_per_block_ <= 64) {
        // Fits within a single 64-bit word
        const uint64_t mask = (1ULL << bits_per_block_) - 1ULL;
        return static_cast<uint32_t>((data_[word_index] >> bit_offset) & mask);
    } else {
        // Crosses a 64-bit boundary
        size_t bits_in_first = 64 - bit_offset;
        size_t bits_in_next = bits_per_block_ - bits_in_first;
        
        uint32_t val1 = static_cast<uint32_t>((data_[word_index] >> bit_offset) & ((1ULL << bits_in_first) - 1ULL));
        uint32_t val2 = static_cast<uint32_t>(data_[word_index + 1] & ((1ULL << bits_in_next) - 1ULL));
        return val1 | (val2 << bits_in_first);
    }
}

void Chunk::setPaletteIndex(uint16_t voxel_index, uint32_t palette_index) {
    size_t bit_index = voxel_index * bits_per_block_;
    size_t word_index = bit_index / 64;
    size_t bit_offset = bit_index % 64;

    if (bit_offset + bits_per_block_ <= 64) {
        uint64_t mask = ((1ULL << bits_per_block_) - 1) << bit_offset;
        data_[word_index] = (data_[word_index] & ~mask) | (static_cast<uint64_t>(palette_index) << bit_offset);
    } else {
        size_t bits_in_first = 64 - bit_offset;
        size_t bits_in_next = bits_per_block_ - bits_in_first;

        uint64_t mask1 = ((1ULL << bits_in_first) - 1) << bit_offset;
        data_[word_index] = (data_[word_index] & ~mask1) | ((static_cast<uint64_t>(palette_index) & ((1ULL << bits_in_first) - 1)) << bit_offset);

        uint64_t mask2 = (1ULL << bits_in_next) - 1;
        data_[word_index + 1] = (data_[word_index + 1] & ~mask2) | (static_cast<uint64_t>(palette_index) >> bits_in_first);
    }
}

void Chunk::resizeBitArray(uint8_t new_bits_per_block) {
    // Calculate new data array size based on 4096 volume
    size_t new_size = (VOLUME * new_bits_per_block + 63) / 64;
    std::vector<uint64_t> new_data(new_size, 0);
    
    // Temporarily copy old state to unpack
    uint8_t old_bits_per_block = bits_per_block_;
    std::vector<uint64_t> old_data = std::move(data_);
    
    data_ = std::move(new_data);
    bits_per_block_ = new_bits_per_block;
    
    // Repack if we had previous data
    if (old_bits_per_block > 0) {
        for (uint16_t i = 0; i < VOLUME; ++i) {
            // Re-implementing a manual inline extraction from the old array here 
            // to bypass the stateful `getPaletteIndex` dependency
            size_t b_idx = i * old_bits_per_block;
            size_t w_idx = b_idx / 64;
            size_t b_off = b_idx % 64;
            uint32_t p_idx = 0;
            
            if (b_off + old_bits_per_block <= 64) {
                p_idx = static_cast<uint32_t>((old_data[w_idx] >> b_off) & ((1ULL << old_bits_per_block) - 1ULL));
            } else {
                size_t first = 64 - b_off;
                size_t next = old_bits_per_block - first;
                uint32_t v1 = static_cast<uint32_t>((old_data[w_idx] >> b_off) & ((1ULL << first) - 1ULL));
                uint32_t v2 = static_cast<uint32_t>(old_data[w_idx + 1] & ((1ULL << next) - 1ULL));
                p_idx = v1 | (v2 << first);
            }
            setPaletteIndex(i, p_idx);
        }
    }
}
