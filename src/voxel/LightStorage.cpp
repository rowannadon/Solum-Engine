#include "solum_engine/voxel/LightStorage.h"

#include <algorithm>

namespace {
constexpr uint8_t kMaxLightLevel = 15;
}

uint8_t LightStorage::skyLight(int index) const {
    if (index < 0 || index >= CHUNK_BLOCKS) {
        return 0;
    }
    return skyLight_[static_cast<std::size_t>(index)];
}

uint8_t LightStorage::blockLight(int index) const {
    if (index < 0 || index >= CHUNK_BLOCKS) {
        return 0;
    }
    return blockLight_[static_cast<std::size_t>(index)];
}

void LightStorage::setSkyLight(int index, uint8_t value) {
    if (index < 0 || index >= CHUNK_BLOCKS) {
        return;
    }
    skyLight_[static_cast<std::size_t>(index)] = clampLight(value);
}

void LightStorage::setBlockLight(int index, uint8_t value) {
    if (index < 0 || index >= CHUNK_BLOCKS) {
        return;
    }
    blockLight_[static_cast<std::size_t>(index)] = clampLight(value);
}

uint8_t LightStorage::clampLight(uint8_t value) {
    return std::min<uint8_t>(value, kMaxLightLevel);
}
