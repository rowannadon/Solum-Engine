#pragma once

#include <array>
#include <cstdint>
#include <glm/glm.hpp>

static constexpr uint32_t MESHLET_QUAD_CAPACITY = 128;
static constexpr uint32_t MESHLET_VERTEX_CAPACITY = MESHLET_QUAD_CAPACITY * 6;

inline uint16_t packMeshletLocalOffset(uint32_t x, uint32_t y, uint32_t z) {
    return static_cast<uint16_t>((x & 0x1Fu) | ((y & 0x1Fu) << 5u) | ((z & 0x1Fu) << 10u));
}

inline uint32_t packMeshletQuadData(uint16_t packedLocalOffset, uint16_t materialId) {
    return static_cast<uint32_t>(packedLocalOffset) | (static_cast<uint32_t>(materialId) << 16u);
}

struct Meshlet {
    glm::ivec3 origin{ 0, 0, 0 };
    uint32_t faceDirection = 0;
    uint32_t quadCount = 0;
    uint32_t voxelScale = 1;
    std::array<uint16_t, MESHLET_QUAD_CAPACITY> packedQuadLocalOffsets{};
    std::array<uint16_t, MESHLET_QUAD_CAPACITY> quadMaterialIds{};
};

struct MeshletMetadataGPU {
    int32_t originX = 0;
    int32_t originY = 0;
    int32_t originZ = 0;
    uint32_t quadCount = 0;
    uint32_t faceDirection = 0;
    uint32_t dataOffset = 0;
    uint32_t voxelScale = 1;
    uint32_t pad1 = 0;
};

static_assert(sizeof(MeshletMetadataGPU) == 32, "Meshlet metadata layout must match shader");
