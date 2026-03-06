#pragma once

#include "solum_engine/voxel/Chunk.h"
#include "solum_engine/resources/Coords.h"
#include "solum_engine/render/MeshletTypes.h"
#include "solum_engine/voxel/BlockModelLibrary.h"
#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

class IBlockSource {
public:
    virtual ~IBlockSource() = default;
    virtual BlockMaterial getBlock(const BlockCoord& coord) const = 0;
};

struct ChunkMeshOutput {
    std::vector<Meshlet> culledMeshlets;
    std::vector<Meshlet> doubleSidedMeshlets;

    bool empty() const {
        return culledMeshlets.empty() && doubleSidedMeshlets.empty();
    }
};

class ChunkMesher {
public:
    static constexpr uint16_t kCulledSolidBlockId = 255u;

    explicit ChunkMesher(std::shared_ptr<const BlockModelLibrary> blockModelLibrary = {})
        : blockModelLibrary_(std::move(blockModelLibrary)) {}

    ChunkMeshOutput mesh(const Chunk& chunk, const ChunkCoord& coord, const std::vector<const Chunk*>& neighbors) const;
    ChunkMeshOutput mesh(const IBlockSource& source,
                         const BlockCoord& sectionOrigin,
                         const glm::ivec3& sectionExtent,
                         const glm::ivec3& meshletOrigin,
                         uint32_t voxelScale = 1u) const;

    static constexpr std::array<glm::ivec3, 6> directionOffsets = {
        glm::ivec3(1, 0, 0),   // PlusX
        glm::ivec3(-1, 0, 0),  // MinusX
        glm::ivec3(0, 1, 0),   // PlusY
        glm::ivec3(0, -1, 0),  // MinusY
        glm::ivec3(0, 0, 1),   // PlusZ (up in z-up world)
        glm::ivec3(0, 0, -1),  // MinusZ (down in z-up world)
    };

private:
    std::shared_ptr<const BlockModelLibrary> blockModelLibrary_;
};
