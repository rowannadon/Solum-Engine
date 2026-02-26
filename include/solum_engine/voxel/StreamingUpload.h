#pragma once

#include <cstdint>
#include <vector>

#include "solum_engine/render/MeshletTypes.h"
#include "solum_engine/resources/Coords.h"

struct StreamingMeshUpload {
    std::vector<MeshletMetadataGPU> metadata;
    std::vector<uint32_t> quadData;
    std::vector<MeshletAabbGPU> meshletAabbsGpu;
    std::vector<MeshletAabb> meshletBounds;
    uint32_t totalMeshletCount = 0;
    uint32_t totalQuadCount = 0;
    uint32_t requiredMeshletCapacity = 0;
    uint32_t requiredQuadCapacity = 0;
    uint64_t meshRevision = 0;
    ColumnCoord centerColumn{0, 0};
};
