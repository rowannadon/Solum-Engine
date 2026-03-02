#pragma once

#include <cstdint>

#include "solum_engine/render/MeshletTypes.h"

struct MeshUploadCapacity {
    uint32_t requiredMeshletCapacity = 0;
    uint32_t requiredQuadCapacity = 0;
};

class MeshUploadCapacityPolicy {
public:
    static MeshUploadCapacity compute(
        uint32_t totalMeshletCount,
        uint32_t totalQuadWordCount,
        uint32_t requiredMeshletCapacityHint = 0u,
        uint32_t requiredQuadCapacityHint = 0u
    ) noexcept;
};
