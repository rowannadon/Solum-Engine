#include "solum_engine/render/MeshUploadCapacityPolicy.h"

#include <algorithm>

MeshUploadCapacity MeshUploadCapacityPolicy::compute(
    uint32_t totalMeshletCount,
    uint32_t totalQuadWordCount,
    uint32_t requiredMeshletCapacityHint,
    uint32_t requiredQuadCapacityHint
) noexcept {
    MeshUploadCapacity capacity{};
    capacity.requiredMeshletCapacity = std::max(
        requiredMeshletCapacityHint,
        std::max(totalMeshletCount + 16u, 64u)
    );

    capacity.requiredQuadCapacity = std::max(
        requiredQuadCapacityHint,
        std::max(
            totalQuadWordCount + (1024u * MESHLET_QUAD_DATA_WORD_STRIDE),
            capacity.requiredMeshletCapacity * MESHLET_QUAD_CAPACITY * MESHLET_QUAD_DATA_WORD_STRIDE
        )
    );

    return capacity;
}
