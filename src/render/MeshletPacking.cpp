#include "solum_engine/render/MeshletPacking.h"

#include <algorithm>
#include <array>

namespace {
MeshletAabb computeMeshletAabb(const Meshlet& meshlet) {
    const float voxelScale = static_cast<float>(std::max(meshlet.voxelScale, 1u));
    const glm::vec3 meshletOrigin = glm::vec3(meshlet.origin);

    if (meshlet.hasCustomBounds) {
        return MeshletAabb{
            meshletOrigin + (meshlet.localBoundsMin * voxelScale),
            meshletOrigin + (meshlet.localBoundsMax * voxelScale)
        };
    }

    if (meshlet.quadCount == 0u) {
        return MeshletAabb{meshletOrigin, meshletOrigin};
    }

    static const std::array<std::array<glm::vec3, 4>, 6> kFaceCornerOffsets{{
        {{glm::vec3{1.0f, 0.0f, 0.0f}, glm::vec3{1.0f, 1.0f, 0.0f}, glm::vec3{1.0f, 0.0f, 1.0f}, glm::vec3{1.0f, 1.0f, 1.0f}}},
        {{glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{0.0f, 0.0f, 1.0f}, glm::vec3{0.0f, 1.0f, 0.0f}, glm::vec3{0.0f, 1.0f, 1.0f}}},
        {{glm::vec3{0.0f, 1.0f, 0.0f}, glm::vec3{0.0f, 1.0f, 1.0f}, glm::vec3{1.0f, 1.0f, 0.0f}, glm::vec3{1.0f, 1.0f, 1.0f}}},
        {{glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{1.0f, 0.0f, 0.0f}, glm::vec3{0.0f, 0.0f, 1.0f}, glm::vec3{1.0f, 0.0f, 1.0f}}},
        {{glm::vec3{0.0f, 0.0f, 1.0f}, glm::vec3{1.0f, 0.0f, 1.0f}, glm::vec3{0.0f, 1.0f, 1.0f}, glm::vec3{1.0f, 1.0f, 1.0f}}},
        {{glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{0.0f, 1.0f, 0.0f}, glm::vec3{1.0f, 0.0f, 0.0f}, glm::vec3{1.0f, 1.0f, 0.0f}}},
    }};

    const uint32_t safeFaceDirection = std::min(meshlet.faceDirection, 5u);

    bool firstVertex = true;
    glm::vec3 minCorner{0.0f};
    glm::vec3 maxCorner{0.0f};

    for (uint32_t quadIndex = 0; quadIndex < meshlet.quadCount; ++quadIndex) {
        const glm::uvec3 local = unpackMeshletLocalOffset(meshlet.packedQuadLocalOffsets[quadIndex]);
        const glm::vec3 quadBase = meshletOrigin + (glm::vec3(local) * voxelScale);
        for (const glm::vec3& cornerOffset : kFaceCornerOffsets[safeFaceDirection]) {
            const glm::vec3 vertex = quadBase + (cornerOffset * voxelScale);
            if (firstVertex) {
                minCorner = vertex;
                maxCorner = vertex;
                firstVertex = false;
                continue;
            }
            minCorner = glm::min(minCorner, vertex);
            maxCorner = glm::max(maxCorner, vertex);
        }
    }

    return MeshletAabb{minCorner, maxCorner};
}

MeshletAabbGPU toGpuAabb(const MeshletAabb& aabb) {
    return MeshletAabbGPU{
        glm::vec4(aabb.minCorner, 0.0f),
        glm::vec4(aabb.maxCorner, 0.0f)
    };
}
}  // namespace

PackedMeshletData packMeshletsForUpload(const std::vector<Meshlet>& meshlets) {
    PackedMeshletData packed;

    uint32_t totalMeshletCount = 0u;
    uint32_t totalQuadWordCount = 0u;
    for (const Meshlet& meshlet : meshlets) {
        if (meshlet.quadCount == 0u) {
            continue;
        }
        ++totalMeshletCount;
        totalQuadWordCount += meshlet.quadCount * MESHLET_QUAD_DATA_WORD_STRIDE;
    }

    packed.metadata.reserve(totalMeshletCount);
    packed.quadData.reserve(totalQuadWordCount);
    packed.aabbGpu.reserve(totalMeshletCount);
    packed.bounds.reserve(totalMeshletCount);

    for (const Meshlet& meshlet : meshlets) {
        if (meshlet.quadCount == 0u) {
            continue;
        }

        MeshletMetadataGPU metadata{};
        metadata.originX = meshlet.origin.x;
        metadata.originY = meshlet.origin.y;
        metadata.originZ = meshlet.origin.z;
        metadata.quadCount = meshlet.quadCount;
        metadata.faceDirection = meshlet.faceDirection;
        metadata.dataOffset = static_cast<uint32_t>(packed.quadData.size());
        metadata.voxelScale = std::max(meshlet.voxelScale, 1u);
        metadata.flags = meshlet.flags;
        packed.metadata.push_back(metadata);

        const MeshletAabb bounds = computeMeshletAabb(meshlet);
        packed.bounds.push_back(bounds);
        packed.aabbGpu.push_back(toGpuAabb(bounds));

        for (uint32_t i = 0; i < meshlet.quadCount; ++i) {
            packed.quadData.push_back(packMeshletQuadData(
                meshlet.packedQuadLocalOffsets[i],
                meshlet.quadMaterialIds[i]
            ));
            packed.quadData.push_back(packMeshletQuadLightData(
                meshlet.quadLightData[i]
            ));
            packed.quadData.push_back(packMeshletQuadAuxData(
                meshlet.quadAoData[i],
                meshlet.quadModelQuadIndices[i],
                meshlet.quadUsesVoxelAo[i] != 0u
            ));
        }
    }

    return packed;
}
