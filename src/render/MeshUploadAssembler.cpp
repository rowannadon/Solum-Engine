#include "solum_engine/render/MeshUploadAssembler.h"

#include <algorithm>
#include <array>

MeshletAabb MeshUploadAssembler::computeMeshletAabb(const Meshlet& meshlet) {
    if (meshlet.quadCount == 0u) {
        const glm::vec3 origin = glm::vec3(meshlet.origin);
        return MeshletAabb{origin, origin};
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
    const float voxelScale = static_cast<float>(std::max(meshlet.voxelScale, 1u));

    bool firstVertex = true;
    glm::vec3 minCorner{0.0f};
    glm::vec3 maxCorner{0.0f};

    for (uint32_t quadIndex = 0; quadIndex < meshlet.quadCount; ++quadIndex) {
        const glm::uvec3 local = unpackMeshletLocalOffset(meshlet.packedQuadLocalOffsets[quadIndex]);
        const glm::vec3 quadBase = glm::vec3(meshlet.origin) + (glm::vec3(local) * voxelScale);
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

MeshletAabbGPU MeshUploadAssembler::toGpuAabb(const MeshletAabb& aabb) {
    return MeshletAabbGPU{
        glm::vec4(aabb.minCorner, 0.0f),
        glm::vec4(aabb.maxCorner, 0.0f)
    };
}

StreamingMeshUpload MeshUploadAssembler::assemble(
    const std::vector<Meshlet>& meshlets,
    uint64_t meshRevision,
    const ColumnCoord& centerColumn
) {
    uint32_t totalMeshletCount = 0u;
    uint32_t totalQuadWordCount = 0u;
    for (const Meshlet& meshlet : meshlets) {
        if (meshlet.quadCount == 0u) {
            continue;
        }
        ++totalMeshletCount;
        totalQuadWordCount += meshlet.quadCount * MESHLET_QUAD_DATA_WORD_STRIDE;
    }

    StreamingMeshUpload upload;
    upload.totalMeshletCount = totalMeshletCount;
    upload.totalQuadCount = totalQuadWordCount;
    upload.meshRevision = meshRevision;
    upload.centerColumn = centerColumn;

    upload.metadata.reserve(totalMeshletCount);
    upload.quadData.reserve(totalQuadWordCount);
    upload.meshletAabbsGpu.reserve(totalMeshletCount);
    upload.meshletBounds.reserve(totalMeshletCount);

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
        metadata.dataOffset = static_cast<uint32_t>(upload.quadData.size());
        metadata.voxelScale = std::max(meshlet.voxelScale, 1u);
        upload.metadata.push_back(metadata);

        const MeshletAabb meshletBounds = computeMeshletAabb(meshlet);
        upload.meshletAabbsGpu.push_back(toGpuAabb(meshletBounds));
        upload.meshletBounds.push_back(meshletBounds);

        for (uint32_t i = 0; i < meshlet.quadCount; ++i) {
            upload.quadData.push_back(packMeshletQuadData(
                meshlet.packedQuadLocalOffsets[i],
                meshlet.quadMaterialIds[i]
            ));
            upload.quadData.push_back(static_cast<uint32_t>(meshlet.quadAoData[i]));
        }
    }

    const MeshUploadCapacity capacity = MeshUploadCapacityPolicy::compute(
        totalMeshletCount,
        totalQuadWordCount
    );
    upload.requiredMeshletCapacity = capacity.requiredMeshletCapacity;
    upload.requiredQuadCapacity = capacity.requiredQuadCapacity;

    return upload;
}
