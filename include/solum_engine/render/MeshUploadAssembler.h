#pragma once

#include <vector>

#include "solum_engine/render/MeshUploadCapacityPolicy.h"
#include "solum_engine/voxel/StreamingUpload.h"

class MeshUploadAssembler {
public:
    static StreamingMeshUpload assemble(
        const std::vector<Meshlet>& meshlets,
        uint64_t meshRevision,
        const ColumnCoord& centerColumn
    );

private:
    static MeshletAabb computeMeshletAabb(const Meshlet& meshlet);
    static MeshletAabbGPU toGpuAabb(const MeshletAabb& aabb);
};
