#pragma once

#include <cstdint>
#include <vector>

#include "solum_engine/render/MeshletTypes.h"

struct PackedMeshletData {
    std::vector<MeshletMetadataGPU> metadata;
    std::vector<uint32_t> quadData;
    std::vector<MeshletAabbGPU> aabbGpu;
    std::vector<MeshletAabb> bounds;
};

PackedMeshletData packMeshletsForUpload(const std::vector<Meshlet>& meshlets);
