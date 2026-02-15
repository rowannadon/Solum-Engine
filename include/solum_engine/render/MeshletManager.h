#pragma once

#include "solum_engine/render/BufferManager.h"
#include "solum_engine/render/MeshletTypes.h"

#include <cstdint>
#include <vector>

struct MeshletGroupHandle {
    uint32_t firstMeshlet = 0;
    uint32_t meshletCount = 0;
    uint32_t firstQuad = 0;
    uint32_t quadCount = 0;
};

class MeshletManager {
public:
    static constexpr const char* kMeshDataBufferName = "meshlet_data_buffer";
    static constexpr const char* kMeshMetadataBufferName = "meshlet_metadata_buffer";

    bool initialize(BufferManager* bufferManager, uint32_t maxMeshlets, uint32_t maxQuads);

    void clear();

    MeshletGroupHandle registerMeshletGroup(const std::vector<Meshlet>& meshlets);

    bool upload();

    uint32_t getMeshletCount() const;
    uint32_t getQuadCount() const;
    uint32_t getVerticesPerMeshlet() const;

private:
    BufferManager* bufferManager = nullptr;
    uint32_t meshletCapacity = 0;
    uint32_t quadCapacity = 0;

    std::vector<MeshletMetadataGPU> metadataCpu;
    std::vector<uint16_t> quadDataCpu;
};
