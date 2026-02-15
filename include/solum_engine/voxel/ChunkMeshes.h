#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <vector>

struct MeshHandle {
    uint32_t index = std::numeric_limits<uint32_t>::max();
    uint32_t generation = 0;

    bool isValid() const {
        return index != std::numeric_limits<uint32_t>::max();
    }

    static MeshHandle invalid() {
        return {};
    }
};

struct PackedVertexAttributes {
    uint32_t xy = 0;
    uint32_t zMaterial = 0;
    uint32_t packedFlags = 0;
};

struct MeshletInfo {
    glm::ivec3 origin{0};
    uint32_t lodLevel = 0;
    uint32_t quadCount = 0;
    uint32_t vertexOffset = 0;
    uint32_t indexOffset = 0;
};

struct MeshData {
    static constexpr uint32_t kMeshletQuadCapacity = 128;
    static constexpr uint32_t kMeshletVertexCapacity = kMeshletQuadCapacity * 4;
    static constexpr uint32_t kMeshletIndexCapacity = kMeshletQuadCapacity * 6;

    std::vector<PackedVertexAttributes> packedVertices;
    std::vector<uint32_t> packedIndices;
    std::vector<MeshletInfo> meshlets;
    glm::vec3 minBounds{0.0f};
    glm::vec3 maxBounds{0.0f};
    uint32_t derivedFromVersion = 0;

    bool empty() const {
        return meshlets.empty();
    }
};

class MeshHandleTable {
public:
    MeshHandle create(MeshData meshData);
    MeshHandle updateOrCreate(MeshHandle handle, MeshData meshData);
    bool release(MeshHandle handle);
    std::optional<MeshData> copy(MeshHandle handle) const;

private:
    struct Entry {
        uint32_t generation = 1;
        bool allocated = false;
        MeshData mesh;
    };

    bool isValidLocked(MeshHandle handle) const;

    mutable std::mutex mutex_;
    std::vector<Entry> entries_;
    std::vector<uint32_t> freeList_;
};
