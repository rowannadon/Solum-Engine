#pragma once

#include "solum_engine/render/VertexAttributes.h"

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

struct MeshData {
    std::vector<VertexAttributes> vertices;
    std::vector<uint16_t> indices;
    glm::vec3 minBounds{0.0f};
    glm::vec3 maxBounds{0.0f};
    uint32_t derivedFromVersion = 0;

    bool empty() const {
        return vertices.empty() || indices.empty();
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
