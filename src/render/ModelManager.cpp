#include "solum_engine/render/ModelManager.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <iostream>
#include <limits>
#include <utility>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader/tiny_obj_loader.h>

namespace {

glm::vec3 computeFaceNormal(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) {
    glm::vec3 n = glm::cross(b - a, c - a);
    const float len = glm::length(n);
    if (len > 1e-8f) {
        n /= len;
    }
    return n;
}

uint8_t preferredFaceFromNormal(const glm::vec3& normal) {
    static const std::array<glm::vec3, 6> kDirections{{
        glm::vec3(1.0f, 0.0f, 0.0f),   // +X
        glm::vec3(-1.0f, 0.0f, 0.0f),  // -X
        glm::vec3(0.0f, 1.0f, 0.0f),   // +Y
        glm::vec3(0.0f, -1.0f, 0.0f),  // -Y
        glm::vec3(0.0f, 0.0f, 1.0f),   // +Z
        glm::vec3(0.0f, 0.0f, -1.0f),  // -Z
    }};

    float maxDot = -std::numeric_limits<float>::infinity();
    uint8_t best = 4u;
    for (size_t i = 0u; i < kDirections.size(); ++i) {
        const float d = glm::dot(normal, kDirections[i]);
        if (d > maxDot) {
            maxDot = d;
            best = static_cast<uint8_t>(i);
        }
    }
    return best;
}

}  // namespace

bool ModelManager::loadModel(const std::string& modelName, const std::filesystem::path& path) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;

    const bool loaded = tinyobj::LoadObj(
        &attrib,
        &shapes,
        &materials,
        &warn,
        &err,
        path.string().c_str(),
        path.parent_path().string().c_str(),
        false
    );

    if (!warn.empty()) {
        std::cerr << "ModelManager: tinyobj warning for '" << path.string() << "': " << warn << std::endl;
    }
    if (!loaded) {
        std::cerr << "ModelManager: tinyobj failed for '" << path.string() << "': " << err << std::endl;
        return false;
    }

    LoadedModel model{};
    model.quads.reserve(256u);
    model.quadMetadata.reserve(256u);
    for (auto& perFace : model.cullableQuadIndices) {
        perFace.clear();
        perFace.reserve(64u);
    }
    model.nonCullableQuadIndices.reserve(128u);

    for (const tinyobj::shape_t& shape : shapes) {
        size_t indexOffset = 0u;
        for (size_t face = 0u; face < shape.mesh.num_face_vertices.size(); ++face) {
            const int faceVertexCount = shape.mesh.num_face_vertices[face];
            if (faceVertexCount != 3 && faceVertexCount != 4) {
                indexOffset += static_cast<size_t>(std::max(faceVertexCount, 0));
                continue;
            }

            std::array<glm::vec3, 4> positions{glm::vec3(0.0f)};
            std::array<glm::vec2, 4> uvs{glm::vec2(0.0f)};

            for (int v = 0; v < faceVertexCount; ++v) {
                const tinyobj::index_t idx = shape.mesh.indices[indexOffset + static_cast<size_t>(v)];
                if (idx.vertex_index >= 0 &&
                    (static_cast<size_t>(idx.vertex_index) * 3u + 2u) < attrib.vertices.size()) {
                    positions[v] = glm::vec3(
                        attrib.vertices[3 * idx.vertex_index + 0],
                        attrib.vertices[3 * idx.vertex_index + 1],
                        attrib.vertices[3 * idx.vertex_index + 2]
                    );
                }

                if (idx.texcoord_index >= 0 &&
                    (static_cast<size_t>(idx.texcoord_index) * 2u + 1u) < attrib.texcoords.size()) {
                    uvs[v] = glm::vec2(
                        attrib.texcoords[2 * idx.texcoord_index + 0],
                        1.0f - attrib.texcoords[2 * idx.texcoord_index + 1]
                    );
                }
            }
            indexOffset += static_cast<size_t>(faceVertexCount);

            if (faceVertexCount == 3) {
                positions[3] = positions[2];
                uvs[3] = uvs[2];
            }

            // Canonicalize to the corner ordering expected by the shader's
            // fixed quad expansion indices (diagonal between corners 1 and 2).
            std::swap(positions[2], positions[3]);
            std::swap(uvs[2], uvs[3]);

            ModelQuadGPU quad{};
            glm::vec3 minCorner = positions[0];
            glm::vec3 maxCorner = positions[0];
            for (int i = 0; i < 4; ++i) {
                quad.vertexPositions[i] = glm::vec4(positions[i], 1.0f);
                quad.uvs[i] = uvs[i];
                quad.aoValues[i] = 1.0f;
                minCorner = glm::min(minCorner, positions[i]);
                maxCorner = glm::max(maxCorner, positions[i]);
            }

            const glm::vec3 normal = computeFaceNormal(positions[0], positions[1], positions[2]);
            quad.normal = glm::vec4(normal, 0.0f);

            const uint32_t quadIndex = static_cast<uint32_t>(model.quads.size());
            model.quads.push_back(quad);

            ModelQuadMetadata metadata{};
            metadata.alignedFace = static_cast<int8_t>(determineAlignedFace(quad));
            metadata.preferredFace = preferredFaceFromNormal(normal);
            metadata.minCorner = minCorner;
            metadata.maxCorner = maxCorner;
            model.quadMetadata.push_back(metadata);

            if (metadata.alignedFace >= 0 && metadata.alignedFace < 6) {
                model.cullableQuadIndices[static_cast<size_t>(metadata.alignedFace)].push_back(quadIndex);
            } else {
                model.nonCullableQuadIndices.push_back(quadIndex);
            }
        }
    }

    model.cullInfo.totalQuads = static_cast<uint32_t>(model.quads.size());
    if (models_.find(modelName) == models_.end()) {
        modelOrder_.push_back(modelName);
    }
    models_[modelName] = std::move(model);
    return true;
}

bool ModelManager::uploadModels(BufferManager& bufferManager) {
    uint32_t totalQuads = 0u;
    for (const auto& [_, model] : models_) {
        totalQuads += static_cast<uint32_t>(model.quads.size());
    }

    if (totalQuads > kMaxTotalQuads) {
        std::cerr << "ModelManager: total model quads (" << totalQuads
                  << ") exceed cap (" << kMaxTotalQuads << ")." << std::endl;
        return false;
    }

    std::vector<ModelQuadGPU> packed;
    packed.reserve(std::max(totalQuads, 1u));
    if (totalQuads == 0u) {
        packed.push_back(ModelQuadGPU{});
    }

    uint32_t currentOffset = 0u;
    for (const std::string& modelName : modelOrder_) {
        auto it = models_.find(modelName);
        if (it == models_.end()) {
            continue;
        }

        LoadedModel& model = it->second;
        model.localToGpuQuadIndex.assign(model.quads.size(), std::numeric_limits<uint32_t>::max());
        model.cullInfo = ModelCullInfo{};
        model.cullInfo.totalQuads = static_cast<uint32_t>(model.quads.size());

        for (uint32_t face = 0u; face < 6u; ++face) {
            model.cullInfo.cullableFaces[face].offset = currentOffset;
            model.cullInfo.cullableFaces[face].count = static_cast<uint32_t>(model.cullableQuadIndices[face].size());

            for (uint32_t localQuadIndex : model.cullableQuadIndices[face]) {
                if (localQuadIndex >= model.quads.size()) {
                    continue;
                }
                model.localToGpuQuadIndex[localQuadIndex] = currentOffset;
                packed.push_back(model.quads[localQuadIndex]);
                ++currentOffset;
            }
        }

        model.cullInfo.nonCullableFaces.offset = currentOffset;
        model.cullInfo.nonCullableFaces.count = static_cast<uint32_t>(model.nonCullableQuadIndices.size());
        for (uint32_t localQuadIndex : model.nonCullableQuadIndices) {
            if (localQuadIndex >= model.quads.size()) {
                continue;
            }
            model.localToGpuQuadIndex[localQuadIndex] = currentOffset;
            packed.push_back(model.quads[localQuadIndex]);
            ++currentOffset;
        }
    }

    wgpu::BufferDescriptor modelBufferDesc = wgpu::Default;
    modelBufferDesc.label = wgpu::StringView("model quad buffer");
    modelBufferDesc.size = static_cast<uint64_t>(packed.size()) * sizeof(ModelQuadGPU);
    modelBufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    modelBufferDesc.mappedAtCreation = false;

    if (!bufferManager.createBuffer(kModelQuadBufferName, modelBufferDesc)) {
        std::cerr << "ModelManager: failed to create model quad GPU buffer." << std::endl;
        return false;
    }

    if (!packed.empty()) {
        bufferManager.writeBuffer(kModelQuadBufferName, 0u, packed.data(), packed.size() * sizeof(ModelQuadGPU));
    }

    packedQuadCount_ = currentOffset;
    return true;
}

bool ModelManager::hasModel(const std::string& modelName) const {
    return models_.find(modelName) != models_.end();
}

const LoadedModel* ModelManager::getModel(const std::string& modelName) const {
    const auto it = models_.find(modelName);
    if (it == models_.end()) {
        return nullptr;
    }
    return &it->second;
}

void ModelManager::terminate(BufferManager& bufferManager) {
    bufferManager.deleteBuffer(kModelQuadBufferName);
    models_.clear();
    modelOrder_.clear();
    packedQuadCount_ = 0u;
}

int ModelManager::determineAlignedFace(const ModelQuadGPU& quad) {
    constexpr float epsilon = 0.001f;

    if (isAlignedWithPlane(quad.vertexPositions, 0, 1.0f, epsilon)) {
        return 0;
    }
    if (isAlignedWithPlane(quad.vertexPositions, 0, 0.0f, epsilon)) {
        return 1;
    }
    if (isAlignedWithPlane(quad.vertexPositions, 1, 1.0f, epsilon)) {
        return 2;
    }
    if (isAlignedWithPlane(quad.vertexPositions, 1, 0.0f, epsilon)) {
        return 3;
    }
    if (isAlignedWithPlane(quad.vertexPositions, 2, 1.0f, epsilon)) {
        return 4;
    }
    if (isAlignedWithPlane(quad.vertexPositions, 2, 0.0f, epsilon)) {
        return 5;
    }

    return -1;
}

bool ModelManager::isAlignedWithPlane(const glm::vec4 positions[4], int axis, float value, float epsilon) {
    for (int i = 0; i < 4; ++i) {
        const float coord = (axis == 0) ? positions[i].x : (axis == 1) ? positions[i].y : positions[i].z;
        if (std::abs(coord - value) > epsilon) {
            return false;
        }
    }
    return true;
}
