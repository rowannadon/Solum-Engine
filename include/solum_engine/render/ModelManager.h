#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

#include "solum_engine/render/BufferManager.h"

struct ModelQuadGPU {
    glm::vec4 vertexPositions[4];
    glm::vec2 uvs[4];
    float aoValues[4];
    glm::vec4 normal;
};

static_assert(sizeof(ModelQuadGPU) == 128, "ModelQuadGPU must stay 128 bytes");

struct ModelCullInfo {
    struct DirectionInfo {
        uint32_t offset = 0u;
        uint32_t count = 0u;
    };

    std::array<DirectionInfo, 6> cullableFaces{};
    DirectionInfo nonCullableFaces{};
    uint32_t totalQuads = 0u;
};

struct ModelQuadMetadata {
    int8_t alignedFace = -1;
    uint8_t preferredFace = 0u;
    glm::vec3 minCorner{0.0f};
    glm::vec3 maxCorner{1.0f};
};

struct LoadedModel {
    std::vector<ModelQuadGPU> quads;
    std::array<std::vector<uint32_t>, 6> cullableQuadIndices{};
    std::vector<uint32_t> nonCullableQuadIndices{};
    std::vector<ModelQuadMetadata> quadMetadata{};
    std::vector<uint32_t> localToGpuQuadIndex{};
    ModelCullInfo cullInfo{};
};

class ModelManager {
public:
    static constexpr uint32_t kMaxTotalQuads = 100000u;
    static constexpr const char* kModelQuadBufferName = "model_quad_buffer";

    bool loadModel(const std::string& modelName, const std::filesystem::path& path);
    bool uploadModels(BufferManager& bufferManager);

    bool hasModel(const std::string& modelName) const;
    const LoadedModel* getModel(const std::string& modelName) const;
    uint32_t totalPackedQuads() const noexcept { return packedQuadCount_; }

    void terminate(BufferManager& bufferManager);

private:
    static int determineAlignedFace(const ModelQuadGPU& quad);
    static bool isAlignedWithPlane(const glm::vec4 positions[4], int axis, float value, float epsilon = 0.001f);

    std::unordered_map<std::string, LoadedModel> models_{};
    std::vector<std::string> modelOrder_{};
    uint32_t packedQuadCount_ = 0u;
};
