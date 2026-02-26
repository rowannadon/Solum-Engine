#pragma once

#include <cstdint>
#include <glm/glm.hpp>

inline constexpr uint32_t kRenderFlagMeshletDebug = 1u << 0;
inline constexpr uint32_t kRenderFlagBoundsDebug = 1u << 1;
inline constexpr uint32_t kRenderFlagBoundsChunks = 1u << 2;
inline constexpr uint32_t kRenderFlagBoundsColumns = 1u << 3;
inline constexpr uint32_t kRenderFlagBoundsRegions = 1u << 4;
inline constexpr uint32_t kRenderFlagBoundsMeshlets = 1u << 5;
inline constexpr uint32_t kRenderFlagBoundsLayerMask =
    kRenderFlagBoundsChunks |
    kRenderFlagBoundsColumns |
    kRenderFlagBoundsRegions |
    kRenderFlagBoundsMeshlets;

struct FrameUniforms {
    glm::mat4 projectionMatrix;
    glm::mat4 viewMatrix;
    glm::mat4 modelMatrix;

    glm::mat4 inverseProjectionMatrix;
    glm::mat4 inverseViewMatrix;

    // renderFlags[0] bit 0: meshlet debug view
    // renderFlags[0] bit 1: bounds debug master view
    // renderFlags[0] bit 2: bounds chunks layer
    // renderFlags[0] bit 3: bounds columns layer
    // renderFlags[0] bit 4: bounds regions layer
    // renderFlags[0] bit 5: bounds meshlets layer
    uint32_t renderFlags[4] = { 0u, 0u, 0u, 0u };

    // occlusionParams[0]: enabled (0.0 disabled, 1.0 enabled)
    // occlusionParams[1]: depth bias in [0, 1]
    // occlusionParams[2]: near-distance occlusion skip (world units)
    // occlusionParams[3]: minimum projected AABB span (pixels) before occlusion tests
    float occlusionParams[4] = { 1.0f, 0.01f, 20.0f, 1.0f };
};

static_assert((sizeof(FrameUniforms) % 16) == 0, "FrameUniforms must remain 16-byte aligned for WGSL uniforms");
