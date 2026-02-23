#pragma once

#include <cstdint>
#include <glm/glm.hpp>

inline constexpr uint32_t kRenderFlagMeshletDebug = 1u << 0;
inline constexpr uint32_t kRenderFlagBoundsDebug = 1u << 1;
inline constexpr uint32_t kRenderFlagBoundsChunks = 1u << 2;
inline constexpr uint32_t kRenderFlagBoundsColumns = 1u << 3;
inline constexpr uint32_t kRenderFlagBoundsRegions = 1u << 4;
inline constexpr uint32_t kRenderFlagBoundsLayerMask =
    kRenderFlagBoundsChunks |
    kRenderFlagBoundsColumns |
    kRenderFlagBoundsRegions;

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
    uint32_t renderFlags[4] = { 0u, 0u, 0u, 0u };
};

static_assert((sizeof(FrameUniforms) % 16) == 0, "FrameUniforms must remain 16-byte aligned for WGSL uniforms");
