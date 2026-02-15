#pragma once

#include <cstdint>
#include <glm/glm.hpp>

struct FrameUniforms {
    glm::mat4 projectionMatrix;
    glm::mat4 viewMatrix;
    glm::mat4 modelMatrix;

    glm::mat4 inverseProjectionMatrix;
    glm::mat4 inverseViewMatrix;

    // renderFlags[0] bit 0: meshlet debug view
    uint32_t renderFlags[4] = { 0u, 0u, 0u, 0u };
};

static_assert((sizeof(FrameUniforms) % 16) == 0, "FrameUniforms must remain 16-byte aligned for WGSL uniforms");
