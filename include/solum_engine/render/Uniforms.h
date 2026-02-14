#pragma once

#include <glm/glm.hpp>

struct FrameUniforms {
    glm::mat4 projectionMatrix;
    glm::mat4 viewMatrix;
    glm::mat4 modelMatrix;

	glm::mat4 inverseProjectionMatrix;
	glm::mat4 inverseViewMatrix;

    glm::uvec4 debugParams{0u, 0u, 0u, 0u};
};
