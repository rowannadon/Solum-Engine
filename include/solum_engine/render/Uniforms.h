#pragma once

#include <glm/glm.hpp>

struct FrameUniforms {
    glm::mat4 projectionMatrix;
    glm::mat4 infiniteProjectionMatrix;
    glm::mat4 viewMatrix;
    glm::mat4 modelMatrix;

	glm::mat4 inverseProjectionMatrix;
	glm::mat4 inverseViewMatrix;

    glm::mat4 lightViewMatrix;
    glm::mat4 lightProjectionMatrix;

    glm::vec3 lightDirection;
    uint32_t transparent;

    glm::ivec3 highlightedVoxelPos;
    float time;

    glm::vec3 cameraWorldPos;
    float padding1;  // For 16-byte alignment

    glm::vec3 lightPosition;
    float padding2;  // For 16-byte alignment

	glm::vec2 screenSize;
    float padding3[2];

    glm::vec3 cameraOffset;  // ADD THIS: High-precision camera offset
    float padding4;
};
