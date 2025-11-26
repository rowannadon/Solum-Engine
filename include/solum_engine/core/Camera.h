#pragma once
#include <glm/glm.hpp>


struct FirstPersonCamera {
    glm::vec3 position = glm::vec3(5.0f, 0.0f, 200.0f);  // Camera position in world space
    glm::vec3 front = glm::vec3(-1.0f, 0.0f, 0.0f);    // Direction camera is looking
    glm::vec3 up = glm::vec3(0.0f, 0.0f, 1.0f);        // Up vector
    glm::vec3 right = glm::vec3(0.0f, 1.0f, 0.0f);     // Right vector (corrected)
    glm::vec3 worldUp = glm::vec3(0.0f, 0.0f, 1.0f);   // World up vector

    // Euler angles
    float yaw = 180.0f;  // Rotation around Z axis (left/right) - corrected initial value
    float pitch = 0.0f;  // Rotation around X axis (up/down)

    // Camera options
    float movementSpeed = 80.0f;
    float mouseSensitivity = 0.1f;
    float zoom = 85.f;

    glm::vec3 velocity = glm::vec3(0.0f);  // Current velocity vector
    glm::vec3 acceleration = glm::vec3(0.0f);  // Current acceleration vector

    void updateCameraVectors() {
        // Calculate the new front vector for Z+ up coordinate system
        glm::vec3 newFront;
        newFront.x = cos(glm::radians(pitch)) * cos(glm::radians(yaw));
        newFront.y = cos(glm::radians(pitch)) * sin(glm::radians(-yaw));
        newFront.z = sin(glm::radians(pitch));
        front = glm::normalize(newFront);

        // Re-calculate the right and up vector
        right = glm::normalize(glm::cross(front, worldUp));
        up = glm::normalize(glm::cross(right, front));
    }
};