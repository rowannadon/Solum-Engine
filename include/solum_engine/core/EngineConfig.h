#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

#include <glm/glm.hpp>

struct EngineConfig {
    struct JobConfig {
        std::size_t workerThreads = 0u;
    };

    struct WindowConfig {
        int width = 1280;
        int height = 720;
        std::string title = "Voxel Engine";
        std::string presentMode = "fifo";
    };

    struct CameraConfig {
        glm::vec3 initialPosition{0.0f, 0.0f, 175.0f};
        float yaw = 180.0f;
        float pitch = 0.0f;
        float movementSpeed = 80.0f;
        float mouseSensitivity = 0.1f;
        float fieldOfViewDegrees = 85.0f;
        float nearClip = 0.1f;
        float farClip = 2500.0f;
    };

    struct FramePacingConfig {
        bool useMonitorRefreshRate = true;
        float targetFps = 60.0f;
    };

    struct TimeConfig {
        float dayDurationSeconds = 300.0f;
        float initialTimeHours = 12.0f;
        float timeMultiplier = 0.5f;
        bool pauseTime = false;
        bool useManualTime = false;
        float manualTimeHours = 12.0f;
    };

    struct DebugConfig {
        bool enableMeshletDebug = false;
        bool enableBoundsDebug = false;
        bool showChunkBounds = true;
        bool showColumnBounds = true;
        bool showRegionBounds = true;
        bool showMeshletBounds = false;
        bool freezeCullingCamera = false;
    };

    struct OcclusionConfig {
        bool enabled = true;
        float depthBias = 0.01f;
        float nearSkipDistance = 20.0f;
        float minProjectedSpanPixels = 1.0f;
    };

    struct InteractionConfig {
        float blockReach = 32.0f;
    };

    struct WorldConfig {
        int32_t columnLoadRadius = 32;
        std::size_t maxInFlightColumnJobs = 0u;
        JobConfig jobConfig{2u};
    };

    struct MeshConfig {
        int32_t lodLevelCount = 4;
        int32_t meshTileSizeChunks = 4;
        int32_t meshTileHeightChunks = 4;
        int32_t activeChunkRadius = 32;
        float lodSseTargetPixels = 16.0f;
        float lodSseHysteresisPixels = 0.25f;
        float lodSseMinDepthBlocks = 4.0f;
        float lodSseFallbackProjectionScale = 390.0f;
        JobConfig jobConfig{2u};
    };

    struct MeshletBufferConfig {
        std::size_t uploadBudgetBytesPerFrame = 0u;
        uint32_t maxActiveMeshlets = 0u;
    };

    struct StreamingConfig {
        std::size_t maxDeltaEntriesPerTick = 64u;
    };

    WindowConfig windowConfig{};
    CameraConfig cameraConfig{};
    FramePacingConfig framePacingConfig{};
    TimeConfig timeConfig{};
    DebugConfig debugConfig{};
    OcclusionConfig occlusionConfig{};
    InteractionConfig interactionConfig{};
    WorldConfig worldConfig{};
    MeshConfig meshConfig{};
    MeshletBufferConfig meshletBufferConfig{};
    StreamingConfig streamingConfig{};

    static std::filesystem::path defaultPath();
    static EngineConfig loadDefault();
    static EngineConfig loadFromFile(const std::filesystem::path& path);

private:
    void sanitize();
};
