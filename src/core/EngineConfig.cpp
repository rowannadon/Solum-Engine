#include "solum_engine/core/EngineConfig.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include "nlohmann_json/json.hpp"

using json = nlohmann::json;

namespace {

#ifdef __APPLE__
constexpr std::size_t kDefaultUploadBudgetBytesPerFrame = 16u * 1024u * 1024u;
constexpr uint32_t kDefaultMaxActiveMeshlets = 1'000'000u;
#else
constexpr std::size_t kDefaultUploadBudgetBytesPerFrame = 128u * 1024u * 1024u;
constexpr uint32_t kDefaultMaxActiveMeshlets = UINT32_MAX;
#endif

const json* findMember(const json& object, const char* key) {
    if (!object.is_object()) {
        return nullptr;
    }
    const auto it = object.find(key);
    if (it == object.end()) {
        return nullptr;
    }
    return &(*it);
}

void readBool(const json& object, const char* key, bool& value) {
    const json* member = findMember(object, key);
    if (member != nullptr && member->is_boolean()) {
        value = member->get<bool>();
    }
}

void readString(const json& object, const char* key, std::string& value) {
    const json* member = findMember(object, key);
    if (member != nullptr && member->is_string()) {
        value = member->get<std::string>();
    }
}

template <typename T>
void readNumber(const json& object, const char* key, T& value) {
    const json* member = findMember(object, key);
    if (member != nullptr && member->is_number()) {
        try {
            value = member->get<T>();
        } catch (const std::exception&) {
        }
    }
}

void readVec3(const json& object, const char* key, glm::vec3& value) {
    const json* member = findMember(object, key);
    if (member == nullptr || !member->is_array() || member->size() != 3u) {
        return;
    }
    if (!(*member)[0].is_number() || !(*member)[1].is_number() || !(*member)[2].is_number()) {
        return;
    }
    value.x = (*member)[0].get<float>();
    value.y = (*member)[1].get<float>();
    value.z = (*member)[2].get<float>();
}

float wrapHours(float hours) {
    if (!std::isfinite(hours)) {
        return 12.0f;
    }
    float wrapped = std::fmod(hours, 24.0f);
    if (wrapped < 0.0f) {
        wrapped += 24.0f;
    }
    return wrapped;
}

std::string toLowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        if (c >= 'A' && c <= 'Z') {
            return static_cast<char>(c - 'A' + 'a');
        }
        return static_cast<char>(c);
    });
    return value;
}

void loadJobConfig(const json& object, EngineConfig::JobConfig& config) {
    const json* jobConfig = findMember(object, "jobConfig");
    if (jobConfig == nullptr || !jobConfig->is_object()) {
        return;
    }
    readNumber(*jobConfig, "workerThreads", config.workerThreads);
}

}  // namespace

std::filesystem::path EngineConfig::defaultPath() {
    const std::filesystem::path resourceDir = std::filesystem::path(RESOURCE_DIR);
    std::filesystem::path baseDir = resourceDir.parent_path();
    if (baseDir.empty()) {
        baseDir = std::filesystem::current_path();
    }
    return baseDir / "config.json";
}

EngineConfig EngineConfig::loadDefault() {
    return loadFromFile(defaultPath());
}

EngineConfig EngineConfig::loadFromFile(const std::filesystem::path& path) {
    EngineConfig config;
    config.meshletBufferConfig.uploadBudgetBytesPerFrame = kDefaultUploadBudgetBytesPerFrame;
    config.meshletBufferConfig.maxActiveMeshlets = kDefaultMaxActiveMeshlets;

    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "EngineConfig: unable to open '" << path.string()
                  << "'. Using built-in defaults." << std::endl;
        config.sanitize();
        return config;
    }

    json root;
    try {
        file >> root;
    } catch (const std::exception& e) {
        std::cerr << "EngineConfig: failed to parse '" << path.string()
                  << "': " << e.what() << ". Using built-in defaults." << std::endl;
        config.sanitize();
        return config;
    }

    if (!root.is_object()) {
        std::cerr << "EngineConfig: '" << path.string()
                  << "' must contain a JSON object. Using built-in defaults." << std::endl;
        config.sanitize();
        return config;
    }

    if (const json* section = findMember(root, "windowConfig"); section != nullptr && section->is_object()) {
        readNumber(*section, "width", config.windowConfig.width);
        readNumber(*section, "height", config.windowConfig.height);
        readString(*section, "title", config.windowConfig.title);
        readString(*section, "presentMode", config.windowConfig.presentMode);
    }

    if (const json* section = findMember(root, "cameraConfig"); section != nullptr && section->is_object()) {
        readVec3(*section, "initialPosition", config.cameraConfig.initialPosition);
        readNumber(*section, "yaw", config.cameraConfig.yaw);
        readNumber(*section, "pitch", config.cameraConfig.pitch);
        readNumber(*section, "movementSpeed", config.cameraConfig.movementSpeed);
        readNumber(*section, "mouseSensitivity", config.cameraConfig.mouseSensitivity);
        readNumber(*section, "fieldOfViewDegrees", config.cameraConfig.fieldOfViewDegrees);
        readNumber(*section, "nearClip", config.cameraConfig.nearClip);
        readNumber(*section, "farClip", config.cameraConfig.farClip);
    }

    if (const json* section = findMember(root, "framePacingConfig"); section != nullptr && section->is_object()) {
        readBool(*section, "useMonitorRefreshRate", config.framePacingConfig.useMonitorRefreshRate);
        readNumber(*section, "targetFps", config.framePacingConfig.targetFps);
    }

    if (const json* section = findMember(root, "timeConfig"); section != nullptr && section->is_object()) {
        readNumber(*section, "dayDurationSeconds", config.timeConfig.dayDurationSeconds);
        readNumber(*section, "initialTimeHours", config.timeConfig.initialTimeHours);
        readNumber(*section, "timeMultiplier", config.timeConfig.timeMultiplier);
        readBool(*section, "pauseTime", config.timeConfig.pauseTime);
        readBool(*section, "useManualTime", config.timeConfig.useManualTime);
        readNumber(*section, "manualTimeHours", config.timeConfig.manualTimeHours);
    }

    if (const json* section = findMember(root, "debugConfig"); section != nullptr && section->is_object()) {
        readBool(*section, "enableMeshletDebug", config.debugConfig.enableMeshletDebug);
        readBool(*section, "enableBoundsDebug", config.debugConfig.enableBoundsDebug);
        readBool(*section, "showChunkBounds", config.debugConfig.showChunkBounds);
        readBool(*section, "showColumnBounds", config.debugConfig.showColumnBounds);
        readBool(*section, "showRegionBounds", config.debugConfig.showRegionBounds);
        readBool(*section, "showMeshletBounds", config.debugConfig.showMeshletBounds);
        readBool(*section, "freezeCullingCamera", config.debugConfig.freezeCullingCamera);
    }

    if (const json* section = findMember(root, "occlusionConfig"); section != nullptr && section->is_object()) {
        readBool(*section, "enabled", config.occlusionConfig.enabled);
        readNumber(*section, "depthBias", config.occlusionConfig.depthBias);
        readNumber(*section, "nearSkipDistance", config.occlusionConfig.nearSkipDistance);
        readNumber(*section, "minProjectedSpanPixels", config.occlusionConfig.minProjectedSpanPixels);
    }

    if (const json* section = findMember(root, "interactionConfig"); section != nullptr && section->is_object()) {
        readNumber(*section, "blockReach", config.interactionConfig.blockReach);
    }

    if (const json* section = findMember(root, "worldConfig"); section != nullptr && section->is_object()) {
        readNumber(*section, "columnLoadRadius", config.worldConfig.columnLoadRadius);
        readNumber(*section, "maxInFlightColumnJobs", config.worldConfig.maxInFlightColumnJobs);
        loadJobConfig(*section, config.worldConfig.jobConfig);
    }

    if (const json* section = findMember(root, "meshConfig"); section != nullptr && section->is_object()) {
        readNumber(*section, "lodLevelCount", config.meshConfig.lodLevelCount);
        readNumber(*section, "meshTileSizeChunks", config.meshConfig.meshTileSizeChunks);
        readNumber(*section, "meshTileHeightChunks", config.meshConfig.meshTileHeightChunks);
        readNumber(*section, "activeChunkRadius", config.meshConfig.activeChunkRadius);
        readNumber(*section, "lodSseTargetPixels", config.meshConfig.lodSseTargetPixels);
        readNumber(*section, "lodSseHysteresisPixels", config.meshConfig.lodSseHysteresisPixels);
        readNumber(*section, "lodSseMinDepthBlocks", config.meshConfig.lodSseMinDepthBlocks);
        readNumber(*section, "lodSseFallbackProjectionScale", config.meshConfig.lodSseFallbackProjectionScale);
        loadJobConfig(*section, config.meshConfig.jobConfig);
    }

    if (const json* section = findMember(root, "meshletBufferConfig"); section != nullptr && section->is_object()) {
        readNumber(*section, "uploadBudgetBytesPerFrame", config.meshletBufferConfig.uploadBudgetBytesPerFrame);
        readNumber(*section, "maxActiveMeshlets", config.meshletBufferConfig.maxActiveMeshlets);
    }

    if (const json* section = findMember(root, "streamingConfig"); section != nullptr && section->is_object()) {
        readNumber(*section, "maxDeltaEntriesPerTick", config.streamingConfig.maxDeltaEntriesPerTick);
    }

    config.sanitize();
    std::cout << "EngineConfig: loaded '" << path.string() << "'." << std::endl;
    return config;
}

void EngineConfig::sanitize() {
    windowConfig.width = std::max(1, windowConfig.width);
    windowConfig.height = std::max(1, windowConfig.height);
    if (windowConfig.title.empty()) {
        windowConfig.title = "Voxel Engine";
    }

    windowConfig.presentMode = toLowerAscii(windowConfig.presentMode);
    if (windowConfig.presentMode.empty()) {
        windowConfig.presentMode = "fifo";
    }
    if (windowConfig.presentMode != "auto" &&
        windowConfig.presentMode != "fifo" &&
        windowConfig.presentMode != "fifo_relaxed" &&
        windowConfig.presentMode != "mailbox" &&
        windowConfig.presentMode != "immediate") {
        windowConfig.presentMode = "fifo";
    }

    if (!std::isfinite(cameraConfig.yaw)) {
        cameraConfig.yaw = 180.0f;
    }
    if (!std::isfinite(cameraConfig.pitch)) {
        cameraConfig.pitch = 0.0f;
    }
    if (!std::isfinite(cameraConfig.movementSpeed) || cameraConfig.movementSpeed <= 0.0f) {
        cameraConfig.movementSpeed = 80.0f;
    }
    if (!std::isfinite(cameraConfig.mouseSensitivity) || cameraConfig.mouseSensitivity <= 0.0f) {
        cameraConfig.mouseSensitivity = 0.1f;
    }
    if (!std::isfinite(cameraConfig.fieldOfViewDegrees) || cameraConfig.fieldOfViewDegrees <= 1.0f) {
        cameraConfig.fieldOfViewDegrees = 85.0f;
    }
    cameraConfig.fieldOfViewDegrees = std::min(cameraConfig.fieldOfViewDegrees, 179.0f);
    if (!std::isfinite(cameraConfig.nearClip) || cameraConfig.nearClip <= 0.0f) {
        cameraConfig.nearClip = 0.1f;
    }
    if (!std::isfinite(cameraConfig.farClip) || cameraConfig.farClip <= cameraConfig.nearClip) {
        cameraConfig.farClip = std::max(cameraConfig.nearClip + 1.0f, 2500.0f);
    }

    if (!std::isfinite(framePacingConfig.targetFps) || framePacingConfig.targetFps <= 1.0f) {
        framePacingConfig.targetFps = 60.0f;
    }

    if (!std::isfinite(timeConfig.dayDurationSeconds) || timeConfig.dayDurationSeconds <= 0.0f) {
        timeConfig.dayDurationSeconds = 300.0f;
    }
    if (!std::isfinite(timeConfig.timeMultiplier) || timeConfig.timeMultiplier < 0.0f) {
        timeConfig.timeMultiplier = 0.5f;
    }
    timeConfig.initialTimeHours = wrapHours(timeConfig.initialTimeHours);
    timeConfig.manualTimeHours = wrapHours(timeConfig.manualTimeHours);

    if (!std::isfinite(occlusionConfig.depthBias) || occlusionConfig.depthBias < 0.0f) {
        occlusionConfig.depthBias = 0.01f;
    }
    if (!std::isfinite(occlusionConfig.nearSkipDistance) || occlusionConfig.nearSkipDistance < 0.0f) {
        occlusionConfig.nearSkipDistance = 20.0f;
    }
    if (!std::isfinite(occlusionConfig.minProjectedSpanPixels) || occlusionConfig.minProjectedSpanPixels < 0.0f) {
        occlusionConfig.minProjectedSpanPixels = 1.0f;
    }

    if (!std::isfinite(interactionConfig.blockReach) || interactionConfig.blockReach <= 0.0f) {
        interactionConfig.blockReach = 32.0f;
    }

    worldConfig.columnLoadRadius = std::max(0, worldConfig.columnLoadRadius);

    meshConfig.lodLevelCount = std::max(1, meshConfig.lodLevelCount);
    meshConfig.meshTileSizeChunks = std::max(1, meshConfig.meshTileSizeChunks);
    meshConfig.meshTileHeightChunks = std::max(1, meshConfig.meshTileHeightChunks);
    meshConfig.activeChunkRadius = std::max(0, meshConfig.activeChunkRadius);
    if (!std::isfinite(meshConfig.lodSseTargetPixels) || meshConfig.lodSseTargetPixels <= 0.0f) {
        meshConfig.lodSseTargetPixels = 16.0f;
    }
    if (!std::isfinite(meshConfig.lodSseHysteresisPixels) || meshConfig.lodSseHysteresisPixels < 0.0f) {
        meshConfig.lodSseHysteresisPixels = 0.25f;
    }
    if (!std::isfinite(meshConfig.lodSseMinDepthBlocks) || meshConfig.lodSseMinDepthBlocks <= 0.0f) {
        meshConfig.lodSseMinDepthBlocks = 4.0f;
    }
    if (!std::isfinite(meshConfig.lodSseFallbackProjectionScale) || meshConfig.lodSseFallbackProjectionScale <= 0.0f) {
        meshConfig.lodSseFallbackProjectionScale = 390.0f;
    }

    if (meshletBufferConfig.uploadBudgetBytesPerFrame == 0u) {
        meshletBufferConfig.uploadBudgetBytesPerFrame = kDefaultUploadBudgetBytesPerFrame;
    }
    if (meshletBufferConfig.maxActiveMeshlets == 0u) {
        meshletBufferConfig.maxActiveMeshlets = kDefaultMaxActiveMeshlets;
    }

    if (streamingConfig.maxDeltaEntriesPerTick == 0u) {
        streamingConfig.maxDeltaEntriesPerTick = 1u;
    }
}
