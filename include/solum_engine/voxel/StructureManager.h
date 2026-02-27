#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "solum_engine/voxel/BlockMaterial.h"

class Column;

class StructureManager {
public:
    enum class Rotation : uint8_t {
        Random = 255,
        Deg0 = 0,
        Deg90 = 1,
        Deg180 = 2,
        Deg270 = 3
    };

    struct ColorMaterialMapping {
        uint8_t r = 0;
        uint8_t g = 0;
        uint8_t b = 0;
        uint8_t a = 255;
        BlockMaterial material{};
    };

    struct StructureDefinition {
        std::string name;
        std::string voxFilePath;
        glm::ivec3 generationOrigin{0, 0, 0};
        std::vector<ColorMaterialMapping> colorMappings;
        uint32_t selectionWeight = 1;
    };

    struct PlacementPoint {
        glm::ivec2 worldXY{0, 0};
        uint64_t key = 0;
    };

    struct SamplerConfig {
        int32_t cellSize = 14;
        int32_t minDistance = 8;
        float cellOccupancy = 0.45f;
        uint32_t seed = 0x51F15EEDu;
    };

    StructureManager();
    explicit StructureManager(const SamplerConfig& samplerConfig);

    void clear();
    bool addStructure(const StructureDefinition& definition);

    bool hasStructures() const noexcept;
    int32_t maxHorizontalReach() const noexcept;

    void collectPointsForBounds(const glm::ivec2& minInclusive,
                                const glm::ivec2& maxExclusive,
                                std::vector<PlacementPoint>& outPoints) const;

    void placeStructureForPoint(const PlacementPoint& point,
                                const glm::ivec3& anchorWorld,
                                const glm::ivec3& clipMinInclusive,
                                const glm::ivec3& clipMaxExclusive,
                                Rotation rotation,
                                Column& column) const;
    void placeStructureForPoint(const PlacementPoint& point,
                                const glm::ivec3& anchorWorld,
                                const glm::ivec3& clipMinInclusive,
                                const glm::ivec3& clipMaxExclusive,
                                Column& column) const;

private:
    struct LoadedVoxel {
        glm::ivec3 local{0, 0, 0};
        BlockMaterial material{};
    };

    struct LoadedStructure {
        std::string name;
        glm::ivec3 generationOrigin{0, 0, 0};
        std::vector<LoadedVoxel> voxels;
        uint32_t selectionWeight = 1;
        int32_t horizontalReach = 0;
    };

    struct CandidatePoint {
        int32_t cellX = 0;
        int32_t cellY = 0;
        int32_t worldX = 0;
        int32_t worldY = 0;
        uint64_t priority = 0;
        uint64_t key = 0;
        bool active = false;
    };

    CandidatePoint makeCandidateForCell(int32_t cellX, int32_t cellY) const;
    bool isCandidateAccepted(const CandidatePoint& candidate) const;
    std::size_t pickStructureIndex(uint64_t pointKey) const;
    Rotation pickRotation(uint64_t pointKey, std::size_t structureIndex) const;
    static glm::ivec3 rotateOffset(const glm::ivec3& offset, Rotation rotation);
    bool loadVoxStructure(const StructureDefinition& definition, LoadedStructure& outStructure) const;

    SamplerConfig samplerConfig_{};
    int32_t cellSize_ = 1;
    int32_t minDistance_ = 1;
    int64_t minDistanceSq_ = 1;
    int32_t neighborRangeCells_ = 1;
    uint64_t seed_ = 0;
    uint32_t occupancyThreshold_ = 0;

    std::vector<LoadedStructure> structures_;
    uint64_t totalSelectionWeight_ = 0;
    int32_t maxHorizontalReach_ = 0;
};
