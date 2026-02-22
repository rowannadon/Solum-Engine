#pragma once
#include "glm/glm.hpp"
#include "solum_engine/voxel/Column.h"

class TerrainGenerator {
public:
    TerrainGenerator() = default;

    void generateColumn(const glm::ivec3& origin, Column& col);
};
