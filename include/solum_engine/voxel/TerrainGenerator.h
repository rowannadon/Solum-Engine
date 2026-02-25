#pragma once
#include "glm/glm.hpp"
#include "solum_engine/voxel/Column.h"
#include "FastNoise/FastNoise.h"

class TerrainGenerator {
private:
    FastNoise::SmartNode<> fnGenerator;

public:
    TerrainGenerator() : fnGenerator(FastNoise::New<FastNoise::Perlin>()) {}


    void generateColumn(const glm::ivec3& origin, Column& col);
};
