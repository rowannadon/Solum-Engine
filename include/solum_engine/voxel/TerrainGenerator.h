#pragma once
#include <cstdint>
#include <random>

#include "glm/glm.hpp"
#include "solum_engine/voxel/Column.h"
#include "FastNoise/FastNoise.h"

class TerrainGenerator {
private:
    FastNoise::SmartNode<> fnGenerator;
    uint32_t seed = 0;
	float noiseScale = 0.004f;

    std::random_device dev;

public:

    TerrainGenerator() : fnGenerator(FastNoise::NewFromEncodedNodeTree("FAAgAA8AAwAAAAAAAEAgACEAFQAAAIA/AAAAAAAAAAAnAAEAAAAnAAAAAAATAI/CdT4PAAMAAADNzIw/DQADAAAAUriePykAAGZmZj8A9ihcPwBSuB6/ALgeBcAQAM3MzD0ZABMAPQrXPg0ABgAAAAAAAEAJAABmZiY/AAAAAD8BBAAAAAAArkehvwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADXo7A/AHsUrj4BBAAAAAAAAADIwgAAAAAAAAAAAAAAAD0KVz8AAAAAAAAAAAAAAMhCAClcDz4AmpkZQAEEAAAAAAAfhVvBAAAAAAAAAAAAAAAAzczMvwAAAAAAAAAAAB+F+0AAAAAAAABSuJ4/AAAAAAAAAAAAAA==")) {};

    void generateColumn(const glm::ivec3& origin, Column& col);
};
