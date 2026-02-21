#include "glm/glm.hpp"
#include "solum_engine/voxel/Column.h"
#include "FastNoise/FastNoise.h"

class TerrainGenerator {
private:
    FastNoise::SmartNode<> fnGenerator;
    uint32_t seed = 0;
	float noiseScale = 0.004f;

public:

    TerrainGenerator() : fnGenerator(FastNoise::NewFromEncodedNodeTree("FQAAAAAAAAAAAGDlwD8VAK5HYT4AAAAAKVyPvhsAEACPwvU/JQAK16M79iicvwAAAAAfhQtAEwDsUfg/DQAEAAAAKVyPPwkAAGZmJj8AAAAAPwC4HoU+ALgehb8=")) {};

    void generateColumn(const glm::ivec3 origin, Column& col);
};