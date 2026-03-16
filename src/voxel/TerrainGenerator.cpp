#include "solum_engine/voxel/TerrainGenerator.h"

#include "terrain/TerrainGeneratorInternal.h"

TerrainGenerator::TerrainGenerator()
    : fnGenerator(terrain_internal::createTerrainNoiseGenerator()) {}

void TerrainGenerator::generateColumn(const glm::ivec3& origin, Column& col) {
    const terrain_internal::HeightmapData& heightmap = terrain_internal::heightmapData();
    const terrain_internal::TerrainDecorationConfig config = terrain_internal::decorationConfig();

    terrain_internal::generateTerrainColumn(origin, col, fnGenerator, heightmap, config);
    terrain_internal::placeColumnStructures(origin, col, fnGenerator, heightmap);
    terrain_internal::bootstrapColumnLighting(col);
}
