#include "solum_engine/voxel/TerrainGenerator.h"
#include "solum_engine/resources/Constants.h"


void TerrainGenerator::generateColumn(const glm::ivec3& origin, Column& col) {
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(1,10);

    UnpackedBlockMaterial solid{1, 0, Direction::PlusZ, 0};

    UnpackedBlockMaterial air{0, 0, Direction::PlusZ, 0};

    BlockMaterial solidPacked = solid.pack();
    BlockMaterial airPacked = air.pack();

    int index = 0;
    // Consume output in FastNoise's required linear order: z -> y -> x.
    // noiseZ corresponds to worldY, noiseY corresponds to worldZ.
    for (int z = 0; z < cfg::COLUMN_HEIGHT_BLOCKS; z++)
    {
        for (int y = 0; y < cfg::CHUNK_SIZE; y++)
        {
            for (int x = 0; x < cfg::CHUNK_SIZE; x++)
            {
                if (fnGenerator->GenSingle3D(
                    static_cast<float>(origin.x + x) * noiseScale, 
                    static_cast<float>(origin.z + z) * noiseScale, 
                    static_cast<float>(origin.y + y) * noiseScale, 
                    seed
                ) > 0.0f) {
                    col.setBlock(x, y, z, solidPacked);
                } else {
                    col.setBlock(x, y, z, airPacked);
                }
            }
        }
    }
}
