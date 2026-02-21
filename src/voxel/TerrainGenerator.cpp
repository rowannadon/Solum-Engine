#include "solum_engine/voxel/TerrainGenerator.h"
#include "solum_engine/resources/Constants.h"

void TerrainGenerator::generateColumn(const glm::ivec3 origin, Column& col) {
    std::array<float, cfg::CHUNK_SIZE * cfg::CHUNK_SIZE * cfg::COLUMN_HEIGHT_BLOCKS> noiseData;

    fnGenerator->GenUniformGrid3D(noiseData.data(), origin.x, origin.z, origin.y, cfg::CHUNK_SIZE, cfg::COLUMN_HEIGHT_BLOCKS, cfg::CHUNK_SIZE, noiseScale, seed);

    UnpackedBlockMaterial solid{1, 0, Direction::PlusX, 0};

    UnpackedBlockMaterial air{0, 0, Direction::PlusX, 0};

    BlockMaterial solidPacked = solid.pack();
    BlockMaterial airPacked = air.pack();

    int index = 0;
    for( int z = 0; z < cfg::CHUNK_SIZE; z++ )
    {
        for( int y = 0; y < cfg::COLUMN_HEIGHT_BLOCKS; y++ )
        {
            for( int x = 0; x < cfg::CHUNK_SIZE; x++ )
            {
                // Process the voxel noise however you need for your use case
                // Create a terrain mesh using marching cubes, etc.
                //std::printf("%f\n", noiseData.at(index++));
                if (y < 100) {
                    col.setBlock(x, y, z, solidPacked);
                } else {
                    col.setBlock(x, y, z, airPacked);
                }
            }
        }
    }
}