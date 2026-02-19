#include "solum_engine/resources/Constants.h"
#include "solum_engine/voxel/BlockMaterial.h"

class RegionGenerator {
private:
    static constexpr int SCRATCH_AREA_SIZE = (REGION_BLOCKS_XY + 2) * (REGION_BLOCKS_XY + 2) * COLUMN_HEIGHT_BLOCKS;

    std::array<BlockMaterial, SCRATCH_AREA_SIZE> genArea;

};