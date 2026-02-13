#include "solum_engine/resources/Constants.h"
#include "solum_engine/resources/Coords.h"
#include "solum_engine/voxel/BlockMaterial.h"
#include <unordered_map>


class ChunkPool {
    static const int POOL_SIZE = 100;
    std::array<BlockMaterial, POOL_SIZE * CHUNK_BLOCKS> data;
    
public:

    
    
};