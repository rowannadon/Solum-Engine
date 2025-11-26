#include <vector>
#include "solum_engine/voxel/BlockMaterial.h"
#include "solum_engine/resources/Constants.h"
#include <glm/glm.hpp>

class Chunk {
    std::vector<BlockMaterial> data;
    
public:
    Chunk();
    bool setBlock(glm::ivec3 pos, UnpackedBlockMaterial mat);
    UnpackedBlockMaterial getBlock(glm::ivec3 pos);

private:
    bool validatePos(glm::ivec3 pos);
    int getIndex(glm::ivec3 pos);
};