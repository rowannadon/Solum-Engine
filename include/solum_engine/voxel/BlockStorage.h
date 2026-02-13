#include "solum_engine/voxel/BlockMaterial.h"
#include "solum_engine/resources/Constants.h"
#include "solum_engine/resources/Coords.h"
class BlockStorage {
    std::array<BlockMaterial, CHUNK_BLOCKS> data;
public:
    bool setBlock(BlockCoord pos, UnpackedBlockMaterial mat);
    UnpackedBlockMaterial getBlock(BlockCoord pos);

    BlockMaterial* getData() {
        return data.data();
    }

private:
    bool validatePos(BlockCoord pos);
    int getIndex(BlockCoord pos);
};