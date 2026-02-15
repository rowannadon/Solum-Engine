#include "solum_engine/voxel/BlockMaterial.h"
#include "solum_engine/resources/Constants.h"
#include "solum_engine/resources/Coords.h"
class BlockStorage {
    BlockMaterial data[CHUNK_BLOCKS];
public:
    bool setBlock(BlockCoord pos, UnpackedBlockMaterial mat);
    UnpackedBlockMaterial getBlock(BlockCoord pos);

    BlockMaterial* getData() {
        return data;
    }

private:
    bool validatePos(BlockCoord pos);
    int getIndex(BlockCoord pos);
};