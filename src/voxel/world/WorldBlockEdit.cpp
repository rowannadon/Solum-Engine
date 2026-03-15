#include "solum_engine/voxel/World.h"

#include <algorithm>
#include <array>
#include <shared_mutex>
#include <unordered_set>

#include "solum_engine/resources/Constants.h"
#include "solum_engine/voxel/Column.h"

namespace {
BlockMaterial airBlock() {
    static const BlockMaterial kAir = UnpackedBlockMaterial{}.pack();
    return kAir;
}

constexpr std::array<glm::ivec2, 4> kHorizontalOffsets = {
    glm::ivec2{1, 0},
    glm::ivec2{-1, 0},
    glm::ivec2{0, 1},
    glm::ivec2{0, -1},
};

constexpr uint8_t kLightingDirtyTopology = 1u << 0u;
constexpr uint8_t kLightingDirtyBoundary = 1u << 1u;
}  // namespace

bool World::breakBlock(const BlockCoord& coord) {
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return false;
    }
    bool changed = false;
    {
        std::unique_lock<std::shared_mutex> lock(worldMutex_);
        changed = applyBlockEditLocked(coord, airBlock(), false, true);
    }
    if (changed) {
        const ChunkCoord editedChunk = block_to_chunk(coord);
        tryApplyImmediateLightingAround(editedChunk);
        pumpChunkPropagationQueue();
    }
    return changed;
}

bool World::placeBlock(const BlockCoord& coord, const BlockMaterial& block) {
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return false;
    }
    bool changed = false;
    {
        std::unique_lock<std::shared_mutex> lock(worldMutex_);
        changed = applyBlockEditLocked(coord, block, true, false);
    }
    if (changed) {
        const ChunkCoord editedChunk = block_to_chunk(coord);
        tryApplyImmediateLightingAround(editedChunk);
        pumpChunkPropagationQueue();
    }
    return changed;
}

bool World::applyBlockEditLocked(const BlockCoord& coord,
                                 const BlockMaterial& newBlock,
                                 bool requireCurrentAir,
                                 bool requireCurrentSolid) {
    if (coord.v.z < 0 || coord.v.z >= cfg::COLUMN_HEIGHT_BLOCKS) {
        return false;
    }
    const ChunkCoord chunkCoord = block_to_chunk(coord);
    if (chunkCoord.v.z < 0 || chunkCoord.v.z >= cfg::COLUMN_HEIGHT) {
        return false;
    }

    const ColumnCoord columnCoord = chunk_to_column(chunkCoord);
    if (!isColumnSkycastCompleteLocked(columnCoord)) {
        return false;
    }

    Column* column = tryGetSkycastColumnLocked(columnCoord);
    if (column == nullptr) {
        return false;
    }

    const uint8_t localX = static_cast<uint8_t>(floor_mod(coord.v.x, cfg::CHUNK_SIZE));
    const uint8_t localY = static_cast<uint8_t>(floor_mod(coord.v.y, cfg::CHUNK_SIZE));
    const uint16_t localZ = static_cast<uint16_t>(floor_mod(coord.v.z, cfg::COLUMN_HEIGHT_BLOCKS));
    const BlockMaterial currentBlock = column->getBlock(localX, localY, localZ);
    if (currentBlock == newBlock) {
        return false;
    }

    const bool currentIsAir = (currentBlock.unpack().id == 0u);
    if (requireCurrentAir && !currentIsAir) {
        return false;
    }
    if (requireCurrentSolid && currentIsAir) {
        return false;
    }

    const uint8_t changedMipMask = column->setBlock(localX, localY, localZ, newBlock);
    if (changedMipMask == 0u) {
        return false;
    }

    const std::vector<ColumnCoord> geometryDirtyColumns =
        collectGeometryDirtyColumnsLocked(columnCoord, localX, localY);
    const std::vector<ChunkCoord> geometryDirtyChunks =
        collectGeometryDirtyChunksLocked(geometryDirtyColumns, chunkCoord, localZ);

    for (const ChunkCoord& dirtyChunk : geometryDirtyChunks) {
        if (!isColumnSkycastCompleteLocked(chunk_to_column(dirtyChunk))) {
            continue;
        }
        recordPlayerEditedChunkEventLocked(WorldChunkEdit{dirtyChunk, changedMipMask});
    }

    for (int32_t oy = -1; oy <= 1; ++oy) {
        for (int32_t ox = -1; ox <= 1; ++ox) {
            int32_t zMin = std::max(0, chunkCoord.v.z - 1);
            const int32_t zMax = std::min(cfg::COLUMN_HEIGHT - 1, chunkCoord.v.z + 1);
            if (ox == 0 && oy == 0) {
                zMin = 0;
            }

            for (int32_t nz = zMin; nz <= zMax; ++nz) {
                bumpChunkTopologyEpochLocked(
                    ChunkCoord{chunkCoord.v.x + ox, chunkCoord.v.y + oy, nz},
                    true,
                    static_cast<uint8_t>(kLightingDirtyTopology | kLightingDirtyBoundary)
                );
            }
        }
    }

    return true;
}

std::vector<ColumnCoord> World::collectGeometryDirtyColumnsLocked(const ColumnCoord& columnCoord,
                                                                  uint8_t localX,
                                                                  uint8_t localY) const {
    std::vector<ColumnCoord> dirtyColumns;
    dirtyColumns.push_back(columnCoord);

    if (localX == 0u) {
        dirtyColumns.push_back(ColumnCoord{columnCoord.v.x - 1, columnCoord.v.y});
    } else if (localX == static_cast<uint8_t>(cfg::CHUNK_SIZE - 1)) {
        dirtyColumns.push_back(ColumnCoord{columnCoord.v.x + 1, columnCoord.v.y});
    }

    if (localY == 0u) {
        dirtyColumns.push_back(ColumnCoord{columnCoord.v.x, columnCoord.v.y - 1});
    } else if (localY == static_cast<uint8_t>(cfg::CHUNK_SIZE - 1)) {
        dirtyColumns.push_back(ColumnCoord{columnCoord.v.x, columnCoord.v.y + 1});
    }

    std::sort(dirtyColumns.begin(), dirtyColumns.end());
    dirtyColumns.erase(std::unique(dirtyColumns.begin(), dirtyColumns.end()), dirtyColumns.end());
    return dirtyColumns;
}

std::vector<ChunkCoord> World::collectGeometryDirtyChunksLocked(const std::vector<ColumnCoord>& dirtyColumns,
                                                                const ChunkCoord& chunkCoord,
                                                                uint16_t localZ) const {
    std::vector<ChunkCoord> dirtyChunks;
    dirtyChunks.reserve(dirtyColumns.size() * 3u);

    for (const ColumnCoord& dirtyColumn : dirtyColumns) {
        dirtyChunks.push_back(ChunkCoord{dirtyColumn.v.x, dirtyColumn.v.y, chunkCoord.v.z});
        if (localZ == 0u && chunkCoord.v.z > 0) {
            dirtyChunks.push_back(ChunkCoord{dirtyColumn.v.x, dirtyColumn.v.y, chunkCoord.v.z - 1});
        } else if (localZ == static_cast<uint16_t>(cfg::CHUNK_SIZE - 1) &&
                   chunkCoord.v.z + 1 < cfg::COLUMN_HEIGHT) {
            dirtyChunks.push_back(ChunkCoord{dirtyColumn.v.x, dirtyColumn.v.y, chunkCoord.v.z + 1});
        }
    }

    std::sort(dirtyChunks.begin(), dirtyChunks.end());
    dirtyChunks.erase(std::unique(dirtyChunks.begin(), dirtyChunks.end()), dirtyChunks.end());
    return dirtyChunks;
}

void World::recordPlayerEditedChunkEventLocked(const WorldChunkEdit& edit) {
    appendRevisionEventLocked(playerEditedChunkEvents_, edit, playerEditChunkRevision_);
}

void World::recordGeneratedColumnEventLocked(const ColumnCoord& coord) {
    appendRevisionEventLocked(generatedColumnEvents_, coord, generationRevision_);
}

void World::recordLightingChangedChunkEventLocked(const ChunkCoord& coord) {
    appendRevisionEventLocked(lightingChangedChunkEvents_, coord, lightingChunkRevision_);
}

void World::enqueueChunkPropagationCandidatesLocked(const ColumnCoord& coord) {
    auto enqueueForColumn = [&](const ColumnCoord& candidateColumn) {
        if (!isColumnSkycastCompleteLocked(candidateColumn)) {
            return;
        }
        for (int32_t z = 0; z < cfg::COLUMN_HEIGHT; ++z) {
            bumpChunkTopologyEpochLocked(
                ChunkCoord{candidateColumn.v.x, candidateColumn.v.y, z},
                false,
                kLightingDirtyBoundary
            );
        }
    };

    enqueueForColumn(coord);
    for (const glm::ivec2& offset : kHorizontalOffsets) {
        enqueueForColumn(ColumnCoord{
            coord.v.x + offset.x,
            coord.v.y + offset.y
        });
    }
}
