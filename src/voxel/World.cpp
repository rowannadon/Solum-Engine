#include "solum_engine/voxel/World.h"

#include "solum_engine/voxel/ChunkMesher.h"
#include "solum_engine/voxel/ChunkResidencyManager.h"
#include "solum_engine/voxel/CompressedStore.h"
#include "solum_engine/voxel/JobScheduler.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace {
constexpr int kApplyBudgetPerFrame = 256;

uint64_t mix64(uint64_t x) {
    x ^= x >> 33u;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33u;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33u;
    return x;
}

uint64_t tileScheduleKey(RegionCoord regionCoord, int lodLevel, RegionLodTileCoord tile) {
    uint64_t key = mix64(static_cast<uint64_t>(static_cast<uint32_t>(regionCoord.x())));
    key ^= mix64(static_cast<uint64_t>(static_cast<uint32_t>(regionCoord.y())) + 0x9E3779B97F4A7C15ULL);
    key ^= mix64(static_cast<uint64_t>(lodLevel + 1));
    key ^= mix64(
        static_cast<uint64_t>(tile.x) |
        (static_cast<uint64_t>(tile.y) << 8u) |
        (static_cast<uint64_t>(tile.z) << 16u)
    );
    return key;
}

int terrainHeight(int worldX, int worldY, uint64_t seed) {
    const uint64_t h0 = mix64(seed ^ static_cast<uint32_t>(worldX));
    const uint64_t h1 = mix64((seed << 1u) ^ static_cast<uint32_t>(worldY));

    const int coarse = static_cast<int>(h0 % 220u);
    const int fine = static_cast<int>(h1 % 50u);

    int height = 260 + coarse - fine;
    height = std::clamp(height, 0, (CHUNK_SIZE * COLUMN_CHUNKS_Z) - 1);
    return height;
}

BlockMaterial terrainMaterial(int worldZ, int height, int seaLevel) {
    UnpackedBlockMaterial mat{};

    if (worldZ > height) {
        if (worldZ <= seaLevel) {
            mat.id = 3;
            mat.waterLevel = std::clamp(seaLevel - worldZ, 0, 15);
        } else {
            mat.id = 0;
        }
        return mat.pack();
    }

    if (worldZ == height) {
        mat.id = 2;
    } else if (worldZ > height - 4) {
        mat.id = 1;
    } else {
        mat.id = 6;
    }

    return mat.pack();
}

std::array<glm::ivec3, 6> neighborOffsets() {
    return {
        glm::ivec3{1, 0, 0},
        glm::ivec3{-1, 0, 0},
        glm::ivec3{0, 1, 0},
        glm::ivec3{0, -1, 0},
        glm::ivec3{0, 0, 1},
        glm::ivec3{0, 0, -1},
    };
}
} // namespace

void WorldInterestSet::clear() {
    regions.clear();
    columns.clear();
    chunks.clear();
}

World::World(std::size_t chunkPoolCapacity, std::size_t workerThreads)
    : chunkPool_(chunkPoolCapacity)
    , compressedStore_(std::make_unique<CompressedStore>())
    , regionManager_(&chunkPool_)
    , residencyManager_(std::make_unique<ChunkResidencyManager>(chunkPool_, *compressedStore_))
    , scheduler_(std::make_unique<JobScheduler>(workerThreads)) {
    scheduler_->setExecutor([this](const VoxelJob& job) {
        return executeJob(job);
    });
}

World::~World() = default;

void World::update(const PlayerStreamingContext& context) {
    const WorldInterestSet interestSet = buildInterestSet(context);
    scheduleStreamingWork(interestSet);
    applyCompletedJobs(kApplyBudgetPerFrame);
}

bool World::setBlock(BlockCoord worldBlockCoord, UnpackedBlockMaterial material) {
    std::scoped_lock lock(worldMutex_);

    ChunkCoord chunkCoord = block_to_chunk(worldBlockCoord);
    Chunk* chunk = regionManager_.tryGetChunk(chunkCoord);
    if (chunk == nullptr) {
        regionManager_.ensureColumn(chunk_to_column(chunkCoord));
        chunk = regionManager_.tryGetChunk(chunkCoord);
    }

    if (chunk == nullptr) {
        return false;
    }

    if (!residencyManager_->ensureResident(*chunk)) {
        return false;
    }

    const glm::ivec3 local = block_local_in_chunk(worldBlockCoord);
    const bool changed = chunk->setBlock(BlockCoord{local.x, local.y, local.z}, material);
    if (!changed) {
        return false;
    }

    if (Region* region = regionManager_.tryGetRegion(chunk_to_region(chunkCoord)); region != nullptr) {
        region->markLodTilesDirtyForChunk(chunkCoord);
    }

    const uint32_t edgeMask = chunk->state().consumeEdgeDirtyMask();
    if (edgeMask != 0u) {
        const auto offsets = neighborOffsets();

        constexpr std::array<ChunkDirtyFlags, 6> edgeFlags = {
            ChunkDirtyFlags::EdgeDirtyPosX,
            ChunkDirtyFlags::EdgeDirtyNegX,
            ChunkDirtyFlags::EdgeDirtyPosY,
            ChunkDirtyFlags::EdgeDirtyNegY,
            ChunkDirtyFlags::EdgeDirtyPosZ,
            ChunkDirtyFlags::EdgeDirtyNegZ,
        };

        for (int i = 0; i < 6; ++i) {
            if ((edgeMask & static_cast<uint32_t>(edgeFlags[static_cast<std::size_t>(i)])) == 0u) {
                continue;
            }

            const ChunkCoord neighborCoord{
                chunkCoord.x() + offsets[static_cast<std::size_t>(i)].x,
                chunkCoord.y() + offsets[static_cast<std::size_t>(i)].y,
                chunkCoord.z() + offsets[static_cast<std::size_t>(i)].z,
            };

            if (Chunk* neighbor = regionManager_.tryGetChunk(neighborCoord); neighbor != nullptr) {
                neighbor->state().markDirty(ChunkDirtyFlags::NeedsMeshL0);
            }
        }
    }

    return true;
}

UnpackedBlockMaterial World::getBlock(BlockCoord worldBlockCoord) const {
    std::scoped_lock lock(worldMutex_);

    const ChunkCoord chunkCoord = block_to_chunk(worldBlockCoord);
    const Chunk* chunk = regionManager_.tryGetChunk(chunkCoord);
    if (chunk == nullptr) {
        return {};
    }

    const glm::ivec3 local = block_local_in_chunk(worldBlockCoord);
    return chunk->getBlock(BlockCoord{local.x, local.y, local.z});
}

RegionManager& World::regions() {
    return regionManager_;
}

const RegionManager& World::regions() const {
    return regionManager_;
}

ChunkPool& World::chunkPool() {
    return chunkPool_;
}

const ChunkPool& World::chunkPool() const {
    return chunkPool_;
}

WorldInterestSet World::buildInterestSet(const PlayerStreamingContext& context) const {
    WorldInterestSet set;

    const BlockCoord playerBlock{
        static_cast<int32_t>(std::floor(context.playerPosition.x)),
        static_cast<int32_t>(std::floor(context.playerPosition.y)),
        static_cast<int32_t>(std::floor(context.playerPosition.z)),
    };

    const ChunkCoord playerChunk = block_to_chunk(playerBlock);
    const int viewDistance = std::max(1, context.viewDistanceChunks);

    const int zMin = std::clamp(context.verticalChunkMin, 0, COLUMN_CHUNKS_Z - 1);
    const int zMax = std::clamp(context.verticalChunkMax, 0, COLUMN_CHUNKS_Z - 1);

    for (int dx = -viewDistance; dx <= viewDistance; ++dx) {
        for (int dy = -viewDistance; dy <= viewDistance; ++dy) {
            const ColumnCoord columnCoord{playerChunk.x() + dx, playerChunk.y() + dy};
            set.columns.insert(columnCoord);
            set.regions.insert(column_to_region(columnCoord));

            for (int z = zMin; z <= zMax; ++z) {
                set.chunks.insert(ChunkCoord{columnCoord.x(), columnCoord.y(), z});
            }
        }
    }

    return set;
}

void World::scheduleStreamingWork(const WorldInterestSet& interestSet) {
    std::scoped_lock lock(worldMutex_);

    for (const RegionCoord regionCoord : interestSet.regions) {
        Region& region = regionManager_.ensureRegion(regionCoord);
        if (region.state().generationState() == RegionGenerationState::Empty) {
            region.state().setGenerationState(RegionGenerationState::Partial);
        }
    }

    for (const ColumnCoord columnCoord : interestSet.columns) {
        Column& column = regionManager_.ensureColumn(columnCoord);

        if (column.state().stage() == ColumnStage::Empty) {
            if (terrainScheduled_.insert(columnCoord).second) {
                VoxelJob job;
                job.type = VoxelJobType::TerrainGeneration;
                job.priority = JobPriority::Medium;
                job.payload = TerrainGenerationJobInput{columnCoord, hashForColumnSeed(columnCoord)};
                scheduler_->enqueue(std::move(job));
            }
        }

        if (column.state().canRunStructureGeneration()) {
            if (structureScheduled_.insert(columnCoord).second) {
                VoxelJob job;
                job.type = VoxelJobType::StructureGeneration;
                job.priority = JobPriority::Medium;
                job.payload = StructureGenerationJobInput{columnCoord};
                scheduler_->enqueue(std::move(job));
            }
        }
    }

    for (const ChunkCoord chunkCoord : interestSet.chunks) {
        Chunk* chunk = regionManager_.tryGetChunk(chunkCoord);
        if (chunk == nullptr) {
            continue;
        }

        residencyManager_->ensureResident(*chunk);

        if (chunk->state().needsLodScan() && lodScanScheduled_.insert(chunkCoord).second) {
            VoxelJob job;
            job.type = VoxelJobType::LodScan;
            job.priority = JobPriority::Medium;
            job.payload = LodScanJobInput{chunkCoord, chunk->state().blockDataVersion()};
            scheduler_->enqueue(std::move(job));
        }

        if (chunk->state().needsMeshL0() && meshScheduled_.insert(chunkCoord).second) {
            VoxelJob job;
            job.type = VoxelJobType::MeshL0;
            job.priority = JobPriority::High;
            job.payload = MeshJobInput{chunkCoord, chunk->state().blockDataVersion()};
            scheduler_->enqueue(std::move(job));
        }
    }

    for (const RegionCoord regionCoord : interestSet.regions) {
        Region* region = regionManager_.tryGetRegion(regionCoord);
        if (region == nullptr) {
            continue;
        }

        for (int lodLevel = 0; lodLevel < 4; ++lodLevel) {
            for (int z = 0; z < COLUMN_CHUNKS_Z; ++z) {
                const std::vector<RegionLodTileCoord> dirtyTiles = region->collectDirtyTiles(lodLevel, static_cast<uint8_t>(z));
                for (const RegionLodTileCoord tile : dirtyTiles) {
                    const uint64_t key = tileScheduleKey(regionCoord, lodLevel, tile);
                    if (!lodTileScheduled_.insert(key).second) {
                        continue;
                    }

                    VoxelJob job;
                    job.type = VoxelJobType::LodTile;
                    job.priority = JobPriority::Low;
                    job.payload = LodTileJobInput{regionCoord, lodLevel, tile, 0};
                    scheduler_->enqueue(std::move(job));
                }
            }
        }

        if (region->state().generationState() == RegionGenerationState::Partial) {
            region->state().setGenerationState(RegionGenerationState::Complete);
        }
    }

    std::vector<Chunk*> evictionCandidates;
    for (const RegionCoord regionCoord : regionManager_.regionCoords()) {
        if (interestSet.regions.find(regionCoord) != interestSet.regions.end()) {
            continue;
        }

        Region* region = regionManager_.tryGetRegion(regionCoord);
        if (region == nullptr) {
            continue;
        }

        for (int lx = 0; lx < REGION_COLS; ++lx) {
            for (int ly = 0; ly < REGION_COLS; ++ly) {
                Column* column = region->tryGetColumn(lx, ly);
                if (column == nullptr) {
                    continue;
                }

                for (int z = 0; z < COLUMN_CHUNKS_Z; ++z) {
                    Chunk* chunk = column->tryGetChunk(z);
                    if (chunk != nullptr && chunk->isPoolResident()) {
                        evictionCandidates.push_back(chunk);
                    }
                }
            }
        }
    }

    const std::size_t targetFreeSlots = std::max<std::size_t>(8, chunkPool_.capacity() / 8);
    residencyManager_->evictChunks(evictionCandidates, targetFreeSlots);
}

void World::applyCompletedJobs(std::size_t budget) {
    for (std::size_t i = 0; i < budget; ++i) {
        JobResult result;
        if (!scheduler_->tryPopResult(result)) {
            return;
        }

        switch (result.type) {
        case VoxelJobType::TerrainGeneration: {
            const auto payload = std::get<TerrainJobResult>(result.payload);
            applyTerrainCompletion(payload.columnCoord, payload.success);
            break;
        }
        case VoxelJobType::StructureGeneration: {
            const auto payload = std::get<StructureJobResult>(result.payload);
            applyStructureCompletion(payload.columnCoord, payload.success);
            break;
        }
        case VoxelJobType::LodScan: {
            const auto payload = std::get<LodScanJobResult>(result.payload);
            applyLodScanCompletion(payload.chunkCoord, payload.derivedVersion, payload.success);
            break;
        }
        case VoxelJobType::MeshL0: {
            auto payload = std::get<MeshJobResult>(std::move(result.payload));
            applyMeshCompletion(payload.chunkCoord, std::move(payload.meshData), payload.derivedVersion, payload.success);
            break;
        }
        case VoxelJobType::LodTile: {
            auto payload = std::get<LodTileJobResult>(std::move(result.payload));
            applyLodTileCompletion(
                payload.regionCoord,
                payload.lodLevel,
                payload.tileCoord,
                std::move(payload.meshData),
                payload.derivedVersion,
                payload.success
            );
            break;
        }
        case VoxelJobType::CompressChunk: {
            const auto payload = std::get<CompressChunkJobResult>(result.payload);
            (void)payload;
            break;
        }
        case VoxelJobType::UncompressChunk: {
            const auto payload = std::get<UncompressChunkJobResult>(result.payload);
            (void)payload;
            break;
        }
        }
    }
}

uint64_t World::hashForColumnSeed(ColumnCoord coord) const {
    uint64_t hash = mix64(worldSeed_ ^ static_cast<uint32_t>(coord.x()));
    hash ^= mix64(static_cast<uint32_t>(coord.y()) + 0x9E3779B97F4A7C15ULL);
    return hash;
}

JobResult World::executeJob(const VoxelJob& job) {
    JobResult result;
    result.type = job.type;
    result.ticket = job.ticket;

    switch (job.type) {
    case VoxelJobType::TerrainGeneration: {
        const auto input = std::get<TerrainGenerationJobInput>(job.payload);

        bool success = false;
        {
            std::scoped_lock lock(worldMutex_);
            Column& column = regionManager_.ensureColumn(input.columnCoord);

            if (column.state().stage() == ColumnStage::Empty) {
                const int seaLevel = 240;

                success = true;
                for (int localChunkZ = 0; localChunkZ < COLUMN_CHUNKS_Z; ++localChunkZ) {
                    Chunk& chunk = column.chunkAt(localChunkZ);
                    if (!residencyManager_->ensureResident(chunk)) {
                        success = false;
                        break;
                    }

                    BlockMaterial* blocks = chunk.getBlockData();
                    if (blocks == nullptr) {
                        success = false;
                        break;
                    }

                    for (int x = 0; x < CHUNK_SIZE; ++x) {
                        for (int y = 0; y < CHUNK_SIZE; ++y) {
                            const int worldX = input.columnCoord.x() * CHUNK_SIZE + x;
                            const int worldY = input.columnCoord.y() * CHUNK_SIZE + y;
                            const int h = terrainHeight(worldX, worldY, input.seed);

                            for (int z = 0; z < CHUNK_SIZE; ++z) {
                                const int worldZ = localChunkZ * CHUNK_SIZE + z;
                                const int idx = (x * CHUNK_SIZE * CHUNK_SIZE) + (y * CHUNK_SIZE) + z;
                                blocks[idx] = terrainMaterial(worldZ, h, seaLevel);
                            }
                        }
                    }

                    chunk.markBulkDataWrite();

                    if (Region* region = regionManager_.tryGetRegion(chunk_to_region(chunk.coord())); region != nullptr) {
                        region->markLodTilesDirtyForChunk(chunk.coord());
                    }
                }

                if (success) {
                    column.state().setStructureSeed(hashForColumnSeed(input.columnCoord));
                    column.state().setStage(ColumnStage::TerrainReady);
                    column.state().bumpContentVersion();

                    for (int ox = -1; ox <= 1; ++ox) {
                        for (int oy = -1; oy <= 1; ++oy) {
                            Column& neighbor = regionManager_.ensureColumn(ColumnCoord{input.columnCoord.x() + ox, input.columnCoord.y() + oy});
                            neighbor.state().incrementNeighborTerrainReadyCount();
                        }
                    }
                }
            }
        }

        result.payload = TerrainJobResult{input.columnCoord, success};
        break;
    }

    case VoxelJobType::StructureGeneration: {
        const auto input = std::get<StructureGenerationJobInput>(job.payload);

        bool success = false;
        {
            std::scoped_lock lock(worldMutex_);
            Column* column = regionManager_.tryGetColumn(input.columnCoord);
            if (column != nullptr && column->state().canRunStructureGeneration()) {
                const uint64_t seed = column->state().structureSeed();
                success = true;

                for (int s = 0; s < 3; ++s) {
                    const uint64_t localSeed = mix64(seed + static_cast<uint64_t>(s));
                    const int baseX = static_cast<int>((localSeed >> 0u) & 31u);
                    const int baseY = static_cast<int>((localSeed >> 8u) & 31u);
                    const int trunkBaseZ = 220 + static_cast<int>((localSeed >> 16u) & 31u);
                    const int trunkHeight = 4 + static_cast<int>((localSeed >> 24u) & 3u);

                    for (int h = 0; h < trunkHeight; ++h) {
                        const int worldZ = trunkBaseZ + h;
                        const int localChunkZ = worldZ / CHUNK_SIZE;
                        const int localBlockZ = worldZ % CHUNK_SIZE;
                        if (localChunkZ < 0 || localChunkZ >= COLUMN_CHUNKS_Z) {
                            continue;
                        }

                        Chunk& chunk = column->chunkAt(localChunkZ);
                        residencyManager_->ensureResident(chunk);

                        UnpackedBlockMaterial m{};
                        m.id = 7;
                        chunk.setBlock(BlockCoord{baseX, baseY, localBlockZ}, m);
                    }
                }

                column->state().setStage(ColumnStage::StructureReady);
                column->state().setStage(ColumnStage::Completed);
                column->state().bumpContentVersion();

                for (int z = 0; z < COLUMN_CHUNKS_Z; ++z) {
                    if (Chunk* chunk = column->tryGetChunk(z); chunk != nullptr) {
                        if (Region* region = regionManager_.tryGetRegion(chunk_to_region(chunk->coord())); region != nullptr) {
                            region->markLodTilesDirtyForChunk(chunk->coord());
                        }
                    }
                }
            }
        }

        result.payload = StructureJobResult{input.columnCoord, success};
        break;
    }

    case VoxelJobType::LodScan: {
        const auto input = std::get<LodScanJobInput>(job.payload);

        uint32_t derivedVersion = 0;
        bool success = false;

        {
            std::scoped_lock lock(worldMutex_);
            Chunk* chunk = regionManager_.tryGetChunk(input.chunkCoord);
            if (chunk != nullptr && residencyManager_->ensureResident(*chunk)) {
                if (input.expectedBlockVersion == 0 || chunk->state().blockDataVersion() == input.expectedBlockVersion) {
                    success = chunk->rebuildLodData();
                    derivedVersion = chunk->state().lodDataVersion();
                }
            }
        }

        result.payload = LodScanJobResult{input.chunkCoord, derivedVersion, success};
        break;
    }

    case VoxelJobType::MeshL0: {
        const auto input = std::get<MeshJobInput>(job.payload);

        MeshData meshData;
        uint32_t derivedVersion = 0;
        bool success = false;

        {
            std::scoped_lock lock(worldMutex_);
            Chunk* chunk = regionManager_.tryGetChunk(input.chunkCoord);
            if (chunk != nullptr && residencyManager_->ensureResident(*chunk)) {
                const uint32_t blockVersion = chunk->state().blockDataVersion();
                if (input.expectedBlockVersion == 0 || input.expectedBlockVersion == blockVersion) {
                    std::vector<Chunk*> neighbors(6, nullptr);
                    const auto offsets = neighborOffsets();

                    for (int i = 0; i < 6; ++i) {
                        const ChunkCoord neighborCoord{
                            input.chunkCoord.x() + offsets[static_cast<std::size_t>(i)].x,
                            input.chunkCoord.y() + offsets[static_cast<std::size_t>(i)].y,
                            input.chunkCoord.z() + offsets[static_cast<std::size_t>(i)].z,
                        };

                        Chunk* neighbor = regionManager_.tryGetChunk(neighborCoord);
                        if (neighbor != nullptr && residencyManager_->ensureResident(*neighbor)) {
                            neighbors[static_cast<std::size_t>(i)] = neighbor;
                        }
                    }

                    ChunkMesher mesher;
                    auto [vertices, indices] = mesher.mesh(*chunk, neighbors);

                    meshData.vertices = std::move(vertices);
                    meshData.indices = std::move(indices);
                    meshData.derivedFromVersion = blockVersion;

                    const BlockCoord origin = chunk_to_block_origin(input.chunkCoord);
                    meshData.minBounds = glm::vec3(
                        static_cast<float>(origin.x()),
                        static_cast<float>(origin.y()),
                        static_cast<float>(origin.z())
                    );
                    meshData.maxBounds = meshData.minBounds + glm::vec3(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE);

                    derivedVersion = blockVersion;
                    success = true;
                }
            }
        }

        result.payload = MeshJobResult{input.chunkCoord, std::move(meshData), derivedVersion, success};
        break;
    }

    case VoxelJobType::LodTile: {
        const auto input = std::get<LodTileJobInput>(job.payload);

        MeshData meshData;
        uint32_t derivedVersion = 0;
        bool success = false;

        {
            std::scoped_lock lock(worldMutex_);
            Region* region = regionManager_.tryGetRegion(input.regionCoord);
            if (region != nullptr && region->tryGetTileState(input.lodLevel, input.tileCoord) != nullptr) {
                const int tileSpan = 1 << (input.lodLevel + 1);
                const int startX = static_cast<int>(input.tileCoord.x) * tileSpan;
                const int startY = static_cast<int>(input.tileCoord.y) * tileSpan;
                const int zSlice = static_cast<int>(input.tileCoord.z);

                for (int y = 0; y < tileSpan; ++y) {
                    for (int x = 0; x < tileSpan; ++x) {
                        Column* column = region->tryGetColumn(startX + x, startY + y);
                        if (column == nullptr) {
                            continue;
                        }

                        const Chunk* chunk = column->tryGetChunk(zSlice);
                        if (chunk == nullptr) {
                            continue;
                        }

                        derivedVersion = std::max(derivedVersion, chunk->state().lodDataVersion());
                    }
                }

                meshData.derivedFromVersion = derivedVersion;
                success = true;
            }
        }

        result.payload = LodTileJobResult{input.regionCoord, input.lodLevel, input.tileCoord, std::move(meshData), derivedVersion, success};
        break;
    }

    case VoxelJobType::CompressChunk: {
        const auto input = std::get<CompressChunkJobInput>(job.payload);

        bool success = false;
        {
            std::scoped_lock lock(worldMutex_);
            if (Chunk* chunk = regionManager_.tryGetChunk(input.chunkCoord); chunk != nullptr) {
                success = residencyManager_->compressChunk(*chunk);
            }
        }

        result.payload = CompressChunkJobResult{input.chunkCoord, success};
        break;
    }

    case VoxelJobType::UncompressChunk: {
        const auto input = std::get<UncompressChunkJobInput>(job.payload);

        bool success = false;
        {
            std::scoped_lock lock(worldMutex_);
            if (Chunk* chunk = regionManager_.tryGetChunk(input.chunkCoord); chunk != nullptr) {
                success = residencyManager_->uncompressChunk(*chunk);
            }
        }

        result.payload = UncompressChunkJobResult{input.chunkCoord, success};
        break;
    }
    }

    return result;
}

void World::applyTerrainCompletion(ColumnCoord columnCoord, bool success) {
    std::scoped_lock lock(worldMutex_);
    terrainScheduled_.erase(columnCoord);

    if (!success) {
        return;
    }

    if (Column* column = regionManager_.tryGetColumn(columnCoord); column != nullptr) {
        if (column->state().canRunStructureGeneration() && structureScheduled_.insert(columnCoord).second) {
            VoxelJob job;
            job.type = VoxelJobType::StructureGeneration;
            job.priority = JobPriority::Medium;
            job.payload = StructureGenerationJobInput{columnCoord};
            scheduler_->enqueue(std::move(job));
        }
    }
}

void World::applyStructureCompletion(ColumnCoord columnCoord, bool success) {
    std::scoped_lock lock(worldMutex_);
    structureScheduled_.erase(columnCoord);

    if (!success) {
        return;
    }

    if (Column* column = regionManager_.tryGetColumn(columnCoord); column != nullptr) {
        column->state().setStage(ColumnStage::Completed);
    }
}

void World::applyLodScanCompletion(ChunkCoord chunkCoord, uint32_t derivedVersion, bool success) {
    std::scoped_lock lock(worldMutex_);
    lodScanScheduled_.erase(chunkCoord);

    if (!success) {
        return;
    }

    if (Chunk* chunk = regionManager_.tryGetChunk(chunkCoord); chunk != nullptr) {
        if (chunk->state().blockDataVersion() == derivedVersion) {
            chunk->state().setLodDataVersion(derivedVersion);
            chunk->state().clearDirty(ChunkDirtyFlags::NeedsLodScan);
        }
    }
}

void World::applyMeshCompletion(ChunkCoord chunkCoord, MeshData meshData, uint32_t derivedVersion, bool success) {
    std::scoped_lock lock(worldMutex_);
    meshScheduled_.erase(chunkCoord);

    if (!success) {
        return;
    }

    Chunk* chunk = regionManager_.tryGetChunk(chunkCoord);
    if (chunk == nullptr) {
        return;
    }

    if (chunk->state().blockDataVersion() != derivedVersion) {
        return;
    }

    MeshHandle handle = chunk->meshHandleL0();
    if (handle.isValid()) {
        handle = meshHandles_.updateOrCreate(handle, std::move(meshData));
    } else {
        handle = meshHandles_.create(std::move(meshData));
    }

    chunk->setMeshHandleL0(handle, derivedVersion);
}

void World::applyLodTileCompletion(
    RegionCoord regionCoord,
    int lodLevel,
    RegionLodTileCoord tileCoord,
    MeshData meshData,
    uint32_t derivedVersion,
    bool success
) {
    std::scoped_lock lock(worldMutex_);
    lodTileScheduled_.erase(tileScheduleKey(regionCoord, lodLevel, tileCoord));

    if (!success) {
        return;
    }

    Region* region = regionManager_.tryGetRegion(regionCoord);
    if (region == nullptr) {
        return;
    }

    RegionLodTileState* tile = region->tryGetTileState(lodLevel, tileCoord);
    if (tile == nullptr) {
        return;
    }

    MeshHandle handle = tile->meshHandle;
    if (handle.isValid()) {
        handle = meshHandles_.updateOrCreate(handle, std::move(meshData));
    } else {
        handle = meshHandles_.create(std::move(meshData));
    }

    region->markTileClean(lodLevel, tileCoord, handle, derivedVersion);
}
