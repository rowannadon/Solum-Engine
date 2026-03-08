#include "solum_engine/voxel/World.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <shared_mutex>
#include <utility>
#include <vector>

#include "solum_engine/resources/Constants.h"
#include "solum_engine/voxel/Column.h"
#include "solum_engine/voxel/MaterialLightProperties.h"
#include "solum_engine/voxel/Region.h"

namespace {
int32_t distanceSqToCenter(const ColumnCoord& coord, const ColumnCoord& center) {
    const int64_t dx = static_cast<int64_t>(coord.v.x) - static_cast<int64_t>(center.v.x);
    const int64_t dy = static_cast<int64_t>(coord.v.y) - static_cast<int64_t>(center.v.y);
    const int64_t distanceSq = (dx * dx) + (dy * dy);
    if (distanceSq > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        return std::numeric_limits<int32_t>::max();
    }
    return static_cast<int32_t>(distanceSq);
}

constexpr int kChunkExtent = cfg::CHUNK_SIZE;
constexpr int kChunkArea = kChunkExtent * kChunkExtent;
constexpr std::array<glm::ivec2, 4> kHorizontalOffsets = {
    glm::ivec2{1, 0},
    glm::ivec2{-1, 0},
    glm::ivec2{0, 1},
    glm::ivec2{0, -1},
};
constexpr std::array<glm::ivec3, 6> kCardinalOffsets = {
    glm::ivec3{1, 0, 0},
    glm::ivec3{-1, 0, 0},
    glm::ivec3{0, 1, 0},
    glm::ivec3{0, -1, 0},
    glm::ivec3{0, 0, 1},
    glm::ivec3{0, 0, -1},
};

constexpr int chunkLocalIndex(int x, int y, int z) {
    return (z * kChunkArea) + (y * kChunkExtent) + x;
}

uint8_t attenuateLight(uint8_t light, uint8_t loss) {
    if (light == 0u || loss == MaterialLightProperties::kOpaqueLightLoss || loss >= light) {
        return 0u;
    }
    return static_cast<uint8_t>(light - loss);
}

constexpr uint8_t kLightingDirtyTopology = 1u << 0u;
constexpr uint8_t kLightingDirtyBoundary = 1u << 1u;

uint64_t hashMix(uint64_t seed, uint64_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6u) + (seed >> 2u);
    return seed;
}
}  // namespace

struct World::ChunkPropagationResult {
    ChunkCoord coord{};
    uint64_t targetEpoch = 0u;
    bool highPriority = false;
    bool propagated = false;
    bool lightChanged = false;
    uint64_t solveSignature = 0u;
};

void World::enqueueChunkPropagationIfReadyLocked(const ChunkCoord& coord, bool highPriority) {
    if (coord.v.z < 0 || coord.v.z >= cfg::COLUMN_HEIGHT) {
        return;
    }
    if (!isChunkKnownLocked(coord)) {
        return;
    }
    LightingChunkState& state = lightingChunkStates_[coord];
    if (state.topologyEpoch == 0u) {
        state.topologyEpoch = 1u;
    }
    const uint64_t targetEpoch = state.topologyEpoch;

    if (state.lightingEpoch >= targetEpoch &&
        state.dirtyFlags == 0u) {
        return;
    }

    if (state.inFlightEpoch >= targetEpoch) {
        return;
    }

    if (state.queuedEpoch >= targetEpoch && !highPriority) {
        return;
    }

    if (!canPropagateChunkLocked(coord)) {
        return;
    }

    state.queuedEpoch = targetEpoch;
    const ChunkPropagationTask task{coord, targetEpoch, highPriority};
    if (highPriority) {
        queuedChunkPropagationJobs_.push_front(task);
    } else {
        queuedChunkPropagationJobs_.push_back(task);
    }
}

void World::collectChunkPropagationJobsLocked(std::vector<ChunkPropagationTask>& outChunks) {
    std::size_t retryBudget = queuedChunkPropagationJobs_.size();
    while (pendingChunkPropagationJobs_.size() < maxInFlightChunkPropagationJobs_ &&
           !queuedChunkPropagationJobs_.empty() &&
           retryBudget > 0u) {
        --retryBudget;
        const ChunkPropagationTask task = queuedChunkPropagationJobs_.front();
        queuedChunkPropagationJobs_.pop_front();
        const ChunkCoord coord = task.coord;
        if (coord.v.z < 0 || coord.v.z >= cfg::COLUMN_HEIGHT) {
            continue;
        }
        if (!isChunkKnownLocked(coord)) {
            continue;
        }

        auto stateIt = lightingChunkStates_.find(coord);
        if (stateIt == lightingChunkStates_.end()) {
            continue;
        }
        LightingChunkState& state = stateIt->second;
        const uint64_t latestEpoch = state.topologyEpoch;
        const uint64_t targetEpoch = std::max(task.targetEpoch, latestEpoch);

        if (state.lightingEpoch >= targetEpoch && state.dirtyFlags == 0u) {
            state.queuedEpoch = 0u;
            continue;
        }
        if (!canPropagateChunkLocked(coord)) {
            queuedChunkPropagationJobs_.push_back(ChunkPropagationTask{
                coord,
                targetEpoch,
                task.highPriority
            });
            continue;
        }

        if (state.inFlightEpoch >= targetEpoch) {
            continue;
        }

        state.queuedEpoch = 0u;
        state.inFlightEpoch = targetEpoch;
        pendingChunkPropagationJobs_[coord] = targetEpoch;
        outChunks.push_back(ChunkPropagationTask{coord, targetEpoch, task.highPriority});
    }
}

void World::dispatchChunkPropagationJobs(std::vector<ChunkPropagationTask>&& chunksToSchedule) {
    for (const ChunkPropagationTask& task : chunksToSchedule) {
        const ChunkCoord coord = task.coord;
        const uint64_t targetEpoch = task.targetEpoch;
        const int32_t distanceSq = hasLastScheduledCenter_
            ? distanceSqToCenter(chunk_to_column(coord), lastScheduledCenter_)
            : 0;
        const jobsystem::Priority priority = task.highPriority
            ? jobsystem::Priority::Critical
            : priorityFromDistanceSq(distanceSq);

        try {
            chunkPropagationJobs_.schedule(
                priority,
                [this, coord, targetEpoch, task]() -> ChunkPropagationResult {
                    bool lightChanged = false;
                    uint64_t solveSignature = 0u;
                    const bool propagated = propagateChunkLighting(
                        coord,
                        targetEpoch,
                        &lightChanged,
                        &solveSignature
                    );
                    return ChunkPropagationResult{
                        coord,
                        targetEpoch,
                        task.highPriority,
                        propagated,
                        lightChanged,
                        solveSignature
                    };
                },
                [this, coord, targetEpoch, task](jobsystem::JobResult<ChunkPropagationResult>&& result) {
                    {
                        std::unique_lock<std::shared_mutex> lock(worldMutex_);
                        auto pendingIt = pendingChunkPropagationJobs_.find(coord);
                        if (pendingIt != pendingChunkPropagationJobs_.end() &&
                            pendingIt->second == targetEpoch) {
                            pendingChunkPropagationJobs_.erase(pendingIt);
                        }

                        auto stateIt = lightingChunkStates_.find(coord);
                        if (stateIt != lightingChunkStates_.end() &&
                            stateIt->second.inFlightEpoch == targetEpoch) {
                            stateIt->second.inFlightEpoch = 0u;
                        }

                        bool propagated = false;
                        if (result.success()) {
                            ChunkPropagationResult propagationResult = std::move(result).value();
                            propagated = propagationResult.propagated;
                        }

                        if (!propagated && !shuttingDown_.load(std::memory_order_acquire)) {
                            auto stateRetryIt = lightingChunkStates_.find(coord);
                            if (stateRetryIt != lightingChunkStates_.end() &&
                                stateRetryIt->second.lightingEpoch < stateRetryIt->second.topologyEpoch) {
                                enqueueChunkPropagationIfReadyLocked(coord, task.highPriority);
                            }
                        }
                    }
                    pumpChunkPropagationQueue();
                }
            );
        } catch (const std::exception&) {
            std::unique_lock<std::shared_mutex> lock(worldMutex_);
            auto pendingIt = pendingChunkPropagationJobs_.find(coord);
            if (pendingIt != pendingChunkPropagationJobs_.end() &&
                pendingIt->second == targetEpoch) {
                pendingChunkPropagationJobs_.erase(pendingIt);
            }
            auto stateIt = lightingChunkStates_.find(coord);
            if (stateIt != lightingChunkStates_.end() &&
                stateIt->second.inFlightEpoch == targetEpoch) {
                stateIt->second.inFlightEpoch = 0u;
            }
            if (!shuttingDown_.load(std::memory_order_acquire)) {
                enqueueChunkPropagationIfReadyLocked(coord, task.highPriority);
            }
        }
    }
}

void World::pumpChunkPropagationQueue() {
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return;
    }

    std::vector<ChunkPropagationTask> chunksToSchedule;
    {
        std::unique_lock<std::shared_mutex> lock(worldMutex_);
        collectChunkPropagationJobsLocked(chunksToSchedule);
    }
    dispatchChunkPropagationJobs(std::move(chunksToSchedule));
}

void World::bumpChunkTopologyEpochLocked(const ChunkCoord& coord,
                                         bool highPriority,
                                         uint8_t dirtyFlags) {
    if (!isChunkKnownLocked(coord)) {
        return;
    }
    LightingChunkState& state = lightingChunkStates_[coord];
    if (state.topologyEpoch == 0u) {
        state.topologyEpoch = 1u;
    } else {
        ++state.topologyEpoch;
    }
    state.dirtyFlags |= dirtyFlags;
    enqueueChunkPropagationIfReadyLocked(coord, highPriority);
}

World::LightingChunkState* World::tryGetLightingChunkStateLocked(const ChunkCoord& coord) {
    auto it = lightingChunkStates_.find(coord);
    if (it == lightingChunkStates_.end()) {
        return nullptr;
    }
    return &it->second;
}

const World::LightingChunkState* World::tryGetLightingChunkStateLocked(const ChunkCoord& coord) const {
    auto it = lightingChunkStates_.find(coord);
    if (it == lightingChunkStates_.end()) {
        return nullptr;
    }
    return &it->second;
}

bool World::isChunkKnownLocked(const ChunkCoord& coord) const {
    if (coord.v.z < 0 || coord.v.z >= cfg::COLUMN_HEIGHT) {
        return false;
    }
    const ColumnCoord columnCoord = chunk_to_column(coord);
    if (!isColumnSkycastCompleteLocked(columnCoord)) {
        return false;
    }
    return tryGetSkycastColumnLocked(columnCoord) != nullptr;
}

uint64_t World::computeChunkSolveSignatureLocked(const ChunkCoord& coord) const {
    if (!isChunkKnownLocked(coord)) {
        return 0u;
    }

    const ColumnCoord columnCoord = chunk_to_column(coord);
    const uint8_t chunkZ = static_cast<uint8_t>(coord.v.z);
    const Column* centerColumn = tryGetSkycastColumnLocked(columnCoord);
    if (centerColumn == nullptr) {
        return 0u;
    }
    const Chunk& centerChunk = centerColumn->getChunk(chunkZ);

    uint64_t signature = 0xcbf29ce484222325ull;
    for (int z = 0; z < kChunkExtent; ++z) {
        for (int y = 0; y < kChunkExtent; ++y) {
            for (int x = 0; x < kChunkExtent; ++x) {
                const BlockMaterial block = centerChunk.getBlock(
                    static_cast<uint8_t>(x),
                    static_cast<uint8_t>(y),
                    static_cast<uint8_t>(z)
                );
                const uint64_t packedLight = static_cast<uint64_t>(centerChunk.getPackedLight(
                    static_cast<uint8_t>(x),
                    static_cast<uint8_t>(y),
                    static_cast<uint8_t>(z)
                ));
                signature = hashMix(signature, static_cast<uint64_t>(block.data));
                signature = hashMix(signature, packedLight);
            }
        }
    }

    const Column* plusXColumn = tryGetSkycastColumnLocked(ColumnCoord{columnCoord.v.x + 1, columnCoord.v.y});
    const Column* minusXColumn = tryGetSkycastColumnLocked(ColumnCoord{columnCoord.v.x - 1, columnCoord.v.y});
    const Column* plusYColumn = tryGetSkycastColumnLocked(ColumnCoord{columnCoord.v.x, columnCoord.v.y + 1});
    const Column* minusYColumn = tryGetSkycastColumnLocked(ColumnCoord{columnCoord.v.x, columnCoord.v.y - 1});
    const Chunk* plusXChunk = (plusXColumn != nullptr) ? &plusXColumn->getChunk(chunkZ) : nullptr;
    const Chunk* minusXChunk = (minusXColumn != nullptr) ? &minusXColumn->getChunk(chunkZ) : nullptr;
    const Chunk* plusYChunk = (plusYColumn != nullptr) ? &plusYColumn->getChunk(chunkZ) : nullptr;
    const Chunk* minusYChunk = (minusYColumn != nullptr) ? &minusYColumn->getChunk(chunkZ) : nullptr;
    const Chunk* plusZChunk =
        (chunkZ + 1u < static_cast<uint8_t>(cfg::COLUMN_HEIGHT))
        ? &centerColumn->getChunk(static_cast<uint8_t>(chunkZ + 1u))
        : nullptr;
    const Chunk* minusZChunk =
        (chunkZ > 0u)
        ? &centerColumn->getChunk(static_cast<uint8_t>(chunkZ - 1u))
        : nullptr;

    auto hashBoundary = [&signature](const Chunk* chunk,
                                     uint8_t fixedAxis,
                                     bool axisX,
                                     bool axisY,
                                     bool axisZ) {
        if (chunk == nullptr) {
            signature = hashMix(signature, 0xDEADBEEFull);
            return;
        }
        for (int b = 0; b < kChunkExtent; ++b) {
            for (int a = 0; a < kChunkExtent; ++a) {
                const uint8_t x = axisX ? fixedAxis : static_cast<uint8_t>(a);
                const uint8_t y = axisY ? fixedAxis : static_cast<uint8_t>(axisX ? a : b);
                const uint8_t z = axisZ ? fixedAxis : static_cast<uint8_t>(b);
                signature = hashMix(signature, static_cast<uint64_t>(chunk->getPackedLight(x, y, z)));
            }
        }
    };

    hashBoundary(plusXChunk, 0u, true, false, false);
    hashBoundary(minusXChunk, static_cast<uint8_t>(kChunkExtent - 1), true, false, false);
    hashBoundary(plusYChunk, 0u, false, true, false);
    hashBoundary(minusYChunk, static_cast<uint8_t>(kChunkExtent - 1), false, true, false);
    hashBoundary(plusZChunk, 0u, false, false, true);
    hashBoundary(minusZChunk, static_cast<uint8_t>(kChunkExtent - 1), false, false, true);

    return signature;
}

bool World::tryApplyImmediateLightingAround(const ChunkCoord& centerChunk) {
    std::array<ChunkCoord, 7> immediateChunks = {
        centerChunk,
        ChunkCoord{centerChunk.v.x + 1, centerChunk.v.y, centerChunk.v.z},
        ChunkCoord{centerChunk.v.x - 1, centerChunk.v.y, centerChunk.v.z},
        ChunkCoord{centerChunk.v.x, centerChunk.v.y + 1, centerChunk.v.z},
        ChunkCoord{centerChunk.v.x, centerChunk.v.y - 1, centerChunk.v.z},
        ChunkCoord{centerChunk.v.x, centerChunk.v.y, centerChunk.v.z + 1},
        ChunkCoord{centerChunk.v.x, centerChunk.v.y, centerChunk.v.z - 1}
    };

    bool anyApplied = false;
    for (const ChunkCoord& chunkCoord : immediateChunks) {
        uint64_t targetEpoch = 0u;
        {
            std::shared_lock<std::shared_mutex> lock(worldMutex_);
            const LightingChunkState* state = tryGetLightingChunkStateLocked(chunkCoord);
            if (state == nullptr) {
                continue;
            }
            targetEpoch = state->topologyEpoch;
        }

        bool lightChanged = false;
        if (propagateChunkLighting(chunkCoord, targetEpoch, &lightChanged, nullptr)) {
            anyApplied = true;
        }
    }

    return anyApplied;
}

bool World::propagateChunkLighting(const ChunkCoord& coord,
                                   uint64_t targetEpoch,
                                   bool* outLightChanged,
                                   uint64_t* outSolveSignature) {
    if (outLightChanged != nullptr) {
        *outLightChanged = false;
    }
    if (outSolveSignature != nullptr) {
        *outSolveSignature = 0u;
    }

    if (coord.v.z < 0 || coord.v.z >= cfg::COLUMN_HEIGHT || targetEpoch == 0u) {
        return false;
    }

    struct ChunkPropagationSnapshot {
        std::array<uint8_t, Chunk::VOLUME> sky{};
        std::array<uint8_t, Chunk::VOLUME> blockLight{};
        std::array<uint8_t, Chunk::VOLUME> emissive{};
        std::array<uint8_t, Chunk::VOLUME> oldPackedLight{};
        std::array<uint8_t, Chunk::VOLUME> blockLightLoss{};
        std::array<uint8_t, Chunk::VOLUME> skyVerticalLoss{};
        std::array<uint8_t, Chunk::VOLUME> blocksLightMask{};
    };

    const ColumnCoord columnCoord = chunk_to_column(coord);
    const uint8_t chunkZ = static_cast<uint8_t>(coord.v.z);
    ChunkPropagationSnapshot snapshot{};
    std::vector<int> skyQueue;
    std::vector<int> blockQueue;
    skyQueue.reserve(Chunk::VOLUME);
    blockQueue.reserve(Chunk::VOLUME);
    uint64_t snapshotSignature = 0u;

    {
        std::shared_lock<std::shared_mutex> lock(worldMutex_);
        if (!canPropagateChunkLocked(coord) || !isChunkKnownLocked(coord)) {
            return false;
        }

        const LightingChunkState* state = tryGetLightingChunkStateLocked(coord);
        if (state == nullptr || state->topologyEpoch != targetEpoch) {
            return false;
        }

        snapshotSignature = computeChunkSolveSignatureLocked(coord);

        const Column* centerColumn = tryGetSkycastColumnLocked(columnCoord);
        if (centerColumn == nullptr) {
            return false;
        }
        const Chunk& centerChunk = centerColumn->getChunk(chunkZ);

        for (int z = 0; z < kChunkExtent; ++z) {
            for (int y = 0; y < kChunkExtent; ++y) {
                for (int x = 0; x < kChunkExtent; ++x) {
                    const int index = chunkLocalIndex(x, y, z);
                    const BlockMaterial block = centerChunk.getBlock(
                        static_cast<uint8_t>(x),
                        static_cast<uint8_t>(y),
                        static_cast<uint8_t>(z)
                    );
                    const uint16_t materialId = block.unpack().id;
                    const uint8_t packedLight = centerChunk.getPackedLight(
                        static_cast<uint8_t>(x),
                        static_cast<uint8_t>(y),
                        static_cast<uint8_t>(z)
                    );
                    const bool blocksLight = MaterialLightProperties::blocksLight(materialId);
                    const uint8_t emissive = MaterialLightProperties::emissiveLight(materialId);
                    snapshot.blocksLightMask[static_cast<size_t>(index)] = blocksLight ? 1u : 0u;
                    snapshot.blockLightLoss[static_cast<size_t>(index)] = MaterialLightProperties::blockLightStepLoss(materialId);
                    snapshot.skyVerticalLoss[static_cast<size_t>(index)] = MaterialLightProperties::skyLightVerticalLoss(materialId);
                    snapshot.sky[static_cast<size_t>(index)] = 0u;
                    snapshot.oldPackedLight[static_cast<size_t>(index)] = packedLight;
                    snapshot.emissive[static_cast<size_t>(index)] = emissive;
                    snapshot.blockLight[static_cast<size_t>(index)] = emissive;
                    if (emissive > 0u) {
                        blockQueue.push_back(index);
                    }
                }
            }
        }

        const Column* plusXColumn = tryGetSkycastColumnLocked(ColumnCoord{columnCoord.v.x + 1, columnCoord.v.y});
        const Column* minusXColumn = tryGetSkycastColumnLocked(ColumnCoord{columnCoord.v.x - 1, columnCoord.v.y});
        const Column* plusYColumn = tryGetSkycastColumnLocked(ColumnCoord{columnCoord.v.x, columnCoord.v.y + 1});
        const Column* minusYColumn = tryGetSkycastColumnLocked(ColumnCoord{columnCoord.v.x, columnCoord.v.y - 1});
        const Chunk* plusXChunk = (plusXColumn != nullptr) ? &plusXColumn->getChunk(chunkZ) : nullptr;
        const Chunk* minusXChunk = (minusXColumn != nullptr) ? &minusXColumn->getChunk(chunkZ) : nullptr;
        const Chunk* plusYChunk = (plusYColumn != nullptr) ? &plusYColumn->getChunk(chunkZ) : nullptr;
        const Chunk* minusYChunk = (minusYColumn != nullptr) ? &minusYColumn->getChunk(chunkZ) : nullptr;
        const Chunk* plusZChunk =
            (chunkZ + 1u < static_cast<uint8_t>(cfg::COLUMN_HEIGHT))
            ? &centerColumn->getChunk(static_cast<uint8_t>(chunkZ + 1u))
            : nullptr;
        const Chunk* minusZChunk =
            (chunkZ > 0u)
            ? &centerColumn->getChunk(static_cast<uint8_t>(chunkZ - 1u))
            : nullptr;

        auto seedSkyFromNeighbor = [&](int lx,
                                       int ly,
                                       int lz,
                                       const Chunk* neighbor,
                                       int nx,
                                       int ny,
                                       int nz,
                                       uint8_t loss) {
            const int localIndex = chunkLocalIndex(lx, ly, lz);
            if (snapshot.blocksLightMask[static_cast<size_t>(localIndex)] != 0u) {
                return;
            }
            if (neighbor == nullptr) {
                return;
            }
            const uint8_t neighborSky = Chunk::unpackSkyLight(neighbor->getPackedLight(
                static_cast<uint8_t>(nx),
                static_cast<uint8_t>(ny),
                static_cast<uint8_t>(nz)
            ));
            const uint8_t candidate = attenuateLight(neighborSky, loss);
            if (candidate == 0u) {
                return;
            }
            uint8_t& current = snapshot.sky[static_cast<size_t>(localIndex)];
            if (candidate <= current) {
                return;
            }
            current = candidate;
            skyQueue.push_back(localIndex);
        };

        auto seedBlockFromNeighbor = [&](int lx,
                                         int ly,
                                         int lz,
                                         const Chunk* neighbor,
                                         int nx,
                                         int ny,
                                         int nz) {
            const int localIndex = chunkLocalIndex(lx, ly, lz);
            if (snapshot.blocksLightMask[static_cast<size_t>(localIndex)] != 0u &&
                snapshot.emissive[static_cast<size_t>(localIndex)] == 0u) {
                return;
            }
            if (neighbor == nullptr) {
                return;
            }
            const uint8_t neighborBlock = Chunk::unpackBlockLight(neighbor->getPackedLight(
                static_cast<uint8_t>(nx),
                static_cast<uint8_t>(ny),
                static_cast<uint8_t>(nz)
            ));
            const uint8_t candidate = attenuateLight(
                neighborBlock,
                snapshot.blockLightLoss[static_cast<size_t>(localIndex)]
            );
            if (candidate == 0u) {
                return;
            }
            uint8_t& current = snapshot.blockLight[static_cast<size_t>(localIndex)];
            if (candidate <= current) {
                return;
            }
            current = candidate;
            blockQueue.push_back(localIndex);
        };

        if (chunkZ == static_cast<uint8_t>(cfg::COLUMN_HEIGHT - 1)) {
            for (int y = 0; y < kChunkExtent; ++y) {
                for (int x = 0; x < kChunkExtent; ++x) {
                    const int localIndex = chunkLocalIndex(x, y, kChunkExtent - 1);
                    if (snapshot.blocksLightMask[static_cast<size_t>(localIndex)] != 0u) {
                        continue;
                    }
                    const uint8_t candidate = attenuateLight(
                        15u,
                        snapshot.skyVerticalLoss[static_cast<size_t>(localIndex)]
                    );
                    if (candidate == 0u) {
                        continue;
                    }
                    uint8_t& current = snapshot.sky[static_cast<size_t>(localIndex)];
                    if (candidate <= current) {
                        continue;
                    }
                    current = candidate;
                    skyQueue.push_back(localIndex);
                }
            }
        }

        for (int z = 0; z < kChunkExtent; ++z) {
            for (int y = 0; y < kChunkExtent; ++y) {
                seedSkyFromNeighbor(
                    kChunkExtent - 1, y, z,
                    plusXChunk, 0, y, z,
                    snapshot.blockLightLoss[static_cast<size_t>(chunkLocalIndex(kChunkExtent - 1, y, z))]
                );
                seedSkyFromNeighbor(
                    0, y, z,
                    minusXChunk, kChunkExtent - 1, y, z,
                    snapshot.blockLightLoss[static_cast<size_t>(chunkLocalIndex(0, y, z))]
                );
                seedBlockFromNeighbor(kChunkExtent - 1, y, z, plusXChunk, 0, y, z);
                seedBlockFromNeighbor(0, y, z, minusXChunk, kChunkExtent - 1, y, z);
            }
        }
        for (int z = 0; z < kChunkExtent; ++z) {
            for (int x = 0; x < kChunkExtent; ++x) {
                seedSkyFromNeighbor(
                    x, kChunkExtent - 1, z,
                    plusYChunk, x, 0, z,
                    snapshot.blockLightLoss[static_cast<size_t>(chunkLocalIndex(x, kChunkExtent - 1, z))]
                );
                seedSkyFromNeighbor(
                    x, 0, z,
                    minusYChunk, x, kChunkExtent - 1, z,
                    snapshot.blockLightLoss[static_cast<size_t>(chunkLocalIndex(x, 0, z))]
                );
                seedBlockFromNeighbor(x, kChunkExtent - 1, z, plusYChunk, x, 0, z);
                seedBlockFromNeighbor(x, 0, z, minusYChunk, x, kChunkExtent - 1, z);
            }
        }
        for (int y = 0; y < kChunkExtent; ++y) {
            for (int x = 0; x < kChunkExtent; ++x) {
                seedSkyFromNeighbor(
                    x, y, kChunkExtent - 1,
                    plusZChunk, x, y, 0,
                    snapshot.skyVerticalLoss[static_cast<size_t>(chunkLocalIndex(x, y, kChunkExtent - 1))]
                );
                const uint8_t upwardLoss = static_cast<uint8_t>(std::min<uint16_t>(
                    15u,
                    static_cast<uint16_t>(snapshot.blockLightLoss[static_cast<size_t>(chunkLocalIndex(x, y, 0))]) + 1u
                ));
                seedSkyFromNeighbor(
                    x, y, 0,
                    minusZChunk, x, y, kChunkExtent - 1,
                    upwardLoss
                );
                seedBlockFromNeighbor(x, y, kChunkExtent - 1, plusZChunk, x, y, 0);
                seedBlockFromNeighbor(x, y, 0, minusZChunk, x, y, kChunkExtent - 1);
            }
        }
    }

    std::size_t skyQueueHead = 0u;
    while (skyQueueHead < skyQueue.size()) {
        const int index = skyQueue[skyQueueHead++];
        const uint8_t current = snapshot.sky[static_cast<size_t>(index)];
        if (current == 0u) {
            continue;
        }

        const int x = index % kChunkExtent;
        const int y = (index / kChunkExtent) % kChunkExtent;
        const int z = index / kChunkArea;

        for (const glm::ivec3& offset : kCardinalOffsets) {
            const int nx = x + offset.x;
            const int ny = y + offset.y;
            const int nz = z + offset.z;
            if (nx < 0 || ny < 0 || nz < 0 ||
                nx >= kChunkExtent || ny >= kChunkExtent || nz >= kChunkExtent) {
                continue;
            }

            const int neighborIndex = chunkLocalIndex(nx, ny, nz);
            if (snapshot.blocksLightMask[static_cast<size_t>(neighborIndex)] != 0u) {
                continue;
            }

            uint8_t loss = snapshot.blockLightLoss[static_cast<size_t>(neighborIndex)];
            if (offset.z < 0) {
                loss = snapshot.skyVerticalLoss[static_cast<size_t>(neighborIndex)];
            } else if (offset.z > 0) {
                const uint16_t upwardLossBase = static_cast<uint16_t>(
                    snapshot.blockLightLoss[static_cast<size_t>(neighborIndex)]
                );
                loss = static_cast<uint8_t>(std::min<uint16_t>(15u, upwardLossBase + 1u));
            }

            const uint8_t propagated = attenuateLight(current, loss);
            if (propagated == 0u) {
                continue;
            }

            uint8_t& neighborSky = snapshot.sky[static_cast<size_t>(neighborIndex)];
            if (propagated <= neighborSky) {
                continue;
            }
            neighborSky = propagated;
            skyQueue.push_back(neighborIndex);
        }
    }

    std::size_t blockQueueHead = 0u;
    while (blockQueueHead < blockQueue.size()) {
        const int index = blockQueue[blockQueueHead++];
        const uint8_t current = snapshot.blockLight[static_cast<size_t>(index)];
        if (current == 0u) {
            continue;
        }

        const int x = index % kChunkExtent;
        const int y = (index / kChunkExtent) % kChunkExtent;
        const int z = index / kChunkArea;

        for (const glm::ivec3& offset : kCardinalOffsets) {
            const int nx = x + offset.x;
            const int ny = y + offset.y;
            const int nz = z + offset.z;
            if (nx < 0 || ny < 0 || nz < 0 ||
                nx >= kChunkExtent || ny >= kChunkExtent || nz >= kChunkExtent) {
                continue;
            }

            const int neighborIndex = chunkLocalIndex(nx, ny, nz);
            if (snapshot.blocksLightMask[static_cast<size_t>(neighborIndex)] != 0u &&
                snapshot.emissive[static_cast<size_t>(neighborIndex)] == 0u) {
                continue;
            }

            const uint8_t propagated = attenuateLight(
                current,
                snapshot.blockLightLoss[static_cast<size_t>(neighborIndex)]
            );
            if (propagated == 0u) {
                continue;
            }
            uint8_t& neighborBlock = snapshot.blockLight[static_cast<size_t>(neighborIndex)];
            if (propagated <= neighborBlock) {
                continue;
            }

            neighborBlock = propagated;
            blockQueue.push_back(neighborIndex);
        }
    }

    std::array<uint8_t, Chunk::VOLUME> propagatedPackedLight{};
    for (int z = 0; z < kChunkExtent; ++z) {
        for (int y = 0; y < kChunkExtent; ++y) {
            for (int x = 0; x < kChunkExtent; ++x) {
                const int index = chunkLocalIndex(x, y, z);
                const bool blocksLight = snapshot.blocksLightMask[static_cast<size_t>(index)] != 0u;
                uint8_t sky = snapshot.sky[static_cast<size_t>(index)];
                uint8_t block = snapshot.blockLight[static_cast<size_t>(index)];
                if (blocksLight) {
                    sky = 0u;
                    if (snapshot.emissive[static_cast<size_t>(index)] == 0u) {
                        block = 0u;
                    }
                }
                propagatedPackedLight[static_cast<size_t>(index)] = Chunk::packLight(sky, block);
            }
        }
    }

    {
        std::unique_lock<std::shared_mutex> lock(worldMutex_);
        if (!canPropagateChunkLocked(coord) || !isChunkKnownLocked(coord)) {
            return false;
        }

        LightingChunkState* state = tryGetLightingChunkStateLocked(coord);
        if (state == nullptr || state->topologyEpoch != targetEpoch) {
            return false;
        }

        const uint64_t commitSignature = computeChunkSolveSignatureLocked(coord);
        if (commitSignature != snapshotSignature) {
            state->dirtyFlags |= kLightingDirtyBoundary;
            return false;
        }

        Column* centerColumn = tryGetSkycastColumnLocked(columnCoord);
        if (centerColumn == nullptr) {
            return false;
        }
        Chunk& centerChunk = centerColumn->getChunk(chunkZ);

        bool chunkLightChanged = false;
        bool plusXChanged = false;
        bool minusXChanged = false;
        bool plusYChanged = false;
        bool minusYChanged = false;
        bool plusZChanged = false;
        bool minusZChanged = false;

        for (int z = 0; z < kChunkExtent; ++z) {
            for (int y = 0; y < kChunkExtent; ++y) {
                for (int x = 0; x < kChunkExtent; ++x) {
                    const int index = chunkLocalIndex(x, y, z);
                    const uint8_t oldPacked = snapshot.oldPackedLight[static_cast<size_t>(index)];
                    const uint8_t newPacked = propagatedPackedLight[static_cast<size_t>(index)];
                    if (oldPacked == newPacked) {
                        continue;
                    }

                    chunkLightChanged = true;
                    if (x == kChunkExtent - 1) {
                        plusXChanged = true;
                    }
                    if (x == 0) {
                        minusXChanged = true;
                    }
                    if (y == kChunkExtent - 1) {
                        plusYChanged = true;
                    }
                    if (y == 0) {
                        minusYChanged = true;
                    }
                    if (z == kChunkExtent - 1) {
                        plusZChanged = true;
                    }
                    if (z == 0) {
                        minusZChanged = true;
                    }
                }
            }
        }

        centerChunk.setPackedLightVolume(propagatedPackedLight);
        state->lightingEpoch = targetEpoch;
        state->lastSolveSignature = snapshotSignature;
        state->dirtyFlags = 0u;

        const bool wasGenerated = generatedColumns_.find(columnCoord) != generatedColumns_.end();
        if (chunkLightChanged && wasGenerated) {
            lightingChangedChunkHistory_.push_back(coord);
            lightingChunkRevision_.fetch_add(1, std::memory_order_release);
        }

        auto queueDependent = [&](const ChunkCoord& dependentCoord) {
            bumpChunkTopologyEpochLocked(dependentCoord, true, kLightingDirtyBoundary);
        };

        if (plusXChanged) {
            queueDependent(ChunkCoord{coord.v.x + 1, coord.v.y, coord.v.z});
        }
        if (minusXChanged) {
            queueDependent(ChunkCoord{coord.v.x - 1, coord.v.y, coord.v.z});
        }
        if (plusYChanged) {
            queueDependent(ChunkCoord{coord.v.x, coord.v.y + 1, coord.v.z});
        }
        if (minusYChanged) {
            queueDependent(ChunkCoord{coord.v.x, coord.v.y - 1, coord.v.z});
        }
        if (plusZChanged) {
            queueDependent(ChunkCoord{coord.v.x, coord.v.y, coord.v.z + 1});
        }
        if (minusZChanged) {
            queueDependent(ChunkCoord{coord.v.x, coord.v.y, coord.v.z - 1});
        }

        bool columnConverged = true;
        for (int32_t z = 0; z < cfg::COLUMN_HEIGHT; ++z) {
            const ChunkCoord chunkCoord{columnCoord.v.x, columnCoord.v.y, z};
            const LightingChunkState* chunkState = tryGetLightingChunkStateLocked(chunkCoord);
            if (chunkState == nullptr || chunkState->lightingEpoch < chunkState->topologyEpoch) {
                columnConverged = false;
                break;
            }
        }
        if (columnConverged && generatedColumns_.insert(columnCoord).second) {
            generatedColumnHistory_.push_back(columnCoord);
            generationRevision_.fetch_add(1, std::memory_order_release);
        }

        if (outLightChanged != nullptr) {
            *outLightChanged = chunkLightChanged;
        }
        if (outSolveSignature != nullptr) {
            *outSolveSignature = snapshotSignature;
        }
    }

    return true;
}

bool World::canPropagateChunkLocked(const ChunkCoord& coord) const {
    if (coord.v.z < 0 || coord.v.z >= cfg::COLUMN_HEIGHT) {
        return false;
    }

    const ColumnCoord columnCoord = chunk_to_column(coord);
    if (!isColumnSkycastCompleteLocked(columnCoord)) {
        return false;
    }

    for (const glm::ivec2& offset : kHorizontalOffsets) {
        const ColumnCoord neighborCoord{
            columnCoord.v.x + offset.x,
            columnCoord.v.y + offset.y
        };
        if (!isColumnSkycastCompleteLocked(neighborCoord)) {
            return false;
        }
    }

    return true;
}

bool World::isColumnSkycastCompleteLocked(const ColumnCoord& coord) const {
    return skycastColumns_.find(coord) != skycastColumns_.end();
}

Column* World::tryGetSkycastColumnLocked(const ColumnCoord& coord) {
    if (!isColumnSkycastCompleteLocked(coord)) {
        return nullptr;
    }

    const RegionCoord regionCoord = column_to_region(coord);
    const auto regionIt = regions_.find(regionCoord);
    if (regionIt == regions_.end() || regionIt->second == nullptr) {
        return nullptr;
    }

    const glm::ivec2 localColumn = column_local_in_region(coord);
    return &regionIt->second->getColumn(
        static_cast<uint8_t>(localColumn.x),
        static_cast<uint8_t>(localColumn.y)
    );
}

const Column* World::tryGetSkycastColumnLocked(const ColumnCoord& coord) const {
    if (!isColumnSkycastCompleteLocked(coord)) {
        return nullptr;
    }

    const RegionCoord regionCoord = column_to_region(coord);
    const auto regionIt = regions_.find(regionCoord);
    if (regionIt == regions_.end() || regionIt->second == nullptr) {
        return nullptr;
    }

    const glm::ivec2 localColumn = column_local_in_region(coord);
    return &regionIt->second->getColumn(
        static_cast<uint8_t>(localColumn.x),
        static_cast<uint8_t>(localColumn.y)
    );
}
