#include "solum_engine/voxel/World.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <shared_mutex>
#include <utility>
#include <vector>

#include "solum_engine/resources/Constants.h"
#include "solum_engine/voxel/Column.h"
#include "solum_engine/voxel/Region.h"
#include "solum_engine/voxel/TerrainGenerator.h"

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

constexpr uint8_t kLightingDirtyTopology = 1u << 0u;
constexpr uint8_t kLightingDirtyBoundary = 1u << 1u;
}  // namespace

struct World::ColumnGenerationResult {
    ColumnCoord coord{};
    Column column{};
    bool generated = false;
};

void World::updatePlayerPosition(const glm::vec3& playerWorldPosition) {
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return;
    }

    const BlockCoord playerBlock{
        static_cast<int32_t>(std::floor(playerWorldPosition.x)),
        static_cast<int32_t>(std::floor(playerWorldPosition.y)),
        static_cast<int32_t>(std::floor(playerWorldPosition.z))
    };
    const ColumnCoord centerColumn = chunk_to_column(block_to_chunk(playerBlock));

    bool centerUnchanged = false;
    {
        // Fast path for unchanged center without taking the write lock. Worker mesh jobs
        // hold shared locks frequently; avoiding a per-frame writer lock reduces stalls.
        std::shared_lock<std::shared_mutex> lock(worldMutex_);
        centerUnchanged = hasLastScheduledCenter_ && centerColumn == lastScheduledCenter_;
    }
    if (centerUnchanged) {
        if (hasPendingJobs()) {
            pumpColumnGenerationQueue();
            pumpChunkPropagationQueue();
        }
        return;
    }

    std::vector<ScheduledColumnJob> jobsToSchedule;
    {
        std::unique_lock<std::shared_mutex> lock(worldMutex_);
        if (hasLastScheduledCenter_ && centerColumn == lastScheduledCenter_) {
            return;
        }

        lastScheduledCenter_ = centerColumn;
        hasLastScheduledCenter_ = true;
        ++queueCenterVersion_;

        // Discard unscheduled work from an old center so movement can immediately
        // shift generation focus to the newest player position.
        queuedColumnJobs_.clear();
        queuedColumnHeap_ = decltype(queuedColumnHeap_){};

        refillQueuedColumnsLocked();
        pruneQueuedColumnsOutsideActiveWindowLocked();
        collectColumnJobsToScheduleLocked(jobsToSchedule);
    }
    dispatchScheduledColumnJobs(std::move(jobsToSchedule));
}

void World::enqueueColumnGenerationLocked(const ColumnCoord& coord) {
    if (!isWithinActiveWindowLocked(coord, 0)) {
        return;
    }
    if (isColumnSkycastCompleteLocked(coord)) {
        return;
    }
    if (pendingColumnJobs_.find(coord) != pendingColumnJobs_.end()) {
        return;
    }
    if (queuedColumnJobs_.find(coord) != queuedColumnJobs_.end()) {
        return;
    }
    queuedColumnJobs_.insert(coord);
    const int32_t distanceSq = hasLastScheduledCenter_
        ? distanceSqToCenter(coord, lastScheduledCenter_)
        : 0;
    queuedColumnHeap_.push(QueuedColumnEntry{
        coord,
        distanceSq,
        queueCenterVersion_,
        queueSequence_++
    });
}

std::size_t World::desiredQueuedColumnCountLocked() const {
    const int32_t radius = std::max(0, config_.columnLoadRadius);
    const uint64_t diameter = (static_cast<uint64_t>(radius) * 2u) + 1u;
    const uint64_t maxWindowColumns64 = diameter * diameter;
    const std::size_t maxWindowColumns = static_cast<std::size_t>(
        std::min<uint64_t>(maxWindowColumns64, std::numeric_limits<std::size_t>::max())
    );

    // Keep a short look-ahead backlog to reduce idle worker time, but cap queue
    // growth so center shifts can be reflected immediately.
    constexpr std::size_t kQueueLookAheadMultiplier = 4;
    const std::size_t lookAheadOutstanding =
        (maxInFlightColumnJobs_ > (std::numeric_limits<std::size_t>::max() / kQueueLookAheadMultiplier))
            ? std::numeric_limits<std::size_t>::max()
            : (maxInFlightColumnJobs_ * kQueueLookAheadMultiplier);
    const std::size_t desiredOutstanding = std::max(maxInFlightColumnJobs_, lookAheadOutstanding);

    const std::size_t pendingCount = pendingColumnJobs_.size();
    const std::size_t desiredQueued =
        (desiredOutstanding > pendingCount) ? (desiredOutstanding - pendingCount) : 0;
    return std::min(desiredQueued, maxWindowColumns);
}

void World::refillQueuedColumnsLocked() {
    if (!hasLastScheduledCenter_) {
        return;
    }

    const std::size_t desiredQueued = desiredQueuedColumnCountLocked();
    if (queuedColumnJobs_.size() >= desiredQueued) {
        return;
    }

    std::size_t remainingToQueue = desiredQueued - queuedColumnJobs_.size();
    const int32_t radius = std::max(0, config_.columnLoadRadius);
    const ColumnCoord center = lastScheduledCenter_;

    auto tryEnqueue = [&](int32_t x, int32_t y) {
        if (remainingToQueue == 0) {
            return;
        }
        const std::size_t queuedBefore = queuedColumnJobs_.size();
        enqueueColumnGenerationLocked(ColumnCoord{x, y});
        if (queuedColumnJobs_.size() > queuedBefore) {
            --remainingToQueue;
        }
    };

    if (remainingToQueue > 0) {
        tryEnqueue(center.v.x, center.v.y);
    }

    for (int32_t ring = 1; ring <= radius && remainingToQueue > 0; ++ring) {
        const int32_t minX = center.v.x - ring;
        const int32_t maxX = center.v.x + ring;
        const int32_t minY = center.v.y - ring;
        const int32_t maxY = center.v.y + ring;

        for (int32_t x = minX; x <= maxX && remainingToQueue > 0; ++x) {
            tryEnqueue(x, minY);
            tryEnqueue(x, maxY);
        }
        for (int32_t y = minY + 1; y <= maxY - 1 && remainingToQueue > 0; ++y) {
            tryEnqueue(minX, y);
            tryEnqueue(maxX, y);
        }
    }
}

void World::pruneQueuedColumnsOutsideActiveWindowLocked() {
    // Bounded heap cleanup to cap per-pump overhead when radius grows.
    constexpr size_t kPruneBudget = 256;
    size_t processed = 0;
    while (processed < kPruneBudget && !queuedColumnHeap_.empty()) {
        const QueuedColumnEntry top = queuedColumnHeap_.top();

        auto queuedIt = queuedColumnJobs_.find(top.coord);
        if (queuedIt == queuedColumnJobs_.end()) {
            queuedColumnHeap_.pop();
            ++processed;
            continue;
        }

        if (!isWithinActiveWindowLocked(top.coord, 0) ||
            isColumnSkycastCompleteLocked(top.coord) ||
            pendingColumnJobs_.find(top.coord) != pendingColumnJobs_.end()) {
            queuedColumnJobs_.erase(queuedIt);
            queuedColumnHeap_.pop();
            ++processed;
            continue;
        }

        if (top.centerVersion != queueCenterVersion_) {
            queuedColumnHeap_.pop();
            queuedColumnHeap_.push(QueuedColumnEntry{
                top.coord,
                hasLastScheduledCenter_ ? distanceSqToCenter(top.coord, lastScheduledCenter_) : 0,
                queueCenterVersion_,
                queueSequence_++
            });
            ++processed;
            continue;
        }

        break;
    }
}

void World::collectColumnJobsToScheduleLocked(std::vector<ScheduledColumnJob>& outJobs) {
    while (pendingColumnJobs_.size() < maxInFlightColumnJobs_ && !queuedColumnHeap_.empty()) {
        const QueuedColumnEntry top = queuedColumnHeap_.top();
        queuedColumnHeap_.pop();

        auto queuedIt = queuedColumnJobs_.find(top.coord);
        if (queuedIt == queuedColumnJobs_.end()) {
            continue;
        }

        if (!isWithinActiveWindowLocked(top.coord, 0) ||
            isColumnSkycastCompleteLocked(top.coord) ||
            pendingColumnJobs_.find(top.coord) != pendingColumnJobs_.end()) {
            queuedColumnJobs_.erase(queuedIt);
            continue;
        }

        if (top.centerVersion != queueCenterVersion_) {
            queuedColumnHeap_.push(QueuedColumnEntry{
                top.coord,
                hasLastScheduledCenter_ ? distanceSqToCenter(top.coord, lastScheduledCenter_) : 0,
                queueCenterVersion_,
                queueSequence_++
            });
            continue;
        }

        queuedColumnJobs_.erase(queuedIt);
        pendingColumnJobs_.insert(top.coord);
        outJobs.push_back(ScheduledColumnJob{
            top.coord,
            priorityFromDistanceSq(top.distanceSq)
        });
    }
}

void World::dispatchScheduledColumnJobs(std::vector<ScheduledColumnJob>&& jobsToSchedule) {
    for (const ScheduledColumnJob& scheduled : jobsToSchedule) {
        const ColumnCoord coord = scheduled.coord;
        try {
            jobs_.schedule(
                scheduled.priority,
                [this, coord]() -> ColumnGenerationResult {
                    {
                        std::shared_lock<std::shared_mutex> lock(worldMutex_);
                        if (!isWithinActiveWindowLocked(coord, 0)) {
                            return ColumnGenerationResult{
                                coord,
                                Column{},
                                false
                            };
                        }
                    }

                    TerrainGenerator generator;
                    Column generatedColumn;

                    const ChunkCoord columnBaseChunk = column_local_to_chunk(coord, 0);
                    const BlockCoord columnOrigin = chunk_to_block_origin(columnBaseChunk);
                    generator.generateColumn(columnOrigin.v, generatedColumn);

                    return ColumnGenerationResult{
                        coord,
                        std::move(generatedColumn),
                        true
                    };
                },
                [this, coord](jobsystem::JobResult<ColumnGenerationResult>&& result) {
                    if (!result.success()) {
                        {
                            std::unique_lock<std::shared_mutex> lock(worldMutex_);
                            pendingColumnJobs_.erase(coord);
                        }
                        pumpColumnGenerationQueue();
                        pumpChunkPropagationQueue();
                        return;
                    }

                    ColumnGenerationResult generated = std::move(result).value();
                    if (!generated.generated) {
                        {
                            std::unique_lock<std::shared_mutex> lock(worldMutex_);
                            pendingColumnJobs_.erase(coord);
                        }
                        pumpColumnGenerationQueue();
                        pumpChunkPropagationQueue();
                        return;
                    }
                    onColumnGenerated(generated.coord, std::move(generated.column));
                    pumpColumnGenerationQueue();
                    pumpChunkPropagationQueue();
                }
            );
        } catch (const std::exception&) {
            {
                std::unique_lock<std::shared_mutex> lock(worldMutex_);
                pendingColumnJobs_.erase(coord);
                if (!shuttingDown_.load(std::memory_order_acquire) &&
                    isWithinActiveWindowLocked(coord, 0) &&
                    !isColumnSkycastCompleteLocked(coord)) {
                    queuedColumnJobs_.insert(coord);
                    const int32_t distanceSq = hasLastScheduledCenter_
                        ? distanceSqToCenter(coord, lastScheduledCenter_)
                        : 0;
                    queuedColumnHeap_.push(QueuedColumnEntry{
                        coord,
                        distanceSq,
                        queueCenterVersion_,
                        queueSequence_++
                    });
                }
            }
        }
    }
}

void World::pumpColumnGenerationQueue() {
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return;
    }

    std::vector<ScheduledColumnJob> jobsToSchedule;
    {
        std::unique_lock<std::shared_mutex> lock(worldMutex_);
        pruneQueuedColumnsOutsideActiveWindowLocked();
        refillQueuedColumnsLocked();
        collectColumnJobsToScheduleLocked(jobsToSchedule);
    }
    dispatchScheduledColumnJobs(std::move(jobsToSchedule));
}

void World::onColumnGenerated(const ColumnCoord& coord, Column&& column) {
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return;
    }

    std::unique_lock<std::shared_mutex> lock(worldMutex_);
    pendingColumnJobs_.erase(coord);
    if (!isWithinActiveWindowLocked(coord, 0)) {
        return;
    }

    // Keep occupancy metadata coherent even if a generator path bypasses Column::setBlock.
    column.rebuildEmptyChunkMask();

    Region* region = getOrCreateRegionLocked(column_to_region(coord));
    if (region == nullptr) {
        return;
    }

    const glm::ivec2 localColumn = column_local_in_region(coord);
    region->getColumn(
        static_cast<uint8_t>(localColumn.x),
        static_cast<uint8_t>(localColumn.y)
    ) = std::move(column);

    skycastColumns_.insert(coord);
    for (int32_t z = 0; z < cfg::COLUMN_HEIGHT; ++z) {
        const ChunkCoord chunkCoord{coord.v.x, coord.v.y, z};
        LightingChunkState& state = lightingChunkStates_[chunkCoord];
        if (state.topologyEpoch == 0u) {
            state.topologyEpoch = 1u;
        } else {
            ++state.topologyEpoch;
        }
        state.lightingEpoch = 0u;
        state.queuedEpoch = 0u;
        state.inFlightEpoch = 0u;
        state.dirtyFlags = static_cast<uint8_t>(kLightingDirtyTopology | kLightingDirtyBoundary);
        state.lastSolveSignature = 0u;
    }
    enqueueChunkPropagationCandidatesLocked(coord);
}
