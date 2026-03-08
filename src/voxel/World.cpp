#include "solum_engine/voxel/World.h"

#include <algorithm>
#include <shared_mutex>
#include <utility>

#include "solum_engine/voxel/Region.h"

World::World()
    : World(Config{}) {}

World::World(Config config)
    : config_(std::move(config)),
      jobs_(config_.jobConfig),
      chunkPropagationJobs_([this]() {
          jobsystem::JobSystem::Config propagationConfig = config_.jobConfig;
          const std::size_t configuredWorkers =
              (propagationConfig.worker_threads > 0) ? propagationConfig.worker_threads : 1u;
          propagationConfig.worker_threads = std::clamp<std::size_t>(configuredWorkers, 1u, 2u);
          return propagationConfig;
      }()) {
    const std::size_t configuredMaxInFlight = config_.maxInFlightColumnJobs;
    const std::size_t workerCount = std::max<std::size_t>(std::size_t{1}, jobs_.worker_count());
    const std::size_t autoMaxInFlight = workerCount * 2;
    maxInFlightColumnJobs_ = std::max<std::size_t>(
        std::size_t{1},
        (configuredMaxInFlight > 0) ? configuredMaxInFlight : autoMaxInFlight
    );
    maxInFlightChunkPropagationJobs_ = std::max<std::size_t>(
        std::size_t{1},
        chunkPropagationJobs_.worker_count()
    );
}

World::~World() {
    shuttingDown_.store(true, std::memory_order_release);
    jobs_.wait_for_idle();
    chunkPropagationJobs_.wait_for_idle();
    jobs_.stop();
    chunkPropagationJobs_.stop();
}

bool World::hasPendingJobs() const {
    std::shared_lock<std::shared_mutex> lock(worldMutex_);
    return !pendingColumnJobs_.empty() ||
           !queuedColumnJobs_.empty() ||
           !pendingChunkPropagationJobs_.empty() ||
           !queuedChunkPropagationJobs_.empty();
}

jobsystem::Priority World::priorityFromDistanceSq(int32_t distanceSq) {
    if (distanceSq <= 0) {
        return jobsystem::Priority::Critical;
    }
    if (distanceSq <= 2) {
        return jobsystem::Priority::High;
    }
    if (distanceSq <= 8) {
        return jobsystem::Priority::Normal;
    }
    return jobsystem::Priority::Low;
}
