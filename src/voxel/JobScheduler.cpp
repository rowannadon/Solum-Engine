#include "solum_engine/voxel/JobScheduler.h"

#include <algorithm>
#include <utility>

JobScheduler::JobScheduler(std::size_t workerThreads) {
    if (workerThreads == 0) {
        workerThreads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    }

    workers_.reserve(workerThreads);
    for (std::size_t i = 0; i < workerThreads; ++i) {
        workers_.emplace_back([this]() { workerMain(); });
    }
}

JobScheduler::~JobScheduler() {
    stopping_.store(true, std::memory_order_release);
    queueCv_.notify_all();

    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void JobScheduler::setExecutor(Executor executor) {
    std::scoped_lock lock(executorMutex_);
    executor_ = std::move(executor);
}

uint64_t JobScheduler::enqueue(VoxelJob job) {
    const uint64_t ticket = nextTicket_.fetch_add(1u, std::memory_order_relaxed);
    job.ticket = ticket;

    {
        std::scoped_lock lock(queueMutex_);
        switch (job.priority) {
        case JobPriority::High:
            highQueue_.push_back(std::move(job));
            break;
        case JobPriority::Medium:
            mediumQueue_.push_back(std::move(job));
            break;
        case JobPriority::Low:
            lowQueue_.push_back(std::move(job));
            break;
        }
    }

    queueCv_.notify_one();
    return ticket;
}

bool JobScheduler::tryPopResult(JobResult& outResult) {
    std::scoped_lock lock(resultMutex_);
    if (completedResults_.empty()) {
        return false;
    }

    outResult = std::move(completedResults_.front());
    completedResults_.pop_front();
    return true;
}

void JobScheduler::workerMain() {
    while (!stopping_.load(std::memory_order_acquire)) {
        VoxelJob job;

        {
            std::unique_lock lock(queueMutex_);
            queueCv_.wait(lock, [this]() {
                return stopping_.load(std::memory_order_acquire)
                    || !highQueue_.empty()
                    || !mediumQueue_.empty()
                    || !lowQueue_.empty();
            });

            if (stopping_.load(std::memory_order_acquire)) {
                return;
            }

            if (!tryPopNextJobLocked(job)) {
                continue;
            }
        }

        Executor executor;
        {
            std::scoped_lock lock(executorMutex_);
            executor = executor_;
        }

        JobResult result;
        if (executor) {
            result = executor(job);
        } else {
            result.type = job.type;
            result.ticket = job.ticket;
            result.payload = TerrainJobResult{};
        }

        result.ticket = job.ticket;

        {
            std::scoped_lock lock(resultMutex_);
            completedResults_.push_back(std::move(result));
        }
    }
}

bool JobScheduler::tryPopNextJobLocked(VoxelJob& job) {
    if (!highQueue_.empty()) {
        job = std::move(highQueue_.front());
        highQueue_.pop_front();
        return true;
    }
    if (!mediumQueue_.empty()) {
        job = std::move(mediumQueue_.front());
        mediumQueue_.pop_front();
        return true;
    }
    if (!lowQueue_.empty()) {
        job = std::move(lowQueue_.front());
        lowQueue_.pop_front();
        return true;
    }

    return false;
}
