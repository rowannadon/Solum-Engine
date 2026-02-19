#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace jobsystem {

class JobSystem;

using JobId = std::uint64_t;

enum class Priority : std::uint8_t {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
};

struct JobContext {
    JobSystem& system;
    std::size_t worker_index;
};

template <typename T>
class JobResult {
public:
    JobResult(JobId id, std::optional<T> value, std::exception_ptr error)
        : id_(id), value_(std::move(value)), error_(std::move(error)) {}

    [[nodiscard]] JobId job_id() const noexcept { return id_; }
    [[nodiscard]] bool success() const noexcept { return error_ == nullptr; }
    [[nodiscard]] explicit operator bool() const noexcept { return success(); }
    [[nodiscard]] const std::exception_ptr& error() const noexcept { return error_; }

    void rethrow_if_error() const {
        if (error_) {
            std::rethrow_exception(error_);
        }
    }

    T& value() & {
        rethrow_if_error();
        return *value_;
    }

    const T& value() const& {
        rethrow_if_error();
        return *value_;
    }

    T&& value() && {
        rethrow_if_error();
        return std::move(*value_);
    }

private:
    JobId id_;
    std::optional<T> value_;
    std::exception_ptr error_;
};

template <>
class JobResult<void> {
public:
    JobResult(JobId id, std::exception_ptr error) : id_(id), error_(std::move(error)) {}

    [[nodiscard]] JobId job_id() const noexcept { return id_; }
    [[nodiscard]] bool success() const noexcept { return error_ == nullptr; }
    [[nodiscard]] explicit operator bool() const noexcept { return success(); }
    [[nodiscard]] const std::exception_ptr& error() const noexcept { return error_; }

    void rethrow_if_error() const {
        if (error_) {
            std::rethrow_exception(error_);
        }
    }

private:
    JobId id_;
    std::exception_ptr error_;
};

class JobSystem {
public:
    struct Config {
        std::size_t worker_threads = 0;
    };

    JobSystem();
    explicit JobSystem(Config config);
    ~JobSystem();

    JobSystem(const JobSystem&) = delete;
    JobSystem& operator=(const JobSystem&) = delete;
    JobSystem(JobSystem&&) = delete;
    JobSystem& operator=(JobSystem&&) = delete;

    template <typename WorkFn, typename CompletionFn>
    JobId schedule(Priority priority, WorkFn&& work, CompletionFn&& on_complete);

    template <typename WorkFn>
    JobId schedule(Priority priority, WorkFn&& work);

    void wait_for_idle();
    void stop();

    [[nodiscard]] std::size_t worker_count() const noexcept { return config_.worker_threads; }

private:
    struct ScheduledJob {
        JobId id;
        Priority priority;
        std::function<void(JobContext&)> run;
    };

    struct CompletionEvent {
        std::function<void(JobSystem&)> dispatch;
    };

    class MPSCCompletionQueue;

    template <typename WorkFn, typename CompletionFn>
    ScheduledJob make_scheduled_job(JobId id, Priority priority, WorkFn&& work, CompletionFn&& on_complete);

    template <typename WorkFn>
    static decltype(auto) invoke_work(WorkFn& work, JobContext& ctx);

    template <typename CompletionFn, typename ResultT>
    static void invoke_completion(CompletionFn& completion, JobSystem& system, JobResult<ResultT>&& result);

    void enqueue_job(ScheduledJob&& job);
    bool has_pending_jobs_locked() const;
    ScheduledJob pop_next_job_locked();

    void worker_loop(std::size_t worker_index);
    void completion_loop();

    void publish_completion(CompletionEvent&& event);
    void mark_job_finished();

    static std::size_t default_worker_count();

    static constexpr std::size_t kPriorityCount = 4;

    Config config_;

    std::atomic<bool> stopping_{false};
    std::atomic<JobId> next_job_id_{1};
    std::atomic<std::size_t> in_flight_jobs_{0};

    std::array<std::deque<ScheduledJob>, kPriorityCount> pending_jobs_;
    mutable std::mutex jobs_mutex_;
    std::condition_variable jobs_cv_;

    std::unique_ptr<MPSCCompletionQueue> completion_queue_;
    std::mutex completion_wait_mutex_;
    std::condition_variable completion_cv_;

    std::mutex idle_mutex_;
    std::condition_variable idle_cv_;

    std::vector<std::thread> workers_;
    std::thread completion_consumer_;
};

namespace detail {
template <typename...>
inline constexpr bool always_false_v = false;
}  // namespace detail

template <typename WorkFn>
decltype(auto) JobSystem::invoke_work(WorkFn& work, JobContext& ctx) {
    if constexpr (std::is_invocable_v<WorkFn&, JobContext&>) {
        return std::invoke(work, ctx);
    } else if constexpr (std::is_invocable_v<WorkFn&>) {
        return std::invoke(work);
    } else {
        static_assert(detail::always_false_v<WorkFn>,
                      "Work function must be invocable with (JobContext&) or ()");
    }
}

template <typename CompletionFn, typename ResultT>
void JobSystem::invoke_completion(CompletionFn& completion,
                                  JobSystem& system,
                                  JobResult<ResultT>&& result) {
    if constexpr (std::is_invocable_v<CompletionFn&, JobSystem&, JobResult<ResultT>&&>) {
        std::invoke(completion, system, std::move(result));
    } else if constexpr (std::is_invocable_v<CompletionFn&, JobResult<ResultT>&&>) {
        std::invoke(completion, std::move(result));
    } else if constexpr (std::is_invocable_v<CompletionFn&, JobSystem&>) {
        std::invoke(completion, system);
    } else if constexpr (std::is_invocable_v<CompletionFn&>) {
        std::invoke(completion);
    } else {
        static_assert(detail::always_false_v<CompletionFn, ResultT>,
                      "Completion function must accept (JobSystem&, JobResult<T>) or compatible subset");
    }
}

template <typename WorkFn, typename CompletionFn>
JobSystem::ScheduledJob JobSystem::make_scheduled_job(JobId id,
                                                      Priority priority,
                                                      WorkFn&& work,
                                                      CompletionFn&& on_complete) {
    using Work = std::decay_t<WorkFn>;
    using Completion = std::decay_t<CompletionFn>;

    ScheduledJob job;
    job.id = id;
    job.priority = priority;

    job.run = [this,
               id,
               work = Work(std::forward<WorkFn>(work)),
               completion = Completion(std::forward<CompletionFn>(on_complete))](JobContext& ctx) mutable {
        using RawResult = decltype(invoke_work(work, ctx));

        if constexpr (std::is_void_v<RawResult>) {
            std::exception_ptr error;
            try {
                invoke_work(work, ctx);
            } catch (...) {
                error = std::current_exception();
            }

            publish_completion(CompletionEvent{[completion = std::move(completion),
                                               result = JobResult<void>(id, std::move(error))](
                                                  JobSystem& system) mutable {
                invoke_completion<Completion, void>(completion, system, std::move(result));
            }});
        } else {
            using Result = std::decay_t<RawResult>;

            std::optional<Result> value;
            std::exception_ptr error;

            try {
                value.emplace(invoke_work(work, ctx));
            } catch (...) {
                error = std::current_exception();
            }

            publish_completion(CompletionEvent{[completion = std::move(completion),
                                               result = JobResult<Result>(id, std::move(value), std::move(error))](
                                                  JobSystem& system) mutable {
                invoke_completion<Completion, Result>(completion, system, std::move(result));
            }});
        }
    };

    return job;
}

template <typename WorkFn, typename CompletionFn>
JobId JobSystem::schedule(Priority priority, WorkFn&& work, CompletionFn&& on_complete) {
    if (stopping_.load(std::memory_order_acquire)) {
        throw std::runtime_error("Cannot schedule jobs after JobSystem::stop()");
    }

    const JobId id = next_job_id_.fetch_add(1, std::memory_order_relaxed);
    in_flight_jobs_.fetch_add(1, std::memory_order_acq_rel);

    enqueue_job(make_scheduled_job(id,
                                   priority,
                                   std::forward<WorkFn>(work),
                                   std::forward<CompletionFn>(on_complete)));

    return id;
}

template <typename WorkFn>
JobId JobSystem::schedule(Priority priority, WorkFn&& work) {
    return schedule(priority,
                    std::forward<WorkFn>(work),
                    [](JobSystem&, auto&&) {});
}

}  // namespace jobsystem
