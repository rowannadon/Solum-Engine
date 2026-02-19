#include "solum_engine/jobsystem/job_system.hpp"

#include <iostream>

namespace jobsystem {

class JobSystem::MPSCCompletionQueue {
public:
    MPSCCompletionQueue() {
        Node* stub = new Node();
        head_.store(stub, std::memory_order_relaxed);
        tail_ = stub;
    }

    ~MPSCCompletionQueue() {
        while (pop().has_value()) {
        }
        delete tail_;
    }

    void push(CompletionEvent&& event) {
        Node* node = new Node(std::move(event));
        Node* prev = head_.exchange(node, std::memory_order_acq_rel);
        prev->next.store(node, std::memory_order_release);
    }

    std::optional<CompletionEvent> pop() {
        Node* tail = tail_;
        Node* next = tail->next.load(std::memory_order_acquire);
        if (next == nullptr) {
            return std::nullopt;
        }

        tail_ = next;
        std::optional<CompletionEvent> event = std::move(next->event);
        delete tail;
        return event;
    }

    bool empty() const {
        return tail_->next.load(std::memory_order_acquire) == nullptr;
    }

private:
    struct Node {
        std::atomic<Node*> next{nullptr};
        std::optional<CompletionEvent> event;

        Node() = default;
        explicit Node(CompletionEvent&& e) : event(std::move(e)) {}
    };

    std::atomic<Node*> head_{nullptr};
    Node* tail_{nullptr};
};

std::size_t JobSystem::default_worker_count() {
    const std::size_t n = std::thread::hardware_concurrency();
    return n == 0 ? 1 : n;
}

JobSystem::JobSystem(Config config) : config_(config), completion_queue_(std::make_unique<MPSCCompletionQueue>()) {
    if (config_.worker_threads == 0) {
        config_.worker_threads = default_worker_count();
    }

    completion_consumer_ = std::thread([this] { completion_loop(); });

    workers_.reserve(config_.worker_threads);
    for (std::size_t i = 0; i < config_.worker_threads; ++i) {
        workers_.emplace_back([this, i] { worker_loop(i); });
    }
}

JobSystem::JobSystem() : JobSystem(Config{}) {}

JobSystem::~JobSystem() {
    stop();
}

void JobSystem::enqueue_job(ScheduledJob&& job) {
    {
        std::lock_guard<std::mutex> lock(jobs_mutex_);
        pending_jobs_[static_cast<std::size_t>(job.priority)].push_back(std::move(job));
    }
    jobs_cv_.notify_one();
}

bool JobSystem::has_pending_jobs_locked() const {
    for (const auto& queue : pending_jobs_) {
        if (!queue.empty()) {
            return true;
        }
    }
    return false;
}

JobSystem::ScheduledJob JobSystem::pop_next_job_locked() {
    for (std::size_t idx = kPriorityCount; idx > 0; --idx) {
        auto& queue = pending_jobs_[idx - 1];
        if (!queue.empty()) {
            ScheduledJob job = std::move(queue.front());
            queue.pop_front();
            return job;
        }
    }

    throw std::runtime_error("pop_next_job_locked called with no pending jobs");
}

void JobSystem::worker_loop(std::size_t worker_index) {
    JobContext ctx{*this, worker_index};

    while (true) {
        ScheduledJob job;

        {
            std::unique_lock<std::mutex> lock(jobs_mutex_);
            jobs_cv_.wait(lock, [this] {
                return stopping_.load(std::memory_order_acquire) || has_pending_jobs_locked();
            });

            if (stopping_.load(std::memory_order_acquire) && !has_pending_jobs_locked()) {
                return;
            }

            job = pop_next_job_locked();
        }

        try {
            job.run(ctx);
        } catch (...) {
            std::exception_ptr error = std::current_exception();
            publish_completion(CompletionEvent{[error = std::move(error)](JobSystem&) {
                try {
                    if (error) {
                        std::rethrow_exception(error);
                    }
                } catch (const std::exception& ex) {
                    std::cerr << "Unhandled worker exception: " << ex.what() << '\n';
                } catch (...) {
                    std::cerr << "Unhandled worker exception: unknown error\n";
                }
            }});
        }
    }
}

void JobSystem::publish_completion(CompletionEvent&& event) {
    completion_queue_->push(std::move(event));
    completion_cv_.notify_one();
}

void JobSystem::mark_job_finished() {
    const std::size_t remaining = in_flight_jobs_.fetch_sub(1, std::memory_order_acq_rel) - 1;
    if (remaining == 0) {
        std::lock_guard<std::mutex> lock(idle_mutex_);
        idle_cv_.notify_all();
    }
}

void JobSystem::completion_loop() {
    while (true) {
        std::optional<CompletionEvent> event = completion_queue_->pop();

        if (!event.has_value()) {
            std::unique_lock<std::mutex> lock(completion_wait_mutex_);
            completion_cv_.wait(lock, [this] {
                return stopping_.load(std::memory_order_acquire) || !completion_queue_->empty();
            });

            event = completion_queue_->pop();

            if (!event.has_value()) {
                if (stopping_.load(std::memory_order_acquire) &&
                    in_flight_jobs_.load(std::memory_order_acquire) == 0) {
                    return;
                }
                continue;
            }
        }

        try {
            event->dispatch(*this);
        } catch (const std::exception& ex) {
            std::cerr << "Completion handler threw exception: " << ex.what() << '\n';
        } catch (...) {
            std::cerr << "Completion handler threw exception: unknown error\n";
        }

        mark_job_finished();

        if (stopping_.load(std::memory_order_acquire) &&
            in_flight_jobs_.load(std::memory_order_acquire) == 0 && completion_queue_->empty()) {
            return;
        }
    }
}

void JobSystem::wait_for_idle() {
    std::unique_lock<std::mutex> lock(idle_mutex_);
    idle_cv_.wait(lock, [this] {
        return in_flight_jobs_.load(std::memory_order_acquire) == 0;
    });
}

void JobSystem::stop() {
    bool was_stopping = stopping_.exchange(true, std::memory_order_acq_rel);

    if (!was_stopping) {
        jobs_cv_.notify_all();
        completion_cv_.notify_all();
    }

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();

    completion_cv_.notify_all();
    if (completion_consumer_.joinable()) {
        completion_consumer_.join();
    }
}

}  // namespace jobsystem
