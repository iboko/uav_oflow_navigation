// Worst-case resource measurement for ofnav15::FixedLag on the HOST.
//
// 1. Stack: the workload runs in a POSIX thread whose stack is painted
//    with a pattern; the deepest overwritten byte is the measured
//    high-water mark (includes libc/libm frames used by the filter).
//    glibc places the thread descriptor and static TLS inside a
//    user-supplied stack, so an EMPTY thread is measured first and
//    subtracted: the remainder is the filter call-chain stack.
// 2. Time: worst and mean wall time of pushImu() (including checkpoint
//    trim replay) and pushDelayedFrame() of a five-event camera frame with
//    FULL IMU history (64 samples) and up to 24 stored aiding events.
//
// Host timing is NOT target WCET; docs/ESKF15_PX4_RUNTIME_RU.md gives the
// operation count used to scale it to a flight controller.
// Exit code != 0 if the stack exceeds the PX4 task budget below.
#include "ofnav/eskf15_workload.hpp"

#include <pthread.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

namespace {

constexpr std::size_t kThreadStack = 64U * 1024U;
// Budget reserved for the filter call chain inside the PX4 ofnav task
// (integrations/px4_ofnav_module/ofnav_px4.cpp, kTaskStackBytes = 8192).
constexpr std::size_t kFilterStackBudget = 3U * 1024U;
constexpr unsigned char kPaint = 0xA5U;

std::uint64_t nowNs() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct Job {
    ofnav15::WorkloadBuffers *buffers{nullptr};
    ofnav15::WorkloadResult result{};
};

void *workload(void *arg) {
    auto *job = static_cast<Job *>(arg);
    // The buffers are static storage in the PX4 module; here they are on
    // the heap so only CALL frames count as stack.
    job->result = ofnav15::runWorkload(*job->buffers, &nowNs, 30);
    return nullptr;
}

void *emptyWorkload(void *) {
    return nullptr;
}

// Returns bytes of the painted stack overwritten by fn, or 0 on error.
std::size_t paintedRun(void *(*fn)(void *), void *arg) {
    void *stack = nullptr;
    if (posix_memalign(&stack, 4096U, kThreadStack) != 0) { return 0; }
    std::memset(stack, kPaint, kThreadStack);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    std::size_t used = 0;
    pthread_t thread;
    if (pthread_attr_setstack(&attr, stack, kThreadStack) == 0 &&
        pthread_create(&thread, &attr, fn, arg) == 0) {
        pthread_join(thread, nullptr);
        // Stack grows downwards on all supported targets: count untouched
        // bytes from the low end.
        const auto *bytes = static_cast<const unsigned char *>(stack);
        std::size_t untouched = 0;
        while (untouched < kThreadStack && bytes[untouched] == kPaint) { ++untouched; }
        used = kThreadStack - untouched;
    }
    pthread_attr_destroy(&attr);
    std::free(stack);
    return used;
}

} // namespace

int main() {
    const std::size_t baseline = paintedRun(emptyWorkload, nullptr);
    Job job{};
    job.buffers = new ofnav15::WorkloadBuffers{};
    const std::size_t total = paintedRun(workload, &job);
    delete job.buffers;
    if (baseline == 0 || total < baseline) { return 2; }
    const std::size_t used = total - baseline;
    const auto &r = job.result;
    // Every event except explicit BUFFER_LIMIT must be processed.
    const bool pass = r.healthy && r.failed_events == 0U &&
                      r.processed >= 4U * r.frames && used <= kFilterStackBudget;
    std::printf("{\"type\":\"HOST_ESKF15_RESOURCE_MEASUREMENT\","
                "\"sizeof_fixed_lag_bytes\":%zu,\"sizeof_filter_bytes\":%zu,"
                "\"sizeof_workspace_bytes\":%zu,\"sizeof_workload_buffers_bytes\":%zu,"
                "\"thread_runtime_baseline_bytes\":%zu,"
                "\"filter_stack_high_water_bytes\":%zu,\"stack_budget_bytes\":%zu,"
                "\"imu_capacity\":%zu,\"event_capacity\":%zu,"
                "\"imu_samples\":%u,\"delayed_frames\":%u,\"processed_events\":%u,"
                "\"push_imu_worst_us\":%.1f,\"push_imu_mean_us\":%.1f,"
                "\"push_frame_worst_us\":%.1f,\"push_frame_mean_us\":%.1f,"
                "\"max_replayed_imu_samples\":%zu,"
                "\"stored_imu_at_worst\":%zu,\"stored_events_at_worst\":%zu,"
                "\"healthy\":%s,\"passed\":%s,\"host_time_is_not_target_wcet\":true}\n",
                sizeof(ofnav15::FixedLag), sizeof(ofnav15::Filter),
                sizeof(ofnav15::Workspace), sizeof(ofnav15::WorkloadBuffers),
                baseline, used, kFilterStackBudget,
                ofnav15::FixedLag::kImuCapacity, ofnav15::FixedLag::kVisualCapacity,
                r.imu_samples, r.frames, r.processed,
                r.imu_worst * 1e-3, r.imu_samples ? r.imu_total * 1e-3 / r.imu_samples : 0.,
                r.frame_worst * 1e-3, r.frames ? r.frame_total * 1e-3 / r.frames : 0.,
                r.max_replay_length, r.stored_imu_at_worst, r.stored_events_at_worst,
                r.healthy ? "true" : "false", pass ? "true" : "false");
    return pass ? 0 : 1;
}
