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
#include "ofnav/eskf15.hpp"

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
// (integrations/px4_ofnav_module/ofnav_px4.cpp, kTaskStackBytes = 6144:
// 3 KiB filter + 3 KiB module loop, uORB and logging).
constexpr std::size_t kFilterStackBudget = 3U * 1024U;
constexpr unsigned char kPaint = 0xA5U;

struct Result {
    double imu_worst_us{0.}, imu_mean_us{0.};
    double delayed_worst_us{0.}, delayed_mean_us{0.};
    unsigned delayed_samples{0}, imu_samples{0};
    unsigned processed{0};
    std::size_t stored_imu_at_worst{0}, stored_events_at_worst{0};
    std::size_t max_replay_length{0};
    bool healthy{false};
};

using Clock = std::chrono::steady_clock;

double elapsedUs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

ofnav15::Imu imu(std::uint64_t t, int i) {
    const double s = static_cast<double>(i) * .01;
    return ofnav15::Imu{t, {.01 * std::sin(.7 * s), .01 * std::cos(.5 * s), .02},
                        {.2 * std::sin(.37 * s), .1 * std::cos(.31 * s),
                         -9.80665 + .02 * std::sin(.41 * s)}};
}

struct Truth {
    std::uint64_t t{0};
    ofnav15::Vec3 p{}, v{};
    ofnav15::Quat q{1., 0., 0., 0.};
};

// Aiding consistent with the SAME IMU stream: values from an unaided
// reference filter (no corrections) plus small deterministic offsets.
ofnav15::Measurement event(const Truth &x, ofnav15::MeasurementKind kind, int n) {
    ofnav15::Measurement m{};
    m.time_us = x.t;
    m.kind = kind;
    const double w = x.q[0], qx = x.q[1], qy = x.q[2], qz = x.q[3];
    const double r[9] = {1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - w * qz), 2 * (qx * qz + w * qy),
                         2 * (qx * qy + w * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - w * qx),
                         2 * (qx * qz - w * qy), 2 * (qy * qz + w * qx), 1 - 2 * (qx * qx + qy * qy)};
    const double e = .01 * std::sin(1.3 * n);
    switch (kind) {
    case ofnav15::MeasurementKind::VelocityNe:
        m.value = {x.v[0] + e, x.v[1] - e};
        m.covariance = {0.04, 0., 0., 0.04};
        break;
    case ofnav15::MeasurementKind::BodyVelocityXy:
        m.value = {r[0] * x.v[0] + r[3] * x.v[1] + r[6] * x.v[2] + e,
                   r[1] * x.v[0] + r[4] * x.v[1] + r[7] * x.v[2] - e};
        m.covariance = {0.04, 0., 0., 0.04};
        break;
    case ofnav15::MeasurementKind::Yaw:
        m.value = {std::atan2(r[3], r[0]) + .1 * e, 0.};
        m.covariance = {0.0004, 0., 0., 0.};
        break;
    case ofnav15::MeasurementKind::PositionNe:
        m.value = {x.p[0] + 10. * e, x.p[1] - 10. * e};
        m.covariance = {4., 0., 0., 4.};
        m.ci_weight = 0.;  // automatic covariance intersection
        break;
    case ofnav15::MeasurementKind::MapGroundPointNe:
        // Nadir axis d_b = (0,0,1): ground point = p + h (R d)_NE / (R d)_D.
        m.value = {x.p[0] + 100. * r[2] / r[8] + e, x.p[1] + 100. * r[5] / r[8] - e};
        m.covariance = {.04, 0., 0., .04};
        m.ci_weight = 0.;
        m.aux = {100., 0., 0., 1.};
        break;
    }
    return m;
}

void *workload(void *arg) {
    auto *result = static_cast<Result *>(arg);
    // The filter object itself is static storage in the PX4 module; here
    // it is placed in heap memory so only CALL frames count as stack.
    void *storage = std::malloc(sizeof(ofnav15::FixedLag));
    if (storage == nullptr) { return nullptr; }
    auto *lag = new (storage) ofnav15::FixedLag({}, 630000U);
    // Unaided reference filter on the heap with its own heap workspace,
    // so it does not contribute to the measured stack.
    auto *reference = new ofnav15::Filter({});
    auto *reference_ws = new ofnav15::Workspace{};
    // Harness buffers on the heap too: only filter call frames are measured.
    auto *history = new Truth[16]{};
    auto *frame_events = new ofnav15::Measurement[5]{};
    std::uint64_t t = 1000000U;
    int i = 0;
    double imu_sum = 0., delayed_sum = 0.;
    bool ok = true;
    // 0.63 s lag at 100 Hz keeps all 64 IMU samples; trim engages after.
    for (int frame = 0; frame < 30 && ok; ++frame) {
        for (int k = 0; k < 10; ++k, ++i) {
            const ofnav15::Imu sample = imu(t, i);
            (void)reference->predict(sample, *reference_ws);
            history[i % 16] = Truth{t, reference->position(), reference->velocity(),
                                    reference->quaternion()};
            const auto a = Clock::now();
            const auto s = lag->pushImu(sample);
            const auto b = Clock::now();
            const double us = elapsedUs(a, b);
            imu_sum += us;
            ++result->imu_samples;
            result->imu_worst_us = std::max(result->imu_worst_us, us);
            if (s != ofnav15::Status::InertialOnly) {
                std::fprintf(stderr, "pushImu status %s at %d\n", ofnav15::statusName(s), i);
                ok = false;
                break;
            }
            t += 10000U;
        }
        if (!ok) { break; }
        // One camera frame per 0.1 s: all five aiding kinds of one
        // exposure, delivered 60 ms late and inserted by ONE replay.
        const Truth &x = history[(i - 6) % 16];
        for (int k = 0; k < 5; ++k) {
            frame_events[k] = event(x, static_cast<ofnav15::MeasurementKind>(k), frame);
        }
        ofnav15::Status statuses[5]{};
        const auto a = Clock::now();
        (void)lag->pushDelayedFrame(frame_events, 5, statuses);
        const auto b = Clock::now();
        const double us = elapsedUs(a, b);
        delayed_sum += us;
        ++result->delayed_samples;
        result->max_replay_length = std::max(result->max_replay_length, lag->lastReplayLength());
        if (us > result->delayed_worst_us) {
            result->delayed_worst_us = us;
            result->stored_imu_at_worst = lag->storedImu();
            result->stored_events_at_worst = lag->storedVisual();
        }
        for (const auto s : statuses) {
            if (ofnav15::measurementProcessed(s)) { ++result->processed; }
            else if (s != ofnav15::Status::BufferLimit) {
                std::fprintf(stderr, "frame %d status %s\n", frame, ofnav15::statusName(s));
            }
        }
    }
    delete[] frame_events;
    delete[] history;
    delete reference_ws;
    delete reference;
    result->imu_mean_us = imu_sum / result->imu_samples;
    result->delayed_mean_us = delayed_sum / result->delayed_samples;
    result->healthy = ok && !lag->failed() && lag->current().healthy();
    lag->~FixedLag();
    std::free(storage);
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
    Result result{};
    const std::size_t total = paintedRun(workload, &result);
    if (baseline == 0 || total < baseline) { return 2; }
    const std::size_t used = total - baseline;
    // Every event except explicit BUFFER_LIMIT must be processed.
    const bool pass = result.healthy && result.processed >= 4U * result.delayed_samples &&
                      used <= kFilterStackBudget;
    std::printf("{\"type\":\"HOST_ESKF15_RESOURCE_MEASUREMENT\","
                "\"sizeof_fixed_lag_bytes\":%zu,\"sizeof_filter_bytes\":%zu,"
                "\"sizeof_workspace_bytes\":%zu,"
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
                sizeof(ofnav15::Workspace), baseline, used, kFilterStackBudget,
                ofnav15::FixedLag::kImuCapacity, ofnav15::FixedLag::kVisualCapacity,
                result.imu_samples, result.delayed_samples, result.processed,
                result.imu_worst_us, result.imu_mean_us,
                result.delayed_worst_us, result.delayed_mean_us,
                result.max_replay_length,
                result.stored_imu_at_worst, result.stored_events_at_worst,
                result.healthy ? "true" : "false", pass ? "true" : "false");
    return pass ? 0 : 1;
}
