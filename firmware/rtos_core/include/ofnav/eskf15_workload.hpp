#pragma once

#include <cstddef>
#include <cstdint>

#include "ofnav/eskf15.hpp"

// Deterministic worst-case workload for the fixed-lag ESKF, shared by the
// host benchmark (tests/eskf15_benchmark.cpp), the Cortex-M7 QEMU harness
// (tests/target/qemu_m7) and the flight-controller command 'ofnav bench'.
// 100 Hz IMU, 300 ms lag as configured in EskfRunner (PX4); every 0.1 s one camera frame
// with all five aiding kinds, 60 ms late, consistent with the same IMU
// stream (built from an unaided reference filter).
namespace ofnav15 {

struct WorkloadTruth {
    std::uint64_t t{0};
    Vec3 p{}, v{};
    Quat q{1., 0., 0., 0.};
};

// ~70 KiB: allocate once (static or heap), never on a task stack.
struct WorkloadBuffers {
    FixedLag lag{Parameters{}, 300000U};
    Filter reference{};
    Workspace reference_ws{};
    WorkloadTruth history[16]{};
    Measurement frame[kMeasurementKinds]{};
};

struct WorkloadResult {
    std::uint32_t imu_samples{0}, frames{0}, processed{0}, failed_events{0};
    std::uint64_t imu_worst{0}, imu_total{0}, frame_worst{0}, frame_total{0};
    std::size_t max_replay_length{0}, stored_imu_at_worst{0}, stored_events_at_worst{0};
    bool healthy{false};
};

// clock: monotonic tick source (any unit); nullptr disables timing.
using WorkloadClock = std::uint64_t (*)();

// imu_period_us: 10000 (100 Hz) or 20000 (50 Hz); camera at 10 Hz.
[[nodiscard]] WorkloadResult runWorkload(WorkloadBuffers &buffers, WorkloadClock clock,
                                         int frames = 30,
                                         std::uint64_t imu_period_us = 10000U) noexcept;

} // namespace ofnav15
