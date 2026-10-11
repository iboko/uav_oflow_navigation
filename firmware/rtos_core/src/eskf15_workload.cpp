#include "ofnav/eskf15_workload.hpp"

#include <cmath>

namespace ofnav15 {
namespace {

Imu imuSample(std::uint64_t t, int i) noexcept {
    const double s = static_cast<double>(i) * .01;
    return Imu{t, {.01 * std::sin(.7 * s), .01 * std::cos(.5 * s), .02},
               {.2 * std::sin(.37 * s), .1 * std::cos(.31 * s),
                -9.80665 + .02 * std::sin(.41 * s)}};
}

Measurement event(const WorkloadTruth &x, MeasurementKind kind, int n) noexcept {
    Measurement m{};
    m.time_us = x.t;
    m.kind = kind;
    const double w = x.q[0], qx = x.q[1], qy = x.q[2], qz = x.q[3];
    const double r[9] = {1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - w * qz), 2 * (qx * qz + w * qy),
                         2 * (qx * qy + w * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - w * qx),
                         2 * (qx * qz - w * qy), 2 * (qy * qz + w * qx), 1 - 2 * (qx * qx + qy * qy)};
    const double e = .01 * std::sin(1.3 * n);
    switch (kind) {
    case MeasurementKind::VelocityNe:
        m.value = {x.v[0] + e, x.v[1] - e};
        m.covariance = {0.04, 0., 0., 0.04};
        break;
    case MeasurementKind::BodyVelocityXy:
        m.value = {r[0] * x.v[0] + r[3] * x.v[1] + r[6] * x.v[2] + e,
                   r[1] * x.v[0] + r[4] * x.v[1] + r[7] * x.v[2] - e};
        m.covariance = {0.04, 0., 0., 0.04};
        break;
    case MeasurementKind::Yaw:
        m.value = {std::atan2(r[3], r[0]) + .1 * e, 0.};
        m.covariance = {0.0004, 0., 0., 0.};
        break;
    case MeasurementKind::PositionNe:
        m.value = {x.p[0] + 10. * e, x.p[1] - 10. * e};
        m.covariance = {4., 0., 0., 4.};
        m.ci_weight = 0.;  // automatic covariance intersection
        break;
    case MeasurementKind::MapGroundPointNe:
        // Nadir axis d_b = (0,0,1): ground point = p + h (R d)_NE / (R d)_D.
        m.value = {x.p[0] + 100. * r[2] / r[8] + e, x.p[1] + 100. * r[5] / r[8] - e};
        m.covariance = {.04, 0., 0., .04};
        m.ci_weight = 0.;
        m.aux = {100., 0., 0., 1.};
        break;
    }
    return m;
}

std::uint64_t ticks(WorkloadClock clock) noexcept { return clock ? clock() : 0U; }

} // namespace

WorkloadResult runWorkload(WorkloadBuffers &b, WorkloadClock clock, int frames,
                           std::uint64_t imu_period_us) noexcept {
    WorkloadResult result{};
    if (imu_period_us < 5000U || imu_period_us > 50000U || 100000U % imu_period_us != 0U) {
        return result;
    }
    const int per_frame = static_cast<int>(100000U / imu_period_us);
    const int delay = static_cast<int>((60000U + imu_period_us - 1U) / imu_period_us);
    b.lag.reset(Parameters{}, 300000U);
    b.reference.reset(Parameters{});
    std::uint64_t t = 1000000U;
    int i = 0;
    bool ok = true;
    for (int frame = 0; frame < frames && ok; ++frame) {
        for (int k = 0; k < per_frame; ++k, ++i) {
            const Imu sample = imuSample(t, i * static_cast<int>(imu_period_us / 10000U));
            (void)b.reference.predict(sample, b.reference_ws);
            b.history[i % 16] = WorkloadTruth{t, b.reference.position(), b.reference.velocity(),
                                              b.reference.quaternion()};
            const std::uint64_t a = ticks(clock);
            const Status s = b.lag.pushImu(sample);
            const std::uint64_t d = ticks(clock) - a;
            result.imu_total += d;
            result.imu_worst = d > result.imu_worst ? d : result.imu_worst;
            ++result.imu_samples;
            if (s != Status::InertialOnly) {
                ok = false;
                break;
            }
            t += imu_period_us;
        }
        if (!ok) { break; }
        const WorkloadTruth &x = b.history[(i - delay) % 16];
        for (std::size_t k = 0; k < kMeasurementKinds; ++k) {
            b.frame[k] = event(x, static_cast<MeasurementKind>(k), frame);
        }
        Status statuses[kMeasurementKinds]{};
        const std::uint64_t a = ticks(clock);
        (void)b.lag.pushDelayedFrame(b.frame, kMeasurementKinds, statuses);
        const std::uint64_t d = ticks(clock) - a;
        result.frame_total += d;
        ++result.frames;
        if (b.lag.lastReplayLength() > result.max_replay_length) {
            result.max_replay_length = b.lag.lastReplayLength();
        }
        if (d > result.frame_worst) {
            result.frame_worst = d;
            result.stored_imu_at_worst = b.lag.storedImu();
            result.stored_events_at_worst = b.lag.storedVisual();
        }
        for (const Status s : statuses) {
            if (measurementProcessed(s)) { ++result.processed; }
            else if (s != Status::BufferLimit) { ++result.failed_events; }
        }
    }
    result.healthy = ok && !b.lag.failed() && b.lag.current().healthy();
    return result;
}

} // namespace ofnav15
