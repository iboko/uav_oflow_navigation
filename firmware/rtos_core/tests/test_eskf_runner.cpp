#include "ofnav/eskf_runner.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>

namespace {

using ofnav::EskfRunner;
using ofnav::EskfRunnerMode;
using ofnav::EskfInitSource;

uint64_t fake_now = 0U;
uint64_t fakeClock() { return fake_now += 7U; }

constexpr double kG = 9.80665;
const ofnav15::Quat kLevelYaw{std::cos(.15), 0., 0., std::sin(.15)};  // yaw 0.3

// 250 Hz vehicle_imu-like packets: 4 ms deltas.
void feed(EskfRunner &r, uint64_t &t, int packets, const ofnav15::Vec3 &rate,
          const ofnav15::Vec3 &force, bool attitude = true) {
    for (int i = 0; i < packets; ++i) {
        t += 4000U;
        const double dt = .004;
        r.addImuDelta(t, {rate[0] * dt, rate[1] * dt, rate[2] * dt}, 4000U,
                      {force[0] * dt, force[1] * dt, force[2] * dt}, 4000U, false,
                      kLevelYaw, attitude);
    }
}

void test_static_alignment_then_running() {
    auto r = std::make_unique<EskfRunner>(ofnav::EskfRunnerConfig{}, fakeClock);
    uint64_t t = 1000000U;
    const double roll = .05;
    const ofnav15::Vec3 still{0., -kG * std::sin(roll), -kG * std::cos(roll)};
    // 4 ms packets accumulate to 12 ms samples: 240 packets = 80 samples.
    feed(*r, t, 240, {.002, -.001, .0005}, still);
    assert(r->mode() == EskfRunnerMode::Aligning);
    feed(*r, t, 60, {.002, -.001, .0005}, still);  // 100th sample aligns
    assert(r->mode() == EskfRunnerMode::Running);
    assert(r->initSource() == EskfInitSource::StaticAlignment);
    const auto &q = r->filter().quaternion();
    const double est_roll = std::atan2(2. * (q[0] * q[1] + q[2] * q[3]),
                                       1. - 2. * (q[1] * q[1] + q[2] * q[2]));
    assert(std::abs(est_roll - roll) < 1.e-3);
    const double yaw = std::atan2(2. * (q[0] * q[3] + q[1] * q[2]),
                                  1. - 2. * (q[2] * q[2] + q[3] * q[3]));
    assert(std::abs(yaw - .3) < 1.e-6);
    assert(std::abs(r->filter().gyroBias()[0] - .002) < 1.e-4);
    feed(*r, t, 100, {.002, -.001, .0005}, still);
    assert(r->stats().imu_samples >= 33U);
    assert(r->stats().imu_worst_us > 0U);
}

void test_moving_start_uses_external_attitude_and_fuses_velocity() {
    auto r = std::make_unique<EskfRunner>(ofnav::EskfRunnerConfig{}, fakeClock);
    uint64_t t = 1000000U;
    // Vibrating vehicle: alignment refuses, attitude fallback is used.
    for (int i = 0; i < 300; ++i) {
        const double a = (i % 2) ? 1.5 : -1.5;
        feed(*r, t, 1, {0., 0., 0.}, {a, 0., -kG});
    }
    assert(r->mode() == EskfRunnerMode::Running);
    assert(r->initSource() == EskfInitSource::ExternalAttitude);
    assert(r->stats().last_alignment == ofnav15::AlignmentStatus::NotStationary);
    feed(*r, t, 30, {0., 0., 0.}, {0., 0., -kG});
    // Delayed body velocity 40 ms in the past: replayed.
    r->addBodyVelocity(t - 40000U, 0., 0., .05);
    assert(r->stats().velocity_processed == 1U);
    // Future sample waits for IMU, then is processed.
    r->addBodyVelocity(t + 6000U, 0., 0., .05);
    assert(r->stats().velocity_deferred == 1U);
    assert(r->stats().velocity_processed == 1U);
    feed(*r, t, 5, {0., 0., 0.}, {0., 0., -kG});
    assert(r->stats().velocity_processed == 2U);
    r->addBodyVelocity(t - 1000U, 0., 0., NAN);
    assert(r->stats().velocity_dropped == 1U);
}

void test_clipping_gap_restarts_filter() {
    auto r = std::make_unique<EskfRunner>(ofnav::EskfRunnerConfig{}, fakeClock);
    uint64_t t = 1000000U;
    feed(*r, t, 300, {0., 0., 0.}, {0., 0., -kG});
    assert(r->mode() == EskfRunnerMode::Running);
    // 80 ms of clipped packets: the next sample is a gap > 50 ms.
    for (int i = 0; i < 20; ++i) {
        t += 4000U;
        r->addImuDelta(t, {0., 0., 0.}, 4000U, {0., 0., -kG * .004}, 4000U, true,
                       kLevelYaw, true);
    }
    assert(r->stats().clipping_resets == 20U);
    feed(*r, t, 5, {0., 0., 0.}, {0., 0., -kG});
    assert(r->stats().filter_restarts == 1U);
    assert(r->mode() == EskfRunnerMode::Aligning);
    feed(*r, t, 300, {0., 0., 0.}, {0., 0., -kG});
    assert(r->mode() == EskfRunnerMode::Running);
}

void test_disabled_runner_ignores_everything() {
    auto r = std::make_unique<EskfRunner>(ofnav::EskfRunnerConfig{}, fakeClock);
    r->disable();
    uint64_t t = 1000000U;
    feed(*r, t, 400, {0., 0., 0.}, {0., 0., -kG});
    r->addBodyVelocity(t, 0., 0., .1);
    assert(r->mode() == EskfRunnerMode::Disabled);
    assert(r->stats().imu_samples == 0U && r->stats().velocity_events == 0U);
}

} // namespace

int main() {
    test_static_alignment_then_running();
    test_moving_start_uses_external_attitude_and_fuses_velocity();
    test_clipping_gap_restarts_filter();
    test_disabled_runner_ignores_everything();
    std::cout << "ESKF runner tests PASS\n";
    return 0;
}
