#include "ofnav/eskf_runner.hpp"

#include <algorithm>
#include <cmath>

namespace ofnav {
namespace {

double yawFromQuaternion(const ofnav15::Quat &q) noexcept {
    return std::atan2(2. * (q[0] * q[3] + q[1] * q[2]),
                      1. - 2. * (q[2] * q[2] + q[3] * q[3]));
}

bool finiteQuat(const ofnav15::Quat &q) noexcept {
    double n = 0.;
    for (double x : q) {
        if (!std::isfinite(x)) { return false; }
        n += x * x;
    }
    return n > .64 && n < 1.44;
}

} // namespace

EskfRunner::EskfRunner(const EskfRunnerConfig &config, Clock clock) noexcept {
    configure(config, clock);
}

void EskfRunner::configure(const EskfRunnerConfig &config, Clock clock) noexcept {
    config_ = config;
    config_.alignment_samples = std::min(std::max<std::size_t>(config_.alignment_samples, 2U),
                                         kMaxAlignmentSamples);
    config_.alignment.min_samples = std::min(config_.alignment.min_samples,
                                             config_.alignment_samples);
    clock_ = clock;
    stats_ = EskfRunnerStats{};
    restart();
}

void EskfRunner::restart() noexcept {
    mode_ = EskfRunnerMode::Aligning;
    init_source_ = EskfInitSource::None;
    alignment_count_ = 0U;
    sum_angle_ = {};
    sum_velocity_ = {};
    sum_dt_us_ = 0U;
    sum_velocity_dt_us_ = 0U;
    last_packet_us_ = 0U;
    has_pending_ = false;
}

void EskfRunner::disable() noexcept {
    mode_ = EskfRunnerMode::Disabled;
}

void EskfRunner::startFilter(const ofnav15::Parameters &parameters,
                             EskfInitSource source) noexcept {
    lag_.reset(parameters, config_.max_lag_us);
    init_source_ = source;
    mode_ = lag_.failed() ? EskfRunnerMode::Aligning : EskfRunnerMode::Running;
    // The last alignment sample starts the filter time line.
    if (mode_ == EskfRunnerMode::Running && alignment_count_ > 0U) {
        (void)lag_.pushImu(alignment_[alignment_count_ - 1U]);
    }
    alignment_count_ = 0U;
}

void EskfRunner::addImuDelta(uint64_t time_us, const ofnav15::Vec3 &dang,
                             uint32_t dang_dt_us, const ofnav15::Vec3 &dvel,
                             uint32_t dvel_dt_us, bool clipping,
                             const ofnav15::Quat &external_q,
                             bool external_q_valid) noexcept {
    if (mode_ == EskfRunnerMode::Disabled) { return; }
    const bool finite = std::isfinite(dang[0]) && std::isfinite(dang[1]) &&
                        std::isfinite(dang[2]) && std::isfinite(dvel[0]) &&
                        std::isfinite(dvel[1]) && std::isfinite(dvel[2]);
    if (clipping || !finite || dang_dt_us == 0U || dvel_dt_us == 0U ||
        time_us <= last_packet_us_) {
        // Saturated or inconsistent inertial data cannot be integrated.
        // Discard the partial sample; a resulting gap restarts the filter.
        ++stats_.clipping_resets;
        sum_angle_ = {};
        sum_velocity_ = {};
        sum_dt_us_ = 0U;
        sum_velocity_dt_us_ = 0U;
        if (time_us > last_packet_us_) { last_packet_us_ = time_us; }
        return;
    }
    last_packet_us_ = time_us;
    for (int i = 0; i < 3; ++i) {
        sum_angle_[i] += dang[i];
        sum_velocity_[i] += dvel[i];
    }
    sum_dt_us_ += dang_dt_us;
    sum_velocity_dt_us_ += dvel_dt_us;
    if (sum_dt_us_ < config_.imu_period_us) { return; }
    // Mean rate and specific force over the accumulated integration spans.
    const double dt = static_cast<double>(sum_dt_us_) * 1.e-6;
    const double dtv = static_cast<double>(sum_velocity_dt_us_) * 1.e-6;
    const ofnav15::Imu sample{time_us,
        {sum_angle_[0] / dt, sum_angle_[1] / dt, sum_angle_[2] / dt},
        {sum_velocity_[0] / dtv, sum_velocity_[1] / dtv, sum_velocity_[2] / dtv}};
    sum_angle_ = {};
    sum_velocity_ = {};
    sum_dt_us_ = 0U;
    sum_velocity_dt_us_ = 0U;
    emitImu(sample, external_q, external_q_valid);
}

void EskfRunner::emitImu(const ofnav15::Imu &sample, const ofnav15::Quat &external_q,
                         bool external_q_valid) noexcept {
    if (mode_ == EskfRunnerMode::Aligning) {
        if (alignment_count_ > 0U &&
            sample.time_us - alignment_[alignment_count_ - 1U].time_us >
                static_cast<uint64_t>(config_.filter.max_dt_s * 1.e6)) {
            alignment_count_ = 0U;  // a gap breaks the stationary window
        }
        alignment_[alignment_count_++] = sample;
        if (alignment_count_ < config_.alignment_samples) { return; }
        const bool have_yaw = external_q_valid && finiteQuat(external_q);
        ofnav15::Parameters p = config_.filter;
        stats_.last_alignment = ofnav15::staticAlignment(
            alignment_.data(), alignment_count_,
            have_yaw ? yawFromQuaternion(external_q) : 0.,
            have_yaw ? config_.external_yaw_sigma_rad : 3.14159265358979323846,
            config_.alignment, p);
        if (stats_.last_alignment == ofnav15::AlignmentStatus::Aligned) {
            startFilter(p, EskfInitSource::StaticAlignment);
        } else if (have_yaw) {
            // Moving vehicle: start from the autopilot attitude with a
            // conservative tilt/yaw uncertainty.
            p.initial_quaternion_body_to_ned = external_q;
            p.initial_sigma[6] = p.initial_sigma[7] = config_.external_tilt_sigma_rad;
            p.initial_sigma[8] = config_.external_yaw_sigma_rad;
            startFilter(p, EskfInitSource::ExternalAttitude);
        } else {
            alignment_count_ = 0U;
        }
        return;
    }
    const uint64_t start = now();
    stats_.last_status = lag_.pushImu(sample);
    const uint64_t elapsed = now() - start;
    ++stats_.imu_samples;
    stats_.imu_total_us += elapsed;
    stats_.imu_worst_us = std::max(stats_.imu_worst_us, elapsed);
    if (lag_.failed() || !lag_.current().healthy()) {
        ++stats_.filter_restarts;
        restart();
        return;
    }
    if (has_pending_ && pending_.time_us <= lag_.latestImuTime()) {
        has_pending_ = false;
        pushVelocity(pending_);
    }
}

void EskfRunner::pushVelocity(const ofnav15::Measurement &m) noexcept {
    ofnav15::Status status = ofnav15::Status::ReplayFailed;
    const uint64_t start = now();
    (void)lag_.pushDelayedFrame(&m, 1U, &status);
    const uint64_t elapsed = now() - start;
    stats_.frame_total_us += elapsed;
    stats_.frame_worst_us = std::max(stats_.frame_worst_us, elapsed);
    stats_.max_replay_length = std::max(stats_.max_replay_length, lag_.lastReplayLength());
    stats_.last_status = status;
    if (ofnav15::measurementProcessed(status)) {
        ++stats_.velocity_processed;
    } else {
        ++stats_.velocity_dropped;
    }
}

void EskfRunner::addBodyVelocity(uint64_t time_us, double vx, double vy,
                                 double sigma) noexcept {
    if (mode_ != EskfRunnerMode::Running) { return; }
    ++stats_.velocity_events;
    if (!std::isfinite(sigma) || !(sigma > 0.)) {
        ++stats_.velocity_dropped;
        return;
    }
    ofnav15::Measurement m{};
    m.time_us = time_us;
    m.kind = ofnav15::MeasurementKind::BodyVelocityXy;
    m.value = {vx, vy};
    m.covariance = {sigma * sigma, 0., 0., sigma * sigma};
    if (time_us > lag_.latestImuTime()) {
        // Flow sample newer than the last integrated IMU sample: keep the
        // newest one until inertial data catches up.
        if (has_pending_) { ++stats_.velocity_dropped; }
        pending_ = m;
        has_pending_ = true;
        ++stats_.velocity_deferred;
        return;
    }
    pushVelocity(m);
}

} // namespace ofnav
