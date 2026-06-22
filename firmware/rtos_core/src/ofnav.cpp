#include "ofnav/ofnav.hpp"

#include <algorithm>
#include <cmath>

namespace ofnav {

namespace {

constexpr float kEps = 1.0e-6F;

[[nodiscard]] float clampf(float v, float lo, float hi) noexcept {
    return std::max(lo, std::min(v, hi));
}

[[nodiscard]] bool timestampFresh(uint64_t now_us, uint64_t sample_us, uint64_t max_age_us) noexcept {
    return sample_us <= now_us && (now_us - sample_us) <= max_age_us;
}

[[nodiscard]] float sq(float v) noexcept {
    return v * v;
}

} // namespace

bool isFinite(float v) noexcept {
    return std::isfinite(v);
}

Vector3f cross(Vector3f a, Vector3f b) noexcept {
    return Vector3f{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

Vector3f matVec(Matrix3f m, Vector3f v) noexcept {
    return Vector3f{
        m(0U, 0U) * v.x + m(0U, 1U) * v.y + m(0U, 2U) * v.z,
        m(1U, 0U) * v.x + m(1U, 1U) * v.y + m(1U, 2U) * v.z,
        m(2U, 0U) * v.x + m(2U, 1U) * v.y + m(2U, 2U) * v.z,
    };
}

Vector2f rotateBodyToNav(float vx_body, float vy_body, float yaw_rad) noexcept {
    const float c = std::cos(yaw_rad);
    const float s = std::sin(yaw_rad);
    return Vector2f{c * vx_body - s * vy_body, s * vx_body + c * vy_body};
}

Vector2f rotateNavToBody(float vn, float ve, float yaw_rad) noexcept {
    const float c = std::cos(yaw_rad);
    const float s = std::sin(yaw_rad);
    return Vector2f{c * vn + s * ve, -s * vn + c * ve};
}

float heightFromRange(float range_m, float roll_rad, float pitch_rad) noexcept {
    if (!isFinite(range_m) || !isFinite(roll_rad) || !isFinite(pitch_rad)) {
        return kNaN;
    }
    return range_m * std::cos(roll_rad) * std::cos(pitch_rad);
}

const char* toString(RejectReason reason) noexcept {
    switch (reason) {
    case RejectReason::Ok: return "OK";
    case RejectReason::NonFiniteInput: return "NON_FINITE_INPUT";
    case RejectReason::BadDeltaTime: return "BAD_DELTA_TIME";
    case RejectReason::LowQuality: return "LOW_QUALITY";
    case RejectReason::BadHeight: return "BAD_HEIGHT";
    case RejectReason::HeightTooLow: return "HEIGHT_TOO_LOW";
    case RejectReason::HeightTooHigh: return "HEIGHT_TOO_HIGH";
    case RejectReason::ExcessiveTilt: return "EXCESSIVE_TILT";
    case RejectReason::FlowRateLimit: return "FLOW_RATE_LIMIT";
    case RejectReason::VelocityLimit: return "VELOCITY_LIMIT";
    case RejectReason::InnovationGate: return "INNOVATION_GATE";
    case RejectReason::StaleRange: return "STALE_RANGE";
    case RejectReason::StaleFlow: return "STALE_FLOW";
    case RejectReason::ImuInvalid: return "IMU_INVALID";
    }
    return "UNKNOWN";
}

const char* toString(NavMode mode) noexcept {
    switch (mode) {
    case NavMode::Standby: return "STANDBY";
    case NavMode::FlowNav: return "FLOW_NAV";
    case NavMode::DegradedHold: return "DEGRADED_HOLD";
    case NavMode::FailsafeLand: return "FAILSAFE_LAND";
    case NavMode::ManualRequired: return "MANUAL_REQUIRED";
    }
    return "UNKNOWN";
}

FlowVelocityEstimate FlowVelocityEstimator::update(const OpticalFlowRadSample& flow,
                                                    const RangeSample& range,
                                                    const ImuSample& imu,
                                                    const AttitudeSample& attitude) const noexcept {
    FlowVelocityEstimate out{};

    if (!flow.valid) {
        out.reason = RejectReason::StaleFlow;
        return out;
    }
    if (!range.valid) {
        out.reason = RejectReason::StaleRange;
        return out;
    }
    if (!imu.valid) {
        out.reason = RejectReason::ImuInvalid;
        return out;
    }

    if (!isFinite(flow.integration_time_s) || flow.integration_time_s <= 0.0F || flow.integration_time_s > 0.20F) {
        out.reason = RejectReason::BadDeltaTime;
        return out;
    }

    if (flow.quality < cfg_.min_quality) {
        out.reason = RejectReason::LowQuality;
        return out;
    }

    if (std::fabs(attitude.roll_rad) > cfg_.max_tilt_rad || std::fabs(attitude.pitch_rad) > cfg_.max_tilt_rad) {
        out.reason = RejectReason::ExcessiveTilt;
        return out;
    }

    const float h = heightFromRange(range.distance_m, attitude.roll_rad, attitude.pitch_rad);
    out.height_m = h;
    if (!isFinite(h)) {
        out.reason = RejectReason::BadHeight;
        return out;
    }
    if (h < cfg_.min_height_m) {
        out.reason = RejectReason::HeightTooLow;
        return out;
    }
    if (h > cfg_.max_height_m) {
        out.reason = RejectReason::HeightTooHigh;
        return out;
    }

    const float ix = flow.integrated_x_rad;
    const float iy = flow.integrated_y_rad;
    const float ixg = flow.integrated_xgyro_rad;
    const float iyg = flow.integrated_ygyro_rad;
    if (!isFinite(ix) || !isFinite(iy) || !isFinite(ixg) || !isFinite(iyg)) {
        out.reason = RejectReason::NonFiniteInput;
        return out;
    }

    const float flow_x_rate = (ix - ixg) / flow.integration_time_s;
    const float flow_y_rate = (iy - iyg) / flow.integration_time_s;
    out.flow_x_rate_rad_s = flow_x_rate;
    out.flow_y_rate_rad_s = flow_y_rate;

    if (std::fabs(flow_x_rate) > cfg_.max_abs_flow_rate_rad_s ||
        std::fabs(flow_y_rate) > cfg_.max_abs_flow_rate_rad_s) {
        out.reason = RejectReason::FlowRateLimit;
        return out;
    }

    const Vector3f velocity_sensor_frame{h * flow_y_rate * cfg_.flow_scale_y, -h * flow_x_rate * cfg_.flow_scale_x, 0.0F};
    Vector3f velocity_body_at_sensor = matVec(cfg_.body_from_sensor, velocity_sensor_frame);
    const Vector3f lever_velocity = cross(imu.gyro_rad_s, cfg_.sensor_offset_body_m);
    const Vector3f velocity_body_at_cg{
        velocity_body_at_sensor.x - lever_velocity.x,
        velocity_body_at_sensor.y - lever_velocity.y,
        velocity_body_at_sensor.z - lever_velocity.z,
    };

    const float speed = std::sqrt(sq(velocity_body_at_cg.x) + sq(velocity_body_at_cg.y));
    if (!isFinite(speed) || speed > cfg_.max_horizontal_speed_m_s * 2.5F) {
        out.reason = RejectReason::VelocityLimit;
        return out;
    }

    out.velocity_body_m_s = Vector2f{velocity_body_at_cg.x, velocity_body_at_cg.y};
    out.velocity_nav_m_s = rotateBodyToNav(velocity_body_at_cg.x, velocity_body_at_cg.y, attitude.yaw_rad);
    out.accepted = true;
    out.reason = RejectReason::Ok;
    return out;
}

HorizontalEkf::HorizontalEkf(Config cfg) noexcept : cfg_(cfg) {
    reset(0U);
}

void HorizontalEkf::setIdentity(Matrix6& m, float diag) noexcept {
    for (auto& row : m) {
        row.fill(0.0F);
    }
    for (uint8_t i = 0; i < 6U; ++i) {
        m[i][i] = diag;
    }
}

void HorizontalEkf::symmetrize(Matrix6& m) noexcept {
    for (uint8_t r = 0; r < 6U; ++r) {
        for (uint8_t c = r + 1U; c < 6U; ++c) {
            const float v = 0.5F * (m[r][c] + m[c][r]);
            m[r][c] = v;
            m[c][r] = v;
        }
    }
}

void HorizontalEkf::reset(uint64_t time_us) noexcept {
    x_ = EkfState{};
    setIdentity(p_, 0.0F);
    p_[0][0] = sq(0.20F);
    p_[1][1] = sq(0.20F);
    p_[2][2] = sq(0.12F);
    p_[3][3] = sq(0.12F);
    p_[4][4] = sq(0.03F);
    p_[5][5] = sq(0.03F);
    last_predict_us_ = time_us;
    initialized_ = true;
    last_innovation_d2_ = 0.0F;
}

void HorizontalEkf::predict(const ImuSample& imu, const AttitudeSample& attitude) noexcept {
    if (!initialized_) {
        reset(imu.time_us);
        return;
    }
    if (!imu.valid || imu.time_us <= last_predict_us_) {
        return;
    }

    float dt = static_cast<float>(imu.time_us - last_predict_us_) * 1.0e-6F;
    if (dt <= 0.0F) {
        return;
    }
    dt = clampf(dt, 0.0F, 0.20F);

    const float ax = imu.accel_m_s2.x - x_.accel_bias_x_m_s2;
    const float ay = imu.accel_m_s2.y - x_.accel_bias_y_m_s2;
    const Vector2f accel_nav = rotateBodyToNav(ax, ay, attitude.yaw_rad);

    x_.n_m += x_.vn_m_s * dt + 0.5F * accel_nav.x * dt * dt;
    x_.e_m += x_.ve_m_s * dt + 0.5F * accel_nav.y * dt * dt;
    x_.vn_m_s += accel_nav.x * dt;
    x_.ve_m_s += accel_nav.y * dt;

    const float c = std::cos(attitude.yaw_rad);
    const float s = std::sin(attitude.yaw_rad);

    Matrix6 f{};
    setIdentity(f, 1.0F);
    f[0][2] = dt;
    f[1][3] = dt;
    f[0][4] = -0.5F * c * dt * dt;
    f[0][5] = 0.5F * s * dt * dt;
    f[1][4] = -0.5F * s * dt * dt;
    f[1][5] = -0.5F * c * dt * dt;
    f[2][4] = -c * dt;
    f[2][5] = s * dt;
    f[3][4] = -s * dt;
    f[3][5] = -c * dt;

    Matrix6 fp{};
    for (uint8_t r = 0; r < 6U; ++r) {
        for (uint8_t c2 = 0; c2 < 6U; ++c2) {
            float acc = 0.0F;
            for (uint8_t k = 0; k < 6U; ++k) {
                acc += f[r][k] * p_[k][c2];
            }
            fp[r][c2] = acc;
        }
    }

    Matrix6 fpf_t{};
    for (uint8_t r = 0; r < 6U; ++r) {
        for (uint8_t c2 = 0; c2 < 6U; ++c2) {
            float acc = 0.0F;
            for (uint8_t k = 0; k < 6U; ++k) {
                acc += fp[r][k] * f[c2][k];
            }
            fpf_t[r][c2] = acc;
        }
    }

    p_ = fpf_t;
    const float q_pos = 0.002F * dt;
    const float q_vel = sq(cfg_.accel_noise_sigma_m_s2 * dt);
    const float q_bias = sq(0.002F * dt);
    p_[0][0] += q_pos;
    p_[1][1] += q_pos;
    p_[2][2] += q_vel;
    p_[3][3] += q_vel;
    p_[4][4] += q_bias;
    p_[5][5] += q_bias;
    symmetrize(p_);
    last_predict_us_ = imu.time_us;
}

bool HorizontalEkf::updateFlowVelocity(const FlowVelocityEstimate& flow, uint8_t quality) noexcept {
    if (!flow.accepted || !initialized_) {
        return false;
    }

    const float y0 = flow.velocity_nav_m_s.x - x_.vn_m_s;
    const float y1 = flow.velocity_nav_m_s.y - x_.ve_m_s;
    const float q = clampf(static_cast<float>(quality), 1.0F, 255.0F);
    float sigma = cfg_.flow_vel_sigma_min_m_s;
    sigma += cfg_.flow_vel_sigma_quality_gain_m_s * (1.0F - q / 255.0F);
    sigma += cfg_.flow_vel_sigma_height_gain_m_s * std::max(flow.height_m - 1.0F, 0.0F);
    sigma = clampf(sigma, cfg_.flow_vel_sigma_min_m_s, 0.35F);
    const float r = sq(sigma);

    const float s00 = p_[2][2] + r;
    const float s01 = p_[2][3];
    const float s10 = p_[3][2];
    const float s11 = p_[3][3] + r;
    const float det = s00 * s11 - s01 * s10;
    if (std::fabs(det) < kEps) {
        return false;
    }

    const float inv00 = s11 / det;
    const float inv01 = -s01 / det;
    const float inv10 = -s10 / det;
    const float inv11 = s00 / det;

    last_innovation_d2_ = y0 * (inv00 * y0 + inv01 * y1) + y1 * (inv10 * y0 + inv11 * y1);
    if (!isFinite(last_innovation_d2_) || last_innovation_d2_ > cfg_.innovation_gate_2d) {
        return false;
    }

    Matrix6x2 k{};
    for (uint8_t row = 0; row < 6U; ++row) {
        const float ph0 = p_[row][2];
        const float ph1 = p_[row][3];
        k[row][0] = ph0 * inv00 + ph1 * inv10;
        k[row][1] = ph0 * inv01 + ph1 * inv11;
    }

    x_.n_m += k[0][0] * y0 + k[0][1] * y1;
    x_.e_m += k[1][0] * y0 + k[1][1] * y1;
    x_.vn_m_s += k[2][0] * y0 + k[2][1] * y1;
    x_.ve_m_s += k[3][0] * y0 + k[3][1] * y1;
    x_.accel_bias_x_m_s2 += k[4][0] * y0 + k[4][1] * y1;
    x_.accel_bias_y_m_s2 += k[5][0] * y0 + k[5][1] * y1;

    Matrix6 new_p = p_;
    for (uint8_t row = 0; row < 6U; ++row) {
        for (uint8_t col = 0; col < 6U; ++col) {
            new_p[row][col] = p_[row][col] - k[row][0] * p_[2][col] - k[row][1] * p_[3][col];
        }
    }

    for (uint8_t row = 0; row < 6U; ++row) {
        for (uint8_t col = 0; col < 6U; ++col) {
            new_p[row][col] += r * (k[row][0] * k[col][0] + k[row][1] * k[col][1]);
        }
    }

    p_ = new_p;
    symmetrize(p_);
    return true;
}

RuntimeOutput SafetyMonitor::evaluate(uint64_t now_us,
                                      const ImuSample& imu,
                                      const RangeSample& range,
                                      const OpticalFlowRadSample& flow,
                                      const FlowVelocityEstimate& flow_estimate,
                                      const EkfState& ekf_state,
                                      bool innovation_ok) noexcept {
    RuntimeOutput out{};
    out.flow = flow_estimate;
    out.state = ekf_state;

    out.health.imu_valid = imu.valid;
    out.health.range_valid = range.valid && timestampFresh(now_us, range.time_us, cfg_.max_range_age_us);
    out.health.flow_valid = flow.valid && timestampFresh(now_us, flow.time_us, cfg_.max_flow_age_us);
    out.health.height_valid = isFinite(flow_estimate.height_m) && flow_estimate.height_m >= cfg_.min_height_m && flow_estimate.height_m <= cfg_.max_height_m;
    out.health.quality_valid = flow.quality >= cfg_.min_quality;
    out.health.innovation_valid = innovation_ok;
    out.health.stale_range = !out.health.range_valid;
    out.health.stale_flow = !out.health.flow_valid;
    out.health.excessive_tilt = flow_estimate.reason == RejectReason::ExcessiveTilt;

    if (!out.health.imu_valid || out.health.stale_range || out.health.excessive_tilt) {
        out.mode = NavMode::FailsafeLand;
        return out;
    }

    if (!out.health.flow_valid || !out.health.quality_valid || !out.health.height_valid || !innovation_ok) {
        out.mode = NavMode::DegradedHold;
        return out;
    }

    out.mode = flow_estimate.accepted ? NavMode::FlowNav : NavMode::DegradedHold;
    return out;
}

OfNavRuntime::OfNavRuntime(Config cfg) noexcept : cfg_(cfg), flow_estimator_(cfg), ekf_(cfg), safety_(cfg) {}

void OfNavRuntime::reset(uint64_t time_us) noexcept {
    ekf_.reset(time_us);
}

RuntimeOutput OfNavRuntime::step(const ImuSample& imu,
                                  const RangeSample& range,
                                  const OpticalFlowRadSample& flow,
                                  const AttitudeSample& attitude) noexcept {
    ekf_.predict(imu, attitude);
    FlowVelocityEstimate flow_estimate = flow_estimator_.update(flow, range, imu, attitude);
    const bool innovation_ok = ekf_.updateFlowVelocity(flow_estimate, flow.quality);

    if (flow_estimate.accepted && !innovation_ok) {
        flow_estimate.accepted = false;
        flow_estimate.reason = RejectReason::InnovationGate;
    }

    return safety_.evaluate(imu.time_us, imu, range, flow, flow_estimate, ekf_.state(), innovation_ok);
}

} // namespace ofnav
