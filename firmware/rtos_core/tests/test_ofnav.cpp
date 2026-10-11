#include "ofnav/ofnav.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

namespace {

bool near(float a, float b, float eps = 1.0e-4F) {
    return std::fabs(a - b) <= eps;
}

ofnav::Config baseConfig() {
    ofnav::Config cfg{};
    cfg.min_quality = 100U;
    cfg.min_height_m = 0.2F;
    cfg.max_height_m = 4.0F;
    cfg.max_horizontal_speed_m_s = 2.0F;
    cfg.sensor_offset_body_m = ofnav::Vector3f{0.10F, 0.0F, 0.0F};
    return cfg;
}

void test_forward_motion_sign() {
    const auto cfg = baseConfig();
    ofnav::FlowVelocityEstimator estimator(cfg);
    const float dt = 0.04F;
    const float height = 1.0F;
    const float vx_body = 0.25F;
    const ofnav::OpticalFlowRadSample flow{1000000U, dt, 0.0F, (vx_body / height) * dt, 0.0F, 0.0F, 0.0F, 220U, true};
    const ofnav::RangeSample range{1000000U, height, true};
    const ofnav::ImuSample imu{1000000U, {}, {}, true};
    const ofnav::AttitudeSample attitude{1000000U, 0.0F, 0.0F, 0.0F};
    const auto result = estimator.update(flow, range, imu, attitude);
    assert(result.accepted);
    assert(near(result.velocity_body_m_s.x, vx_body));
    assert(near(result.velocity_body_m_s.y, 0.0F));
    assert(near(result.velocity_nav_m_s.x, vx_body));
}

void test_right_motion_sign() {
    const auto cfg = baseConfig();
    ofnav::FlowVelocityEstimator estimator(cfg);
    const float dt = 0.04F;
    const float height = 1.0F;
    const float vy_body = 0.20F;
    const ofnav::OpticalFlowRadSample flow{1000000U, dt, (-vy_body / height) * dt, 0.0F, 0.0F, 0.0F, 0.0F, 220U, true};
    const ofnav::RangeSample range{1000000U, height, true};
    const ofnav::ImuSample imu{1000000U, {}, {}, true};
    const ofnav::AttitudeSample attitude{1000000U, 0.0F, 0.0F, 0.0F};
    const auto result = estimator.update(flow, range, imu, attitude);
    assert(result.accepted);
    assert(near(result.velocity_body_m_s.x, 0.0F));
    assert(near(result.velocity_body_m_s.y, vy_body));
}

void test_lever_arm_correction() {
    const auto cfg = baseConfig();
    ofnav::FlowVelocityEstimator estimator(cfg);
    const float dt = 0.04F;
    const float height = 1.0F;
    const float sensor_vy = 0.10F;
    const ofnav::OpticalFlowRadSample flow{1000000U, dt, (-sensor_vy / height) * dt, 0.0F, 0.0F, 0.0F, 1.0F * dt, 220U, true};
    const ofnav::RangeSample range{1000000U, height, true};
    const ofnav::ImuSample imu{1000000U, {0.0F, 0.0F, 1.0F}, {}, true};
    const ofnav::AttitudeSample attitude{1000000U, 0.0F, 0.0F, 0.0F};
    const auto result = estimator.update(flow, range, imu, attitude);
    assert(result.accepted);
    assert(near(result.velocity_body_m_s.x, 0.0F));
    assert(near(result.velocity_body_m_s.y, 0.0F));
}

void test_quality_rejection() {
    const auto cfg = baseConfig();
    ofnav::FlowVelocityEstimator estimator(cfg);
    const ofnav::OpticalFlowRadSample flow{1000000U, 0.04F, 0.0F, 0.01F, 0.0F, 0.0F, 0.0F, 20U, true};
    const ofnav::RangeSample range{1000000U, 1.0F, true};
    const ofnav::ImuSample imu{1000000U, {}, {}, true};
    const ofnav::AttitudeSample attitude{1000000U, 0.0F, 0.0F, 0.0F};
    const auto result = estimator.update(flow, range, imu, attitude);
    assert(!result.accepted);
    assert(result.reason == ofnav::RejectReason::LowQuality);
}

void test_runtime_failsafe_without_range() {
    auto cfg = baseConfig();
    ofnav::OfNavRuntime runtime(cfg);
    const ofnav::ImuSample imu{1000000U, {}, {}, true};
    const ofnav::RangeSample range{1000000U, ofnav::kNaN, false};
    const ofnav::OpticalFlowRadSample flow{1000000U, 0.04F, 0.0F, 0.01F, 0.0F, 0.0F, 0.0F, 220U, true};
    const ofnav::AttitudeSample attitude{1000000U, 0.0F, 0.0F, 0.0F};
    const auto out = runtime.step(imu, range, flow, attitude);
    assert(out.mode == ofnav::NavMode::FailsafeLand);
}

void test_ekf_innovation_gate_rejects_outlier() {
    auto cfg = baseConfig();
    cfg.innovation_gate_2d = 5.99F;
    ofnav::HorizontalEkf ekf(cfg);
    ekf.reset(0U);
    const ofnav::FlowVelocityEstimate outlier{true, ofnav::RejectReason::Ok, 1.0F, {10.0F, 0.0F}, {10.0F, 0.0F}, 0.0F, 0.0F};
    const bool ok = ekf.updateFlowVelocity(outlier, 220U);
    assert(!ok);
    assert(ekf.lastInnovationD2() > cfg.innovation_gate_2d);
}

void test_velocity_reset_after_aiding_outage_but_not_during_aiding() {
    auto cfg = baseConfig();
    ofnav::OfNavRuntime runtime(cfg);
    runtime.reset(1000000U);
    // Right after a reset an outlier is gated: no unaided interval yet.
    {
        const uint64_t t0 = 1001000U;
        runtime.predictImu({t0, {}, {0.0F, 0.0F, -9.81F}, true}, {t0, 0.0F, 0.0F, 0.0F});
        const ofnav::OpticalFlowRadSample fast{t0, 0.04F, 0.04F, 0.0F, 0.0F, 0.0F, 0.0F, 220U, true};
        const auto out = runtime.step({t0, {}, {0.0F, 0.0F, -9.81F}, true}, {t0, 1.0F, true}, fast,
                                      {t0, 0.0F, 0.0F, 0.0F}, t0);
        assert(out.flow.reason == ofnav::RejectReason::InnovationGate);
        assert(runtime.velocityResets() == 0U);
    }
    // 3 s unaided with a 1 m/s^2 accelerometer error: ~3 m/s velocity drift.
    uint64_t t = 1001000U;
    for (int i = 0; i < 300; ++i) {
        t += 10000U;
        runtime.predictImu({t, {}, {1.0F, 0.0F, -9.81F}, true}, {t, 0.0F, 0.0F, 0.0F});
    }
    const ofnav::RangeSample range{t, 1.0F, true};
    const ofnav::OpticalFlowRadSample still{t, 0.04F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 220U, true};
    const auto first = runtime.step({t, {}, {0.0F, 0.0F, -9.81F}, true}, range, still,
                                    {t, 0.0F, 0.0F, 0.0F}, t);
    assert(first.flow.accepted);
    assert(runtime.velocityResets() == 1U);
    assert(near(first.state.vn_m_s, 0.0F, 1.0e-3F));
    // Continuous aiding: an outlier right after a fusion stays gated.
    t += 40000U;
    runtime.predictImu({t, {}, {0.0F, 0.0F, -9.81F}, true}, {t, 0.0F, 0.0F, 0.0F});
    const ofnav::OpticalFlowRadSample jump{t, 0.04F, 0.04F, 0.0F, 0.0F, 0.0F, 0.0F, 220U, true};
    const auto second = runtime.step({t, {}, {0.0F, 0.0F, -9.81F}, true}, {t, 1.0F, true}, jump,
                                     {t, 0.0F, 0.0F, 0.0F}, t);
    assert(!second.flow.accepted);
    assert(second.flow.reason == ofnav::RejectReason::InnovationGate);
    assert(runtime.velocityResets() == 1U);
}

void test_safety_action_config_validation() {
    ofnav::SafetyActionConfig cfg{};
    assert(ofnav::validSafetyActionConfig(cfg));
    cfg.degraded_to_land_us = cfg.degraded_confirm_us;  // escalation must come later
    assert(!ofnav::validSafetyActionConfig(cfg));
    cfg = ofnav::SafetyActionConfig{};
    cfg.failsafe_confirm_us = 0U;
    assert(!ofnav::validSafetyActionConfig(cfg));
}

void test_range_rate_is_a_measured_vertical_velocity() {
    ofnav::RangeRateEstimator rate(0.02F);
    float vd = 0.0F;
    float sigma = 0.0F;
    assert(!rate.push(1000000U, 2.0F, vd, sigma));          // first sample: no rate
    assert(rate.push(1100000U, 2.1F, vd, sigma));           // climbing 1 m/s
    assert(near(vd, -1.0F, 1.0e-4F));
    assert(near(sigma, 1.41421356F * 0.02F / 0.1F, 1.0e-5F));
    assert(!rate.push(1105000U, 2.1F, vd, sigma));          // < 20 ms: refused
    assert(!rate.push(1600000U, 2.1F, vd, sigma));          // > 300 ms gap: refused
    assert(!rate.push(1700000U, ofnav::kNaN, vd, sigma));    // invalid height resets
    assert(!rate.push(1800000U, 2.0F, vd, sigma));
}

void test_body_velocity_3d_from_horizontal_body_and_vertical_ned() {
    ofnav::BodyVelocity3d out{};
    // Level: body z equals NED down.
    assert(ofnav::bodyVelocity3d({1.0F, -0.5F}, 0.1F, 0.3F, 0.2F, {0U, 0.0F, 0.0F, 1.0F}, 0.45F, out));
    assert(near(out.velocity_m_s.z, 0.3F, 1.0e-6F));
    assert(near(out.variance_m2_s2.z, 0.04F, 1.0e-6F));
    // Pitched 0.2 rad, level forward flight: NED down velocity zero.
    const float pitch = 0.2F;
    assert(ofnav::bodyVelocity3d({2.0F, 0.0F}, 0.1F, 0.0F, 0.2F, {0U, 0.0F, pitch, 0.0F}, 0.45F, out));
    // R20*vx + R22*vz = 0  ->  vz = sin(p)*vx/cos(p)
    assert(near(out.velocity_m_s.z, 2.0F * std::tan(pitch), 1.0e-5F));
    assert(out.variance_m2_s2.z > 0.04F);
    // Beyond the tilt envelope and non-finite input are refused.
    assert(!ofnav::bodyVelocity3d({1.0F, 0.0F}, 0.1F, 0.0F, 0.2F, {0U, 0.6F, 0.0F, 0.0F}, 0.45F, out));
    assert(!ofnav::bodyVelocity3d({ofnav::kNaN, 0.0F}, 0.1F, 0.0F, 0.2F, {0U, 0.0F, 0.0F, 0.0F}, 0.45F, out));
}

void test_covariance_joseph_update() {
    auto cfg = baseConfig();
    ofnav::HorizontalEkf ekf(cfg);
    ekf.reset(0U);
    const float p_before = ekf.covariance(2, 2);
    const ofnav::FlowVelocityEstimate flow{true, ofnav::RejectReason::Ok,
        1.0F, {0.0F, 0.0F}, {0.0F, 0.0F}, 0.0F, 0.0F};
    assert(ekf.updateFlowVelocity(flow, 220U));
    const float quality = 220.0F;
    float sigma = cfg.flow_vel_sigma_min_m_s +
        cfg.flow_vel_sigma_quality_gain_m_s * (1.0F - quality / 255.0F);
    const float r = sigma * sigma;
    const float expected = p_before * r / (p_before + r);
    assert(near(ekf.covariance(2, 2), expected, 1.0e-6F));
    assert(ekf.covariance(2, 2) > 0.0F);
}

void test_gravity_projection_does_not_create_horizontal_velocity() {
    const auto cfg = baseConfig();
    ofnav::HorizontalEkf ekf(cfg);
    ekf.reset(1000000U);
    const float pitch = 0.20F;
    const float g = 9.81F;
    // NED gravity, specific force for a stationary body with pitch.
    const ofnav::ImuSample imu{1010000U, {},
        {g * std::sin(pitch), 0.0F, -g * std::cos(pitch)}, true};
    const ofnav::AttitudeSample att{1010000U, 0.0F, pitch, 0.0F};
    ekf.predict(imu, att);
    assert(near(ekf.state().vn_m_s, 0.0F, 1.0e-4F));
    assert(near(ekf.state().ve_m_s, 0.0F, 1.0e-4F));
}

void test_stale_flow_is_never_fused() {
    auto cfg = baseConfig();
    cfg.max_flow_age_us = 100000U;
    ofnav::OfNavRuntime runtime(cfg);
    runtime.reset(1000000U);
    const ofnav::ImuSample imu{1200000U, {}, {}, true};
    const ofnav::RangeSample range{1200000U, 1.0F, true};
    const ofnav::OpticalFlowRadSample flow{1000000U, 0.04F, 0.0F, 0.01F,
        0.0F, 0.0F, 0.0F, 220U, true};
    const ofnav::AttitudeSample att{1200000U, 0.0F, 0.0F, 0.0F};
    const auto out = runtime.step(imu, range, flow, att, 1200000U);
    assert(out.mode != ofnav::NavMode::FlowNav);
    assert(!out.flow.accepted);
}

void test_timeout_without_new_flow_samples() {
    auto cfg = baseConfig();
    ofnav::OfNavRuntime runtime(cfg);
    const ofnav::ImuSample imu{1200000U, {}, {}, true};
    const ofnav::RangeSample range{1200000U, 1.0F, true};
    const ofnav::OpticalFlowRadSample flow{1000000U, 0.04F, 0.0F, 0.01F,
        0.0F, 0.0F, 0.0F, 220U, true};
    const ofnav::AttitudeSample att{1200000U, 0.0F, 0.0F, 0.0F};
    const auto out = runtime.monitor(1200000U, imu, range, flow, att);
    assert(out.mode == ofnav::NavMode::DegradedHold);
    assert(out.health.stale_flow);
}

void test_asynchronous_imu_prediction_between_flow_frames() {
    ofnav::Config cfg = baseConfig();
    ofnav::OfNavRuntime runtime(cfg);
    runtime.reset(1000000U);
    const ofnav::AttitudeSample attitude{1010000U, 0.0F, 0.0F, 0.0F};
    const ofnav::ImuSample sample1{1010000U, {}, {1.0F, 0.0F, -9.81F}, true};
    runtime.predictImu(sample1, attitude);
    const ofnav::AttitudeSample attitude2{1020000U, 0.0F, 0.0F, 0.0F};
    const ofnav::ImuSample sample2{1020000U, {}, {1.0F, 0.0F, -9.81F}, true};
    runtime.predictImu(sample2, attitude2);
    const ofnav::RangeSample range{1020000U, 1.0F, true};
    const ofnav::OpticalFlowRadSample flow{};  // No optical-flow update at all.
    const auto result = runtime.monitor(1020000U, sample2, range, flow, attitude2);
    assert(near(result.state.vn_m_s, 0.02F, 1.0e-5F));
    assert(result.mode != ofnav::NavMode::FlowNav);
}

void test_reused_imu_timestamp_not_predicted_twice() {
    ofnav::Config cfg = baseConfig();
    ofnav::OfNavRuntime runtime(cfg);
    runtime.reset(1000000U);
    const ofnav::AttitudeSample att{1010000U, 0.0F, 0.0F, 0.0F};
    const ofnav::ImuSample imu{1010000U, {}, {1.0F, 0.0F, -9.81F}, true};
    runtime.predictImu(imu, att);
    runtime.predictImu(imu, att);
    const auto result = runtime.monitor(1010000U, imu, {}, {}, att);
    assert(near(result.state.vn_m_s, 0.01F, 1.0e-5F));
}

void test_missing_gyro_compensation_is_rejected() {
    const auto cfg = baseConfig();
    ofnav::FlowVelocityEstimator estimator(cfg);
    const ofnav::OpticalFlowRadSample flow{1000000U, 0.04F, 0.01F, 0.01F,
        ofnav::kNaN, ofnav::kNaN, ofnav::kNaN, 220U, true};
    const ofnav::RangeSample range{1000000U, 1.0F, true};
    const ofnav::ImuSample imu{1000000U, {}, {}, true};
    const ofnav::AttitudeSample att{1000000U, 0.0F, 0.0F, 0.0F};
    const auto out = estimator.update(flow, range, imu, att);
    assert(!out.accepted);
    assert(out.reason == ofnav::RejectReason::NonFiniteInput);
}

void test_velocity_rotation_respects_pitch() {
    auto cfg = baseConfig();
    cfg.sensor_offset_body_m = {0.0F, 0.0F, 0.0F};
    ofnav::FlowVelocityEstimator estimator(cfg);
    const float pitch = 0.20F;
    const float height = 1.0F;
    const float v_sensor_x = 0.25F;
    const float dt = 0.04F;
    const ofnav::OpticalFlowRadSample flow{1000000U, dt, 0.0F,
        v_sensor_x / height * dt, 0.0F, 0.0F, 0.0F, 220U, true};
    const ofnav::RangeSample range{1000000U, height / std::cos(pitch), true};
    const ofnav::ImuSample imu{1000000U, {}, {}, true};
    const ofnav::AttitudeSample attitude{1000000U, 0.0F, pitch, 0.0F};
    const auto result = estimator.update(flow, range, imu, attitude);
    assert(result.accepted);
    assert(near(result.velocity_body_m_s.x, v_sensor_x));
    assert(near(result.velocity_nav_m_s.x, v_sensor_x * std::cos(pitch)));
}

void test_range_sample_skew_must_reject_flow_correction() {
    auto cfg = baseConfig();
    cfg.max_measurement_skew_us = 10000U;
    ofnav::OfNavRuntime runtime(cfg);
    runtime.reset(1000000U);
    const ofnav::ImuSample imu{1080000U, {}, {}, true};
    const ofnav::RangeSample range{1000000U, 1.0F, true};
    const ofnav::OpticalFlowRadSample flow{1080000U, 0.04F,
        0.0F, 0.002F, 0.0F, 0.0F, 0.0F, 220U, true};
    const ofnav::AttitudeSample attitude{1080000U, 0.0F, 0.0F, 0.0F};
    const auto result = runtime.step(imu, range, flow, attitude, 1080000U);
    assert(!result.flow.accepted);
    assert(result.mode != ofnav::NavMode::FlowNav);
}

void test_excessive_tilt_detected_during_flow_dropout() {
    const auto cfg = baseConfig();
    ofnav::OfNavRuntime runtime(cfg);
    runtime.reset(1000000U);
    const ofnav::ImuSample imu{1010000U, {}, {0.0F, 0.0F, -9.81F}, true};
    const ofnav::RangeSample range{1010000U, 1.0F, true};
    const ofnav::AttitudeSample attitude{
        1010000U, 2.0F * cfg.max_tilt_rad, 0.0F, 0.0F
    };
    const auto result = runtime.monitor(1010000U, imu, range, {}, attitude);
    assert(result.health.excessive_tilt);
    assert(result.mode == ofnav::NavMode::FailsafeLand);
}

void test_nonfinite_imu_does_not_enter_flow_mode() {
    const auto cfg = baseConfig();
    ofnav::OfNavRuntime runtime(cfg);
    runtime.reset(1000000U);
    const ofnav::ImuSample imu{1010000U, {},
        {ofnav::kNaN, 0.0F, -9.81F}, true};
    const ofnav::RangeSample range{1010000U, 1.0F, true};
    const ofnav::OpticalFlowRadSample flow{1010000U, 0.04F,
        0.0F, 0.01F, 0.0F, 0.0F, 0.0F, 220U, true};
    const ofnav::AttitudeSample attitude{1010000U, 0.0F, 0.0F, 0.0F};
    const auto result = runtime.step(imu, range, flow, attitude, 1010000U);
    assert(!result.health.imu_valid);
    assert(!result.flow.accepted);
    assert(result.mode == ofnav::NavMode::FailsafeLand);
}

void test_missing_imu_history_requires_explicit_reset() {
    const auto cfg = baseConfig();
    ofnav::OfNavRuntime runtime(cfg);
    runtime.reset(1000000U);
    const ofnav::ImuSample imu_gap{1250000U, {},
        {0.0F, 0.0F, -9.81F}, true};
    const ofnav::RangeSample range{1250000U, 1.0F, true};
    const ofnav::OpticalFlowRadSample flow{1250000U, 0.04F,
        0.0F, 0.01F, 0.0F, 0.0F, 0.0F, 220U, true};
    const ofnav::AttitudeSample att{1250000U, 0.0F, 0.0F, 0.0F};
    auto out = runtime.step(imu_gap, range, flow, att, 1250000U);
    assert(out.mode == ofnav::NavMode::FailsafeLand);
    assert(!out.flow.accepted);

    // New valid samples must not silently re-arm a discontinuous estimator.
    const ofnav::ImuSample imu_next{1260000U, {},
        {0.0F, 0.0F, -9.81F}, true};
    const ofnav::AttitudeSample att_next{1260000U, 0.0F, 0.0F, 0.0F};
    out = runtime.monitor(1260000U, imu_next, range, flow, att_next);
    assert(out.mode == ofnav::NavMode::FailsafeLand);

    runtime.reset(1260000U);
    const auto reset_out = runtime.monitor(1260000U, imu_next, range, flow, att_next);
    assert(reset_out.mode != ofnav::NavMode::FailsafeLand);
}

void test_combined_roll_pitch_exceeds_inclination_limit() {
    const auto cfg = baseConfig();
    ofnav::FlowVelocityEstimator estimator(cfg);
    const ofnav::OpticalFlowRadSample flow{1000000U, 0.04F,
        0.0F, 0.003F, 0.0F, 0.0F, 0.0F, 220U, true};
    const ofnav::RangeSample range{1000000U, 1.2F, true};
    const ofnav::ImuSample imu{1000000U, {}, {}, true};
    const ofnav::AttitudeSample att{1000000U, 0.35F, 0.35F, 0.0F};
    const auto result = estimator.update(flow, range, imu, att);
    assert(!result.accepted);
    assert(result.reason == ofnav::RejectReason::ExcessiveTilt);
}

ofnav::OpticalFlowRadSample cameraFlow(float height, float vx_body, float dt) {
    return ofnav::OpticalFlowRadSample{1000000U, dt, 0.0F, (vx_body / height) * dt,
                                       0.0F, 0.0F, 0.0F, 200U, true};
}

void test_flow_sensor_profile_still_rejects_100m() {
    const auto cfg = ofnav::makeProfileConfig(ofnav::SensorProfile::OpticalFlowSensor);
    assert(near(cfg.max_height_m, 4.0F));
    ofnav::FlowVelocityEstimator estimator(cfg);
    const ofnav::RangeSample range{1000000U, 100.0F, true};
    const ofnav::ImuSample imu{1000000U, {}, {}, true};
    const ofnav::AttitudeSample att{1000000U, 0.0F, 0.0F, 0.0F};
    const auto result = estimator.update(cameraFlow(100.0F, 1.0F, 0.2F), range, imu, att);
    assert(!result.accepted);
    assert(result.reason == ofnav::RejectReason::HeightTooHigh);
}

void test_camera_profile_accepts_100m_with_height_scaled_sigma() {
    const auto cfg = ofnav::makeProfileConfig(ofnav::SensorProfile::DownwardCameraVo);
    ofnav::FlowVelocityEstimator estimator(cfg);
    const ofnav::RangeSample range{1000000U, 100.0F, true};
    const ofnav::ImuSample imu{1000000U, {}, {}, true};
    const ofnav::AttitudeSample att{1000000U, 0.0F, 0.0F, 0.0F};
    const auto result = estimator.update(cameraFlow(100.0F, 1.2F, 0.2F), range, imu, att);
    assert(result.accepted);
    assert(near(result.velocity_nav_m_s.x, 1.2F, 1.0e-3F));
    // sqrt(0.05^2 + (100*0.0015)^2 + (1.2*0.01)^2)
    const float expected = std::sqrt(0.0025F + 0.0225F + 0.000144F);
    assert(near(result.velocity_sigma_m_s, expected, 1.0e-4F));
    const auto low = estimator.update(cameraFlow(20.0F, 1.2F, 0.2F),
        ofnav::RangeSample{1000000U, 20.0F, true}, imu, att);
    assert(low.accepted && low.velocity_sigma_m_s < result.velocity_sigma_m_s);
}

void test_camera_profile_rejects_coarse_measurement_by_uncertainty() {
    auto cfg = ofnav::makeProfileConfig(ofnav::SensorProfile::DownwardCameraVo);
    cfg.max_velocity_sigma_m_s = 0.12F;  // 100 m * 1.5 mrad/s = 0.15 m/s
    ofnav::FlowVelocityEstimator estimator(cfg);
    const ofnav::ImuSample imu{1000000U, {}, {}, true};
    const ofnav::AttitudeSample att{1000000U, 0.0F, 0.0F, 0.0F};
    const auto result = estimator.update(cameraFlow(100.0F, 1.0F, 0.2F),
        ofnav::RangeSample{1000000U, 100.0F, true}, imu, att);
    assert(!result.accepted);
    assert(result.reason == ofnav::RejectReason::VelocityUncertainty);
    const auto below = estimator.update(cameraFlow(3.0F, 1.0F, 0.2F),
        ofnav::RangeSample{1000000U, 3.0F, true}, imu, att);
    assert(below.reason == ofnav::RejectReason::HeightTooLow);
}

void test_camera_profile_ekf_uses_estimator_sigma() {
    const auto cfg = ofnav::makeProfileConfig(ofnav::SensorProfile::DownwardCameraVo);
    ofnav::OfNavRuntime runtime(cfg);
    const ofnav::AttitudeSample att{1000000U, 0.0F, 0.0F, 0.0F};
    for (uint64_t t = 1000000U; t <= 1200000U; t += 10000U) {
        const ofnav::ImuSample imu{t, {}, {0.0F, 0.0F, -9.80665F}, true};
        runtime.predictImu(imu, ofnav::AttitudeSample{t, 0.0F, 0.0F, 0.0F});
    }
    const uint64_t t = 1200000U;
    const ofnav::ImuSample imu{t, {}, {0.0F, 0.0F, -9.80665F}, true};
    ofnav::OpticalFlowRadSample flow = cameraFlow(100.0F, 0.3F, 0.2F);
    flow.time_us = t;
    const auto out = runtime.step(imu, ofnav::RangeSample{t, 100.0F, true}, flow,
                                  ofnav::AttitudeSample{t, 0.0F, 0.0F, 0.0F}, t);
    (void)att;
    assert(out.flow.accepted);
    assert(out.mode == ofnav::NavMode::FlowNav);
    assert(out.state.vn_m_s > 0.0F && out.state.vn_m_s < 0.3F);
}

ofnav::SafetyActionConfig unengaged() {
    ofnav::SafetyActionConfig cfg{};
    cfg.require_engagement = false;  // these tests start mid-flight
    return cfg;
}

void test_safety_requires_air_and_established_flow_navigation() {
    ofnav::SafetyActionManager m{};
    using ofnav::NavMode;
    using ofnav::SafetyAction;
    // On the ground after arming (range below the envelope): nothing.
    for (uint64_t t = 1000000U; t < 8000000U; t += 100000U) {
        const auto d = m.update(NavMode::DegradedHold, t, false);
        assert(d.action == SafetyAction::None && d.reason == ofnav::SafetyReason::OnGround);
    }
    // Takeoff below the envelope: in air but not engaged yet.
    for (uint64_t t = 8000000U; t < 15000000U; t += 100000U) {
        const auto d = m.update(NavMode::DegradedHold, t, true);
        assert(d.action == SafetyAction::None && d.reason == ofnav::SafetyReason::NotEngaged);
    }
    // 1 s of healthy flow engages; a later loss acts.
    for (uint64_t t = 15000000U; t <= 16000000U; t += 100000U) {
        (void)m.update(NavMode::FlowNav, t, true);
    }
    assert(m.engaged());
    (void)m.update(NavMode::DegradedHold, 16100000U, true);
    assert(m.update(NavMode::DegradedHold, 16500000U, true).action == SafetyAction::AltitudeHold);
    // Landing resets the non-latched hold.
    assert(m.update(NavMode::FlowNav, 20000000U, false).action == SafetyAction::None);
    assert(!m.engaged());
}

void test_safety_failsafe_is_debounced_then_latched_land() {
    ofnav::SafetyActionManager m{unengaged()};
    using ofnav::NavMode;
    using ofnav::SafetyAction;
    assert(m.update(NavMode::FlowNav, 1000000U).action == SafetyAction::None);
    // A single 100 ms FAILSAFE glitch does not land the vehicle.
    assert(m.update(NavMode::FailsafeLand, 1100000U).action == SafetyAction::None);
    assert(m.update(NavMode::FailsafeLand, 1200000U).action == SafetyAction::None);
    assert(m.update(NavMode::FlowNav, 1210000U).action == SafetyAction::None);
    // Persisting failsafe lands; land is latched against later recovery.
    (void)m.update(NavMode::FailsafeLand, 2000000U);
    const auto land = m.update(NavMode::FailsafeLand, 2200000U);
    assert(land.action == SafetyAction::Land && land.changed && land.latched);
    assert(land.reason == ofnav::SafetyReason::FailsafeConfirmed);
    const auto latched = m.update(NavMode::FlowNav, 9000000U);
    assert(latched.action == SafetyAction::Land);
    // The latched action keeps reporting its cause, not NOMINAL.
    assert(latched.reason == ofnav::SafetyReason::FailsafeConfirmed);
    assert(!m.update(NavMode::FlowNav, 9100000U).changed);
    m.reset();
    assert(m.update(NavMode::FlowNav, 9200000U).action == SafetyAction::None);
}

void test_safety_degraded_hold_escalates_and_recovers_with_hysteresis() {
    ofnav::SafetyActionManager m{unengaged()};
    using ofnav::NavMode;
    using ofnav::SafetyAction;
    (void)m.update(NavMode::FlowNav, 1000000U);
    (void)m.update(NavMode::DegradedHold, 1100000U);
    assert(m.update(NavMode::DegradedHold, 1300000U).action == SafetyAction::None);
    const auto hold = m.update(NavMode::DegradedHold, 1400000U);
    assert(hold.action == SafetyAction::AltitudeHold && hold.changed);
    // Short healthy interval keeps the hold (hysteresis).
    (void)m.update(NavMode::FlowNav, 1500000U);
    assert(m.update(NavMode::FlowNav, 2000000U).action == SafetyAction::AltitudeHold);
    const auto rec = m.update(NavMode::FlowNav, 2500000U);
    assert(rec.action == SafetyAction::None && rec.reason == ofnav::SafetyReason::Recovered);
    // Intermittent degradation without 1 s of recovery escalates to land.
    uint64_t t = 3000000U;
    ofnav::SafetyDecision d{};
    for (int i = 0; i < 12; ++i) {
        d = m.update(NavMode::DegradedHold, t);
        t += 400000U;
        d = m.update(NavMode::FlowNav, t);
        t += 200000U;
    }
    assert(d.action == SafetyAction::Land && d.latched);
}

void test_safety_manual_request_and_time_reversal() {
    ofnav::SafetyActionManager m{unengaged()};
    using ofnav::NavMode;
    using ofnav::SafetyAction;
    (void)m.update(NavMode::FlowNav, 1000000U);
    const auto manual = m.update(NavMode::ManualRequired, 1010000U);
    assert(manual.action == SafetyAction::ManualTakeover && manual.latched);
    ofnav::SafetyActionManager clock{unengaged()};
    (void)clock.update(NavMode::FlowNav, 2000000U);
    const auto back = clock.update(NavMode::FlowNav, 1000000U);
    assert(back.action == SafetyAction::Land);
    assert(back.reason == ofnav::SafetyReason::NonMonotonicTime);
}

void test_gyro_integrator_prorates_and_requires_full_coverage() {
    ofnav::GyroIntegrator g{};
    // 250 Hz packets of a constant 0.5 rad/s rate about body x.
    for (uint64_t t = 1004000U; t <= 1200000U; t += 4000U) {
        g.push(t, ofnav::Vector3f{0.5F * 0.004F, 0.0F, -0.1F * 0.004F}, 4000U);
    }
    ofnav::Vector3f a{};
    assert(g.integrate(1050000U, 1150000U, a));
    assert(near(a.x, 0.05F, 1.0e-5F));
    assert(near(a.z, -0.01F, 1.0e-5F));
    // Interval with a partial first and last packet.
    assert(g.integrate(1051000U, 1149000U, a));
    assert(near(a.x, 0.5F * 0.098F, 1.0e-5F));
    // Not yet covered by gyro data -> refused.
    assert(!g.integrate(1150000U, 1210000U, a));
    assert(!g.integrate(900000U, 1010000U, a));
    // A 20 ms hole resets coverage.
    g.push(1220000U, ofnav::Vector3f{0.002F, 0.0F, 0.0F}, 4000U);
    assert(!g.integrate(1100000U, 1150000U, a));
}

} // namespace

int main() {
    test_forward_motion_sign();
    test_right_motion_sign();
    test_lever_arm_correction();
    test_quality_rejection();
    test_runtime_failsafe_without_range();
    test_ekf_innovation_gate_rejects_outlier();
    test_velocity_reset_after_aiding_outage_but_not_during_aiding();
    test_range_rate_is_a_measured_vertical_velocity();
    test_safety_action_config_validation();
    test_body_velocity_3d_from_horizontal_body_and_vertical_ned();
    test_covariance_joseph_update();
    test_gravity_projection_does_not_create_horizontal_velocity();
    test_stale_flow_is_never_fused();
    test_timeout_without_new_flow_samples();
    test_asynchronous_imu_prediction_between_flow_frames();
    test_reused_imu_timestamp_not_predicted_twice();
    test_missing_gyro_compensation_is_rejected();
    test_velocity_rotation_respects_pitch();
    test_range_sample_skew_must_reject_flow_correction();
    test_excessive_tilt_detected_during_flow_dropout();
    test_nonfinite_imu_does_not_enter_flow_mode();
    test_missing_imu_history_requires_explicit_reset();
    test_combined_roll_pitch_exceeds_inclination_limit();
    test_flow_sensor_profile_still_rejects_100m();
    test_camera_profile_accepts_100m_with_height_scaled_sigma();
    test_camera_profile_rejects_coarse_measurement_by_uncertainty();
    test_camera_profile_ekf_uses_estimator_sigma();
    test_safety_requires_air_and_established_flow_navigation();
    test_safety_failsafe_is_debounced_then_latched_land();
    test_safety_degraded_hold_escalates_and_recovers_with_hysteresis();
    test_safety_manual_request_and_time_reversal();
    test_gyro_integrator_prorates_and_requires_full_coverage();
    std::cout << "ofnav RTOS core tests passed\n";
    return 0;
}
