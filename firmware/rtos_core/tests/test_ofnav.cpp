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
        {-g * std::sin(pitch), 0.0F, -g * std::cos(pitch)}, true};
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

} // namespace

int main() {
    test_forward_motion_sign();
    test_right_motion_sign();
    test_lever_arm_correction();
    test_quality_rejection();
    test_runtime_failsafe_without_range();
    test_ekf_innovation_gate_rejects_outlier();
    test_covariance_joseph_update();
    test_gravity_projection_does_not_create_horizontal_velocity();
    test_stale_flow_is_never_fused();
    test_timeout_without_new_flow_samples();
    std::cout << "ofnav RTOS core tests passed\n";
    return 0;
}
