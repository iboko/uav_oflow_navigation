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

} // namespace

int main() {
    test_forward_motion_sign();
    test_right_motion_sign();
    test_lever_arm_correction();
    test_quality_rejection();
    test_runtime_failsafe_without_range();
    test_ekf_innovation_gate_rejects_outlier();
    std::cout << "ofnav RTOS core tests passed\n";
    return 0;
}
