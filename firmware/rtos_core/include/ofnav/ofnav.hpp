#pragma once

#include "ofnav/compat.hpp"
#include <cstdint>

namespace ofnav {

constexpr float kPi = 3.14159265358979323846F;
constexpr float kNaN = kQuietNaNf;

struct Vector2f {
    float x{0.0F};
    float y{0.0F};
};

struct Vector3f {
    float x{0.0F};
    float y{0.0F};
    float z{0.0F};
};

struct Matrix3f {
    Array<float, 9> data{1.0F, 0.0F, 0.0F,
                              0.0F, 1.0F, 0.0F,
                              0.0F, 0.0F, 1.0F};

    [[nodiscard]] float operator()(const uint8_t r, const uint8_t c) const noexcept {
        return data[static_cast<unsigned>(r) * 3U + static_cast<unsigned>(c)];
    }
};

struct AttitudeSample {
    uint64_t time_us{0U};
    float roll_rad{0.0F};
    float pitch_rad{0.0F};
    float yaw_rad{0.0F};
};

struct ImuSample {
    uint64_t time_us{0U};
    Vector3f gyro_rad_s{};
    Vector3f accel_m_s2{};
    bool valid{false};
};

struct RangeSample {
    uint64_t time_us{0U};
    float distance_m{kNaN};
    bool valid{false};
};

struct OpticalFlowRadSample {
    uint64_t time_us{0U};
    float integration_time_s{0.0F};
    float integrated_x_rad{0.0F};
    float integrated_y_rad{0.0F};
    float integrated_xgyro_rad{0.0F};
    float integrated_ygyro_rad{0.0F};
    float integrated_zgyro_rad{0.0F};
    uint8_t quality{0U};
    bool valid{false};
};

enum class RejectReason : uint8_t {
    Ok = 0,
    NonFiniteInput,
    BadDeltaTime,
    LowQuality,
    BadHeight,
    HeightTooLow,
    HeightTooHigh,
    ExcessiveTilt,
    FlowRateLimit,
    VelocityLimit,
    InnovationGate,
    StaleRange,
    StaleFlow,
    ImuInvalid,
    // Physics-based velocity uncertainty h * sigma_flow exceeds the
    // configured limit: the measurement is too coarse at this height.
    VelocityUncertainty
};

// Explicit measurement-source envelope. The legacy optical-flow sensor
// envelope (0.2-4 m) stays the default; a downward camera VO source has its
// own height range and an uncertainty model scaled by height.
enum class SensorProfile : uint8_t {
    OpticalFlowSensor = 0,
    DownwardCameraVo = 1
};

enum class NavMode : uint8_t {
    Standby = 0,
    FlowNav,
    DegradedHold,
    FailsafeLand,
    ManualRequired
};

// Reaction requested from the autopilot for the current navigation health.
// The autopilot integration decides whether it is executed (opt-in).
enum class SafetyAction : uint8_t {
    None = 0,          // no request; never commands a return to position mode
    AltitudeHold = 1,  // stop relying on horizontal navigation: altitude mode
    Land = 2,          // latched: land in place
    ManualTakeover = 3 // latched: hand over to the pilot (stabilized mode)
};

struct SafetyActionConfig {
    uint64_t failsafe_confirm_us{200000U};   // FAILSAFE_LAND must persist
    uint64_t degraded_confirm_us{300000U};   // DEGRADED_HOLD must persist
    uint64_t degraded_to_land_us{5000000U};  // unrecovered degradation -> land
    uint64_t recovery_us{1000000U};          // healthy FLOW_NAV clears hold
    // Act only after flow navigation was established in this flight
    // (FLOW_NAV for recovery_us while in air): before that there is no
    // navigation to lose (ground, takeoff below the sensor envelope).
    bool require_engagement{true};
};

enum class SafetyReason : uint8_t {
    Nominal = 0,
    DegradedConfirmed,
    DegradedTooLong,
    FailsafeConfirmed,
    ManualRequested,
    Recovered,
    NonMonotonicTime,
    OnGround,
    NotEngaged
};

struct SafetyDecision {
    SafetyAction action{SafetyAction::None};
    SafetyReason reason{SafetyReason::Nominal};
    bool changed{false};
    bool latched{false};
    uint64_t action_since_us{0U};
};

struct HealthFlags {
    bool imu_valid{false};
    bool range_valid{false};
    bool flow_valid{false};
    bool height_valid{false};
    bool quality_valid{false};
    bool innovation_valid{true};
    bool stale_range{true};
    bool stale_flow{true};
    bool excessive_tilt{false};
};

struct FlowVelocityEstimate {
    bool accepted{false};
    RejectReason reason{RejectReason::Ok};
    float height_m{kNaN};
    Vector2f velocity_body_m_s{};
    Vector2f velocity_nav_m_s{};
    float flow_x_rate_rad_s{0.0F};
    float flow_y_rate_rad_s{0.0F};
    // One-sigma horizontal velocity uncertainty used for the EKF update.
    float velocity_sigma_m_s{kNaN};
};

struct EkfState {
    float n_m{0.0F};
    float e_m{0.0F};
    float vn_m_s{0.0F};
    float ve_m_s{0.0F};
    float accel_bias_x_m_s2{0.0F};
    float accel_bias_y_m_s2{0.0F};
};

struct RuntimeOutput {
    NavMode mode{NavMode::Standby};
    HealthFlags health{};
    FlowVelocityEstimate flow{};
    EkfState state{};
};

struct Config {
    float min_height_m{0.20F};
    float max_height_m{4.00F};
    float max_tilt_rad{0.45F};
    float max_abs_flow_rate_rad_s{7.4F};
    float max_horizontal_speed_m_s{1.5F};
    uint8_t min_quality{100U};

    float flow_scale_x{1.0F};
    float flow_scale_y{1.0F};

    Vector3f sensor_offset_body_m{};
    Matrix3f body_from_sensor{};

    uint64_t max_range_age_us{200000U};
    uint64_t max_flow_age_us{120000U};
    uint64_t max_imu_age_us{120000U};
    uint64_t max_attitude_age_us{120000U};
    uint64_t max_measurement_skew_us{100000U};

    float accel_noise_sigma_m_s2{0.08F};
    float flow_vel_sigma_min_m_s{0.035F};
    float flow_vel_sigma_quality_gain_m_s{0.18F};
    float flow_vel_sigma_height_gain_m_s{0.025F};
    float innovation_gate_2d{9.21F};
    // Inertial-only time after which a gate-rejected but otherwise valid
    // flow velocity re-initialises the EKF velocity (EKF2-style aiding
    // reset): unaided accelerometer integration drifts by metres per
    // second while its covariance stays optimistic, which would otherwise
    // lock the innovation gate shut for good. 0 disables the reset.
    uint64_t velocity_reset_timeout_us{1000000U};

    float max_integration_time_s{0.20F};
    // If > 0: sigma_v^2 = sigma_min^2 + (h*sigma_flow)^2 + (|v|*sigma_h_rel)^2,
    // inflated by low quality, and rejected above max_velocity_sigma_m_s.
    // If == 0: legacy empirical optical-flow sensor model (clamped 0.35 m/s).
    float flow_rate_sigma_rad_s{0.0F};
    float height_rel_sigma{0.0F};
    float max_velocity_sigma_m_s{0.35F};
};

// Validated configuration for a measurement-source envelope.
[[nodiscard]] Config makeProfileConfig(SensorProfile profile, uint8_t min_quality = 120U) noexcept;
[[nodiscard]] const char* toString(SensorProfile profile) noexcept;
// One-sigma velocity uncertainty for given height, speed and quality.
[[nodiscard]] float flowVelocitySigma(const Config& cfg, float height_m,
                                      float speed_m_s, uint8_t quality) noexcept;

[[nodiscard]] bool isFinite(float v) noexcept;
[[nodiscard]] Vector3f cross(Vector3f a, Vector3f b) noexcept;
[[nodiscard]] Vector3f matVec(Matrix3f m, Vector3f v) noexcept;
[[nodiscard]] Vector2f rotateBodyToNav(float vx_body, float vy_body, float yaw_rad) noexcept;
[[nodiscard]] Vector2f rotateNavToBody(float vn, float ve, float yaw_rad) noexcept;
[[nodiscard]] float heightFromRange(float range_m, float roll_rad, float pitch_rad) noexcept;
[[nodiscard]] const char* toString(RejectReason reason) noexcept;
[[nodiscard]] const char* toString(NavMode mode) noexcept;

class FlowVelocityEstimator final {
public:
    explicit FlowVelocityEstimator(Config cfg) noexcept : cfg_(cfg) {}

    [[nodiscard]] FlowVelocityEstimate update(const OpticalFlowRadSample& flow,
                                              const RangeSample& range,
                                              const ImuSample& imu,
                                              const AttitudeSample& attitude) const noexcept;

private:
    Config cfg_{};
};

class HorizontalEkf final {
public:
    explicit HorizontalEkf(Config cfg) noexcept;

    void reset(uint64_t time_us) noexcept;
    void predict(const ImuSample& imu, const AttitudeSample& attitude) noexcept;
    [[nodiscard]] bool updateFlowVelocity(const FlowVelocityEstimate& flow, uint8_t quality) noexcept;
    // Replaces the velocity by a measurement; velocity cross-covariances
    // are cleared, position and accelerometer bias are kept.
    bool resetVelocity(const FlowVelocityEstimate& flow) noexcept;

    [[nodiscard]] EkfState state() const noexcept { return x_; }
    [[nodiscard]] bool healthy() const noexcept { return prediction_healthy_; }
    [[nodiscard]] float lastInnovationD2() const noexcept { return last_innovation_d2_; }
    [[nodiscard]] float covariance(uint8_t row, uint8_t col) const noexcept {
        return (row < 6U && col < 6U) ? p_[row][col] : kNaN;
    }

private:
    using Matrix6 = Array<Array<float, 6>, 6>;
    using Matrix6x2 = Array<Array<float, 2>, 6>;

    Config cfg_{};
    EkfState x_{};
    Matrix6 p_{};
    uint64_t last_predict_us_{0U};
    bool initialized_{false};
    bool prediction_healthy_{true};
    float last_innovation_d2_{0.0F};

    static void setIdentity(Matrix6& m, float diag) noexcept;
    static void symmetrize(Matrix6& m) noexcept;
};

class SafetyMonitor final {
public:
    explicit SafetyMonitor(Config cfg) noexcept : cfg_(cfg) {}

    [[nodiscard]] RuntimeOutput evaluate(uint64_t now_us,
                                         const ImuSample& imu,
                                         const RangeSample& range,
                                         const OpticalFlowRadSample& flow,
                                         const FlowVelocityEstimate& flow_estimate,
                                         const EkfState& ekf_state,
                                         bool innovation_ok,
                                         const AttitudeSample& attitude) noexcept;
private:
    Config cfg_{};
};

// Integrates vehicle gyro delta angles over an optical-flow integration
// interval for sensors without their own gyro (PAW3902-class, Gazebo).
// Only an interval FULLY covered by stored IMU packets is integrated;
// partial packets are prorated linearly. No interval -> no compensation
// (the flow sample is then rejected, never compensated with zeros).
class GyroIntegrator final {
public:
    static constexpr uint32_t kCapacity = 96U;  // > 0.35 s at 250 Hz

    void reset() noexcept { count_ = 0U; head_ = 0U; }
    // Packet covering (end_us - dt_us, end_us], delta angle in body FRD.
    void push(uint64_t end_us, const Vector3f& delta_angle_rad, uint32_t dt_us) noexcept;
    [[nodiscard]] bool integrate(uint64_t start_us, uint64_t end_us,
                                 Vector3f& delta_angle_rad) const noexcept;

private:
    struct Packet {
        uint64_t end_us{0U};
        uint32_t dt_us{0U};
        Vector3f angle{};
    };
    Packet packets_[kCapacity]{};
    uint32_t count_{0U};
    uint32_t head_{0U};  // next write position
};

// Debounced, escalating and latching mapping NavMode -> SafetyAction.
// Land and ManualTakeover are latched until reset() (pilot / ground crew).
// AltitudeHold clears to None only after recovery_us of healthy FLOW_NAV;
// the manager never requests a return to a horizontal-position mode.
class SafetyActionManager final {
public:
    explicit SafetyActionManager(SafetyActionConfig cfg = SafetyActionConfig{}) noexcept
        : cfg_(cfg) {}

    // in_air: vehicle not landed (PX4 vehicle_land_detected). On the ground
    // the manager is reset and requests nothing.
    [[nodiscard]] SafetyDecision update(NavMode mode, uint64_t now_us,
                                        bool in_air = true) noexcept;
    void reset() noexcept;
    [[nodiscard]] bool engaged() const noexcept { return engaged_; }
    [[nodiscard]] SafetyAction action() const noexcept { return action_; }

private:
    SafetyDecision decide(SafetyAction action, SafetyReason reason, uint64_t now_us) noexcept;

    SafetyActionConfig cfg_{};
    SafetyAction action_{SafetyAction::None};
    NavMode last_mode_{NavMode::Standby};
    uint64_t mode_since_us_{0U};
    uint64_t degraded_since_us_{0U};
    uint64_t action_since_us_{0U};
    uint64_t last_us_{0U};
    bool started_{false};
    bool degraded_episode_{false};
    bool engaged_{false};
};

[[nodiscard]] const char* toString(SafetyAction action) noexcept;
[[nodiscard]] const char* toString(SafetyReason reason) noexcept;

class OfNavRuntime final {
public:
    explicit OfNavRuntime(Config cfg) noexcept;

    void reset(uint64_t time_us) noexcept;

    // Feed each IMU sample separately from the slower optical-flow rate.
    void predictImu(const ImuSample& imu, const AttitudeSample& attitude) noexcept;

    [[nodiscard]] RuntimeOutput step(const ImuSample& imu,
                                     const RangeSample& range,
                                     const OpticalFlowRadSample& flow,
                                     const AttitudeSample& attitude,
                                     uint64_t now_us = 0U) noexcept;

    [[nodiscard]] RuntimeOutput monitor(uint64_t now_us,
                                        const ImuSample& imu,
                                        const RangeSample& range,
                                        const OpticalFlowRadSample& flow,
                                        const AttitudeSample& attitude) noexcept;
    [[nodiscard]] uint32_t velocityResets() const noexcept { return velocity_resets_; }
private:
    FlowVelocityEstimate last_flow_estimate_{};
    bool last_innovation_ok_{false};
    // Last accepted flow fusion, or the start of aiding (reset / first
    // measurement): the unaided interval for the velocity reset.
    uint64_t last_fusion_us_{0U};
    uint32_t velocity_resets_{0U};
    Config cfg_{};
    FlowVelocityEstimator flow_estimator_;
    HorizontalEkf ekf_;
    SafetyMonitor safety_;
};

} // namespace ofnav
