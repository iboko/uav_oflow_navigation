#pragma once

#include <array>
#include <cstdint>
#include <limits>

namespace ofnav {

constexpr float kPi = 3.14159265358979323846F;
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

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
    std::array<float, 9> data{1.0F, 0.0F, 0.0F,
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
    ImuInvalid
};

enum class NavMode : uint8_t {
    Standby = 0,
    FlowNav,
    DegradedHold,
    FailsafeLand,
    ManualRequired
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
    uint64_t max_flow_age_us{200000U};

    float accel_noise_sigma_m_s2{0.08F};
    float flow_vel_sigma_min_m_s{0.035F};
    float flow_vel_sigma_quality_gain_m_s{0.18F};
    float flow_vel_sigma_height_gain_m_s{0.025F};
    float innovation_gate_2d{9.21F};
};

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

    [[nodiscard]] EkfState state() const noexcept { return x_; }
    [[nodiscard]] float lastInnovationD2() const noexcept { return last_innovation_d2_; }

private:
    using Matrix6 = std::array<std::array<float, 6>, 6>;
    using Matrix6x2 = std::array<std::array<float, 2>, 6>;

    Config cfg_{};
    EkfState x_{};
    Matrix6 p_{};
    uint64_t last_predict_us_{0U};
    bool initialized_{false};
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
                                         bool innovation_ok) noexcept;

private:
    Config cfg_{};
};

class OfNavRuntime final {
public:
    explicit OfNavRuntime(Config cfg) noexcept;

    void reset(uint64_t time_us) noexcept;

    [[nodiscard]] RuntimeOutput step(const ImuSample& imu,
                                     const RangeSample& range,
                                     const OpticalFlowRadSample& flow,
                                     const AttitudeSample& attitude) noexcept;

private:
    Config cfg_{};
    FlowVelocityEstimator flow_estimator_;
    HorizontalEkf ekf_;
    SafetyMonitor safety_;
};

} // namespace ofnav
