#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

// Research-only 15-state error-state inertial filter and bounded fixed-lag
// replay. No PX4 uORB, EKF2, controller or actuator APIs in this header.
// SI units, FRD body, NED navigation, quaternion wxyz maps FRD -> NED.
// Right orientation error: Rtrue = Rnom Exp([dtheta]x).
namespace ofnav15 {

using Vec3 = std::array<double, 3>;
using Quat = std::array<double, 4>;
using Matrix15 = std::array<double, 225>;

struct Imu {
    std::uint64_t time_us{0};
    Vec3 gyro_rad_s{};
    Vec3 specific_force_m_s2{};
};

struct Visual {
    std::uint64_t time_us{0};
    std::array<double, 2> velocity_ne_m_s{};
    std::array<double, 4> covariance_ne_m2_s2{0.04, 0., 0., 0.04};
};

struct Parameters {
    std::array<double, 15> initial_sigma{
        10., 10., 10., 1., 1., 1., .1, .1, .1,
        .02, .02, .02, .2, .2, .2
    };
    double gyro_noise{.003};
    double accel_noise{.03};
    double gyro_bias_random_walk{.00005};
    double accel_bias_random_walk{.0003};
    double max_dt_s{.05};
    double nis_gate{9.210340371976184};
    double max_gyro_bias{.3};
    double max_accel_bias{3.};
};

enum class Status {
    WaitingForImu,
    InertialOnly,
    VisualCorrected,
    VisualOutlier,
    ImuGap,
    NumericalFailure,
    InvalidVisual,
    RepeatedVisual,
    UnalignedVisual,
    BiasLimit,
    PosteriorInvalid,
    NoHistory,
    DelayExceeded,
    FutureVisual,
    DuplicateVisual,
    BufferLimit,
    ReplayFailed
};

const char *statusName(Status value) noexcept;

class Filter {
public:
    explicit Filter(const Parameters &config = Parameters{}) noexcept;

    [[nodiscard]] Status predict(const Imu &sample) noexcept;
    [[nodiscard]] Status updateVelocity(const Visual &sample) noexcept;
    [[nodiscard]] bool healthy() const noexcept;
    [[nodiscard]] Status status() const noexcept { return status_; }
    [[nodiscard]] std::uint64_t timestamp() const noexcept { return last_.time_us; }
    [[nodiscard]] const Vec3 &position() const noexcept { return p_; }
    [[nodiscard]] const Vec3 &velocity() const noexcept { return v_; }
    [[nodiscard]] const Quat &quaternion() const noexcept { return q_; }
    [[nodiscard]] const Vec3 &gyroBias() const noexcept { return bg_; }
    [[nodiscard]] const Vec3 &accelBias() const noexcept { return ba_; }
    [[nodiscard]] const Matrix15 &covariance() const noexcept { return p_cov_; }
    [[nodiscard]] double lastNis() const noexcept { return nis_; }
    [[nodiscard]] unsigned accepted() const noexcept { return accepted_; }
    [[nodiscard]] unsigned rejected() const noexcept { return rejected_; }
    [[nodiscard]] bool hasImu() const noexcept { return has_imu_; }

private:
    Parameters config_{};
    Vec3 p_{}, v_{}, bg_{}, ba_{};
    Quat q_{1., 0., 0., 0.};
    Matrix15 p_cov_{};
    Imu last_{};
    std::uint64_t last_visual_us_{0};
    double nis_{std::numeric_limits<double>::quiet_NaN()};
    unsigned accepted_{0}, rejected_{0};
    bool has_imu_{false};
    bool valid_config_{false};
    Status status_{Status::WaitingForImu};
};

// This class stores only a small fixed number of IMU/visual events and a
// checkpoint. No heap, no unbounded growth. Replay is transactional:
// rejected delayed updates do NOT alter the latest valid filter state.
// It is diagnostic; operation count and WCET on target hardware untested.
class FixedLag {
public:
    static constexpr std::size_t kImuCapacity = 64;
    static constexpr std::size_t kVisualCapacity = 12;

    explicit FixedLag(const Parameters &parameters = Parameters{},
                      std::uint64_t max_lag_us = 300000U) noexcept;
    [[nodiscard]] Status pushImu(const Imu &sample) noexcept;
    [[nodiscard]] Status pushDelayedVelocity(const Visual &sample) noexcept;

    [[nodiscard]] const Filter &current() const noexcept { return current_; }
    [[nodiscard]] std::size_t storedImu() const noexcept { return count_; }
    [[nodiscard]] std::size_t storedVisual() const noexcept { return visual_count_; }
    [[nodiscard]] std::uint64_t latestImuTime() const noexcept { return latest_; }
    [[nodiscard]] bool failed() const noexcept { return failed_; }

private:
    [[nodiscard]] bool replayPrefix(std::size_t count, const Visual *observations,
                                    std::size_t observations_count,
                                    Filter &result) const noexcept;
    [[nodiscard]] bool replay(const Visual *observations,
                              std::size_t observations_count,
                              Filter &result) const noexcept;
    void trim() noexcept;

    Parameters parameters_{};
    std::uint64_t max_lag_us_{300000U};
    std::uint64_t latest_{0};
    Filter checkpoint_;
    Filter current_;
    Imu checkpoint_imu_{};
    bool initialized_{false}, failed_{false};
    std::array<Imu, kImuCapacity> imu_{};
    std::size_t count_{0};
    std::array<Visual, kVisualCapacity> visual_{};
    std::size_t visual_count_{0};
};

} // namespace ofnav15
