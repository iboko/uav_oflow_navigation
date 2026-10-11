#pragma once

#include <cstddef>
#include <cstdint>

#include "ofnav/eskf15.hpp"

// Platform-independent runner of the 15-state fixed-lag ESKF for an
// autopilot task: integrates delta-angle/delta-velocity IMU packets to a
// fixed rate, initializes by static alignment (fallback: external attitude),
// accepts delayed body-frame velocity from the optical-flow/VO estimator
// and restarts after a latched filter failure.
//
// DIAGNOSTIC ONLY: the output is never an EKF2 input or a control command.
// No heap; the object (~56 KiB) must be statically allocated once.
namespace ofnav {

enum class EskfRunnerMode : uint8_t {
    Disabled = 0,
    Aligning = 1,
    Running = 2
};

enum class EskfInitSource : uint8_t {
    None = 0,
    StaticAlignment = 1,
    ExternalAttitude = 2
};

struct EskfRunnerConfig {
    uint64_t imu_period_us{10000U};        // 100 Hz filter rate
    uint64_t max_lag_us{300000U};          // must fit 64 IMU samples
    std::size_t alignment_samples{100U};   // 1 s at 100 Hz
    double external_yaw_sigma_rad{0.1};    // heading from the autopilot
    double external_tilt_sigma_rad{0.05};  // fallback initialization
    ofnav15::AlignmentLimits alignment{};
    ofnav15::Parameters filter{};
};

struct EskfRunnerStats {
    uint32_t imu_samples{0U};
    uint32_t velocity_events{0U};
    uint32_t velocity_processed{0U};
    uint32_t velocity_deferred{0U};
    uint32_t velocity_dropped{0U};
    uint32_t filter_restarts{0U};
    uint32_t clipping_resets{0U};
    uint64_t imu_worst_us{0U};
    uint64_t imu_total_us{0U};
    uint64_t frame_worst_us{0U};
    uint64_t frame_total_us{0U};
    std::size_t max_replay_length{0U};
    ofnav15::Status last_status{ofnav15::Status::WaitingForImu};
    ofnav15::AlignmentStatus last_alignment{ofnav15::AlignmentStatus::TooFewSamples};
};

class EskfRunner final {
public:
    static constexpr std::size_t kMaxAlignmentSamples = 200U;
    // Plain function pointer (no noexcept in the type: C++14 ABI stable).
    using Clock = uint64_t (*)();

    explicit EskfRunner(const EskfRunnerConfig &config = EskfRunnerConfig{},
                        Clock clock = nullptr) noexcept;

    void configure(const EskfRunnerConfig &config, Clock clock) noexcept;
    void restart() noexcept;
    void disable() noexcept;

    // One vehicle_imu-like packet. external_q (FRD->NED, wxyz) is used only
    // for the yaw of the static alignment or as fallback initialization.
    void addImuDelta(uint64_t time_us, const ofnav15::Vec3 &delta_angle_rad,
                     uint32_t delta_angle_dt_us, const ofnav15::Vec3 &delta_velocity_m_s,
                     uint32_t delta_velocity_dt_us, bool clipping,
                     const ofnav15::Quat &external_q, bool external_q_valid) noexcept;

    // Body-frame (FRD) x/y velocity of the centre of mass at time_us.
    void addBodyVelocity(uint64_t time_us, double vx_m_s, double vy_m_s,
                         double sigma_m_s) noexcept;

    [[nodiscard]] EskfRunnerMode mode() const noexcept { return mode_; }
    [[nodiscard]] EskfInitSource initSource() const noexcept { return init_source_; }
    [[nodiscard]] const ofnav15::Filter &filter() const noexcept { return lag_.current(); }
    [[nodiscard]] const EskfRunnerStats &stats() const noexcept { return stats_; }

private:
    void emitImu(const ofnav15::Imu &sample, const ofnav15::Quat &external_q,
                 bool external_q_valid) noexcept;
    void startFilter(const ofnav15::Parameters &parameters, EskfInitSource source) noexcept;
    void pushVelocity(const ofnav15::Measurement &m) noexcept;
    [[nodiscard]] uint64_t now() const noexcept { return clock_ ? clock_() : 0U; }

    EskfRunnerConfig config_{};
    Clock clock_{nullptr};
    EskfRunnerMode mode_{EskfRunnerMode::Aligning};
    EskfInitSource init_source_{EskfInitSource::None};
    EskfRunnerStats stats_{};
    ofnav15::FixedLag lag_{};
    Array<ofnav15::Imu, kMaxAlignmentSamples> alignment_{};
    std::size_t alignment_count_{0U};
    ofnav15::Vec3 sum_angle_{}, sum_velocity_{};
    uint64_t sum_dt_us_{0U};
    uint64_t sum_velocity_dt_us_{0U};
    uint64_t last_packet_us_{0U};
    bool has_pending_{false};
    ofnav15::Measurement pending_{};
};

} // namespace ofnav
