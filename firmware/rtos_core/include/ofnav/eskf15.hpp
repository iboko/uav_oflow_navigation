#pragma once

#include "ofnav/compat.hpp"
#include <cstddef>
#include <cstdint>

// Research-only 15-state error-state inertial filter and bounded fixed-lag
// replay. No PX4 uORB, EKF2, controller or actuator APIs in this header.
// SI units, FRD body, NED navigation, quaternion wxyz maps FRD -> NED.
// Right orientation error: Rtrue = Rnom Exp([dtheta]x).
namespace ofnav15 {

using ofnav::Array;
using ofnav::kQuietNaN;

using Vec3 = Array<double, 3>;
using Quat = Array<double, 4>;
using Matrix15 = Array<double, 225>;

struct Imu {
    std::uint64_t time_us{0};
    Vec3 gyro_rad_s{};
    Vec3 specific_force_m_s2{};
};

struct Visual {
    std::uint64_t time_us{0};
    Array<double, 2> velocity_ne_m_s{};
    Array<double, 4> covariance_ne_m2_s2{0.04, 0., 0., 0.04};
};

// Aiding measurement kinds. The numeric order is the deterministic
// processing order of events sharing one timestamp.
enum class MeasurementKind : std::uint8_t {
    VelocityNe = 0,      // VN, VE of the centre of mass, m/s
    BodyVelocityXy = 1,  // FRD x, y velocity of the centre of mass, m/s
    Yaw = 2,             // heading of FRD x axis from north, rad
    PositionNe = 3,      // local N, E position (orthophoto fix), m
    // N, E of the ground point hit by the camera optical axis (orthophoto
    // match of the principal point). Attitude-free measurement; model
    // z = p_NE + h (R d)_NE / (R d)_D couples it to the tilt states.
    MapGroundPointNe = 4
};
constexpr std::size_t kMeasurementKinds = 5;

struct Measurement {
    std::uint64_t time_us{0};
    MeasurementKind kind{MeasurementKind::VelocityNe};
    Array<double, 2> value{};
    // 2x2 row-major covariance; a scalar yaw uses only element [0].
    Array<double, 4> covariance{0.04, 0., 0., 0.04};
    // Covariance-intersection weight for an estimate whose error
    // correlation with the filter is UNKNOWN (orthophoto fix computed
    // from the same frames as VO). 1: independent, standard update.
    // (0,1): full-state CI update with P/w and R/(1-w).
    // <= 0: choose w on a fixed grid minimising det of the FULL posterior.
    double ci_weight{1.};
    // MapGroundPointNe only: [height above ground plane along NED down, m;
    // optical axis unit vector in FRD body axes x, y, z].
    Array<double, 4> aux{0., 0., 0., 1.};
};

[[nodiscard]] Measurement fromVisual(const Visual &visual) noexcept;

struct Parameters {
    Quat initial_quaternion_body_to_ned{1., 0., 0., 0.};
    Vec3 initial_gyro_bias_rad_s{};
    Vec3 initial_accel_bias_m_s2{};
    Array<double, 15> initial_sigma{
        10., 10., 10., 1., 1., 1., .1, .1, .1,
        .02, .02, .02, .2, .2, .2
    };
    double gyro_noise{.003};
    double accel_noise{.03};
    double gyro_bias_random_walk{.00005};
    double accel_bias_random_walk{.0003};
    double max_dt_s{.05};
    double nis_gate{9.210340371976184};
    double nis_gate_1d{6.6348966010212145};
    double max_gyro_bias{.3};
    double max_accel_bias{3.};
};

// Coarse static alignment from a stationary IMU window: roll/pitch from
// mean specific force, gyro bias from mean angular rate. Yaw is NOT
// observable from a static IMU and must be supplied (map, compass) with
// its sigma. Accelerometer bias is inseparable from tilt here and stays
// at its prior; its sigma is folded into the roll/pitch sigma.
struct AlignmentLimits {
    std::size_t min_samples{50};
    double max_gyro_std_rad_s{.01};
    double max_accel_std_m_s2{.15};
    double max_gravity_error_m_s2{.5};
};

enum class AlignmentStatus {
    Aligned,
    TooFewSamples,
    NotStationary,
    GravityMismatch,
    InvalidInput
};

const char *alignmentStatusName(AlignmentStatus value) noexcept;

[[nodiscard]] AlignmentStatus staticAlignment(const Imu *samples, std::size_t count,
                                              double yaw_rad, double yaw_sigma_rad,
                                              const AlignmentLimits &limits,
                                              Parameters &parameters) noexcept;

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
    ReplayFailed,
    YawCorrected,
    YawOutlier,
    PositionCorrected,
    PositionOutlier
};

const char *statusName(Status value) noexcept;
// Corrected or a statistically rejected outlier: both are valid outcomes
// of an accepted, well-formed measurement event.
[[nodiscard]] bool measurementProcessed(Status value) noexcept;

// Scratch matrices for one predict/update. Kept OUTSIDE the stack so the
// filter runs inside a small RTOS task stack (see docs/ESKF15_PX4_RUNTIME_RU.md).
// Not shared between threads; one workspace per executing thread.
struct Workspace {
    Array<Matrix15, 8> m{};
    Array<double, 225> cholesky{};
    Array<double, 180> noise_map{};
    // Measurement update scratch (n <= 2 rows).
    double h[2][15]{};
    double pht[15][2]{};
    double gain[15][2]{};
    double dx[15]{};
};

class Filter {
public:
    explicit Filter(const Parameters &config = Parameters{}) noexcept;
    // Re-initialize in place (no temporary Filter on the stack).
    void reset(const Parameters &config) noexcept;

#if !defined(__PX4_NUTTX)
    // Host convenience overloads: they place a Workspace (~18 KiB) on the
    // stack, so they are compiled out of NuttX flight builds (PX4 enforces
    // -Wframe-larger-than=4096). RTOS code uses the explicit-workspace API.
    [[nodiscard]] Status predict(const Imu &sample) noexcept;
    [[nodiscard]] Status updateVelocity(const Visual &sample) noexcept;
    [[nodiscard]] Status update(const Measurement &sample) noexcept;
#endif

    [[nodiscard]] Status predict(const Imu &sample, Workspace &ws) noexcept;
    [[nodiscard]] Status update(const Measurement &sample, Workspace &ws) noexcept;

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
    [[nodiscard]] double lastCiWeight() const noexcept { return ci_weight_; }
    [[nodiscard]] unsigned accepted() const noexcept { return accepted_; }
    [[nodiscard]] unsigned rejected() const noexcept { return rejected_; }
    [[nodiscard]] bool hasImu() const noexcept { return has_imu_; }

private:
    Parameters config_{};
    Vec3 p_{}, v_{}, bg_{}, ba_{};
    Quat q_{1., 0., 0., 0.};
    Matrix15 p_cov_{};
    Imu last_{};
    Array<std::uint64_t, kMeasurementKinds> last_measurement_us_{};
    double nis_{kQuietNaN};
    double ci_weight_{1.};
    unsigned accepted_{0}, rejected_{0};
    bool has_imu_{false};
    bool valid_config_{false};
    Status status_{Status::WaitingForImu};
};

// This class stores only a small fixed number of IMU/aiding events and a
// checkpoint. No heap, no unbounded growth. Replay is transactional:
// rejected delayed updates do NOT alter the latest valid filter state.
// All large temporaries live in members: the call stack stays small and
// is measured by tests/eskf15_benchmark.cpp.
//
// Cost control: (1) all measurements of one camera frame are inserted by a
// SINGLE replay (pushDelayedFrame); (2) up to kSnapshots intermediate
// states, taken every kSnapshotStride IMU samples, let a replay start at
// the newest state older than the measurement instead of the checkpoint.
// Snapshots touched by a failed replay are invalidated, never trusted.
class FixedLag {
public:
    static constexpr std::size_t kImuCapacity = 64;
    static constexpr std::size_t kVisualCapacity = 32;
    static constexpr std::size_t kSnapshots = 10;
    static constexpr std::size_t kSnapshotStride = 4;
    static constexpr std::size_t kMaxFrameEvents = kMeasurementKinds;

    explicit FixedLag(const Parameters &parameters = Parameters{},
                      std::uint64_t max_lag_us = 300000U) noexcept;
    // Re-initialize in place: discards all history (no large temporaries).
    void reset(const Parameters &parameters, std::uint64_t max_lag_us) noexcept;
    [[nodiscard]] Status pushImu(const Imu &sample) noexcept;
    [[nodiscard]] Status pushDelayedVelocity(const Visual &sample) noexcept;
    [[nodiscard]] Status pushDelayed(const Measurement &sample) noexcept;
    // All events must share one exposure time and have distinct kinds.
    // statuses[i] receives the outcome of events[i]. Returns ReplayFailed
    // if the joint replay failed (nothing committed), otherwise the status
    // of the first event.
    [[nodiscard]] Status pushDelayedFrame(const Measurement *events, std::size_t count,
                                          Status *statuses) noexcept;

    [[nodiscard]] const Filter &current() const noexcept { return current_; }
    [[nodiscard]] std::size_t storedImu() const noexcept { return count_; }
    [[nodiscard]] std::size_t storedVisual() const noexcept { return visual_count_; }
    [[nodiscard]] std::uint64_t latestImuTime() const noexcept { return latest_; }
    [[nodiscard]] bool failed() const noexcept { return failed_; }
    // IMU predictions executed by the most recent delayed-frame replay.
    [[nodiscard]] std::size_t lastReplayLength() const noexcept { return last_replay_length_; }

private:
    struct Snapshot {
        bool valid{false};
        std::uint64_t time_us{0};
        Filter state{};
    };
    [[nodiscard]] bool replayRange(const Filter &start, const Imu &start_imu,
                                   std::size_t first, std::size_t last,
                                   const Measurement *observations,
                                   std::size_t observations_count, Filter &result,
                                   std::uint64_t watch_time, Status *watch_status,
                                   bool refresh_snapshots) noexcept;
    void trim() noexcept;
    void dropBefore(std::size_t samples) noexcept;
    void invalidateSnapshotsFrom(std::uint64_t time_us) noexcept;

    Parameters parameters_{};
    std::uint64_t max_lag_us_{300000U};
    std::uint64_t latest_{0};
    Filter checkpoint_;
    Filter current_;
    Filter scratch_;
    Filter replay_;
    Imu checkpoint_imu_{};
    bool initialized_{false}, failed_{false};
    Array<Imu, kImuCapacity> imu_{};
    std::size_t count_{0};
    Array<Measurement, kVisualCapacity> visual_{};
    Array<Measurement, kVisualCapacity> candidate_{};
    Array<Measurement, kMaxFrameEvents> batch_{};
    std::size_t visual_count_{0};
    Array<Snapshot, kSnapshots> snapshots_{};
    std::size_t since_snapshot_{0};
    std::size_t last_replay_length_{0};
    Workspace ws_{};
};

} // namespace ofnav15
