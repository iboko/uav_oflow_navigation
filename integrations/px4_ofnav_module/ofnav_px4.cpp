#include <px4_platform_common/module.h>
#include <px4_platform_common/log.h>
#include <px4_platform_common/px4_config.h>
#include <px4_platform_common/posix.h>
#include <px4_platform_common/tasks.h>
#include <px4_platform_common/getopt.h>
#include <drivers/drv_hrt.h>
#include <drivers/drv_sensor.h>

#include <uORB/uORB.h>
#include <uORB/Publication.hpp>
#include <uORB/topics/sensor_optical_flow.h>
#include <uORB/topics/distance_sensor.h>
#include <uORB/topics/vehicle_imu.h>
#include <uORB/topics/vehicle_attitude.h>
#include <uORB/topics/debug_vect.h>
#include <uORB/topics/vehicle_command.h>
#include <uORB/topics/vehicle_status.h>
#include <uORB/topics/vehicle_land_detected.h>
#include <commander/px4_custom_mode.h>

#include <poll.h>
#include <unistd.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#if defined(__PX4_POSIX)
#include <chrono>
#endif

#include "ofnav/ofnav.hpp"
#include "ofnav/eskf_runner.hpp"
#include "ofnav/eskf15_workload.hpp"

#if defined(__PX4_NUTTX) && defined(CONFIG_STACK_COLORATION)
#include <nuttx/arch.h>
#include <nuttx/sched.h>
#define OFNAV_STACK_MONITOR 1
#endif

// Flight builds never include the POSIX-only synthetic CSV replay commands.
#if defined(__PX4_POSIX)
int ofnav_sitl_csv_replay(const char *filename, ofnav::SensorProfile profile);
int ofnav_sitl_eskf_replay(const char *filename);
#endif

namespace {

constexpr uint64_t kMainLoopTimeoutMs = 100;
constexpr unsigned kDefaultRateHz = 100U;
constexpr unsigned kMinRateHz = 20U;
constexpr unsigned kMaxRateHz = 250U;
// Task stack: measured filter call chain <= 2.1 KiB on the host
// (tests/eskf15_benchmark.cpp, budget 3 KiB) + module loop, uORB and
// logging. Must be re-checked on the target with 'top' / stack monitor.
constexpr int kTaskStackBytes = 8192;
constexpr uint64_t kEskfPublishPeriodUs = 100000U;
constexpr uint64_t kMaxAttitudeAgeUs = 100000U;

static uint64_t px4_clock()
{
    return hrt_absolute_time();
}

// Flight configuration of the diagnostic ESKF: 50 Hz filter rate halves the
// prediction load of 100 Hz (docs/ESKF15_FMU_V6X_RU.md), 1 s alignment.
static ofnav::EskfRunnerConfig eskf_config()
{
    ofnav::EskfRunnerConfig cfg{};
    cfg.imu_period_us = 20000U;
    cfg.alignment_samples = 50U;
    cfg.alignment.min_samples = 50U;
    return cfg;
}

#if defined(__PX4_POSIX)
// SITL-only input masking at the module boundary for closed-loop failure
// scenarios in Gazebo (gz_bridge does not implement 'failure optical_flow'
// or 'failure distance_sensor'). Not compiled into flight builds.
static volatile bool g_fault_flow = false;
static volatile bool g_fault_range = false;
#endif

// ---- 'ofnav bench': the shared worst-case workload on THIS processor ----
struct BenchJob {
    ofnav15::WorkloadBuffers *buffers{nullptr};
    ofnav15::WorkloadResult result[2]{};
    long stack_used{-1};
    volatile bool done{false};
};
static BenchJob *g_bench_job = nullptr;

#if defined(__PX4_POSIX)
// Lockstep SITL without a simulator does not advance hrt time: wall clock.
static uint64_t bench_clock()
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch()).count());
}
#else
static uint64_t bench_clock()
{
    return hrt_absolute_time();
}
#endif

static int bench_task_main(int argc, char *argv[])
{
    BenchJob *job = g_bench_job;
    const uint64_t periods[2] = {10000U, 20000U};

    for (int k = 0; k < 2; ++k) {
        job->result[k] = ofnav15::runWorkload(*job->buffers, &bench_clock, 30, periods[k]);
    }

#if defined(OFNAV_STACK_MONITOR)
    job->stack_used = static_cast<long>(up_check_tcbstack(nxsched_self()));
#endif
    job->done = true;
    return 0;
}

static int run_bench()
{
    // Static: a task that outlives the wait below must never write into a
    // returned stack frame. A new run is refused while one is still running.
    static BenchJob job{};

    if (g_bench_job != nullptr && !job.done) {
        PX4_ERR("OFNAV_BENCH previous run still active");
        return -1;
    }

    job = BenchJob{};
    job.buffers = new ofnav15::WorkloadBuffers{};

    if (job.buffers == nullptr) {
        PX4_ERR("OFNAV_BENCH out of memory");
        return -1;
    }

    g_bench_job = &job;
#if defined(__PX4_NUTTX)
    // Same stack size and priority class as the ofnav task; NuttX stack
    // coloration reports the real high-water mark of that task.
    const int tid = px4_task_spawn_cmd("ofnav_bench", SCHED_DEFAULT, SCHED_PRIORITY_DEFAULT,
                                       kTaskStackBytes, bench_task_main, nullptr);

    if (tid < 0) {
        delete job.buffers;
        job.buffers = nullptr;
        job.done = true;
        PX4_ERR("OFNAV_BENCH task spawn failed");
        return -1;
    }

    for (int i = 0; i < 600 && !job.done; ++i) { px4_usleep(100000); }

    if (!job.done) {
        PX4_ERR("OFNAV_BENCH timeout");
        return -1;  // job and buffers stay owned by the still-running task
    }

#else
    (void)bench_task_main(0, nullptr);
#endif

    const uint64_t rates[2] = {100U, 50U};
    bool ok = true;

    for (int k = 0; k < 2; ++k) {
        const auto &r = job.result[k];
        const uint64_t imu_mean = r.imu_samples ? r.imu_total / r.imu_samples : 0U;
        const uint64_t frame_mean = r.frames ? r.frame_total / r.frames : 0U;
        // Busy microseconds per second = CPU load in units of 0.01 %.
        const uint64_t load = imu_mean * rates[k] + frame_mean * 10U;
        ok = ok && r.healthy && r.failed_events == 0U;
        PX4_INFO("OFNAV_BENCH imu_hz=%llu imu_us mean=%llu worst=%llu frame_us mean=%llu worst=%llu",
                 static_cast<unsigned long long>(rates[k]),
                 static_cast<unsigned long long>(imu_mean),
                 static_cast<unsigned long long>(r.imu_worst),
                 static_cast<unsigned long long>(frame_mean),
                 static_cast<unsigned long long>(r.frame_worst));
        PX4_INFO("OFNAV_BENCH imu_hz=%llu cpu_load=%llu.%02llu%% processed=%lu failed=%lu replay<=%u healthy=%u",
                 static_cast<unsigned long long>(rates[k]),
                 static_cast<unsigned long long>(load / 10000U),
                 static_cast<unsigned long long>((load / 100U) % 100U),
                 static_cast<unsigned long>(r.processed), static_cast<unsigned long>(r.failed_events),
                 static_cast<unsigned>(r.max_replay_length), static_cast<unsigned>(r.healthy));
    }

    PX4_INFO("OFNAV_BENCH stack_used=%ld of %d bytes (-1: no NuttX stack coloration) buffers=%u bytes",
             job.stack_used, kTaskStackBytes, static_cast<unsigned>(sizeof(ofnav15::WorkloadBuffers)));
    delete job.buffers;
    job.buffers = nullptr;
    return ok ? 0 : -1;
}

static float safe_div(float n, uint32_t dt_us) noexcept
{
    if (dt_us == 0U) {
        return 0.0f;
    }

    return n / (static_cast<float>(dt_us) * 1.0e-6f);
}

static bool finite(float v) noexcept
{
    return ofnav::isFiniteValue(v);
}

static void quat_to_euler(const float q[4], float &roll, float &pitch, float &yaw) noexcept
{
    const float qw = q[0];
    const float qx = q[1];
    const float qy = q[2];
    const float qz = q[3];

    const float sinr_cosp = 2.0f * (qw * qx + qy * qz);
    const float cosr_cosp = 1.0f - 2.0f * (qx * qx + qy * qy);
    roll = atan2f(sinr_cosp, cosr_cosp);

    const float sinp = 2.0f * (qw * qy - qz * qx);

    if (fabsf(sinp) >= 1.0f) {
        pitch = (sinp > 0.0f) ? ofnav::kPi / 2.0f : -ofnav::kPi / 2.0f;

    } else {
        pitch = asinf(sinp);
    }

    const float siny_cosp = 2.0f * (qw * qz + qx * qy);
    const float cosy_cosp = 1.0f - 2.0f * (qy * qy + qz * qz);
    yaw = atan2f(siny_cosp, cosy_cosp);
}

static bool parse_profile(const char *text, ofnav::SensorProfile &profile) noexcept
{
    if (text == nullptr || std::strcmp(text, "flow") == 0) {
        profile = ofnav::SensorProfile::OpticalFlowSensor;
        return true;
    }

    if (std::strcmp(text, "camera") == 0) {
        profile = ofnav::SensorProfile::DownwardCameraVo;
        return true;
    }

    return false;
}

static void set_debug_name(debug_vect_s &dbg, const char *name) noexcept
{
    std::memset(dbg.name, 0, sizeof(dbg.name));
    std::strncpy(dbg.name, name, sizeof(dbg.name) - 1U);
}

} // namespace

class Ofnav final : public ModuleBase<Ofnav>
{
public:
    Ofnav(unsigned rate_hz, uint8_t min_quality, bool strict_downward_range,
          ofnav::SensorProfile profile, bool eskf_enabled, bool actions_enabled) :
        _rate_hz(rate_hz),
        _min_quality(min_quality),
        _strict_downward_range(strict_downward_range),
        _profile(profile),
        _runtime(ofnav::makeProfileConfig(profile, min_quality))
    {
        // The 15-state ESKF runs as a DIAGNOSTIC estimator in this task.
        // Its output goes to debug_vect only, never to EKF2 or control.
        _eskf.configure(eskf_config(), &px4_clock);
        _actions_enabled = actions_enabled;

        if (!eskf_enabled) {
            _eskf.disable();
        }
    }

    ~Ofnav() override
    {
        if (_flow_sub >= 0) { orb_unsubscribe(_flow_sub); }
        if (_range_sub >= 0) { orb_unsubscribe(_range_sub); }
        if (_imu_sub >= 0) { orb_unsubscribe(_imu_sub); }
        if (_att_sub >= 0) { orb_unsubscribe(_att_sub); }
        if (_status_sub >= 0) { orb_unsubscribe(_status_sub); }
        if (_land_sub >= 0) { orb_unsubscribe(_land_sub); }
    }

    static int task_spawn(int argc, char *argv[])
    {
        _task_id = px4_task_spawn_cmd("ofnav",
                                      SCHED_DEFAULT,
                                      SCHED_PRIORITY_DEFAULT + 20,
                                      kTaskStackBytes,
                                      reinterpret_cast<px4_main_t>(&run_trampoline),
                                      argv);

        if (_task_id < 0) {
            _task_id = -1;
            return -errno;
        }

        return 0;
    }

    static Ofnav *instantiate(int argc, char *argv[])
    {
        unsigned rate_hz = kDefaultRateHz;
        unsigned min_quality = 120U;
        bool strict_downward_range = true;
        bool publish_ev_velocity = false;
        bool eskf_enabled = true;
        bool actions_enabled = false;
        ofnav::SensorProfile profile = ofnav::SensorProfile::OpticalFlowSensor;

        int ch = 0;
        int myoptind = 1;
        const char *myoptarg = nullptr;

        while ((ch = px4_getopt(argc, argv, "r:q:p:ndae", &myoptind, &myoptarg)) != EOF) {
            switch (ch) {
            case 'r': {
                    const int value = std::atoi(myoptarg);
                    rate_hz = static_cast<unsigned>(value > 0 ? value : static_cast<int>(kDefaultRateHz));
                    break;
                }

            case 'q': {
                    const int value = std::atoi(myoptarg);
                    min_quality = static_cast<unsigned>(value < 0 ? 0 : (value > 255 ? 255 : value));
                    break;
                }

            case 'p':
                if (!parse_profile(myoptarg, profile)) {
                    PX4_ERR("unknown profile, use flow or camera");
                    return nullptr;
                }

                break;

            case 'n':
                strict_downward_range = false;
                break;

            case 'd':
                eskf_enabled = false;
                break;

            case 'a':
                actions_enabled = true;
                break;

            case 'e':
                publish_ev_velocity = true;
                break;

            default:
                PX4_ERR("unknown option");
                return nullptr;
            }
        }

        if (rate_hz < kMinRateHz) { rate_hz = kMinRateHz; }
        if (rate_hz > kMaxRateHz) { rate_hz = kMaxRateHz; }

        if (publish_ev_velocity) {
            PX4_ERR("external velocity fusion unavailable: no validated vertical velocity");
            return nullptr;
        }
        return new Ofnav(rate_hz, static_cast<uint8_t>(min_quality), strict_downward_range, profile,
                         eskf_enabled, actions_enabled);
    }

    static int custom_command(int argc, char *argv[])
    {
        if (argc == 1 && std::strcmp(argv[0], "bench") == 0) {
            return run_bench();
        }

#if defined(__PX4_POSIX)

        if (argc == 2 && std::strcmp(argv[0], "fault") == 0) {
            g_fault_flow = std::strcmp(argv[1], "flow") == 0;
            g_fault_range = std::strcmp(argv[1], "range") == 0;
            PX4_INFO("OFNAV_FAULT flow=%u range=%u", static_cast<unsigned>(g_fault_flow),
                     static_cast<unsigned>(g_fault_range));
            return 0;
        }

#endif

#if defined(__PX4_POSIX)
        if ((argc == 2 || argc == 3) && std::strcmp(argv[0], "sitl-replay") == 0) {
            ofnav::SensorProfile profile = ofnav::SensorProfile::OpticalFlowSensor;

            if (!parse_profile(argc == 3 ? argv[2] : nullptr, profile)) {
                return print_usage("unknown profile, use flow or camera");
            }

            return ofnav_sitl_csv_replay(argv[1], profile);
        }

        if (argc == 2 && std::strcmp(argv[0], "sitl-eskf") == 0) {
            return ofnav_sitl_eskf_replay(argv[1]);
        }
#endif
        return print_usage("unknown command");
    }

    static int print_usage(const char *reason = nullptr)
    {
        if (reason) {
            PX4_WARN("%s", reason);
        }

        PRINT_MODULE_DESCRIPTION(
            R"DESCR_STR(
### Description
PX4/NuttX module for running the ofnav RTOS optical-flow navigation core inside PX4.

The module subscribes to uORB topics:
- sensor_optical_flow
- distance_sensor
- vehicle_imu
- vehicle_attitude

It publishes debug_vect diagnostics:
- OFNAV_V: estimated VN, VE and mode
- OFNAV_H: height, quality and reject/accept reason code
- OFNAV_E: 15-state ESKF VN, VE and 100*runner_mode + filter status
- OFNAV_T: ESKF worst frame replay us, worst IMU step us, max replay length

The 15-state ESKF is aligned on the ground (static alignment, autopilot yaw)
and aided by the body-frame flow velocity. It is a diagnostic estimator.

Optional fusion-output path:
- -e is rejected until a validated 3D velocity source is available
- never substitute invented vertical speed for missing measurements

Safety reactions: a debounced, escalating SafetyActionManager maps the
navigation mode to NONE / ALTITUDE_HOLD / LAND / MANUAL_TAKEOVER and
publishes it as OFNAV_A. Only with -a, and only while armed, each NEW
action is sent once as vehicle_command (DO_SET_MODE ALTCTL or
STABILIZED, NAV_LAND). It never commands a return to position mode.

Safe-by-default behavior: without -a this module does not command
the vehicle. It never drives actuators directly.
Fusion output is not supported. The -e option is intentionally rejected.
)DESCR_STR");

        PRINT_MODULE_USAGE_NAME("ofnav", "navigation");
        PRINT_MODULE_USAGE_COMMAND("start");
        PRINT_MODULE_USAGE_PARAM_INT('r', static_cast<int>(kDefaultRateHz), static_cast<int>(kMinRateHz), static_cast<int>(kMaxRateHz), "Module loop rate, Hz", true);
        PRINT_MODULE_USAGE_PARAM_INT('q', 120, 0, 255, "Minimum optical-flow quality", true);
        PRINT_MODULE_USAGE_PARAM_STRING('p', "flow", "flow|camera", "Measurement envelope: optical-flow sensor 0.2-4 m or downward camera VO 5-120 m", true);
        PRINT_MODULE_USAGE_PARAM_FLAG('n', "Do not require downward-facing distance_sensor orientation", true);
        PRINT_MODULE_USAGE_PARAM_FLAG('d', "Disable the diagnostic 15-state ESKF", true);
        PRINT_MODULE_USAGE_PARAM_FLAG('a', "Send safety actions as vehicle_command while armed (SITL/bench first)", true);
        PRINT_MODULE_USAGE_PARAM_FLAG('e', "Reserved: reject unsupported EKF2 velocity fusion until validated 3D velocity exists", true);
        PRINT_MODULE_USAGE_COMMAND("stop");
        PRINT_MODULE_USAGE_COMMAND_DESCR("bench", "Run the worst-case ESKF workload on this CPU: time, load, stack");
        PRINT_MODULE_USAGE_COMMAND("status");
#if defined(__PX4_POSIX)
        PRINT_MODULE_USAGE_COMMAND_DESCR("sitl-replay", "<csv> [flow|camera]: read a synthetic sensor CSV through C++ diagnostics only; no uORB publish, EKF2 or control");
        PRINT_MODULE_USAGE_COMMAND_DESCR("fault", "flow|range|none: SITL-only input masking for failure scenarios");
        PRINT_MODULE_USAGE_COMMAND_DESCR("sitl-eskf", "<csv>: run the 15-state fixed-lag ESKF on a synthetic IMU+VO stream inside PX4; no uORB publish");
#endif
        return 0;
    }

    int print_status() override
    {
        PX4_INFO("profile=%s rate=%u Hz min_quality=%u downward_range=%s",
                 ofnav::toString(_profile),
                 _rate_hz,
                 static_cast<unsigned>(_min_quality),
                 _strict_downward_range ? "true" : "false");
        PX4_INFO("samples=%lu accepted=%lu degraded=%lu failsafe=%lu sim_placeholders=%lu",
                 static_cast<unsigned long>(_samples),
                 static_cast<unsigned long>(_accepted),
                 static_cast<unsigned long>(_degraded),
                 static_cast<unsigned long>(_failsafe),
                 static_cast<unsigned long>(_flow_placeholders));
#if defined(OFNAV_STACK_MONITOR)
        PX4_INFO("task stack min free=%ld of %d bytes", _stack_free_min, kTaskStackBytes);
#endif
        for (unsigned r = 0; r < kReasonSlots; ++r) {
            if (_reasons[r] != 0U) {
                PX4_INFO("flow reason %s: %lu", ofnav::toString(static_cast<ofnav::RejectReason>(r)),
                         static_cast<unsigned long>(_reasons[r]));
            }
        }

        PX4_INFO("last flow raw=(%.4f %.4f) gyro=(%.4f %.4f) dt=%.3f q=%u h=%.2f v_nav=(%.3f %.3f) ekf v=(%.3f %.3f) vel_resets=%lu",
                 static_cast<double>(_last_flow.integrated_x_rad), static_cast<double>(_last_flow.integrated_y_rad),
                 static_cast<double>(_last_flow.integrated_xgyro_rad), static_cast<double>(_last_flow.integrated_ygyro_rad),
                 static_cast<double>(_last_flow.integration_time_s), static_cast<unsigned>(_last_flow.quality), static_cast<double>(_last_out.flow.height_m),
                 static_cast<double>(_last_out.flow.velocity_nav_m_s.x), static_cast<double>(_last_out.flow.velocity_nav_m_s.y),
                 static_cast<double>(_last_out.state.vn_m_s), static_cast<double>(_last_out.state.ve_m_s),
                 static_cast<unsigned long>(_runtime.velocityResets()));
        PX4_INFO("safety engaged=%u", static_cast<unsigned>(_safety.engaged()));
        PX4_INFO("safety action=%s reason=%s commands=%s sent=%lu",
                 ofnav::toString(_last_decision.action), ofnav::toString(_last_decision.reason),
                 _actions_enabled ? "armed-only" : "advisory", static_cast<unsigned long>(_commands_sent));
        const auto &st = _eskf.stats();
        const auto &f = _eskf.filter();
        PX4_INFO("eskf mode=%u init=%u status=%s align=%s restarts=%lu",
                 static_cast<unsigned>(_eskf.mode()), static_cast<unsigned>(_eskf.initSource()),
                 ofnav15::statusName(st.last_status), ofnav15::alignmentStatusName(st.last_alignment),
                 static_cast<unsigned long>(st.filter_restarts));
        PX4_INFO("eskf imu=%lu vel=%lu processed=%lu deferred=%lu dropped=%lu clip=%lu",
                 static_cast<unsigned long>(st.imu_samples), static_cast<unsigned long>(st.velocity_events),
                 static_cast<unsigned long>(st.velocity_processed),
                 static_cast<unsigned long>(st.velocity_deferred),
                 static_cast<unsigned long>(st.velocity_dropped),
                 static_cast<unsigned long>(st.clipping_resets));
        PX4_INFO("eskf time imu worst/mean=%llu/%llu us frame worst/mean=%llu/%llu us replay<=%u",
                 static_cast<unsigned long long>(st.imu_worst_us),
                 static_cast<unsigned long long>(st.imu_samples ? st.imu_total_us / st.imu_samples : 0U),
                 static_cast<unsigned long long>(st.frame_worst_us),
                 static_cast<unsigned long long>(st.velocity_processed + st.velocity_dropped ?
                         st.frame_total_us / (st.velocity_processed + st.velocity_dropped) : 0U),
                 static_cast<unsigned>(st.max_replay_length));
        PX4_INFO("eskf v=(%.3f %.3f %.3f) bg=(%.4f %.4f %.4f)",
                 f.velocity()[0], f.velocity()[1], f.velocity()[2],
                 f.gyroBias()[0], f.gyroBias()[1], f.gyroBias()[2]);
        return 0;
    }

    void run() override
    {
        _flow_sub = orb_subscribe(ORB_ID(sensor_optical_flow));
        _range_sub = orb_subscribe(ORB_ID(distance_sensor));
        _imu_sub = orb_subscribe(ORB_ID(vehicle_imu));
        _att_sub = orb_subscribe(ORB_ID(vehicle_attitude));
        _status_sub = orb_subscribe(ORB_ID(vehicle_status));
        _land_sub = orb_subscribe(ORB_ID(vehicle_land_detected));

        if (_flow_sub < 0 || _range_sub < 0 || _imu_sub < 0 || _att_sub < 0) {
            PX4_ERR("failed to subscribe to required uORB topics");
            return;
        }

        orb_set_interval(_flow_sub, static_cast<unsigned>(1000U / _rate_hz));
        // vehicle_imu is NOT rate-limited: every delta-angle packet is needed
        // for the gyro record of flow sensors without a gyro and for the ESKF.

        px4_pollfd_struct_t fds[2]{};
        fds[0].fd = _flow_sub;
        fds[0].events = POLLIN;
        fds[1].fd = _imu_sub;
        fds[1].events = POLLIN;

        sensor_optical_flow_s flow_msg{};
        distance_sensor_s range_msg{};
        vehicle_imu_s imu_msg{};
        vehicle_attitude_s att_msg{};

        while (!should_exit()) {
            const int pret = px4_poll(fds, 2, static_cast<int>(kMainLoopTimeoutMs));

            if (pret < 0) {
                PX4_ERR("poll error");
                px4_usleep(100000);
                continue;
            }

            bool flow_updated = false;
            bool range_updated = false;
            bool imu_updated = false;
            bool att_updated = false;

            orb_check(_flow_sub, &flow_updated);
            orb_check(_range_sub, &range_updated);
            orb_check(_imu_sub, &imu_updated);
            orb_check(_att_sub, &att_updated);

            if (flow_updated) {
                orb_copy(ORB_ID(sensor_optical_flow), _flow_sub, &flow_msg);
                // Gazebo flow plugin (PX4 OpticalFlow library limitRate):
                // a frame that is not yet published comes with quality -1
                // (255 in uORB) and its single-frame flow, which is ALSO
                // summed into the next published sample, whose timespan
                // covers only its own frame. Merge the placeholder's
                // timespan into the next sample and never fuse it.
                const bool sim = ((flow_msg.device_id >> 16) & 0xFFU) == DRV_FLOW_DEVTYPE_SIM;

                if (sim && flow_msg.quality == 255U) {
                    _flow_carry_us += flow_msg.integration_timespan_us;
                    ++_flow_placeholders;
                    flow_updated = false;

                } else if (_flow_carry_us != 0U) {
                    flow_msg.integration_timespan_us += _flow_carry_us;
                    _flow_carry_us = 0U;
                }
            }
            if (range_updated) { orb_copy(ORB_ID(distance_sensor), _range_sub, &range_msg); }
            if (imu_updated) { orb_copy(ORB_ID(vehicle_imu), _imu_sub, &imu_msg); }
            if (att_updated) { orb_copy(ORB_ID(vehicle_attitude), _att_sub, &att_msg); }

            bool status_updated = false;
            orb_check(_status_sub, &status_updated);

            if (status_updated) { orb_copy(ORB_ID(vehicle_status), _status_sub, &_status_msg); }

            bool land_updated = false;
            orb_check(_land_sub, &land_updated);

            if (land_updated) { orb_copy(ORB_ID(vehicle_land_detected), _land_sub, &_land_msg); }

            const uint64_t now_us = hrt_absolute_time();
#if defined(OFNAV_STACK_MONITOR)

            if (now_us - _last_stack_check_us > 1000000U) {
                const long remain = static_cast<long>(up_check_tcbstack_remain(nxsched_self()));
                _stack_free_min = (_stack_free_min < 0 || remain < _stack_free_min) ? remain : _stack_free_min;
                _last_stack_check_us = now_us;
            }

#endif
            const auto imu = make_imu(imu_msg);
            const auto attitude = make_attitude(att_msg);
            if (imu_updated) {
                // Gyro record for flow sensors without their own gyro.
                _gyro.push(imu_msg.timestamp_sample,
                {imu_msg.delta_angle[0], imu_msg.delta_angle[1], imu_msg.delta_angle[2]},
                imu_msg.delta_angle_dt);
                // Independent IMU prediction, regardless of optical-flow rate.
                _runtime.predictImu(imu, attitude);
                const bool att_fresh = att_msg.timestamp != 0U &&
                                       now_us >= att_msg.timestamp &&
                                       now_us - att_msg.timestamp <= kMaxAttitudeAgeUs;
                _eskf.addImuDelta(imu_msg.timestamp_sample,
                {imu_msg.delta_angle[0], imu_msg.delta_angle[1], imu_msg.delta_angle[2]},
                imu_msg.delta_angle_dt,
                {imu_msg.delta_velocity[0], imu_msg.delta_velocity[1], imu_msg.delta_velocity[2]},
                imu_msg.delta_velocity_dt,
                imu_msg.delta_angle_clipping != 0U || imu_msg.delta_velocity_clipping != 0U,
                {att_msg.q[0], att_msg.q[1], att_msg.q[2], att_msg.q[3]}, att_fresh);
                publish_eskf(now_us);
            }
            if (flow_updated) {
                if (!_flow_pending) { _flow_pending_since_us = now_us; }

                _flow_pending = true;
            }

            bool flow_ready = false;

            if (_flow_pending) {
                if (flow_msg.delta_angle_available) {
                    flow_ready = true;

                } else {
                    // Integrate vehicle gyro over (end - span, end]. If the
                    // IMU record does not yet cover the end of the interval,
                    // wait for the next IMU packet (at most 50 ms).
                    const uint64_t t1 = flow_msg.timestamp_sample;
                    const uint64_t t0 = t1 - flow_msg.integration_timespan_us;
                    _flow_gyro_valid = flow_msg.integration_timespan_us > 0U && t1 > t0 &&
                                       _gyro.integrate(t0, t1, _flow_gyro);
                    flow_ready = _flow_gyro_valid || now_us > _flow_pending_since_us + 50000U;
                }
            }

            if (!flow_ready) {
                // A watchdog must keep running when optical flow stops.
                if (now_us - _last_health_pub_us >= 100000U) {
                    const auto health = _runtime.monitor(now_us, imu,
                                        make_range(flow_msg, range_msg), make_flow(flow_msg),
                                        attitude);
                    publish_debug(health, flow_msg.quality);
                    handle_safety(health, now_us);
                    _last_health_pub_us = now_us;
                }
                continue;
            }

            _flow_pending = false;
            const ofnav::RangeSample range = make_range(flow_msg, range_msg);
            const ofnav::OpticalFlowRadSample flow = make_flow(flow_msg);

            const ofnav::RuntimeOutput out = _runtime.step(imu, range, flow, attitude, now_us);
            ++_samples;
            ++_reasons[static_cast<unsigned>(out.flow.reason) % kReasonSlots];
            _last_out = out;
            _last_flow = flow;

            if (out.flow.accepted) {
                // Body-frame centre-of-mass velocity of this flow interval:
                // independent of the autopilot heading used for NE output.
                _eskf.addBodyVelocity(flow.time_us, out.flow.velocity_body_m_s.x,
                                      out.flow.velocity_body_m_s.y, out.flow.velocity_sigma_m_s);
            }

            if (out.mode == ofnav::NavMode::FlowNav) {
                ++_accepted;

            } else if (out.mode == ofnav::NavMode::FailsafeLand) {
                ++_failsafe;

            } else {
                ++_degraded;
            }

            publish_debug(out, flow_msg.quality);
            handle_safety(out, now_us);
            _last_health_pub_us = now_us;

        }
    }

private:
    ofnav::OpticalFlowRadSample make_flow(const sensor_optical_flow_s &msg) const noexcept
    {
        ofnav::OpticalFlowRadSample flow{};
        flow.time_us = msg.timestamp_sample;
        flow.integration_time_s = static_cast<float>(msg.integration_timespan_us) * 1.0e-6f;
        flow.integrated_x_rad = msg.pixel_flow[0];
        flow.integrated_y_rad = msg.pixel_flow[1];
        // A zero gyro correction is NOT valid when the sensor has no gyro:
        // then the vehicle gyro integrated over the same interval is used
        // (sensor axes == body axes: pixel_flow is raw, SENS_FLOW_ROT=0).
        const bool gyro = msg.delta_angle_available || _flow_gyro_valid;
        flow.integrated_xgyro_rad = msg.delta_angle_available ? msg.delta_angle[0] :
                                    (_flow_gyro_valid ? _flow_gyro.x : NAN);
        flow.integrated_ygyro_rad = msg.delta_angle_available ? msg.delta_angle[1] :
                                    (_flow_gyro_valid ? _flow_gyro.y : NAN);
        flow.integrated_zgyro_rad = msg.delta_angle_available ? msg.delta_angle[2] :
                                    (_flow_gyro_valid ? _flow_gyro.z : NAN);
        flow.quality = msg.quality;
        flow.valid = msg.timestamp_sample != 0U && msg.integration_timespan_us > 0U &&
                     gyro && finite(flow.integrated_x_rad) &&
                     finite(flow.integrated_y_rad) && finite(flow.integrated_xgyro_rad) &&
                     finite(flow.integrated_ygyro_rad);
#if defined(__PX4_POSIX)

        if (g_fault_flow) { flow.valid = false; }

#endif
        return flow;
    }

    ofnav::RangeSample make_range(const sensor_optical_flow_s &flow_msg, const distance_sensor_s &range_msg) const noexcept
    {
        ofnav::RangeSample range{};

        if (flow_msg.distance_available && finite(flow_msg.distance_m) && flow_msg.distance_m > 0.0f) {
            range.time_us = flow_msg.timestamp_sample;
            range.distance_m = flow_msg.distance_m;
            range.valid = true;
            return range;
        }

        const bool orientation_ok = !_strict_downward_range || range_msg.orientation == distance_sensor_s::ROTATION_DOWNWARD_FACING;
        const bool distance_ok = finite(range_msg.current_distance) &&
                                 range_msg.current_distance >= range_msg.min_distance &&
                                 range_msg.current_distance <= range_msg.max_distance;

        range.time_us = range_msg.timestamp;
        range.distance_m = range_msg.current_distance;
        range.valid = range_msg.timestamp != 0U && orientation_ok && distance_ok;
#if defined(__PX4_POSIX)

        if (g_fault_range) { range.valid = false; }

#endif
        return range;
    }

    ofnav::ImuSample make_imu(const vehicle_imu_s &msg) const noexcept
    {
        ofnav::ImuSample imu{};
        imu.time_us = msg.timestamp_sample;
        imu.gyro_rad_s = {
            safe_div(msg.delta_angle[0], msg.delta_angle_dt),
            safe_div(msg.delta_angle[1], msg.delta_angle_dt),
            safe_div(msg.delta_angle[2], msg.delta_angle_dt)
        };
        imu.accel_m_s2 = {
            safe_div(msg.delta_velocity[0], msg.delta_velocity_dt),
            safe_div(msg.delta_velocity[1], msg.delta_velocity_dt),
            safe_div(msg.delta_velocity[2], msg.delta_velocity_dt)
        };
        imu.valid = msg.timestamp_sample != 0U && msg.delta_angle_dt > 0U && msg.delta_velocity_dt > 0U &&
                    msg.delta_angle_clipping == 0U && msg.delta_velocity_clipping == 0U;
        return imu;
    }

    ofnav::AttitudeSample make_attitude(const vehicle_attitude_s &msg) const noexcept
    {
        float roll = 0.0f;
        float pitch = 0.0f;
        float yaw = 0.0f;
        quat_to_euler(msg.q, roll, pitch, yaw);
        return ofnav::AttitudeSample{msg.timestamp_sample, roll, pitch, yaw};
    }

    void publish_eskf(uint64_t now_us)
    {
        if (_eskf.mode() == ofnav::EskfRunnerMode::Disabled ||
            now_us - _last_eskf_pub_us < kEskfPublishPeriodUs) {
            return;
        }

        _last_eskf_pub_us = now_us;
        const auto &f = _eskf.filter();
        const auto &st = _eskf.stats();
        debug_vect_s state{};
        state.timestamp = now_us;
        set_debug_name(state, "OFNAV_E");
        state.x = static_cast<float>(f.velocity()[0]);
        state.y = static_cast<float>(f.velocity()[1]);
        state.z = static_cast<float>(100U * static_cast<unsigned>(_eskf.mode()) +
                                     static_cast<unsigned>(st.last_status));
        publish_vect(state, _debug_eskf_pub, _debug_eskf_instance);
        debug_vect_s timing{};
        timing.timestamp = now_us;
        set_debug_name(timing, "OFNAV_T");
        timing.x = static_cast<float>(st.frame_worst_us);
        timing.y = static_cast<float>(st.imu_worst_us);
        timing.z = static_cast<float>(st.max_replay_length);
        publish_vect(timing, _debug_timing_pub, _debug_timing_instance);
    }

    static void publish_vect(const debug_vect_s &msg, orb_advert_t &pub, int &instance)
    {
        if (pub == nullptr) {
            pub = orb_advertise_multi(ORB_ID(debug_vect), &msg, &instance);

        } else {
            orb_publish(ORB_ID(debug_vect), pub, &msg);
        }
    }

    void handle_safety(const ofnav::RuntimeOutput &out, uint64_t now_us)
    {
        const bool armed_now = _status_msg.arming_state == vehicle_status_s::ARMING_STATE_ARMED;

        if (armed_now && !_was_armed) {
            // Pre-flight conditions (e.g. range below the envelope on the
            // ground) must not latch an in-flight action.
            _safety.reset();
        }

        _was_armed = armed_now;
        // Unknown land state (never published) counts as on the ground.
        const bool in_air = armed_now && _land_msg.timestamp != 0U && !_land_msg.landed;
        _last_decision = _safety.update(out.mode, now_us, in_air);
        debug_vect_s action{};
        action.timestamp = now_us;
        set_debug_name(action, "OFNAV_A");
        action.x = static_cast<float>(_last_decision.action);
        action.y = static_cast<float>(out.mode);
        action.z = static_cast<float>(_commands_sent);
        publish_vect(action, _debug_action_pub, _debug_action_instance);

        const bool armed = _status_msg.arming_state == vehicle_status_s::ARMING_STATE_ARMED;

        if (!_last_decision.changed || !_actions_enabled || !armed) {
            return;
        }

        vehicle_command_s cmd{};
        cmd.timestamp = now_us;
        cmd.target_system = _status_msg.system_id;
        cmd.target_component = _status_msg.component_id;
        cmd.source_system = _status_msg.system_id;
        cmd.source_component = _status_msg.component_id;
        cmd.confirmation = 0;
        cmd.from_external = false;

        switch (_last_decision.action) {
        case ofnav::SafetyAction::AltitudeHold:
            cmd.command = vehicle_command_s::VEHICLE_CMD_DO_SET_MODE;
            cmd.param1 = 1.0f;  // MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
            cmd.param2 = static_cast<float>(PX4_CUSTOM_MAIN_MODE_ALTCTL);
            break;

        case ofnav::SafetyAction::ManualTakeover:
            cmd.command = vehicle_command_s::VEHICLE_CMD_DO_SET_MODE;
            cmd.param1 = 1.0f;
            cmd.param2 = static_cast<float>(PX4_CUSTOM_MAIN_MODE_STABILIZED);
            break;

        case ofnav::SafetyAction::Land:
            cmd.command = vehicle_command_s::VEHICLE_CMD_NAV_LAND;
            // Land in place: no latitude/longitude/altitude given.
            cmd.param5 = static_cast<double>(NAN);
            cmd.param6 = static_cast<double>(NAN);
            cmd.param7 = NAN;
            break;

        case ofnav::SafetyAction::None:
            return;  // never command a return to a position mode
        }

        // Queued topic (ORB_QUEUE_LENGTH from the message definition).
        _command_pub.publish(cmd);
        ++_commands_sent;
        PX4_WARN("ofnav safety action %s (%s)", ofnav::toString(_last_decision.action),
                 ofnav::toString(_last_decision.reason));
    }

    void publish_debug(const ofnav::RuntimeOutput &out, uint8_t quality)
    {
        debug_vect_s vel{};
        vel.timestamp = hrt_absolute_time();
        set_debug_name(vel, "OFNAV_V");
        vel.x = out.state.vn_m_s;
        vel.y = out.state.ve_m_s;
        vel.z = static_cast<float>(out.mode);

        if (_debug_vel_pub == nullptr) {
            _debug_vel_pub = orb_advertise_multi(ORB_ID(debug_vect), &vel,
                                                &_debug_vel_instance);

        } else {
            orb_publish(ORB_ID(debug_vect), _debug_vel_pub, &vel);
        }

        debug_vect_s health{};
        health.timestamp = vel.timestamp;
        set_debug_name(health, "OFNAV_H");
        health.x = out.flow.height_m;
        health.y = static_cast<float>(quality);
        health.z = static_cast<float>(out.flow.reason);

        if (_debug_health_pub == nullptr) {
            _debug_health_pub = orb_advertise_multi(ORB_ID(debug_vect), &health,
                                                   &_debug_health_instance);

        } else {
            orb_publish(ORB_ID(debug_vect), _debug_health_pub, &health);
        }
    }

    unsigned _rate_hz{kDefaultRateHz};
    uint8_t _min_quality{120U};
    bool _strict_downward_range{true};
    ofnav::SensorProfile _profile{ofnav::SensorProfile::OpticalFlowSensor};

    int _flow_sub{-1};
    int _range_sub{-1};
    int _imu_sub{-1};
    int _att_sub{-1};

    bool _actions_enabled{false};
    bool _was_armed{false};
    static constexpr unsigned kReasonSlots = 16U;
    uint32_t _reasons[kReasonSlots] {};
    ofnav::RuntimeOutput _last_out{};
    uint32_t _flow_carry_us{0U};
    uint32_t _flow_placeholders{0U};
    ofnav::OpticalFlowRadSample _last_flow{};
    ofnav::GyroIntegrator _gyro{};
    ofnav::Vector3f _flow_gyro{};
    bool _flow_gyro_valid{false};
    bool _flow_pending{false};
    uint64_t _flow_pending_since_us{0U};
    long _stack_free_min{-1};
    uint64_t _last_stack_check_us{0U};
    int _status_sub{-1};
    int _land_sub{-1};
    vehicle_land_detected_s _land_msg{};
    vehicle_status_s _status_msg{};
    ofnav::SafetyActionManager _safety{};
    ofnav::SafetyDecision _last_decision{};
    uORB::Publication<vehicle_command_s> _command_pub{ORB_ID(vehicle_command)};
    orb_advert_t _debug_action_pub{nullptr};
    int _debug_action_instance{-1};
    uint32_t _commands_sent{0U};
    orb_advert_t _debug_eskf_pub{nullptr};
    orb_advert_t _debug_timing_pub{nullptr};
    int _debug_eskf_instance{-1};
    int _debug_timing_instance{-1};
    uint64_t _last_eskf_pub_us{0U};
    ofnav::EskfRunner _eskf{};
    orb_advert_t _debug_vel_pub{nullptr};
    orb_advert_t _debug_health_pub{nullptr};
    int _debug_vel_instance{-1};
    int _debug_health_instance{-1};

    ofnav::OfNavRuntime _runtime;

    uint32_t _samples{0U};
    uint32_t _accepted{0U};
    uint32_t _degraded{0U};
    uint32_t _failsafe{0U};
    uint64_t _last_health_pub_us{0U};
};

extern "C" __EXPORT int ofnav_main(int argc, char *argv[])
{
    return Ofnav::main(argc, argv);
}
