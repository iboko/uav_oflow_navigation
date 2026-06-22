#include <px4_platform_common/module.h>
#include <px4_platform_common/log.h>
#include <px4_platform_common/px4_config.h>
#include <px4_platform_common/posix.h>
#include <px4_platform_common/tasks.h>
#include <drivers/drv_hrt.h>

#include <uORB/uORB.h>
#include <uORB/topics/sensor_optical_flow.h>
#include <uORB/topics/distance_sensor.h>
#include <uORB/topics/vehicle_imu.h>
#include <uORB/topics/vehicle_attitude.h>
#include <uORB/topics/debug_vect.h>

#include <poll.h>
#include <unistd.h>
#include <cmath>
#include <cstring>
#include <cstdio>

#include "ofnav/ofnav.hpp"

using namespace time_literals;

namespace {

constexpr uint64_t kMainLoopTimeoutMs = 100;
constexpr unsigned kDefaultRateHz = 100U;
constexpr unsigned kMinRateHz = 20U;
constexpr unsigned kMaxRateHz = 250U;

static float safe_div(float n, uint32_t dt_us) noexcept
{
    if (dt_us == 0U) {
        return 0.0f;
    }

    return n / (static_cast<float>(dt_us) * 1.0e-6f);
}

static bool finite(float v) noexcept
{
    return std::isfinite(v);
}

static void quat_to_euler(const float q[4], float &roll, float &pitch, float &yaw) noexcept
{
    const float qw = q[0];
    const float qx = q[1];
    const float qy = q[2];
    const float qz = q[3];

    const float sinr_cosp = 2.0f * (qw * qx + qy * qz);
    const float cosr_cosp = 1.0f - 2.0f * (qx * qx + qy * qy);
    roll = std::atan2(sinr_cosp, cosr_cosp);

    const float sinp = 2.0f * (qw * qy - qz * qx);

    if (std::fabs(sinp) >= 1.0f) {
        pitch = std::copysign(ofnav::kPi / 2.0f, sinp);

    } else {
        pitch = std::asin(sinp);
    }

    const float siny_cosp = 2.0f * (qw * qz + qx * qy);
    const float cosy_cosp = 1.0f - 2.0f * (qy * qy + qz * qz);
    yaw = std::atan2(siny_cosp, cosy_cosp);
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
    Ofnav(unsigned rate_hz, uint8_t min_quality, bool strict_downward_range) :
        _rate_hz(rate_hz),
        _min_quality(min_quality),
        _strict_downward_range(strict_downward_range),
        _runtime(make_config(min_quality))
    {
    }

    ~Ofnav() override
    {
        if (_flow_sub >= 0) { orb_unsubscribe(_flow_sub); }
        if (_range_sub >= 0) { orb_unsubscribe(_range_sub); }
        if (_imu_sub >= 0) { orb_unsubscribe(_imu_sub); }
        if (_att_sub >= 0) { orb_unsubscribe(_att_sub); }
    }

    static int task_spawn(int argc, char *argv[])
    {
        _task_id = px4_task_spawn_cmd("ofnav",
                                      SCHED_DEFAULT,
                                      SCHED_PRIORITY_DEFAULT + 20,
                                      4096,
                                      reinterpret_cast<px4_main_t>(run_trampoline),
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

        int ch = 0;
        int myoptind = 1;
        const char *myoptarg = nullptr;

        while ((ch = px4_getopt(argc, argv, "r:q:n", &myoptind, &myoptarg)) != EOF) {
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

            case 'n':
                strict_downward_range = false;
                break;

            default:
                PX4_ERR("unknown option");
                return nullptr;
            }
        }

        if (rate_hz < kMinRateHz) { rate_hz = kMinRateHz; }
        if (rate_hz > kMaxRateHz) { rate_hz = kMaxRateHz; }

        return new Ofnav(rate_hz, static_cast<uint8_t>(min_quality), strict_downward_range);
    }

    static int custom_command(int argc, char *argv[])
    {
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
- OFNAV_H: height, quality and innovation value

Safe-by-default behavior: this module does not replace EKF2 and does not command actuators.
Use it first as a flight-stack-native monitor before enabling any fusion/control path.
)DESCR_STR");

        PRINT_MODULE_USAGE_NAME("ofnav", "navigation");
        PRINT_MODULE_USAGE_COMMAND("start");
        PRINT_MODULE_USAGE_PARAM_INT('r', static_cast<int>(kDefaultRateHz), static_cast<int>(kMinRateHz), static_cast<int>(kMaxRateHz), "Module loop rate, Hz", true);
        PRINT_MODULE_USAGE_PARAM_INT('q', 120, 0, 255, "Minimum optical-flow quality", true);
        PRINT_MODULE_USAGE_PARAM_FLAG('n', "Do not require downward-facing distance_sensor orientation", true);
        PRINT_MODULE_USAGE_COMMAND("stop");
        PRINT_MODULE_USAGE_COMMAND("status");
        return 0;
    }

    int print_status() override
    {
        PX4_INFO("running: rate=%u Hz min_quality=%u strict_downward_range=%s",
                 _rate_hz,
                 static_cast<unsigned>(_min_quality),
                 _strict_downward_range ? "true" : "false");
        PX4_INFO("samples=%lu accepted=%lu degraded=%lu failsafe=%lu",
                 static_cast<unsigned long>(_samples),
                 static_cast<unsigned long>(_accepted),
                 static_cast<unsigned long>(_degraded),
                 static_cast<unsigned long>(_failsafe));
        return 0;
    }

    void run() override
    {
        _flow_sub = orb_subscribe(ORB_ID(sensor_optical_flow));
        _range_sub = orb_subscribe(ORB_ID(distance_sensor));
        _imu_sub = orb_subscribe(ORB_ID(vehicle_imu));
        _att_sub = orb_subscribe(ORB_ID(vehicle_attitude));

        if (_flow_sub < 0 || _range_sub < 0 || _imu_sub < 0 || _att_sub < 0) {
            PX4_ERR("failed to subscribe to required uORB topics");
            return;
        }

        orb_set_interval(_flow_sub, static_cast<unsigned>(1000U / _rate_hz));

        pollfd fds{};
        fds.fd = _flow_sub;
        fds.events = POLLIN;

        sensor_optical_flow_s flow_msg{};
        distance_sensor_s range_msg{};
        vehicle_imu_s imu_msg{};
        vehicle_attitude_s att_msg{};

        while (!should_exit()) {
            const int pret = px4_poll(&fds, 1, static_cast<int>(kMainLoopTimeoutMs));

            if (pret < 0) {
                PX4_ERR("poll error");
                px4_usleep(100_ms);
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

            if (flow_updated) { orb_copy(ORB_ID(sensor_optical_flow), _flow_sub, &flow_msg); }
            if (range_updated) { orb_copy(ORB_ID(distance_sensor), _range_sub, &range_msg); }
            if (imu_updated) { orb_copy(ORB_ID(vehicle_imu), _imu_sub, &imu_msg); }
            if (att_updated) { orb_copy(ORB_ID(vehicle_attitude), _att_sub, &att_msg); }

            if (!flow_updated) {
                continue;
            }

            const ofnav::ImuSample imu = make_imu(imu_msg);
            const ofnav::RangeSample range = make_range(flow_msg, range_msg);
            const ofnav::OpticalFlowRadSample flow = make_flow(flow_msg);
            const ofnav::AttitudeSample attitude = make_attitude(att_msg);

            const ofnav::RuntimeOutput out = _runtime.step(imu, range, flow, attitude);
            ++_samples;

            if (out.mode == ofnav::NavMode::FlowNav) {
                ++_accepted;

            } else if (out.mode == ofnav::NavMode::FailsafeLand) {
                ++_failsafe;

            } else {
                ++_degraded;
            }

            publish_debug(out, flow_msg.quality);
        }
    }

private:
    static ofnav::Config make_config(uint8_t min_quality) noexcept
    {
        ofnav::Config cfg{};
        cfg.min_height_m = 0.20f;
        cfg.max_height_m = 4.00f;
        cfg.max_tilt_rad = 0.45f;
        cfg.max_abs_flow_rate_rad_s = 7.4f;
        cfg.max_horizontal_speed_m_s = 1.5f;
        cfg.min_quality = min_quality;
        cfg.flow_scale_x = 1.0f;
        cfg.flow_scale_y = 1.0f;
        cfg.sensor_offset_body_m = {0.0f, 0.0f, 0.0f};
        return cfg;
    }

    ofnav::OpticalFlowRadSample make_flow(const sensor_optical_flow_s &msg) const noexcept
    {
        ofnav::OpticalFlowRadSample flow{};
        flow.time_us = msg.timestamp;
        flow.integration_time_s = static_cast<float>(msg.integration_timespan_us) * 1.0e-6f;
        flow.integrated_x_rad = msg.pixel_flow[0];
        flow.integrated_y_rad = msg.pixel_flow[1];
        flow.integrated_xgyro_rad = msg.delta_angle_available ? msg.delta_angle[0] : 0.0f;
        flow.integrated_ygyro_rad = msg.delta_angle_available ? msg.delta_angle[1] : 0.0f;
        flow.integrated_zgyro_rad = msg.delta_angle_available ? msg.delta_angle[2] : 0.0f;
        flow.quality = msg.quality;
        flow.valid = msg.timestamp != 0U && msg.integration_timespan_us > 0U && finite(flow.integrated_x_rad) && finite(flow.integrated_y_rad);
        return flow;
    }

    ofnav::RangeSample make_range(const sensor_optical_flow_s &flow_msg, const distance_sensor_s &range_msg) const noexcept
    {
        ofnav::RangeSample range{};

        if (flow_msg.distance_available && finite(flow_msg.distance_m) && flow_msg.distance_m > 0.0f) {
            range.time_us = flow_msg.timestamp;
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
        return range;
    }

    ofnav::ImuSample make_imu(const vehicle_imu_s &msg) const noexcept
    {
        ofnav::ImuSample imu{};
        imu.time_us = msg.timestamp;
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
        imu.valid = msg.timestamp != 0U && msg.delta_angle_dt > 0U && msg.delta_velocity_dt > 0U &&
                    msg.delta_angle_clipping == 0U && msg.delta_velocity_clipping == 0U;
        return imu;
    }

    ofnav::AttitudeSample make_attitude(const vehicle_attitude_s &msg) const noexcept
    {
        float roll = 0.0f;
        float pitch = 0.0f;
        float yaw = 0.0f;
        quat_to_euler(msg.q, roll, pitch, yaw);
        return ofnav::AttitudeSample{msg.timestamp, roll, pitch, yaw};
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
            _debug_vel_pub = orb_advertise(ORB_ID(debug_vect), &vel);

        } else {
            orb_publish(ORB_ID(debug_vect), _debug_vel_pub, &vel);
        }

        debug_vect_s health{};
        health.timestamp = vel.timestamp;
        set_debug_name(health, "OFNAV_H");
        health.x = out.flow.height_m;
        health.y = static_cast<float>(quality);
        health.z = _runtime_diag_innovation;

        if (_debug_health_pub == nullptr) {
            _debug_health_pub = orb_advertise(ORB_ID(debug_vect), &health);

        } else {
            orb_publish(ORB_ID(debug_vect), _debug_health_pub, &health);
        }
    }

    unsigned _rate_hz{kDefaultRateHz};
    uint8_t _min_quality{120U};
    bool _strict_downward_range{true};

    int _flow_sub{-1};
    int _range_sub{-1};
    int _imu_sub{-1};
    int _att_sub{-1};

    orb_advert_t _debug_vel_pub{nullptr};
    orb_advert_t _debug_health_pub{nullptr};

    ofnav::OfNavRuntime _runtime;
    float _runtime_diag_innovation{0.0f};

    uint32_t _samples{0U};
    uint32_t _accepted{0U};
    uint32_t _degraded{0U};
    uint32_t _failsafe{0U};
};

extern "C" __EXPORT int ofnav_main(int argc, char *argv[])
{
    return Ofnav::main(argc, argv);
}
