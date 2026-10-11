// POSIX/PX4 SITL only: run the C++17 15-state fixed-lag ESKF INSIDE the
// PX4 process on an arrival-ordered synthetic IMU + delayed VO stream
// (src/export_px4_sitl_replay.py --eskf-events-csv). Measures execution
// time with the host monotonic clock: in lockstep SITL without a simulator
// hrt_absolute_time() is simulated time and does not advance. Never
// publishes uORB topics, never feeds EKF2 and cannot command actuators.
#include <px4_platform_common/px4_config.h>

#if defined(__PX4_POSIX)

#include <px4_platform_common/log.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

#include "ofnav/eskf15.hpp"

namespace {

struct Event {
    char kind{0};
    unsigned long long arrival{0};
    unsigned long long time{0};
    double v[6]{};
};

bool parse(const char *line, Event &e)
{
    return std::sscanf(line, "%c,%llu,%llu,%lf,%lf,%lf,%lf,%lf,%lf",
                       &e.kind, &e.arrival, &e.time, &e.v[0], &e.v[1], &e.v[2],
                       &e.v[3], &e.v[4], &e.v[5]) == 9 &&
           (e.kind == 'I' || e.kind == 'B') && e.time > 0ULL && e.arrival >= e.time;
}

uint64_t wallMicros()
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch()).count());
}

} // namespace

int ofnav_sitl_eskf_replay(const char *filename)
{
    FILE *fp = std::fopen(filename, "r");

    if (fp == nullptr) {
        PX4_ERR("OFNAV_SITL_ESKF cannot open CSV");
        return -1;
    }

    char line[512] {};

    if (std::fgets(line, sizeof(line), fp) == nullptr ||
        std::strncmp(line, "kind,arrival_us,time_us,", 24) != 0) {
        PX4_ERR("OFNAV_SITL_ESKF invalid header");
        std::fclose(fp);
        return -1;
    }

    // ~48 KiB: heap for this POSIX diagnostic command, never the stack.
    auto *lag = new (std::nothrow) ofnav15::FixedLag(ofnav15::Parameters{}, 300000U);

    if (lag == nullptr) {
        std::fclose(fp);
        return -1;
    }

    unsigned imu = 0, frames = 0, processed = 0, failed = 0;
    uint64_t imu_worst = 0, frame_worst = 0, imu_total = 0, frame_total = 0;
    unsigned long long last_arrival = 0;
    bool invalid = false;

    while (std::fgets(line, sizeof(line), fp) != nullptr) {
        if (line[0] == '\n' || line[0] == '\r') { continue; }

        Event e{};

        if (!parse(line, e) || e.arrival < last_arrival) {
            invalid = true;
            break;
        }

        last_arrival = e.arrival;
        const uint64_t start = wallMicros();

        if (e.kind == 'I') {
            const ofnav15::Imu sample{e.time, {e.v[0], e.v[1], e.v[2]}, {e.v[3], e.v[4], e.v[5]}};

            if (lag->pushImu(sample) != ofnav15::Status::InertialOnly) { ++failed; }

            const uint64_t dt = wallMicros() - start;
            imu_total += dt;
            imu_worst = dt > imu_worst ? dt : imu_worst;
            ++imu;

        } else {
            ofnav15::Measurement m{};
            m.time_us = e.time;
            m.kind = ofnav15::MeasurementKind::BodyVelocityXy;
            m.value = {e.v[0], e.v[1]};
            m.covariance = {e.v[2], e.v[3], e.v[3], e.v[4]};
            ofnav15::Status status = ofnav15::Status::ReplayFailed;
            (void)lag->pushDelayedFrame(&m, 1, &status);

            if (ofnav15::measurementProcessed(status)) { ++processed; }

            const uint64_t dt = wallMicros() - start;
            frame_total += dt;
            frame_worst = dt > frame_worst ? dt : frame_worst;
            ++frames;
        }
    }

    std::fclose(fp);
    const ofnav15::Filter &f = lag->current();
    const bool healthy = !lag->failed() && f.healthy();

    if (invalid || imu == 0U || !healthy) {
        PX4_ERR("OFNAV_SITL_ESKF INVALID imu=%u frames=%u invalid=%u healthy=%u",
                imu, frames, static_cast<unsigned>(invalid), static_cast<unsigned>(healthy));
        delete lag;
        return -1;
    }

    // PX4 log lines are length-limited: state and timing on separate lines.
    PX4_INFO("OFNAV_SITL_ESKF_OK imu=%u frames=%u processed=%u failed=%u accepted=%u rejected=%u",
             imu, frames, processed, failed, f.accepted(), f.rejected());
    PX4_INFO("OFNAV_SITL_ESKF_STATE pn=%.9f pe=%.9f vn=%.9f ve=%.9f",
             f.position()[0], f.position()[1], f.velocity()[0], f.velocity()[1]);
    PX4_INFO("OFNAV_SITL_ESKF_TIME imu_worst_us=%llu imu_mean_us=%llu frame_worst_us=%llu frame_mean_us=%llu",
             static_cast<unsigned long long>(imu_worst),
             static_cast<unsigned long long>(imu ? imu_total / imu : 0U),
             static_cast<unsigned long long>(frame_worst),
             static_cast<unsigned long long>(frames ? frame_total / frames : 0U));
    delete lag;
    return 0;
}

#endif // __PX4_POSIX
