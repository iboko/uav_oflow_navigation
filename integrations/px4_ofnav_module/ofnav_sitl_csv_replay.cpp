// POSIX/PX4 SITL only: read a synthetic navigation CSV into the existing
// C++ ofnav core. Never publishes uORB sensors, vehicle_odometry, actuators
// or any EKF2 input. No motion command is possible from this path.
#include <px4_platform_common/px4_config.h>

#if defined(__PX4_POSIX)

#include <px4_platform_common/log.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "ofnav/ofnav.hpp"

namespace {

struct Record {
    unsigned long long stamp{0};
    unsigned long long arrival{0};
    float height{0.0f};
    float yaw{0.0f};
    float gx{0.0f};
    float gy{0.0f};
    float gz{0.0f};
    float ax{0.0f};
    float ay{0.0f};
    float az{0.0f};
    float integrated_x{0.0f};
    float integrated_y{0.0f};
    float dt{0.0f};
    unsigned quality{0};
    unsigned valid{0};
};

bool parse(const char *line, Record &r)
{
    return std::sscanf(line,
        "%llu,%llu,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%u,%u",
        &r.stamp, &r.arrival, &r.height, &r.yaw, &r.gx, &r.gy, &r.gz,
        &r.ax, &r.ay, &r.az, &r.integrated_x, &r.integrated_y,
        &r.dt, &r.quality, &r.valid) == 15;
}

bool finite(const Record &r)
{
    return std::isfinite(r.height) && std::isfinite(r.yaw) &&
           std::isfinite(r.gx) && std::isfinite(r.gy) &&
           std::isfinite(r.gz) && std::isfinite(r.ax) &&
           std::isfinite(r.ay) && std::isfinite(r.az) &&
           std::isfinite(r.integrated_x) && std::isfinite(r.integrated_y) &&
           std::isfinite(r.dt);
}

} // namespace

int ofnav_sitl_csv_replay(const char *filename)
{
    FILE *fp = std::fopen(filename, "r");
    if (fp == nullptr) {
        PX4_ERR("OFNAV_SITL_REPLAY cannot open CSV");
        return -1;
    }

    char line[512]{};
    if (std::fgets(line, sizeof(line), fp) == nullptr ||
            std::strncmp(line, "timestamp_us,arrival_us,height_m,yaw_rad,", 40) != 0) {
        PX4_ERR("OFNAV_SITL_REPLAY invalid header");
        std::fclose(fp);
        return -1;
    }

    ofnav::Config cfg{};
    // Preserve the existing flight-module safety envelope!
    cfg.min_height_m = 0.2f;
    cfg.max_height_m = 4.0f;
    cfg.min_quality = 120U;
    cfg.max_horizontal_speed_m_s = 1.5f;
    ofnav::FlowVelocityEstimator estimator(cfg);

    unsigned rows = 0;
    unsigned valid_rows = 0;
    unsigned height_rejected = 0;
    unsigned other_rejected = 0;
    unsigned flow_accepted = 0;
    unsigned explicitly_invalid = 0;
    unsigned line_count = 1;
    uint64_t last_stamp = 0;
    uint64_t last_arrival = 0;
    bool unsafe_acceptance = false;
    bool invalid_file = false;

    while (std::fgets(line, sizeof(line), fp) != nullptr) {
        ++line_count;
        if (line[0] == '\n' || line[0] == '\r') {
            continue;
        }
        Record r{};
        if (!parse(line, r) || r.stamp == 0ULL ||
                r.stamp <= last_stamp || r.arrival < r.stamp ||
                r.arrival <= last_arrival ||
                r.height <= 0.f || r.quality > 255U || r.valid > 1U ||
                (r.valid != 0U && !finite(r)) ||
                (r.valid == 0U && !std::isfinite(r.height))) {
            PX4_ERR("OFNAV_SITL_REPLAY malformed line=%u", line_count);
            invalid_file = true;
            break;
        }
        last_stamp = static_cast<uint64_t>(r.stamp);
        last_arrival = static_cast<uint64_t>(r.arrival);
        ++rows;
        if (r.valid == 0U) {
            ++explicitly_invalid;
            continue;
        }
        ++valid_rows;

        // Supplied gyro and acceleration originate from the same generated
        // trajectory. No made-up gyro compensation: x/y rotation is zero
        // in current small-yaw simulation. z-rotation is recorded explicitly.
        const ofnav::ImuSample imu{
            static_cast<uint64_t>(r.stamp),
            {r.gx, r.gy, r.gz}, {r.ax, r.ay, r.az}, true
        };
        const ofnav::AttitudeSample attitude{
            static_cast<uint64_t>(r.stamp), 0.f, 0.f, r.yaw
        };
        const ofnav::RangeSample range{
            static_cast<uint64_t>(r.stamp), r.height, true
        };
        const ofnav::OpticalFlowRadSample flow{
            static_cast<uint64_t>(r.stamp),
            r.dt, r.integrated_x, r.integrated_y,
            0.f, 0.f, r.gz * r.dt,
            static_cast<uint8_t>(r.quality), true
        };
        if (r.arrival - r.stamp > cfg.max_flow_age_us) {
            // Time freshness would reject such a sample in the live module.
            // Do not claim it was processed as a current flow observation.
            ++other_rejected;
            continue;
        }
        const auto estimate = estimator.update(flow, range, imu, attitude);
        if (estimate.accepted) {
            ++flow_accepted;
            if (r.height > cfg.max_height_m) {
                unsafe_acceptance = true;
            }
        } else if (estimate.reason == ofnav::RejectReason::HeightTooHigh) {
            ++height_rejected;
        } else {
            ++other_rejected;
        }
    }

    std::fclose(fp);
    if (rows == 0U || valid_rows == 0U || invalid_file || unsafe_acceptance) {
        PX4_ERR("OFNAV_SITL_REPLAY INVALID rows=%u valid=%u unsafe=%u",
                rows, valid_rows, static_cast<unsigned>(unsafe_acceptance));
        return -1;
    }
    PX4_INFO("OFNAV_SITL_REPLAY_OK rows=%u valid=%u accepted=%u height_rejected=%u other_rejected=%u invalid=%u",
             rows, valid_rows, flow_accepted, height_rejected,
             other_rejected, explicitly_invalid);
    // An expected rejected 100 m frame is a PASS of the safety test.
    // No uORB topic was advertised and no estimator/control state updated.
    return 0;
}

#endif // __PX4_POSIX
