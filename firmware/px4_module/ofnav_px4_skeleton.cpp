// PX4/NuttX integration skeleton for ofnav_rtos_core.
//
// This file is intentionally not compiled by the standalone CMake project.
// It shows how the deterministic core should be wrapped inside a PX4 module.
// Replace comments with actual uORB subscriptions/publications in a PX4 tree.

#include "ofnav/ofnav.hpp"

namespace ofnav_px4_skeleton {

class OpticalFlowNavigationModule {
public:
    OpticalFlowNavigationModule()
        : runtime_(makeConfig()) {}

    void runOnce() {
        // 1. Read uORB topics, for example:
        //    vehicle_imu
        //    vehicle_attitude
        //    distance_sensor
        //    optical_flow
        //
        // 2. Convert all timestamps to PX4 hrt_absolute_time() compatible microseconds.
        // 3. Fill ofnav::ImuSample, RangeSample, OpticalFlowRadSample, AttitudeSample.
        // 4. Call runtime_.step(...).
        // 5. Publish diagnostic topic or feed validated velocity to the selected fusion path.

        ofnav::ImuSample imu{};
        ofnav::RangeSample range{};
        ofnav::OpticalFlowRadSample flow{};
        ofnav::AttitudeSample attitude{};

        const ofnav::RuntimeOutput out = runtime_.step(imu, range, flow, attitude);

        switch (out.mode) {
        case ofnav::NavMode::FlowNav:
            // Publish valid diagnostic velocity/state.
            break;
        case ofnav::NavMode::DegradedHold:
            // Request hold/brake or mark optical-flow aiding invalid.
            break;
        case ofnav::NavMode::FailsafeLand:
            // Trigger commander-compatible failsafe path.
            break;
        case ofnav::NavMode::ManualRequired:
        case ofnav::NavMode::Standby:
            break;
        }
    }

private:
    static ofnav::Config makeConfig() {
        ofnav::Config cfg{};
        cfg.min_height_m = 0.20F;
        cfg.max_height_m = 4.00F;
        cfg.min_quality = 120U;
        cfg.max_tilt_rad = 0.35F;
        cfg.max_horizontal_speed_m_s = 1.5F;
        cfg.sensor_offset_body_m = {0.0F, 0.0F, 0.0F};
        cfg.flow_scale_x = 1.0F;
        cfg.flow_scale_y = 1.0F;
        return cfg;
    }

    ofnav::OfNavRuntime runtime_;
};

} // namespace ofnav_px4_skeleton
