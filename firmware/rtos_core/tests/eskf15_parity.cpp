// Deterministic numeric parity fixture for the Python implementation.
// Standalone host executable, NOT a command to PX4/uORB/EKF2.
#include "ofnav/eskf15.hpp"

#include <cmath>
#include <cstdio>
#include <cstdint>

int main() {
    ofnav15::FixedLag replay({},500000U);
    for(int i=0;i<50;++i) {
        const double t=static_cast<double>(i)*.01;
        const ofnav15::Imu reading{
            static_cast<std::uint64_t>(1000000+10000*i),
            {0.,0.,.02},
            {.1*std::sin(.37*t),
             .02*std::cos(.31*t),
             -9.80665 + .02*std::sin(.41*t)}
        };
        if (replay.pushImu(reading)!=ofnav15::Status::InertialOnly) {
            std::fprintf(stderr,"C++ predict failure at %d\n",i);
            return 1;
        }
    }
    const ofnav15::Visual late{
        1255000U,{.13,.03},{.04,0.,0.,.04}
    };
    const ofnav15::Visual early{
        1120000U,{.07,.02},{.04,0.,0.,.04}
    };
    if (replay.pushDelayedVelocity(late)!=ofnav15::Status::VisualCorrected ||
        replay.pushDelayedVelocity(early)!=ofnav15::Status::VisualCorrected) {
        std::fprintf(stderr,"C++ delayed replay failure\n");
        return 1;
    }
    const auto &f=replay.current();
    std::printf("{\"timestamp_us\":%llu,\"accepted\":%u,\"rejected\":%u,",
                static_cast<unsigned long long>(f.timestamp()),
                f.accepted(),f.rejected());
    auto print_array=[](const char *key, const auto &values, bool comma) {
        std::printf("\"%s\":[",key);
        bool first=true;
        for(double value:values) {
            if (!first) {std::printf(",");}
            first=false;
            std::printf("%.17g",value);
        }
        std::printf(comma?"],":"]");
    };
    print_array("position",f.position(),true);
    print_array("velocity",f.velocity(),true);
    print_array("quaternion",f.quaternion(),true);
    print_array("gyro_bias",f.gyroBias(),true);
    print_array("accel_bias",f.accelBias(),true);
    print_array("covariance",f.covariance(),true);
    std::printf("\"nis\":%.17g}\n",f.lastNis());
    return 0;
}
