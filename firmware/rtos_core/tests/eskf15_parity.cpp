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
    // Frame 1 (exposure between IMU samples): all four aiding kinds,
    // orthophoto fix with automatic covariance intersection. Delivered
    // BEFORE the older frame 2 (velocity + independent map fix).
    using K=ofnav15::MeasurementKind;
    auto make=[](std::uint64_t t, K kind, double a, double b,
                 double c00, double c01, double c11, double ci) {
        ofnav15::Measurement m{};
        m.time_us=t; m.kind=kind; m.value={a,b};
        m.covariance={c00,c01,c01,c11}; m.ci_weight=ci;
        return m;
    };
    const ofnav15::Measurement late[4]={
        make(1255000U,K::VelocityNe,.13,.03,.04,0.,.04,1.),
        make(1255000U,K::BodyVelocityXy,.128,.036,.05,.004,.05,1.),
        make(1255000U,K::Yaw,.006,0.,.0004,0.,0.,1.),
        make(1255000U,K::PositionNe,.4,-.3,4.,.5,3.,0.)
    };
    ofnav15::Measurement early[3]={
        make(1120000U,K::VelocityNe,.07,.02,.04,0.,.04,1.),
        make(1120000U,K::PositionNe,.1,.05,9.,0.,9.,1.),
        make(1120000U,K::MapGroundPointNe,.3,-.2,.25,.02,.36,0.)
    };
    early[2].aux={100.,.02,-.01,1.};
    ofnav15::Status s_late[4]{}, s_early[3]{};
    (void)replay.pushDelayedFrame(late,4,s_late);
    (void)replay.pushDelayedFrame(early,3,s_early);
    for(const auto s:s_late) {
        if (!ofnav15::measurementProcessed(s)) {
            std::fprintf(stderr,"C++ delayed replay failure %s\n",ofnav15::statusName(s));
            return 1;
        }
    }
    for(const auto s:s_early) {
        if (!ofnav15::measurementProcessed(s)) {
            std::fprintf(stderr,"C++ delayed replay failure %s\n",ofnav15::statusName(s));
            return 1;
        }
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
    std::printf("\"nis\":%.17g,\"ci_weight\":%.17g}\n",f.lastNis(),f.lastCiWeight());
    return 0;
}
