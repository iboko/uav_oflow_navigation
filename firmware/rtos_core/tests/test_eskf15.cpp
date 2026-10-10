#include "ofnav/eskf15.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

namespace {

using ofnav15::Status;
using ofnav15::Visual;
using ofnav15::Imu;
using ofnav15::Filter;
using ofnav15::FixedLag;

bool near(double a, double b, double eps=1.e-7) {
    return std::abs(a-b)<eps;
}
bool finiteCov(const ofnav15::Matrix15 &p) {
    for (int i=0;i<15;++i) {
        if (!(p[15*i+i]>0.)) {return false;}
        for(int j=0;j<15;++j) {
            if (!std::isfinite(p[15*i+j]) ||
                !near(p[15*i+j],p[15*j+i],1.e-9)) {return false;}
        }
    }
    return true;
}
Imu sample(std::uint64_t t, double ax=0., double wz=0.) {
    return Imu{t,{0.,0.,wz},{ax,0.,-9.80665}};
}
Visual visual(std::uint64_t t, double vn, double ve=0.,
              double variance=.04) {
    return Visual{t,{vn,ve},{variance,0.,0.,variance}};
}
void test_static_frd_ned_gravity_and_spd_15() {
    Filter f{};
    for(int i=0;i<=100;++i) {
        assert(f.predict(sample(1000000U+static_cast<std::uint64_t>(10000*i)))==
               Status::InertialOnly);
    }
    assert(f.timestamp()==2000000U);
    for(double x:f.position()) {assert(std::abs(x)<1.e-8);}
    for(double x:f.velocity()) {assert(std::abs(x)<1.e-8);}
    assert(finiteCov(f.covariance()));
    assert(near(f.quaternion()[0],1.));
}
void test_constant_acceleration_and_yaw_rotation() {
    Filter f{};
    for(int i=0;i<=100;++i) {
        assert(f.predict(sample(1000000U+static_cast<std::uint64_t>(10000*i),1.))==
               Status::InertialOnly);
    }
    assert(near(f.velocity()[0],1.,1.e-9));
    assert(near(f.position()[0],.5,1.e-9));
    assert(finiteCov(f.covariance()));
    Filter yaw{};
    for(int i=0;i<=100;++i) {
        assert(yaw.predict(sample(1000000U+static_cast<std::uint64_t>(10000*i),
                                  0.,.5))==Status::InertialOnly);
    }
    const double angle=2.*std::atan2(yaw.quaternion()[3],yaw.quaternion()[0]);
    assert(near(angle,.5,1.e-7));
    assert(finiteCov(yaw.covariance()));
}
void test_visual_joseph_and_rejection_are_safe() {
    Filter f{};
    assert(f.predict(sample(1000000U))==Status::InertialOnly);
    assert(f.predict(sample(1010000U))==Status::InertialOnly);
    const double before=f.covariance()[15*3+3];
    assert(f.updateVelocity(visual(1010000U,.1))==Status::VisualCorrected);
    assert(f.accepted()==1U);
    assert(f.covariance()[15*3+3]<before);
    assert(finiteCov(f.covariance()));
    assert(f.updateVelocity(visual(1010000U,.1))==Status::RepeatedVisual);
    assert(f.predict(sample(1020000U))==Status::InertialOnly);
    const double predicted_velocity=f.velocity()[0];
    const auto predicted_covariance=f.covariance();
    assert(f.updateVelocity(visual(1020000U,500.))==Status::VisualOutlier);
    assert(f.rejected()==1U);
    assert(near(f.velocity()[0],predicted_velocity,1.e-10));
    assert(f.covariance()==predicted_covariance);
    assert(finiteCov(f.covariance()));
}
void test_invalid_covariances_and_imu_gap_fail_closed() {
    Filter f{};
    assert(f.predict(sample(1000000U))==Status::InertialOnly);
    assert(f.predict(sample(1010000U))==Status::InertialOnly);
    Visual invalid=visual(1010000U,0.);
    invalid.covariance_ne_m2_s2={1.,2.,0.,1.};
    assert(f.updateVelocity(invalid)==Status::InvalidVisual);
    assert(f.accepted()==0U);
    assert(f.predict(sample(1020000U))==Status::InertialOnly);
    assert(f.predict(sample(1030000U))==Status::InertialOnly);
    assert(f.predict(sample(1200000U))==Status::ImuGap);
    assert(!f.healthy());
    assert(f.predict(sample(1210000U))==Status::ImuGap);
}
void test_delayed_measurements_replay_transactionally_same_as_time_sorted() {
    FixedLag chronological({},500000U);
    FixedLag delayed({},500000U);
    for(int i=0;i<=45;++i) {
        const auto s=sample(1000000U+static_cast<std::uint64_t>(10000*i),.12);
        assert(chronological.pushImu(s)==Status::InertialOnly);
        assert(delayed.pushImu(s)==Status::InertialOnly);
    }
    assert(chronological.pushDelayedVelocity(visual(1200000U,.025))==
           Status::VisualCorrected);
    assert(chronological.pushDelayedVelocity(visual(1305000U,.042))==
           Status::VisualCorrected);
    assert(delayed.pushDelayedVelocity(visual(1305000U,.042))==
           Status::VisualCorrected);
    assert(delayed.pushDelayedVelocity(visual(1200000U,.025))==
           Status::VisualCorrected);
    assert(chronological.current().timestamp()==1450000U);
    assert(delayed.current().timestamp()==1450000U);
    assert(chronological.storedVisual()==2);
    assert(delayed.storedVisual()==2);
    for(int i=0;i<3;++i) {
        assert(near(chronological.current().position()[i],
                    delayed.current().position()[i],1.e-10));
        assert(near(chronological.current().velocity()[i],
                    delayed.current().velocity()[i],1.e-10));
    }
    for(std::size_t i=0;i<225;++i) {
        assert(near(chronological.current().covariance()[i],
                    delayed.current().covariance()[i],1.e-10));
    }
    assert(finiteCov(delayed.current().covariance()));
}
void test_bad_delayed_visual_not_accepted_or_mutate_state() {
    FixedLag lag({},350000U);
    for(int i=0;i<=45;++i) {
        assert(lag.pushImu(sample(1000000U+
                     static_cast<std::uint64_t>(10000*i)))==Status::InertialOnly);
    }
    const auto old=lag.current().covariance();
    assert(lag.pushDelayedVelocity(visual(1010000U,0.))==
           Status::DelayExceeded);
    assert(lag.pushDelayedVelocity(visual(1600000U,0.))==
           Status::FutureVisual);
    auto m=visual(1400000U,0.);
    m.covariance_ne_m2_s2={0.,0.,0.,0.};
    assert(lag.pushDelayedVelocity(m)==Status::InvalidVisual);
    assert(lag.storedVisual()==0);
    assert(lag.current().covariance()==old);
    assert(lag.pushDelayedVelocity(visual(1400000U,0.))==
           Status::VisualCorrected);
    const auto accepted=lag.current().covariance();
    assert(lag.pushDelayedVelocity(visual(1400000U,0.))==
           Status::DuplicateVisual);
    assert(lag.current().covariance()==accepted);
}
void test_fixed_memory_history_and_checkpoint_compaction() {
    FixedLag lag({},300000U);
    for(int i=0;i<=120;++i) {
        assert(lag.pushImu(sample(1000000U+
               static_cast<std::uint64_t>(10000*i)))==Status::InertialOnly);
        assert(lag.storedImu()<=FixedLag::kImuCapacity);
    }
    assert(lag.storedImu()<=35U);
    assert(lag.pushDelayedVelocity(visual(2180000U,0.))==
           Status::VisualCorrected);
    assert(lag.current().timestamp()==2200000U);
    assert(!lag.failed());
}
void test_capacity_fails_explicitly_without_heap_growth() {
    FixedLag lag({},5000000U);
    for(int i=0;i<=64;++i) {
        assert(lag.pushImu(sample(1000000U+
               static_cast<std::uint64_t>(10000*i)))==Status::InertialOnly);
    }
    assert(lag.storedImu()==FixedLag::kImuCapacity);
    assert(lag.pushImu(sample(1650000U))==Status::BufferLimit);
    assert(lag.current().timestamp()==1640000U);
    assert(lag.storedImu()==FixedLag::kImuCapacity);
}
} // namespace

int main() {
    test_static_frd_ned_gravity_and_spd_15();
    test_constant_acceleration_and_yaw_rotation();
    test_visual_joseph_and_rejection_are_safe();
    test_invalid_covariances_and_imu_gap_fail_closed();
    test_delayed_measurements_replay_transactionally_same_as_time_sorted();
    test_bad_delayed_visual_not_accepted_or_mutate_state();
    test_fixed_memory_history_and_checkpoint_compaction();
    test_capacity_fails_explicitly_without_heap_growth();
    std::cout<<"C++ 15-state ESKF and fixed-lag regression PASS\n";
    return 0;
}
