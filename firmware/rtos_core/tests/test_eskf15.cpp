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
void test_nonzero_calibrated_initial_attitude_and_imu_biases() {
    ofnav15::Parameters config{};
    const double roll=.2;
    config.initial_quaternion_body_to_ned={
        std::cos(.5*roll),std::sin(.5*roll),0.,0.
    };
    config.initial_gyro_bias_rad_s={0.,0.,.02};
    config.initial_accel_bias_m_s2={.1,0.,0.};
    Filter f(config);
    for(int i=0;i<=100;++i) {
        const double g=9.80665;
        const Imu m{
            1000000U+static_cast<std::uint64_t>(i*10000),
            {0.,0.,.02},
            {.1,-g*std::sin(roll),-g*std::cos(roll)}
        };
        assert(f.predict(m)==Status::InertialOnly);
    }
    for(double x:f.position()) {assert(std::abs(x)<1.e-7);}
    for(double x:f.velocity()) {assert(std::abs(x)<1.e-7);}
    assert(near(f.quaternion()[1],std::sin(.5*roll),1.e-10));
    assert(finiteCov(f.covariance()));
    ofnav15::Parameters invalid=config;
    invalid.initial_quaternion_body_to_ned={0.,0.,0.,0.};
    Filter rejected(invalid);
    assert(!rejected.healthy());
    assert(rejected.predict(sample(1000000U))==Status::NumericalFailure);
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
ofnav15::Measurement meas(std::uint64_t t, ofnav15::MeasurementKind kind,
                         double a, double b, double var, double ci=1.) {
    ofnav15::Measurement m{};
    m.time_us=t; m.kind=kind; m.value={a,b};
    m.covariance={var,0.,0.,kind==ofnav15::MeasurementKind::Yaw?0.:var};
    m.ci_weight=ci;
    return m;
}
double yawOf(const ofnav15::Quat &q) {
    return std::atan2(2.*(q[0]*q[3]+q[1]*q[2]),1.-2.*(q[2]*q[2]+q[3]*q[3]));
}
Imu moving(std::uint64_t t, int i) {
    const double s=static_cast<double>(i)*.01;
    return Imu{t,{.004*std::sin(.9*s),-.003*std::cos(.7*s),.03*std::cos(.5*s)},
               {.3*std::sin(.6*s),.2*std::cos(.4*s),-9.80665+.05*std::sin(.8*s)}};
}
void test_snapshot_replay_equals_sequential_filter() {
    // Exposures coincide with IMU times: a plain sequential filter performs
    // exactly the same operations as the replay -> results must agree.
    using K=ofnav15::MeasurementKind;
    Filter reference{};
    FixedLag lag({},630000U);
    ofnav15::Workspace ws{};
    struct Frame {std::uint64_t t; ofnav15::Measurement m[4];};
    Frame frames[6]{};
    for(int k=0;k<6;++k) {
        const std::uint64_t t=1000000U+static_cast<std::uint64_t>(10000*(20+12*k));
        frames[k].t=t;
        frames[k].m[0]=meas(t,K::VelocityNe,.01*k,-.02,.04);
        frames[k].m[1]=meas(t,K::BodyVelocityXy,.012*k,-.018,.05);
        frames[k].m[2]=meas(t,K::Yaw,.001*k,0.,.01);
        frames[k].m[3]=meas(t,K::PositionNe,.03*k,-.01*k,4.,0.);
    }
    int delivered=0;
    for(int i=0;i<=110;++i) {
        const std::uint64_t t=1000000U+static_cast<std::uint64_t>(10000*i);
        const Imu m=moving(t,i);
        assert(reference.predict(m,ws)==Status::InertialOnly);
        for(const auto &fr:frames) {
            if (fr.t==t) {for(const auto &e:fr.m) {
                assert(ofnav15::measurementProcessed(reference.update(e,ws)));
            }}
        }
        assert(lag.pushImu(m)==Status::InertialOnly);
        // Deliver frames late and pairwise out of order.
        if (i==60) {
            Status st[4]{};
            assert(ofnav15::measurementProcessed(lag.pushDelayedFrame(frames[1].m,4,st)));
            assert(ofnav15::measurementProcessed(lag.pushDelayedFrame(frames[0].m,4,st)));
            delivered+=2;
        }
        if (i==95) {
            Status st[4]{};
            for(int k:{3,2,4}) {
                assert(ofnav15::measurementProcessed(lag.pushDelayedFrame(frames[k].m,4,st)));
                for(const auto x:st) {assert(ofnav15::measurementProcessed(x));}
                ++delivered;
            }
        }
        if (i==105) {
            Status st[4]{};
            assert(ofnav15::measurementProcessed(lag.pushDelayedFrame(frames[5].m,4,st)));
            ++delivered;
            assert(lag.lastReplayLength()<lag.storedImu());
        }
    }
    assert(delivered==6);
    const auto &f=lag.current();
    for(int i=0;i<3;++i) {
        assert(near(f.position()[i],reference.position()[i],1.e-9));
        assert(near(f.velocity()[i],reference.velocity()[i],1.e-9));
    }
    for(int i=0;i<4;++i) {assert(near(f.quaternion()[i],reference.quaternion()[i],1.e-12));}
    for(std::size_t i=0;i<225;++i) {assert(near(f.covariance()[i],reference.covariance()[i],1.e-9));}
    assert(f.accepted()==reference.accepted());
}
void test_frame_rejects_duplicate_kind_and_keeps_state() {
    using K=ofnav15::MeasurementKind;
    FixedLag lag({},400000U);
    for(int i=0;i<=30;++i) {
        assert(lag.pushImu(sample(1000000U+static_cast<std::uint64_t>(10000*i)))==Status::InertialOnly);
    }
    const auto before=lag.current().covariance();
    ofnav15::Measurement twice[2]={meas(1200000U,K::VelocityNe,0.,0.,.04),
                                   meas(1200000U,K::VelocityNe,0.,0.,.04)};
    Status st[2]{};
    (void)lag.pushDelayedFrame(twice,2,st);
    assert(st[0]==Status::VisualCorrected);
    assert(st[1]==Status::InvalidVisual);
    ofnav15::Measurement bad=meas(1210000U,K::VelocityNe,0.,0.,.04,.5);
    assert(lag.pushDelayed(bad)==Status::InvalidVisual);  // CI only for position
    assert(lag.storedVisual()==1);
    assert(!(lag.current().covariance()==before));
}
void test_body_velocity_makes_yaw_observable() {
    using K=ofnav15::MeasurementKind;
    ofnav15::Parameters cfg{};
    cfg.initial_sigma[8]=.5;
    Filter f(cfg);
    ofnav15::Workspace ws{};
    const double true_yaw=.2;
    // True motion: 1 m/s north. Body-frame velocity seen by the camera
    // depends on the true heading; NE velocity from another aid.
    for(int i=0;i<=200;++i) {
        const std::uint64_t t=1000000U+static_cast<std::uint64_t>(10000*i);
        assert(f.predict(sample(t),ws)==Status::InertialOnly);
        if (i%10==0) {
            assert(ofnav15::measurementProcessed(f.update(meas(t,K::VelocityNe,1.,0.,.0004),ws)));
            assert(ofnav15::measurementProcessed(f.update(
                meas(t,K::BodyVelocityXy,std::cos(true_yaw),-std::sin(true_yaw),.0004),ws)));
        }
    }
    assert(std::abs(yawOf(f.quaternion())-true_yaw)<.02);
    assert(f.covariance()[15*8+8]<.01);
    assert(finiteCov(f.covariance()));
}
void test_yaw_update_with_tilt_and_wraparound() {
    using K=ofnav15::MeasurementKind;
    ofnav15::Parameters cfg{};
    const double roll=.25, pitch=-.15;
    // q = Rz(0) Ry(pitch) Rx(roll)
    const double cr=std::cos(.5*roll),sr=std::sin(.5*roll),cp=std::cos(.5*pitch),sp=std::sin(.5*pitch);
    cfg.initial_quaternion_body_to_ned={cr*cp,sr*cp,cr*sp,-sr*sp};
    cfg.initial_sigma[8]=1.;
    Filter f(cfg);
    ofnav15::Workspace ws{};
    assert(f.predict(Imu{1000000U,{},{0.,0.,-9.80665}},ws)==Status::InertialOnly);
    // One linearized step: residual error is second order in the step
    // (3.9e-4 rad for 0.1 rad with this tilt); a second step reduces it.
    assert(f.update(meas(1000000U,K::Yaw,.1,0.,1.e-6),ws)==Status::YawCorrected);
    const double first=std::abs(yawOf(f.quaternion())-.1);
    assert(first<1.e-3);
    assert(f.predict(Imu{1010000U,{},{0.,0.,-9.80665}},ws)==Status::InertialOnly);
    assert(f.update(meas(1010000U,K::Yaw,.1,0.,1.e-6),ws)==Status::YawCorrected);
    assert(std::abs(yawOf(f.quaternion())-.1)<.6*first);
    Filter w(cfg);
    assert(w.predict(Imu{1000000U,{},{0.,0.,-9.80665}},ws)==Status::InertialOnly);
    // Nominal yaw 0; measurement 2*pi - 0.05 must act as -0.05.
    assert(w.update(meas(1000000U,K::Yaw,6.283185307179586-.05,0.,1.e-6),ws)==Status::YawCorrected);
    assert(std::abs(yawOf(w.quaternion())+.05)<1.e-3);
}
void test_position_ci_update_is_conservative() {
    using K=ofnav15::MeasurementKind;
    ofnav15::Workspace ws{};
    Filter independent{}, ci{};
    for(Filter *f:{&independent,&ci}) {
        assert(f->predict(sample(1000000U),ws)==Status::InertialOnly);
        assert(f->predict(sample(1010000U),ws)==Status::InertialOnly);
    }
    const double prior=independent.covariance()[0];
    assert(independent.update(meas(1010000U,K::PositionNe,1.,-1.,9.),ws)==Status::PositionCorrected);
    assert(ci.update(meas(1010000U,K::PositionNe,1.,-1.,9.,0.),ws)==Status::PositionCorrected);
    assert(ci.lastCiWeight()>0. && ci.lastCiWeight()<1.);
    assert(near(independent.lastCiWeight(),1.));
    assert(ci.covariance()[0]>independent.covariance()[0]);
    assert(ci.covariance()[0]<prior);
    assert(finiteCov(ci.covariance()));
    Filter far{};
    assert(far.predict(sample(1000000U),ws)==Status::InertialOnly);
    assert(far.update(meas(1000000U,K::PositionNe,500.,0.,1.),ws)==Status::PositionOutlier);
}
void test_static_alignment_roll_pitch_gyro_bias() {
    const double roll=.1, pitch=-.05, g=9.80665;
    const ofnav15::Vec3 bias{.01,-.02,.005};
    Imu window[200]{};
    for(int i=0;i<200;++i) {
        const double n=.01*std::sin(1.7*i);
        window[i]=Imu{1000000U+static_cast<std::uint64_t>(5000*i),
            {bias[0]+.001*std::sin(2.1*i),bias[1],bias[2]},
            {g*std::sin(pitch)+n,-g*std::sin(roll)*std::cos(pitch),
             -g*std::cos(roll)*std::cos(pitch)-n}};
    }
    ofnav15::Parameters p{};
    assert(ofnav15::staticAlignment(window,200,.7,.05,{},p)==ofnav15::AlignmentStatus::Aligned);
    const auto &q=p.initial_quaternion_body_to_ned;
    const double r=std::atan2(2.*(q[0]*q[1]+q[2]*q[3]),1.-2.*(q[1]*q[1]+q[2]*q[2]));
    const double pt=std::asin(2.*(q[0]*q[2]-q[3]*q[1]));
    assert(std::abs(r-roll)<1.e-3 && std::abs(pt-pitch)<1.e-3);
    assert(std::abs(yawOf(q)-.7)<1.e-9);
    for(int k=0;k<3;++k) {assert(std::abs(p.initial_gyro_bias_rad_s[k]-bias[k])<1.e-4);}
    assert(near(p.initial_sigma[8],.05));
    Filter f(p);
    assert(f.healthy());
    for(int i=0;i<200;++i) {
        Imu m=window[i]; m.time_us=2000000U+static_cast<std::uint64_t>(5000*i);
        assert(f.predict(m)==Status::InertialOnly);
    }
    for(double v:f.velocity()) {assert(std::abs(v)<.02);}
    window[50].specific_force_m_s2[0]+=5.;
    assert(ofnav15::staticAlignment(window,200,0.,.1,{},p)==ofnav15::AlignmentStatus::NotStationary);
    assert(ofnav15::staticAlignment(window,10,0.,.1,{},p)==ofnav15::AlignmentStatus::TooFewSamples);
}

} // namespace

int main() {
    test_static_frd_ned_gravity_and_spd_15();
    test_constant_acceleration_and_yaw_rotation();
    test_nonzero_calibrated_initial_attitude_and_imu_biases();
    test_visual_joseph_and_rejection_are_safe();
    test_invalid_covariances_and_imu_gap_fail_closed();
    test_delayed_measurements_replay_transactionally_same_as_time_sorted();
    test_bad_delayed_visual_not_accepted_or_mutate_state();
    test_fixed_memory_history_and_checkpoint_compaction();
    test_capacity_fails_explicitly_without_heap_growth();
    test_snapshot_replay_equals_sequential_filter();
    test_frame_rejects_duplicate_kind_and_keeps_state();
    test_body_velocity_makes_yaw_observable();
    test_yaw_update_with_tilt_and_wraparound();
    test_position_ci_update_is_conservative();
    test_static_alignment_roll_pitch_gyro_bias();
    std::cout<<"C++ 15-state ESKF and fixed-lag regression PASS\n";
    return 0;
}
