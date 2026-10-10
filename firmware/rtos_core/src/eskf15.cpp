#include "ofnav/eskf15.hpp"

#include <algorithm>
#include <cmath>

namespace ofnav15 {
namespace {

constexpr double kGravity = 9.80665;
using Mat3 = std::array<double, 9>;
using Mat2 = std::array<double, 4>;

double &at(Matrix15 &a, int i, int j) noexcept { return a[15 * i + j]; }
double at(const Matrix15 &a, int i, int j) noexcept { return a[15 * i + j]; }

Matrix15 identity() noexcept {
    Matrix15 r{};
    for (int i = 0; i < 15; ++i) { at(r, i, i) = 1.; }
    return r;
}
Matrix15 transpose(const Matrix15 &a) noexcept {
    Matrix15 r{};
    for (int i = 0; i < 15; ++i) {
        for (int j = 0; j < 15; ++j) { at(r, i, j) = at(a, j, i); }
    }
    return r;
}
Matrix15 multiply(const Matrix15 &a, const Matrix15 &b) noexcept {
    Matrix15 r{};
    for (int i = 0; i < 15; ++i) {
        for (int k = 0; k < 15; ++k) {
            const double v = at(a, i, k);
            for (int j = 0; j < 15; ++j) { at(r, i, j) += v * at(b, k, j); }
        }
    }
    return r;
}
void symmetric(Matrix15 &p) noexcept {
    for (int i = 0; i < 15; ++i) {
        for (int j = 0; j < i; ++j) {
            const double v = .5 * (at(p, i, j) + at(p, j, i));
            at(p, i, j) = v;
            at(p, j, i) = v;
        }
    }
}
bool positiveDefinite(const Matrix15 &p) noexcept {
    double l[15][15]{};
    for (int i = 0; i < 15; ++i) {
        for (int j = 0; j <= i; ++j) {
            double x = at(p, i, j);
            if (!std::isfinite(x)) { return false; }
            for (int k = 0; k < j; ++k) { x -= l[i][k] * l[j][k]; }
            if (i == j) {
                if (!(x > 1.e-20)) { return false; }
                l[i][j] = std::sqrt(x);
            } else {
                l[i][j] = x / l[j][j];
            }
        }
    }
    return true;
}
bool finite3(const Vec3 &v) noexcept {
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}
bool finiteImu(const Imu &v) noexcept {
    return v.time_us > 0 && finite3(v.gyro_rad_s) && finite3(v.specific_force_m_s2);
}
double length(const Vec3 &v) noexcept {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}
Vec3 minus(const Vec3 &a, const Vec3 &b) noexcept {
    return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};
}
Vec3 midpoint(const Vec3 &a, const Vec3 &b) noexcept {
    return { .5*(a[0]+b[0]), .5*(a[1]+b[1]), .5*(a[2]+b[2]) };
}
Quat product(const Quat &a, const Quat &b) noexcept {
    return {a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
            a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
            a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
            a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]};
}
Quat unit(Quat q) noexcept {
    double ss = 0;
    for (double x:q) { ss += x*x; }
    if (!std::isfinite(ss) || ss < .64 || ss > 1.44) {
        return {std::numeric_limits<double>::quiet_NaN(),0.,0.,0.};
    }
    for (double &x:q) { x /= std::sqrt(ss); }
    return q;
}
Quat expQuat(const Vec3 &angle) noexcept {
    const double theta = length(angle);
    const double s = theta < 1.e-8 ? .5-theta*theta/48. : std::sin(.5*theta)/theta;
    return {std::cos(.5*theta),s*angle[0],s*angle[1],s*angle[2]};
}
Mat3 matrixFromQuat(const Quat &q) noexcept {
    const double w=q[0], x=q[1], y=q[2], z=q[3];
    return {1.-2.*(y*y+z*z), 2.*(x*y-w*z),2.*(x*z+w*y),
            2.*(x*y+w*z),1.-2.*(x*x+z*z),2.*(y*z-w*x),
            2.*(x*z-w*y),2.*(y*z+w*x),1.-2.*(x*x+y*y)};
}
Vec3 matVec(const Mat3 &a, const Vec3 &v) noexcept {
    return {a[0]*v[0]+a[1]*v[1]+a[2]*v[2],
            a[3]*v[0]+a[4]*v[1]+a[5]*v[2],
            a[6]*v[0]+a[7]*v[1]+a[8]*v[2]};
}
Mat3 skew(const Vec3 &v) noexcept {
    return {0.,-v[2],v[1],v[2],0.,-v[0],-v[1],v[0],0.};
}
Matrix15 jacobian(const Mat3 &r, const Vec3 &omega, const Vec3 &force) noexcept {
    Matrix15 f{};
    const Mat3 s = skew(force);
    const Mat3 sw = skew(omega);
    for (int i = 0; i < 3; ++i) {
        at(f,i,3+i)=1.;
        at(f,6+i,9+i)=-1.;
        for (int j=0;j<3;++j) {
            at(f,3+i,12+j)=-r[3*i+j];
            at(f,6+i,6+j)=-sw[3*i+j];
            double sum=0.;
            for (int k=0;k<3;++k) { sum += r[3*i+k]*s[3*k+j]; }
            at(f,3+i,6+j)=-sum;
        }
    }
    return f;
}
Matrix15 transition(const Matrix15 &f, double dt) noexcept {
    Matrix15 a=f;
    for (double &x:a) { x *= dt; }
    const Matrix15 a2=multiply(a,a);
    const Matrix15 a3=multiply(a2,a);
    const Matrix15 a4=multiply(a3,a);
    Matrix15 phi=identity();
    for (std::size_t i=0;i<phi.size();++i) {
        phi[i]+=a[i]+.5*a2[i]+a3[i]/6.+a4[i]/24.;
    }
    return phi;
}
Matrix15 processCov(const Matrix15 &f, const Mat3 &r, double dt,
                    const Parameters &cfg) noexcept {
    std::array<double, 180> b{}; // row-major 15 x 12
    const double sigma[4] = {
        cfg.accel_noise, cfg.gyro_noise,
        cfg.gyro_bias_random_walk, cfg.accel_bias_random_walk
    };
    for (int i=0;i<3;++i) {
        for(int j=0;j<3;++j) { b[(3+i)*12+j]=-r[3*i+j]*sigma[0]; }
        b[(6+i)*12+3+i]=-sigma[1];
        b[(9+i)*12+6+i]=sigma[2];
        b[(12+i)*12+9+i]=sigma[3];
    }
    Matrix15 q{};
    constexpr double nodes[3] = {0.,-.7745966692414834,.7745966692414834};
    constexpr double weights[3] = {8./9.,5./9.,5./9.};
    for (int n=0;n<3;++n) {
        const double tau=.5*dt*(1.+nodes[n]);
        const double weight=.5*dt*weights[n];
        const Matrix15 phi=transition(f,tau);
        double z[15][12]{};
        for(int i=0;i<15;++i) {
            for(int k=0;k<15;++k) {
                for(int col=0;col<12;++col) {
                    z[i][col]+=at(phi,i,k)*b[k*12+col];
                }
            }
        }
        for(int i=0;i<15;++i) {
            for(int j=0;j<15;++j) {
                double value=0.;
                for(int col=0;col<12;++col) {value+=z[i][col]*z[j][col];}
                at(q,i,j)+=weight*value;
            }
        }
    }
    symmetric(q);
    return q;
}
bool validParameters(const Parameters &c) noexcept {
    if (!(c.max_dt_s>=.001 && c.max_dt_s<=.05 &&
          c.nis_gate>0 && c.max_gyro_bias>0 && c.max_accel_bias>0)) {
        return false;
    }
    for (double x:c.initial_sigma) {
        if (!std::isfinite(x) || x<=0.) {return false;}
    }
    for (double x : {c.gyro_noise,c.accel_noise,
                     c.gyro_bias_random_walk,c.accel_bias_random_walk}) {
        if (!std::isfinite(x) || !(x>0 && x<100.)) {return false;}
    }
    return true;
}
bool visualValid(const Visual &m) noexcept {
    if (m.time_us==0 || !std::isfinite(m.velocity_ne_m_s[0]) ||
        !std::isfinite(m.velocity_ne_m_s[1])) {return false;}
    const auto &r=m.covariance_ne_m2_s2;
    return std::isfinite(r[0]) && std::isfinite(r[1]) &&
        std::isfinite(r[2]) && std::isfinite(r[3]) &&
        std::abs(r[1]-r[2])<=1.e-10 && r[0]>0 && r[3]>0 &&
        r[0]*r[3]-r[1]*r[2]>1.e-16;
}
Imu interpolated(const Imu &a, const Imu &b, std::uint64_t t) noexcept {
    const double alpha=static_cast<double>(t-a.time_us) /
                       static_cast<double>(b.time_us-a.time_us);
    Imu v{}; v.time_us=t;
    for(int i=0;i<3;++i) {
        v.gyro_rad_s[i]=(1.-alpha)*a.gyro_rad_s[i]+alpha*b.gyro_rad_s[i];
        v.specific_force_m_s2[i]=(1.-alpha)*a.specific_force_m_s2[i]+
                                  alpha*b.specific_force_m_s2[i];
    }
    return v;
}
} // namespace

const char *statusName(Status s) noexcept {
    switch(s) {
        case Status::WaitingForImu:return "WAITING_FOR_IMU";
        case Status::InertialOnly:return "INERTIAL_ONLY";
        case Status::VisualCorrected:return "VISUAL_CORRECTED";
        case Status::VisualOutlier:return "VISUAL_OUTLIER_REJECTED";
        case Status::ImuGap:return "IMU_INVALID_OR_GAP";
        case Status::NumericalFailure:return "NUMERICAL_FAILURE";
        case Status::InvalidVisual:return "INVALID_VISUAL_MEASUREMENT";
        case Status::RepeatedVisual:return "REPEATED_VISUAL";
        case Status::UnalignedVisual:return "UNALIGNED_VISUAL_TIMESTAMP";
        case Status::BiasLimit:return "BIAS_LIMIT_EXCEEDED";
        case Status::PosteriorInvalid:return "POSTERIOR_INVALID";
        case Status::NoHistory:return "NO_IMU_REFERENCE";
        case Status::DelayExceeded:return "VISUAL_DELAY_EXCEEDED";
        case Status::FutureVisual:return "FUTURE_VISUAL_MEASUREMENT";
        case Status::DuplicateVisual:return "REPEATED_VISUAL_MEASUREMENT";
        case Status::BufferLimit:return "BUFFER_LIMIT";
        case Status::ReplayFailed:return "REPLAY_FAILED";
    }
    return "UNKNOWN";
}
Filter::Filter(const Parameters &c) noexcept : config_(c),valid_config_(validParameters(c)) {
    for(int i=0;i<15;++i) {at(p_cov_,i,i)=c.initial_sigma[i]*c.initial_sigma[i];}
    if (!valid_config_) {status_=Status::NumericalFailure;}
}
bool Filter::healthy() const noexcept {
    return status_!=Status::ImuGap && status_!=Status::NumericalFailure &&
           status_!=Status::BiasLimit && status_!=Status::PosteriorInvalid;
}
Status Filter::predict(const Imu &m) noexcept {
    if (!healthy()) {return status_;}
    if (!finiteImu(m) || (has_imu_ && (m.time_us<=last_.time_us ||
        static_cast<double>(m.time_us-last_.time_us)*1.e-6>config_.max_dt_s))) {
        status_=Status::ImuGap;
        return status_;
    }
    if (!has_imu_) {
        last_=m;has_imu_=true;status_=Status::InertialOnly;
        return status_;
    }
    const double dt=static_cast<double>(m.time_us-last_.time_us)*1.e-6;
    const Vec3 omega=minus(midpoint(m.gyro_rad_s,last_.gyro_rad_s),bg_);
    const Vec3 force=minus(midpoint(m.specific_force_m_s2,last_.specific_force_m_s2),ba_);
    const Vec3 halfangle={.5*dt*omega[0],.5*dt*omega[1],.5*dt*omega[2]};
    const Quat qmid=unit(product(q_,expQuat(halfangle)));
    const Mat3 rmid=matrixFromQuat(qmid);
    Vec3 a=matVec(rmid,force);a[2]+=kGravity;
    const Matrix15 f=jacobian(rmid,omega,force);
    const Matrix15 phi=transition(f,dt);
    const Matrix15 qprocess=processCov(f,rmid,dt,config_);
    Matrix15 pnew=multiply(multiply(phi,p_cov_),transpose(phi));
    for(std::size_t i=0;i<pnew.size();++i) {pnew[i]+=qprocess[i];}
    symmetric(pnew);
    const Vec3 angle={dt*omega[0],dt*omega[1],dt*omega[2]};
    const Quat qnext=unit(product(q_,expQuat(angle)));
    if (!positiveDefinite(pnew) || !std::isfinite(qnext[0]) ||
        !finite3(a)) {
        status_=Status::NumericalFailure;
        return status_;
    }
    for(int i=0;i<3;++i) {
        p_[i]+=dt*v_[i]+.5*dt*dt*a[i];
        v_[i]+=dt*a[i];
    }
    p_cov_=pnew;q_=qnext;last_=m;status_=Status::InertialOnly;
    return status_;
}
Status Filter::updateVelocity(const Visual &m) noexcept {
    if (!healthy()) {return status_;}
    if (!has_imu_) {return Status::NoHistory;}
    if (m.time_us!=last_.time_us) {return Status::UnalignedVisual;}
    if (m.time_us<=last_visual_us_) {return Status::RepeatedVisual;}
    if (!visualValid(m)) {++rejected_;return Status::InvalidVisual;}
    const double y0=m.velocity_ne_m_s[0]-v_[0];
    const double y1=m.velocity_ne_m_s[1]-v_[1];
    const auto &r=m.covariance_ne_m2_s2;
    const double s00=at(p_cov_,3,3)+r[0];
    const double s01=at(p_cov_,3,4)+r[1];
    const double s10=at(p_cov_,4,3)+r[2];
    const double s11=at(p_cov_,4,4)+r[3];
    const double det=s00*s11-s01*s10;
    if (!(s00>0 && s11>0 && det>1.e-16) ||
        !std::isfinite(det)) {++rejected_;return Status::InvalidVisual;}
    const double inv00=s11/det,inv01=-s01/det,inv10=-s10/det,inv11=s00/det;
    nis_=y0*(inv00*y0+inv01*y1)+y1*(inv10*y0+inv11*y1);
    last_visual_us_=m.time_us;
    if (!std::isfinite(nis_)) {++rejected_;return Status::InvalidVisual;}
    if (nis_>config_.nis_gate) {
        ++rejected_;status_=Status::VisualOutlier;
        return status_;
    }
    double k[15][2]{};
    double dx[15]{};
    for(int i=0;i<15;++i) {
        k[i][0]=at(p_cov_,i,3)*inv00+at(p_cov_,i,4)*inv10;
        k[i][1]=at(p_cov_,i,3)*inv01+at(p_cov_,i,4)*inv11;
        dx[i]=k[i][0]*y0+k[i][1]*y1;
    }
    Vec3 bg_new=bg_,ba_new=ba_;
    for(int i=0;i<3;++i) {
        bg_new[i]+=dx[9+i];
        ba_new[i]+=dx[12+i];
    }
    if (length(bg_new)>config_.max_gyro_bias ||
        length(ba_new)>config_.max_accel_bias) {
        status_=Status::BiasLimit;
        return status_;
    }
    // Joseph form (I-KH)P(I-KH)^T+KRK^T, algebraically
    // P-KHP-PH^TK^T+K(HPH^T+R)K^T.
    Matrix15 pnew=p_cov_;
    for(int i=0;i<15;++i) {
        for(int j=0;j<15;++j) {
            at(pnew,i,j) -= k[i][0]*at(p_cov_,3,j)+
                             k[i][1]*at(p_cov_,4,j);
            at(pnew,i,j) -= at(p_cov_,i,3)*k[j][0]+
                             at(p_cov_,i,4)*k[j][1];
            at(pnew,i,j) += k[i][0]*(s00*k[j][0]+s01*k[j][1])+
                             k[i][1]*(s10*k[j][0]+s11*k[j][1]);
        }
    }
    const Vec3 dth={dx[6],dx[7],dx[8]};
    const Mat3 st=skew(dth);
    Matrix15 reset=identity();
    for(int i=0;i<3;++i) {
        for(int j=0;j<3;++j) {
            at(reset,6+i,6+j)-=.5*st[3*i+j];
        }
    }
    pnew=multiply(multiply(reset,pnew),transpose(reset));
    symmetric(pnew);
    const Quat qnew=unit(product(q_,expQuat(dth)));
    if (!positiveDefinite(pnew) || !std::isfinite(qnew[0])) {
        status_=Status::PosteriorInvalid;
        return status_;
    }
    p_cov_=pnew;q_=qnew;bg_=bg_new;ba_=ba_new;
    for(int i=0;i<3;++i) {p_[i]+=dx[i];v_[i]+=dx[3+i];}
    ++accepted_;status_=Status::VisualCorrected;
    return status_;
}

FixedLag::FixedLag(const Parameters &params, std::uint64_t lag) noexcept :
    parameters_(params),max_lag_us_(lag),checkpoint_(params),current_(params) {
    if (lag==0 || lag>5000000U || !current_.healthy()) {failed_=true;}
}
Status FixedLag::pushImu(const Imu &sample) noexcept {
    if (failed_) {return Status::ImuGap;}
    if (initialized_ && count_>=kImuCapacity) {
        trim();
        if (count_>=kImuCapacity) {return Status::BufferLimit;}
    }
    if (!finiteImu(sample) || (latest_!=0 && sample.time_us<=latest_)) {
        failed_=true;return Status::ImuGap;
    }
    const Status s=current_.predict(sample);
    if (!current_.healthy()) {failed_=true;return s;}
    latest_=sample.time_us;
    if (!initialized_) {
        checkpoint_=current_;
        checkpoint_imu_=sample;
        initialized_=true;
    } else {
        imu_[count_++]=sample;
    }
    trim();
    return s;
}
bool FixedLag::replayPrefix(std::size_t count, const Visual *obs,
                            std::size_t obs_count, Filter &result,
                            std::uint64_t watch_time,
                            Status *watch_status) const noexcept {
    if (!initialized_ || count>count_) {return false;}
    Filter f=checkpoint_;
    Imu prev=checkpoint_imu_;
    std::size_t next=0;
    for(std::size_t i=0;i<count;++i) {
        const Imu &raw=imu_[i];
        while(next<obs_count && obs[next].time_us<raw.time_us) {
            const Visual &v=obs[next];
            if (v.time_us<=prev.time_us) {return false;}
            if (f.predict(interpolated(prev,raw,v.time_us))!=Status::InertialOnly) {
                return false;
            }
            const Status status=f.updateVelocity(v);
            if (v.time_us==watch_time && watch_status) {*watch_status=status;}
            if (status!=Status::VisualCorrected && status!=Status::VisualOutlier) {
                return false;
            }
            ++next;
        }
        if (f.predict(raw)!=Status::InertialOnly) {return false;}
        while(next<obs_count && obs[next].time_us==raw.time_us) {
            const Status status=f.updateVelocity(obs[next]);
            if (obs[next].time_us==watch_time && watch_status) {*watch_status=status;}
            if (status!=Status::VisualCorrected && status!=Status::VisualOutlier) {
                return false;
            }
            ++next;
        }
        prev=raw;
    }
    if (count==count_ && next!=obs_count) {return false;}
    result=f;
    return true;
}
bool FixedLag::replay(const Visual *obs, std::size_t n, Filter &out,
                      std::uint64_t time, Status *status) const noexcept {
    return replayPrefix(count_,obs,n,out,time,status);
}
void FixedLag::trim() noexcept {
    if (!initialized_) {return;}
    while(count_>1 && latest_-checkpoint_imu_.time_us>max_lag_us_) {
        Filter state(parameters_);
        if (!replayPrefix(1,visual_.data(),visual_count_,state)) {
            failed_=true;return;
        }
        checkpoint_=state;
        checkpoint_imu_=imu_[0];
        for(std::size_t i=1;i<count_;++i) {imu_[i-1]=imu_[i];}
        --count_;
        std::size_t kept=0;
        for(std::size_t i=0;i<visual_count_;++i) {
            if(visual_[i].time_us>checkpoint_imu_.time_us) {
                visual_[kept++]=visual_[i];
            }
        }
        visual_count_=kept;
    }
}
Status FixedLag::pushDelayedVelocity(const Visual &m) noexcept {
    if (failed_) {return Status::ImuGap;}
    if (!visualValid(m)) {return Status::InvalidVisual;}
    if (!initialized_ || count_==0) {return Status::NoHistory;}
    if (m.time_us>latest_) {return Status::FutureVisual;}
    if (latest_-m.time_us>max_lag_us_) {return Status::DelayExceeded;}
    if (m.time_us<=checkpoint_imu_.time_us) {return Status::NoHistory;}
    if (visual_count_>=kVisualCapacity) {return Status::BufferLimit;}
    std::array<Visual,kVisualCapacity> candidate=visual_;
    std::size_t insert=0;
    while(insert<visual_count_ && candidate[insert].time_us<m.time_us) {
        ++insert;
    }
    if (insert<visual_count_ && candidate[insert].time_us==m.time_us) {
        return Status::DuplicateVisual;
    }
    for(std::size_t i=visual_count_;i>insert;--i) {
        candidate[i]=candidate[i-1];
    }
    candidate[insert]=m;
    Filter new_state(parameters_);
    Status observed=Status::ReplayFailed;
    if (!replay(candidate.data(),visual_count_+1,new_state,m.time_us,&observed)) {
        return Status::ReplayFailed;
    }
    if (observed!=Status::VisualCorrected && observed!=Status::VisualOutlier) {
        return Status::ReplayFailed;
    }
    visual_=candidate;
    ++visual_count_;
    current_=new_state;
    trim();
    return failed_?Status::ReplayFailed:observed;
}

} // namespace ofnav15
