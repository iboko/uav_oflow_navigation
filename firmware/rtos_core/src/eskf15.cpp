#include "ofnav/eskf15.hpp"

#include <algorithm>
#include <cmath>

namespace ofnav15 {
namespace {

constexpr double kGravity = 9.80665;
constexpr double kPiD = 3.14159265358979323846;
using Mat3 = std::array<double, 9>;

double &at(Matrix15 &a, int i, int j) noexcept { return a[15 * i + j]; }
double at(const Matrix15 &a, int i, int j) noexcept { return a[15 * i + j]; }

void setIdentity(Matrix15 &r) noexcept {
    r.fill(0.);
    for (int i = 0; i < 15; ++i) { at(r, i, i) = 1.; }
}
// r = a * b; r must not alias a or b.
void multiplyInto(const Matrix15 &a, const Matrix15 &b, Matrix15 &r) noexcept {
    r.fill(0.);
    for (int i = 0; i < 15; ++i) {
        for (int k = 0; k < 15; ++k) {
            const double v = at(a, i, k);
            if (v == 0.) { continue; }
            for (int j = 0; j < 15; ++j) { at(r, i, j) += v * at(b, k, j); }
        }
    }
}
// r = a * b^T; r must not alias a or b.
void multiplyTransposeInto(const Matrix15 &a, const Matrix15 &b, Matrix15 &r) noexcept {
    for (int i = 0; i < 15; ++i) {
        for (int j = 0; j < 15; ++j) {
            double sum = 0.;
            for (int k = 0; k < 15; ++k) { sum += at(a, i, k) * at(b, j, k); }
            at(r, i, j) = sum;
        }
    }
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
bool positiveDefinite(const Matrix15 &p, std::array<double, 225> &l) noexcept {
    l.fill(0.);
    for (int i = 0; i < 15; ++i) {
        for (int j = 0; j <= i; ++j) {
            double x = at(p, i, j);
            if (!std::isfinite(x)) { return false; }
            for (int k = 0; k < j; ++k) { x -= l[15 * i + k] * l[15 * j + k]; }
            if (i == j) {
                if (!(x > 1.e-20)) { return false; }
                l[15 * i + j] = std::sqrt(x);
            } else {
                l[15 * i + j] = x / l[15 * j + j];
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
Vec3 matTransposeVec(const Mat3 &a, const Vec3 &v) noexcept {
    return {a[0]*v[0]+a[3]*v[1]+a[6]*v[2],
            a[1]*v[0]+a[4]*v[1]+a[7]*v[2],
            a[2]*v[0]+a[5]*v[1]+a[8]*v[2]};
}
Mat3 skew(const Vec3 &v) noexcept {
    return {0.,-v[2],v[1],v[2],0.,-v[0],-v[1],v[0],0.};
}
double wrapPi(double a) noexcept {
    return std::atan2(std::sin(a), std::cos(a));
}
void jacobianInto(const Mat3 &r, const Vec3 &omega, const Vec3 &force,
                  Matrix15 &f) noexcept {
    f.fill(0.);
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
}
// Fourth-order exp(F dt) by Horner: I + A(I + A/2(I + A/3(I + A/4))).
// Uses two scratch matrices, never more stack than a few scalars.
void transitionInto(const Matrix15 &f, double dt, Matrix15 &phi,
                    Matrix15 &a, Matrix15 &tmp) noexcept {
    for (std::size_t i = 0; i < a.size(); ++i) { a[i] = f[i] * dt; }
    for (std::size_t i = 0; i < phi.size(); ++i) { phi[i] = a[i] / 4.; }
    for (int i = 0; i < 15; ++i) { at(phi, i, i) += 1.; }
    for (const double k : {3., 2., 1.}) {
        multiplyInto(a, phi, tmp);
        for (std::size_t i = 0; i < phi.size(); ++i) { phi[i] = tmp[i] / k; }
        for (int i = 0; i < 15; ++i) { at(phi, i, i) += 1.; }
    }
}
void processCovInto(const Matrix15 &f, const Mat3 &r, double dt,
                    const Parameters &cfg, Matrix15 &q, Matrix15 &phi,
                    Matrix15 &a, Matrix15 &tmp,
                    std::array<double, 180> &z) noexcept {
    const double sigma[4] = {
        cfg.accel_noise, cfg.gyro_noise,
        cfg.gyro_bias_random_walk, cfg.accel_bias_random_walk
    };
    q.fill(0.);
    constexpr double nodes[3] = {0.,-.7745966692414834,.7745966692414834};
    constexpr double weights[3] = {8./9.,5./9.,5./9.};
    for (int n=0;n<3;++n) {
        const double tau=.5*dt*(1.+nodes[n]);
        const double weight=.5*dt*weights[n];
        transitionInto(f,tau,phi,a,tmp);
        // z = Phi(tau) B with the sparse 15x12 noise map B.
        for(int i=0;i<15;++i) {
            for(int j=0;j<3;++j) {
                double accel=0.;
                for(int k=0;k<3;++k) { accel += at(phi,i,3+k)*(-r[3*k+j]); }
                z[i*12+j]=accel*sigma[0];
                z[i*12+3+j]=-at(phi,i,6+j)*sigma[1];
                z[i*12+6+j]=at(phi,i,9+j)*sigma[2];
                z[i*12+9+j]=at(phi,i,12+j)*sigma[3];
            }
        }
        for(int i=0;i<15;++i) {
            for(int j=0;j<15;++j) {
                double value=0.;
                for(int col=0;col<12;++col) {value+=z[i*12+col]*z[j*12+col];}
                at(q,i,j)+=weight*value;
            }
        }
    }
    symmetric(q);
}
bool validParameters(const Parameters &c) noexcept {
    if (!(c.max_dt_s>=.001 && c.max_dt_s<=.05 &&
          c.nis_gate>0 && c.nis_gate_1d>0 &&
          c.max_gyro_bias>0 && c.max_accel_bias>0)) {
        return false;
    }
    const auto &q=c.initial_quaternion_body_to_ned;
    double qnorm2=0.;
    for(double x:q) {
        if (!std::isfinite(x)) {return false;}
        qnorm2+=x*x;
    }
    if (qnorm2<.64 || qnorm2>1.44 ||
        !finite3(c.initial_gyro_bias_rad_s) ||
        !finite3(c.initial_accel_bias_m_s2) ||
        length(c.initial_gyro_bias_rad_s)>c.max_gyro_bias ||
        length(c.initial_accel_bias_m_s2)>c.max_accel_bias) {
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
bool measurementValid(const Measurement &m) noexcept {
    if (m.time_us==0 || static_cast<std::size_t>(m.kind)>=kMeasurementKinds ||
        !std::isfinite(m.ci_weight) || m.ci_weight>1.) {
        return false;
    }
    if (m.ci_weight<1. && m.kind!=MeasurementKind::PositionNe) {return false;}
    if (m.ci_weight>0. && m.ci_weight<1. && (m.ci_weight<.01 || m.ci_weight>.99)) {
        return false;
    }
    const auto &r=m.covariance;
    if (m.kind==MeasurementKind::Yaw) {
        return std::isfinite(m.value[0]) && std::isfinite(r[0]) && r[0]>1.e-12 &&
               r[0]<10.;
    }
    if (!std::isfinite(m.value[0]) || !std::isfinite(m.value[1])) {return false;}
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
bool before(const Measurement &a, const Measurement &b) noexcept {
    return a.time_us<b.time_us ||
        (a.time_us==b.time_us && static_cast<int>(a.kind)<static_cast<int>(b.kind));
}
bool sameEvent(const Measurement &a, const Measurement &b) noexcept {
    return a.time_us==b.time_us && a.kind==b.kind;
}
// CI weight minimising det of the fused 2x2 position covariance:
// argmax_w log det(w P^-1 + (1-w) R^-1) on a fixed 91-point grid.
// The SAME grid is used by src/error_state_inertial_filter.py.
double chooseCiWeight(const double p[4], const double r[4]) noexcept {
    const double pdet=p[0]*p[3]-p[1]*p[2];
    const double rdet=r[0]*r[3]-r[1]*r[2];
    const double pi[4]={p[3]/pdet,-p[1]/pdet,-p[2]/pdet,p[0]/pdet};
    const double ri[4]={r[3]/rdet,-r[1]/rdet,-r[2]/rdet,r[0]/rdet};
    double best_w=.5, best=-std::numeric_limits<double>::infinity();
    for(int step=5;step<=95;++step) {
        const double w=static_cast<double>(step)/100.;
        const double c0=w*pi[0]+(1.-w)*ri[0];
        const double c1=w*pi[1]+(1.-w)*ri[1];
        const double c2=w*pi[2]+(1.-w)*ri[2];
        const double c3=w*pi[3]+(1.-w)*ri[3];
        const double det=c0*c3-c1*c2;
        if (det>best) {best=det;best_w=w;}
    }
    return best_w;
}
} // namespace

Measurement fromVisual(const Visual &visual) noexcept {
    Measurement m{};
    m.time_us=visual.time_us;
    m.kind=MeasurementKind::VelocityNe;
    m.value=visual.velocity_ne_m_s;
    m.covariance=visual.covariance_ne_m2_s2;
    return m;
}

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
        case Status::YawCorrected:return "YAW_CORRECTED";
        case Status::YawOutlier:return "YAW_OUTLIER_REJECTED";
        case Status::PositionCorrected:return "POSITION_CORRECTED";
        case Status::PositionOutlier:return "POSITION_OUTLIER_REJECTED";
    }
    return "UNKNOWN";
}
bool measurementProcessed(Status s) noexcept {
    return s==Status::VisualCorrected || s==Status::VisualOutlier ||
           s==Status::YawCorrected || s==Status::YawOutlier ||
           s==Status::PositionCorrected || s==Status::PositionOutlier;
}
const char *alignmentStatusName(AlignmentStatus s) noexcept {
    switch(s) {
        case AlignmentStatus::Aligned:return "ALIGNED";
        case AlignmentStatus::TooFewSamples:return "TOO_FEW_SAMPLES";
        case AlignmentStatus::NotStationary:return "NOT_STATIONARY";
        case AlignmentStatus::GravityMismatch:return "GRAVITY_MISMATCH";
        case AlignmentStatus::InvalidInput:return "INVALID_INPUT";
    }
    return "UNKNOWN";
}
AlignmentStatus staticAlignment(const Imu *samples, std::size_t n, double yaw,
                                double yaw_sigma, const AlignmentLimits &limits,
                                Parameters &out) noexcept {
    if (samples==nullptr || !std::isfinite(yaw) || !std::isfinite(yaw_sigma) ||
        !(yaw_sigma>0.) || limits.min_samples<2) {
        return AlignmentStatus::InvalidInput;
    }
    if (n<limits.min_samples) {return AlignmentStatus::TooFewSamples;}
    Vec3 mw{}, mf{};
    for(std::size_t i=0;i<n;++i) {
        if (!finiteImu(samples[i]) || (i>0 && samples[i].time_us<=samples[i-1].time_us)) {
            return AlignmentStatus::InvalidInput;
        }
        for(int k=0;k<3;++k) {
            mw[k]+=samples[i].gyro_rad_s[k];
            mf[k]+=samples[i].specific_force_m_s2[k];
        }
    }
    const double count=static_cast<double>(n);
    for(int k=0;k<3;++k) {mw[k]/=count;mf[k]/=count;}
    double vw=0., vf=0.;
    for(std::size_t i=0;i<n;++i) {
        for(int k=0;k<3;++k) {
            const double dw=samples[i].gyro_rad_s[k]-mw[k];
            const double df=samples[i].specific_force_m_s2[k]-mf[k];
            vw+=dw*dw; vf+=df*df;
        }
    }
    const double sw=std::sqrt(vw/(3.*(count-1.)));
    const double sf=std::sqrt(vf/(3.*(count-1.)));
    if (sw>limits.max_gyro_std_rad_s || sf>limits.max_accel_std_m_s2 ||
        length(mw)>out.max_gyro_bias) {
        return AlignmentStatus::NotStationary;
    }
    if (std::abs(length(mf)-kGravity)>limits.max_gravity_error_m_s2) {
        return AlignmentStatus::GravityMismatch;
    }
    // At rest f_b = -R^T g_n: f = (g sin(pitch), -g sin(roll)cos(pitch), -g cos(roll)cos(pitch)).
    const double roll=std::atan2(-mf[1],-mf[2]);
    const double pitch=std::atan2(mf[0],std::sqrt(mf[1]*mf[1]+mf[2]*mf[2]));
    const double cr=std::cos(.5*roll), sr=std::sin(.5*roll);
    const double cp=std::cos(.5*pitch), sp=std::sin(.5*pitch);
    const double cy=std::cos(.5*yaw), sy=std::sin(.5*yaw);
    out.initial_quaternion_body_to_ned={
        cr*cp*cy+sr*sp*sy,
        sr*cp*cy-cr*sp*sy,
        cr*sp*cy+sr*cp*sy,
        cr*cp*sy-sr*sp*cy
    };
    out.initial_gyro_bias_rad_s=mw;
    // Tilt sigma: averaged accelerometer noise plus the unobservable
    // accelerometer bias prior divided by gravity.
    const double tilt=std::sqrt(sf*sf/count/(kGravity*kGravity) +
        std::pow(out.initial_sigma[12]/kGravity,2.));
    out.initial_sigma[6]=std::max(tilt,1.e-4);
    out.initial_sigma[7]=std::max(tilt,1.e-4);
    out.initial_sigma[8]=yaw_sigma;
    // Gyro-bias sigma: standard error of the mean plus Earth rate (7.29e-5),
    // which a MEMS stationary alignment does not separate from the bias.
    const double gb=std::sqrt(sw*sw/count+7.2921159e-5*7.2921159e-5);
    for(int k=9;k<12;++k) {out.initial_sigma[k]=std::max(gb,1.e-6);}
    for(int k=3;k<6;++k) {out.initial_sigma[k]=std::min(out.initial_sigma[k],.05);}
    return AlignmentStatus::Aligned;
}

Filter::Filter(const Parameters &c) noexcept :
    config_(c),
    bg_(c.initial_gyro_bias_rad_s),
    ba_(c.initial_accel_bias_m_s2),
    q_(unit(c.initial_quaternion_body_to_ned)),
    valid_config_(validParameters(c)) {
    for(int i=0;i<15;++i) {at(p_cov_,i,i)=c.initial_sigma[i]*c.initial_sigma[i];}
    if (!valid_config_) {status_=Status::NumericalFailure;}
}
bool Filter::healthy() const noexcept {
    return status_!=Status::ImuGap && status_!=Status::NumericalFailure &&
           status_!=Status::BiasLimit && status_!=Status::PosteriorInvalid;
}
Status Filter::predict(const Imu &m) noexcept {
    Workspace ws{};
    return predict(m, ws);
}
Status Filter::updateVelocity(const Visual &m) noexcept {
    return update(fromVisual(m));
}
Status Filter::update(const Measurement &m) noexcept {
    Workspace ws{};
    return update(m, ws);
}
Status Filter::predict(const Imu &m, Workspace &ws) noexcept {
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
    Matrix15 &f=ws.m[0], &phi=ws.m[1], &q=ws.m[2], &sa=ws.m[3],
             &st=ws.m[4], &phitau=ws.m[5], &fp=ws.m[6], &pnew=ws.m[7];
    jacobianInto(rmid,omega,force,f);
    transitionInto(f,dt,phi,sa,st);
    processCovInto(f,rmid,dt,config_,q,phitau,sa,st,ws.noise_map);
    multiplyInto(phi,p_cov_,fp);
    multiplyTransposeInto(fp,phi,pnew);
    for(std::size_t i=0;i<pnew.size();++i) {pnew[i]+=q[i];}
    symmetric(pnew);
    const Vec3 angle={dt*omega[0],dt*omega[1],dt*omega[2]};
    const Quat qnext=unit(product(q_,expQuat(angle)));
    if (!positiveDefinite(pnew,ws.cholesky) || !std::isfinite(qnext[0]) ||
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
Status Filter::update(const Measurement &m, Workspace &ws) noexcept {
    if (!healthy()) {return status_;}
    if (!has_imu_) {return Status::NoHistory;}
    if (m.time_us!=last_.time_us) {return Status::UnalignedVisual;}
    const std::size_t kind=static_cast<std::size_t>(m.kind);
    if (kind<kMeasurementKinds && m.time_us<=last_measurement_us_[kind]) {
        return Status::RepeatedVisual;
    }
    if (!measurementValid(m)) {++rejected_;return Status::InvalidVisual;}
    Status corrected=Status::VisualCorrected, outlier=Status::VisualOutlier;
    // Measurement model: residual y = z - h(x), Jacobian H (n x 15).
    const int n=m.kind==MeasurementKind::Yaw?1:2;
    double h[2][15]{};
    double y[2]{};
    double r[4]{m.covariance[0],m.covariance[1],m.covariance[2],m.covariance[3]};
    const Mat3 rot=matrixFromQuat(q_);
    switch(m.kind) {
    case MeasurementKind::VelocityNe:
        y[0]=m.value[0]-v_[0]; y[1]=m.value[1]-v_[1];
        h[0][3]=1.; h[1][4]=1.;
        break;
    case MeasurementKind::BodyVelocityXy: {
        // z = [R^T v]_xy; R_true^T v_true ~ R^T v + R^T dv + [R^T v]x dtheta.
        const Vec3 u=matTransposeVec(rot,v_);
        const Mat3 su=skew(u);
        y[0]=m.value[0]-u[0]; y[1]=m.value[1]-u[1];
        for(int i=0;i<2;++i) {
            for(int j=0;j<3;++j) {
                h[i][3+j]=rot[3*j+i];
                h[i][6+j]=su[3*i+j];
            }
        }
        break;
    }
    case MeasurementKind::Yaw: {
        // yaw = atan2(R10, R00); right-error derivative.
        const double den=rot[0]*rot[0]+rot[3]*rot[3];
        if (!(den>1.e-6)) {++rejected_;return Status::InvalidVisual;}
        y[0]=wrapPi(m.value[0]-std::atan2(rot[3],rot[0]));
        h[0][7]=(rot[3]*rot[2]-rot[0]*rot[5])/den;
        h[0][8]=(rot[0]*rot[4]-rot[3]*rot[1])/den;
        r[1]=r[2]=r[3]=0.;
        corrected=Status::YawCorrected; outlier=Status::YawOutlier;
        break;
    }
    case MeasurementKind::PositionNe:
        y[0]=m.value[0]-p_[0]; y[1]=m.value[1]-p_[1];
        h[0][0]=1.; h[1][1]=1.;
        corrected=Status::PositionCorrected; outlier=Status::PositionOutlier;
        break;
    }
    // Innovation covariance S = H P H^T + R (honest, unscaled) for the gate.
    double pht[15][2]{};
    for(int i=0;i<15;++i) {
        for(int k=0;k<n;++k) {
            double sum=0.;
            for(int j=0;j<15;++j) {sum+=at(p_cov_,i,j)*h[k][j];}
            pht[i][k]=sum;
        }
    }
    double s[4]{};
    for(int a=0;a<n;++a) {
        for(int b=0;b<n;++b) {
            double sum=0.;
            for(int i=0;i<15;++i) {sum+=h[a][i]*pht[i][b];}
            s[2*a+b]=sum+r[2*a+b];
        }
    }
    double nis=0.;
    if (n==1) {
        if (!(s[0]>0.) || !std::isfinite(s[0])) {++rejected_;return Status::InvalidVisual;}
        nis=y[0]*y[0]/s[0];
    } else {
        const double det=s[0]*s[3]-s[1]*s[2];
        if (!(s[0]>0 && s[3]>0 && det>1.e-16) || !std::isfinite(det)) {
            ++rejected_;return Status::InvalidVisual;
        }
        nis=(y[0]*(s[3]*y[0]-s[1]*y[1])+y[1]*(-s[2]*y[0]+s[0]*y[1]))/det;
    }
    nis_=nis;
    last_measurement_us_[kind]=m.time_us;
    if (!std::isfinite(nis_)) {++rejected_;return Status::InvalidVisual;}
    if (nis_>(n==1?config_.nis_gate_1d:config_.nis_gate)) {
        ++rejected_;status_=outlier;
        return status_;
    }
    // Covariance intersection for unknown cross-correlation (position only).
    double w=1.;
    if (m.kind==MeasurementKind::PositionNe && m.ci_weight<1.) {
        if (m.ci_weight>0.) {
            w=m.ci_weight;
        } else {
            const double pp[4]={at(p_cov_,0,0),at(p_cov_,0,1),at(p_cov_,1,0),at(p_cov_,1,1)};
            w=chooseCiWeight(pp,r);
        }
    }
    const double pscale=1./w;
    const double rscale=w<1.?1./(1.-w):1.;
    double sc[4]{};
    for(int a=0;a<n;++a) {
        for(int b=0;b<n;++b) {
            double sum=0.;
            for(int i=0;i<15;++i) {sum+=h[a][i]*pht[i][b];}
            sc[2*a+b]=sum*pscale+r[2*a+b]*rscale;
        }
    }
    double si[4]{};
    if (n==1) {
        si[0]=1./sc[0];
    } else {
        const double det=sc[0]*sc[3]-sc[1]*sc[2];
        if (!(det>1.e-16)) {++rejected_;return Status::InvalidVisual;}
        si[0]=sc[3]/det; si[1]=-sc[1]/det; si[2]=-sc[2]/det; si[3]=sc[0]/det;
    }
    double k[15][2]{};
    double dx[15]{};
    for(int i=0;i<15;++i) {
        for(int b=0;b<n;++b) {
            double sum=0.;
            for(int a=0;a<n;++a) {sum+=pht[i][a]*pscale*si[2*a+b];}
            k[i][b]=sum;
        }
        for(int b=0;b<n;++b) {dx[i]+=k[i][b]*y[b];}
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
    // Joseph form (I-KH)(P/w)(I-KH)^T + K (R/(1-w)) K^T; w=1: standard.
    Matrix15 &ikh=ws.m[0], &pw=ws.m[1], &tmp=ws.m[2], &pnew=ws.m[3],
             &reset=ws.m[4], &tmp2=ws.m[5];
    setIdentity(ikh);
    for(int i=0;i<15;++i) {
        for(int j=0;j<15;++j) {
            for(int b=0;b<n;++b) {at(ikh,i,j)-=k[i][b]*h[b][j];}
        }
    }
    for(std::size_t i=0;i<pw.size();++i) {pw[i]=p_cov_[i]*pscale;}
    multiplyInto(ikh,pw,tmp);
    multiplyTransposeInto(tmp,ikh,pnew);
    for(int i=0;i<15;++i) {
        for(int j=0;j<15;++j) {
            double sum=0.;
            for(int a=0;a<n;++a) {
                for(int b=0;b<n;++b) {sum+=k[i][a]*r[2*a+b]*rscale*k[j][b];}
            }
            at(pnew,i,j)+=sum;
        }
    }
    const Vec3 dth={dx[6],dx[7],dx[8]};
    const Mat3 st=skew(dth);
    setIdentity(reset);
    for(int i=0;i<3;++i) {
        for(int j=0;j<3;++j) {
            at(reset,6+i,6+j)-=.5*st[3*i+j];
        }
    }
    multiplyInto(reset,pnew,tmp2);
    multiplyTransposeInto(tmp2,reset,pnew);
    symmetric(pnew);
    const Quat qnew=unit(product(q_,expQuat(dth)));
    if (!positiveDefinite(pnew,ws.cholesky) || !std::isfinite(qnew[0])) {
        status_=Status::PosteriorInvalid;
        return status_;
    }
    p_cov_=pnew;q_=qnew;bg_=bg_new;ba_=ba_new;ci_weight_=w;
    for(int i=0;i<3;++i) {p_[i]+=dx[i];v_[i]+=dx[3+i];}
    ++accepted_;status_=corrected;
    return status_;
}

FixedLag::FixedLag(const Parameters &params, std::uint64_t lag) noexcept :
    parameters_(params),max_lag_us_(lag),checkpoint_(params),current_(params),
    scratch_(params),replay_(params) {
    if (lag==0 || lag>5000000U || !current_.healthy()) {failed_=true;}
}
void FixedLag::invalidateSnapshotsFrom(std::uint64_t time_us) noexcept {
    for (auto &s:snapshots_) {
        if (s.valid && s.time_us>=time_us) {s.valid=false;}
    }
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
    const Status s=current_.predict(sample,ws_);
    if (!current_.healthy()) {failed_=true;return s;}
    latest_=sample.time_us;
    if (!initialized_) {
        checkpoint_=current_;
        checkpoint_imu_=sample;
        initialized_=true;
    } else {
        imu_[count_++]=sample;
        if (++since_snapshot_>=kSnapshotStride) {
            // Replace an invalid or the oldest snapshot.
            std::size_t slot=0;
            for(std::size_t i=0;i<kSnapshots;++i) {
                if (!snapshots_[i].valid) {slot=i;break;}
                if (snapshots_[i].time_us<snapshots_[slot].time_us) {slot=i;}
            }
            snapshots_[slot].valid=true;
            snapshots_[slot].time_us=sample.time_us;
            snapshots_[slot].state=current_;
            since_snapshot_=0;
        }
    }
    trim();
    return s;
}
bool FixedLag::replayRange(const Filter &start, const Imu &start_imu,
                           std::size_t first, std::size_t last,
                           const Measurement *obs, std::size_t obs_count,
                           Filter &result, std::uint64_t watch_time,
                           Status *watch_status, bool refresh) noexcept {
    if (!initialized_ || first>last || last>count_) {return false;}
    Filter &f=replay_;
    f=start;
    Imu prev=start_imu;
    std::size_t next=0;
    // Only events strictly after the start state belong to this replay.
    while(next<obs_count && obs[next].time_us<=start_imu.time_us) {++next;}
    auto apply=[&](const Measurement &v) noexcept {
        const Status status=f.update(v,ws_);
        if (watch_status && v.time_us==watch_time) {
            watch_status[static_cast<std::size_t>(v.kind)]=status;
        }
        return measurementProcessed(status);
    };
    for(std::size_t i=first;i<last;++i) {
        const Imu &raw=imu_[i];
        while(next<obs_count && obs[next].time_us<raw.time_us) {
            const std::uint64_t t=obs[next].time_us;
            if (t<=prev.time_us) {return false;}
            if (f.predict(interpolated(prev,raw,t),ws_)!=Status::InertialOnly) {
                return false;
            }
            while(next<obs_count && obs[next].time_us==t) {
                if (!apply(obs[next])) {return false;}
                ++next;
            }
        }
        if (f.predict(raw,ws_)!=Status::InertialOnly) {return false;}
        while(next<obs_count && obs[next].time_us==raw.time_us) {
            if (!apply(obs[next])) {return false;}
            ++next;
        }
        if (refresh) {
            for(auto &snap:snapshots_) {
                if (snap.valid && snap.time_us==raw.time_us) {snap.state=f;}
            }
        }
        prev=raw;
    }
    if (last==count_ && next!=obs_count) {return false;}
    result=f;
    return true;
}
void FixedLag::trim() noexcept {
    if (!initialized_) {return;}
    while(count_>1 && latest_-checkpoint_imu_.time_us>max_lag_us_) {
        if (!replayRange(checkpoint_,checkpoint_imu_,0,1,visual_.data(),
                         visual_count_,scratch_,0,nullptr,false)) {
            failed_=true;return;
        }
        checkpoint_=scratch_;
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
        for (auto &snap:snapshots_) {
            if (snap.valid && snap.time_us<=checkpoint_imu_.time_us) {snap.valid=false;}
        }
    }
}
Status FixedLag::pushDelayedVelocity(const Visual &m) noexcept {
    return pushDelayed(fromVisual(m));
}
Status FixedLag::pushDelayed(const Measurement &m) noexcept {
    Status status=Status::ReplayFailed;
    const Status overall=pushDelayedFrame(&m,1,&status);
    return overall==Status::ReplayFailed?overall:status;
}
Status FixedLag::pushDelayedFrame(const Measurement *events, std::size_t n,
                                  Status *statuses) noexcept {
    if (events==nullptr || statuses==nullptr || n==0 || n>kMaxFrameEvents) {
        return Status::InvalidVisual;
    }
    const std::uint64_t t=events[0].time_us;
    // Per-event admission; rejected events never enter the history.
    std::array<bool,kMaxFrameEvents> admit{};
    std::size_t admitted=0;
    for(std::size_t e=0;e<n;++e) {
        const Measurement &m=events[e];
        Status s=Status::ReplayFailed;
        bool repeated_kind=false;
        for(std::size_t j=0;j<e;++j) {
            if (events[j].kind==m.kind) {repeated_kind=true;}
        }
        if (failed_) {s=Status::ImuGap;}
        else if (m.time_us!=t || repeated_kind || !measurementValid(m)) {s=Status::InvalidVisual;}
        else if (!initialized_ || count_==0) {s=Status::NoHistory;}
        else if (m.time_us>latest_) {s=Status::FutureVisual;}
        else if (latest_-m.time_us>max_lag_us_) {s=Status::DelayExceeded;}
        else if (m.time_us<=checkpoint_imu_.time_us) {s=Status::NoHistory;}
        else if (visual_count_+admitted>=kVisualCapacity) {s=Status::BufferLimit;}
        else {
            bool duplicate=false;
            for(std::size_t i=0;i<visual_count_;++i) {
                if (sameEvent(visual_[i],m)) {duplicate=true;}
            }
            if (duplicate) {s=Status::DuplicateVisual;}
            else {admit[e]=true;++admitted;}
        }
        statuses[e]=s;
    }
    if (admitted==0) {return statuses[0];}
    // Merge admitted events into a sorted candidate history.
    std::size_t c=0;
    std::size_t i=0;
    std::array<Measurement,kMaxFrameEvents> batch{};
    std::size_t batch_count=0;
    for(std::size_t e=0;e<n;++e) {if (admit[e]) {batch[batch_count++]=events[e];}}
    for(std::size_t x=1;x<batch_count;++x) {
        for(std::size_t y=x;y>0 && before(batch[y],batch[y-1]);--y) {
            const Measurement tmp=batch[y];batch[y]=batch[y-1];batch[y-1]=tmp;
        }
    }
    std::size_t b=0;
    while(i<visual_count_ || b<batch_count) {
        if (b>=batch_count || (i<visual_count_ && before(visual_[i],batch[b]))) {
            candidate_[c++]=visual_[i++];
        } else {
            candidate_[c++]=batch[b++];
        }
    }
    // Newest valid state strictly older than the exposure.
    const Filter *start=&checkpoint_;
    Imu start_imu=checkpoint_imu_;
    std::size_t first=0;
    for(const auto &snap:snapshots_) {
        if (!snap.valid || snap.time_us>=t || snap.time_us<=start_imu.time_us) {continue;}
        for(std::size_t k=0;k<count_;++k) {
            if (imu_[k].time_us==snap.time_us) {
                start=&snap.state;start_imu=imu_[k];first=k+1;
                break;
            }
        }
    }
    std::array<Status,kMeasurementKinds> observed{};
    observed.fill(Status::ReplayFailed);
    last_replay_length_=count_-first;
    // Refreshed snapshots are strictly newer than the start state, so the
    // replay never overwrites the state it started from.
    if (!replayRange(*start,start_imu,first,count_,candidate_.data(),c,scratch_,
                     t,observed.data(),true)) {
        invalidateSnapshotsFrom(t);
        for(std::size_t e=0;e<n;++e) {if (admit[e]) {statuses[e]=Status::ReplayFailed;}}
        return Status::ReplayFailed;
    }
    for(std::size_t e=0;e<n;++e) {
        if (!admit[e]) {continue;}
        statuses[e]=observed[static_cast<std::size_t>(events[e].kind)];
        if (!measurementProcessed(statuses[e])) {
            invalidateSnapshotsFrom(t);
            for(std::size_t j=0;j<n;++j) {if (admit[j]) {statuses[j]=Status::ReplayFailed;}}
            return Status::ReplayFailed;
        }
    }
    visual_=candidate_;
    visual_count_=c;
    current_=scratch_;
    trim();
    if (failed_) {return Status::ReplayFailed;}
    return statuses[0];
}

} // namespace ofnav15
