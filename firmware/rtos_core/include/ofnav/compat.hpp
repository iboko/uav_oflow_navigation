#pragma once

// Freestanding replacements for the parts of the C++ standard library the
// flight code needs. PX4 on NuttX ships only the C wrappers (<cmath>,
// <cstddef>, <cstdint>, ...): <array>, <limits> and <algorithm> are not
// available there, and NuttX <cmath> has no std::isfinite / std::abs.
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ofnav {

// Aggregate fixed-size array: brace-initializable like std::array.
template <class T, std::size_t N>
struct Array {
    T elems[N];

    T &operator[](std::size_t i) noexcept { return elems[i]; }
    const T &operator[](std::size_t i) const noexcept { return elems[i]; }
    T *data() noexcept { return elems; }
    const T *data() const noexcept { return elems; }
    static constexpr std::size_t size() noexcept { return N; }
    T *begin() noexcept { return elems; }
    T *end() noexcept { return elems + N; }
    const T *begin() const noexcept { return elems; }
    const T *end() const noexcept { return elems + N; }
    void fill(const T &value) noexcept {
        for (std::size_t i = 0; i < N; ++i) { elems[i] = value; }
    }
    friend bool operator==(const Array &a, const Array &b) noexcept {
        for (std::size_t i = 0; i < N; ++i) {
            if (!(a.elems[i] == b.elems[i])) { return false; }
        }
        return true;
    }
    friend bool operator!=(const Array &a, const Array &b) noexcept { return !(a == b); }
};

template <class T>
constexpr const T &minOf(const T &a, const T &b) noexcept { return (b < a) ? b : a; }
template <class T>
constexpr const T &maxOf(const T &a, const T &b) noexcept { return (a < b) ? b : a; }

constexpr double kQuietNaN = __builtin_nan("");
constexpr float kQuietNaNf = __builtin_nanf("");
constexpr double kInfinity = __builtin_inf();

inline bool isFiniteValue(double x) noexcept { return __builtin_isfinite(x); }
inline bool isFiniteValue(float x) noexcept { return __builtin_isfinite(x); }
inline double absValue(double x) noexcept { return __builtin_fabs(x); }
inline float absValue(float x) noexcept { return __builtin_fabsf(x); }

} // namespace ofnav
