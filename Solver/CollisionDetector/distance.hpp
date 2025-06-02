// File: distance.hpp
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef DISTANCE_HPP
#define DISTANCE_HPP

// #include <Eigen/Cholesky>
#include "Core/float_nxn.h"
#include "Core/scalar.h"
#include "SimulationCore/simulation_type.h"
#include "luisa/dsl/sugar.h"
#include <vector>
#include <string>
#include <luisa/luisa-compute.h>
#include <Eigen/Cholesky>

namespace lcsv {

namespace distance {

static inline Var<float> squared_norm(const Var<luisa::float3>& vec)
{
    return luisa::compute::dot(vec, vec);
}
static inline Var<bool> all(const Var<luisa::bool3>& vec)
{
    return luisa::compute::all(vec);
}

using Mat2x2f = Float2x2;
using Mat2x3f = Float3x2;
using Mat3x2f = Float2x3;
using Vec2f = Float2;
using Vec3f = Float3;
using Vec4f = Float4;

inline Var<bool> solve(const Mat2x2f &a, const Vec2f &b, Vec2f &x) {
    auto det = a[0][0] * a[1][1] - a[1][0] * a[0][1];
    Var<bool> is_safe = false;
    $if (det != 0.0f) {
        // a_inv << a(1, 1) / det, -a(0, 1) / det, -a(1, 0) / det, a(0, 0) / det;
        Mat2x2f a_inv = makeFloat2x2(
            makeFloat2( a[1][1] / det, -a[0][1] / det),
            makeFloat2(-a[1][0] / det,  a[0][0] / det)
        );
        x = a_inv * b;
        is_safe = (true);
    };
    return (is_safe);
}

// TODO: Check Me !!!!!!!!
inline Vec2f point_edge_distance_coeff(const Vec3f &p,
                                       const Vec3f &e0,
                                       const Vec3f &e1) {
    Vec3f r = (e1 - e0);
    auto d = squared_norm(r) ; // r.squaredNorm();
    Vec2f bary = Vec2f(0.5f, 0.5f);
    $if (d > Epsilon) {
        auto x = dot_vec(r, p - e0) / d;
        bary = Vec2f(1.0f - x, x);
    };
    return bary;
}

inline Vec3f point_triangle_distance_coeff(const Vec3f &p,
                                           const Vec3f &t0,
                                           const Vec3f &t1,
                                           const Vec3f &t2) {
    Vec3f r0 = (t1 - t0);
    Vec3f r1 = (t2 - t0);
    Mat3x2f a = makeFloat2x3(r0, r1);
    Mat2x3f a_t = transpose_2x3(a);
    Vec2f c;
    $if (!solve(mult(a_t,  a), mult(a_t, (p - t0)), c)) {
        c = Vec2f(1.0f / 3.0f, 1.0f / 3.0f);
    };
    return Vec3f(1.0f - c[0] - c[1], c[0], c[1]);
}

inline Vec4f edge_edge_distance_coeff(const Vec3f &ea0,
                                      const Vec3f &ea1,
                                      const Vec3f &eb0,
                                      const Vec3f &eb1) {

    Vec3f r0 = (ea1 - ea0);
    Vec3f r1 = (eb1 - eb0);
    Mat3x2f a = makeFloat2x3(r0, -r1);
    Mat2x3f a_t = transpose_2x3(a);
    Vec2f x;
    Vec4f bary(0.5f, 0.5f, 0.5f, 0.5f);
    $if (solve(mult(a_t, a), mult(a_t, (eb0 - ea0)), x)) {
        // luisa::compute::device_log("r0 = {}, r1 = {}, x = {}", r0, r1, x);
        bary = Vec4f(1.0f - x[0], x[0], 1.0f - x[1], x[1]);
    };
    return bary;
}

inline Vec3f point_triangle_distance_coeff_unclassified(
    const Vec3f &p, 
    const Vec3f &t0, 
    const Vec3f &t1,
    const Vec3f &t2) {

    Vec3f c = point_triangle_distance_coeff(p, t0, t1, t2);
    Vec3f result = c;
    $if (all(c >= 0.0f) & all(c <= 1.0f)) {

    }
    $elif (c[0] < 0.0f) {
        Vec2f c = point_edge_distance_coeff(p, t1, t2);
        $if (c[0] >= 0.0f & c[0] <= 1.0f) {
            result = Vec3f(0.0f, c[0], c[1]);
        } $else {
            $if (c[0] > 1.0f) {
                result = Vec3f(0.0f, 1.0f, 0.0f);
            } $else {
                result = Vec3f(0.0f, 0.0f, 1.0f);
            };
        };
    } $elif (c[1] < 0.0f) {
        Vec2f c = point_edge_distance_coeff(p, t0, t2);
        $if (c[0] >= 0.0f & c[0] <= 1.0f) {
            result = Vec3f(c[0], 0.0f, c[1]);
        } $else {
            $if (c[0] > 1.0f) {
                result = Vec3f(1.0f, 0.0f, 0.0f);
            } $else {
                result = Vec3f(0.0f, 0.0f, 1.0f);
            };
        };
    } $else {
        Vec2f c = point_edge_distance_coeff(p, t0, t1);
        $if (c[0] >= 0.0f & c[0] <= 1.0f) {
            result = Vec3f(c[0], c[1], 0.0f);
        } $else {
            $if (c[0] > 1.0f) {
                result = Vec3f(1.0f, 0.0f, 0.0f);
            } $else {
                result = Vec3f(0.0f, 1.0f, 0.0f);
            };
        };
    };
    return result;
}

inline auto point_triangle_distance_squared_unclassified(
    const Vec3f &p, 
    const Vec3f &t0, 
    const Vec3f &t1,
    const Vec3f &t2) {
    Vec3f c = point_triangle_distance_coeff_unclassified(p, t0, t1, t2);
    Vec3f x = c[0] * (t0 - p) +
              c[1] * (t1 - p) +
              c[2] * (t2 - p);
    return squared_norm(x);
}

inline Vec4f edge_edge_distance_coeff_unclassified(const Vec3f &ea0,
                                                  const Vec3f &ea1,
                                                  const Vec3f &eb0,
                                                  const Vec3f &eb1) {

    Vec4f c = edge_edge_distance_coeff(ea0, ea1, eb0, eb1);
    // luisa::compute::device_log("c = {}", c);
    Vec4f result = c;
    $if (all(c >= 0.0f) & all(c <= 1.0f))  {
    } $else {
        Vec2f c1 = point_edge_distance_coeff(ea0, eb0, eb1);
        Vec2f c2 = point_edge_distance_coeff(ea1, eb0, eb1);
        Vec2f c3 = point_edge_distance_coeff(eb0, ea0, ea1);
        Vec2f c4 = point_edge_distance_coeff(eb1, ea0, ea1);
        $if (c1[0] < 0.0f) {
            c1 = Vec2f(0.0f, 1.0f);
        } $elif (c1[0] > 1.0f) {
            c1 = Vec2f(1.0f, 0.0f);
        };
        $if (c2[0] < 0.0f) {
            c2 = Vec2f(0.0f, 1.0f);
        } $elif (c2[0] > 1.0f) {
            c2 = Vec2f(1.0f, 0.0f);
        };
        $if (c3[0] < 0.0f) {
            c3 = Vec2f(0.0f, 1.0f);
        } $elif (c3[0] > 1.0f) {
            c3 = Vec2f(1.0f, 0.0f);
        };
        $if (c4[0] < 0.0f) {
            c4 = Vec2f(0.0f, 1.0f);
        } $elif (c4[0] > 1.0f) {
            c4 = Vec2f(1.0f, 0.0f);
        };
        Vec4f types[4] = {
            Vec4f(1.0f, 0.0f, c1[0], c1[1]), 
            Vec4f(0.0f, 1.0f, c2[0], c2[1]),
            Vec4f(c3[0], c3[1], 1.0f, 0.0f), 
            Vec4f(c4[0], c4[1], 0.0f, 1.0f)};
        uint index = 0;
        Var<float> di = Float_max;
        for (unsigned i = 0; i < 4; ++i) {
            const auto &c = types[i];
            Vec3f x0 = c[0] * ea0 +
                       c[1] * ea1;
            Vec3f x1 = c[2] * eb0 +
                       c[3] * eb1;
            Var<float> d = squared_norm(x0 - x1);
            $if (d < di) {
                index = i;
                di = d;
            };
        }
        result = types[index];
    };
    return result;
}

inline auto edge_edge_distance_squared_unclassified(
    const Vec3f &ea0, 
    const Vec3f &ea1, 
    const Vec3f &eb0,
    const Vec3f &eb1) {
    Vec4f c = edge_edge_distance_coeff_unclassified(ea0, ea1, eb0, eb1);
    Vec3f x0 = c[0] * ea0 +
               c[1] * ea1;
    Vec3f x1 = c[2] * eb0 +
               c[3] * eb1;
    return squared_norm(x1 - x0);
}

} // namespace distance


namespace host_distance {

using Eigen::Map;
template <class T, unsigned N> using SVec = Eigen::Vector<T, N>;
template <unsigned N> using SVecf = SVec<float, N>;
template <unsigned N> using SVecu = SVec<unsigned, N>;

using Vec2f = SVecf<2>;
using Vec3f = SVecf<3>;
using Vec4f = SVecf<4>;
using Vec6f = SVecf<6>;
using Vec9f = SVecf<9>;
using Vec12f = SVecf<12>;
using Vec1u = SVecu<1>;
using Vec2u = SVecu<2>;
using Vec3u = SVecu<3>;
using Vec4u = SVecu<4>;

template <class T, unsigned R, unsigned C>
using SMat = Eigen::Matrix<T, R, C, Eigen::ColMajor>;
template <unsigned R, unsigned C>
using SMatf = Eigen::Matrix<float, R, C, Eigen::ColMajor>;

using Mat2x3f = SMatf<2, 3>;
using Mat3x2f = SMatf<3, 2>;
using Mat2x2f = SMatf<2, 2>;
using Mat3x3f = SMatf<3, 3>;
using Mat3x4f = SMatf<3, 4>;
using Mat3x6f = SMatf<3, 6>;
using Mat4x3f = SMatf<4, 3>;
using Mat4x4f = SMatf<4, 4>;
using Mat3x9f = SMatf<3, 9>;
using Mat6x6f = SMatf<6, 6>;
using Mat6x9f = SMatf<6, 9>;
using Mat9x9f = SMatf<9, 9>;
using Mat9x12f = SMatf<9, 12>;
using Mat12x12f = SMatf<12, 12>;

const float EPSILON = 1e-8;
const float FLT_MAX = 1e8;

inline bool solve(const Mat2x2f &a, const Vec2f &b, Vec2f &x) {
    float det = a(0, 0) * a(1, 1) - a(1, 0) * a(0, 1);
    if (det) {
        Mat2x2f a_inv;
        a_inv << a(1, 1) / det, -a(0, 1) / det, -a(1, 0) / det, a(0, 0) / det;
        x = a_inv * b;
        return true;
    }
    return false;
}

template <class T>
inline Vec2f point_edge_distance_coeff(const SVec<T, 3> &p,
                                           const SVec<T, 3> &e0,
                                           const SVec<T, 3> &e1) {
    Vec3f r = (e1 - e0).template cast<float>();
    float d = r.squaredNorm();
    if (d > EPSILON) {
        float x = r.dot((p - e0).template cast<float>()) / d;
        return Vec2f(1.0f - x, x);
    } else {
        return Vec2f(0.5f, 0.5f);
    }
}

template <class T>
inline Vec3f point_triangle_distance_coeff(const SVec<T, 3> &p,
                                               const SVec<T, 3> &t0,
                                               const SVec<T, 3> &t1,
                                               const SVec<T, 3> &t2) {
    Vec3f r0 = (t1 - t0).template cast<float>();
    Vec3f r1 = (t2 - t0).template cast<float>();
    Mat3x2f a;
    a << r0, r1;
    Eigen::Transpose<Mat3x2f> a_t = a.transpose();
    Vec2f c;
    if (!solve(a_t * a, a_t * (p - t0).template cast<float>(), c)) {
        c = Vec2f(1.0f / 3.0f, 1.0f / 3.0f);
    }
    return Vec3f(1.0f - c[0] - c[1], c[0], c[1]);
}

template <class T>
inline Vec4f edge_edge_distance_coeff(const SVec<T, 3> &ea0,
                                          const SVec<T, 3> &ea1,
                                          const SVec<T, 3> &eb0,
                                          const SVec<T, 3> &eb1) {
    Vec3f r0 = (ea1 - ea0).template cast<float>();
    Vec3f r1 = (eb1 - eb0).template cast<float>();
    Mat3x2f a;
    a << r0, -r1;
    Eigen::Transpose<Mat3x2f> a_t = a.transpose();
    Vec2f x;
    if (solve(a.transpose() * a, a.transpose() * (eb0 - ea0).template cast<float>(), x)) {
        // luisa::log_info("r0 = {}, r1 = {}, x = {}", r0, r1, x);
        return Vec4f(1.0f - x[0], x[0], 1.0f - x[1], x[1]);
    } else {
        return Vec4f(0.5f, 0.5f, 0.5f, 0.5f);
    }
}

template <class T>
inline Vec3f point_triangle_distance_coeff_unclassified(
    const SVec<T, 3> &p, const SVec<T, 3> &t0, const SVec<T, 3> &t1,
    const SVec<T, 3> &t2) {

    Vec3f c = point_triangle_distance_coeff(p, t0, t1, t2);
    if (c[0] >= 0.0f && c[0] <= 1.0f && c[1] >= 0.0f && c[1] <= 1.0f &&
        c[2] >= 0.0f && c[2] <= 1.0f) {
        return c;
    } else if (c[0] < 0.0f) {
        Vec2f c = point_edge_distance_coeff(p, t1, t2);
        if (c(0) >= 0.0f && c(0) <= 1.0f) {
            return Vec3f(0.0f, c(0), c(1));
        } else {
            if (c(0) > 1.0f) {
                return Vec3f(0.0f, 1.0f, 0.0f);
            } else {
                return Vec3f(0.0f, 0.0f, 1.0f);
            }
        }
    } else if (c[1] < 0.0f) {
        Vec2f c = point_edge_distance_coeff(p, t0, t2);
        if (c(0) >= 0.0f && c(0) <= 1.0f) {
            return Vec3f(c(0), 0.0f, c(1));
        } else {
            if (c(0) > 1.0f) {
                return Vec3f(1.0f, 0.0f, 0.0f);
            } else {
                return Vec3f(0.0f, 0.0f, 1.0f);
            }
        }
    } else {
        Vec2f c = point_edge_distance_coeff(p, t0, t1);
        if (c(0) >= 0.0f && c(0) <= 1.0f) {
            return Vec3f(c(0), c(1), 0.0f);
        } else {
            if (c(0) > 1.0f) {
                return Vec3f(1.0f, 0.0f, 0.0f);
            } else {
                return Vec3f(0.0f, 1.0f, 0.0f);
            }
        }
    }
}

template <class T>
inline float point_triangle_distance_squared_unclassified(
    const SVec<T, 3> &p, const SVec<T, 3> &t0, const SVec<T, 3> &t1,
    const SVec<T, 3> &t2) {
    Vec3f c = point_triangle_distance_coeff_unclassified(p, t0, t1, t2);
    Vec3f x = c(0) * (t0 - p).template cast<float>() +
              c(1) * (t1 - p).template cast<float>() +
              c(2) * (t2 - p).template cast<float>();
    return x.squaredNorm();
}

template <class T>
inline Vec4f edge_edge_distance_coeff_unclassified(const SVec<T, 3> &ea0,
                                                       const SVec<T, 3> &ea1,
                                                       const SVec<T, 3> &eb0,
                                                       const SVec<T, 3> &eb1) {

    Vec4f c = edge_edge_distance_coeff(ea0, ea1, eb0, eb1);
    // luisa::compute::log_info("c = {}", c);
    if (c[0] >= 0.0f && c[0] <= 1.0f && c[1] >= 0.0f && c[1] <= 1.0f &&
        c[2] >= 0.0f && c[2] <= 1.0f && c[3] >= 0.0f && c[3] <= 1.0f) {
        return c;
    } else {
        Vec2f c1 = point_edge_distance_coeff(ea0, eb0, eb1);
        Vec2f c2 = point_edge_distance_coeff(ea1, eb0, eb1);
        Vec2f c3 = point_edge_distance_coeff(eb0, ea0, ea1);
        Vec2f c4 = point_edge_distance_coeff(eb1, ea0, ea1);
        if (c1(0) < 0.0f) {
            c1 = Vec2f(0.0f, 1.0f);
        } else if (c1(0) > 1.0f) {
            c1 = Vec2f(1.0f, 0.0f);
        }
        if (c2(0) < 0.0f) {
            c2 = Vec2f(0.0f, 1.0f);
        } else if (c2(0) > 1.0f) {
            c2 = Vec2f(1.0f, 0.0f);
        }
        if (c3(0) < 0.0f) {
            c3 = Vec2f(0.0f, 1.0f);
        } else if (c3(0) > 1.0f) {
            c3 = Vec2f(1.0f, 0.0f);
        }
        if (c4(0) < 0.0f) {
            c4 = Vec2f(0.0f, 1.0f);
        } else if (c4(0) > 1.0f) {
            c4 = Vec2f(1.0f, 0.0f);
        }
        Vec4f types[] = {
            Vec4f(1.0f, 0.0f, c1(0), c1(1)), Vec4f(0.0f, 1.0f, c2(0), c2(1)),
            Vec4f(c3(0), c3(1), 1.0f, 0.0f), Vec4f(c4(0), c4(1), 0.0f, 1.0f)};
        unsigned index = 0;
        float di = FLT_MAX;
        for (unsigned i = 0; i < 4; ++i) {
            const auto &c = types[i];
            Vec3f x0 = c(0) * ea0.template cast<float>() +
                       c(1) * ea1.template cast<float>();
            Vec3f x1 = c(2) * eb0.template cast<float>() +
                       c(3) * eb1.template cast<float>();
            float d = (x0 - x1).squaredNorm();
            if (d < di) {
                index = i;
                di = d;
            }
        }
        return types[index];
    }
}

template <class T>
inline float edge_edge_distance_squared_unclassified(
    const SVec<T, 3> &ea0, const SVec<T, 3> &ea1, const SVec<T, 3> &eb0,
    const SVec<T, 3> &eb1) {
    Vec4f c = edge_edge_distance_coeff_unclassified(ea0, ea1, eb0, eb1);
    Vec3f x0 = c[0] * ea0.template cast<float>() +
               c[1] * ea1.template cast<float>();
    Vec3f x1 = c[2] * eb0.template cast<float>() +
               c[3] * eb1.template cast<float>();
    return (x1 - x0).squaredNorm();
}

} // namespace host_distance


} // namespace lcsv

#endif
