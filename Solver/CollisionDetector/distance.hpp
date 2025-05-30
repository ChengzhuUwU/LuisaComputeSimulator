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
        Vec4f types[] = {
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
            auto d = squared_norm(x0 - x1);
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

} // namespace lcsv

#endif
