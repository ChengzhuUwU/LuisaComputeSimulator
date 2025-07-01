// File: distance.hpp
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef DISTANCE_HPP
#define DISTANCE_HPP

// #include <Eigen/Cholesky>
#include "Core/constant_value.h"
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

// Get bary
inline Vec3f point_triangle_distance_coeff_unclassified(
    const Vec3f &p, 
    const Vec3f &t0, 
    const Vec3f &t1,
    const Vec3f &t2) {

    Vec3f c = point_triangle_distance_coeff(p, t0, t1, t2);
    Vec3f result = c;
    $if (all(c >= 0.0f) & all(c <= 1.0f)) { // VF

    }
    $elif (c[0] < 0.0f) {
        Vec2f c = point_edge_distance_coeff(p, t1, t2);
        $if (c[0] >= 0.0f & c[0] <= 1.0f) { // V-E1
            result = Vec3f(0.0f, c[0], c[1]); 
        } $else {
            $if (c[0] > 1.0f) {
                result = Vec3f(0.0f, 1.0f, 0.0f); // V-V2
            } $else {
                result = Vec3f(0.0f, 0.0f, 1.0f); // V-V2
            };
        };
    } $elif (c[1] < 0.0f) {
        Vec2f c = point_edge_distance_coeff(p, t0, t2);
        $if (c[0] >= 0.0f & c[0] <= 1.0f) { // V-E2
            result = Vec3f(c[0], 0.0f, c[1]); 
        } $else {
            $if (c[0] > 1.0f) {
                result = Vec3f(1.0f, 0.0f, 0.0f); // V-V1
            } $else {
                result = Vec3f(0.0f, 0.0f, 1.0f); // V-V3
            };
        };
    } $else {
        Vec2f c = point_edge_distance_coeff(p, t0, t1);
        $if (c[0] >= 0.0f & c[0] <= 1.0f) { // V-E3
            result = Vec3f(c[0], c[1], 0.0f); 
        } $else {
            $if (c[0] > 1.0f) {
                result = Vec3f(1.0f, 0.0f, 0.0f); // V-V1
            } $else {
                result = Vec3f(0.0f, 1.0f, 0.0f); // V-V2
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

inline auto point_point_distance_squared_unclassified(
    const Vec3f &p0, 
    const Vec3f &p1) 
{
    return squared_norm(p0 - p1);
}
inline auto point_edge_distance_squared_unclassified(
    const Vec3f &p, 
    const Vec3f &e0,
    const Vec3f &e1
) 
{
    return squared_norm(cross_vec(e0 - p, e1 - p)) / squared_norm(e1 - e0);
}

} // namespace distance

namespace distance 
{

// enum class PointPointDistanceType : unsigned char
// {
//     PP = 0,  // point-point, the shortest distance is the distance between the two points
// };

// enum class PointEdgeDistanceType : unsigned char
// {
//     PP_PE0 = 0,  // point-edge, the shortest distance is the distance between the point and the point 0 in edge
//     PP_PE1 = 1,  // point-edge, the shortest distance is the distance between the point and the point 1 in edge
//     PE = 2,  // point-edge, the shortest distance is the distance between the point and some point on the edge
// };

// enum class PointTriangleDistanceType : unsigned char
// {
//     PP_PT0 = 0,  // point-triangle, the closest point is the point 0 in triangle
//     PP_PT1 = 1,  // point-triangle, the closest point is the point 1 in triangle
//     PP_PT2 = 2,  // point-triangle, the closest point is the point 2 in triangle
//     PE_PT0T1 = 3,  // point-triangle, the closest point is on the edge (t0, t1)
//     PE_PT1T2 = 4,  // point-triangle, the closest point is on the edge (t1, t2)
//     PE_PT2T0 = 5,  // point-triangle, the closest point is on the edge (t2, t0)
//     PT       = 6,  // point-triangle, the closest point is on the triangle
// };

// enum class EdgeEdgeDistanceType : unsigned char
// {
//     PP_Ea0Eb0 = 0,  // point-point, the shortest distance is the distance between the point 0 in edge a and the point 0 in edge b
//     PP_Ea0Eb1 = 1,  // point-point, the shortest distance is the distance between the point 0 in edge a and the point 1 in edge b
//     PE_Ea0Eb0Eb1 = 2,  // point-edge, the shortest distance is the distance between the point 0 in edge a and some point the edge b
//     PP_Ea1Eb0 = 3,  // point-point, the shortest distance is the distance between the point 1 in edge a and the point 0 in edge b
//     PP_Ea1Eb1 = 4,  // point-point, the shortest distance is the distance between the point 1 in edge a and the point 1 in edge b
//     PE_Ea1Eb0Eb1 = 5,  // point-edge, the shortest distance is the distance between the point 1 in edge a and some point the edge b
//     PE_Eb0Ea0Ea1 = 6,  // point-edge, the shortest distance is the distance between the point 0 in edge b and some point the edge a
//     PE_Eb1Ea0Ea1 = 7,  // point-edge, the shortest distance is the distance between the point 1 in edge b and some point the edge a
//     EE = 8,  // edge-edge, the shortest distance is the distance between some point on edge a and some point on edge b
// };

// inline Vec3f point_triangle_type(
//     const Vec3f &p, 
//     const Vec3f &t0, 
//     const Vec3f &t1,
//     const Vec3f &t2,
//     PointTriangleDistanceType& vf_type) {
//     Vec3f c = point_triangle_distance_coeff(p, t0, t1, t2);
//     Vec3f bary = c;
//     $if (all(c >= 0.0f) & all(c <= 1.0f)) { // VF
//         { vf_type = PointTriangleDistanceType::PT; }
//         bary = c;
//     } $elif (c[0] < 0.0f) {
//         Float2 c = distance::point_edge_distance_coeff(p, t1, t2);
//         $if (c[0] >= 0.0f & c[0] <= 1.0f) { // V-E0
//             { vf_type = PointTriangleDistanceType::PE_PT1T2; }
//             bary = Float3(0.0f, c[0], c[1]); 
//         } $else {
//             $if (c[0] > 1.0f) {
//                 { vf_type = PointTriangleDistanceType::PP_PT1; }
//                 bary = Float3(0.0f, 1.0f, 0.0f); // V-V1
//             } $else {
//                 { vf_type = PointTriangleDistanceType::PP_PT2; }
//                 bary = Float3(0.0f, 0.0f, 1.0f); // V-V2
//             };
//         };
//     } $elif (c[1] < 0.0f) {
//         Float2 c = distance::point_edge_distance_coeff(p, t0, t2);
//         $if (c[0] >= 0.0f & c[0] <= 1.0f) { // V-E1
//             { vf_type = PointTriangleDistanceType::PE_PT2T0; }
//             bary = Float3(c[0], 0.0f, c[1]); 
//         } $else {
//             $if (c[0] > 1.0f) {
//                 { vf_type = PointTriangleDistanceType::PP_PT0; }
//                 bary = Float3(1.0f, 0.0f, 0.0f); // V-V0
//             } $else {
//                 { vf_type = PointTriangleDistanceType::PP_PT2; }
//                 bary = Float3(0.0f, 0.0f, 1.0f); // V-V2
//             };
//         };
//     } $else {
//         Float2 c = distance::point_edge_distance_coeff(p, t0, t1);
//         $if (c[0] >= 0.0f & c[0] <= 1.0f) { // V-E2
//             { vf_type = PointTriangleDistanceType::PE_PT0T1; }
//             bary = Float3(c[0], c[1], 0.0f); 
//         } $else {
//             $if (c[0] > 1.0f) {
//                 { vf_type = PointTriangleDistanceType::PP_PT0; }
//                 bary = Float3(1.0f, 0.0f, 0.0f); // V-V0
//             } $else {
//                 { vf_type = PointTriangleDistanceType::PP_PT1; }
//                 bary = Float3(0.0f, 1.0f, 0.0f); // V-V1
//             };
//         };
//     };
//     return bary;
// }

inline uint point_triangle_type(
    const Vec3f &bary, 
    luisa::uint3& valid_indices
    ) 
{
    Bool3 is_valid = bary == 0.0f;
    uint valid_count = 0;

    $if (is_valid[0]) { valid_indices[valid_count] = 0; valid_count += 1; };
    $if (is_valid[1]) { valid_indices[valid_count] = 1; valid_count += 1; };
    $if (is_valid[2]) { valid_indices[valid_count] = 2; valid_count += 1; };
    return valid_count;
}

inline uint2 edge_edge_type(
    const Vec4f &bary, 
    luisa::uint2& valid_indices1,
    luisa::uint2& valid_indices2
    ) 
{
    Bool4 is_valid = bary == 0.0f;
    uint valid_count1 = 0;
    uint valid_count2 = 0;
    $if (is_valid[0]) { valid_indices1[valid_count1] = 0; valid_count1 += 1; };
    $if (is_valid[1]) { valid_indices1[valid_count1] = 1; valid_count1 += 1; };
    $if (is_valid[2]) { valid_indices2[valid_count2] = 0; valid_count2 += 1; };
    $if (is_valid[2]) { valid_indices2[valid_count2] = 1; valid_count2 += 1; };
    return makeUint2(valid_count1, valid_count2);
}



}


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

// const float EPSILON = 1e-8;
// const float FLT_MAX = 1e8;

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
    if (d > Epsilon) {
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
        float di = Float_max;
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

inline auto point_point_distance_squared_unclassified(
    const Vec3f &p0, 
    const Vec3f &p1) 
{
    return (p0 - p1).squaredNorm();
}
inline auto point_edge_distance_squared_unclassified(
    const Vec3f &p, 
    const Vec3f &e0,
    const Vec3f &e1
) 
{
    return (e0 - p).cross(e1 - p).squaredNorm() / (e1 - e0).squaredNorm();
}

} // namespace host_distance

namespace host_distance
{

inline uint point_triangle_type(
    const Vec3f &bary, 
    luisa::uint3& valid_indices
    ) 
{
    // Bool3 is_valid = bary == 0.0f;
    uint valid_count = 0;

    if (bary[0] == 0.0f) { valid_indices[valid_count] = 0; valid_count += 1; };
    if (bary[1] == 0.0f) { valid_indices[valid_count] = 1; valid_count += 1; };
    if (bary[2] == 0.0f) { valid_indices[valid_count] = 2; valid_count += 1; };
    return valid_count;
}

inline uint2 edge_edge_type(
    const Vec4f &bary, 
    luisa::uint2& valid_indices1,
    luisa::uint2& valid_indices2
    ) 
{
    // Bool4 is_valid = bary == 0.0f;
    uint valid_count1 = 0;
    uint valid_count2 = 0;
    if (bary[0] == 0.0f) { valid_indices1[valid_count1] = 0; valid_count1 += 1; };
    if (bary[1] == 0.0f) { valid_indices1[valid_count1] = 1; valid_count1 += 1; };
    if (bary[2] == 0.0f) { valid_indices2[valid_count2] = 0; valid_count2 += 1; };
    if (bary[2] == 0.0f) { valid_indices2[valid_count2] = 1; valid_count2 += 1; };
    return makeUint2(valid_count1, valid_count2);
}

} // namespace host_distance

} // namespace lcsv

#endif
