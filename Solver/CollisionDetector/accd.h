#pragma once

#include <luisa/luisa-compute.h>
#include "CollisionDetector/distance.hpp"
#include "Core/float_nxn.h"
#include "Core/scalar.h"
#include "SimulationCore/simulation_type.h"
#include "SimulationCore/simulation_data.h"
#include <vector>
#include <string>


namespace lcsv {

namespace accd {

using Mat2x2f = luisa::compute::Float2x2;
using Mat2x3f = luisa::compute::Float3x2;
using Mat3x2f = luisa::compute::Float2x3;
using Mat3x4f = luisa::compute::Float4x3;
using Vec2f = luisa::compute::Float2;
using Vec3f = luisa::compute::Float3;
using Vec4f = luisa::compute::Float4;


inline void centerize(Mat3x4f &x) {
    Vec3f mov = makeFloat3(0.0f);
    Var<float> scale(0.25f);
    for (int k = 0; k < 4; k++) {
        mov += scale * x.cols[k];
    }
    for (int k = 0; k < 4; k++) {
        x.cols[k] -= mov;
    }
}

constexpr float ccd_reduction = 0.01f;
constexpr float line_search_max_t = 1.25f;

template <typename F>
inline Var<float> ccd_helper(
    const Mat3x4f &x0, 
    const Mat3x4f &dx, 
    const Var<float> u_max,
    F square_dist_func, 
    const Var<float> offset) 
{
    Var<float> toi = 0.0f;
    Var<float> max_t = line_search_max_t;
    Var<float> eps = ccd_reduction * (sqrt_scalar(square_dist_func(x0)) - offset);
    Var<float> target = eps + offset;
    Var<float> eps_sqr = eps * eps;
    Var<float> inv_u_max = 1.0f / u_max;
    $while (true) {
        Var<float> d2 = square_dist_func(add(x0, mult(toi, dx)));
        Var<float> d_minus_target = (d2 - target * target) / (sqrt_scalar(d2) + target);
        $if ((max_t - toi) * u_max < d_minus_target - eps) {
            toi = max_t;
            $break;
        } $elif (toi > 0.0f & d_minus_target * d_minus_target < eps_sqr) {
            $break;
        };
        Var<float> toi_next = toi + d_minus_target * inv_u_max;
        $if (toi_next != toi) {
            toi = toi_next;
        } $else {
            $break;
        };
        $if (toi > max_t) {
            toi = max_t;
            $break;
        };
    };
    assert(toi > 0.0f);
    return toi;
}

struct EdgeEdgeSquaredDist {
    inline Var<float> operator()(const Mat3x4f &x) {
        const Vec3f &p0 = x.cols[0];
        const Vec3f &p1 = x.cols[1];
        const Vec3f &q0 = x.cols[2];
        const Vec3f &q1 = x.cols[3];
        return distance::edge_edge_distance_squared_unclassified(p0, p1, q0,
                                                                 q1);
    }
};

struct PointTriangleSquaredDist {
    inline Var<float> operator()(const Mat3x4f &x) {
        const Vec3f &p =  x.cols[0];
        const Vec3f &t0 = x.cols[1];
        const Vec3f &t1 = x.cols[2];
        const Vec3f &t2 = x.cols[3];
        return distance::point_triangle_distance_squared_unclassified(p, t0, t1,
                                                                      t2);
    }
};

inline Var<float> max_relative_u(const Mat3x4f &u) {
    Var<float> max_u = 0.0f;
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 4; j++) {
            Vec3f du = u.cols[i] - u.cols[j];
            max_u = max_scalar(max_u, length_squared_vec(du));
        }
    }
    return sqrt(max_u);
}

inline Var<float> point_triangle_ccd(const Vec3f &p0,  const Vec3f &p1,
                               const Vec3f &t00, const Vec3f &t01,
                               const Vec3f &t02, const Vec3f &t10,
                               const Vec3f &t11, const Vec3f &t12,
                               float offset) {
    Vec3f dp = p1 - p0;
    Vec3f dt0 = t10 - t00;
    Vec3f dt1 = t11 - t01;
    Vec3f dt2 = t12 - t02;
    Mat3x4f x0 = makeFloat4x3(p0, t00, t01, t02);
    Mat3x4f dx = makeFloat4x3(dp, dt0, dt1, dt2);
    centerize(x0);
    centerize(dx);
    Var<float> u_max = max_relative_u(dx);
    Var<float> toi = line_search_max_t;
    $if (u_max != 0.0f) {
        PointTriangleSquaredDist dist_func;
        toi = ccd_helper(x0, dx, u_max, dist_func, offset);
    } ;
    return toi;
}

inline Var<float> edge_edge_ccd(const Vec3f &ea00, const Vec3f &ea01,
                               const Vec3f &eb00, const Vec3f &eb01,
                               const Vec3f &ea10, const Vec3f &ea11,
                               const Vec3f &eb10, const Vec3f &eb11,
                               float offset) {
    Vec3f dea0 = ea10 - ea00;
    Vec3f dea1 = ea11 - ea01;
    Vec3f deb0 = eb10 - eb00;
    Vec3f deb1 = eb11 - eb01;
    Mat3x4f x0 = makeFloat4x3(ea00, ea01, eb00, eb01);
    Mat3x4f dx = makeFloat4x3(dea0, dea1, deb0, deb1);
    centerize(x0);
    centerize(dx);
    Var<float> u_max = max_relative_u(dx);
    Var<float> toi = line_search_max_t;
    $if (u_max != 0.0f) {
        EdgeEdgeSquaredDist dist_func;
        toi = ccd_helper(x0, dx, u_max, dist_func, offset);
    };
    return toi;
}

} // namespace accd



class AccdDetector
{


private:
    CollisionDataCCD<luisa::compute::Buffer>* ccd_data;

};



} // namespace lcsv