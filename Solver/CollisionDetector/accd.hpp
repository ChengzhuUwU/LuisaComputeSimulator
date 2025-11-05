#pragma once

#include <luisa/luisa-compute.h>
#include "CollisionDetector/distance.hpp"
#include "Core/float_nxn.h"
#include "Core/scalar.h"
#include "SimulationCore/simulation_type.h"
#include "SimulationCore/simulation_data.h"
#include <vector>
#include <string>


namespace lcs
{

namespace accd
{

    using Mat2x2f = luisa::compute::Float2x2;
    using Mat2x3f = luisa::compute::Float3x2;
    using Mat3x2f = luisa::compute::Float2x3;
    using Mat3x4f = luisa::compute::Float4x3;
    using Vec2f   = luisa::compute::Float2;
    using Vec3f   = luisa::compute::Float3;
    using Vec4f   = luisa::compute::Float4;
    using Float   = luisa::compute::Float;


    inline void centerize(Mat3x4f& x)
    {
        Vec3f      mov   = makeFloat3Var(0.0f);
        Var<float> scale = 0.25f;
        for (int k = 0; k < 4; k++)
        {
            mov += scale * x.cols[k];
        }
        for (int k = 0; k < 4; k++)
        {
            x.cols[k] -= mov;
        }
    }

    constexpr float ccd_reduction        = 0.01f;
    constexpr float line_search_max_t    = 1.25f;
    constexpr bool  print_ccd_iter_count = false;

    template <typename F>
    inline Var<float> ccd_helper(
        const Mat3x4f& x0, const Mat3x4f& dx, const Var<float> u_max, F square_dist_func, const Var<float> offset)
    {
        Var<float> toi        = 0.0f;
        Var<float> max_t      = line_search_max_t;
        Var<float> eps        = ccd_reduction * (sqrt_scalar(square_dist_func(x0)) - offset);
        Var<float> target     = eps + offset;
        Var<float> eps_sqr    = eps * eps;
        Var<float> inv_u_max  = 1.0f / u_max;
        Var<uint>  iter_count = 0;
        $while(true)
        {
            $if(iter_count > 10000)
            {
                // toi = 0.01f;
                luisa::compute::device_assert(false, "CCD iteration not converged in 10000 iteration");
                // luisa::compute::device_log("x0 = {}, offset = {}, u_max = {}", x0, offset, u_max);
                $break;
            };
            // if constexpr (print_ccd_iter_count)
            iter_count += 1;

            Var<float> d2             = square_dist_func(add(x0, mult(toi, dx)));
            Var<float> d_minus_target = (d2 - target * target) / (sqrt_scalar(d2) + target);
            $if((max_t - toi) * u_max < d_minus_target - eps)
            {
                toi = max_t;
                $break;
            }
            $elif(toi > 0.0f & d_minus_target * d_minus_target < eps_sqr)
            {
                $break;
            };
            Var<float> toi_next = toi + d_minus_target * inv_u_max;
            $if(toi_next != toi)
            {
                toi = toi_next;
            }
            $else
            {
                $break;
            };
            $if(toi > max_t)
            {
                toi = max_t;
                $break;
            };
        };
        if constexpr (print_ccd_iter_count)
            $if(iter_count != 1)
            {
                luisa::compute::device_log("CCD iter for {}, toi = {}", iter_count, toi);
            };

        // $if(toi != accd::line_search_max_t)
        // {
        //     luisa::compute::device_log(
        //         "CCD detect toi at {}: init dist = {}, offset = {}", toi, sqrt_scalar(square_dist_func(x0)), offset);
        // };
        luisa::compute::device_assert(toi > 0.0f);
        return toi;
    }

    struct EdgeEdgeSquaredDist
    {
        inline Var<float> operator()(const Mat3x4f& x)
        {
            const Vec3f& p0 = x.cols[0];
            const Vec3f& p1 = x.cols[1];
            const Vec3f& q0 = x.cols[2];
            const Vec3f& q1 = x.cols[3];
            return distance::edge_edge_distance_squared_unclassified(p0, p1, q0, q1);
        }
    };

    struct PointTriangleSquaredDist
    {
        inline Var<float> operator()(const Mat3x4f& x)
        {
            const Vec3f& p  = x.cols[0];
            const Vec3f& t0 = x.cols[1];
            const Vec3f& t1 = x.cols[2];
            const Vec3f& t2 = x.cols[3];
            return distance::point_triangle_distance_squared_unclassified(p, t0, t1, t2);
        }
    };

    inline Var<float> max_relative_u(const Mat3x4f& u)
    {
        Var<float> max_u = 0.0f;
        for (int i = 0; i < 4; i++)
        {
            for (int j = i + 1; j < 4; j++)
            {
                Vec3f du = u.cols[i] - u.cols[j];
                max_u    = max_scalar(max_u, length_squared_vec(du));
            }
        }
        return luisa::compute::sqrt(max_u);
    }

    inline Var<float> point_triangle_ccd(const Vec3f& p0,
                                         const Vec3f& p1,
                                         const Vec3f& t00,
                                         const Vec3f& t01,
                                         const Vec3f& t02,
                                         const Vec3f& t10,
                                         const Vec3f& t11,
                                         const Vec3f& t12,
                                         Float        offset)
    {
        Vec3f   dp  = p1 - p0;
        Vec3f   dt0 = t10 - t00;
        Vec3f   dt1 = t11 - t01;
        Vec3f   dt2 = t12 - t02;
        Mat3x4f x0  = makeFloat4x3(p0, t00, t01, t02);
        Mat3x4f dx  = makeFloat4x3(dp, dt0, dt1, dt2);
        centerize(x0);
        centerize(dx);
        Var<float> u_max = max_relative_u(dx);
        Var<float> toi   = line_search_max_t;
        $if(u_max != 0.0f)
        {
            PointTriangleSquaredDist dist_func;
            toi = ccd_helper(x0, dx, u_max, dist_func, offset);
        };
        return toi;
    }

    inline Var<float> edge_edge_ccd(const Vec3f& ea00,
                                    const Vec3f& ea01,
                                    const Vec3f& eb00,
                                    const Vec3f& eb01,
                                    const Vec3f& ea10,
                                    const Vec3f& ea11,
                                    const Vec3f& eb10,
                                    const Vec3f& eb11,
                                    Float        offset)
    {
        Vec3f   dea0 = ea10 - ea00;
        Vec3f   dea1 = ea11 - ea01;
        Vec3f   deb0 = eb10 - eb00;
        Vec3f   deb1 = eb11 - eb01;
        Mat3x4f x0   = makeFloat4x3(ea00, ea01, eb00, eb01);
        Mat3x4f dx   = makeFloat4x3(dea0, dea1, deb0, deb1);
        centerize(x0);
        centerize(dx);
        Var<float> u_max = max_relative_u(dx);
        Var<float> toi   = line_search_max_t;
        $if(u_max != 0.0f)
        {
            EdgeEdgeSquaredDist dist_func;
            toi = ccd_helper(x0, dx, u_max, dist_func, offset);
        };
        return toi;
    }

}  // namespace accd

namespace host_accd
{

    using namespace host_distance;

    // Name: ACCD Reduction Factor
    // Recommended Range: 1e-2 to 0.5
    // Description:
    // ACCD needs some small number to determine that the gap distance is close enough to the surface.
    // This factor is multiplied to the initial gap to set this threshold.
    constexpr float ccd_reduction = 0.01f;

    // Name: Extended Line Search Maximum Time
    // Recommended Range: 1.25 to 1.75
    // Description:
    // Continuous Collision Detection (CCD) is used to determine the time of impact (TOI),
    // but if we advance the time to the actual TOI, contact gaps can be nearly zero.
    // Such a small gap can cause the solver to diverge, so we extend the time to the TOI by this
    // factor, and the actual TOI is recaled by dividing by this factor.
    // For example, if the actual TOI is 1.0 and this value is 1.25, the actual TOI is 1.0/1.25.
    constexpr float line_search_max_t = 1.25f;

    template <class T>
    inline void centerize(SMat<T, 3, 4>& x)
    {
        SVec<T, 3> mov = SVec<T, 3>::Zero();
        T          scale(0.25f);
        for (int k = 0; k < 4; k++)
        {
            mov += scale * x.col(k);
        }
        for (int k = 0; k < 4; k++)
        {
            x.col(k) -= mov;
        }
    }

    constexpr bool print_ccd_iter_count = false;
    template <typename F>
    inline float ccd_helper(const Mat3x4f& x0, const Mat3x4f& dx, float u_max, F square_dist_func, float offset)
    {
        float toi        = 0.0f;
        float max_t      = line_search_max_t;
        float eps        = ccd_reduction * (sqrtf(square_dist_func(x0)) - offset);
        float target     = eps + offset;
        float eps_sqr    = eps * eps;
        float inv_u_max  = 1.0f / u_max;
        uint  iter_count = 0;
        while (true)
        {
            if (iter_count > 10000)
            {
                LUISA_ASSERT(false, "CCD iteration not converged in 1000 iteration!");
                break;
            }
            iter_count += 1;

            // Removed redundant increment for print_ccd_iter_count
            float d2             = square_dist_func(x0 + toi * dx);
            float d_minus_target = (d2 - target * target) / (sqrtf(d2) + target);
            if ((max_t - toi) * u_max < d_minus_target - eps)
            {
                toi = max_t;
                break;
            }
            else if (toi > 0.0f && d_minus_target * d_minus_target < eps_sqr)
            {
                break;
            }
            float toi_next = toi + d_minus_target * inv_u_max;
            if (toi_next != toi)
            {
                toi = toi_next;
            }
            else
            {
                break;
            }
            if (toi > max_t)
            {
                toi = max_t;
                break;
            }
        }
        if constexpr (print_ccd_iter_count)
            if (iter_count != 1)
                LUISA_INFO("CCD iter for {}, toi = {}", iter_count, toi);
        assert(toi > 0.0f);
        return toi;
    }

    struct EdgeEdgeSquaredDist
    {
        inline float operator()(const Mat3x4f& x)
        {
            const Vec3f& p0 = x.col(0);
            const Vec3f& p1 = x.col(1);
            const Vec3f& q0 = x.col(2);
            const Vec3f& q1 = x.col(3);
            return host_distance::edge_edge_distance_squared_unclassified(p0, p1, q0, q1);
        }
    };

    struct PointTriangleSquaredDist
    {
        inline float operator()(const Mat3x4f& x)
        {
            const Vec3f& p  = x.col(0);
            const Vec3f& t0 = x.col(1);
            const Vec3f& t1 = x.col(2);
            const Vec3f& t2 = x.col(3);
            return host_distance::point_triangle_distance_squared_unclassified(p, t0, t1, t2);
        }
    };

    inline float max_relative_u(const Mat3x4f& u)
    {
        float max_u = 0.0f;
        for (int i = 0; i < 4; i++)
        {
            for (int j = i + 1; j < 4; j++)
            {
                Vec3f du = u.col(i) - u.col(j);
                max_u    = std::max(max_u, du.squaredNorm());
            }
        }
        return sqrt(max_u);
    }

    inline float point_triangle_ccd(const Vec3f& p0,
                                    const Vec3f& p1,
                                    const Vec3f& t00,
                                    const Vec3f& t01,
                                    const Vec3f& t02,
                                    const Vec3f& t10,
                                    const Vec3f& t11,
                                    const Vec3f& t12,
                                    float        offset)
    {
        Vec3f   dp  = p1 - p0;
        Vec3f   dt0 = t10 - t00;
        Vec3f   dt1 = t11 - t01;
        Vec3f   dt2 = t12 - t02;
        Mat3x4f x0;
        Mat3x4f dx;
        x0 << p0, t00, t01, t02;
        centerize(x0);
        dx << dp, dt0, dt1, dt2;
        centerize(dx);
        float u_max = max_relative_u(dx);
        if (u_max)
        {
            PointTriangleSquaredDist dist_func;
            return ccd_helper(x0, dx, u_max, dist_func, offset);
        }
        else
        {
            return line_search_max_t;
        }
    }

    inline float edge_edge_ccd(const Vec3f& ea00,
                               const Vec3f& ea01,
                               const Vec3f& eb00,
                               const Vec3f& eb01,
                               const Vec3f& ea10,
                               const Vec3f& ea11,
                               const Vec3f& eb10,
                               const Vec3f& eb11,
                               float        offset)
    {
        Vec3f   dea0 = ea10 - ea00;
        Vec3f   dea1 = ea11 - ea01;
        Vec3f   deb0 = eb10 - eb00;
        Vec3f   deb1 = eb11 - eb01;
        Mat3x4f x0;
        Mat3x4f dx;
        x0 << ea00, ea01, eb00, eb01;
        centerize(x0);
        dx << dea0, dea1, deb0, deb1;
        centerize(dx);
        float u_max = max_relative_u(dx);
        if (u_max)
        {
            EdgeEdgeSquaredDist dist_func;
            return ccd_helper(x0, dx, u_max, dist_func, offset);
        }
        else
        {
            return line_search_max_t;
        }
    }

}  // namespace host_accd

}  // namespace lcs