#pragma once

#include "CollisionDetector/cipc_kernel.hpp"
#include "Core/scalar.h"
#include "Core/xbasic_types.h"
#include "luisa/dsl/sugar.h"
#include "distance.hpp"

namespace lcs
{

namespace Friction
{
    namespace GaussNewton
    {
        //
        // Modified from [https://github.com/st-tech/ppf-contact-solver/blob/main/src/cpp/energy/model/friction.hpp]
        //
        constexpr float friction_eps = 1e-5f;

        template <typename T>
        inline auto get_projection(const T& normal)
        {
            return Identity3x3 - outer_product(normal, normal);
        }
        inline std::pair<float, float3x3> get_friction_lambda_P(
            const float3& grad_contact, const float3& dx, const float3& normal, const float mu, const float min_dx)
        {
            float    contact = -dot(grad_contact, normal);
            float3x3 P       = get_projection(normal);

            float lambda;
            if (mu > 0.0f)
            {
                float denom = length_squared_vec(P * dx);
                if (denom > 0.0f)
                {
                    lambda = mu * contact / luisa::max(min_dx, luisa::sqrt(denom));
                }
                else
                {
                    lambda = mu * contact / min_dx;
                }
            }
            else
            {
                lambda = 0.0f;
            }
            return std::make_pair(lambda, P);
        }
        inline std::pair<Var<float>, Var<float3x3>> get_friction_lambda_P(const Var<float3>& grad_contact,
                                                                          const Var<float3>& dx,
                                                                          const Var<float3>& normal,
                                                                          const Var<float>   mu,
                                                                          const Var<float>   min_dx)
        {
            Var<float>    contact = -luisa::compute::dot(grad_contact, normal);
            Var<float3x3> P       = get_projection(normal);

            Var<float> lambda;
            $if(mu > 0.0f)
            {
                Var<float> denom = length_squared_vec(P * dx);
                $if(denom > 0.0f)
                {
                    lambda = mu * contact / luisa::compute::max(min_dx, luisa::compute::sqrt(denom));
                }
                $else
                {
                    lambda = mu * contact / min_dx;
                };
            }
            $else
            {
                lambda = 0.0f;
            };
            return std::make_pair(lambda, P);
        }
        inline std::pair<float, float3x3> get_friction_lambda_P(
            const float dbdd, const float3& dx, const float3& normal, const float mu, const float min_dx)
        {
            float    contact = -dbdd;
            float3x3 P       = get_projection(normal);

            float lambda;
            if (mu > 0.0f)
            {
                float denom = length_squared_vec(P * dx);
                if (denom > 0.0f)
                {
                    lambda = mu * contact / luisa::max(min_dx, luisa::sqrt(denom));
                }
                else
                {
                    lambda = mu * contact / min_dx;
                }
            }
            else
            {
                lambda = 0.0f;
            }
            return std::make_pair(lambda, P);
        }
        inline std::pair<Var<float>, Var<float3x3>> get_friction_lambda_P(const Var<float>& dbdd,  // First derivative of energy to distance
                                                                          const Var<float3>& dx,
                                                                          const Var<float3>& normal,
                                                                          const Var<float>   mu,
                                                                          const Var<float>   min_dx)
        {
            Var<float>    contact = -dbdd;
            Var<float3x3> P       = get_projection(normal);

            Var<float> lambda;
            $if(mu > 0.0f)
            {
                Var<float> denom = length_squared_vec(P * dx);
                $if(denom > 0.0f)
                {
                    lambda = mu * contact / luisa::compute::max(min_dx, luisa::compute::sqrt(denom));
                }
                $else
                {
                    lambda = mu * contact / min_dx;
                };
            }
            $else
            {
                lambda = 0.0f;
            };
            return std::make_pair(lambda, P);
        }

        template <typename PairType, typename Vec>
        inline auto compute_gradient_hessian(const PairType& lambda_P, const Vec& dx)
        {
            const auto& lambda   = lambda_P.first;
            const auto& P        = lambda_P.second;
            auto        gradient = lambda * (P * dx);
            auto        hessian  = lambda * P;
            return std::make_pair(gradient, hessian);
        }
        template <typename FloatType, typename Vec>
        inline auto compute_gradient_hessian(const FloatType& lambda, const Vec& normal, const Vec& dx)
        {
            const auto P        = Identity3x3 - outer_product(normal, normal);
            auto       gradient = lambda * (P * dx);
            auto       hessian  = lambda * P;
            return std::make_pair(gradient, hessian);
        }
        template <typename FloatType, typename Vec>
        inline auto compute_hessian(const FloatType& lambda, const Vec& normal)
        {
            const auto P       = Identity3x3 - outer_product(normal, normal);
            auto       hessian = lambda * P;
            return hessian;
        }
    }  // namespace GaussNewton
}  // namespace Friction

namespace Friction
{
    // C1 clamping
    inline void f0(float x2, float epsvh, float& f0)
    {
        if (x2 >= epsvh * epsvh)
        {
            //tex: $$y$$
            f0 = sqrt_scalar(x2);
        }
        else
        {
            //tex: $$\frac{y^{2}}{\epsilon_{x}} + \frac{1}{3 \epsilon_{x}} - \frac{y^{3}}{3 \epsilon_{x}^{2}}$$
            f0 = x2 * (-sqrt_scalar(x2) / 3.0f + epsvh) / (epsvh * epsvh) + epsvh / 3.0f;
        }
    }
    inline void f0(Var<float> x2, Var<float> epsvh, Var<float>& f0)
    {
        $if(x2 >= epsvh * epsvh)
        {
            //tex: $$y$$
            f0 = sqrt_scalar(x2);
        }
        $else
        {
            //tex: $$\frac{y^{2}}{\epsilon_{x}} + \frac{1}{3 \epsilon_{x}} - \frac{y^{3}}{3 \epsilon_{x}^{2}}$$
            f0 = x2 * (-sqrt_scalar(x2) / 3.0f + epsvh) / (epsvh * epsvh) + epsvh / 3.0f;
        };
    }

    inline void f1_div_rel_dx_norm(float x2, float epsvh, float& result)
    {
        if (x2 >= epsvh * epsvh)
        {
            //tex: $$ \frac{1}{y}$$
            result = 1 / sqrt_scalar(x2);
        }
        else
        {
            //tex: $$ \frac{2 \epsilon_{x} - y}{ \epsilon_{x}^{2}}$$
            result = (-sqrt_scalar(x2) + 2.0f * epsvh) / (epsvh * epsvh);
        }
    }
    inline void f1_div_rel_dx_norm(Var<float> x2, Var<float> epsvh, Var<float>& result)
    {
        $if(x2 >= epsvh * epsvh)
        {
            //tex: $$ \frac{1}{y}$$
            result = 1.0f / sqrt_scalar(x2);
        }
        $else
        {
            //tex: $$ \frac{2 \epsilon_{x} - y}{ \epsilon_{x}^{2}}$$
            result = (-sqrt_scalar(x2) + 2.0f * epsvh) / (epsvh * epsvh);
        };
    }


    template <typename T>
    inline void f2_term(T x2, T epsvh, T& term)
    {
        term = -1 / (epsvh * epsvh);
        // same for x2 >= epsvh * epsvh for C1 clamped friction
    }

    template <typename T, typename Vec2>
    inline T friction_energy(T mu, T lambda, T eps_vh, Vec2 tan_rel_x)
    {
        T f0_val;
        f0(dot_vec(tan_rel_x, tan_rel_x), eps_vh, f0_val);
        return mu * lambda * f0_val;
    }

    template <typename T, typename Vec2>
    inline void friction_gradient(Vec2& G2, T mu, T lambda, T eps_vh, Vec2 tan_rel_x)
    {
        T f1_val;
        f1_div_rel_dx_norm(dot_vec(tan_rel_x), eps_vh, f1_val);
        G2 = mu * lambda * f1_val * tan_rel_x;
    }
}  // namespace Friction

namespace Friction
{
    namespace basis
    {
        inline void tangent_rel_dx(const float3& rel_dx, const float2x3& basis, float2& tan_rel_dx)
        {
            tan_rel_dx[0] = luisa::dot(basis.cols[0], rel_dx);
            tan_rel_dx[1] = luisa::dot(basis.cols[1], rel_dx);
        }
        inline void tangent_rel_dx(const Var<float3>& rel_dx, const Var<float2x3>& basis, Var<float2>& tan_rel_dx)
        {
            tan_rel_dx[0] = luisa::compute::dot(basis.cols[0], rel_dx);
            tan_rel_dx[1] = luisa::compute::dot(basis.cols[1], rel_dx);
        }

        inline void point_triangle_tangent_basis(luisa::float2&       beta,
                                                 float2x3&            basis,
                                                 const luisa::float3& p,
                                                 const luisa::float3& t0,
                                                 const luisa::float3& t1,
                                                 const luisa::float3& t2)
        {
            luisa::float3 v12 = t1 - t0;
            luisa::float3 v13 = t2 - t0;
            basis.cols[0]     = normalize_vec(v12);
            basis.cols[1]     = normalize_vec(cross_vec(cross_vec(v12, v13), v12));

            luisa::float2x2 BBT = luisa::make_float2x2(
                luisa::dot(v12, v12), luisa::dot(v12, v13), luisa::dot(v13, v12), luisa::dot(v13, v13));
            luisa::float2 rhs = luisa::make_float2(luisa::dot(v12, p - t0), luisa::dot(v13, p - t0));
            beta              = luisa::inverse(BBT) * rhs;
        }
        inline void point_triangle_tangent_basis(  // out
            Var<luisa::float2>&       beta,
            Var<float2x3>&            basis,
            const Var<luisa::float3>& p,
            const Var<luisa::float3>& t0,
            const Var<luisa::float3>& t1,
            const Var<luisa::float3>& t2)
        {
            Var<luisa::float3> v12 = t1 - t0;
            Var<luisa::float3> v13 = t2 - t0;
            basis.cols[0]          = normalize_vec(v12);
            basis.cols[1]          = normalize_vec(cross_vec(cross_vec(v12, v13), v12));

            Var<float2x2>      BBT = luisa::compute::make_float2x2(luisa::compute::dot(v12, v12),
                                                              luisa::compute::dot(v12, v13),
                                                              luisa::compute::dot(v13, v12),
                                                              luisa::compute::dot(v13, v13));
            Var<luisa::float2> rhs = luisa::compute::make_float2(luisa::compute::dot(v12, p - t0),
                                                                 luisa::compute::dot(v13, p - t0));
            beta                   = luisa::compute::inverse(BBT) * rhs;
        }
        inline void point_triangle_tan_rel_dx(const luisa::float2& beta,
                                              const float2x3&      basis,
                                              float2&              tan_rel_dx,
                                              const luisa::float3& dP,
                                              const luisa::float3& dT0,
                                              const luisa::float3& dT1,
                                              const luisa::float3& dT2)
        {
            float3 rel_dx = dP - (dT0 + beta[0] * (dT1 - dT0) + beta[1] * (dT2 - dT0));
            tangent_rel_dx(rel_dx, basis, tan_rel_dx);
        }

    }  // namespace basis
}  // namespace Friction

namespace Friction
{
    inline void PT_friction_basis(
        // out
        float&         f,
        luisa::float2& beta,
        float2x3&      basis,
        luisa::float2& tan_rel_dx,
        // in
        float                kappa,
        float                d_hat,
        float                thickness,
        const luisa::float3& prev_P,
        const luisa::float3& prev_T0,
        const luisa::float3& prev_T1,
        const luisa::float3& prev_T2,
        const luisa::float3& P,
        const luisa::float3& T0,
        const luisa::float3& T1,
        const luisa::float3& T2)
    {
        using namespace distance;
        // using namespace friction;

        // using the prev values to compute normal force
        basis::point_triangle_tangent_basis(beta, basis, prev_P, prev_T0, prev_T1, prev_T2);

        float prev_D;
        point_triangle_distance_squared_unclassified(prev_P, prev_T0, prev_T1, prev_T2);

        float prev_d = sqrt_scalar(prev_D);
        f            = -ipc::barrier_first_derivative(prev_d - thickness, d_hat);

        luisa::float3 dP  = P - prev_P;
        luisa::float3 dT0 = T0 - prev_T0;
        luisa::float3 dT1 = T1 - prev_T1;
        luisa::float3 dT2 = T2 - prev_T2;

        basis::point_triangle_tan_rel_dx(beta, basis, tan_rel_dx, dP, dT0, dT1, dT2);
    }
}  // namespace Friction


}  // namespace lcs