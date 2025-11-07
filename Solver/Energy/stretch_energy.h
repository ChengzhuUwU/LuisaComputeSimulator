// Modified from [https://github.com/st-tech/ppf-contact-solver/blob/main/src/cpp/energy/model/baraffwitkin.hpp]

#pragma once

#include "Core/float_n.h"
#include "Core/float_nxn.h"
#include "Core/lc_to_eigen.h"
#include "Core/svd_2x2.h"
#include <luisa/luisa-compute.h>

namespace lcs
{

namespace StretchEnergy
{
    namespace detail
    {
        template <typename T>
        inline T sqr(T x)
        {
            return x * x;
        }

        inline constexpr float2x3 make_diff_mat3x2()
        {
            float2x3 result;
            result.set_zero();
            // x2 - x1
            result[0][0] = float(-1.0f);
            result[0][1] = float(1.0f);
            // x3 - x1
            result[1][0] = float(-1.0f);
            result[1][2] = float(1.0f);
            return result;
        }
        inline Var<float2x3> make_diff_mat3x2_Var()
        {
            Var<float2x3> result;
            result->set_zero();
            // x2 - x1
            result.cols[0][0] = -1.0f;
            result.cols[0][1] = 1.0f;
            // x3 - x1
            result.cols[1][0] = -1.0f;
            result.cols[1][2] = 1.0f;
            return result;
        }

        inline float6 flatten(const float2x3& F)
        {
            float6 R;
            R.vec[0] = F.cols[0];
            R.vec[1] = F.cols[1];
            return R;
        }
        inline Var<float6> flatten(const Var<float2x3>& F)
        {
            Var<float6> R;
            R.vec[0] = F.cols[0];
            R.vec[1] = F.cols[1];
            return R;
        }

        inline LargeMatrix<9, 6> get_dFdx(const luisa::float2x2& InverseDm)
        {
            const float d0 = InverseDm[0][0];
            const float d1 = InverseDm[0][1];
            const float d2 = InverseDm[1][0];
            const float d3 = InverseDm[1][1];
            const float s0 = d0 + d1;
            const float s1 = d2 + d3;

            lcs::LargeMatrix<9, 6> result;
            for (int i = 0; i < 3; i++)
            {
                result.scalar(i, i)     = -s0;
                result.scalar(i, i + 3) = -s1;
            }
            for (int i = 0; i < 3; i++)
            {
                result.scalar(i + 3, i)     = d0;
                result.scalar(i + 3, i + 3) = d2;
            }
            for (int i = 0; i < 3; i++)
            {
                result.scalar(i + 6, i)     = d1;
                result.scalar(i + 6, i + 3) = d3;
            }
            return result;
        }
        inline LargeMatrix<6, 9> get_dFdx_T(const luisa::float2x2& InverseDm)
        {
            lcs::LargeMatrix<6, 9> dfdx_T = lcs::LargeMatrix<6, 9>::zero();

            const float d0 = InverseDm[0][0];
            const float d1 = InverseDm[0][1];
            const float d2 = InverseDm[1][0];
            const float d3 = InverseDm[1][1];
            const float s0 = d0 + d1;
            const float s1 = d2 + d3;

            dfdx_T.scalar<0, 0>() = -s0;
            dfdx_T.scalar<3, 0>() = -s1;
            dfdx_T.scalar<1, 1>() = -s0;
            dfdx_T.scalar<4, 1>() = -s1;
            dfdx_T.scalar<2, 2>() = -s0;
            dfdx_T.scalar<5, 2>() = -s1;
            dfdx_T.scalar<0, 3>() = d0;
            dfdx_T.scalar<3, 3>() = d2;
            dfdx_T.scalar<1, 4>() = d0;
            dfdx_T.scalar<4, 4>() = d2;
            dfdx_T.scalar<2, 5>() = d0;
            dfdx_T.scalar<5, 5>() = d2;
            dfdx_T.scalar<0, 6>() = d1;
            dfdx_T.scalar<3, 6>() = d3;
            dfdx_T.scalar<1, 7>() = d1;
            dfdx_T.scalar<4, 7>() = d3;
            dfdx_T.scalar<2, 8>() = d1;
            dfdx_T.scalar<5, 8>() = d3;
            return dfdx_T;
        }
        inline Var<LargeMatrix<6, 9>> get_dFdx_T(const Var<luisa::float2x2>& InverseDm)
        {
            Var<LargeMatrix<6, 9>> dfdx_T;
            dfdx_T->set_zero();

            const auto d0 = InverseDm[0][0];
            const auto d1 = InverseDm[0][1];
            const auto d2 = InverseDm[1][0];
            const auto d3 = InverseDm[1][1];
            const auto s0 = d0 + d1;
            const auto s1 = d2 + d3;

            dfdx_T->scalar<0, 0>() = -s0;
            dfdx_T->scalar<3, 0>() = -s1;
            dfdx_T->scalar<1, 1>() = -s0;
            dfdx_T->scalar<4, 1>() = -s1;
            dfdx_T->scalar<2, 2>() = -s0;
            dfdx_T->scalar<5, 2>() = -s1;
            dfdx_T->scalar<0, 3>() = d0;
            dfdx_T->scalar<3, 3>() = d2;
            dfdx_T->scalar<1, 4>() = d0;
            dfdx_T->scalar<4, 4>() = d2;
            dfdx_T->scalar<2, 5>() = d0;
            dfdx_T->scalar<5, 5>() = d2;
            dfdx_T->scalar<0, 6>() = d1;
            dfdx_T->scalar<3, 6>() = d3;
            dfdx_T->scalar<1, 7>() = d1;
            dfdx_T->scalar<4, 7>() = d3;
            dfdx_T->scalar<2, 8>() = d1;
            dfdx_T->scalar<5, 8>() = d3;
            return dfdx_T;
        }

        // dedF * dFdx (6x1 mult 6x9 => 1x9)
        inline luisa::float3x3 convert_force(const float2x3& dedF, const luisa::float2x2& inv_rest2x2)
        {
            const float3x2  g_T    = (make_diff_mat3x2() * inv_rest2x2).transpose();
            const float3x2  dedF_T = dedF.transpose();
            luisa::float3x3 result;
            for (unsigned i = 0; i < 3; ++i)
            {
                for (unsigned dim = 0; dim < 3; ++dim)
                {
                    result[i][dim] = luisa::dot(g_T[i], dedF_T[dim]);
                }
            }
            return result;
        }
        inline float9x9 convert_hessian(const float6x6& d2ed2f, const luisa::float2x2& inv_rest2x2)
        {
            lcs::LargeMatrix<6, 9> dfdx_T = get_dFdx_T(inv_rest2x2);

            float9x9 result;
            result.set_zero();
            for (unsigned i = 0; i < 6; ++i)
            {
                for (unsigned j = 0; j < 6; ++j)
                {
                    result = result
                             + d2ed2f.scalar(j, i) * float9x9::outer_product(dfdx_T.column(i), dfdx_T.column(j));
                }
            }
            return result;  // dfdx.transpose() * d2ed2f * dfdx;
        }
        inline Var<luisa::float3x3> convert_force(const Var<float2x3>& dedF, const Var<luisa::float2x2>& inv_rest2x2)
        {
            const Var<float3x2>  g_T    = transpose(make_diff_mat3x2_Var() * inv_rest2x2);
            const Var<float3x2>  dedF_T = transpose(dedF);
            Var<luisa::float3x3> result;
            for (unsigned i = 0; i < 3; ++i)
            {
                for (unsigned dim = 0; dim < 3; ++dim)
                {
                    result[i][dim] = luisa::compute::dot(g_T.cols[i], dedF_T.cols[dim]);
                }
            }
            return result;
        }
        inline Var<float9x9> convert_hessian(const Var<float6x6>& d2ed2f, const Var<luisa::float2x2>& inv_rest2x2)
        {
            Var<LargeMatrix<6, 9>> dfdx_T = get_dFdx_T(inv_rest2x2);
            Var<float9x9>          result;
            result->set_zero();
            for (unsigned i = 0; i < 6; ++i)
            {
                for (unsigned j = 0; j < 6; ++j)
                {
                    result = result + d2ed2f->scalar(j, i) * outer_product(dfdx_T->column(i), dfdx_T->column(j));
                }
            }
            return result;  // dfdx.transpose() * d2ed2f * dfdx;
        }

        inline float stretch_energy(const float2x3& F, float mu)
        {
            const auto i5u = luisa::dot(F[0], F[0]);
            const auto i5v = luisa::dot(F[1], F[1]);
            return 0.5f * mu * (sqr(luisa::sqrt(i5u) - 1.0f) + sqr(luisa::sqrt(i5v) - 1.0f));
        }
        inline float shear_energy(const float2x3& F, float lmd)
        {
            const auto i6 = luisa::dot(F[0], F[1]);
            return 0.5f * lmd * sqr(i6);
        }
        inline Var<float> stretch_energy(const Var<float2x3>& F, Var<float> mu)
        {
            const auto i5u = luisa::compute::dot(F.cols[0], F.cols[0]);
            const auto i5v = luisa::compute::dot(F.cols[1], F.cols[1]);
            return 0.5f * mu * (sqr(luisa::compute::sqrt(i5u) - 1.0f) + sqr(luisa::compute::sqrt(i5v) - 1.0f));
        }
        inline Var<float> shear_energy(const Var<float2x3>& F, Var<float> lmd)
        {
            const auto i6 = luisa::compute::dot(F.cols[0], F.cols[1]);
            return 0.5f * lmd * sqr(i6);
        }

        inline float2x3 stretch_gradient(const float2x3& F, const float mu)
        {
            const float3& Fu = F.cols[0];
            const float3& Fv = F.cols[1];

            const auto I5u = luisa::dot(Fu, Fu);
            const auto I5v = luisa::dot(Fv, Fv);

            float sqrtI5u    = luisa::sqrt(I5u);
            float sqrtI5v    = luisa::sqrt(I5v);
            float invSqrtI5u = 1.0f / sqrtI5u;
            float invSqrtI5v = 1.0f / sqrtI5v;

            float2x3 result;
            result.cols[0] = (sqrtI5u - 1.0f) * invSqrtI5u * Fu;
            result.cols[1] = (sqrtI5v - 1.0f) * invSqrtI5v * Fv;
            return mu * result;
        }
        inline float2x3 shear_gradient(const float2x3& F, const float lmd)
        {
            float    w = luisa::dot(F.cols[0], F.cols[1]);
            float2x3 result;
            result[0] = w * F.cols[1];
            result[1] = w * F.cols[0];
            return lmd * result;
        }

        inline float6x6 stretch_hessian(const float2x3& F, float mu)
        {
            float6x6 H = float6x6::zero();

            const float3& Fu = F.cols[0];
            const float3& Fv = F.cols[1];

            const auto I5u = luisa::dot(Fu, Fu);
            const auto I5v = luisa::dot(Fv, Fv);

            float sqrtI5u    = luisa::sqrt(I5u);
            float sqrtI5v    = luisa::sqrt(I5v);
            float invSqrtI5u = 1.0f / sqrtI5u;
            float invSqrtI5v = 1.0f / sqrtI5v;

            H.scalar(0, 0) = H.scalar(1, 1) = H.scalar(2, 2) = luisa::max(0.0f, 1.0f - invSqrtI5u);
            H.scalar(3, 3) = H.scalar(4, 4) = H.scalar(5, 5) = luisa::max(0.0f, 1.0f - invSqrtI5v);

            auto fu = F.cols[0] * invSqrtI5u;
            auto fv = F.cols[1] * invSqrtI5v;

            float uCoeff  = (invSqrtI5u < 1.0f) ? invSqrtI5u : 1.0f;
            float vCoeff  = (invSqrtI5v < 1.0f) ? invSqrtI5v : 1.0f;
            H.block(0, 0) = H.block(0, 0) + uCoeff * outer_product(fu, fu);
            H.block(1, 1) = H.block(1, 1) + vCoeff * outer_product(fv, fv);
            return mu * H;
        }
        inline float6x6 shear_hessian(const float2x3& F, float mu)
        {
            float6x6 H = float6x6::zero();

            const float3& Fu = F.cols[0];
            const float3& Fv = F.cols[1];

            const float I6     = luisa::dot(Fu, Fv);
            const float signI6 = luisa::sign(I6);

            H.scalar<3, 0>() = H.scalar<4, 1>() = H.scalar<5, 2>() = H.scalar<0, 3>() =
                H.scalar<1, 4>() = H.scalar<2, 5>() = 1.0f;

            const float6 g = flatten(F * luisa::make_float2x2(0, 1, 1, 0));  // F * (a b^T + b a^T)

            const float I2      = luisa::dot(Fu, Fu) + luisa::dot(Fv, Fv);  // F.squaredNorm();
            const float lambda0 = 0.5f * (I2 + luisa::sqrt(I2 * I2 + 12.0f * I6 * I6));

            const float6 q0     = (I6 * H * g + lambda0 * g).normalize();
            float6x6     T      = float6x6::identity();
            T                   = 0.5f * (T + signI6 * H);
            const float6 Tq     = T * q0;
            const float  normTq = Tq.squared_norm();

            H = luisa::abs(I6) * (T - float6x6::outer_product(Tq, Tq) / normTq)
                + lambda0 * float6x6::outer_product(q0, q0);

            return mu * H;
        }


        inline Var<float2x3> stretch_gradient(const Var<float2x3>& F, const Var<float> mu)
        {
            const auto& Fu = F.cols[0];
            const auto& Fv = F.cols[1];

            const auto I5u = luisa::compute::dot(Fu, Fu);
            const auto I5v = luisa::compute::dot(Fv, Fv);

            const auto sqrtI5u    = luisa::compute::sqrt(I5u);
            const auto sqrtI5v    = luisa::compute::sqrt(I5v);
            const auto invSqrtI5u = 1.0f / sqrtI5u;
            const auto invSqrtI5v = 1.0f / sqrtI5v;

            Var<float2x3> result;
            result.cols[0] = (sqrtI5u - 1.0f) * invSqrtI5u * Fu;
            result.cols[1] = (sqrtI5v - 1.0f) * invSqrtI5v * Fv;
            return mu * result;
        }
        inline Var<float2x3> shear_gradient(const Var<float2x3>& F, const Var<float> lmd)
        {
            Var<float>    w = luisa::compute::dot(F.cols[0], F.cols[1]);
            Var<float2x3> result;
            result.cols[0] = w * F.cols[1];
            result.cols[1] = w * F.cols[0];
            return lmd * result;
        }

        inline Var<float6x6> stretch_hessian(const Var<float2x3>& F, Var<float> mu)
        {
            Var<float6x6> H;
            H->set_zero();

            const auto& Fu = F.cols[0];
            const auto& Fv = F.cols[1];

            const auto I5u = luisa::compute::dot(Fu, Fu);
            const auto I5v = luisa::compute::dot(Fv, Fv);

            const auto sqrtI5u    = luisa::compute::sqrt(I5u);
            const auto sqrtI5v    = luisa::compute::sqrt(I5v);
            const auto invSqrtI5u = 1.0f / sqrtI5u;
            const auto invSqrtI5v = 1.0f / sqrtI5v;

            H->scalar(0, 0) = H->scalar(1, 1) = H->scalar(2, 2) = luisa::compute::max(0.0f, 1.0f - invSqrtI5u);
            H->scalar(3, 3) = H->scalar(4, 4) = H->scalar(5, 5) = luisa::compute::max(0.0f, 1.0f - invSqrtI5v);

            auto fu = F.cols[0] * invSqrtI5u;
            auto fv = F.cols[1] * invSqrtI5v;

            Var<float> uCoeff = luisa::compute::min(invSqrtI5u, 1.0f);
            Var<float> vCoeff = luisa::compute::min(invSqrtI5v, 1.0f);
            H->block(0, 0)    = H->block(0, 0) + uCoeff * outer_product(fu, fu);
            H->block(1, 1)    = H->block(1, 1) + vCoeff * outer_product(fv, fv);
            return mu * H;
        }
        inline Var<float6x6> shear_hessian(const Var<float2x3>& F, Var<float> mu)
        {
            using Float    = Var<float>;
            using Float3   = Var<float3>;
            using Float6   = Var<float6>;
            using Float6x6 = Var<float6x6>;

            Float6x6 H;
            H->set_zero();

            const Float3& Fu = F.cols[0];
            const Float3& Fv = F.cols[1];

            const Float I6     = luisa::compute::dot(Fu, Fv);
            const Float signI6 = luisa::compute::sign(I6);

            H->scalar<3, 0>() = H->scalar<4, 1>() = H->scalar<5, 2>() = H->scalar<0, 3>() =
                H->scalar<1, 4>() = H->scalar<2, 5>() = 1.0f;

            Var<float2x2> tmp = luisa::compute::make_float2x2(luisa::compute::make_float2(0.0f, 1.0f),
                                                              luisa::compute::make_float2(1.0f, 0.0f));
            const Float6  g   = flatten(F * tmp);  // F * (a b^T + b a^T)

            const Float I2 = luisa::compute::dot(Fu, Fu) + luisa::compute::dot(Fv, Fv);  // F.squaredNorm();
            const Float lambda0 = 0.5f * (I2 + luisa::compute::sqrt(I2 * I2 + 12.0f * I6 * I6));

            const Float6 q0 = (I6 * H * g + lambda0 * g)->normalize();

            Float6x6 T;
            T->set_identity();
            T                   = 0.5f * (T + signI6 * H);
            const Float6 Tq     = T * q0;
            const auto   normTq = Tq->squared_norm();

            H = luisa::compute::abs(I6) * (T - outer_product(Tq, Tq) / normTq) + lambda0 * outer_product(q0, q0);

            return mu * H;
        }
    }  // namespace detail

    inline std::array<double, 2> convert_prop(const float young_mod, const float poiss_rat)
    {
        double mu     = young_mod / (2.0 * (1.0 + poiss_rat));
        double lambda = young_mod * poiss_rat / ((1.0 + poiss_rat) * (1.0 - 2.0 * poiss_rat));
        return {mu, lambda};
    }
    inline float2x2 get_Dm_inv(const float3& x_0, const float3& x_1, const float3& x_2)
    {
        float3         r_1     = x_1 - x_0;
        float3         r_2     = x_2 - x_0;
        float3         cross   = cross_vec(r_1, r_2);
        float3         axis_1  = normalize_vec(r_1);
        float3         axis_2  = normalize_vec(cross_vec(cross, axis_1));
        float2         uv0     = float2(dot_vec(axis_1, x_0), dot_vec(axis_2, x_0));
        float2         uv1     = float2(dot_vec(axis_1, x_1), dot_vec(axis_2, x_1));
        float2         uv2     = float2(dot_vec(axis_1, x_2), dot_vec(axis_2, x_2));
        float2         duv0    = uv1 - uv0;
        float2         duv1    = uv2 - uv0;
        const float2x2 duv     = float2x2(duv0, duv1);
        const float2x2 inv_duv = luisa::inverse(duv);
        return inv_duv;
    }
    inline void compute_gradient_hessian(const float3&   x0,
                                         const float3&   x1,
                                         const float3&   x2,
                                         const float2x2& Dm,
                                         const float     mu,
                                         const float     lambda,
                                         const float     area,
                                         float3x3&       dedx,
                                         float9x9&       d2edx2)
    {
        dedx = luisa::make_float3x3(0.0f);
        d2edx2.set_zero();

        float2x3 F = makeFloat2x3(x1 - x0, x2 - x0) * Dm;

        float2x3 de0dF   = detail::stretch_gradient(F, mu);
        float6x6 d2e0dF2 = detail::stretch_hessian(F, mu);

        float2x3 de1dF   = detail::shear_gradient(F, lambda);
        float6x6 d2e1dF2 = detail::shear_hessian(F, lambda);

        float2x3 dedF   = de0dF + de1dF;
        float6x6 d2edF2 = d2e0dF2 + d2e1dF2;

        dedx   = area * detail::convert_force(dedF, Dm);
        d2edx2 = area * detail::convert_hessian(d2edF2, Dm);

        // gradient[0] = dedx[0];
        // gradient[1] = dedx[1];
        // gradient[2] = dedx[2];
        // for (uint ii = 0; ii < 3; ii++)
        // {
        //     for (uint jj = 0; jj < 3; jj++)
        //     {
        //         hessian[jj][ii] = d2edx2.block(ii, jj);
        //     }
        // }
    }
    inline void compute_gradient_hessian(const Var<float3>&   x0,
                                         const Var<float3>&   x1,
                                         const Var<float3>&   x2,
                                         const Var<float2x2>& Dm,
                                         const Var<float>     mu,
                                         const Var<float>     lambda,
                                         const Var<float>     area,
                                         Var<float3x3>&       dedx,
                                         Var<float9x9>&       d2edx2)
    {
        // dedx = luisa::make_float3x3(0.0f);
        // d2edx2->set_zero();

        Var<float2x3> F = makeFloat2x3(x1 - x0, x2 - x0) * Dm;

        // float2x3 de0dF   = libuipc::stretch_gradient(F, mu, 1.0f);
        // float6x6 d2e0dF2 = libuipc::stretch_hessian(F, mu, 1.0f);
        auto de0dF   = detail::stretch_gradient(F, mu);
        auto d2e0dF2 = detail::stretch_hessian(F, mu);

        auto de1dF   = detail::shear_gradient(F, lambda);
        auto d2e1dF2 = detail::shear_hessian(F, lambda);

        auto dedF   = de0dF + de1dF;
        auto d2edF2 = d2e0dF2 + d2e1dF2;

        dedx   = area * detail::convert_force(dedF, Dm);
        d2edx2 = area * detail::convert_hessian(d2edF2, Dm);
    }

    inline float compute_energy(const float3&   x0,
                                const float3&   x1,
                                const float3&   x2,
                                const float2x2& Dm,
                                const float     mu,
                                const float     lambda,
                                const float     area)
    {

        const float2x3 F      = makeFloat2x3(x1 - x0, x2 - x0) * Dm;
        auto           energy = detail::stretch_energy(F, mu) + detail::shear_energy(F, lambda);
        return area * energy;
    }
    inline Var<float> compute_energy(const Var<float3>&   x0,
                                     const Var<float3>&   x1,
                                     const Var<float3>&   x2,
                                     const Var<float2x2>& Dm,
                                     const Var<float>     mu,
                                     const Var<float>     lambda,
                                     const Var<float>     area)
    {

        const Var<float2x3> F      = makeFloat2x3(x1 - x0, x2 - x0) * Dm;
        auto                energy = detail::stretch_energy(F, mu) + detail::shear_energy(F, lambda);
        return area * energy;
    }

    // inline float compute_theta(const float3& x2, const float3& x1, const float3& x0, const float3& x3)
    // {
    //     return detail::face_dihedral_angle(x0, x1, x2, x3);
    // }
    // inline float compute_theta(const Float3& x2, const Float3& x1, const Float3& x0, const Float3& x3)
    // {
    //     return detail::face_dihedral_angle(x0, x1, x2, x3);
    // }

};  // namespace StretchEnergy


};  // namespace lcs