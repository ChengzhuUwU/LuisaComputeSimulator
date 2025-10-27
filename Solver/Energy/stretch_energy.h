//
// This file incorporates code from baraffwitkin.hpp (by Ryoichi Ando) and strain_limiting_baraff_witkin_shell_2d.h (by Kemeng Huang and Xinyu Lu)
// Original Author: Ryoichi Ando (ryoichi.ando@zozo.com, License: Apache v2.0), Kemeng Huang and Xinyu Lu
//

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
    namespace libuipc
    {
        inline Eigen::Vector<float, 6> flatten(const Eigen::Matrix<float, 3, 2>& F)
        {
            Eigen::Vector<float, 6> R;
            R.segment<3>(0) = F.col(0);
            R.segment<3>(3) = F.col(1);
            return R;
        }

        inline Eigen::Matrix<float, 3, 2> F3x2(const Eigen::Matrix<float, 3, 2>& Ds3x2,
                                               const Eigen::Matrix<float, 2, 2>& Dms2x2_inv)
        {
            return Ds3x2 * Dms2x2_inv;
        }

        inline Eigen::Matrix<float, 2, 2> Dm2x2(const Eigen::Vector3f& x0,
                                                const Eigen::Vector3f& x1,
                                                const Eigen::Vector3f& x2)
        {
            Eigen::Vector3f v01 = x1 - x0;
            Eigen::Vector3f v02 = x2 - x0;
            // compute uv coordinates by rotating each triangle normal to (0, 1, 0)
            Eigen::Vector3f normal = v01.cross(v02).normalized();
            Eigen::Vector3f target = Eigen::Vector3f(0, 1, 0);

            Eigen::Vector3f vec      = normal.cross(target);
            float           cos      = normal.dot(target);
            Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();
            Eigen::Matrix3f cross_vec;

            cross_vec << 0, -vec.z(), vec.y(),  //
                vec.z(), 0, -vec.x(),           //
                -vec.y(), vec.x(), 0;

            rotation += cross_vec + cross_vec * cross_vec / (1 + cos);

            Eigen::Vector3f rotate_uv0 = rotation * x0;
            Eigen::Vector3f rotate_uv1 = rotation * x1;
            Eigen::Vector3f rotate_uv2 = rotation * x2;

            auto            uv0 = Eigen::Vector2f(rotate_uv0.x(), rotate_uv0.z());
            auto            uv1 = Eigen::Vector2f(rotate_uv1.x(), rotate_uv1.z());
            auto            uv2 = Eigen::Vector2f(rotate_uv2.x(), rotate_uv2.z());
            Eigen::Matrix2f M;
            M.col(0) = uv1 - uv0;
            M.col(1) = uv2 - uv0;
            return M;
        }
        inline Eigen::Matrix<float, 3, 2> Ds3x2(const Eigen::Vector3f& x0,
                                                const Eigen::Vector3f& x1,
                                                const Eigen::Vector3f& x2)
        {
            Eigen::Matrix<float, 3, 2> M;
            M.col(0) = x1 - x0;
            M.col(1) = x2 - x0;
            return M;
        }

        namespace BWS
        {
            inline Eigen::Matrix<float, 6, 9> dFdX(const Eigen::Matrix<float, 2, 2>& DmI)
            {
                Eigen::Matrix<float, 6, 9> dfdx;
                dfdx.setZero();
                float d0 = DmI(0, 0);
                float d1 = DmI(1, 0);
                float d2 = DmI(0, 1);
                float d3 = DmI(1, 1);
                float s0 = d0 + d1;
                float s1 = d2 + d3;

                for (int i = 0; i < 3; i++)
                {
                    dfdx(i, i)     = -s0;
                    dfdx(i + 3, i) = -s1;
                }
                for (int i = 0; i < 3; i++)
                {
                    dfdx(i, i + 3)     = d0;
                    dfdx(i + 3, i + 3) = d2;
                }
                for (int i = 0; i < 3; i++)
                {
                    dfdx(i, i + 6)     = d1;
                    dfdx(i + 3, i + 6) = d3;
                }
                return dfdx;
            }

            inline void dEdF(Eigen::Matrix<float, 3, 2>&       R,
                             const Eigen::Matrix<float, 3, 2>& F,
                             const Eigen::Vector2f&            anisotropic_a,
                             const Eigen::Vector2f&            anisotropic_b,
                             float                             stretchS,
                             float                             shearS,
                             float                             strainRate)
            {
                stretchS /= strainRate;
                float I6 = anisotropic_a.transpose() * F.transpose() * F * anisotropic_b;
                Eigen::Matrix<float, 3, 2> stretch_pk1, shear_pk1;

                shear_pk1 = 2 * (I6 - anisotropic_a.transpose() * anisotropic_b)
                            * (F * anisotropic_a * anisotropic_b.transpose()
                               + F * anisotropic_b * anisotropic_a.transpose());
                float I5u    = (F * anisotropic_a).transpose() * F * anisotropic_a;
                float I5v    = (F * anisotropic_b).transpose() * F * anisotropic_b;
                float ucoeff = float{1} - float{1} / sqrt(I5u);
                float vcoeff = float{1} - float{1} / sqrt(I5v);

                if (I5u > 1)
                {
                    ucoeff += 1.5 * strainRate * (sqrt(I5u) + 1 / sqrt(I5u) - 2);
                }
                if (I5v > 1)
                {
                    vcoeff += 1.5 * strainRate * (sqrt(I5v) + 1 / sqrt(I5v) - 2);
                }


                stretch_pk1 = ucoeff * float{2} * F * anisotropic_a * anisotropic_a.transpose()
                              + vcoeff * float{2} * F * anisotropic_b * anisotropic_b.transpose();

                R = (stretchS * stretch_pk1 + shearS * shear_pk1);
            }
            inline void ddEddF(Eigen::Matrix<float, 6, 6>&       R,
                               const Eigen::Matrix<float, 3, 2>& F,
                               const Eigen::Vector2f&            anisotropic_a,
                               const Eigen::Vector2f&            anisotropic_b,
                               float                             stretchS,
                               float                             shearS,
                               float                             strainRate)
            {

                stretchS /= strainRate;

                Eigen::Matrix<float, 6, 6> final_H = Eigen::Matrix<float, 6, 6>::Zero();
                {
                    Eigen::Matrix<float, 6, 6> H;
                    H.setZero();
                    float I5u        = (F * anisotropic_a).transpose() * F * anisotropic_a;
                    float I5v        = (F * anisotropic_b).transpose() * F * anisotropic_b;
                    float invSqrtI5u = float{1} / sqrt(I5u);
                    float invSqrtI5v = float{1} / sqrt(I5v);

                    float sqrtI5u = sqrt(I5u);
                    float sqrtI5v = sqrt(I5v);

                    if (sqrtI5u > 1)
                        H(0, 0) = H(1, 1) = H(2, 2) =
                            2 * (((sqrtI5u - 1) * (3 * sqrtI5u * strainRate - 3 * strainRate + 2)) / (2 * sqrtI5u));
                    if (sqrtI5v > 1)
                        H(3, 3) = H(4, 4) = H(5, 5) =
                            2 * (((sqrtI5v - 1) * (3 * sqrtI5v * strainRate - 3 * strainRate + 2)) / (2 * sqrtI5v));
                    auto fu = F.col(0).normalized();
                    auto fv = F.col(1).normalized();

                    float uCoeff =
                        (sqrtI5u > float{1.0}) ? (3 * I5u * strainRate - 3 * strainRate + 2) / (sqrt(I5u)) : 2.0;
                    float vCoeff = (sqrtI5v > float{1.0}) ?
                                       (3 * I5v * strainRate - 3 * strainRate + 2) / (sqrt(I5v)) :
                                       float{2.0};


                    H.block<3, 3>(0, 0) += uCoeff * (fu * fu.transpose());
                    H.block<3, 3>(3, 3) += vCoeff * (fv * fv.transpose());

                    final_H += stretchS * H;
                }
                {
                    Eigen::Matrix<float, 6, 6> H_shear;
                    H_shear.setZero();
                    Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
                    H(3, 0) = H(4, 1) = H(5, 2) = H(0, 3) = H(1, 4) = H(2, 5) = 1.0;
                    float I6     = anisotropic_a.transpose() * F.transpose() * F * anisotropic_b;
                    float signI6 = (I6 >= 0) ? 1.0 : -1.0;
                    auto  g =
                        F * (anisotropic_a * anisotropic_b.transpose() + anisotropic_b * anisotropic_a.transpose());
                    Eigen::Matrix<float, 6, 1> vec_g = Eigen::Matrix<float, 6, 1>::Zero();

                    vec_g.block(0, 0, 3, 1)            = g.col(0);
                    vec_g.block(3, 0, 3, 1)            = g.col(1);
                    float                      I2      = F.squaredNorm();
                    float                      lambda0 = 0.5 * (I2 + sqrt(I2 * I2 + 12.0 * I6 * I6));
                    Eigen::Matrix<float, 6, 1> q0      = (I6 * H * vec_g + lambda0 * vec_g).normalized();
                    Eigen::Matrix<float, 6, 6> T       = Eigen::Matrix<float, 6, 6>::Identity();
                    T                                  = float{0.5} * (T + signI6 * H);
                    auto  Tq                           = T * q0;
                    float normTq                       = Tq.squaredNorm();
                    H_shear = fabs(I6) * (T - (Tq * Tq.transpose()) / normTq) + lambda0 * (q0 * q0.transpose());
                    H_shear *= 2;
                    final_H += shearS * H_shear;
                }
                R = final_H;
            }
        }  // namespace BWS

        inline void compute_gradient_hessian(const float3&               x0,
                                             const float3&               x1,
                                             const float3&               x2,
                                             const float3&               X0,
                                             const float3&               X1,
                                             const float3&               X2,
                                             const float                 mu,
                                             const float                 lambda,
                                             const float                 area,
                                             Eigen::Vector<float, 9>&    G,
                                             Eigen::Matrix<float, 9, 9>& H)
        {
            float       Vdt2       = area;
            const float strainRate = 100.0f;

            Eigen::Vector<float, 9> X;
            X.block<3, 1>(0, 0) = float3_to_eigen3(x0);
            X.block<3, 1>(3, 0) = float3_to_eigen3(x1);
            X.block<3, 1>(6, 0) = float3_to_eigen3(x2);

            Eigen::Vector<float, 9> x_bars;
            x_bars.block<3, 1>(0, 0) = float3_to_eigen3(X0);
            x_bars.block<3, 1>(3, 0) = float3_to_eigen3(X1);
            x_bars.block<3, 1>(6, 0) = float3_to_eigen3(X2);

            Eigen::Matrix<float, 2, 2> IB =
                Dm2x2(x_bars.segment<3>(0), x_bars.segment<3>(3), x_bars.segment<3>(6));
            IB                              = IB.inverse();
            Eigen::Matrix<float, 3, 2> Ds   = Ds3x2(X.segment<3>(0), X.segment<3>(3), X.segment<3>(6));
            Eigen::Matrix<float, 3, 2> F    = Ds * IB;
            auto                       dFdx = BWS::dFdX(IB);

            Eigen::Vector2f anisotropic_a = Eigen::Vector2f(1, 0);
            Eigen::Vector2f anisotropic_b = Eigen::Vector2f(0, 1);


            Eigen::Matrix<float, 3, 2> dEdF;
            BWS::dEdF(dEdF, F, anisotropic_a, anisotropic_b, mu, lambda, strainRate);

            auto VecdEdF = flatten(dEdF);
            G            = dFdx.transpose() * VecdEdF;
            G *= Vdt2;

            Eigen::Matrix<float, 6, 6> ddEddF;
            BWS::ddEddF(ddEddF, F, anisotropic_a, anisotropic_b, mu, lambda, strainRate);
            ddEddF *= Vdt2;
            H = dFdx.transpose() * ddEddF * dFdx;
        }

    }  // namespace libuipc

    namespace libuipc
    {
        inline float2x3 stretch_gradient(const float2x3& F, float mu, const float strainRate)
        {
            mu /= strainRate;

            const auto I5u        = luisa::dot(F.cols[0], F.cols[0]);
            const auto I5v        = luisa::dot(F.cols[1], F.cols[1]);
            float      sqrtI5u    = luisa::sqrt(I5u);
            float      sqrtI5v    = luisa::sqrt(I5v);
            float      invSqrtI5u = 1.0f / sqrtI5u;
            float      invSqrtI5v = 1.0f / sqrtI5v;
            float      ucoeff     = float{1} - float{1} / sqrt(I5u);
            float      vcoeff     = float{1} - float{1} / sqrt(I5v);

            if (I5u > 1)
            {
                ucoeff += 1.5 * strainRate * (sqrt(I5u) + 1 / sqrt(I5u) - 2);
            }
            if (I5v > 1)
            {
                vcoeff += 1.5 * strainRate * (sqrt(I5v) + 1 / sqrt(I5v) - 2);
            }


            float2x3 result;
            result.cols[0] = ucoeff * 2.0f * F.cols[0];
            result.cols[1] = vcoeff * 2.0f * F.cols[1];
            // LUISA_INFO("Fu = {}, Fv = {}, norm_u -1 = {}, norm_v -1 = {}", fu, fv, norm_u - 1, norm_v - 1);
            return mu * result;
        }
        inline float6x6 stretch_hessian(const float2x3& F, float mu, const float strainRate)
        {
            mu /= strainRate;

            float6x6 H = float6x6::zero();

            const auto I5u        = luisa::dot(F.cols[0], F.cols[0]);
            const auto I5v        = luisa::dot(F.cols[1], F.cols[1]);
            float      sqrtI5u    = luisa::sqrt(I5u);
            float      sqrtI5v    = luisa::sqrt(I5v);
            float      invSqrtI5u = 1.0f / sqrtI5u;
            float      invSqrtI5v = 1.0f / sqrtI5v;
            if (sqrtI5u > 1.0f)
                H.scalar(0, 0) = H.scalar(1, 1) = H.scalar(2, 2) =
                    2 * (((sqrtI5u - 1) * (3 * sqrtI5u * strainRate - 3 * strainRate + 2)) / (2 * sqrtI5u));

            if (sqrtI5v > 1.0f)
                H.scalar(3, 3) = H.scalar(4, 4) = H.scalar(5, 5) =
                    2 * (((sqrtI5v - 1) * (3 * sqrtI5v * strainRate - 3 * strainRate + 2)) / (2 * sqrtI5v));

            auto fu = F.cols[0] * invSqrtI5u;
            auto fv = F.cols[1] * invSqrtI5v;
            float uCoeff = (sqrtI5u > 1.0f) ? (3 * I5u * strainRate - 3 * strainRate + 2) * invSqrtI5u : 2.0f;
            float vCoeff = (sqrtI5v > 1.0f) ? (3 * I5v * strainRate - 3 * strainRate + 2) * invSqrtI5v : 2.0f;
            H.block(0, 0) = H.block(0, 0) + uCoeff * outer_product(fu, fu);
            H.block(1, 1) = H.block(1, 1) + vCoeff * outer_product(fv, fv);
            return mu * H;
        }
    }  // namespace libuipc

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

        inline LargeMatrix<9, 6> get_dFdx(const luisa::float2x2& InverseDm)
        {
            const float d0 = InverseDm[0][0];
            const float d1 = InverseDm[0][1];
            const float d2 = InverseDm[1][0];
            const float d3 = InverseDm[1][1];
            const float s0 = d0 + d1;
            const float s1 = d2 + d3;

            lcs::LargeMatrix<9, 6> result;
            // const float2x3         diff_mat = make_diff_mat3x2();
            // for (unsigned i = 0; i < 3; ++i)
            // {
            //     for (unsigned j = 0; j < 2; ++j)
            //     {
            //         for (unsigned dim = 0; dim < 3; ++dim)
            //         {
            //             result.scalar(i, 3 * j + dim)     = diff_mat[j][dim] * inv_rest2x2[0][i];
            //             result.scalar(i + 3, 3 * j + dim) = diff_mat[j][dim] * inv_rest2x2[1][i];
            //         }
            //     }
            // }

            // for(int i = 0; i < 3; i++)
            // {
            //     dfdx(i, i)     = -s0;
            //     dfdx(i + 3, i) = -s1;
            // }
            // for(int i = 0; i < 3; i++)
            // {
            //     dfdx(i, i + 3)     = d0;
            //     dfdx(i + 3, i + 3) = d2;
            // }
            // for(int i = 0; i < 3; i++)
            // {
            //     dfdx(i, i + 6)     = d1;
            //     dfdx(i + 3, i + 6) = d3;
            // }

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
            // inv_rest2x2;
            lcs::LargeMatrix<6, 9> dfdx_T = lcs::LargeMatrix<6, 9>::zero();

            const auto& InverseDm = inv_rest2x2;
            const float d0        = InverseDm[0][0];
            const float d1        = InverseDm[0][1];
            const float d2        = InverseDm[1][0];
            const float d3        = InverseDm[1][1];
            const float s0        = d0 + d1;
            const float s1        = d2 + d3;
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

            // LUISA_INFO("dFdx = {}", dfdx_T);

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

        inline float2x3 shear_gradient(const float2x3& F, float lmd)
        {
            float    w = luisa::dot(F[0], F[1]);
            float2x3 result;
            result[0] = w * F[1];
            result[1] = w * F[0];
            return lmd * result;
        }

        inline float2x3 stretch_gradient(const float2x3& F, float mu)
        {
            const float3& Fu = F.cols[0];
            const float3& Fv = F.cols[1];

            const auto I5u = luisa::dot(Fu, Fu);
            const auto I5v = luisa::dot(Fv, Fv);

            float sqrtI5u    = luisa::sqrt(I5u);
            float sqrtI5v    = luisa::sqrt(I5v);
            float invSqrtI5u = 1.0f / sqrtI5u;
            float invSqrtI5v = 1.0f / sqrtI5v;

            // float uCoeff = (invSqrtI5u < 1.0f) ? invSqrtI5u : 1.0f;
            // float vCoeff = (invSqrtI5v < 1.0f) ? invSqrtI5v : 1.0f;

            float2x3 result;
            result.cols[0] = (sqrtI5u - 1.0f) * invSqrtI5u * F.cols[0];
            result.cols[1] = (sqrtI5v - 1.0f) * invSqrtI5v * F.cols[1];
            // LUISA_INFO("Fu = {}, Fv = {}, norm_u -1 = {}, norm_v -1 = {}", fu, fv, norm_u - 1, norm_v - 1);
            return mu * result;
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
    }  // namespace detail


    inline void compute_gradient_hessian(const float3&   x0,
                                         const float3&   x1,
                                         const float3&   x2,
                                         const float2x2& Dm,
                                         const float     mu,
                                         const float     lambda,
                                         const float     area,
                                         float3          gradient[3],
                                         float3x3        hessian[3][3])
    {
        float3x3 dedx   = luisa::make_float3x3(0.0f);
        float9x9 d2edx2 = float9x9::zero();
        float2x3 F      = makeFloat2x3(x1 - x0, x2 - x0) * Dm;
        // float2x3 de0dF   = libuipc::stretch_gradient(F, mu, 1.0f);
        // float6x6 d2e0dF2 = libuipc::stretch_hessian(F, mu, 1.0f);
        float2x3 de0dF   = detail::stretch_gradient(F, mu);
        float6x6 d2e0dF2 = detail::stretch_hessian(F, mu);

        float2x3 dedF   = de0dF;
        float6x6 d2edF2 = d2e0dF2;

        dedx   = area * detail::convert_force(dedF, Dm);
        d2edx2 = area * detail::convert_hessian(d2edF2, Dm);

        gradient[0] = dedx[0];
        gradient[1] = dedx[1];
        gradient[2] = dedx[2];
        for (uint ii = 0; ii < 3; ii++)
        {
            for (uint jj = 0; jj < 3; jj++)
            {
                hessian[ii][jj] = d2edx2.block(ii, jj);
            }
        }
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