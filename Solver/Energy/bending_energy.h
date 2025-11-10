// Modified from the appendix of [A Finite Element Formulation of Baraff-Witkin Cloth]
//          and [https://github.com/st-tech/ppf-contact-solver]
//          and [https://github.com/KemengHuang/GPU_IPC/blob/main/GPU_IPC/femEnergy.cu]

#pragma once

#include "Core/float_n.h"
#include "Core/float_nxn.h"
#include <luisa/luisa-compute.h>

namespace lcs
{

namespace BendingEnergy
{

    namespace LibShell
    {
        static luisa::float3x3 crossMatrix(const luisa::float3& v)
        {
            //    0   -v.z    v.y
            //  v.z      0   -v.x
            // -v.y    v.x      0
            return luisa::make_float3x3(luisa::make_float3(0, v.z, -v.y),
                                        luisa::make_float3(-v.z, 0, v.x),
                                        luisa::make_float3(v.y, -v.x, 0));
        }
        static float angle(const luisa::float3& v,
                           const luisa::float3& w,
                           const luisa::float3& axis,
                           luisa::float3        derivative[3],  // v, w
                           luisa::float3x3      hessian[3][3])
        {
            // double theta = 2.0 * luisa::atan2((v.cross(w).dot(axis) / axis.norm()), v.dot(w) + v.norm() * w.norm());
            float theta = 2.0f
                          * luisa::atan2((luisa::dot(luisa::cross(v, w), axis) / luisa::length(axis)),
                                         luisa::dot(v, w) + luisa::length(v) * luisa::length(w));

            if (derivative)
            {
                derivative[0] = -luisa::cross(axis, v) / luisa::dot(v, v) / luisa::length(axis);
                derivative[1] = luisa::cross(axis, w) / luisa::dot(w, w) / luisa::length(axis);
                derivative[2] = luisa::make_float3(0.0f);
            }
            if (hessian)
            {
                hessian[0][1] = luisa::float3x3::fill(0.0f);
                hessian[1][0] = luisa::float3x3::fill(0.0f);
                hessian[1][2] = luisa::float3x3::fill(0.0f);
                hessian[2][0] = luisa::float3x3::fill(0.0f);
                hessian[2][1] = luisa::float3x3::fill(0.0f);
                hessian[2][2] = luisa::float3x3::fill(0.0f);

                hessian[0][0] = 2.0f * outer_product(luisa::cross(axis, v), v) / luisa::dot(v, v)
                                / luisa::dot(v, v) / luisa::length(axis);
                hessian[1][1] = -2.0f * outer_product(luisa::cross(axis, w), w) / luisa::dot(w, w)
                                / luisa::dot(w, w) / luisa::length(axis);
                hessian[0][0] = hessian[0][0] - crossMatrix(axis) / luisa::dot(v, v) / luisa::length(axis);
                hessian[1][1] = hessian[1][1] + crossMatrix(axis) / luisa::dot(w, w) / luisa::length(axis);

                luisa::float3x3 dahat = luisa::float3x3::eye(1.0f) / luisa::length(axis)
                                        - outer_product(axis, axis) / luisa::length(axis)
                                              / luisa::length(axis) / luisa::length(axis);
                hessian[0][2] = crossMatrix(v) * dahat / luisa::dot(v, v);
                hessian[1][2] = crossMatrix(-w) * dahat / luisa::dot(w, w);
            }

            return theta;
        }
        static float edgeTheta(const luisa::float3& q0,
                               const luisa::float3& q1,
                               const luisa::float3& q2,
                               const luisa::float3& q3,
                               luisa::float3   derivative[4],  // edgeVertex, then edgeOppositeVertex
                               luisa::float3x3 hessian[4][4])
        {
            luisa::float3   n0   = luisa::cross(q0 - q2, q1 - q2);
            luisa::float3   n1   = luisa::cross(q1 - q3, q0 - q3);
            luisa::float3   axis = q1 - q0;
            luisa::float3   angderiv[3];
            luisa::float3x3 anghess[3][3];

            float theta =
                angle(n0, n1, axis, *((derivative || hessian) ? &angderiv : nullptr), *(hessian ? &anghess : nullptr));

            if (derivative)
            {
                derivative[0] += luisa::cross(angderiv[0], q2 - q1);
                derivative[1] += luisa::cross(angderiv[0], q0 - q2);
                derivative[2] += luisa::cross(angderiv[0], q1 - q0);
                derivative[0] += luisa::cross(angderiv[1], q1 - q3);
                derivative[1] += luisa::cross(angderiv[1], q3 - q0);
                derivative[3] += luisa::cross(angderiv[1], q0 - q1);
            }

            if (hessian)
            {
                luisa::float3x3 vqm[3];
                vqm[0] = crossMatrix(q0 - q2);
                vqm[1] = crossMatrix(q1 - q0);
                vqm[2] = crossMatrix(q2 - q1);
                luisa::float3x3 wqm[3];
                wqm[0] = crossMatrix(q0 - q1);
                wqm[1] = crossMatrix(q1 - q3);
                wqm[2] = crossMatrix(q3 - q0);

                int vindices[3] = {1, 2, 0};
                int windices[3] = {3, 0, 1};

                for (int i = 0; i < 3; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        hessian[vindices[i]][vindices[j]] =
                            hessian[vindices[i]][vindices[j]] + luisa::transpose(vqm[i]) * anghess[0][0] * vqm[j];
                        hessian[vindices[i]][windices[j]] =
                            hessian[vindices[i]][windices[j]] + luisa::transpose(vqm[i]) * anghess[0][1] * wqm[j];
                        hessian[windices[i]][vindices[j]] =
                            hessian[windices[i]][vindices[j]] + luisa::transpose(wqm[i]) * anghess[1][0] * vqm[j];
                        hessian[windices[i]][windices[j]] =
                            hessian[windices[i]][windices[j]] + luisa::transpose(wqm[i]) * anghess[1][1] * wqm[j];
                    }

                    hessian[vindices[i]][0] = hessian[vindices[i]][0] + luisa::transpose(vqm[i]) * anghess[0][2];
                    hessian[0][vindices[i]] = hessian[0][vindices[i]] + anghess[2][0] * vqm[i];
                    hessian[windices[i]][0] = hessian[windices[i]][0] - luisa::transpose(wqm[i]) * anghess[1][2];
                    hessian[0][windices[i]] = hessian[0][windices[i]] - anghess[2][1] * wqm[i];

                    hessian[vindices[i]][1] = hessian[vindices[i]][1] + luisa::transpose(vqm[i]) * anghess[0][2];
                    hessian[1][vindices[i]] = hessian[1][vindices[i]] + anghess[2][0] * vqm[i];
                    hessian[windices[i]][1] = hessian[windices[i]][1] - luisa::transpose(wqm[i]) * anghess[1][2];
                    hessian[1][windices[i]] = hessian[1][windices[i]] + anghess[2][1] * wqm[i];
                }

                luisa::float3 dang1 = angderiv[0];
                luisa::float3 dang2 = angderiv[1];

                luisa::float3x3 dang1mat = crossMatrix(dang1);
                luisa::float3x3 dang2mat = crossMatrix(dang2);

                hessian[6 / 3][3 / 3] = hessian[6 / 3][3 / 3] + dang1mat;
                hessian[0 / 3][3 / 3] = hessian[0 / 3][3 / 3] - dang1mat;
                hessian[0 / 3][6 / 3] = hessian[0 / 3][6 / 3] + dang1mat;
                hessian[3 / 3][0 / 3] = hessian[3 / 3][0 / 3] + dang1mat;
                hessian[3 / 3][6 / 3] = hessian[3 / 3][6 / 3] - dang1mat;
                hessian[6 / 3][0 / 3] = hessian[6 / 3][0 / 3] - dang1mat;

                hessian[9 / 3][0 / 3] = hessian[9 / 3][0 / 3] + dang2mat;
                hessian[3 / 3][0 / 3] = hessian[3 / 3][0 / 3] - dang2mat;
                hessian[3 / 3][9 / 3] = hessian[3 / 3][9 / 3] + dang2mat;
                hessian[0 / 3][3 / 3] = hessian[0 / 3][3 / 3] + dang2mat;
                hessian[0 / 3][9 / 3] = hessian[0 / 3][9 / 3] - dang2mat;
                hessian[9 / 3][3 / 3] = hessian[9 / 3][3 / 3] - dang2mat;
            }

            return theta;
        }

    };  // namespace LibShell

    using Float3 = luisa::compute::Float3;
    using Float  = luisa::compute::Float;

    namespace detail
    {
        using HostVector12 = std::array<float3, 4>;
        static inline HostVector12 face_dihedral_angle_grad(const float3& v2, const float3& v0, const float3& v1, const float3& v3)
        {
            const float3 e0      = v1 - v0;
            const float3 e1      = v2 - v0;
            const float3 e2      = v3 - v0;
            const float3 e3      = v2 - v1;
            const float3 e4      = v3 - v1;
            const float3 n1      = luisa::cross(e0, e1);
            const float3 n2      = luisa::cross(e2, e0);
            const float  n1_sqnm = luisa::dot(n1, n1);
            const float  n2_sqnm = luisa::dot(n2, n2);
            const float  e0_norm = luisa::length(e0);
            assert(n1_sqnm > 0.0f);
            assert(n2_sqnm > 0.0f);
            assert(e0_norm > 0.0f);

            HostVector12 grad;
            grad[0] = -e0_norm / n1_sqnm * n1;
            grad[1] = -luisa::dot(e0, e3) / (e0_norm * n1_sqnm) * n1 - luisa::dot(e0, e4) / (e0_norm * n2_sqnm) * n2;
            grad[2] = luisa::dot(e0, e1) / (e0_norm * n1_sqnm) * n1 + luisa::dot(e0, e2) / (e0_norm * n2_sqnm) * n2;
            grad[3] = -e0_norm / n2_sqnm * n2;
            return grad;
        }

        using DeviceVector12 = luisa::compute::ArrayFloat3<4>;
        static inline DeviceVector12 face_dihedral_angle_grad(const Float3& v2,
                                                              const Float3& v0,
                                                              const Float3& v1,
                                                              const Float3& v3)
        {

            const Float3 e0      = v1 - v0;
            const Float3 e1      = v2 - v0;
            const Float3 e2      = v3 - v0;
            const Float3 e3      = v2 - v1;
            const Float3 e4      = v3 - v1;
            const Float3 n1      = luisa::compute::cross(e0, e1);
            const Float3 n2      = luisa::compute::cross(e2, e0);
            const Float  n1_sqnm = luisa::compute::dot(n1, n1);
            const Float  n2_sqnm = luisa::compute::dot(n2, n2);
            const Float  e0_norm = luisa::compute::length(e0);
            luisa::compute::device_assert(n1_sqnm > 0.0f);
            luisa::compute::device_assert(n2_sqnm > 0.0f);
            luisa::compute::device_assert(e0_norm > 0.0f);

            DeviceVector12 grad;
            grad[0] = -e0_norm / n1_sqnm * n1;
            grad[1] = -luisa::compute::dot(e0, e3) / (e0_norm * n1_sqnm) * n1
                      - luisa::compute::dot(e0, e4) / (e0_norm * n2_sqnm) * n2;
            grad[2] = luisa::compute::dot(e0, e1) / (e0_norm * n1_sqnm) * n1
                      + luisa::compute::dot(e0, e2) / (e0_norm * n2_sqnm) * n2;
            grad[3] = -e0_norm / n2_sqnm * n2;
            return grad;
        }

        static inline float face_dihedral_angle(const float3& v0, const float3& v1, const float3& v2, const float3& v3)
        {
            const float3 n1  = luisa::cross(v1 - v0, v2 - v0);
            const float3 n2  = luisa::cross(v2 - v3, v1 - v3);
            float        dot = luisa::dot(n1, n2) / luisa::sqrt(luisa::dot(n1, n1) * luisa::dot(n2, n2));
            float        angle = luisa::acos(luisa::max(-1.0f, luisa::min(1.0f, dot)));
            float        sign  = luisa::sign(luisa::dot(luisa::cross(n2, n1), v1 - v2));
            angle              = sign * angle;
            return angle;
        }

        static inline Float face_dihedral_angle(const Float3& v0, const Float3& v1, const Float3& v2, const Float3& v3)
        {
            const Float3 n1  = luisa::compute::cross(v1 - v0, v2 - v0);
            const Float3 n2  = luisa::compute::cross(v2 - v3, v1 - v3);
            Float        dot = luisa::compute::dot(n1, n2)
                        / luisa::compute::sqrt(luisa::compute::dot(n1, n1) * luisa::compute::dot(n2, n2));
            Float angle = luisa::compute::acos(luisa::compute::max(-1.0f, luisa::compute::min(1.0f, dot)));
            Float sign = luisa::compute::sign(luisa::compute::dot(luisa::compute::cross(n2, n1), v1 - v2));
            angle = sign * angle;
            return angle;
        }

        inline uint4 remap(const uint4& hinge)
        {
            return luisa::make_uint4(hinge[2], hinge[1], hinge[0], hinge[3]);
        }
        inline luisa::compute::Uint4 remap(const luisa::compute::Uint4& hinge)
        {
            return luisa::compute::make_uint4(hinge[2], hinge[1], hinge[0], hinge[3]);
        }
    }  // namespace detail

    inline float compute_d_theta_d_x(const float3& x2, const float3& x1, const float3& x0, const float3& x3, float3 gradient[4])
    {
        const auto angle      = detail::face_dihedral_angle(x0, x1, x2, x3);
        const auto angle_grad = detail::face_dihedral_angle_grad(x0, x1, x2, x3);
        gradient[2]           = angle_grad[0];
        gradient[1]           = angle_grad[1];
        gradient[0]           = angle_grad[2];
        gradient[3]           = angle_grad[3];
        return angle;
    }
    inline Float compute_d_theta_d_x(const Float3& x2, const Float3& x1, const Float3& x0, const Float3& x3, Float3 gradient[4])
    {
        const auto angle      = detail::face_dihedral_angle(x0, x1, x2, x3);
        const auto angle_grad = detail::face_dihedral_angle_grad(x0, x1, x2, x3);
        gradient[2]           = angle_grad[0];
        gradient[1]           = angle_grad[1];
        gradient[0]           = angle_grad[2];
        gradient[3]           = angle_grad[3];
        return angle;
    }
    inline float compute_theta(const float3& x2, const float3& x1, const float3& x0, const float3& x3)
    {
        return detail::face_dihedral_angle(x0, x1, x2, x3);
    }
    inline Float compute_theta(const Float3& x2, const Float3& x1, const Float3& x0, const Float3& x3)
    {
        return detail::face_dihedral_angle(x0, x1, x2, x3);
    }

    // inline float face_dihedral_angle(const float3& v0, const float3& v1, const float3& v2, const float3& v3)
    // {
    //     const float3& x3 = v0;
    //     const float3& x4 = v1;
    //     const float3& x1 = v2;
    //     const float3& x2 = v3;

    //     const float3 N1 = luisa::cross(x1 - x3, x1 - x4);
    //     const float3 N2 = luisa::cross(x2 - x4, x2 - x3);

    //     const float3 E         = x4 - x3;
    //     const float  E_square  = luisa::dot(E, E);
    //     const float  N1_square = luisa::dot(N1, N1);
    //     const float  N2_square = luisa::dot(N2, N2);
    //     const float  E_len     = luisa::sqrt(E_square);
    //     const float  N1_len    = luisa::sqrt(N1_square);
    //     const float  N2_len    = luisa::sqrt(N2_square);
    //     const float3 normal1   = N1 / N1_len;
    //     const float3 normal2   = N2 / N2_len;

    //     const float3 u1 = E_len * normal1;
    //     const float3 u2 = E_len * normal2;
    //     const float3 u3 = luisa::dot(x1 - x4, E) * normal1 / E_len + luisa::dot(x2 - x4, E) * normal2 / E_len;
    //     const float3 u4 = -luisa::dot(x1 - x3, E) * normal1 / E_len - luisa::dot(x2 - x3, E) * normal2 / E_len;
    //     const float3 u[4] = {u1, u2, u3, u4};

    //     const float sign = luisa::sign(luisa::dot(luisa::cross(normal1, normal2), E));
    //     return sign * luisa::acos(luisa::dot(normal1, normal2));
    // }
    // inline void face_dihedral_gradient(
    //     const float3& v0, const float3& v1, const float3& v2, const float3& v3, float3& g0, float3& g1, float3& g2, float3& g3)
    // {
    //     const float3& x3 = v0;
    //     const float3& x4 = v1;
    //     const float3& x1 = v2;
    //     const float3& x2 = v3;

    //     const float3 N1 = luisa::cross(x1 - x3, x1 - x4);
    //     const float3 N2 = luisa::cross(x2 - x4, x2 - x3);

    //     const float3 E         = x4 - x3;
    //     const float  E_square  = luisa::dot(E, E);
    //     const float  N1_square = luisa::dot(N1, N1);
    //     const float  N2_square = luisa::dot(N2, N2);
    //     const float  E_len     = luisa::sqrt(E_square);
    //     const float  N1_len    = luisa::sqrt(N1_square);
    //     const float  N2_len    = luisa::sqrt(N2_square);
    //     const float3 normal1   = N1 / N1_len;
    //     const float3 normal2   = N2 / N2_len;

    //     const float3 u1 = E_len * normal1;
    //     const float3 u2 = E_len * normal2;
    //     const float3 u3 = luisa::dot(x1 - x4, E) * normal1 / E_len + luisa::dot(x2 - x4, E) * normal2 / E_len;
    //     const float3 u4 = -luisa::dot(x1 - x3, E) * normal1 / E_len - luisa::dot(x2 - x3, E) * normal2 / E_len;
    //     const float3 u[4] = {u1, u2, u3, u4};

    //     const float sign = luisa::sign(luisa::dot(luisa::cross(normal1, normal2), E));

    //     const float sin_half_theta = sign * luisa::sqrt(1 - luisa::dot(normal1, normal2) / 2);
    // }

};  // namespace BendingEnergy


};  // namespace lcs