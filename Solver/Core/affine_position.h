#pragma once

#include <luisa/luisa-compute.h>
#include "Core/lc_to_eigen.h"
#include "Core/xbasic_types.h"
#include "Core/float_nxn.h"

namespace lcs
{

static inline luisa::float4x4 scale(const luisa::float3& v)
{
    // | sx  0   0   0 |
    // | 0   sy  0   0 |
    // | 0   0   sz  0 |
    // | 0   0   0   1 |

    luisa::float4x4 result = luisa::make_float4x4(luisa::make_float4(v[0], 0.0f, 0.0f, 0.0f),
                                                  luisa::make_float4(0.0f, v[1], 0.0f, 0.0f),
                                                  luisa::make_float4(0.0f, 0.0f, v[2], 0.0f),
                                                  luisa::make_float4(0.0f, 0.0f, 0.0f, 1.0f));
    return result;
}
static inline luisa::float4x4 translate(const luisa::float3& v)
{
    // | 1 0 0 x |
    // | 0 1 0 y |
    // | 0 0 1 z |
    // | 0 0 0 1 |

    // luisa::float4x4 result = Identity4x4;
    // set(result, 0, 0, 1.0f);
    // set(result, 1, 1, 1.0f);
    // set(result, 2, 2, 1.0f);
    // set(result, 3, 3, 1.0f);
    // set(result, 3, 0, v[0]);
    // set(result, 3, 1, v[1]);
    // set(result, 3, 2, v[2]);
    // return result;

    return luisa::transpose(luisa::make_float4x4(luisa::make_float4(1.0f, 0.0f, 0.0f, v[0]),
                                                 luisa::make_float4(0.0f, 1.0f, 0.0f, v[1]),
                                                 luisa::make_float4(0.0f, 0.0f, 1.0f, v[2]),
                                                 luisa::make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
}
static inline luisa::float4x4 rorateX(float angleX)
{
    float cosX = luisa::cos(angleX);
    float sinX = luisa::sin(angleX);

    // |  1     0      0     0 |
    // |  0   cos(θ) -sin(θ) 0 |
    // |  0   sin(θ)  cos(θ) 0 |
    // |  0     0      0     1 |

    return luisa::transpose(luisa::make_float4x4(luisa::make_float4(1.0f, 0.0f, 0.0f, 0.0f),
                                                 luisa::make_float4(0.0f, cosX, sinX, 0.0f),
                                                 luisa::make_float4(0.0f, -sinX, cosX, 0.0f),
                                                 luisa::make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
}
static inline luisa::float4x4 rorateY(float angleY)
{
    float cosY = luisa::cos(angleY);
    float sinY = luisa::sin(angleY);

    // |  cos(θ)  0  sin(θ)  0 |
    // |    0     1    0     0 |
    // | -sin(θ)  0  cos(θ)  0 |
    // |    0     0    0     1 |
    return luisa::transpose(luisa::make_float4x4(luisa::make_float4(cosY, 0.0f, -sinY, 0.0f),
                                                 luisa::make_float4(0.0f, 1.0f, 0.0f, 0.0f),
                                                 luisa::make_float4(sinY, 0.0f, cosY, 0.0f),
                                                 luisa::make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
}
static inline luisa::float4x4 rorateZ(float angleZ)
{
    float cosZ = luisa::cos(angleZ);
    float sinZ = luisa::sin(angleZ);

    // | cos(θ) -sin(θ)  0  0 |
    // | sin(θ)  cos(θ)  0  0 |
    // |   0       0     1  0 |
    // |   0       0     0  1 |

    return luisa::transpose(luisa::make_float4x4(luisa::make_float4(cosZ, -sinZ, 0.0f, 0.0f),
                                                 luisa::make_float4(sinZ, cosZ, 0.0f, 0.0f),
                                                 luisa::make_float4(0.0f, 0.0f, 1.0f, 0.0f),
                                                 luisa::make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
}
static inline luisa::float4x4 rotate(const luisa::float3& axis)
{
    return rorateX(axis[0]) * rorateY(axis[1]) * rorateZ(axis[2]);
}

inline luisa::float4x4 make_model_matrix(const luisa::float3& t, const luisa::float3& r, const luisa::float3& s)
{
    return translate(t) * rotate(r) * scale(s);
    // return scale(s) * (rotate(r) * translate(t));
    // return translate(t) ;
}

inline luisa::float3 affine_position(const luisa::float4x4& model_matrix, const luisa::float3& model_position)
{
    luisa::float4 mult_position =
        model_matrix * luisa::make_float4(model_position[0], model_position[1], model_position[2], 1.0f);
    return luisa::make_float3(mult_position[0], mult_position[1], mult_position[2]);
}

namespace AffineBodyDynamics
{

    inline Eigen::Matrix<float, 3, 12> get_jacobian_dxdq(const luisa::float3& model_position)
    {
        Eigen::Matrix<float, 3, 12> J = Eigen::Matrix<float, 3, 12>::Zero();
        J.block<3, 3>(0, 0)           = float3x3_to_eigen3x3(float3x3::eye(1.0f));
        J.block<3, 3>(0, 3)           = float3x3_to_eigen3x3(float3x3::eye(model_position.x));
        J.block<3, 3>(0, 6)           = float3x3_to_eigen3x3(float3x3::eye(model_position.y));
        J.block<3, 3>(0, 9)           = float3x3_to_eigen3x3(float3x3::eye(model_position.z));
        return J;
    }
    inline auto extract_q_from_affine_matrix(const luisa::float4x4& A)
    {
        float4x3 q;
        q.cols[0] = A[3].xyz();
        q.cols[1] = A[0].xyz();
        q.cols[2] = A[1].xyz();
        q.cols[3] = A[2].xyz();
        return q;
    }
    inline auto extract_q_from_affine_matrix(const Var<luisa::float4x4>& A)
    {
        Var<float4x3> q;
        q.cols[0] = A[3].xyz();
        q.cols[1] = A[0].xyz();
        q.cols[2] = A[1].xyz();
        q.cols[3] = A[2].xyz();
        return q;
    }

    inline void extract_Ap_from_q(const lcs::float4x3& q, float3x3& A, float3& p)
    {
        p    = q[0];
        A[0] = q[1];
        A[1] = q[2];
        A[2] = q[3];
    }
    inline void extract_Ap_from_q(const lcs::float3* q, float3x3& A, float3& p)
    {
        p    = q[0];
        A[0] = q[1];
        A[1] = q[2];
        A[2] = q[3];
    }
    inline void extract_Ap_from_q(const luisa::compute::BufferView<float3>& q,
                                  const luisa::compute::Uint                body_idx,
                                  luisa::compute::Float3x3&                 A,
                                  luisa::compute::Float3&                   p)
    {
        p    = q->read(4 * body_idx + 0);
        A[0] = q->read(4 * body_idx + 1);
        A[1] = q->read(4 * body_idx + 2);
        A[2] = q->read(4 * body_idx + 3);
    }

    template <typename Vec>
    inline auto affine_Jacobian_to_gradient(const Vec& model_position, const Vec& vertex_force)
    {
        return makeFloat4x3(vertex_force,
                            vertex_force * model_position.x,
                            vertex_force * model_position.y,
                            vertex_force * model_position.z);
    }
    template <typename Vec>
    inline auto affine_Jacobian_to_gradient(const Vec& model_position, const Vec& vertex_force, Vec output_force[4])
    {
        output_force[0] = vertex_force;
        output_force[1] = vertex_force * model_position.x;
        output_force[2] = vertex_force * model_position.y;
        output_force[3] = vertex_force * model_position.z;
    }
    template <typename Vec, typename Mat>
    inline auto affine_Jacobian_to_hessian(const Vec& X1, const Vec& X2, const Mat& hessian, Mat output_hessian[10])
    {
        //  0            1            2          3
        // t1            4            5          6
        // t2           t5            7          8
        // t3           t6           t8          9
        //
        //      H           x21*H        x22*H       x23*H
        //  x11*H       x11*x21*H    x11*x22*H   x11*x23*H
        //  x12*H       x12*x21*H    x12*x22*H   x12*x23*H
        //  x13*H       x13*x21*H    x13*x22*H   x13*x23*H

        // Diag
        output_hessian[0] = hessian;
        output_hessian[4] = X1[0] * X2[0] * hessian;
        output_hessian[7] = X1[1] * X2[1] * hessian;
        output_hessian[9] = X1[2] * X2[2] * hessian;

        // Offi-diag
        output_hessian[1] = X2[0] * hessian;
        output_hessian[2] = X2[1] * hessian;
        output_hessian[3] = X2[2] * hessian;

        output_hessian[5] = X1[0] * X2[1] * hessian;
        output_hessian[6] = X1[0] * X2[2] * hessian;
        output_hessian[8] = X1[1] * X2[2] * hessian;
    }

}  // namespace AffineBodyDynamics


}  // namespace lcs