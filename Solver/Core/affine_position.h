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

    luisa::float4x4 result = luisa::make_float4x4(
        luisa::make_float4(v[0], 0.0f, 0.0f, 0.0f),
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

    return luisa::transpose(luisa::make_float4x4(
        luisa::make_float4(1.0f, 0.0f, 0.0f, v[0]),
        luisa::make_float4(0.0f, 1.0f, 0.0f, v[1]),
        luisa::make_float4(0.0f, 0.0f, 1.0f, v[2]),
        luisa::make_float4(0.0f, 0.0f, 0.0f, 1.0f)
    ));
}
static inline luisa::float4x4 rorateX(float angleX) 
{
    float cosX = luisa::cos(angleX);
    float sinX = luisa::sin(angleX);

    // |  1     0      0     0 |
    // |  0   cos(θ) -sin(θ) 0 |
    // |  0   sin(θ)  cos(θ) 0 |
    // |  0     0      0     1 |

    return luisa::transpose(luisa::make_float4x4(
        luisa::make_float4(1.0f,  0.0f, 0.0f, 0.0f),
        luisa::make_float4(0.0f,  cosX, sinX, 0.0f),
        luisa::make_float4(0.0f, -sinX, cosX, 0.0f),
        luisa::make_float4(0.0f,  0.0f, 0.0f, 1.0f)
    ));
}
static inline luisa::float4x4 rorateY(float angleY) 
{
    float cosY = luisa::cos(angleY);
    float sinY = luisa::sin(angleY);

    // |  cos(θ)  0  sin(θ)  0 |
    // |    0     1    0     0 |
    // | -sin(θ)  0  cos(θ)  0 |
    // |    0     0    0     1 |
    return luisa::transpose(luisa::make_float4x4(
        luisa::make_float4(cosY, 0.0f, -sinY, 0.0f),
        luisa::make_float4(0.0f, 1.0f,  0.0f, 0.0f),
        luisa::make_float4(sinY, 0.0f,  cosY, 0.0f),
        luisa::make_float4(0.0f, 0.0f,  0.0f, 1.0f)
    ));
}
static inline luisa::float4x4 rorateZ(float angleZ) 
{
    float cosZ = luisa::cos(angleZ);
    float sinZ = luisa::sin(angleZ);

    // | cos(θ) -sin(θ)  0  0 |
    // | sin(θ)  cos(θ)  0  0 |
    // |   0       0     1  0 |
    // |   0       0     0  1 |

    return luisa::transpose(luisa::make_float4x4(
        luisa::make_float4(cosZ, -sinZ, 0.0f, 0.0f),
        luisa::make_float4(sinZ,  cosZ, 0.0f, 0.0f),
        luisa::make_float4(0.0f,  0.0f, 1.0f, 0.0f),
        luisa::make_float4(0.0f,  0.0f, 0.0f, 1.0f)
    ));
}
static inline luisa::float4x4 rotate(const luisa::float3& axis) 
{
    return rorateX(axis[0]) * rorateY(axis[1]) * rorateZ(axis[2]);
}

inline luisa::float4x4 make_model_matrix(const luisa::float3& t, const luisa::float3& r, const luisa::float3& s)
{
    return translate(t) * rotate(r) * scale(s)  ;
    // return scale(s) * (rotate(r) * translate(t));
    // return translate(t) ;
}

inline luisa::float3 affine_position(const luisa::float4x4& model_matrix, const luisa::float3& model_position)
{
    luisa::float4 mult_position = model_matrix * luisa::make_float4(model_position[0], model_position[1], model_position[2], 1.0f);
    return luisa::make_float3(mult_position[0], mult_position[1], mult_position[2]);
}

inline Eigen::Matrix<float, 3, 12> get_jacobian_dxdq(const luisa::float3& model_position)
{
    Eigen::Matrix<float, 3, 12> J = Eigen::Matrix<float, 3, 12>::Zero();
    J.block<3, 3>(0, 0) = float3x3_to_eigen3x3(Identity3x3);
    J.block<3, 3>(0, 3) = float3x3_to_eigen3x3(luisa::transpose(float3x3(model_position, Zero3,  Zero3)));
    J.block<3, 3>(0, 6) = float3x3_to_eigen3x3(luisa::transpose(float3x3(Zero3,  model_position, Zero3)));
    J.block<3, 3>(0, 9) = float3x3_to_eigen3x3(luisa::transpose(float3x3(Zero3,  Zero3,  model_position)));
    return J;
}
inline auto extract_q_from_affine_matrix(const luisa::float4x4& A)
{
    float4x3 q;
    q.cols[0] = A[3].xyz();
    auto T = luisa::transpose(A);
    q.cols[1] = T[0].xyz();
    q.cols[2] = T[1].xyz();
    q.cols[3] = T[2].xyz();
    return q;
}
inline auto extract_q_from_affine_matrix(const Var<luisa::float4x4>& A)
{
    Var<float4x3> q;
    q.cols[0] = A[3].xyz();
    auto T = luisa::compute::transpose(A);
    q.cols[1] = T[0].xyz();
    q.cols[2] = T[1].xyz();
    q.cols[3] = T[2].xyz();
    return q;
}

inline void extract_Ap_from_q(const lcs::float4x3& q, float3x3& A, float3& p)
{
    p = q[0];
    A[0] = q[1];
    A[1] = q[2];
    A[2] = q[3];
    A = luisa::transpose(A);
}
inline void extract_Ap_from_q(const lcs::float3* q, float3x3& A, float3& p)
{
    p = q[0];
    A[0] = q[1];
    A[1] = q[2];
    A[2] = q[3];
    A = luisa::transpose(A);
}

template <typename Vec>
inline auto affine_Jacobian_to_gradient(const Vec& rest_position, const Vec& vertex_force)
{
    return makeFloat4x3(
        vertex_force,
        vertex_force.x * rest_position,
        vertex_force.y * rest_position,
        vertex_force.z * rest_position
    );
}
template <typename Vec>
inline auto affine_Jacobian_to_gradient(const Vec& model_position, const Vec& vertex_force, Vec output_force[4])
{
    output_force[0] = vertex_force;
    output_force[1] = vertex_force.x * model_position;
    output_force[2] = vertex_force.y * model_position;
    output_force[3] = vertex_force.z * model_position;
}
template <typename Vec, typename Mat>
inline auto affine_Jacobian_to_hessian(const Vec& X1, const Vec& X2, const Mat& hessian, Mat output_hessian[10])
{
    //  0            1            2          3
    // t1            4            5          6
    // t2           t5            7          8
    // t3           t6           t8          9 
    // 
    //  H           H.c1 * x2T   H.c2 * x2T  H.c3 * x2T
    //  x1 * H.r1   H11*x1*x2T   H12*x1*x2T  H13*x1*x2T 
    //  x1 * H.r2   H21*x1*x2T   H22*x1*x2T  H23*x1*x2T
    //  x1 * H.r3   H31*x1*x2T   H32*x1*x2T  H33*x1*x2T

    // Diag
    Mat x1x2T = outer_product(X1, X2);
    output_hessian[0] = hessian;
    output_hessian[4] = hessian[0][0] * x1x2T;
    output_hessian[7] = hessian[1][1] * x1x2T;
    output_hessian[9] = hessian[2][2] * x1x2T;

    // Offi-diag
    Mat trans = transpose_mat(hessian);
    output_hessian[1] = outer_product(trans[0], X2);
    output_hessian[2] = outer_product(trans[1], X2);
    output_hessian[3] = outer_product(trans[2], X2);

    output_hessian[5] = hessian[1][0] * outer_product(X1, X2);
    output_hessian[6] = hessian[2][0] * outer_product(X1, X2);
    output_hessian[8] = hessian[2][1] * outer_product(X1, X2);

}

}