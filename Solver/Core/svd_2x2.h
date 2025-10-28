#pragma once

#include "Core/float_n.h"
#include "Core/float_nxn.h"

namespace lcs
{

inline luisa::float2 rot90(const luisa::float2& x)
{
    return luisa::make_float2(x[1], -x[0]);
}

inline luisa::float2 find_ortho2(const luisa::float2x2& A, const luisa::float2& x, float sqr_eps)
{
    luisa::float2 u(rot90(A[0])), v(rot90(A[1]));
    if (luisa::dot(u, u) > sqr_eps)
    {
        return luisa::normalize(u);
    }
    else if (luisa::dot(v, v) > sqr_eps)
    {
        return luisa::normalize(v);
    }
    else
    {
        return luisa::normalize(rot90(x));
    }
}

inline luisa::float2 eigvalues(const luisa::float2x2& A)
{
    float a00(A[0][0]);
    float a01(A[1][0]);
    float a11(A[1][1]);
    float tmp = a00 - a11;
    float D   = 0.5f * luisa::sqrt(tmp * tmp + 4.0f * a01 * a01);
    float mid = 0.5f * (a00 + a11);
    return luisa::make_float2(mid - D, mid + D);
}
inline luisa::float2x2 eigvectors2(const luisa::float2x2& A, const luisa::float2& lmd)
{
    float eps     = 1e-8;
    float sqr_eps = eps * eps;
    luisa::float2 u = find_ortho2(A - lmd[0] * luisa::float2x2::eye(1.0f), luisa::float2(0.0, 1.0), sqr_eps);
    luisa::float2 v = find_ortho2(A - lmd[1] * luisa::float2x2::eye(1.0f), -u, sqr_eps);

    luisa::float2x2 result = luisa::make_float2x2(u, v);
    return result;
}

inline luisa::compute::Float2 eigvalues(const luisa::compute::Float2x2& A)
{
    luisa::compute::Float a00(A[0][0]);
    luisa::compute::Float a01(A[1][0]);
    luisa::compute::Float a11(A[1][1]);
    luisa::compute::Float tmp = a00 - a11;
    luisa::compute::Float D   = 0.5f * luisa::compute::sqrt(tmp * tmp + 4.0f * a01 * a01);
    luisa::compute::Float mid = 0.5f * (a00 + a11);
    return luisa::compute::make_float2(mid - D, mid + D);
}

inline std::tuple<luisa::float2, luisa::float2x2> sym_eigsolve_2x2(const luisa::float2x2& A)
{
    // float scale   = A.norm();
    float scale       = luisa::sqrt(luisa::dot(A[0], A[0]) + luisa::dot(A[1], A[1]));  // Frobenius norm
    luisa::float2x2 B = A / scale;
    luisa::float2   lmd     = eigvalues(B);
    luisa::float2x2 eigvecs = eigvectors2(B, lmd);
    return std::tuple<luisa::float2, luisa::float2x2>(scale * lmd, eigvecs);
}
inline std::tuple<float2x3, luisa::float2, luisa::float2x2> svd3x2(const float2x3& F)
{
    auto            result = sym_eigsolve_2x2(transpose_2x3(F) * F);
    luisa::float2   sigma  = std::get<0>(result);
    luisa::float2x2 V      = std::get<1>(result);
    for (int i = 0; i < 2; ++i)
    {
        sigma[i] = luisa::sqrt(luisa::max(0.0f, sigma[i]));
    }
    float2x3 U = F * V;
    for (int i = 0; i < 2; i++)
    {
        U[i] = luisa::normalize(U[i]);
    }
    return {U, sigma, luisa::transpose(V)};
}

}  // namespace lcs