#pragma once

#include "Core/float_n.h"
#include "Core/xbasic_types.h"

namespace lcs
{

#define Identity2x2 luisa::float2x2::eye(1.0f)
#define Identity3x3 luisa::float3x3::eye(1.0f)
#define Identity4x4 luisa::float4x4::eye(1.0f)

#define Zero3x3 luisa::make_float3x3(0.0f)
#define Zero4x4 luisa::make_float4x4(0.0f)

// makeFloat NxN
// diag scalar
// diag vec
// columns

// float2x2
constexpr inline float2x2 makeFloat2x2(const float diag)
{
    return luisa::make_float2x2(diag, 0.0f, 0.0f, diag);
}
constexpr inline float2x2 makeFloat2x2(const float2 diag)
{
    return luisa::make_float2x2(luisa::make_float2(diag[0], 0.0f), luisa::make_float2(0.0f, diag[1]));
}
constexpr inline float2x2 makeFloat2x2(const float2& column0, const float2& column1)
{
    return luisa::make_float2x2(column0, column1);
}

// float3x3
constexpr inline float3x3 makeFloat3x3(const float diag)
{
    return luisa::make_float3x3(luisa::make_float3(diag, 0.0f, 0.0f),
                                luisa::make_float3(0.0f, diag, 0.0f),
                                luisa::make_float3(0.0f, 0.0f, diag));
}
constexpr inline float3x3 makeFloat3x3(const float3& diag)
{
    return luisa::make_float3x3(luisa::make_float3(diag[0], 0.0f, 0.0f),
                                luisa::make_float3(0.0f, diag[1], 0.0f),
                                luisa::make_float3(0.0f, 0.0f, diag[2]));
}
constexpr inline float3x3 makeFloat3x3(const float3& column0, const float3& column1, const float3& column2)
{
    return luisa::make_float3x3(column0, column1, column2);
}

// float4x4
constexpr inline float4x4 makeFloat4x4(const float diag)
{
    return luisa::make_float4x4(luisa::make_float4(diag, 0.0f, 0.0f, 0.0f),
                                luisa::make_float4(0.0f, diag, 0.0f, 0.0f),
                                luisa::make_float4(0.0f, 0.0f, diag, 0.0f),
                                luisa::make_float4(0.0f, 0.0f, 0.0f, diag));
}
constexpr inline float4x4 makeFloat4x4(const float4& diag)
{
    return luisa::make_float4x4(luisa::make_float4(diag[0], 0.0f, 0.0f, 0.0f),
                                luisa::make_float4(0.0f, diag[1], 0.0f, 0.0f),
                                luisa::make_float4(0.0f, 0.0f, diag[2], 0.0f),
                                luisa::make_float4(0.0f, 0.0f, 0.0f, diag[3]));
}
constexpr inline float4x4 makeFloat4x4(const float4& column0, const float4& column1, const float4& column2, const float4& column3)
{
    return luisa::make_float4x4(column0, column1, column2, column3);
}

// Var<float2x2>
inline Var<float2x2> makeFloat2x2(const Var<float> x)
{
    return luisa::compute::make_float2x2(luisa::compute::make_float2(x, 0.0f),
                                         luisa::compute::make_float2(0.0f, x));
}
inline Var<float2x2> makeFloat2x2(const Var<float2> diag)
{
    return luisa::compute::make_float2x2(luisa::compute::make_float2(diag[0], 0.0f),
                                         luisa::compute::make_float2(0.0f, diag[1]));
}
inline Var<float2x2> makeFloat2x2(const Var<float2>& column0, const Var<float2>& column1)
{
    return luisa::compute::make_float2x2(column0, column1);
}

// Var<float3x3>
inline Var<float3x3> makeFloat3x3(const Var<float> diag)
{
    return luisa::compute::make_float3x3(luisa::compute::make_float3(diag, 0.0f, 0.0f),
                                         luisa::compute::make_float3(0.0f, diag, 0.0f),
                                         luisa::compute::make_float3(0.0f, 0.0f, diag));
}
inline Var<float3x3> makeFloat3x3(const Var<float3>& column0, const Var<float3>& column1, const Var<float3>& column2)
{
    return luisa::compute::make_float3x3(column0, column1, column2);
}
inline Var<float3x3> makeFloat3x3(const Var<float3> diag)
{
    return luisa::compute::make_float3x3(luisa::compute::make_float3(diag[0], 0.0f, 0.0f),
                                         luisa::compute::make_float3(0.0f, diag[1], 0.0f),
                                         luisa::compute::make_float3(0.0f, 0.0f, diag[2]));
}

// Var<float4x4>
inline Var<float4x4> makeFloat4x4(const Var<float> x)
{
    return luisa::compute::make_float4x4(luisa::compute::make_float4(x, 0.0f, 0.0f, 0.0f),
                                         luisa::compute::make_float4(0.0f, x, 0.0f, 0.0f),
                                         luisa::compute::make_float4(0.0f, 0.0f, x, 0.0f),
                                         luisa::compute::make_float4(0.0f, 0.0f, 0.0f, x));
}
inline Var<float4x4> makeFloat4x4(const Var<float4>& column0,
                                  const Var<float4>& column1,
                                  const Var<float4>& column2,
                                  const Var<float4>& column3)
{
    return luisa::compute::make_float4x4(column0, column1, column2, column3);
}
inline Var<float4x4> makeFloat4x4(const Var<float> x, const Var<float4> diag)
{
    return luisa::compute::make_float4x4(luisa::compute::make_float4(diag[0], 0.0f, 0.0f, 0.0f),
                                         luisa::compute::make_float4(0.0f, diag[1], 0.0f, 0.0f),
                                         luisa::compute::make_float4(0.0f, 0.0f, diag[2], 0.0f),
                                         luisa::compute::make_float4(0.0f, 0.0f, 0.0f, diag[3]));
}

// 2x3
// 2x4
// 3x2
// 3x4
// 4x2
// 4x3

// float2x3
[[nodiscard]] inline float2x3 makeFloat2x3(const float3& column0, const float3& column1) noexcept
{
    float2x3 mat;
    mat.cols[0] = column0;
    mat.cols[1] = column1;
    return mat;
}
[[nodiscard]] inline Var<float2x3> makeFloat2x3(const Var<float3>& column0, const Var<float3>& column1) noexcept
{
    Var<float2x3> mat;
    mat.cols[0] = column0;
    mat.cols[1] = column1;
    return mat;
}

// float2x3
[[nodiscard]] inline float2x4 makeFloat2x4(const float4& column0, const float4& column1) noexcept
{
    float2x4 mat;
    mat.cols[0] = column0;
    mat.cols[1] = column1;
    return mat;
}
[[nodiscard]] inline Var<float2x4> makeFloat2x4(const Var<float4>& column0, const Var<float4>& column1) noexcept
{
    Var<float2x4> mat;
    mat.cols[0] = column0;
    mat.cols[1] = column1;
    return mat;
}

// float3x2
[[nodiscard]] inline float3x2 makeFloat3x2(const float2& column0, const float2& column1, const float2& column2) noexcept
{
    float3x2 mat;
    mat.cols[0] = column0;
    mat.cols[1] = column1;
    mat.cols[2] = column2;
    return mat;
}
[[nodiscard]] inline Var<float3x2> makeFloat3x2(const Var<float2>& column0,
                                                const Var<float2>& column1,
                                                const Var<float2>& column2) noexcept
{
    Var<float3x2> mat;
    mat.cols[0] = column0;
    mat.cols[1] = column1;
    mat.cols[2] = column2;
    return mat;
}


[[nodiscard]] inline float2x3 transpose_3x2(const float3x2& mat) noexcept
{
    return makeFloat2x3(luisa::make_float3(mat.cols[0][0], mat.cols[1][0], mat.cols[2][0]),
                        luisa::make_float3(mat.cols[0][1], mat.cols[1][1], mat.cols[2][1]));
}
[[nodiscard]] inline float3x2 transpose_2x3(const float2x3& mat) noexcept
{
    return makeFloat3x2(luisa::make_float2(mat.cols[0][0], mat.cols[1][0]),
                        luisa::make_float2(mat.cols[0][1], mat.cols[1][1]),
                        luisa::make_float2(mat.cols[0][2], mat.cols[1][2]));
}
[[nodiscard]] inline Var<float2x3> transpose_3x2(const Var<float3x2>& mat) noexcept
{
    return makeFloat2x3(luisa::compute::make_float3(mat.cols[0][0], mat.cols[1][0], mat.cols[2][0]),
                        luisa::compute::make_float3(mat.cols[0][1], mat.cols[1][1], mat.cols[2][1]));
}
[[nodiscard]] inline Var<float3x2> transpose_2x3(const Var<float2x3>& mat) noexcept
{
    return makeFloat3x2(luisa::compute::make_float2(mat.cols[0][0], mat.cols[1][0]),
                        luisa::compute::make_float2(mat.cols[0][1], mat.cols[1][1]),
                        luisa::compute::make_float2(mat.cols[0][2], mat.cols[1][2]));
}

// float3x4
[[nodiscard]] inline float3x4 makeFloat3x4(const float4& column0, const float4& column1, const float4& column2) noexcept
{
    float3x4 mat;
    mat.cols[0] = column0;
    mat.cols[1] = column1;
    mat.cols[2] = column2;
    return mat;
}
[[nodiscard]] inline Var<float3x4> makeFloat3x4(const Var<float4>& column0,
                                                const Var<float4>& column1,
                                                const Var<float4>& column2) noexcept
{
    Var<float3x4> mat;
    mat.cols[0] = column0;
    mat.cols[1] = column1;
    mat.cols[2] = column2;
    return mat;
}

// float4x2
[[nodiscard]] inline float4x2 makeFloat4x2(const float2& column0,
                                           const float2& column1,
                                           const float2& column2,
                                           const float2& column3) noexcept
{
    float4x2 mat;
    mat.cols[0] = column0;
    mat.cols[1] = column1;
    mat.cols[2] = column2;
    mat.cols[3] = column3;
    return mat;
}
[[nodiscard]] inline Var<float4x2> makeFloat4x2(const Var<float2>& column0,
                                                const Var<float2>& column1,
                                                const Var<float2>& column2,
                                                const Var<float2>& column3) noexcept
{
    Var<float4x2> mat;
    mat.cols[0] = column0;
    mat.cols[1] = column1;
    mat.cols[2] = column2;
    mat.cols[3] = column3;
    return mat;
}

// float4x3
[[nodiscard]] inline float4x3 makeFloat4x3(const float3& column0,
                                           const float3& column1,
                                           const float3& column2,
                                           const float3& column3) noexcept
{
    float4x3 mat;
    mat.cols[0] = column0;
    mat.cols[1] = column1;
    mat.cols[2] = column2;
    mat.cols[3] = column3;
    return mat;
}
[[nodiscard]] inline Var<float4x3> makeFloat4x3(const Var<float3>& column0,
                                                const Var<float3>& column1,
                                                const Var<float3>& column2,
                                                const Var<float3>& column3) noexcept
{
    Var<float4x3> mat;
    mat.cols[0] = column0;
    mat.cols[1] = column1;
    mat.cols[2] = column2;
    mat.cols[3] = column3;
    return mat;
}
[[nodiscard]] inline Var<float4x3> add(const Var<float4x3>& left, const Var<float4x3>& right) noexcept
{
    Var<float4x3> output;
    output.cols[0] = left.cols[0] + right.cols[0];
    output.cols[1] = left.cols[1] + right.cols[1];
    output.cols[2] = left.cols[2] + right.cols[2];
    output.cols[3] = left.cols[3] + right.cols[3];
    return output;
}


// When N != L
template <size_t M, size_t N, size_t L, std::enable_if_t<(N != L), int> = 0>
[[nodiscard]] XMatrix<L, N> mult(const XMatrix<M, N>& left, const XMatrix<L, M>& right)
{
    XMatrix<L, N> output;
    for (uint j = 0; j < L; ++j)
    {  // output column
        for (uint i = 0; i < N; ++i)
        {  // output row
            output.cols[j][i] = 0.0f;
            for (uint k = 0; k < M; ++k)
            {
                output.cols[j][i] += left.cols[k][i] * right.cols[j][k];
            }
        }
    }
    return output;
}
// When N == L
template <size_t M, size_t N, size_t L, std::enable_if_t<(N == L), int> = 0>
[[nodiscard]] luisa::Matrix<float, N> mult(const XMatrix<M, N>& left, const XMatrix<L, M>& right)
{
    luisa::Matrix<float, N> output;
    for (uint j = 0; j < N; ++j)
    {
        for (uint i = 0; i < N; ++i)
        {
            output[j][i] = 0.0f;
            for (uint k = 0; k < M; ++k)
            {
                output[j][i] += left.cols[k][i] * right.cols[j][k];
            }
        }
    }
    return output;
}
template luisa::Matrix<float, 3> mult(const XMatrix<2, 3>&, const XMatrix<3, 2>&);
template luisa::Matrix<float, 2> mult(const XMatrix<3, 2>&, const XMatrix<2, 3>&);


// When N != L
template <size_t M, size_t N, size_t L, std::enable_if_t<(N != L), int> = 0>
[[nodiscard]] Var<XMatrix<L, N>> mult(const Var<XMatrix<M, N>>& left, const Var<XMatrix<L, M>>& right)
{
    Var<XMatrix<L, N>> output;
    for (uint j = 0; j < L; ++j)
    {  // output column
        for (uint i = 0; i < N; ++i)
        {  // output row
            output.cols[j][i] = 0.0f;
            for (uint k = 0; k < M; ++k)
            {
                output.cols[j][i] += left.cols[k][i] * right.cols[j][k];
            }
        }
    }
    return output;
}
// When N == L
template <size_t M, size_t N, size_t L, std::enable_if_t<(N == L), int> = 0>
[[nodiscard]] Var<luisa::Matrix<float, N>> mult(const Var<XMatrix<M, N>>& left, const Var<XMatrix<L, M>>& right)
{
    Var<luisa::Matrix<float, N>> output;
    for (uint j = 0; j < N; ++j)
    {
        for (uint i = 0; i < N; ++i)
        {
            output[j][i] = 0.0f;
            for (uint k = 0; k < M; ++k)
            {
                output[j][i] += left.cols[k][i] * right.cols[j][k];
            }
        }
    }
    return output;
}
template Var<luisa::Matrix<float, 3>> mult(const Var<XMatrix<2, 3>>&, const Var<XMatrix<3, 2>>&);
template Var<luisa::Matrix<float, 2>> mult(const Var<XMatrix<3, 2>>&, const Var<XMatrix<2, 3>>&);

template <size_t M, size_t N>
[[nodiscard]] luisa::Vector<float, N> mult(const XMatrix<M, N>& left, const luisa::Vector<float, M>& right)
{
    luisa::Vector<float, N> output;
    for (uint i = 0; i < N; ++i)
    {
        output[i] = 0.0f;
        for (uint j = 0; j < M; ++j)
        {
            output[i] += left.cols[j][i] * right[j];
        }
    }
    return output;
}
template <size_t M, size_t N>
[[nodiscard]] Var<luisa::Vector<float, N>> mult(const Var<XMatrix<M, N>>&           left,
                                                const Var<luisa::Vector<float, M>>& right)
{
    Var<luisa::Vector<float, N>> output;
    for (uint i = 0; i < N; ++i)
    {
        output[i] = 0.0f;
        for (uint j = 0; j < M; ++j)
        {
            output[i] += left.cols[j][i] * right[j];
        }
    }
    return output;
}
template <size_t M, size_t N>
[[nodiscard]] auto mult(const Var<XMatrix<M, N>>& left, const Var<float>& alpha)
{
    Var<XMatrix<M, N>> output;
    for (uint j = 0; j < M; ++j)
    {  // output column
        output.cols[j] = alpha * left.cols[j];
    }
    return output;
}
template <size_t M, size_t N>
[[nodiscard]] auto mult(const Var<float>& alpha, const Var<XMatrix<M, N>>& left)
{
    Var<XMatrix<M, N>> output;
    for (uint j = 0; j < M; ++j)
    {  // output column
        output.cols[j] = alpha * left.cols[j];
    }
    return output;
}

// auto mult(const Var<XMatrix<2, 3>>& left, const Var<XMatrix<3, 2>>& right);
// auto mult(const Var<XMatrix<3, 2>>& left, const Var<XMatrix<2, 3>>& right);

// [[nodiscard]] inline Var<float2x2> mult(const Var<float3x2>& left, const Var<float2x3>& right) noexcept
// {
//     Var<float2x2> output;
//     for (int i = 0; i < 2; ++i) { // row
//         for (int j = 0; j < 2; ++j) { // col
//             output[j][i] = left.cols[0][i] * right.cols[j][0]
//                          + left.cols[1][i] * right.cols[j][1]
//                          + left.cols[2][i] * right.cols[j][2];
//         }
//     }
//     return output;
// }
// [[nodiscard]] inline Var<float2> mult(const Var<float3x2>& left, const Var<float3>& right) noexcept
// {
//     Var<float2> output;
//     for (int i = 0; i < 2; ++i) { // row
//         output[i] = left.cols[0][i] * right[0]
//                   + left.cols[1][i] * right[1]
//                   + left.cols[2][i] * right[2];
//     }
//     return output;
// }
// [[nodiscard]] inline Var<float4x3> mult(const Var<float4x3>& mat, const Var<float> alpha) noexcept
// {
//     Var<float4x3> output;
//     output.cols[0] *= alpha;
//     output.cols[1] *= alpha;
//     output.cols[2] *= alpha;
//     output.cols[3] *= alpha;
//     return mat;
// }


template <typename Mat>
inline float determinant_mat(const Mat& mat)
{
    return luisa::determinant(mat);
}
template <typename Mat>
inline Var<float> determinant_mat(const Var<Mat>& mat)
{
    return luisa::compute::determinant(mat);
}

template <typename Mat>
inline auto transpose_mat(const Mat& mat)
{
    return luisa::transpose(mat);
}
template <typename Mat>
inline auto transpose_mat(const Var<Mat>& mat)
{
    return luisa::compute::transpose(mat);
}

inline float2 get_diag(const float2x2& mat)
{
    return luisa::make_float2(mat[0][0], mat[1][1]);
}
inline float3 get_diag(const float3x3& mat)
{
    return luisa::make_float3(mat[0][0], mat[1][1], mat[2][2]);
}
inline float4 get_diag(const float4x4& mat)
{
    return luisa::make_float4(mat[0][0], mat[1][1], mat[2][2], mat[3][3]);
}
inline Var<float2> get_diag(const Var<float2x2>& mat)
{
    return luisa::compute::make_float2(mat[0][0], mat[1][1]);
}
inline Var<float3> get_diag(const Var<float3x3>& mat)
{
    return luisa::compute::make_float3(mat[0][0], mat[1][1], mat[2][2]);
}
inline Var<float4> get_diag(const Var<float4x4>& mat)
{
    return luisa::compute::make_float4(mat[0][0], mat[1][1], mat[2][2], mat[3][3]);
}

template <size_t N>
static inline auto trace_mat(luisa::Matrix<float, N> mat)
{
    return sum_vec(get_diag(mat));
}


inline float3x3 kronecker_product(const float3& left, const float3& right)
{
    return makeFloat3x3(left[0] * right, left[1] * right, left[2] * right);
}
inline float4x4 kronecker_product(const float4& left, const float4& right)
{
    return makeFloat4x4(left[0] * right, left[1] * right, left[2] * right, left[3] * right);
}
template <typename Vec, uint N>
inline void kronecker_product(Vec output[N], const Vec& left, const Vec& right)
{
    for (uint i = 0; i < N; i++)
    {
        output[i] = left[i] * right;
    }
}
template <typename Vec>
inline void kronecker_product(Vec output[4], const Vec& left, const Vec& right)
{
    output[0] = left[0] * right;
    output[1] = left[1] * right;
    output[2] = left[2] * right;
    output[3] = left[3] * right;
}

inline Var<float3x3> kronecker_product(const Var<float3>& left, const Var<float3>& right)
{
    return makeFloat3x3(left[0] * right, left[1] * right, left[2] * right);
}
inline Var<float4x4> kronecker_product(const Var<float4>& left, const Var<float4>& right)
{
    return makeFloat4x4(left[0] * right, left[1] * right, left[2] * right, left[3] * right);
}

inline auto outer_product(const float2& left, const float2& right)
{
    return makeFloat2x2(left * right[0], left * right[1]);
}
inline auto outer_product(const float3& left, const float3& right)
{
    return makeFloat3x3(left * right[0], left * right[1], left * right[2]);
}
inline auto outer_product(const float4& left, const float4& right)
{
    return makeFloat4x4(left * right[0], left * right[1], left * right[2], left * right[3]);
}
inline auto outer_product(const Var<float2>& left, const Var<float2>& right)
{
    return makeFloat2x2(left * right[0], left * right[1]);
}
inline auto outer_product(const Var<float3>& left, const Var<float3>& right)
{
    return makeFloat3x3(left * right[0], left * right[1], left * right[2]);
}
inline auto outer_product(const Var<float4>& left, const Var<float4>& right)
{
    return makeFloat4x4(left * right[0], left * right[1], left * right[2], left * right[3]);
}


};  // namespace lcs