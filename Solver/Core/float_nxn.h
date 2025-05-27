#pragma once

#include "Core/float_n.h"
#include "Core/xbasic_types.h"

namespace lcsv
{

#define Identity2x2 make_float2x2(1.0f)
#define Identity3x3 make_float3x3(1.0f)
#define Identity4x4 make_float4x4(1.0f)

#define Zero3x3 make_float3x3(0.0f)
#define Zero4x4 make_float4x4(0.0f)

// makeFloat NxN
// diag scalar
// diag vec
// columns

// float2x2
constexpr inline float2x2 makeFloat2x2(const float diag) { return luisa::make_float2x2(
    diag, 0.0f, 
    0.0f, diag
); }
constexpr inline float2x2 makeFloat2x2(const float2 diag) { return luisa::make_float2x2(
    luisa::make_float2(diag[0], 0.0f),
    luisa::make_float2(0.0f, diag[1])
); }
constexpr inline float2x2 makeFloat2x2(const float2& column0, const float2& column1) { return luisa::make_float2x2(
    column0, 
    column1
); }

// float3x3
constexpr inline float3x3 makeFloat3x3(const float diag)  { return luisa::make_float3x3(
    luisa::make_float3(diag, 0.0f, 0.0f), 
    luisa::make_float3(0.0f, diag, 0.0f),
    luisa::make_float3(0.0f, 0.0f, diag)
); }
constexpr inline float3x3 makeFloat3x3(const float3& diag)  { return luisa::make_float3x3(
    luisa::make_float3(diag[0], 0.0f, 0.0f), 
    luisa::make_float3(0.0f, diag[1], 0.0f),
    luisa::make_float3(0.0f, 0.0f, diag[2])
); }
constexpr inline float3x3 makeFloat3x3(const float3& column0, const float3& column1, const float3& column2) { return luisa::make_float3x3(
    column0, 
    column1, 
    column2
); }

// float4x4
constexpr inline float4x4 makeFloat4x4(const float diag)  { return luisa::make_float4x4(
    luisa::make_float4(diag, 0.0f, 0.0f, 0.0f), 
    luisa::make_float4(0.0f, diag, 0.0f, 0.0f),
    luisa::make_float4(0.0f, 0.0f, diag, 0.0f),
    luisa::make_float4(0.0f, 0.0f, 0.0f, diag)
); }
constexpr inline float4x4 makeFloat4x4(const float4& diag)  { return luisa::make_float4x4(
    luisa::make_float4(diag[0], 0.0f, 0.0f, 0.0f), 
    luisa::make_float4(0.0f, diag[1], 0.0f, 0.0f),
    luisa::make_float4(0.0f, 0.0f, diag[2], 0.0f),
    luisa::make_float4(0.0f, 0.0f, 0.0f, diag[3])
); }
constexpr inline float4x4 makeFloat4x4(const float4& column0, const float4& column1, const float4& column2, const float4& column3) { return luisa::make_float4x4(
    column0, 
    column1, 
    column2, 
    column3
); }

// Var<float2x2>
inline Var<float2x2> makeFloat2x2(const Var<float> x) { return luisa::compute::make_float2x2(
    luisa::compute::make_float2(x, 0.0f), 
    luisa::compute::make_float2(0.0f, x)
); }
inline Var<float2x2> makeFloat2x2(const Var<float2> diag) { return luisa::compute::make_float2x2(
    luisa::compute::make_float2(diag[0], 0.0f), 
    luisa::compute::make_float2(0.0f, diag[1])
); }
inline Var<float2x2> makeFloat2x2(const Var<float2>& column0, const Var<float2>& column1) { return luisa::compute::make_float2x2(
    column0, 
    column1
); }

// Var<float3x3>
inline Var<float3x3> makeFloat3x3(const Var<float> diag)  { return luisa::compute::make_float3x3(
    luisa::compute::make_float3(diag, 0.0f, 0.0f), 
    luisa::compute::make_float3(0.0f, diag, 0.0f),
    luisa::compute::make_float3(0.0f, 0.0f, diag)
); }
inline Var<float3x3> makeFloat3x3(const Var<float3>& column0, const Var<float3>& column1, const Var<float3>& column2) { return luisa::compute::make_float3x3(
    column0, 
    column1, 
    column2
); }
inline Var<float3x3> makeFloat3x3(const Var<float3> diag)  { return luisa::compute::make_float3x3(
    luisa::compute::make_float3(diag[0], 0.0f, 0.0f), 
    luisa::compute::make_float3(0.0f, diag[1], 0.0f),
    luisa::compute::make_float3(0.0f, 0.0f, diag[2])
); }

// Var<float4x4>
inline Var<float4x4> makeFloat4x4(const Var<float> x)  { return luisa::compute::make_float4x4(
    luisa::compute::make_float4(x, 0.0f, 0.0f, 0.0f), 
    luisa::compute::make_float4(0.0f, x, 0.0f, 0.0f),
    luisa::compute::make_float4(0.0f, 0.0f, x, 0.0f),
    luisa::compute::make_float4(0.0f, 0.0f, 0.0f, x)
); }
inline Var<float4x4> makeFloat4x4(const Var<float4>& column0, const Var<float4>& column1, const Var<float4>& column2, const Var<float4>& column3) { return luisa::compute::make_float4x4(
    column0, 
    column1, 
    column2, 
    column3
); }
inline Var<float4x4> makeFloat4x4(const Var<float> x, const Var<float4> diag)  { return luisa::compute::make_float4x4(
    luisa::compute::make_float4(diag[0], 0.0f, 0.0f, 0.0f), 
    luisa::compute::make_float4(0.0f, diag[1], 0.0f, 0.0f),
    luisa::compute::make_float4(0.0f, 0.0f, diag[2], 0.0f),
    luisa::compute::make_float4(0.0f, 0.0f, 0.0f, diag[3])
); }



// float2x3
[[nodiscard]] inline float2x3 makeFloat2x3(const float3& column0, const float3& column1) noexcept 
{ 
    return luisa::float2x3{
        column0, 
        column1
    }; 
}
[[nodiscard]] inline Var<float2x3> makeFloat2x3(const Var<float3>& column0, const Var<float3>& column1) noexcept 
{ 
    Var<float2x3> mat;
    mat.cols[0] = column0;
    mat.cols[1] = column1;
    return mat;
}

// float4x3
[[nodiscard]] inline float4x3 makeFloat2x3(const float3& column0, const float3& column1, const float3& column2, const float3& column3) noexcept 
{ 
    return luisa::float4x3{
        column0, 
        column1,
        column2,
        column3,
    }; 
}
[[nodiscard]] inline Var<float4x3> make_float4x3(const Var<float3>& column0, const Var<float3>& column1, const Var<float3>& column2, const Var<float3>& column3) noexcept 
{
    Var<float4x3> mat;
    mat.cols[0] = column0;
    mat.cols[1] = column1;
    mat.cols[2] = column2;
    mat.cols[3] = column3;
    return mat;
}

template<typename Mat> inline float determinant_mat(const Mat& mat) { return luisa::determinant(mat); }
template<typename Mat> inline Var<float> determinant_mat(const Var<Mat>& mat) { return luisa::compute::determinant(mat); }

template<typename Mat> inline auto transpose_mat(const Mat& mat) { return luisa::transpose(mat); }
template<typename Mat> inline auto transpose_mat(const Var<Mat>& mat)  { return luisa::compute::transpose(mat); }

inline float2 get_diag(const float2x2& mat) { return luisa::make_float2(mat[0][0], mat[1][1]);  }
inline float3 get_diag(const float3x3& mat) { return luisa::make_float3(mat[0][0], mat[1][1], mat[2][2]);  }
inline float4 get_diag(const float4x4& mat) { return luisa::make_float4(mat[0][0], mat[1][1], mat[2][2], mat[3][3]);  }
inline Var<float2> get_diag(const Var<float2x2>& mat) { return luisa::compute::make_float2(mat[0][0], mat[1][1]);  }
inline Var<float3> get_diag(const Var<float3x3>& mat) { return luisa::compute::make_float3(mat[0][0], mat[1][1], mat[2][2]);  }
inline Var<float4> get_diag(const Var<float4x4>& mat) { return luisa::compute::make_float4(mat[0][0], mat[1][1], mat[2][2], mat[3][3]);  }

template<size_t N>
static inline auto trace_mat(luisa::Matrix<N> mat) 
{
    return sum_vec(get_diag(mat));
}


inline float3x3 kronecker_product(const float3& left, const float3& right)
{
	return makeFloat3x3(left[0] * right, 
                        left[1] * right, 
                        left[2] * right); 
}
inline float4x4 kronecker_product(const float4& left, const float4& right)
{
	return makeFloat4x4(left[0] * right, 
                        left[1] * right, 
                        left[2] * right,
                        left[3] * right); 
}
template<typename Vec, uint N>
inline void kronecker_product(Vec output[N], const Vec& left, const Vec& right)
{
    for (uint i = 0; i < N; i++) 
    {
        output[i] = left[i] * right;
    }
}
template<typename Vec>
inline void kronecker_product(Vec output[4], const Vec& left, const Vec& right)
{
	output[0] = left[0] * right;
    output[1] = left[1] * right;
    output[2] = left[2] * right;
    output[3] = left[3] * right;
}

inline Var<float3x3> kronecker_product(const Var<float3>& left, const Var<float3>& right)
{
	return makeFloat3x3(left[0] * right, 
                        left[1] * right, 
                        left[2] * right); 
}
inline Var<float4x4> kronecker_product(const Var<float4>& left, const Var<float4>& right)
{
	return makeFloat4x4(left[0] * right, 
                        left[1] * right, 
                        left[2] * right,
                        left[3] * right); 
}

inline auto outer_product(const float2& left, const float2& right)
{
	return makeFloat2x2(left * right[0], 
                          left * right[1]); 
}
inline auto outer_product(const float3& left, const float3& right)
{
	return makeFloat3x3(left * right[0], 
                        left * right[1], 
                        left * right[2]); 
}
inline auto outer_product(const float4& left, const float4& right)
{
	return makeFloat4x4(left * right[0], 
                          left * right[1], 
                          left * right[2],
                          left * right[3]); 
}
inline auto outer_product(const Var<float2>& left, const Var<float2>& right)
{
	return makeFloat2x2(left * right[0], 
                          left * right[1]); 
}
inline auto outer_product(const Var<float3>& left, const Var<float3>& right)
{
	return makeFloat3x3(left * right[0], 
                        left * right[1], 
                        left * right[2]); 
}
inline auto outer_product(const Var<float4>& left, const Var<float4>& right)
{
	return makeFloat4x4(left * right[0], 
                          left * right[1], 
                          left * right[2],
                          left * right[3]); 
}


}; // namespace lcsv