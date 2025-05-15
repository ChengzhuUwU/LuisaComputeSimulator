#pragma once

#include "Core/float_n.h"
#include "Core/xbasic_types.h"

namespace lcsv
{

#define Identity2x2 luisa::make_float2x2(1.0f)
#define Identity3x3 luisa::make_float3x3(1.0f)
#define Identity4x4 luisa::make_float4x4(1.0f)

#define Zero3x3 luisa::make_float3x3(0.0f)
#define Zero4x4 luisa::make_float4x4(0.0f)

template<typename Mat>
inline float determinant_mat(const Mat& mat) 
{
    return luisa::determinant(mat);
}

template<typename Mat>
inline auto transpose_mat(const Mat& mat) 
{
    return luisa::transpose(mat);
}

template<size_t N>
Float3 get_diag(luisa::Matrix<N> mat) { return luisa::make_float3(mat[0][0], mat[1][1], mat[2][2]);  }

template<size_t N>
static inline float trace_mat(luisa::Matrix<N> mat) 
{
    return sum_vec(get_diag(mat));
}


inline Float3x3 kronecker_product(const Float3& left, const Float3& right)
{
	return make_float3x3(left[0] * right, 
                        left[1] * right, 
                        left[2] * right); 
}
inline void kronecker_product(Float3 output[3], const Float3& left, const Float3& right)
{
	output[0] = left[0] * right;
    output[1] = left[1] * right;
    output[2] = left[2] * right;
}
inline Float4x3 kronecker_product(const Float4& left, const Float3& right)
{
	return luisa::make_float4x3(left[0] * right, 
                          left[1] * right, 
                          left[2] * right,
                          left[3] * right);
}
inline void kronecker_product(Float3 output[4], const Float4& left, const Float3& right)
{
	output[0] = left[0] * right;
    output[1] = left[1] * right;
    output[2] = left[2] * right;
    output[3] = left[3] * right;
}
inline Float2x2 outer_product(const Float2& left, const Float2& right)
{
	return luisa::make_float2x2(left * right[0], 
                          left * right[1]); 
}
inline Float3x3 outer_product(const Float3& left, const Float3& right)
{
	return luisa::make_float3x3(left * right[0], 
                          left * right[1], 
                          left * right[2]); 
}
inline Float4x4 outer_product(const Float4& left, const Float4& right)
{
	return luisa::make_float4x4(left * right[0], 
                          left * right[1], 
                          left * right[2],
                          left * right[3]); 
}

inline void outer_product(Float3x3& result, const Float3& left, const Float3& right)
{
	result[0] = left * right[0];
	result[1] = left * right[1];
	result[2] = left * right[2];
}


}; // namespace lcsv