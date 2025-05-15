#pragma once

#include "Core/xbasic_types.h"
#include "Core/constant_value.h"
#include "luisa/core/mathematics.h"

namespace lcsv
{

using Float2 = luisa::float2;
using Float3 = luisa::float3;
using Float4 = luisa::float4;
using Int2 = luisa::uint2;
using Int3 = luisa::uint3;
using Int4 = luisa::uint4;
using Uchar2 = luisa::ubyte2;
using Uchar4 = luisa::ubyte4;
using Float2x2 = luisa::float2x2;
using Float3x3 = luisa::float3x3;
using Float4x4 = luisa::float4x4;
// using Float2x3 = luisa::float2x3;
// using Float3x2 = luisa::float3x2;
using Float2x3 = luisa::float2x3;
using Float4x3 = luisa::float4x3;
using ElementOffset = luisa::ubyte4;




#define Float3_zero make<Float3>(0)
#define Float3_one  make<Float3>(1)
#define FLOAT3_MAX  make<Float3>(Float_max)
#define FLOAT3_EPSILON make<Float3>(EPSILON)

#define Color_Rre    luisa::make_float4(0.9f, 0.1f, 0.1f, 1.f)
#define Color_Green  luisa::make_float4(0.1f, 0.9f, 0.1f, 1.f)
#define Color_Blue   luisa::make_float4(0.1f, 0.1f, 0.9f, 1.f)
#define Color_Yellow luisa::make_float4(0.9f, 0.9f, 0.1f, 1.f)
#define Color_Orange luisa::make_float4(0.9f, 0.5f, 0.1f, 1.f)
#define Color_Purple luisa::make_float4(0.5f, 0.1f, 0.9f, 1.f)
#define Color_Cyan   luisa::make_float4(0.1f, 0.9f, 0.9f, 1.f)
#define Zero2        luisa::make_float2(0.f, 0.f)
#define Zero3        luisa::make_float3(0.f, 0.f, 0.f)
#define Zero4        luisa::make_float4(0.f, 0.f, 0.f, 0.f)


template<typename Vec> inline Vec normalize_vec(const Vec& vec) {return luisa::normalize(vec);}
template<typename Vec> inline auto length_vec(const Vec& vec) { return luisa::length(vec); }
template<typename Vec> inline Vec reverse_vec(const Vec& vec) { return 1.f / vec;}
template<typename Vec> inline Vec abs_vec(const Vec& vec) { return luisa::abs(vec); }

template<typename Vec> inline Vec cross_vec(const Vec& vec1, const Vec& vec2)   { return luisa::cross(vec1, vec2);   }
template<typename Vec> inline auto dot_vec(const Vec& vec1, const Vec& vec2)    { return luisa::dot(vec1, vec2);     }
template<typename Vec> inline Vec max_vec(const Vec& vec1, const Vec& vec2) { return luisa::max(vec1, vec2); }
template<typename Vec> inline Vec min_vec(const Vec& vec1, const Vec& vec2) { return luisa::min(vec1, vec2); }

template<typename Vec> inline auto safe_length_vec(const Vec& vec) { return length_vec(vec) + lcsv::Epsilon; }

// template<typename Vec, uint N = Meta::get_vec_length<Vec>()> 
// inline float sum_vec(const Vec& vec) { 
//     float value = vec[0];
//     for(uint i = 1; i < N; i++){ value += vec[i];} 
//     return value;
// }

template<typename Vec, uint N> inline auto min_component_vec(const Vec& vec) 
{ 
    auto min_value = vec[0];
    for (uint i = 1; i < N; i++){ min_value = min_scalar(min_value, vec[i]);} 
    return min_value;
}
template<typename Vec, uint N> inline auto max_component_vec(const Vec& vec) 
{ 
    auto max_value = vec[0];
    for (uint i = 1; i < N; i++){ max_value = max_scalar(max_value, vec[i]); } 
    return max_value;
}

template<typename Vec, uint N>
inline float infinity_norm_vec(const Vec& vec)
{
    return max_component_vec<Vec, N>(abs_vec(vec));
}

template<typename Vec> 
inline auto length_squared_vec(const Vec& vec) 
{ 
    return dot_vec(vec, vec);
}

template<typename Vec>
inline Vec clamp_vec(const Vec& vec, const Vec& lower, const Vec& upper) 
{
    return max_vec(min_vec(vec, upper), lower);
}
template<typename Vec> 
inline constexpr Vec lerp_vec(const Vec& left, const Vec& right, const float lerp_value) 
{ 
    return left + lerp_value * (right - left);
}

template<typename T, uint N> 
inline bool is_inf_vec(const luisa::Vector<T, N>& vec) 
{ 
    bool is_inf = false; 
    for (uint i = 0; i < N; i++) {
        if(is_inf_scalar(vec[i])) {
            is_inf = true;
        }
    } 
    return is_inf; 
}
template<typename T, uint N> 
inline bool is_nan_vec(const luisa::Vector<T, N>& vec) 
{ 
    bool is_nan = false; 
    for (uint i = 0; i < N; i++) {
        if(is_nan_scalar(vec[i])) {
            is_nan = true;
        }
    } 
    return is_nan; 
}


template<typename T, size_t N> 
inline float sum_vec(luisa::Vector<T, N> vec) { 
    float value = vec[0];
    for (uint i = 1; i < N; i++){ value += vec[i];} 
    return value;
}


template<typename Vec> 
inline Vec project_vec(const Vec& vec1, const Vec& vec2) 
{
    auto length_squred_vec2 = dot_vec(vec2, vec2); // u^2
    if (length_squred_vec2 != 0) return (dot_vec(vec1, vec2) / length_squred_vec2) * vec2; // dot(u, v)/dot(v, v)*v (u proj to v)
    else return make<Vec>(0);
}


inline float compute_face_area(const Float3& pos0, const Float3& pos1, const Float3& pos2)
{
    Float3 vec0 = pos1 - pos0;
    Float3 vec1 = pos2 - pos0;
    float area = length_vec(cross_vec(vec0, vec1)) * 0.5f;
    return area;
}
inline Float3 compute_face_normal(const Float3& p1, const Float3& p2, const Float3& p3)
{
	Float3 s = p2 - p1;
	Float3 t = p3 - p1;
	Float3 n = normalize_vec(cross_vec(s, t));
	return n;
}

}; // namespace lcsv