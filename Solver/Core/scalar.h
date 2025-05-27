#pragma once

#include <luisa/core/mathematics.h>
#include "Core/constant_value.h"
#include "luisa/dsl/var.h"
#include "luisa/dsl/builtin.h"

namespace lcsv
{


template<typename T> inline constexpr T select(const bool condition, const T& true_return, const T& false_return) { if (condition) return true_return; else return false_return; }
template<typename T> inline constexpr T select(const luisa::compute::Var<bool>& condition, const T& true_return, const T& false_return) { return luisa::compute::select(false_return, true_return, condition); }
template<typename T> inline constexpr T abs_scalar(const T value) { return value > T(0) ? value : -value; }
template<typename T> inline constexpr T lerp_scalar(const T left, const T right, const T lerp_value) { return left + lerp_value * (right - left);}
// inline constexpr uint abs_scalar(CONST(int) value) { return value > (0) ? value : -value; }
// inline constexpr float abs_scalar(CONST(float) value) { return value > (0.f) ? value : -value; }


template<typename T1, typename T2> inline constexpr T1 max_scalar(const T1& left, const T2& right) { return left > right ? left : right; }
template<typename T1, typename T2> inline constexpr T1 min_scalar(const T1& left, const T2& right) { return left < right ? left : right; }

template<typename T1, typename T2> 
inline constexpr luisa::compute::Var<T1> max_scalar(const luisa::compute::Var<T1>& left, const luisa::compute::Var<T2>& right) 
{ 
    return luisa::compute::select(right, left, left > right); 
}
template<typename T1, typename T2> 
inline constexpr luisa::compute::Var<T1> min_scalar(const luisa::compute::Var<T1>& left, const luisa::compute::Var<T2>& right) 
{ 
    return luisa::compute::select(right, left, left < right); 
}

// template<typename T1, typename T2, typename T3> inline constexpr T1 max_scalar(const T1& v1, const T2& v2, const T3& v3) 
// { return (v1 >= v2 && v1 >= v3) ? v1 : (v2 >= v1 && v2 >= v3) ? v2 : v3; }
// template<typename T1, typename T2, typename T3> inline constexpr T1 min_scalar(const T1& v1, const T2& v2, const T3& v3) 
// { return (v1 <= v2 && v1 <= v3) ? v1 : (v2 <= v1 && v2 <= v3) ? v2 : v3; }
// template<typename T1, typename T2, typename T3, typename T4> inline constexpr T1 max_scalar(const T1& v1, const T2& v2, const T3& v3, const T4& v4) 
// { return (v1 >= v2 && v1 >= v3 && v1 >= v4) ? v1 : (v2 >= v1 && v2 >= v3 && v2 >= v4) ? v2 : (v3 >= v1 && v3 >= v2 && v3 >= v4) ? v3 : v4; }
// template<typename T1, typename T2, typename T3, typename T4> inline constexpr T1 min_scalar(const T1& v1, const T2& v2, const T3& v3, const T4& v4) 
// { return (v1 <= v2 && v1 <= v3 && v1 <= v4) ? v1 : (v2 <= v1 && v2 <= v3 && v2 <= v4) ? v2 : (v3 <= v1 && v3 <= v2 && v3 <= v4) ? v3 : v4; }

template<typename T> inline constexpr void swap_scalar(T& left, T& right) { T tmp = left; left = right; right = tmp; }
template<typename T1, typename T2, typename T3>  inline constexpr T1 clamp_scalar(const T1& value, const T2& lower, const T3& upper) { return max_scalar(min_scalar(value, upper), lower);}

template<typename T> inline constexpr T square_scalar(const T value) { return value * value; }
template<typename T> inline constexpr T is_equal_scalar(const T left, const T right)   { return abs_scalar(left - right) < Epsilon; }


                    //  inline constexpr int floor_scalar(const float value)  { return luisa::floor(value); }
template<typename T> inline constexpr T sign_scalar(const T value)  { return luisa::sign(value); }
template<typename T> inline constexpr T sin_scalar(const T value)   { return luisa::sin(value); }
template<typename T> inline constexpr T cos_scalar(const T value)   { return luisa::cos(value); }
template<typename T> inline constexpr T tan_scalar(const T value)   { return luisa::tan(value); }
template<typename T> inline constexpr T asin_scalar(const T value)   { return luisa::asin(value); }
template<typename T> inline constexpr T acos_scalar(const T value)   { return luisa::acos(value); }
template<typename T> inline constexpr T atan_scalar(const T value)   { return luisa::atan(value); }
template<typename T> inline constexpr T atan2_scalar(const T y, const T x)   { return x != T(0) ? luisa::atan2(y, x) : T(0); }
template<typename T> inline constexpr T sqrt_scalar(const T value)  { return luisa::sqrt(value); }
template<typename T> inline constexpr T rsqrt_scalar(const T value) { return T(1.0f) / sqrt_scalar(value); }
template<typename T> inline constexpr T is_inf_scalar(const T value) { return luisa::isinf(value); }
template<typename T> inline constexpr T is_nan_scalar(const T value) { return luisa::isnan(value); }
template<typename T1, typename T2> inline constexpr T1 pow_scalar(const T1& x, const T2& y) {return luisa::pow(x, y);}

                    //  inline constexpr luisa::compute::Var<int> floor_scalar(const luisa::compute::Var<float> value)  { return luisa::compute::floor(value); }
template<typename T> inline constexpr luisa::compute::Var<T> sign_scalar(const luisa::compute::Var<T> value)  { return luisa::compute::sign(value); }
template<typename T> inline constexpr luisa::compute::Var<T> sin_scalar(const luisa::compute::Var<T> value)   { return luisa::compute::sin(value); }
template<typename T> inline constexpr luisa::compute::Var<T> cos_scalar(const luisa::compute::Var<T> value)   { return luisa::compute::cos(value); }
template<typename T> inline constexpr luisa::compute::Var<T> tan_scalar(const luisa::compute::Var<T> value)   { return luisa::compute::tan(value); }
template<typename T> inline constexpr luisa::compute::Var<T> asin_scalar(const luisa::compute::Var<T> value)  { return luisa::compute::asin(value); }
template<typename T> inline constexpr luisa::compute::Var<T> acos_scalar(const luisa::compute::Var<T> value)  { return luisa::compute::acos(value); }
template<typename T> inline constexpr luisa::compute::Var<T> atan_scalar(const luisa::compute::Var<T> value)  { return luisa::compute::atan(value); }
template<typename T> inline constexpr luisa::compute::Var<T> atan2_scalar(const luisa::compute::Var<T> y, const T x)   { return x != T(0) ? luisa::compute::atan2(y, x) : T(0); }
template<typename T> inline constexpr luisa::compute::Var<T> sqrt_scalar(const luisa::compute::Var<T> value)  { return luisa::compute::sqrt(value); }
template<typename T> inline constexpr luisa::compute::Var<T> rsqrt_scalar(const luisa::compute::Var<T> value) { return T(1.0f) / sqrt_scalar(value); }
template<typename T> inline constexpr luisa::compute::Var<T> is_inf_scalar(const luisa::compute::Var<T> value) { return luisa::compute::isinf(value); }
template<typename T> inline constexpr luisa::compute::Var<T> is_nan_scalar(const luisa::compute::Var<T> value) { return luisa::compute::isnan(value); }
template<typename T1, typename T2> inline constexpr luisa::compute::Var<T1> pow_scalar(const luisa::compute::Var<T1>& x, const luisa::compute::Var<T2>& y) {return luisa::compute::pow(x, y);}

template<typename Int1, typename Int2>
inline Int1 get_dispatch_block(const Int1 num_threads, const Int2 block_dim = 256)
{
    return (num_threads + block_dim - 1) / block_dim;
}
template<typename Int1, typename Int2>
inline Int1 get_dispatch_threads(Int1 num_threads, Int2 block_dim = 256)
{
    Int1 mask_block_dim = ~(block_dim - 1);
    return ((num_threads - 1) & mask_block_dim) + block_dim;
}

/*
#if SIM_USE_SIMD
#include <simd/simd.h>
#elif SIM_USE_GLM
#include <glm/glm.hpp>
#include <glm/detail/qualifier.hpp>
#endif

#if SIM_USE_SIMD
                     inline constexpr int floor_scalar(CONST(float) value)  { return simd::floor(value); }
template<typename T> inline constexpr T sign_scalar(const T value)  { return simd::sign(value); }
template<typename T> inline constexpr T sin_scalar(const T value)   { return simd::sin(value); }
template<typename T> inline constexpr T cos_scalar(const T value)   { return simd::cos(value); }
template<typename T> inline constexpr T tan_scalar(const T value)   { return simd::tan(value); }
template<typename T> inline constexpr T asin_scalar(const T value)   { return simd::asin(value); }
template<typename T> inline constexpr T acos_scalar(const T value)   { return simd::acos(value); }
template<typename T> inline constexpr T atan_scalar(const T value)   { return simd::atan(value); }
template<typename T> inline constexpr T atan2_scalar(const T y, const T x)   { return x != T(0) ? simd::atan2(y, x) : T(0); }
template<typename T> inline constexpr T sqrt_scalar(const T value)  { return simd::sqrt(value); }
template<typename T> inline constexpr T rsqrt_scalar(const T value) { return simd::rsqrt(value); }
template<typename T> inline constexpr T is_inf_scalar(const T value) { return simd::isinf(value); }
template<typename T> inline constexpr T is_nan_scalar(const T value) { return simd::isnan(value); }
template<typename T1, typename T2> inline constexpr T1 pow_scalar(const T1& x, const T2& y) {return simd::pow(x, y);}

#elif SIM_USE_GLM
                     inline constexpr int floor_scalar(CONST(float) value)  { return glm::floor(value); }
template<typename T> inline T sign_scalar(const T value)  { return glm::sign(value);}
template<typename T> inline constexpr T sin_scalar(const T value)   { return glm::sin(value); }
template<typename T> inline constexpr T cos_scalar(const T value)   { return glm::cos(value); }
template<typename T> inline constexpr T tan_scalar(const T value)   { return glm::tan(value); }
template<typename T> inline constexpr T asin_scalar(const T value)   { return glm::asin(value); }
template<typename T> inline constexpr T acos_scalar(const T value)   { return glm::acos(value); }
template<typename T> inline constexpr T atan_scalar(const T value)   { return glm::atan(value); }
template<typename T> inline constexpr T atan2_scalar(const T y, const T x)   { return x != T(0) ? glm::atan2(y, x) : T(0); }
template<typename T> inline constexpr T sqrt_scalar(const T value)  { return glm::sqrt(value);}
template<typename T> inline constexpr T rsqrt_scalar(const T value) { return 1.f / glm::sqrt(value);}
template<typename T> inline constexpr T is_inf_scalar(const T value) { return glm::isinf(value); }
template<typename T> inline constexpr T is_nan_scalar(const T value) { return glm::isnan(value); }
template<typename T1, typename T2> inline constexpr T1 pow_scalar(const T1& x, const T2& y)  {return glm::pow(x, y);}

#endif

*/


} // namespace lcsv