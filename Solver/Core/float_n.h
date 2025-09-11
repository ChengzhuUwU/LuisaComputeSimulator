#pragma once

#include "Core/xbasic_types.h"
#include "Core/constant_value.h"
#include "luisa/core/mathematics.h"
#include "luisa/dsl/var.h"


// LUISA_STRUCT(luisa::compute::float2x3, cols1, cols2) {};
// LUISA_STRUCT(luisa::compute::float4x3, cols1, cols2, cols3, cols4) {};
// LUISA_STRUCT(luisa::compute::float3x4, cols1, cols2, cols3) {};




namespace luisa::compute 
{
using Uint = luisa::compute::Var<uint>;
using Uint2 = luisa::compute::Var<uint2>;
using Uint3 = luisa::compute::Var<uint3>;
using Uint4 = luisa::compute::Var<uint4>;
using Float2x3 = luisa::compute::Var<lcs::float2x3>;
using Float2x4 = luisa::compute::Var<lcs::float2x4>;
using Float3x2 = luisa::compute::Var<lcs::float3x2>;
using Float3x4 = luisa::compute::Var<lcs::float3x4>;
using Float4x2 = luisa::compute::Var<lcs::float4x2>;
using Float4x3 = luisa::compute::Var<lcs::float4x3>;

/*

namespace detail 
{

template<>
struct TypeDesc<float4x3> {
    static constexpr luisa::string_view description() noexcept {
        using namespace std::string_view_literals;
        return "Xmatrix<4, 3>"sv;
    }
};

template<typename T, size_t M = 0u, size_t N = 0u>
struct is_xmatrix_impl : std::false_type {};

template<> struct is_xmatrix_impl<luisa::float4x3, 2u, 3u> : std::true_type {};
template<> struct is_xmatrix_impl<luisa::float4x3, 4u, 3u> : std::true_type {};


/// Ref class common definition
#define LUISA_REF_COMMON(...)                                              \
private:                                                                   \
    const Expression *_expression;                                         \
                                                                           \
public:                                                                    \
    explicit Ref(const Expression *e) noexcept : _expression{e} {}         \
    [[nodiscard]] auto expression() const noexcept { return _expression; } \
    Ref(Ref &&) noexcept = default;                                        \
    Ref(Var<__VA_ARGS__> &&other) noexcept                                 \
        : Ref{static_cast<Ref &&>(other)} {}                               \
    Ref(const Ref &) noexcept = default;                                   \
    template<typename Rhs>                                                 \
    void operator=(Rhs &&rhs) & noexcept {                                 \
        dsl::assign(*this, std::forward<Rhs>(rhs));                        \
    }                                                                      \
    [[nodiscard]] operator Expr<__VA_ARGS__>() const noexcept {            \
        return Expr<__VA_ARGS__>{this->expression()};                      \
    }                                                                      \
    void operator=(Ref rhs) & noexcept { (*this) = Expr<__VA_ARGS__>{rhs}; }

/// Ref<Matrix<N>>
template<size_t M, size_t N>
struct Ref<XMatrix<M, N>>
    : detail::ExprEnableBitwiseCast<Ref<XMatrix<M, N>>>,
      detail::RefEnableSubscriptAccess<Ref<XMatrix<M, N>>>,
      detail::RefEnableGetMemberByIndex<Ref<XMatrix<M, N>>>,
      detail::RefEnableGetAddress<Ref<XMatrix<M, N>>> {
    LUISA_REF_COMMON(XMatrix<M, N>)

#undef LUISA_REF_COMMON

};

inline namespace dsl
{

using luisa::make_float2x3;
using luisa::make_float4x3;

#define LUISA_EXPR(value) \
    detail::extract_expression(std::forward<decltype(value)>(value))

/// Make float4x3 from 4 column vector float3
template<typename C0, typename C1, typename C2, typename C3>
    requires any_dsl_v<C0, C1, C2, C3> &&
             is_same_expr_v<C0, float3> &&
             is_same_expr_v<C1, float3> &&
             is_same_expr_v<C2, float3> &&
             is_same_expr_v<C3, float3>
[[nodiscard]] inline auto make_float4x3(C0 &&c0, C1 &&c1, C2 &&c2, C3 &&c3) noexcept {
    return def<float4x3>(
        detail::FunctionBuilder::current()->call(
            Type::of<float4x3>(), CallOp::MAKE_FLOAT4X3,
            {LUISA_EXPR(c0), LUISA_EXPR(c1), LUISA_EXPR(c2), LUISA_EXPR(c3)}));
}

#undef LUISA_EXPR

} // namespace dsl
} // namespace detail

template<typename T, size_t M = 0u, size_t N = 0u>
using is_xmatrix = detail::is_xmatrix_impl<std::remove_cvref_t<T>, M, N>;

template<typename T>
using is_matrix23 = is_xmatrix<T, 2u, 3u>;

template<typename T>
using is_matrix43 = is_xmatrix<T, 4u, 3u>;

template<>
struct is_custom_struct<luisa::XMatrix<4, 3>> : std::true_type {};

template<>
struct struct_member_tuple<luisa::XMatrix<4, 3>> {
    using type = std::tuple<luisa::float3, luisa::float3, luisa::float3, luisa::float3>;
};

*/

} // namespace luisa::compute


namespace lcs
{

using float2 = luisa::float2;
using float3 = luisa::float3;
using float4 = luisa::float4;
using uint = unsigned int; // luisa::uint;
using int2 = luisa::int2;
using int3 = luisa::int3;
using int4 = luisa::int4;
using uint2 = luisa::uint2;
using uint3 = luisa::uint3;
using uint4 = luisa::uint4;
using uchar2 = luisa::ubyte2;
using uchar4 = luisa::ubyte4;
using float2x2 = luisa::float2x2;
using float3x3 = luisa::float3x3;
using float4x4 = luisa::float4x4;

using float6 = VECTOR6;
using float9 = VECTOR9;
using float12 = VECTOR12;
using float6x6 = MATRIX6;
using float9x9 = MATRIX9;
using float12x12 = MATRIX12;


// using float2x3 = luisa::float2x3;
// using float2x4 = luisa::float2x4;
// using float3x2 = luisa::float3x2;
// using float3x4 = luisa::float3x4;
// using float4x2 = luisa::float4x2;
// using float4x3 = luisa::float4x3;

using Int = luisa::compute::Int;
using Int2 = luisa::compute::Int2;
using Int3 = luisa::compute::Int3;
using Int4 = luisa::compute::Int4;
using UInt = luisa::compute::UInt;
using UInt2 = luisa::compute::UInt2;
using UInt3 = luisa::compute::UInt3;
using UInt4 = luisa::compute::UInt4;
using Float = luisa::compute::Float;
using Float2 = luisa::compute::Float2;
using Float3 = luisa::compute::Float3;
using Float4 = luisa::compute::Float4;
using Bool = luisa::compute::Bool;
using Bool2 = luisa::compute::Bool2;
using Bool3 = luisa::compute::Bool3;
using Bool4 = luisa::compute::Bool4;
using Float2x2 = luisa::compute::Float2x2;
using Float3x3 = luisa::compute::Float3x3;
using Float4x4 = luisa::compute::Float4x4;
using Short = luisa::compute::Short;
using Short2 = luisa::compute::Short2;
using Short3 = luisa::compute::Short3;
using Short4 = luisa::compute::Short4;
using UShort = luisa::compute::UShort;
using UShort2 = luisa::compute::UShort2;
using UShort3 = luisa::compute::UShort3;
using UShort4 = luisa::compute::UShort4;
using SLong = luisa::compute::SLong;
using SLong2 = luisa::compute::SLong2;
using SLong3 = luisa::compute::SLong3;
using SLong4 = luisa::compute::SLong4;
using ULong = luisa::compute::ULong;
using ULong2 = luisa::compute::ULong2;
using ULong3 = luisa::compute::ULong3;
using ULong4 = luisa::compute::ULong4;
using Half = luisa::compute::Half;
using Half2 = luisa::compute::Half2;
using Half3 = luisa::compute::Half3;
using Half4 = luisa::compute::Half4;

using Float2x3 = luisa::compute::Var<float2x3>;
using Float2x4 = luisa::compute::Var<float2x4>;
using Float3x2 = luisa::compute::Var<float3x2>;
using Float3x4 = luisa::compute::Var<float3x4>;
using Float4x2 = luisa::compute::Var<float4x2>;
using Float4x3 = luisa::compute::Var<float4x3>;

using Float6 = luisa::compute::Var<float6>;
using Float9 = luisa::compute::Var<float9>;
using Float12 = luisa::compute::Var<float12>;
using Float6x6 = luisa::compute::Var<MATRIX6>;
using Float9x9 = luisa::compute::Var<MATRIX9>;
using Float12x12 = luisa::compute::Var<MATRIX12>;


// using ElementOffset = luisa::ubyte4;

template<typename T>
using Var = luisa::compute::Var<T>;

#define Float3_zero makeFloat3(0)
#define Float3_one  makeFloat3(1)
#define FLOAT3_MAX  makeFloat3(Float_max)
#define FLOAT3_EPSILON makeFloat3(EPSILON)

#define Color_Rre    makeFloat4(0.9f, 0.1f, 0.1f, 1.f)
#define Color_Green  makeFloat4(0.1f, 0.9f, 0.1f, 1.f)
#define Color_Blue   makeFloat4(0.1f, 0.1f, 0.9f, 1.f)
#define Color_Yellow makeFloat4(0.9f, 0.9f, 0.1f, 1.f)
#define Color_Orange makeFloat4(0.9f, 0.5f, 0.1f, 1.f)
#define Color_Purple makeFloat4(0.5f, 0.1f, 0.9f, 1.f)
#define Color_Cyan   makeFloat4(0.1f, 0.9f, 0.9f, 1.f)
#define Zero2        makeFloat2(0.f, 0.f)
#define Zero3        makeFloat3(0.f, 0.f, 0.f)
#define Zero4        makeFloat4(0.f, 0.f, 0.f, 0.f)


constexpr inline float2 makeFloat2(const float x = 0.0f) { return luisa::make_float2(x, x); }
constexpr inline float2 makeFloat2(const float x, const float y) { return luisa::make_float2(x, y); }
constexpr inline float3 makeFloat3(const float x = 0.0f) { return luisa::make_float3(x, x, x); }
constexpr inline float3 makeFloat3(const float x, const float y, const float z)  { return luisa::make_float3(x, y, z); }
constexpr inline float4 makeFloat4(const float x = 0.0f)  { return luisa::make_float4(x, x, x, x); }
constexpr inline float4 makeFloat4(const float x, const float y, const float z, const float w)  { return luisa::make_float4(x, y, z, w); }

inline Var<float2> makeFloat2(const Var<float> x = 0.0f) { return luisa::compute::make_float2(x, x); }
inline Var<float2> makeFloat2(const Var<float> x, const Var<float> y) { return luisa::compute::make_float2(x, y); }
inline Var<float3> makeFloat3Var(const Var<float> x = 0.0f) { return luisa::compute::make_float3(x, x, x); }
inline Var<float3> makeFloat3(const Var<float> x = 0.0f) { return luisa::compute::make_float3(x, x, x); }
inline Var<float3> makeFloat3(const Var<float> x, const Var<float> y, const Var<float> z)  { return luisa::compute::make_float3(x, y, z); }
inline Var<float4> makeFloat4(const Var<float> x = 0.0f)  { return luisa::compute::make_float4(x, x, x, x); }
inline Var<float4> makeFloat4(const Var<float> x, const Var<float> y, const Var<float> z, const Var<float> w)  { return luisa::compute::make_float4(x, y, z, w); }


constexpr inline int2 makeInt2(const int x = 0) { return luisa::make_int2(x, x); }
constexpr inline int2 makeInt2(const int x, const int y) { return luisa::make_int2(x, y); }
constexpr inline int3 makeInt3(const int x = 0) { return luisa::make_int3(x, x, x); }
constexpr inline int3 makeInt3(const int x, const int y, const int z)  { return luisa::make_int3(x, y, z); }
constexpr inline int4 makeInt4(const int x = 0)  { return luisa::make_int4(x, x, x, x); }
constexpr inline int4 makeInt4(const int x, const int y, const int z, const int w)  { return luisa::make_int4(x, y, z, w); }

inline Var<int2> makeInt2(const Var<int> x = 0) { return luisa::compute::make_int2(x, x); }
inline Var<int2> makeInt2(const Var<int> x, const Var<int> y) { return luisa::compute::make_int2(x, y); }
inline Var<int3> makeInt3(const Var<int> x = 0) { return luisa::compute::make_int3(x, x, x); }
inline Var<int3> makeInt3(const Var<int> x, const Var<int> y, const Var<int> z)  { return luisa::compute::make_int3(x, y, z); }
inline Var<int4> makeInt4(const Var<int> x = 0)  { return luisa::compute::make_int4(x, x, x, x); }
inline Var<int4> makeInt4(const Var<int> x, const Var<int> y, const Var<int> z, const Var<int> w)  { return luisa::compute::make_int4(x, y, z, w); }


constexpr inline uint2 makeUint2(const uint x = 0) { return luisa::make_uint2(x, x); }
constexpr inline uint2 makeUint2(const uint x, const int y) { return luisa::make_uint2(x, y); }
constexpr inline uint3 makeUint3(const uint x = 0) { return luisa::make_uint3(x, x, x); }
constexpr inline uint3 makeUint3(const uint x, const int y, const int z)  { return luisa::make_uint3(x, y, z); }
constexpr inline uint4 makeUint4(const uint x = 0)  { return luisa::make_uint4(x, x, x, x); }
constexpr inline uint4 makeUint4(const uint x, const int y, const int z, const int w)  { return luisa::make_uint4(x, y, z, w); }

inline Var<uint2> makeUint2(const Var<uint> x = 0) { return luisa::compute::make_uint2(x, x); }
inline Var<uint2> makeUint2(const Var<uint> x, const Var<uint> y) { return luisa::compute::make_uint2(x, y); }
inline Var<uint3> makeUint3(const Var<uint> x = 0) { return luisa::compute::make_uint3(x, x, x); }
inline Var<uint3> makeUint3(const Var<uint> x, const Var<uint> y, const Var<uint> z)  { return luisa::compute::make_uint3(x, y, z); }
inline Var<uint4> makeUint4(const Var<uint> x = 0)  { return luisa::compute::make_uint4(x, x, x, x); }
inline Var<uint4> makeUint4(const Var<uint> x, const Var<uint> y, const Var<uint> z, const Var<uint> w)  { return luisa::compute::make_uint4(x, y, z, w); }

namespace Meta
{

template <typename T> struct get_vec_length;
template <typename T, uint N> struct get_vec_length<luisa::Vector<T, N>> 
{
    static constexpr uint value = N;
};
template <typename T, uint N> struct get_vec_length<Var<luisa::compute::Vector<T, N>>> 
{
    static constexpr uint value = N;
};

template <typename T> struct get_matrix_length;
template <uint N> struct get_matrix_length<luisa::Matrix<N>> 
{
    static constexpr uint value = N;
};



};


template<typename Vec> inline Vec normalize_vec(const Vec& vec) {return luisa::normalize(vec);}
template<typename Vec> inline float length_vec(const Vec& vec) { return luisa::length(vec); }
template<typename Vec> inline Vec abs_vec(const Vec& vec) { return luisa::abs(vec); }
template<typename Vec> inline Vec cross_vec(const Vec& vec1, const Vec& vec2)   { return luisa::cross(vec1, vec2);   }
template<typename Vec> inline float dot_vec(const Vec& vec1, const Vec& vec2)    { return luisa::dot(vec1, vec2);     }
template<typename Vec> inline Vec max_vec(const Vec& vec1, const Vec& vec2) { return luisa::max(vec1, vec2); }
template<typename Vec> inline Vec min_vec(const Vec& vec1, const Vec& vec2) { return luisa::min(vec1, vec2); }
template<typename Vec> inline auto all_vec(const Vec& vec) { return luisa::all(vec); }
template<typename Vec> inline auto any_vec(const Vec& vec) { return luisa::any(vec); }

template<typename Vec> inline Var<Vec> normalize_vec(const Var<Vec>& vec) {return luisa::compute::normalize(vec);}
template<typename Vec> inline Var<float> length_vec(const Var<Vec>& vec) { return luisa::compute::length(vec); }
template<typename Vec> inline Var<Vec> abs_vec(const Var<Vec>& vec) { return luisa::compute::abs(vec); }
template<typename Vec> inline Var<Vec> cross_vec(const Var<Vec>& vec1, const Var<Vec>& vec2)   { return luisa::compute::cross(vec1, vec2);   }
template<typename Vec> inline Var<float> dot_vec(const Var<Vec>& vec1, const Var<Vec>& vec2)    { return luisa::compute::dot(vec1, vec2);     }
template<typename Vec> inline Var<Vec> max_vec(const Var<Vec>& vec1, const Var<Vec>& vec2) { return luisa::compute::max(vec1, vec2); }
template<typename Vec> inline Var<Vec> min_vec(const Var<Vec>& vec1, const Var<Vec>& vec2) { return luisa::compute::min(vec1, vec2); }
template<typename Vec> inline auto all_vec(const Var<Vec>& vec) { return luisa::compute::all(vec); }
template<typename Vec> inline auto any_vec(const Var<Vec>& vec) { return luisa::compute::any(vec); }

template<typename Vec> inline Vec reverse_vec(const Vec& vec) { return 1.f / vec;}
template<typename Vec> inline auto safe_length_vec(const Vec& vec) { return length_vec(vec) + lcs::Epsilon; }



template<typename Vec, uint N> inline auto min_component_vec(const Vec& vec) 
{ 
    auto min_value = vec[0];
    for (uint i = 1; i < N; i++) { min_value = min_scalar(min_value, vec[i]);} 
    return min_value;
}
template<typename Vec, uint N> inline auto max_component_vec(const Vec& vec) 
{ 
    auto max_value = vec[0];
    for (uint i = 1; i < N; i++) { max_value = max_scalar(max_value, vec[i]); } 
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
inline float sum_vec(luisa::Vector<T, N> vec) 
{ 
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


inline float compute_face_area(const float3& pos0, const float3& pos1, const float3& pos2)
{
    float3 vec0 = pos1 - pos0;
    float3 vec1 = pos2 - pos0;
    float area = length_vec(cross_vec(vec0, vec1)) * 0.5f;
    return area;
}
inline float3 compute_face_normal(const float3& p1, const float3& p2, const float3& p3)
{
	float3 s = p2 - p1;
	float3 t = p3 - p1;
	float3 n = normalize_vec(cross_vec(s, t));
	return n;
}

}; // namespace lcs