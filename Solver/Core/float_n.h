#pragma once

#include "Core/xbasic_types.h"
#include "Core/constant_value.h"
#include "luisa/core/mathematics.h"
#include "luisa/dsl/var.h"


namespace luisa::compute 
{
using Uint = luisa::compute::Var<uint>;
using Uint2 = luisa::compute::Var<uint2>;
using Uint3 = luisa::compute::Var<uint3>;
using Uint4 = luisa::compute::Var<uint4>;
using Float2x3 = luisa::compute::Var<float2x3>;
using Float4x3 = luisa::compute::Var<float4x3>;

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

template<>
struct is_xmatrix_impl<luisa::float4x3, 4u, 3u> : std::true_type {};

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
*/

} // namespace luisa::compute






namespace lcsv
{

using float2 = luisa::float2;
using float3 = luisa::float3;
using float4 = luisa::float4;
using uint2 = luisa::uint2;
using uint3 = luisa::uint3;
using uint4 = luisa::uint4;
using uchar2 = luisa::ubyte2;
using uchar4 = luisa::ubyte4;
using float2x2 = luisa::float2x2;
using float3x3 = luisa::float3x3;
using float4x4 = luisa::float4x4;
// using Float2x3 = luisa::float2x3;
// using Float3x2 = luisa::float3x2;
using float2x3 = luisa::float2x3;
using float4x3 = luisa::float4x3;
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

}; // namespace lcsv