#pragma once

#include <cstddef>
#include <array>
#include <luisa/core/basic_types.h>
#include <luisa/dsl/struct.h>
#include "luisa/core/logging.h"
// #include <luisa/core/stl/hash_fwd.h>
// #include <luisa/core/basic_traits.h>

#define PTR(T) luisa::compute::BufferVar<T>

namespace lcs {

using uint = unsigned int;
/*
/// Matrix only allows size of 2, 3, 4
template<size_t M, size_t N>
struct XMatrix {
    static_assert(always_false_v<std::integral_constant<size_t, N>>, "Invalid matrix type");
};

/// 4x3 matrix
template<>
struct XMatrix<4, 3> {

    float3 cols[4];

    constexpr XMatrix() noexcept
        : cols{float3{0.0f}, float3{0.0f}, float3{0.0f}, float3{0.0f}} {}

    constexpr XMatrix(const float3 c0, const float3 c1, const float3 c2, const float3 c3) noexcept
        : cols{c0, c1, c2, c3} {}

    static constexpr XMatrix fill(const float c) noexcept {
        return XMatrix{
            float3{c, c, c},
            float3{c, c, c},
            float3{c, c, c},
            float3{c, c, c}};
    }

    [[nodiscard]] constexpr float3 &operator[](size_t i) noexcept { return cols[i]; }
    [[nodiscard]] constexpr const float3 &operator[](size_t i) const noexcept { return cols[i]; }
    // constexpr void operator+=(
    //     const luisa::XMatrix<4, 3>& right) noexcept {
    //     for (size_t i = 0; i < 4; i++) {
    //         cols[i] += right[i];
    //     }
    // }
};

/// 4x3 matrix
template<>
struct XMatrix<2, 3> {

    float3 cols[2];

    constexpr XMatrix() noexcept
        : cols{float3{0.0f}, float3{0.0f}} {}

    constexpr XMatrix(const float3 c0, const float3 c1) noexcept
        : cols{c0, c1} {}

    static constexpr XMatrix fill(const float c) noexcept {
        return XMatrix{
            float3{c, c, c},
            float3{c, c, c}};
    }

    [[nodiscard]] constexpr float3 &operator[](size_t i) noexcept { return cols[i]; }
    [[nodiscard]] constexpr const float3 &operator[](size_t i) const noexcept { return cols[i]; }
};

/// 3x4 matrix
template<>
struct XMatrix<3, 4> {

    float4 cols[3];

    constexpr XMatrix() noexcept
        : cols{float4{0.0f}, float4{0.0f}, float4{0.0f}} {}

    constexpr XMatrix(const float4 c0, const float4 c1, const float4 c2) noexcept
        : cols{c0, c1, c2} {}

    static constexpr XMatrix fill(const float c) noexcept {
        return XMatrix{
            float4{c, c, c, c},
            float4{c, c, c, c},
            float4{c, c, c, c}};
    }

    [[nodiscard]] constexpr float4 &operator[](size_t i) noexcept { return cols[i]; }
    [[nodiscard]] constexpr const float4 &operator[](size_t i) const noexcept { return cols[i]; }
};



// template<> struct XMatrix<2, 3>; 
// template<> struct XMatrix<4, 3>; 
// template<> struct XMatrix<3, 4>; 
*/



template<size_t M, size_t N>
struct XMatrix {
    luisa::Vector<float, N> cols[M];
    [[nodiscard]] constexpr luisa::Vector<float, N> &operator[](size_t i) noexcept { return cols[i]; }
    [[nodiscard]] constexpr const luisa::Vector<float, N> &operator[](size_t i) const noexcept { return cols[i]; }
    constexpr void operator+=(
        const XMatrix<M, N>& right) noexcept {
        for (size_t i = 0; i < M; i++) {
            cols[i] += right[i];
        }
    }
    constexpr void operator-=(
        const XMatrix<M, N>& right) noexcept {
        for (size_t i = 0; i < M; i++) {
            cols[i] -= right[i];
        }
    }
    constexpr void operator*=(
        const float& right) noexcept {
        for (size_t i = 0; i < M; i++) {
            cols[i] *= right;
        }
    }
    // constexpr auto operator+(
    //     const XMatrix<M, N>& right) noexcept {
    //     XMatrix<M, N> result;
    //     for (size_t i = 0; i < M; i++) {
    //         result[i] = cols[i] + right[i];
    //     }
    //     return result;
    // }
    // constexpr auto operator*(
    //     const float right) noexcept {
    //     XMatrix<M, N> result;
    //     for (size_t i = 0; i < M; i++) {
    //         result[i] = cols[i] * right;
    //     }
    //     return result;
    // }
    // constexpr auto operator/(
    //     const float right) noexcept {
    //     XMatrix<M, N> result;
    //     for (size_t i = 0; i < M; i++) {
    //         result[i] = cols[i] / right;
    //     }
    //     return result;
    // }
};



template<size_t N>
struct LargeVector{
    using Float3 = luisa::float3;
    Float3 vec[N/3];

public:
    Float3& operator[](size_t idx) { return vec[idx]; }
    Float3& operator()(size_t idx) { return vec[idx]; }
    const Float3& operator[](size_t idx) const { return vec[idx]; }
    const Float3& operator()(size_t idx) const { return vec[idx]; }
    Float3 to_Float3() { static_assert(N == 3, "Array is not Float3 Type!"); return vec[0]; }
    void operator=(const LargeVector<N>& input_vec) { for (size_t i = 0; i < N/3; i++) { vec[i] = input_vec[i]; }} 
};

template<size_t M, size_t N>
struct LargeMatrix {
    using Float3 = luisa::float3;
    using Float3x3 = luisa::float3x3;
    Float3x3 mat[M/3][N/3];

    //       Float3x3& operator()(size_t idx1, size_t idx2)        { return mat[idx1 * (N/3) + idx2]; }
    // const Float3x3& operator()(size_t idx1, size_t idx2) const  { return mat[idx1 * (N/3) + idx2]; }

    //       float& get_scalar(size_t idx1, size_t idx2)        { return mat[(idx1/3) * (N/3) + (idx2/3)][(idx1%3)][(idx2%3)]; }
    // const float& get_scalar(size_t idx1, size_t idx2) const  { return mat[(idx1/3) * (N/3) + (idx2/3)][(idx1%3)][(idx2%3)]; }

private:
    void setHelper(const Float3x3& input){
        for (size_t i = 0; i < M/3; i++) 
            for(size_t j = 0; j < N/3; j++)
                (*this)(i, j) = input;
    }
    void setHelper(const LargeMatrix<M, N>& input_mat){
        for (size_t i = 0; i < M/3; i++) 
            for(size_t j = 0; j < N/3; j++)
                (*this)(i, j) = input_mat(i, j);
    }
    void setDiag(const Float3x3 input){
        static_assert(M == N, "Matrix is not Square Matrix");
        for (size_t i = 0; i < M/3; i++) 
            for(size_t j = 0; j < N/3; j++)
                if(i == j){
                    (*this)(i, j) = input;
                }
    }
};


// template<size_t M, size_t N>
// struct hash<XMatrix<M, N>> {
//     using is_avalanching = void;
//     [[nodiscard]] uint64_t operator()(XMatrix<M, N> m, uint64_t seed = hash64_default_seed) const noexcept {
//         std::array<float, M * N> data{};
//         for (size_t i = 0u; i < M; i++) {
//             for (size_t j = 0u; j < N; j++) {
//                 data[i * N + j] = m[i][j];
//             }
//         }
//         return hash64(data.data(), data.size() * sizeof(float), seed);
//     }
// };

}// namespace luisa



// template<typename T, typename... Args>
// static inline constexpr T make_matrix(Args... args) {
//     return T{ args... };
// }

//      M              L              L
//    |||||           ||||||         |||||
//  N |||||    *    M ||||||   => N  |||||
//    |||||           ||||||         |||||
// template<size_t M, size_t N, size_t L> // TODO: use column acceleration
// [[nodiscard]] constexpr auto operator*(
//     const luisa::XMatrix<M, N>& left, 
//     const luisa::XMatrix<L, M>& right) noexcept {
//     luisa::XMatrix<L, N> result;
//     for (size_t i = 0; i < L; i++) {
//         for(size_t j = 0; j < N; j++){
//             result[i][j] = 0.0f;
//             for(size_t k = 0; k < M; k++){
//                 result[i][j] += left[k][j] * right[i][k];
//             }
//         }
//     }
//     return result;
// }
// template<size_t M, size_t N>
// [[nodiscard]] constexpr luisa::XMatrix<M, N> operator+(
//     const luisa::XMatrix<M, N>& left, 
//     const luisa::XMatrix<M, N>& right) noexcept {
//     luisa::XMatrix<M, N> result;
//     for (size_t i = 0; i < M; i++) {
//         result[i] = left[i] + right[i];
//     }
//     return result;
// }

namespace lcs {

using float2x3 = XMatrix<2, 3>;
using float2x4 = XMatrix<2, 4>;
using float3x2 = XMatrix<3, 2>;
using float3x4 = XMatrix<3, 4>;
using float4x2 = XMatrix<4, 2>;
using float4x3 = XMatrix<4, 3>;

using VECTOR3 = LargeVector<3>;
using VECTOR6 = LargeVector<6>;
using VECTOR9 = LargeVector<9>;
using VECTOR12 = LargeVector<12>;

using MATRIX3 = LargeMatrix<3, 3>;
using MATRIX6 = LargeMatrix<6, 6>;
using MATRIX9 = LargeMatrix<9, 9>;
using MATRIX12 = LargeMatrix<12, 12>;
using MATRIX12x3 = LargeMatrix<12, 3>;
using MATRIX3x12 = LargeMatrix<3, 12>;

// #define MAKE_XMATRIX_TYPE(M, N) \
//     struct float##M##x##N { \
//         luisa::float##N cols[M]; \
//         [[nodiscard]] constexpr luisa::float##N &operator[](size_t i) noexcept { return cols[i]; } \
//         [[nodiscard]] constexpr const luisa::float##N &operator[](size_t i) const noexcept { return cols[i]; } \
//         constexpr void operator+=(const float##M##x##N &right) noexcept { \
//             for (size_t i = 0; i < M; i++) { \
//                 cols[i] += right[i]; \
//             } \
//         } \
//     };

// MAKE_XMATRIX_TYPE(2, 3);
// MAKE_XMATRIX_TYPE(2, 4);
// MAKE_XMATRIX_TYPE(3, 2);
// MAKE_XMATRIX_TYPE(3, 4);
// MAKE_XMATRIX_TYPE(4, 2);
// MAKE_XMATRIX_TYPE(4, 3);

// #undef MAKE_XMATRIX_TYPE


// // float2x3
// [[nodiscard]] inline float2x3 make_float2x3(const luisa::float3& column0, const luisa::float3& column1) noexcept 
// { 
//     float2x3 mat;
//     mat.cols[0] = column0;
//     mat.cols[1] = column1;
//     return mat;
// }

// // float2x3
// [[nodiscard]] inline float2x4 make_float2x4(const luisa::float4& column0, const luisa::float4& column1) noexcept 
// { 
//     float2x4 mat;
//     mat.cols[0] = column0;
//     mat.cols[1] = column1;
//     return mat;
// }

// // float3x2
// [[nodiscard]] inline float3x2 make_float3x2(const luisa::float2& column0, const luisa::float2& column1, const luisa::float2& column2) noexcept 
// { 
//     float3x2 mat;
//     mat.cols[0] = column0;
//     mat.cols[1] = column1;
//     mat.cols[2] = column2;
//     return mat;
// }

// // float3x4
// [[nodiscard]] inline float3x4 make_float3x4(const luisa::float4& column0, const luisa::float4& column1, const luisa::float4& column2) noexcept 
// { 
//     float3x4 mat;
//     mat.cols[0] = column0;
//     mat.cols[1] = column1;
//     mat.cols[2] = column2;
//     return mat;
// }

// // float4x2
// [[nodiscard]] inline float4x2 make_float4x2(const luisa::float2& column0, const luisa::float2& column1, const luisa::float2& column2, const luisa::float2& column3) noexcept 
// { 
//     float4x2 mat;
//     mat.cols[0] = column0;
//     mat.cols[1] = column1;
//     mat.cols[2] = column2;
//     mat.cols[3] = column3;
//     return mat;
// }

// template<size_t M, size_t N, size_t L> // TODO: use column acceleration
// [[nodiscard]] constexpr auto mult_mat(
//     const luisa::compute::Var<luisa::compute::XMatrix<M, N>>& left, 
//     const luisa::compute::Var<luisa::compute::XMatrix<L, M>>& right) noexcept {
//     luisa::compute::Var<luisa::compute::XMatrix<L, N>> result;
//     for (size_t i = 0; i < L; i++) {
//         for(size_t j = 0; j < N; j++){
//             result[i][j] = 0.0f;
//             for(size_t k = 0; k < M; k++){
//                 result[i][j] += left[k][j] * right[i][k];
//             }
//         }
//     }
//     return result;
// }
// template<size_t M, size_t N>
// [[nodiscard]] constexpr luisa::compute::Var<luisa::XMatrix<M, N>> add(
//     const luisa::compute::Var<luisa::XMatrix<M, N>>& left, 
//     const luisa::compute::Var<luisa::XMatrix<M, N>>& right) noexcept {
//     luisa::compute::Var<luisa::XMatrix<M, N>> result;
//     for (size_t i = 0; i < M; i++) {
//         result.cols[i] = left.cols[i] + right.cols[i];
//     }
//     return result;
// }
// template<size_t M, size_t N>
// constexpr void add(
//     luisa::compute::Var<luisa::XMatrix<M, N>>& result,
//     const luisa::compute::Var<luisa::XMatrix<M, N>>& left, 
//     const luisa::compute::Var<luisa::XMatrix<M, N>>& right) noexcept {
//     for (size_t i = 0; i < M; i++) {
//         result.cols[i] = left.cols[i] + right.cols[i];
//     }
// }
// template<size_t M, size_t N>
// [[nodiscard]] constexpr luisa::compute::Var<luisa::XMatrix<M, N>> sub(
//     const luisa::compute::Var<luisa::XMatrix<M, N>>& left, 
//     const luisa::compute::Var<luisa::XMatrix<M, N>>& right) noexcept {
//     luisa::compute::Var<luisa::XMatrix<M, N>> result;
//     for (size_t i = 0; i < M; i++) {
//         result.cols[i] = left.cols[i] - right.cols[i];
//     }
//     return result;
// }

} // namespace luisa::compute

LUISA_STRUCT(lcs::float2x3, cols) {};
LUISA_STRUCT(lcs::float2x4, cols) {};
LUISA_STRUCT(lcs::float3x2, cols) {};
LUISA_STRUCT(lcs::float3x4, cols) {};
LUISA_STRUCT(lcs::float4x2, cols) {};
LUISA_STRUCT(lcs::float4x3, cols) {};

LUISA_STRUCT(lcs::VECTOR3, vec) {};
LUISA_STRUCT(lcs::VECTOR6, vec) {};
LUISA_STRUCT(lcs::VECTOR9, vec) {};
LUISA_STRUCT(lcs::VECTOR12, vec) {};

LUISA_STRUCT(lcs::MATRIX3, mat) {};
LUISA_STRUCT(lcs::MATRIX6, mat) {};
LUISA_STRUCT(lcs::MATRIX9, mat) {};
LUISA_STRUCT(lcs::MATRIX12, mat) {};
LUISA_STRUCT(lcs::MATRIX12x3, mat) {};
LUISA_STRUCT(lcs::MATRIX3x12, mat) {};


template<size_t M, size_t N>
[[nodiscard]] inline lcs::XMatrix<M, N> operator+(const lcs::XMatrix<M, N>& lhs, const lcs::XMatrix<M, N>& rhs) noexcept
{
    lcs::XMatrix<M, N> result;
    for (uint ii = 0; ii < M; ii++)
    {
        result.cols[ii] = lhs.cols[ii] + rhs.cols[ii];
    }
    return result;
}
template<size_t M, size_t N>
[[nodiscard]] inline lcs::XMatrix<M, N> operator-(const lcs::XMatrix<M, N>& lhs, const lcs::XMatrix<M, N>& rhs) noexcept
{
    lcs::XMatrix<M, N> result;
    for (uint ii = 0; ii < M; ii++)
    {
        result.cols[ii] = lhs.cols[ii] - rhs.cols[ii];
    }
    return result;
}
template<size_t M, size_t N>
[[nodiscard]] inline lcs::XMatrix<M, N> operator*(const lcs::XMatrix<M, N>& lhs, const float& rhs) noexcept
{
    lcs::XMatrix<M, N> result;
    for (uint ii = 0; ii < M; ii++)
    {
        result.cols[ii] = rhs * lhs.cols[ii];
    }
    return result;
}
template<size_t M, size_t N>
[[nodiscard]] inline lcs::XMatrix<M, N> operator*(const float& lhs, const lcs::XMatrix<M, N>& rhs) noexcept
{
    lcs::XMatrix<M, N> result;
    for (uint ii = 0; ii < M; ii++)
    {
        result.cols[ii] = lhs * rhs.cols[ii];
    }
    return result;
}
template<size_t M, size_t N>
[[nodiscard]] inline lcs::XMatrix<M, N> operator/(const lcs::XMatrix<M, N>& lhs, const float& rhs) noexcept
{
    lcs::XMatrix<M, N> result;
    for (uint ii = 0; ii < M; ii++)
    {
        result.cols[ii] = lhs.cols[ii] / rhs;
    }
    return result;
}
template<size_t M, size_t N>
[[nodiscard]] inline lcs::XMatrix<M, N> operator/(const float& lhs, const lcs::XMatrix<M, N>& rhs) noexcept
{
    lcs::XMatrix<M, N> result;
    for (uint ii = 0; ii < M; ii++)
    {
        result.cols[ii] = rhs.cols[ii] / lhs;
    }
    return result;
}


template<size_t M, size_t N>
[[nodiscard]] inline luisa::compute::Var<lcs::XMatrix<M, N>> operator+(const luisa::compute::Var<lcs::XMatrix<M, N>>& lhs, const luisa::compute::Var<lcs::XMatrix<M, N>>& rhs) noexcept
{
    luisa::compute::Var<lcs::XMatrix<M, N>> result;
    for (uint ii = 0; ii < M; ii++)
    {
        result.cols[ii] = lhs.cols[ii] + rhs.cols[ii];
    }
    return result;
}
template<size_t M, size_t N>
[[nodiscard]] inline luisa::compute::Var<lcs::XMatrix<M, N>> operator-(const luisa::compute::Var<lcs::XMatrix<M, N>>& lhs, const luisa::compute::Var<lcs::XMatrix<M, N>>& rhs) noexcept
{
    luisa::compute::Var<lcs::XMatrix<M, N>> result;
    for (uint ii = 0; ii < M; ii++)
    {
        result.cols[ii] = lhs.cols[ii] - rhs.cols[ii];
    }
    return result;
}
template<size_t M, size_t N>
[[nodiscard]] inline luisa::compute::Var<lcs::XMatrix<M, N>> operator*(const luisa::compute::Var<lcs::XMatrix<M, N>>& lhs, const luisa::compute::Var<float>& rhs) noexcept
{
    luisa::compute::Var<lcs::XMatrix<M, N>> result;
    for (uint ii = 0; ii < M; ii++)
    {
        result.cols[ii] = rhs * lhs.cols[ii];
    }
    return result;
}
template<size_t M, size_t N>
[[nodiscard]] inline luisa::compute::Var<lcs::XMatrix<M, N>> operator*(const luisa::compute::Var<float>& lhs, const luisa::compute::Var<lcs::XMatrix<M, N>>& rhs) noexcept
{
    luisa::compute::Var<lcs::XMatrix<M, N>> result;
    for (uint ii = 0; ii < M; ii++)
    {
        result.cols[ii] = lhs * rhs.cols[ii];
    }
    return result;
}
template<size_t M, size_t N>
[[nodiscard]] inline luisa::compute::Var<lcs::XMatrix<M, N>> operator/(const luisa::compute::Var<lcs::XMatrix<M, N>>& lhs, const luisa::compute::Var<float>& rhs) noexcept
{
    luisa::compute::Var<lcs::XMatrix<M, N>> result;
    for (uint ii = 0; ii < M; ii++)
    {
        result.cols[ii] = lhs.cols[ii] / rhs;
    }
    return result;
}
template<size_t M, size_t N>
[[nodiscard]] inline luisa::compute::Var<lcs::XMatrix<M, N>> operator/(const luisa::compute::Var<float>& lhs, const luisa::compute::Var<lcs::XMatrix<M, N>>& rhs) noexcept
{
    luisa::compute::Var<lcs::XMatrix<M, N>> result;
    for (uint ii = 0; ii < M; ii++)
    {
        result.cols[ii] = rhs.cols[ii] / lhs;
    }
    return result;
}

// #define REGIRSTER_XMATRIX_TO_STRUCT(M, N) \
//     LUISA_STRUCT(lcs::float##M##x##N, cols) {  \
//         [[nodiscard]] auto mult(const luisa::compute::Expr<float> alpha) const noexcept {   \
//             luisa::compute::Var<lcs::float##M##x##N> result;  \
//             for (size_t i = 0; i < M; i++) {  \
//                 result.cols[i] = alpha * cols[i];  \
//             }  \
//             return result;  \
//         }  \
//         [[nodiscard]] inline auto mult(const luisa::compute::Expr<luisa::float##N>& vec) noexcept   \
//         {  \
//             luisa::compute::Var<luisa::float##N> output;  \
//             for (int i = 0; i < M; ++i) { \
//                 output[i] = 0.0f;         \
//                 for (int j = 0; j < N; j++) { \
//                     output[i] += cols[j][i] * vec[j]; \
//                 }  \
//             }  \
//             return output;  \
//         }  \
//     };  

// REGIRSTER_XMATRIX_TO_STRUCT(2, 3);
// REGIRSTER_XMATRIX_TO_STRUCT(2, 4);
// REGIRSTER_XMATRIX_TO_STRUCT(3, 2);
// REGIRSTER_XMATRIX_TO_STRUCT(3, 4);
// REGIRSTER_XMATRIX_TO_STRUCT(4, 2);
// REGIRSTER_XMATRIX_TO_STRUCT(4, 3);

// auto operator=(const luisa::compute::Expr<lcs::float##M##x##N> right) noexcept {   \
//     for (size_t i = 0; i < M; i++) {  \
//         cols[i] = right.cols[i];  \
//     }  \
// }  \

// [[nodiscard]] inline Var<float2x2> mult(const luisa::compute::Var<float3x2>& right) noexcept 
// {
//     luisa::compute::Var<float2x2> output;
//     for (int i = 0; i < 2; ++i) { // row
//         for (int j = 0; j < 2; ++j) { // col
//             output[j][i] = left.cols[0][i] * right.cols[j][0]
//                          + left.cols[1][i] * right.cols[j][1]
//                          + left.cols[2][i] * right.cols[j][2];
//         }
//     }
//     return output;
// }

namespace lcs {

template<typename T>
using Var = luisa::compute::Var<T>;

// [[nodiscard]] inline Var<float2x3> make_float2x3(const Var<luisa::float3>& column0, const Var<luisa::float3>& column1) noexcept 
// { 
//     Var<float2x3> mat;
//     mat.cols[0] = column0;
//     mat.cols[1] = column1;
//     return mat;
// }
// [[nodiscard]] inline Var<float2x4> make_float2x4(const Var<luisa::float4>& column0, const Var<luisa::float4>& column1) noexcept 
// { 
//     Var<float2x4> mat;
//     mat.cols[0] = column0;
//     mat.cols[1] = column1;
//     return mat;
// }
// [[nodiscard]] inline Var<float3x2> make_float3x2(const Var<luisa::float2>& column0, const Var<luisa::float2>& column1, const Var<luisa::float2>& column2) noexcept 
// {
//     Var<float3x2> mat;
//     mat.cols[0] = column0;
//     mat.cols[1] = column1;
//     mat.cols[2] = column2;
//     return mat;
// }
// [[nodiscard]] inline Var<float3x4> make_float3x4(const Var<luisa::float4>& column0, const Var<luisa::float4>& column1, const Var<luisa::float4>& column2) noexcept 
// {
//     Var<float3x4> mat;
//     mat.cols[0] = column0;
//     mat.cols[1] = column1;
//     mat.cols[2] = column2;
//     return mat;
// }
// [[nodiscard]] inline Var<float4x2> make_float4x2(const Var<luisa::float2>& column0, const Var<luisa::float2>& column1, const Var<luisa::float2>& column2, const Var<luisa::float2>& column3) noexcept 
// {
//     Var<float4x2> mat;
//     mat.cols[0] = column0;
//     mat.cols[1] = column1;
//     mat.cols[2] = column2;
//     mat.cols[3] = column3;
//     return mat;
// }




// Vector set
template<size_t N>
void set_largevec(LargeVector<N>& left_vec, float value) {
    for (size_t i = 0; i < N/3; i++) { left_vec.vec[i] = luisa::make_float3(value); }
}
template<size_t N>
void set_largevec(LargeVector<N>& left_vec, const luisa::float3& vec) {
    for (size_t i = 0; i < N/3; i++) { left_vec.vec[i] = vec; }
}
template<size_t N>
void set_largevec(LargeVector<N>& left_vec, const luisa::float3 vec[N]) {
    for (size_t i = 0; i < N/3; i++) { left_vec.vec[i] = vec[i]; }
}

template<size_t N>
void set_largevec(Var<LargeVector<N>>& left_vec, Var<float> value) {
    for (size_t i = 0; i < N/3; i++) { left_vec.vec[i] = luisa::compute::make_float3(value); }
}
template<size_t N>
void set_largevec(Var<LargeVector<N>>& left_vec, const luisa::float3& vec) {
    for (size_t i = 0; i < N/3; i++) { left_vec.vec[i] = vec; }
}
template<size_t N>
void set_largevec(Var<LargeVector<N>>& left_vec, const luisa::float3 vec[N]) {
    for (size_t i = 0; i < N/3; i++) { left_vec.vec[i] = vec[i]; }
}


// Vector add
template<size_t N>
auto add_largevec_vec(const LargeVector<N>& left_vec, const luisa::float3& value) {
    LargeVector<N> output;
    for (size_t i = 0; i < N/3; i++) { output.vec[i] = left_vec.vec[i] + value; }
    return output;
}
template<size_t N>
auto sub_largevec_vec(const LargeVector<N>& left_vec, const luisa::float3& value) {
    LargeVector<N> output;
    for (size_t i = 0; i < N/3; i++) { output.vec[i] = left_vec.vec[i] - value; }
    return output;
}
template<size_t N>
auto add_largevec_vec(const LargeVector<N>& left_vec, const LargeVector<N>& right_vec) {
    LargeVector<N> output;
    for (size_t i = 0; i < N/3; i++) { output.vec[i] = left_vec.vec[i] + right_vec[i]; }
    return output;
}
template<size_t N>
auto sub_largevec_vec(const LargeVector<N>& left_vec, const LargeVector<N>& right_vec) {
    LargeVector<N> output;
    for (size_t i = 0; i < N/3; i++) { output.vec[i] = left_vec.vec[i] - right_vec[i]; }
    return output;
}

template<size_t N>
auto add_largevec_vec(const Var<LargeVector<N>>& left_vec, const Var<luisa::float3>& value) {
    Var<LargeVector<N>> output;
    for (size_t i = 0; i < N/3; i++) { output.vec[i] = left_vec.vec[i] + value; }
    return output;
}
template<size_t N>
auto sub_largevec_vec(const Var<LargeVector<N>>& left_vec, const Var<luisa::float3>& value) {
    Var<LargeVector<N>> output;
    for (size_t i = 0; i < N/3; i++) { output.vec[i] = left_vec.vec[i] - value; }
    return output;
}
template<size_t N>
auto add_largevec_vec(const Var<LargeVector<N>>& left_vec, const Var<LargeVector<N>>& right_vec) {
    Var<LargeVector<N>> output;
    for (size_t i = 0; i < N/3; i++) { output.vec[i] = left_vec.vec[i] + right_vec.vec[i]; }
    return output;
}
template<size_t N>
auto sub_largevec_vec(const Var<LargeVector<N>>& left_vec, const Var<LargeVector<N>>& right_vec) {
    Var<LargeVector<N>> output;
    for (size_t i = 0; i < N/3; i++) { output.vec[i] = left_vec.vec[i] - right_vec.vec[i]; }
    return output;
}

// Vector mult scalar
template<size_t N>
auto mult_largevec_scalar(LargeVector<N>& output, const LargeVector<N>& left_vec, const float value) {
    for (size_t i = 0; i < N/3; i++) { output.vec[i] = left_vec.vec[i] * value; }
}
template<size_t N>
auto mult_largevec_scalar(const LargeVector<N>& left_vec, const float value) {
    LargeVector<N> output;
    mult_largevec_scalar(output, left_vec, value);
    return output;
}
template<size_t N>
auto mult_largevec_scalar(Var<LargeVector<N>>& output, const Var<LargeVector<N>>& left_vec, const Var<float> value) {
    for (size_t i = 0; i < N/3; i++) { output.vec[i] = left_vec.vec[i] * value; }
    return output;
}
template<size_t N>
auto mult_largevec_scalar(const Var<LargeVector<N>>& left_vec, const Var<float>& value) {
    Var<LargeVector<N>> output;
    mult_largevec_scalar(output, left_vec, value);
    return output;
}


// Matrix set
template<size_t M, size_t N>
void set_largemat_zero(LargeMatrix<M, N>& left_mat) 
{
    for (size_t i = 0; i < M/3; i++) 
        for(size_t j = 0; j < N/3; j++)
            left_mat.mat[i][j] = luisa::make_float3x3(0.0f);
}
template<size_t M, size_t N>
void set_largemat_identity(LargeMatrix<M, N>& left_mat) 
{
    set_largemat_zero(left_mat);
    for (size_t i = 0; i < M/3; i++) 
        left_mat.mat[i][i] = luisa::make_float3x3(1.0f);
}
// template<size_t M, size_t N> auto get_largemat_zero() { LargeMatrix<M, N> output; set_largemat_zero(output); return output; }
// template<size_t M, size_t N> auto get_largemat_identity() { LargeMatrix<M, N> output; set_largemat_identity(output); return output; }

template<size_t M, size_t N>
void set_largemat_zero(Var<LargeMatrix<M, N>>& left_mat) 
{
    for (size_t i = 0; i < M/3; i++) 
        for(size_t j = 0; j < N/3; j++)
            left_mat.mat[i][j] = luisa::compute::make_float3x3(0.0f);
}
template<size_t M, size_t N>
void set_largemat_identity(Var<LargeMatrix<M, N>>& left_mat) 
{
    set_largemat_zero(left_mat);
    for (size_t i = 0; i < M/3; i++) 
        left_mat.mat[i][i] = luisa::make_float3x3(1.0f);
}

// Matrix column set/get
template<size_t M, size_t N>
auto get_colomn_largemat(LargeMatrix<M, N>& left_mat, const uint col_idx) {
    LargeVector<N> vec;
    for (size_t i = 0; i < N/3; i++)
        vec.vec[i] = left_mat.mat[col_idx/3][i][col_idx%3];
    return vec;
}
template<size_t M, size_t N>
void set_colomn_largemat(LargeMatrix<M, N>& left_mat, const uint col_idx, const LargeVector<N>& vec) {
    for (size_t i = 0; i < N/3; i++)
        left_mat.mat[col_idx/3][i][col_idx%3] = (vec.vec[i]);
}

template<size_t M, size_t N>
auto get_colomn_largemat(Var<LargeMatrix<M, N>>& left_mat, const uint col_idx) {
    Var<LargeVector<N>> vec;
    for (size_t i = 0; i < N/3; i++)
        vec.vec[i] = left_mat.mat[col_idx/3][i][col_idx%3];
    return vec;
}
template<size_t M, size_t N>
void set_colomn_largemat(Var<LargeMatrix<M, N>>& left_mat, const uint col_idx, const Var<LargeVector<N>>& vec) {
    for (size_t i = 0; i < N/3; i++)
        left_mat.mat[col_idx/3][i][col_idx%3] = (vec.vec[i]);
}

// Matrix row set/get
template<size_t M, size_t N>
auto get_row_largemat(const LargeMatrix<M, N>& left_mat, const size_t row_idx) {
    LargeVector<M> vec;
    for (size_t i = 0; i < M/3; i++) {
        auto& mat3x3 = left_mat.mat[i][row_idx/3];
        vec.vec[i] = luisa::make_float3(
            mat3x3[0][row_idx%3], 
            mat3x3[1][row_idx%3], 
            mat3x3[2][row_idx%3]);
    }
    return vec;
}
template<size_t M, size_t N>
void set_row_largemat(LargeMatrix<M, N>& left_mat, const size_t row_idx, const LargeVector<M>& vec) {
    for (size_t i = 0; i < M/3; i++) {
        auto& mat3x3 = left_mat.mat[i][row_idx/3];
        mat3x3[0][row_idx%3] = vec.vec[i][0];
        mat3x3[1][row_idx%3] = vec.vec[i][1];
        mat3x3[2][row_idx%3] = vec.vec[i][2];
    }
}

template<size_t M, size_t N>
auto get_row_largemat(const Var<LargeMatrix<M, N>>& left_mat, const size_t row_idx) {
    Var<LargeVector<M>> vec;
    for (size_t i = 0; i < M/3; i++) {
        auto& mat3x3 = left_mat.mat[i][row_idx/3];
        vec.vec[i] = luisa::compute::make_float3(
            mat3x3[0][row_idx%3], 
            mat3x3[1][row_idx%3], 
            mat3x3[2][row_idx%3]);
    }
    return vec;
}
template<size_t M, size_t N>
void set_row_largemat(Var<LargeMatrix<M, N>>& left_mat, const size_t row_idx, const Var<LargeVector<M>>& vec) {
    for (size_t i = 0; i < M/3; i++) {
        auto& mat3x3 = left_mat.mat[i][row_idx/3];
        mat3x3[0][row_idx%3] = vec.vec[i][0];
        mat3x3[1][row_idx%3] = vec.vec[i][1];
        mat3x3[2][row_idx%3] = vec.vec[i][2];
    }
}


// Matrix add
template<size_t M, size_t N>
auto add_largemat(LargeMatrix<M, N>& output, const LargeMatrix<M, N>& left_mat, const LargeMatrix<M, N>& right_mat) 
{
    for (size_t i = 0; i < M/3; i++) 
        for(size_t j = 0; j < N/3; j++)
            output.mat[i][j] = left_mat.mat[i][j] + right_mat.mat[i][j];
}
template<size_t M, size_t N>
auto sub_largemat(LargeMatrix<M, N>& output, const LargeMatrix<M, N>& left_mat, const LargeMatrix<M, N>& right_mat) 
{
    for (size_t i = 0; i < M/3; i++) 
        for(size_t j = 0; j < N/3; j++)
            output.mat[i][j] = left_mat.mat[i][j] - right_mat.mat[i][j];
}
template<size_t M, size_t N>
auto add_largemat(const LargeMatrix<M, N>& left_mat, const LargeMatrix<M, N>& right_mat) 
{
    LargeMatrix<M, N> output;
    add_largemat(output, left_mat, right_mat);
    return output;
}
template<size_t M, size_t N>
auto sub_largemat(const LargeMatrix<M, N>& left_mat, const LargeMatrix<M, N>& right_mat) 
{
    LargeMatrix<M, N> output;
    sub_largemat(output, left_mat, right_mat);
    return output;
}

template<size_t M, size_t N>
auto add_largemat(Var<LargeMatrix<M, N>>& output, const Var<LargeMatrix<M, N>>& left_mat, const Var<LargeMatrix<M, N>>& right_mat) {
    for (size_t i = 0; i < M/3; i++)
        for (size_t j = 0; j < N/3; j++)
            output.mat[i][j] = left_mat.mat[i][j] + right_mat.mat[i][j];
}
template<size_t M, size_t N>
auto sub_largemat(Var<LargeMatrix<M, N>>& output, const Var<LargeMatrix<M, N>>& left_mat, const Var<LargeMatrix<M, N>>& right_mat) {
    for (size_t i = 0; i < M/3; i++)
        for (size_t j = 0; j < N/3; j++)
            output.mat[i][j] = left_mat.mat[i][j] - right_mat.mat[i][j];
}
template<size_t M, size_t N>
auto add_largemat(const Var<LargeMatrix<M, N>>& left_mat, const Var<LargeMatrix<M, N>>& right_mat) {
    Var<LargeMatrix<M, N>> output;
    add_largemat(output, left_mat, right_mat);
    return output;
}
template<size_t M, size_t N>
auto sub_largemat(const Var<LargeMatrix<M, N>>& left_mat, const Var<LargeMatrix<M, N>>& right_mat) {
    Var<LargeMatrix<M, N>> output;
    sub_largemat(output, left_mat, right_mat);
    return output;
}


// Matrix mult scalar
template<size_t M, size_t N>
void mult_largemat_scalar(LargeMatrix<M, N>& output, const LargeMatrix<M, N>& left_mat, float value) {
    for (size_t i = 0; i < M/3; i++) 
        for(size_t j = 0; j < N/3; j++)
            output.mat[i][j] = left_mat.mat[i][j] * value;
}
template<size_t M, size_t N>
auto mult_largemat_scalar(const LargeMatrix<M, N>& left_mat, const float value) {
    LargeMatrix<M, N> output;
    mult_largemat_scalar(output, left_mat, value);
    return output;
}


template<size_t M, size_t N>
void mult_largemat_scalar(Var<LargeMatrix<M, N>>& output, const Var<LargeMatrix<M, N>>& left_mat, const Var<float>& value) {
    for (size_t i = 0; i < M/3; i++) 
        for(size_t j = 0; j < N/3; j++)
            output.mat[i][j] = left_mat.mat[i][j] * value;
}
template<size_t M, size_t N>
auto mult_largemat_scalar(const Var<LargeMatrix<M, N>>& left_mat, const Var<float>& value) {
    Var<LargeMatrix<M, N>> output;
    mult_largemat_scalar(output, left_mat, value);
    return output;
}


// Matrix mult matrix
template<size_t M, size_t N, size_t L>
auto mult_largemat_mat(LargeMatrix<L, N>& result, const LargeMatrix<M, N>& left_mat, const LargeMatrix<L, M>& right_mat) {
    for (size_t i = 0; i < L/3; i++) {
        for (size_t j = 0; j < N/3; j++){
            result.mat[i][j] = luisa::make_float3x3(0.0f);
            for (size_t k = 0; k < M/3; k++){
                result.mat[i][j] = result.mat[i][j] + left_mat.mat[k][j] * right_mat.mat[i][k];
            }
        }
    }
}
template<size_t M, size_t N, size_t L>
auto mult_largemat_mat(const LargeMatrix<M, N>& left_mat, const LargeMatrix<L, M>& right_mat) {
    LargeMatrix<L, N> result;
    mult_largemat_mat(result, left_mat, right_mat);
    return result;
}

template<size_t M, size_t N, size_t L>
auto mult_largemat_mat(Var<LargeMatrix<L, N>>& result, const Var<LargeMatrix<M, N>>& left_mat, const Var<LargeMatrix<L, M>>& right_mat) {
    for (size_t i = 0; i < L/3; i++) {
        for (size_t j = 0; j < N/3; j++){
            result.mat[i][j] = luisa::compute::make_float3x3(0.0f);
            for(size_t k = 0; k < M/3; k++){
                result.mat[i][j] = result.mat[i][j] + left_mat.mat[k][j] * right_mat.mat[i][k];
            }
        }
    }
}
template<size_t M, size_t N, size_t L>
auto mult_largemat_mat(const Var<LargeMatrix<M, N>>& left_mat, const Var<LargeMatrix<L, M>>& right_mat) {
    Var<LargeMatrix<L, N>> result;
    mult_largemat_mat(result, left_mat, right_mat);
    return result;
}


// Matrix mult vec
template<size_t M, size_t N>
auto mult_largemat_vec(LargeVector<N>& result, const LargeMatrix<M, N>& left_mat, const luisa::float3& vec){
    static_assert(M == 3, "Can not multiply!");
    for (size_t i = 0; i < N/3; i++){
        result.vec[i] = left_mat.mat[0][i] * vec;
    }
}
template<size_t M, size_t N>
auto mult_largemat_vec(const LargeMatrix<M, N>& left_mat, const luisa::float3& vec){
    static_assert(M == 3, "Can not multiply!");
    LargeVector<N> result;
    mult_largemat_vec(result, left_mat, vec);
    return result;
}

template<size_t M, size_t N>
auto mult_largemat_vec(LargeVector<N>& result, const LargeMatrix<M, N>& left_mat, const LargeVector<M>& vec) {
    // static_assert(M == L, "Can not multiply!");
    for (size_t i = 0; i < N/3; i++) {
        result.vec[i] = luisa::make_float3(0.0f);
        for (size_t j = 0; j < M/3; j++){
            result.vec[i] += left_mat.mat[j][i] * vec.vec[j];
        }
    }
}
template<size_t M, size_t N>
auto mult_largemat_vec(const LargeMatrix<M, N>& left_mat, const LargeVector<M>& vec) {
    // static_assert(M == L, "Can not multiply!");
    LargeVector<N> result;
    mult_largemat_vec(result, left_mat, vec);
    return result;
}

template<size_t M, size_t N>
auto mult_largemat_vec(Var<LargeVector<N>>& result, const Var<LargeMatrix<M, N>>& left_mat, const Var<luisa::float3>& vec) {
    static_assert(M == 3, "Can not multiply!");
    for (size_t i = 0; i < N/3; i++) {
        result.vec[i] = left_mat.mat[0][i] * vec;
    }
}
template<size_t M, size_t N>
auto mult_largemat_vec(const Var<LargeMatrix<M, N>>& left_mat, const Var<luisa::float3>& vec) {
    static_assert(M == 3, "Can not multiply!");
    Var<LargeVector<N>> result;
    mult_largemat_vec(result, left_mat, vec);
    return result;
}

template<size_t M, size_t N>
auto mult_largemat_vec(Var<LargeVector<N>>& result, const Var<LargeMatrix<M, N>>& left_mat, const Var<LargeVector<M>>& vec) {
    for (size_t i = 0; i < N/3; i++) {
        result.vec[i] = luisa::compute::make_float3(0.0f);
        for (size_t j = 0; j < M/3; j++){
            result.vec[i] += left_mat.mat[j][i] * vec.vec[j];
        }
    }
}
template<size_t M, size_t N>
auto mult_largemat_vec(const Var<LargeMatrix<M, N>>& left_mat, const Var<LargeVector<M>>& vec) {
    Var<LargeVector<N>> result;
    mult_largemat_vec(result, left_mat, vec);
    return result;
}

// Matrix transfer
template<size_t M, size_t N>
auto transpose_largemat(const LargeMatrix<M, N>& left_mat) {
    LargeMatrix<N, M> output;
    for (size_t i = 0; i < N/3; i++) 
        for(size_t j = 0; j < M/3; j++)
            output.mat[i][j] = transpose_mat(left_mat[j][i]);
    return output;
}
template<size_t M, size_t N>
auto transpose_largemat(const Var<LargeMatrix<M, N>>& left_mat) {
    Var<LargeMatrix<N, M>> output;
    for (size_t i = 0; i < N/3; i++)
        for (size_t j = 0; j < M/3; j++)
            output.mat[i][j] = transpose_mat(left_mat.mat[j][i]);
    return output;
}




// = left*(right)T
template<size_t M, size_t N>
auto outer_product_largevec(const LargeVector<N>& left_vec, const LargeVector<M>& right_vec){
    // 对于每一列，拿左向量乘以右向量的对应元素
    LargeMatrix<M, N> output;
    for (size_t i = 0; i < M; i++){
        LargeVector<N> current_culumn = mult_largevec_scalar(left_vec, right_vec.vec[i/3][i%3]);
        set_colomn_largemat(output, i, current_culumn);
    }  
    return output;
} 
template<size_t M, size_t N>
auto outer_product_largevec(const Var<LargeVector<N>>& left_vec, const Var<LargeVector<M>>& right_vec) {
    Var<LargeMatrix<M, N>> output;
    for (size_t i = 0; i < M; i++){
        Var<LargeVector<N>> current_culumn = mult_largevec_scalar(left_vec, right_vec.vec[i/3][i%3]);
        set_colomn_largemat(output, i, current_culumn);
    }  
    return output;
}



inline void print_largevec(const LargeVector<3>& vec)  { luisa::log_info("({:>10.5f} {:>10.5f} {:>10.5f})", vec.vec[0].x, vec.vec[0].y, vec.vec[0].z); }
inline void print_largevec(const LargeVector<6>& vec)  { luisa::log_info("({:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f})", vec.vec[0].x, vec.vec[0].y, vec.vec[0].z, vec.vec[1].x, vec.vec[1].y, vec.vec[1].z); }
inline void print_largevec(const LargeVector<9>& vec)  { luisa::log_info("({:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f})", vec.vec[0].x, vec.vec[0].y, vec.vec[0].z, vec.vec[1].x, vec.vec[1].y, vec.vec[1].z, vec.vec[2].x, vec.vec[2].y, vec.vec[2].z); }
inline void print_largevec(const LargeVector<12>& vec)  { luisa::log_info("({:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f})", vec.vec[0].x, vec.vec[0].y, vec.vec[0].z, vec.vec[1].x, vec.vec[1].y, vec.vec[1].z, vec.vec[2].x, vec.vec[2].y, vec.vec[2].z, vec.vec[3].x, vec.vec[3].y, vec.vec[3].z); }
template <size_t M, size_t N> inline void print_largemat(const LargeMatrix<M, N>&  mat) { for (uint row = 0; row < N; row++) print_largevec(get_row_largemat(mat, row)); }


// inline void print_largevec(const Var<LargeVector<3>>& vec)   { luisa::compute::device_log("({} {} {})", vec.vec[0].x, vec.vec[0].y, vec.vec[0].z); }
// inline void print_largevec(const Var<LargeVector<6>>& vec)   { luisa::compute::device_log("({} {} {} {} {} {})", vec.vec[0].x, vec.vec[0].y, vec.vec[0].z, vec.vec[1].x, vec.vec[1].y, vec.vec[1].z); }
// inline void print_largevec(const Var<LargeVector<9>>& vec)   { luisa::compute::device_log("({} {} {} {} {} {} {} {} {})", vec.vec[0].x, vec.vec[0].y, vec.vec[0].z, vec.vec[1].x, vec.vec[1].y, vec.vec[1].z, vec.vec[2].x, vec.vec[2].y, vec.vec[2].z); }
// inline void print_largevec(const Var<LargeVector<12>>& vec)  { luisa::compute::device_log("({} {} {} {} {} {} {} {} {} {} {} {})", vec.vec[0].x, vec.vec[0].y, vec.vec[0].z, vec.vec[1].x, vec.vec[1].y, vec.vec[1].z, vec.vec[2].x, vec.vec[2].y, vec.vec[2].z, vec.vec[3].x, vec.vec[3].y, vec.vec[3].z); }
inline void print_largevec(const Var<LargeVector<3>>& vec)  { luisa::compute::device_log("({})", vec.vec[0]); }
inline void print_largevec(const Var<LargeVector<6>>& vec)  { luisa::compute::device_log("({} {})", vec.vec[0], vec.vec[1]); }
inline void print_largevec(const Var<LargeVector<9>>& vec)  { luisa::compute::device_log("({} {} {})", vec.vec[0], vec.vec[1], vec.vec[2]); }
inline void print_largevec(const Var<LargeVector<12>>& vec) { luisa::compute::device_log("({} {} {} {})", vec.vec[0], vec.vec[1], vec.vec[2], vec.vec[3]); }
// inline void print_largevec(const Var<LargeVector<3>>& vec)   { luisa::compute::device_log("({:>10.5f} {:>10.5f} {:>10.5f})", vec.vec[0].x, vec.vec[0].y, vec.vec[0].z); }
// inline void print_largevec(const Var<LargeVector<6>>& vec)   { luisa::compute::device_log("({:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f})", vec.vec[0].x, vec.vec[0].y, vec.vec[0].z, vec.vec[1].x, vec.vec[1].y, vec.vec[1].z); }
// inline void print_largevec(const Var<LargeVector<9>>& vec)   { luisa::compute::device_log("({:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f})", vec.vec[0].x, vec.vec[0].y, vec.vec[0].z, vec.vec[1].x, vec.vec[1].y, vec.vec[1].z, vec.vec[2].x, vec.vec[2].y, vec.vec[2].z); }
// inline void print_largevec(const Var<LargeVector<12>>& vec)  { luisa::compute::device_log("({:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f})", vec.vec[0].x, vec.vec[0].y, vec.vec[0].z, vec.vec[1].x, vec.vec[1].y, vec.vec[1].z, vec.vec[2].x, vec.vec[2].y, vec.vec[2].z, vec.vec[3].x, vec.vec[3].y, vec.vec[3].z); }
template <size_t M, size_t N> inline void print_largemat(const Var<LargeMatrix<M, N>>&  mat) { for (uint row = 0; row < N; row++) print_largevec(get_row_largemat(mat, row)); }


}

/*
/// float4x3 multiplied by float
[[nodiscard]] constexpr auto operator*(const luisa::float4x3 m, float s) noexcept {
    return luisa::float4x3{m[0] * s, m[1] * s, m[2] * s, m[3] * s};
}

/// float4x3 multiplied by float
[[nodiscard]] constexpr auto operator*(float s, const luisa::float4x3 m) noexcept {
    return m * s;
}

/// float4x3 divided by float
[[nodiscard]] constexpr auto operator/(const luisa::float4x3 m, float s) noexcept {
    return m * (1.0f / s);
}

/// floa4x3 dot float2
[[nodiscard]] constexpr auto operator*(const luisa::float4x3 m, const luisa::float2 v) noexcept {
    return v.x * m[0] + v.y * m[1];
}

// /// float4x3 multiply(matmul)
// [[nodiscard]] constexpr auto operator*(const luisa::float4x3 lhs, const luisa::float3x4 rhs) noexcept {
//     return luisa::float4x3{lhs * rhs[0], lhs * rhs[1], lhs * rhs[2], lhs * rhs[3]};
// }

/// float4x3 plus
[[nodiscard]] constexpr auto operator+(const luisa::float4x3 lhs, const luisa::float4x3 rhs) noexcept {
    return luisa::float4x3{lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2], lhs[3] + rhs[3]};
}

/// float4x3 minus
[[nodiscard]] constexpr auto operator-(const luisa::float4x3 lhs, const luisa::float4x3 rhs) noexcept {
    return luisa::float4x3{lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2], lhs[3] - rhs[3]};
}




/// float2x3 multiplied by float
[[nodiscard]] constexpr auto operator*(const luisa::float2x3 m, float s) noexcept {
    return luisa::float2x3{m[0] * s, m[1] * s};
}

/// float2x3 multiplied by float
[[nodiscard]] constexpr auto operator*(float s, const luisa::float2x3 m) noexcept {
    return m * s;
}

/// float2x3 divided by float
[[nodiscard]] constexpr auto operator/(const luisa::float2x3 m, float s) noexcept {
    return m * (1.0f / s);
}

/// floa2x3 dot float3 // mult template
[[nodiscard]] constexpr auto operator*(const luisa::float2x3 m, const luisa::float2 v) noexcept {
    return v.x * m[0] + v.y * m[1];
}

/// float2x3 multiply(matmul)
[[nodiscard]] constexpr auto operator*(const luisa::float2x3 lhs, const luisa::float2x2 rhs) noexcept {
    return luisa::float2x3{lhs * rhs[0], lhs * rhs[1]};
}

/// float2x3 plus
[[nodiscard]] constexpr auto operator+(const luisa::float2x3 lhs, const luisa::float2x3 rhs) noexcept {
    return luisa::float2x3{lhs[0] + rhs[0], lhs[1] + rhs[1]};
}

/// float2x3 minus
[[nodiscard]] constexpr auto operator-(const luisa::float2x3 lhs, const luisa::float2x3 rhs) noexcept {
    return luisa::float2x3{lhs[0] - rhs[0], lhs[1] - rhs[1]};
}

namespace luisa {

/// make float4x3
[[nodiscard]] constexpr auto make_float4x3(float v) noexcept {
    return float4x3{float3{v},
                   float3{v},
                   float3{v},
                   float3{v}};
}

/// make float4x3
[[nodiscard]] constexpr auto make_float4x3(float3 c0, float3 c1, float3 c2, float3 c3) noexcept {
    return float4x3{c0, c1, c2, c3};
}

/// make float4x3
[[nodiscard]] constexpr auto make_float4x3(
    float m00, float m01, float m02,
    float m10, float m11, float m12,
    float m20, float m21, float m22,
    float m30, float m31, float m32) noexcept {
    return float4x3{float3{m00, m01, m02},
                   float3{m10, m11, m12},
                   float3{m20, m21, m22},
                   float3{m30, m31, m32}};
}

/// make float4x3
[[nodiscard]] constexpr auto make_float4x3(float4x3 m) noexcept {
    return m;
}

// Matrix Functions
[[nodiscard]] constexpr auto transpose(const float4x3 m) noexcept {
    return float3x4{float4{m[0].x, m[1].x, m[2].x, m[3].x},
                   float4{m[0].y, m[1].y, m[2].y, m[3].y},
                   float4{m[0].z, m[1].z, m[2].z, m[3].z}};
}



/// make float2x3
[[nodiscard]] constexpr auto make_float2x3(float v) noexcept {
    return float2x3{float3{v},
                   float3{v}};
}

/// make float2x3
[[nodiscard]] constexpr auto make_float2x3(float3 c0, float3 c1, float3 c2, float3 c3) noexcept {
    return float2x3{c0, c1};
}

/// make float2x3
[[nodiscard]] constexpr auto make_float2x3(
    float m00, float m01, float m02,
    float m10, float m11, float m12,
    float m20, float m21, float m22,
    float m30, float m31, float m32) noexcept {
    return float2x3{float3{m00, m01, m02},
                   float3{m10, m11, m12}};
}

/// make float2x3
[[nodiscard]] constexpr auto make_float4x3(float2x3 m) noexcept {
    return m;
}
}// namespace luisa
*/