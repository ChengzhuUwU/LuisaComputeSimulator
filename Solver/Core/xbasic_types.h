#pragma once

#include <cstddef>
#include <array>
#include <luisa/core/basic_types.h>
#include <luisa/dsl/struct.h>
// #include <luisa/core/stl/hash_fwd.h>
// #include <luisa/core/basic_traits.h>

#define PTR(T) luisa::compute::BufferVar<T>

namespace lcsv {

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
    //     for (uint i = 0; i < 4; i++) {
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
        for (uint i = 0; i < M; i++) {
            cols[i] += right[i];
        }
    }
    constexpr void operator+(
        const XMatrix<M, N>& right) noexcept {
        XMatrix<M, N> result;
        for (uint i = 0; i < M; i++) {
            result[i] = cols[i] = right[i];
        }
        return result;
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
// template<uint M, uint N, uint L> // TODO: use column acceleration
// [[nodiscard]] constexpr auto operator*(
//     const luisa::XMatrix<M, N>& left, 
//     const luisa::XMatrix<L, M>& right) noexcept {
//     luisa::XMatrix<L, N> result;
//     for (uint i = 0; i < L; i++) {
//         for(uint j = 0; j < N; j++){
//             result[i][j] = 0.0f;
//             for(uint k = 0; k < M; k++){
//                 result[i][j] += left[k][j] * right[i][k];
//             }
//         }
//     }
//     return result;
// }
// template<uint M, uint N>
// [[nodiscard]] constexpr luisa::XMatrix<M, N> operator+(
//     const luisa::XMatrix<M, N>& left, 
//     const luisa::XMatrix<M, N>& right) noexcept {
//     luisa::XMatrix<M, N> result;
//     for (uint i = 0; i < M; i++) {
//         result[i] = left[i] + right[i];
//     }
//     return result;
// }

namespace lcsv {

using float2x3 = XMatrix<2, 3>;
using float2x4 = XMatrix<2, 4>;
using float3x2 = XMatrix<3, 2>;
using float3x4 = XMatrix<3, 4>;
using float4x2 = XMatrix<4, 2>;
using float4x3 = XMatrix<4, 3>;

// #define MAKE_XMATRIX_TYPE(M, N) \
//     struct float##M##x##N { \
//         luisa::float##N cols[M]; \
//         [[nodiscard]] constexpr luisa::float##N &operator[](size_t i) noexcept { return cols[i]; } \
//         [[nodiscard]] constexpr const luisa::float##N &operator[](size_t i) const noexcept { return cols[i]; } \
//         constexpr void operator+=(const float##M##x##N &right) noexcept { \
//             for (uint i = 0; i < M; i++) { \
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

// template<uint M, uint N, uint L> // TODO: use column acceleration
// [[nodiscard]] constexpr auto mult_mat(
//     const luisa::compute::Var<luisa::compute::XMatrix<M, N>>& left, 
//     const luisa::compute::Var<luisa::compute::XMatrix<L, M>>& right) noexcept {
//     luisa::compute::Var<luisa::compute::XMatrix<L, N>> result;
//     for (uint i = 0; i < L; i++) {
//         for(uint j = 0; j < N; j++){
//             result[i][j] = 0.0f;
//             for(uint k = 0; k < M; k++){
//                 result[i][j] += left[k][j] * right[i][k];
//             }
//         }
//     }
//     return result;
// }
// template<uint M, uint N>
// [[nodiscard]] constexpr luisa::compute::Var<luisa::XMatrix<M, N>> add(
//     const luisa::compute::Var<luisa::XMatrix<M, N>>& left, 
//     const luisa::compute::Var<luisa::XMatrix<M, N>>& right) noexcept {
//     luisa::compute::Var<luisa::XMatrix<M, N>> result;
//     for (uint i = 0; i < M; i++) {
//         result.cols[i] = left.cols[i] + right.cols[i];
//     }
//     return result;
// }
// template<uint M, uint N>
// constexpr void add(
//     luisa::compute::Var<luisa::XMatrix<M, N>>& result,
//     const luisa::compute::Var<luisa::XMatrix<M, N>>& left, 
//     const luisa::compute::Var<luisa::XMatrix<M, N>>& right) noexcept {
//     for (uint i = 0; i < M; i++) {
//         result.cols[i] = left.cols[i] + right.cols[i];
//     }
// }
// template<uint M, uint N>
// [[nodiscard]] constexpr luisa::compute::Var<luisa::XMatrix<M, N>> sub(
//     const luisa::compute::Var<luisa::XMatrix<M, N>>& left, 
//     const luisa::compute::Var<luisa::XMatrix<M, N>>& right) noexcept {
//     luisa::compute::Var<luisa::XMatrix<M, N>> result;
//     for (uint i = 0; i < M; i++) {
//         result.cols[i] = left.cols[i] - right.cols[i];
//     }
//     return result;
// }

} // namespace luisa::compute

// LUISA_STRUCT(lcsv::float2x3, cols) {};
// LUISA_STRUCT(lcsv::float2x4, cols) {};
// LUISA_STRUCT(lcsv::float3x2, cols) {};
// LUISA_STRUCT(lcsv::float3x4, cols) {};
// LUISA_STRUCT(lcsv::float4x2, cols) {};
// LUISA_STRUCT(lcsv::float4x3, cols) {};

#define REGIRSTER_XMATRIX_TO_STRUCT(M, N) \
    LUISA_STRUCT(lcsv::float##M##x##N, cols) {  \
        [[nodiscard]] auto mult(const luisa::compute::Expr<float> alpha) const noexcept {   \
            luisa::compute::Var<lcsv::float##M##x##N> result;  \
            for (uint i = 0; i < M; i++) {  \
                result.cols[i] = alpha * cols[i];  \
            }  \
            return result;  \
        }  \
        [[nodiscard]] inline auto mult(const luisa::compute::Expr<luisa::float##N>& vec) noexcept   \
        {  \
            luisa::compute::Var<luisa::float##N> output;  \
            for (int i = 0; i < M; ++i) { \
                output[i] = 0.0f;         \
                for (int j = 0; j < N; j++) { \
                    output[i] += cols[j][i] * vec[j]; \
                }  \
            }  \
            return output;  \
        }  \
    };  

REGIRSTER_XMATRIX_TO_STRUCT(2, 3);
REGIRSTER_XMATRIX_TO_STRUCT(2, 4);
REGIRSTER_XMATRIX_TO_STRUCT(3, 2);
REGIRSTER_XMATRIX_TO_STRUCT(3, 4);
REGIRSTER_XMATRIX_TO_STRUCT(4, 2);
REGIRSTER_XMATRIX_TO_STRUCT(4, 3);

// auto operator=(const luisa::compute::Expr<lcsv::float##M##x##N> right) noexcept {   \
//     for (uint i = 0; i < M; i++) {  \
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

namespace lcsv {

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