#pragma once

#include <cstddef>
#include <array>

#include <luisa/core/basic_types.h>
// #include <luisa/core/stl/hash_fwd.h>
// #include <luisa/core/basic_traits.h>

#define PTR(T) luisa::compute::BufferVar<T>

namespace luisa {

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

template<size_t M, size_t N>
struct hash<XMatrix<M, N>> {
    using is_avalanching = void;
    [[nodiscard]] uint64_t operator()(XMatrix<M, N> m, uint64_t seed = hash64_default_seed) const noexcept {
        std::array<float, M * N> data{};
        for (size_t i = 0u; i < M; i++) {
            for (size_t j = 0u; j < N; j++) {
                data[i * N + j] = m[i][j];
            }
        }
        return hash64(data.data(), data.size() * sizeof(float), seed);
    }
};

using float4x3 = XMatrix<4, 3>;
using float2x3 = XMatrix<2, 3>;
using float3x4 = XMatrix<3, 4>;

}// namespace luisa


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
