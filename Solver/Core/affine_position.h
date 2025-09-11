#pragma once

#include <luisa/luisa-compute.h>

namespace lcs 
{

static inline luisa::float4x4 scale(const luisa::float3& v) 
{
    // | sx  0   0   0 |
    // | 0   sy  0   0 |
    // | 0   0   sz  0 |
    // | 0   0   0   1 |

    luisa::float4x4 result = luisa::make_float4x4(
        luisa::make_float4(v[0], 0.0f, 0.0f, 0.0f),
        luisa::make_float4(0.0f, v[1], 0.0f, 0.0f),
        luisa::make_float4(0.0f, 0.0f, v[2], 0.0f),
        luisa::make_float4(0.0f, 0.0f, 0.0f, 1.0f));
    return result;
}
static inline luisa::float4x4 translate(const luisa::float3& v) 
{
    // | 1 0 0 x |
    // | 0 1 0 y |
    // | 0 0 1 z |
    // | 0 0 0 1 |
    
    // luisa::float4x4 result = Identity4x4;
    // set(result, 0, 0, 1.0f);
    // set(result, 1, 1, 1.0f);
    // set(result, 2, 2, 1.0f);
    // set(result, 3, 3, 1.0f);
    // set(result, 3, 0, v[0]);
    // set(result, 3, 1, v[1]);
    // set(result, 3, 2, v[2]);
    // return result;

    return luisa::make_float4x4(
        luisa::make_float4(1.0f, 0.0f, 0.0f, 0.0f),
        luisa::make_float4(0.0f, 1.0f, 0.0f, 0.0f),
        luisa::make_float4(0.0f, 0.0f, 1.0f, 0.0f),
        luisa::make_float4(v[0], v[1], v[2], 1.0f)
    );
}
static inline luisa::float4x4 rorateX(float angleX) 
{
    float cosX = luisa::cos(angleX);
    float sinX = luisa::sin(angleX);

    // |  1     0      0     0 |
    // |  0   cos(θ) -sin(θ) 0 |
    // |  0   sin(θ)  cos(θ) 0 |
    // |  0     0      0     1 |

    return luisa::transpose(luisa::make_float4x4(
        luisa::make_float4(1.0f, 0.0f, 0.0f,  0.0f),
        luisa::make_float4(0.0f, cosX, sinX, 0.0f),
        luisa::make_float4(0.0f, -sinX, cosX,  0.0f),
        luisa::make_float4(0.0f, 0.0f, 0.0f,  1.0f)
    ));
}
static inline luisa::float4x4 rorateY(float angleY) 
{
    float cosY = luisa::cos(angleY);
    float sinY = luisa::sin(angleY);

    // |  cos(θ)  0  sin(θ)  0 |
    // |    0     1    0     0 |
    // | -sin(θ)  0  cos(θ)  0 |
    // |    0     0    0     1 |
    return luisa::transpose(luisa::make_float4x4(
        luisa::make_float4(cosY,  0.0f, -sinY, 0.0f),
        luisa::make_float4(0.0f,  1.0f, 0.0f, 0.0f),
        luisa::make_float4(sinY, 0.0f, cosY, 0.0f),
        luisa::make_float4(0.0f,  0.0f, 0.0f, 1.0f)
    ));
}
static inline luisa::float4x4 rorateZ(float angleZ) 
{
    float cosZ = luisa::cos(angleZ);
    float sinZ = luisa::sin(angleZ);

    // | cos(θ) -sin(θ)  0  0 |
    // | sin(θ)  cos(θ)  0  0 |
    // |   0       0     1  0 |
    // |   0       0     0  1 |

    return luisa::transpose(luisa::make_float4x4(
        luisa::make_float4(cosZ, -sinZ, 0.0f, 0.0f),
        luisa::make_float4(sinZ, cosZ,  0.0f, 0.0f),
        luisa::make_float4(0.0f, 0.0f,  1.0f, 0.0f),
        luisa::make_float4(0.0f, 0.0f,  0.0f, 1.0f)
    ));
}
static inline luisa::float4x4 rotate(const luisa::float3& axis) 
{
    return rorateX(axis[0]) * rorateY(axis[1]) * rorateZ(axis[2]);
}

inline luisa::float4x4 make_model_matrix(const luisa::float3& t, const luisa::float3& r, const luisa::float3& s)
{
    return translate(t) * rotate(r) * scale(s)  ;
    // return scale(s) * (rotate(r) * translate(t));
    // return translate(t) ;
}

inline luisa::float3 affine_position(const luisa::float4x4& model_matrix, const luisa::float3& model_position)
{
    luisa::float4 mult_position = model_matrix * luisa::make_float4(model_position[0], model_position[1], model_position[2], 1.0f);
    return luisa::make_float3(mult_position[0], mult_position[1], mult_position[2]);
}


}