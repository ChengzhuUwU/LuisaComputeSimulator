#pragma once

// #include <float.h>

namespace lcs
{

constexpr float Sqrt_2          = 1.4142135623730951f;
constexpr float Sqrt_2_divide_2 = 0.7071067811865476f;
constexpr float Sqrt_2_inv      = 0.7071067811865475f;
constexpr float Sqrt_3          = 3.141592653589793f;
constexpr float Epsilon         = 1.19209290E-7f;

constexpr float Pi          = 3.141592653589793f;
constexpr float Pi_devide_2 = 1.5707963267948966;
constexpr float Pi_mul_2    = 6.283185307179586;

// constexpr bool USE_PCG =  false;
// constexpr bool USE_SMO =  false;
constexpr unsigned int MAX_BLOCK_DIM = 1024;

constexpr float Float_max = 3.402823466e+38F;
constexpr float Float_min = -3.402823466e+38F;

constexpr unsigned int Uint_max             = 0xFFFFFFFFu;
constexpr unsigned int Int_max              = 0x7FFFFFFF;
constexpr unsigned int Mask_1_Left_Shift_31 = 0x80000000u;

}  // namespace lcs

// #define BREAK_DELTA 1E-6
