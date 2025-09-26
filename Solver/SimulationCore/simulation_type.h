#pragma once

#include "Core/float_n.h"
#include "Core/xbasic_types.h"
#include <luisa/luisa-compute.h>

class SimulationType
{
  protected:
    // using Device      = luisa::compute::Device;
    // using CommandList = luisa::compute::CommandList;
    // using Stream      = luisa::compute::Stream;
    // using Type        = luisa::compute::Type;

    using float2 = luisa::float2;
    using float3 = luisa::float3;
    using float4 = luisa::float4;
    // using float3x3    = lcs::float3x3;
    // using float4x4    = lcs::float4x4;
    // using float4x3    = lcs::float4x3;
    using uint   = luisa::uint;
    using uint2  = luisa::uint2;
    using uint3  = luisa::uint3;
    using uint4  = luisa::uint4;
    using ulong  = luisa::ulong;
    using uchar  = luisa::ubyte;
    using uchar2 = luisa::ubyte2;
    using uchar4 = luisa::ubyte4;
};
