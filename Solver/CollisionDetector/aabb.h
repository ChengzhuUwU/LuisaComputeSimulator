#pragma once
#include <array>
#include <algorithm>
#include "Core/float_n.h"
#include "Core/float_nxn.h"

namespace lcsv
{

namespace AABB
{

inline float3 get_aabb_min(const float2x3& aabb) { return aabb[0]; }
inline float3 get_aabb_max(const float2x3& aabb) { return aabb[1]; }
inline Var<float3> get_aabb_min(const Var<float2x3>& aabb) { return aabb.cols[0]; }
inline Var<float3> get_aabb_max(const Var<float2x3>& aabb) { return aabb.cols[1]; }


inline auto make_aabb()  
{ 
    return makeFloat2x3(
        makeFloat3(1000.0f), 
        makeFloat3(1000.0f)
    ); 
}
template<typename Vec3> inline auto make_aabb(const Vec3& p1)  
{ 
    return makeFloat2x3(p1, p1); 
}
template<typename Vec3> inline auto make_aabb(const Vec3& p1, const Vec3& p2)  
{ 
    return makeFloat2x3(min_vec(p1, p2), max_vec(p1, p2)); 
}
template<typename Vec3>
inline auto make_aabb(const Vec3& p1, const Vec3& p2, const Vec3& p3) 
{
    return make_aabb(
        min_vec(p1, min_vec(p2, p3)),
        max_vec(p1, max_vec(p2, p3))
    );
}


template<typename AabbType, typename Float>
inline AabbType add_thickness(const AabbType& aabb, const Float thickness) 
{
    return makeFloat2x3(
        get_aabb_min(aabb) - thickness,
        get_aabb_max(aabb) + thickness
    );
}

template<typename AabbType>
inline AabbType add_aabb(const AabbType& aabb1, const AabbType& aabb2) 
{
    return {
        min_vec(get_aabb_min(aabb1), get_aabb_min(aabb2)),
        max_vec(get_aabb_max(aabb1), get_aabb_max(aabb2))
    };
}

template<typename AabbType, typename Float3>
inline bool is_overlap_pos(const AabbType& aabb, const Float3& pos) 
{
    const auto& mn = get_aabb_min(aabb);
    const auto& mx = get_aabb_max(aabb);
    return (pos.x >= mn.x && pos.y >= mn.y && pos.z >= mn.z) &&
           (pos.x <= mx.x && pos.y <= mx.y && pos.z <= mx.z);
}


} // namespace AABB
} // namespace lcsv

