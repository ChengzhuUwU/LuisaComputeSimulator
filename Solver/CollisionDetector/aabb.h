#pragma once
#include <array>
#include <algorithm>
#include "Core/float_n.h"
#include "Core/float_nxn.h"

namespace lcs
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
        makeFloat3( 1000.0f), 
        makeFloat3(-1000.0f)
    ); 
}
inline auto make_aabbVar()  
{ 
    return makeFloat2x3(
        makeFloat3Var( 1000.0f), 
        makeFloat3Var(-1000.0f)
    ); 
}

template<typename Vec3> inline auto make_aabb_from_min_max(const Vec3& min_vec, const Vec3& max_vec)  
{ 
    return makeFloat2x3(
        min_vec, 
        max_vec); 
}

template<typename Vec3> inline auto make_aabb(const Vec3& p1)  
{ 
    return makeFloat2x3(
        p1, 
        p1); 
}
template<typename Vec3> inline auto make_aabb(const Vec3& p1, const Vec3& p2)  
{ 
    return makeFloat2x3(
        min_vec(p1, p2), 
        max_vec(p1, p2)); 
}
template<typename Vec3>
inline auto make_aabb(const Vec3& p1, const Vec3& p2, const Vec3& p3) 
{
    return makeFloat2x3(
        min_vec(p1, min_vec(p2, p3)),
        max_vec(p1, max_vec(p2, p3))
    );
}
template<typename Vec3>
inline auto make_aabb(const Vec3& p1, const Vec3& p2, const Vec3& p3, const Vec3& p4) 
{
    return makeFloat2x3(
        min_vec(min_vec(p1, p2), min_vec(p3, p4)),
        max_vec(max_vec(p1, p2), max_vec(p3, p4))
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
    return makeFloat2x3(
        min_vec(get_aabb_min(aabb1), get_aabb_min(aabb2)),
        max_vec(get_aabb_max(aabb1), get_aabb_max(aabb2))
    );
}

inline void reduce_aabb(Var<float2x3> & aabb1, const Var<float2x3> & aabb2) 
{
    aabb1.cols[0] = min_vec(get_aabb_min(aabb1), get_aabb_min(aabb2));
    aabb1.cols[1] = max_vec(get_aabb_max(aabb1), get_aabb_max(aabb2));
}

template<typename AabbType, typename Float3>
inline auto is_overlap_pos(const AabbType& aabb, const Float3& pos) 
{
    const auto& mn = get_aabb_min(aabb);
    const auto& mx = get_aabb_max(aabb);
    return all_vec(pos >= mn) &
           all_vec(pos <= mx);
}

template<typename AabbType>
inline auto is_overlap_aabb(const AabbType& left, const AabbType& right) 
{
    return all_vec(get_aabb_min(left) < get_aabb_max(right)) & all_vec(get_aabb_max(left) > get_aabb_min(right));
}



} // namespace AABB
} // namespace lcs

