#pragma once

#include "Core/scalar.h"
#include "SimulationCore/simulation_type.h"
#include "Utils/device_parallel.h"
#include <vector>
#include <string>
#include <luisa/luisa-compute.h>

namespace lcsv
{

using morton32 = unsigned int;
using morton64 = uint64_t;
using Morton32 = luisa::compute::Var<morton32>;
using Morton64 = luisa::compute::Var<morton64>;

template<typename UintType>
static inline UintType expand_bits(UintType bits)
{
    bits = (bits | (bits << 16)) & 0x030000FF;
    bits = (bits | (bits << 8))  & 0x0300F00F;
    bits = (bits | (bits << 4))  & 0x030C30C3;
    return (bits | (bits << 2))  & 0x09249249;
}

inline auto make_morton32(const luisa::compute::Float3& pos) 
{
    using namespace luisa::compute;
    const Uint precision = 10;
    const Float min_value = Float(0.0f);
    const Float max_value = Float((1 << precision) - 1);
    const Float range = Float(1 << precision);

    Float x = clamp_scalar(pos[0] * max_value, min_value, max_value);
    Float y = clamp_scalar(pos[1] * max_value, min_value, max_value);
    Float z = clamp_scalar(pos[2] * max_value, min_value, max_value);

    Uint xx = expand_bits(static_cast<Uint>(x));
    Uint yy = expand_bits(static_cast<Uint>(y));
    Uint zz = expand_bits(static_cast<Uint>(z));

    return (xx << 2) | (yy << 1) | zz;
}
inline auto make_morton32(const luisa::float3& pos) 
{
    const uint precision = 10;

    float x = clamp_scalar(pos[0] * (1 << precision), static_cast<float>(0.0f), (1 << precision) - 1.0f);
    float y = clamp_scalar(pos[1] * (1 << precision), static_cast<float>(0.0f), (1 << precision) - 1.0f);
    float z = clamp_scalar(pos[2] * (1 << precision), static_cast<float>(0.0f), (1 << precision) - 1.0f);

    uint xx = expand_bits(static_cast<uint>(x));
    uint yy = expand_bits(static_cast<uint>(y));
    uint zz = expand_bits(static_cast<uint>(z));

    return (xx << 2) | (yy << 1) | zz;
}

template<typename Float3, typename Uint>
inline auto make_morton64(const Float3& pos, const Uint index) 
{
    return make_morton32(pos) | (static_cast<morton64>(index) << 32);
}

using AabbData = float2x3;
enum LBVHTreeType{
    LBVHTreeTypeVert,
    LBVHTreeTypeFace,
    LBVHTreeTypeEdge
};
enum LBVHUpdateType{
    LBVHUpdateTypeCloth,
    LBVHUpdateTypeObstacle
};

template<template<typename...> typename BufferType>
struct LbvhData 
{
    BufferType<float3> sa_leaf_center;
    BufferType<AabbData> sa_block_aabb;
    BufferType<morton64> sa_morton;
    BufferType<morton64> sa_morton_sorted;
    BufferType<uint> sa_sorted_get_original;
    BufferType<uint> sa_parrent;
    BufferType<uint2> sa_children;
    BufferType<uint> sa_escape_index;
    BufferType<uint2> sa_left_and_escape;
    BufferType<AabbData> sa_node_aabb;
    BufferType<uint> sa_is_healthy; 
    BufferType<uint> sa_apply_flag;
    BufferType<AabbData> sa_node_aabb_model_position;

    uint num_leaves;
    uint num_nodes;
    uint num_inner_nodes;

    LBVHTreeType tree_type;
    LBVHUpdateType update_type;
};

class LBVH
{
    template<typename T>
    using Buffer = luisa::compute::Buffer<T>;
    using Stream = luisa::compute::Stream;
    using Device = luisa::compute::Device;
    
public:
    LBVH() = default;
    void init(luisa::compute::Device& device, luisa::compute::Stream& stream, 
         const uint input_num, const LBVHTreeType tree_type, const LBVHUpdateType update_type);
    void compile(luisa::compute::Device& device);

public:
    void construct_tree(Stream& stream);
    void refit(Stream& stream);
    void update_vert_tree_leave_aabb(Stream& stream, const Buffer<float3>& input_position);
    void update_face_tree_leave_aabb(Stream& stream, const Buffer<float3>& input_position, const Buffer<uint3>& input_faces);
    void broad_phase_query(Stream& stream, const Buffer<float3>& input_position, Buffer<uint>& broad_phase_list, Buffer<uint>& broadphase_count, const float thickness);
    
    // void construct_tree();
    // void refit();
    // void update_vert_tree_leave_aabb(const Buffer<float3>& input_position);
    // void update_face_tree_leave_aabb(const Buffer<float3>& input_position, const Buffer<uint3>& input_faces);
    // void broad_phase_query(const Buffer<float3>& input_position, Buffer<uint>& broad_phase_list, Buffer<uint>& broadphase_count, const float thickness);

private:
    // void reduce_vert_tree_global_aabb();
    // void reduce_face_tree_global_aabb();
    // LbvhData<luisa::compute::Buffer>& get_lbvh_data() { return lbvh_data; }

private:
    LbvhData<luisa::compute::Buffer>* lbvh_data;
    DeviceParallel* device_parallel;

};
    
};