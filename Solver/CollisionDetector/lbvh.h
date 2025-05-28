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

template<typename T>
static inline void resize_buffer(luisa::compute::Device& device, luisa::compute::Buffer<T>& buffer, const uint size)
{
    buffer = device.create_buffer<T>(size);
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

    void allocate(
        luisa::compute::Device& device, 
        const uint input_num, 
        const LBVHTreeType input_tree_type,
        const LBVHUpdateType input_update_type
    )
    {
        const uint num_leaves = input_num;
        const uint num_inner_nodes = num_leaves - 1;
        const uint num_nodes = num_leaves + num_inner_nodes;

        this->num_leaves = num_leaves;
        this->num_inner_nodes = num_inner_nodes;
        this->num_nodes = num_nodes;

        this->tree_type = input_tree_type;
        this->update_type = input_update_type;

        resize_buffer(device, this->sa_leaf_center, num_leaves);
        resize_buffer(device, this->sa_block_aabb, get_dispatch_block(num_leaves, 256));
        resize_buffer(device, this->sa_morton, num_leaves);
        resize_buffer(device, this->sa_morton_sorted, num_leaves);
        resize_buffer(device, this->sa_sorted_get_original, num_leaves);
        resize_buffer(device, this->sa_parrent, num_nodes);
        resize_buffer(device, this->sa_children, num_nodes);
        resize_buffer(device, this->sa_escape_index, num_nodes);
        resize_buffer(device, this->sa_left_and_escape, num_nodes);
        resize_buffer(device, this->sa_node_aabb, num_nodes);
        resize_buffer(device, this->sa_apply_flag, num_nodes);
        resize_buffer(device, this->sa_is_healthy, 1);
    }
};

class LBVH
{
    template<typename T>
    using Buffer = luisa::compute::Buffer<T>;
    using Stream = luisa::compute::Stream;
    using Device = luisa::compute::Device;
    
public:
    // void init(luisa::compute::Device& device, luisa::compute::Stream& stream, 
    //      const uint input_num, const LBVHTreeType tree_type, const LBVHUpdateType update_type);
    void unit_test(luisa::compute::Device& device, luisa::compute::Stream& stream);
    void set_lbvh_data(LbvhData<luisa::compute::Buffer>* input_ptr) { lbvh_data = input_ptr; }

private:
    void compile(luisa::compute::Device& device);

public:
    void reduce_vert_tree_aabb(Stream& stream, const Buffer<float3>& input_position);
    void reduce_face_tree_aabb(Stream& stream, const Buffer<float3>& input_position, const Buffer<uint3>& input_faces);
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

private:

    luisa::compute::Shader<1, luisa::compute::BufferView<float3>> fn_reduce_vert_tree_global_aabb;
    luisa::compute::Shader<1, luisa::compute::BufferView<float3>, luisa::compute::BufferView<uint3>> fn_reduce_face_tree_global_aabb;
    
    luisa::compute::Shader<1> fn_reduce_aabb_2_pass_template;
    luisa::compute::Shader<1> fn_reset_tree;
    luisa::compute::Shader<1> fn_compute_mortons;
    luisa::compute::Shader<1> fn_apply_sorted;
    luisa::compute::Shader<1> fn_build_inner_nodes ;
    luisa::compute::Shader<1> fn_check_construction ;
    luisa::compute::Shader<1> fn_set_escape_index;
    luisa::compute::Shader<1> fn_set_left_and_escape;

    // Refit
    luisa::compute::Shader<1, luisa::compute::BufferView<float3>, float> fn_update_vert_tree_leave_aabb;
    luisa::compute::Shader<1, luisa::compute::BufferView<float3>, luisa::compute::BufferView<uint3>, float>  fn_update_face_tree_leave_aabb ;
    luisa::compute::Shader<1> fn_clear_apply_flag ;
    luisa::compute::Shader<1> fn_refit_kernel;
    luisa::compute::Shader<1, luisa::compute::BufferView<float3>,
                       luisa::compute::BufferView<uint>,
                       luisa::compute::BufferView<uint>, float> fn_query_kernel;


};
    
};