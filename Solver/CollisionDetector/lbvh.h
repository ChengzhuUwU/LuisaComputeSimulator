#pragma once

#include "Core/scalar.h"
#include "SimulationCore/simulation_type.h"
#include <vector>
#include <string>
#include <luisa/luisa-compute.h>

namespace lcsv
{

using morton32 = unsigned int;
using morton64 = uint64_t;
using Morton32 = luisa::compute::Var<morton32>;
using Morton64 = luisa::compute::Var<morton64>;
using aabbData = float2x3;

template<typename T>
static inline void resize_buffer(luisa::compute::Device& device, luisa::compute::Buffer<T>& buffer, const uint size)
{
    buffer = device.create_buffer<T>(size);
}


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
    BufferType<aabbData> sa_block_aabb;
    BufferType<morton64> sa_morton;
    BufferType<morton64> sa_morton_sorted;
    BufferType<uint> sa_sorted_get_original;
    BufferType<uint> sa_parrent;
    BufferType<uint2> sa_children;
    BufferType<uint> sa_escape_index;
    BufferType<uint2> sa_left_and_escape;
    BufferType<aabbData> sa_node_aabb;
    BufferType<uint> sa_is_healthy; 
    BufferType<uint> sa_apply_flag;
    // BufferType<AabbData> sa_node_aabb_model_position;

    std::vector<morton64> host_morton64;
    std::vector<uint> host_sorted_get_original;
    std::vector<uint> host_parrent;
    std::vector<uint2> host_children;
    std::vector<aabbData> host_node_aabb;
    std::vector<uint> host_apply_flag;
    std::vector<uint> host_is_healthy;

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

        luisa::log_info("Allocate for {}-LBVH data : num_leaves = {}", 
            input_tree_type == LBVHTreeTypeVert ? "Vert" : input_tree_type == LBVHTreeTypeFace ? "Face" : "Edge", num_leaves);

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

        this->host_morton64.resize(num_leaves);
        this->host_sorted_get_original.resize(num_leaves);
        this->host_parrent.resize(num_nodes);
        this->host_children.resize(num_nodes);
        this->host_node_aabb.resize(num_nodes);
        this->host_apply_flag.resize(num_nodes);
        this->host_is_healthy.resize(1);
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
    void compile(luisa::compute::Device& device);
    void set_lbvh_data(LbvhData<luisa::compute::Buffer>* input_ptr) { lbvh_data = input_ptr; }

public:
    void reduce_vert_tree_aabb(Stream& stream, const Buffer<float3>& input_position);
    void reduce_edge_tree_aabb(Stream& stream, const Buffer<float3>& input_position, const Buffer<uint2>& input_edges);
    void reduce_face_tree_aabb(Stream& stream, const Buffer<float3>& input_position, const Buffer<uint3>& input_faces);
    void construct_tree(Stream& stream);
    void refit(Stream& stream);
    void update_vert_tree_leave_aabb(Stream& stream, const Buffer<float3>& start_position, const Buffer<float3>& end_position);
    void update_edge_tree_leave_aabb(Stream& stream, const Buffer<float3>& start_position, const Buffer<float3>& end_position, const Buffer<uint2>& input_edges);
    void update_face_tree_leave_aabb(Stream& stream, const Buffer<float3>& start_position, const Buffer<float3>& end_position, const Buffer<uint3>& input_faces);
    void broad_phase_query_from_verts(Stream& stream, 
        const Buffer<float3>& sa_x_begin, 
        const Buffer<float3>& sa_x_end, 
        Buffer<uint>& broadphase_count, 
        Buffer<uint>& broad_phase_list, 
        const float thickness);
    void broad_phase_query_from_edges(Stream& stream, 
        const Buffer<float3>& sa_x_begin, 
        const Buffer<float3>& sa_x_end, 
        const Buffer<uint2>& sa_edges, 
        Buffer<uint>& broadphase_count, 
        Buffer<uint>& broad_phase_list, 
        const float thickness);
    
private:
    // void reduce_vert_tree_global_aabb();
    // void reduce_face_tree_global_aabb();
    // LbvhData<luisa::compute::Buffer>& get_lbvh_data() { return lbvh_data; }

public:
    LbvhData<luisa::compute::Buffer>* lbvh_data;

private:

    // Compute Morton
    luisa::compute::Shader<1, luisa::compute::BufferView<float3>> fn_reduce_vert_tree_global_aabb;
    luisa::compute::Shader<1, luisa::compute::BufferView<float3>, luisa::compute::BufferView<uint2>> fn_reduce_edge_tree_global_aabb;
    luisa::compute::Shader<1, luisa::compute::BufferView<float3>, luisa::compute::BufferView<uint3>> fn_reduce_face_tree_global_aabb;
    luisa::compute::Shader<1> fn_reduce_aabb_2_pass;
    luisa::compute::Shader<1> fn_reset_tree;
    luisa::compute::Shader<1> fn_compute_mortons;

    // Construct
    luisa::compute::Shader<1> fn_apply_sorted;
    luisa::compute::Shader<1> fn_build_inner_nodes ;
    luisa::compute::Shader<1> fn_check_construction ;
    luisa::compute::Shader<1> fn_set_escape_index;
    luisa::compute::Shader<1> fn_set_left_and_escape;

    // Refit
    luisa::compute::Shader<1, luisa::compute::BufferView<float3>, luisa::compute::BufferView<float3>, float> fn_update_vert_tree_leave_aabb;
    luisa::compute::Shader<1, luisa::compute::BufferView<float3>, luisa::compute::BufferView<float3>, luisa::compute::BufferView<uint2>, float>  fn_update_edge_tree_leave_aabb ;
    luisa::compute::Shader<1, luisa::compute::BufferView<float3>, luisa::compute::BufferView<float3>, luisa::compute::BufferView<uint3>, float>  fn_update_face_tree_leave_aabb ;
    luisa::compute::Shader<1> fn_clear_apply_flag ;
    luisa::compute::Shader<1> fn_refit_tree_aabb; // Invalid!!!!

    // Query
    luisa::compute::Shader<1, luisa::compute::BufferView<uint>> fn_reset_collision_count;
    luisa::compute::Shader<1, luisa::compute::BufferView<float3>, luisa::compute::BufferView<float3>,
                       luisa::compute::BufferView<uint>,
                       luisa::compute::BufferView<uint>, float> fn_query_from_verts;
    luisa::compute::Shader<1, luisa::compute::BufferView<float3>, luisa::compute::BufferView<float3>, luisa::compute::BufferView<uint2>,
                       luisa::compute::BufferView<uint>,
                       luisa::compute::BufferView<uint>, float> fn_query_from_edges;


};
    
};