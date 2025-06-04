#pragma once

#include "SimulationCore/simulation_type.h"
#include "Utils/buffer_allocator.h"
#include <vector>
#include <string>
#include <luisa/luisa-compute.h>
// #include <glm/glm.hpp>


namespace lcsv 
{

template<template<typename...> typename BufferType>
struct SimulationData : SimulationType
{
    // template<typename T>
    // using BufferType = Buffer<T>;
    BufferType<float3> sa_x_tilde;
    BufferType<float3> sa_x;
    BufferType<float3> sa_v;
    BufferType<float3> sa_v_step_start;
    BufferType<float3> sa_x_step_start;
    BufferType<float3> sa_x_iter_start;

    // Merged constraints
    BufferType<uint2> sa_merged_edges; 
    BufferType<float> sa_merged_edges_rest_length;

    BufferType<uint4> sa_merged_bending_edges; 
    BufferType<float> sa_merged_bending_edges_angle;
    BufferType<float4x4> sa_merged_bending_edges_Q;

    // Coloring
    // Spring constraint
    uint num_clusters_springs = 0;
    BufferType<uint> sa_clusterd_springs; 
    BufferType<uint> sa_prefix_merged_springs;
    BufferType<float> sa_lambda_stretch_mass_spring;

    // Bending constraint
    uint num_clusters_bending_edges = 0;
    BufferType<uint> sa_clusterd_bending_edges; 
    BufferType<uint> sa_prefix_merged_bending_edges; 
    BufferType<float> sa_lambda_bending;

    // Hessian non-conflict set
    uint num_clusters_hessian_pairs = 0;
    BufferType<uint2> sa_hessian_pairs; // Constaints the needed hessian pair 
    BufferType<uint> sa_clusterd_hessian_pairs;
    BufferType<uint> sa_hessian_slot_per_edge; 
    BufferType<uint2> sa_merged_hessian_pairs; // TODO
    BufferType<uint> sa_prefix_merged_hessian_pairs; // TODO
    BufferType<uint> sa_merged_hessian_slot_per_edge; // TODO
    // BufferType<uint> sa_clusterd_hessian_slot_per_dehedral_angle; 
    // BufferType<uint> sa_clusterd_hessian_slot_per_triangle; 

    // VBD
    uint num_clusters_per_vertex_bending = 0; 
    BufferType<uint> prefix_per_vertex_bending; 
    BufferType<uint> clusterd_per_vertex_bending; 
    BufferType<uint> per_vertex_bending_cluster_id; // ubyte
    
    BufferType<float> sa_Hf; 
    BufferType<float4x3> sa_Hf1; 

    // PCG
    BufferType<float3> sa_cgX;
    BufferType<float3> sa_cgB;
    BufferType<float3x3> sa_cgA_diag;
    BufferType<float3x3> sa_cgA_offdiag; // Row-major for simplier SpMV
 
    BufferType<float3x3> sa_cgMinv;
    BufferType<float3> sa_cgP;
    BufferType<float3> sa_cgQ;
    BufferType<float3> sa_cgR;
    BufferType<float3> sa_cgZ;
    BufferType<float> sa_block_result;
    BufferType<float> sa_convergence;
};

} // namespace lcsv 




namespace lcsv 
{

struct CollisionPairVV
{
    lcsv::uint2 indices; // vid1:1, vid2:1
    lcsv::float4 vec1; // normal:3, stiff:1
};
struct CollisionPairVE
{
    lcsv::uint2 edge; 
    uint vid;
    float bary;
    lcsv::float4 vec1; // normal:3, stiff 1
};
struct CollisionPairVF
{
    lcsv::uint4 indices; // vid:1, face:3
    lcsv::float4 vec1; // normal:3, stiff:1
    lcsv::float3 bary; // bary
};
struct CollisionPairEE
{
    lcsv::uint4 indices;
    lcsv::float4 vec1; // normal:3, stiff 1
    lcsv::float4 vec2; // 
};

// enum CollisionListType
// {
//     CollisionListTypeVV,
//     CollisionListTypeVF,
//     CollisionListTypeEE,
//     CollisionListTypeEF,
// };

}

LUISA_STRUCT(lcsv::CollisionPairVV, indices, vec1) {};
LUISA_STRUCT(lcsv::CollisionPairVE, edge, vid, bary, vec1) {};
LUISA_STRUCT(lcsv::CollisionPairVF, indices, vec1, bary) {};
LUISA_STRUCT(lcsv::CollisionPairEE, indices, vec1, vec2) {};


namespace lcsv 
{
namespace CollisionPair
{
    
    template<typename T> auto get_indices(const T& pair) { return pair.indices; }
    
    template<typename T> auto get_vv_vid1(const T& pair)  { return pair.indices[0]; }
    template<typename T> auto get_vv_vid2(const T& pair)  { return pair.indices[1]; }
    template<typename T> auto get_ve_vid(const T& pair)  { return pair.vid; }
    template<typename T> auto get_ve_edge(const T& pair) { return pair.edge; }
    template<typename T> auto get_vf_vid(const T& pair)  { return pair.indices[0]; }
    template<typename T> auto get_vf_face(const T& pair)  { return pair.indices.yzw(); }
    template<typename T> auto get_ee_edge1(const T& pair)  { return pair.indices.xy(); }
    template<typename T> auto get_ee_edge2(const T& pair)  { return pair.indices.zw(); }

    template<typename T> auto get_stiff(const T& pair) { return pair.vec1[0]; }
    template<typename T> auto get_direction(const T& pair) { return pair.vec1.yzw(); }

    // inline auto get_vv_bary(const CollisionPairVV& pair) { return makeFloat2(1.0f, 1.0f); }
    inline auto get_ve_edge_bary (const CollisionPairVE& pair) { return makeFloat2(pair.bary, 1.0f - pair.bary); }
    inline auto get_vf_face_bary (const CollisionPairVF& pair) { return pair.bary; }
    inline auto get_ee_edge1_bary(const CollisionPairEE& pair) { return pair.vec2.xy(); }
    inline auto get_ee_edge2_bary(const CollisionPairEE& pair) { return pair.vec2.zw(); }
    inline auto get_ve_edge_bary (const Var<CollisionPairVE>& pair) { return makeFloat2(pair.bary, 1.0f - pair.bary); }
    inline auto get_vf_face_bary (const Var<CollisionPairVF>& pair) { return pair.bary; }
    inline auto get_ee_edge1_bary(const Var<CollisionPairEE>& pair) { return pair.vec2.xy(); }
    inline auto get_ee_edge2_bary(const Var<CollisionPairEE>& pair) { return pair.vec2.zw(); }

} // namespace CollisionPair
} // namespace lcsv 


namespace lcsv {


template<template<typename...> typename BufferType>
struct CollisionData : SimulationType
{
    BufferType<uint> broad_phase_collision_count; // 0: VV, 1: VE, 2: VF, 3: EE
    BufferType<uint> narrow_phase_collision_count; // 0: VV, 1: VE, 2: VF, 3: EE

    BufferType<uint> broad_phase_list_vf;
    BufferType<uint> broad_phase_list_ee;
    BufferType<float> toi_per_vert;
    BufferType<float> contact_energy;

    BufferType<CollisionPairVV> narrow_phase_list_vv; // 0
    BufferType<CollisionPairVE> narrow_phase_list_ve; // 1
    BufferType<CollisionPairVF> narrow_phase_list_vf; // 2
    BufferType<CollisionPairEE> narrow_phase_list_ee; // 3
    // BufferType<uint> narrow_phase_indices_ef; 

    BufferType<uint> per_vert_num_broad_phase_vf; 
    BufferType<uint> per_vert_num_broad_phase_ee; 

    BufferType<uint> per_vert_num_narrow_phase_vv; 
    BufferType<uint> per_vert_num_narrow_phase_ve; 
    BufferType<uint> per_vert_num_narrow_phase_vf; 
    BufferType<uint> per_vert_num_narrow_phase_ee; 
    BufferType<uint> per_vert_prefix_narrow_phase_vv; 
    BufferType<uint> per_vert_prefix_narrow_phase_ve; 
    BufferType<uint> per_vert_prefix_narrow_phase_vf; 
    BufferType<uint> per_vert_prefix_narrow_phase_ee; 
    luisa::compute::IndirectDispatchBuffer collision_indirect_cmd_buffer_broad_phase; 
    luisa::compute::IndirectDispatchBuffer collision_indirect_cmd_buffer_narrow_phase; 


    constexpr uint get_vv_count_offset() { return 0; }
    constexpr uint get_ve_count_offset() { return 1; }
    constexpr uint get_vf_count_offset() { return 2; }
    constexpr uint get_ee_count_offset() { return 3; }

    // template<template<typename...> typename BufferType>
    inline void resize_collision_data(
            luisa::compute::Device& device, 
            const uint num_verts, const uint num_faces, const uint num_edges)
    {
        // const uint num_verts = mesh_data->num_verts;
        // const uint num_edges = mesh_data->num_edges;
        
        const uint per_element_count_BP = 128;
        const uint per_element_count_NP = 64;
        
        lcsv::Initializer::resize_buffer(device, this->broad_phase_collision_count, 4); 
        lcsv::Initializer::resize_buffer(device, this->narrow_phase_collision_count, 4); 
        lcsv::Initializer::resize_buffer(device, this->contact_energy, 4); 
        lcsv::Initializer::resize_buffer(device, this->toi_per_vert, num_verts); 
        lcsv::Initializer::resize_buffer(device, this->broad_phase_list_vf, per_element_count_BP * num_verts); 
        lcsv::Initializer::resize_buffer(device, this->broad_phase_list_ee, per_element_count_BP * num_edges); 
        lcsv::Initializer::resize_buffer(device, this->narrow_phase_list_vv, per_element_count_NP * num_verts); 
        lcsv::Initializer::resize_buffer(device, this->narrow_phase_list_ve, per_element_count_NP * num_verts); 
        lcsv::Initializer::resize_buffer(device, this->narrow_phase_list_vf, per_element_count_NP * num_verts); 
        lcsv::Initializer::resize_buffer(device, this->narrow_phase_list_ee, per_element_count_NP * num_edges); 
        lcsv::Initializer::resize_buffer(device, this->per_vert_num_broad_phase_vf, num_verts); 
        lcsv::Initializer::resize_buffer(device, this->per_vert_num_broad_phase_ee, num_verts); 
        lcsv::Initializer::resize_buffer(device, this->per_vert_num_narrow_phase_vv, num_verts); 
        lcsv::Initializer::resize_buffer(device, this->per_vert_num_narrow_phase_ve, num_verts); 
        lcsv::Initializer::resize_buffer(device, this->per_vert_num_narrow_phase_vf, num_verts); 
        lcsv::Initializer::resize_buffer(device, this->per_vert_num_narrow_phase_ee, num_verts); 
        lcsv::Initializer::resize_buffer(device, this->per_vert_prefix_narrow_phase_vv, num_verts); 
        lcsv::Initializer::resize_buffer(device, this->per_vert_prefix_narrow_phase_ve, num_verts); 
        lcsv::Initializer::resize_buffer(device, this->per_vert_prefix_narrow_phase_vf, num_verts); 
        lcsv::Initializer::resize_buffer(device, this->per_vert_prefix_narrow_phase_ee, num_verts); 
        this->collision_indirect_cmd_buffer_broad_phase = device.create_indirect_dispatch_buffer(2); 
        this->collision_indirect_cmd_buffer_narrow_phase = device.create_indirect_dispatch_buffer(4); 

        const uint collision_pair_bytes = 
            sizeof(uint) * this->broad_phase_list_vf.size() +
            sizeof(uint) * this->broad_phase_list_ee.size() +
            sizeof(CollisionPairVV) * this->narrow_phase_list_vv.size() +
            sizeof(CollisionPairVE) * this->narrow_phase_list_ve.size() +
            sizeof(CollisionPairVF) * this->narrow_phase_list_vf.size() +
            sizeof(CollisionPairEE) * this->narrow_phase_list_ee.size()
        ;
        
        luisa::log_info("Allocated collision buffer size {} MB", collision_pair_bytes / (1024 * 1024));
        if (float(collision_pair_bytes) / (1024 * 1024 * 1024) > 1.0f) luisa::log_info("Allocated buffer size for collision pair = {} GB", collision_pair_bytes / (1024 * 1024 * 1024));
    }
};

// template<>
// struct CollisionDataCCD<luisa::compute::Buffer>
// {
//     void print() {}
// };

}



/*
struct BaseSimulationData
{

using uint = unsigned int;
using Float3 = luisa::float3;
using Int2 = luisa::uint2;
using Int3 = luisa::uint3;
using Int4 = luisa::uint4;
using uchar = luisa::uchar;
using Float3x3 = luisa::float3x3;
using Float4x4 = luisa::float4x4;



public:
    bool simulate_cloth = false;
    std::vector<float> edges_rest_state_length;
    std::vector<float> bending_edges_rest_angle;
    std::vector<Float4x4> bending_edges_Q;

public:
    uint num_verts_cloth;
    bool simulate_tet = false;
    std::vector<float> rest_volumn;
    std::vector<Float3x3> Dm;
    std::vector<Float3x3> inv_Dm;

public:
    std::vector< std::vector<uint> > cloth_vert_adj_verts;
    std::vector< std::vector<uint> > cloth_vert_adj_verts_with_bending;
    std::vector< std::vector<uint> > cloth_vert_adj_faces;
    std::vector< std::vector<uint> > cloth_vert_adj_edges;
    std::vector< std::vector<uint> > cloth_vert_adj_bending_edges;

    std::vector< std::vector<uint> > tet_vert_adj_verts;
    std::vector< std::vector<uint> > tet_vert_adj_faces;
    std::vector< std::vector<uint> > tet_vert_adj_tets;

public:
    uint num_verts_total;
    uint num_edges_total;
    uint num_faces_total;

public:
    std::vector<Float3> x_frame_start;
    std::vector<Float3> v_frame_start;
    std::vector<Float3> x_frame_saved;
    std::vector<Float3> v_frame_saved;
    std::vector<Float3> x_frame_end;
    std::vector<Float3> v_frame_end;

    std::vector<Int3> rendering_triangles;

};

struct SimulationData
{

using uint = unsigned int;
using Float3 = luisa::float3;
using Int2 = luisa::uint2;
using Int3 = luisa::uint3;
using Int4 = luisa::uint4;
using uchar = luisa::uchar;
using Float3x3 = luisa::float3x3;
using Float4x4 = luisa::float4x4;

template<typename T>
using Buffer = luisa::compute::Buffer<T>;

public:
    Buffer<Float3> sa_x_start; // For calculating velocity
    Buffer<Float3> sa_v_start;
    Buffer<Float3> sa_x;
    Buffer<Float3> sa_v;

public:
    Buffer<Float3> sa_x_tilde;
    Buffer<Float3> sa_x_prev_1;
    Buffer<Float3> sa_x_prev_2;
    Buffer<Float3> sa_x_jacobi;
    Buffer<Float3> sa_dx;
public:
public:
    void assemble_from_scene()
    {

    }
    void write_to_scene()
    {

    }
};

*/