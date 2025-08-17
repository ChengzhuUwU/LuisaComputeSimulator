#pragma once

#include "Core/float_nxn.h"
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

    // Energy
    BufferType<uint2> sa_stretch_springs;
    BufferType<float> sa_stretch_spring_rest_state_length;
    BufferType<uint3> sa_stretch_faces;
    BufferType<float2x2> sa_stretch_faces_Dm_inv;
    BufferType<uint4> sa_bending_edges;
    BufferType<float> sa_bending_edges_rest_angle;
    BufferType<float4x4> sa_bending_edges_Q;

    // Merged constraints
    BufferType<uint2> sa_merged_stretch_springs; 
    BufferType<float> sa_merged_stretch_spring_rest_length;

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
    uint num_clusters_per_vertex_with_material_constraints = 0; 
    BufferType<uint> prefix_per_vertex_with_material_constraints; 
    BufferType<uint> clusterd_per_vertex_with_material_constraints; 
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

// struct CollisionPairVV
// {
//     uint2 indices; // vid1:1, vid2:1
//     float4 vec1; // normal:3, stiff:1
//     float3 gradient[2];
//     float3x3 hessian[3];
// };
// struct CollisionPairVE
// {
//     uint2 edge; 
//     uint vid;
//     float bary;
//     float4 vec1; // normal:3, stiff 1
//     float3 gradient[3];
//     float3x3 hessian[6];
// };
// struct CollisionPairVF
// {
//     uint4 indices; // vid:1, face:3
//     float4 vec1; // normal:3, stiff:1
//     float2 bary; // bary
//     float2 vec2;
//     float3 gradient[4];
//     float3x3 hessian[10];
// };
// struct CollisionPairEE
// {
//     uint4 indices;
//     float4 vec1; // normal:3, stiff 1
//     float2 bary; // 
//     float2 vec2;
//     float3 gradient[4];
//     float3x3 hessian[10];
// };
struct CollisionPairVV
{
    uint2 indices; // vid1:1, vid2:1
    float4 vec1; // normal:3, stiff:1
};
struct CollisionPairVE
{
    uint2 edge; 
    uint vid;
    float bary;
    float4 vec1; // normal:3, stiff 1
};
struct CollisionPairVF
{
    uint4 indices; // vid:1, face:3
    float4 vec1; // normal:3, stiff:1
    float2 bary; // bary
    float2 vec2;
};
struct CollisionPairEE
{
    uint4 indices;
    float4 vec1; // normal:3, stiff 1
    float2 bary; // 
    float2 vec2;
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
LUISA_STRUCT(lcsv::CollisionPairVF, indices, vec1, bary, vec2) {};
LUISA_STRUCT(lcsv::CollisionPairEE, indices, vec1, bary, vec2) {};

// LUISA_STRUCT(lcsv::CollisionPairVV, indices, vec1, gradient, hessian) {};
// LUISA_STRUCT(lcsv::CollisionPairVE, edge, vid, bary, vec1, gradient, hessian) {};
// LUISA_STRUCT(lcsv::CollisionPairVF, indices, vec1, bary, vec2, gradient, hessian) {};
// LUISA_STRUCT(lcsv::CollisionPairEE, indices, vec1, bary, vec2, gradient, hessian) {};


namespace lcsv 
{
namespace CollisionPair
{
    
// template<typename T> auto get_indices(const T& pair) { return pair.indices; }
inline auto get_indices(const CollisionPairVV& pair) { return pair.indices; }
inline auto get_indices(const CollisionPairVE& pair) { return makeUint3(pair.vid, pair.edge[0], pair.edge[1]); }
inline auto get_indices(const CollisionPairVF& pair) { return pair.indices; }
inline auto get_indices(const CollisionPairEE& pair) { return pair.indices; }
inline auto get_indices(const Var<CollisionPairVV>& pair) { return pair.indices; }
inline auto get_indices(const Var<CollisionPairVE>& pair) { return makeUint3(pair.vid, pair.edge[0], pair.edge[1]); }
inline auto get_indices(const Var<CollisionPairVF>& pair) { return pair.indices; }
inline auto get_indices(const Var<CollisionPairEE>& pair) { return pair.indices; }

template<typename T> auto get_vv_vid1(const T& pair)  { return pair.indices[0]; }
template<typename T> auto get_vv_vid2(const T& pair)  { return pair.indices[1]; }
template<typename T> auto get_ve_vid(const T& pair)  { return pair.vid; }
template<typename T> auto get_ve_edge(const T& pair) { return pair.edge; }
template<typename T> auto get_vf_vid(const T& pair)  { return pair.indices[0]; }
template<typename T> auto get_vf_face(const T& pair)  { return pair.indices.yzw(); }
template<typename T> auto get_ee_edge1(const T& pair)  { return pair.indices.xy(); }
template<typename T> auto get_ee_edge2(const T& pair)  { return pair.indices.zw(); }

// template<typename T> auto get_stiff(const T& pair) { return pair.vec1[3]; }
template<typename T> auto get_direction(const T& pair) { return pair.vec1.xyz(); }
template<typename T> auto get_area(const T& pair) { return pair.vec1[3]; }

// inline auto get_vv_bary(const CollisionPairVV& pair) { return makeFloat2(1.0f, 1.0f); }
inline auto get_ve_edge_bary (const CollisionPairVE& pair) { return makeFloat2(pair.bary, 1.0f - pair.bary); }
inline auto get_vf_face_bary (const CollisionPairVF& pair) { return makeFloat3(pair.bary[0], pair.bary[1], 1.0f - pair.bary[0] - pair.bary[1]); }
inline auto get_ee_edge1_bary(const CollisionPairEE& pair) { return makeFloat2(pair.bary[0], 1.0f - pair.bary[0]); }
inline auto get_ee_edge2_bary(const CollisionPairEE& pair) { return makeFloat2(pair.bary[1], 1.0f - pair.bary[1]); }

inline auto get_ve_edge_bary (const Var<CollisionPairVE>& pair) { return makeFloat2(pair.bary, 1.0f - pair.bary); }
inline auto get_vf_face_bary (const Var<CollisionPairVF>& pair) { return makeFloat3(pair.bary[0], pair.bary[1], 1.0f - pair.bary[0] - pair.bary[1]); }
inline auto get_ee_edge1_bary(const Var<CollisionPairEE>& pair) { return makeFloat2(pair.bary[0], 1.0f - pair.bary[0]); }
inline auto get_ee_edge2_bary(const Var<CollisionPairEE>& pair) { return makeFloat2(pair.bary[1], 1.0f - pair.bary[1]); }

template<typename T> auto get_vf_weight(const T& pair)  
{ 
    return makeFloat4(
        1.0f, 
        -pair.bary[0], 
        -pair.bary[1], 
        pair.bary[0] + pair.bary[1] - 1.0f); 
}
template<typename T> auto get_ee_weight(const T& pair)  
{ 
    return makeFloat4(
        pair.bary[0], 
        1.0f - pair.bary[0], 
        -pair.bary[1], 
        pair.bary[1] - 1.0f); 
}
template<typename T, typename Vec3> void write_vf_weight(T& pair, const Vec3& face_bary)  
{ 
    pair.bary = makeFloat2(face_bary[0], face_bary[1]);
}
template<typename T, typename Vec4> void write_ee_weight(T& pair, const Vec4& edge_bary)  
{ 
    pair.bary = makeFloat2(edge_bary[0], edge_bary[2]);
}

template<typename T> auto get_vf_stiff(const T& pair)  { return pair.vec2; }
template<typename T> auto get_ee_stiff(const T& pair) { return pair.vec2; }

template<typename T> auto get_vf_k1(const T& pair) { return pair.vec2[0]; }
template<typename T> auto get_vf_k2(const T& pair) { return pair.vec2[1]; }
template<typename T> auto get_ee_k1(const T& pair) { return pair.vec2[0]; }
template<typename T> auto get_ee_k2(const T& pair) { return pair.vec2[1]; }

template<typename T, typename FloatType> void write_vf_stiff(T& pair, const FloatType& k1, const FloatType& k2)  
{ 
    pair.vec2 = makeFloat2(k1, k2);
}
template<typename T, typename FloatType> void write_ee_stiff(T& pair, const FloatType& k1, const FloatType& k2)  
{ 
    pair.vec2 = makeFloat2(k1, k2);
}

inline void write_upper_hessian(luisa::compute::ArrayFloat3x3<3>& hessian, Float6x6& H)
{
    //   0   2  
    //  t2   1  
    hessian[0] = H.mat[0][0];
    hessian[1] = H.mat[1][1];
    hessian[2] = H.mat[1][0];
}
inline void write_upper_hessian(float3x3 hessian[3], float6x6& H)
{
    //   0   2  
    //  t2   1  
    hessian[0] = H.mat[0][0];
    hessian[1] = H.mat[1][1];
    hessian[2] = H.mat[1][0];
}
inline void extract_upper_hessian(float3x3 hessian[3], float6x6& H)
{
    //   0   2  
    //  t2   1  
    H.mat[0][0] = hessian[0];
    H.mat[0][1] = transpose_mat(hessian[2]);
    H.mat[1][0] = hessian[2];
    H.mat[1][1] = hessian[1];
}

inline void write_upper_hessian(luisa::compute::ArrayFloat3x3<6>& hessian, Float9x9& H)
{
    //   0   3   5  
    //  t3   1   4  
    //  t5  t4   2    
    hessian[0] = H.mat[0][0];
    hessian[1] = H.mat[1][1];
    hessian[2] = H.mat[2][2];
    hessian[3] = H.mat[1][0];
    hessian[4] = H.mat[2][1];
    hessian[5] = H.mat[2][0];
}
inline void write_upper_hessian(float3x3 hessian[6], float9x9& H)
{
    //   0   3   5  
    //  t3   1   4  
    //  t5  t4   2    
    hessian[0] = H.mat[0][0];
    hessian[1] = H.mat[1][1];
    hessian[2] = H.mat[2][2];
    hessian[3] = H.mat[1][0];
    hessian[4] = H.mat[2][1];
    hessian[5] = H.mat[2][0];
}
inline void extract_upper_hessian(float3x3 hessian[6], float9x9& H)
{
    //   0   3   5  
    //  t3   1   4  
    //  t5  t4   2           
    H.mat[0][0] = hessian[0];
    H.mat[0][1] = transpose_mat(hessian[3]);
    H.mat[0][2] = transpose_mat(hessian[5]);
    H.mat[1][0] = hessian[3];
    H.mat[1][1] = hessian[1];
    H.mat[1][2] = transpose_mat(hessian[4]);
    H.mat[2][0] = hessian[5];
    H.mat[2][1] = hessian[4];
    H.mat[2][2] = hessian[2];
}

inline void write_upper_hessian(luisa::compute::ArrayFloat3x3<10>& hessian, Float12x12& H)
{
    //   0   4   7   9
    //  t4   1   5   8
    //  t7  t5   2   6
    //  t9  t8  t6   3
    hessian[0] = H.mat[0][0];
    hessian[1] = H.mat[1][1];
    hessian[2] = H.mat[2][2];
    hessian[3] = H.mat[3][3];
    hessian[4] = H.mat[1][0];
    hessian[5] = H.mat[2][1];
    hessian[6] = H.mat[3][2];
    hessian[7] = H.mat[2][0];
    hessian[8] = H.mat[3][1];
    hessian[9] = H.mat[3][0];
}
inline void write_upper_hessian(float3x3 hessian[10], float12x12& H)
{
    
    //   0   4   7   9
    //  t4   1   5   8
    //  t7  t5   2   6
    //  t9  t8  t6   3
    hessian[0] = H.mat[0][0];
    hessian[1] = H.mat[1][1];
    hessian[2] = H.mat[2][2];
    hessian[3] = H.mat[3][3];
    hessian[4] = H.mat[1][0];
    hessian[5] = H.mat[2][1];
    hessian[6] = H.mat[3][2];
    hessian[7] = H.mat[2][0];
    hessian[8] = H.mat[3][1];
    hessian[9] = H.mat[3][0];
}
inline void extract_upper_hessian(float3x3 hessian[10], float12x12& H)
{
    //   0   4   7   9
    //  t4   1   5   8
    //  t7  t5   2   6
    //  t9  t8  t6   3
    H.mat[0][0] = hessian[0];
    H.mat[0][1] = transpose_mat(hessian[4]);
    H.mat[0][2] = transpose_mat(hessian[7]);
    H.mat[0][3] = transpose_mat(hessian[9]);
    H.mat[1][0] = hessian[4];
    H.mat[1][1] = hessian[1];
    H.mat[1][2] = transpose_mat(hessian[5]);
    H.mat[1][3] = transpose_mat(hessian[8]);
    H.mat[2][0] = hessian[7];
    H.mat[2][1] = hessian[5];
    H.mat[2][2] = hessian[2];
    H.mat[2][3] = transpose_mat(hessian[3]);
    H.mat[3][0] = hessian[9];
    H.mat[3][1] = hessian[8];
    H.mat[3][2] = hessian[6];
    H.mat[3][3] = hessian[3];
}


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
        
        const uint per_element_count_BP = 256;
        const uint per_element_count_NP = 96;
        
        constexpr bool use_vv_ve = false;
        lcsv::Initializer::resize_buffer(device, this->broad_phase_collision_count, 4); 
        lcsv::Initializer::resize_buffer(device, this->narrow_phase_collision_count, 4); 
        lcsv::Initializer::resize_buffer(device, this->contact_energy, 4); 
        lcsv::Initializer::resize_buffer(device, this->toi_per_vert, num_verts); 
        lcsv::Initializer::resize_buffer(device, this->broad_phase_list_vf, per_element_count_BP * num_verts); 
        lcsv::Initializer::resize_buffer(device, this->broad_phase_list_ee, per_element_count_BP * num_edges); 
        // lcsv::Initializer::resize_buffer(device, this->narrow_phase_list_vv, per_element_count_NP * num_verts); 
        // lcsv::Initializer::resize_buffer(device, this->narrow_phase_list_ve, per_element_count_NP * num_verts); 
        if (use_vv_ve) lcsv::Initializer::resize_buffer(device, this->narrow_phase_list_vv, per_element_count_NP * num_verts); 
        if (use_vv_ve) lcsv::Initializer::resize_buffer(device, this->narrow_phase_list_ve, per_element_count_NP * num_verts); 
        if (!use_vv_ve) lcsv::Initializer::resize_buffer(device, this->narrow_phase_list_vv, 1); 
        if (!use_vv_ve) lcsv::Initializer::resize_buffer(device, this->narrow_phase_list_ve, 1); 
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
    std::vector< std::vector<uint> > cloth_vert_adj_verts_with_material_constraints;
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