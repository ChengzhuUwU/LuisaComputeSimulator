#pragma once

#include "SimulationCore/simulation_type.h"
#include <vector>
#include <string>
#include <luisa/luisa-compute.h>
// #include <glm/glm.hpp>

namespace lcsv
{

template<template<typename...> typename BasicBuffer>
struct BasicMeshData : SimulationType
{
    uint num_verts;
    uint num_faces;
    uint num_edges;
    uint num_bending_edges;

    BasicBuffer<float3> sa_rest_x;
    BasicBuffer<float3> sa_rest_v;

    BasicBuffer<float3> sa_x_frame_start;
    BasicBuffer<float3> sa_v_frame_start;
    BasicBuffer<float3> sa_x_frame_saved;
    BasicBuffer<float3> sa_v_frame_saved;
    BasicBuffer<float3> sa_x_frame_end;
    BasicBuffer<float3> sa_v_frame_end;

    BasicBuffer<uint3> sa_faces;
    BasicBuffer<uint2> sa_edges;
    BasicBuffer<uint4> sa_bending_edges;

    BasicBuffer<float> sa_vert_mass;
    BasicBuffer<float> sa_vert_mass_inv;
    BasicBuffer<uint> sa_is_fixed; // ubyte

    BasicBuffer<float> sa_edges_rest_state_length;
    BasicBuffer<float> sa_bending_edges_rest_angle;
    BasicBuffer<float4x4> sa_bending_edges_Q;

    BasicBuffer<uint> sa_vert_adj_verts; std::vector< std::vector<uint> > vert_adj_verts;
    BasicBuffer<uint> sa_vert_adj_verts_with_bending; std::vector< std::vector<uint> > vert_adj_verts_with_bending;
    BasicBuffer<uint> sa_vert_adj_faces; std::vector< std::vector<uint> > vert_adj_faces;
    BasicBuffer<uint> sa_vert_adj_edges; std::vector< std::vector<uint> > vert_adj_edges;
    BasicBuffer<uint> sa_vert_adj_bending_edges; std::vector< std::vector<uint> > vert_adj_bending_edges;

    BasicBuffer<float> sa_system_energy;
};

template<template<typename...> typename BasicBuffer>
struct XpbdData : SimulationType
{
    // template<typename T>
    // using BasicBuffer = Buffer<T>;
    BasicBuffer<float3> sa_x_tilde;
    BasicBuffer<float3> sa_x;
    BasicBuffer<float3> sa_v;
    BasicBuffer<float3> sa_v_start;
    BasicBuffer<float3> sa_x_start; // For calculating velocity

    BasicBuffer<uint2> sa_merged_edges; 
    BasicBuffer<float> sa_merged_edges_rest_length;

    BasicBuffer<uint4> sa_merged_bending_edges; 
    BasicBuffer<float> sa_merged_bending_edges_angle;
    BasicBuffer<float4x4> sa_merged_bending_edges_Q;

    uint num_clusters_stretch_mass_spring = 0;
    BasicBuffer<uint> clusterd_constraint_stretch_mass_spring; 
    BasicBuffer<uint> prefix_stretch_mass_spring;
    BasicBuffer<float> sa_lambda_stretch_mass_spring;

    uint num_clusters_bending = 0;
    BasicBuffer<uint> clusterd_constraint_bending; 
    BasicBuffer<uint> prefix_bending; 
    BasicBuffer<float> sa_lambda_bending;

    // VBD
    uint num_clusters_per_vertex_bending = 0; 
    BasicBuffer<uint> prefix_per_vertex_bending; 
    BasicBuffer<uint> clusterd_per_vertex_bending; 
    BasicBuffer<uint> per_vertex_bending_cluster_id; // ubyte
    // BasicBuffer<float4x3> sa_Hf; 
};


/*
struct BaseClothMeshData
{    
    using uint = unsigned int;
    using Float2 = luisa::float2;
    using Float3 = luisa::float3;
    using Int2 = luisa::uint2;
    using Int3 = luisa::uint3;
    using Int4 = luisa::uint4;
    using uchar = luisa::uchar;
    using Float4x4 = luisa::float4x4;

public:
    std::string name;
    uint num_verts = 0;
    uint num_faces = 0;
    uint num_edges = 0;
    uint num_bending_edges = 0;

public:
    std::vector<Float3> rest_x;
    std::vector<Float3> rest_v;
    std::vector<Float2> rest_uv;

    std::vector<Int3> faces;
    std::vector<Int2> edges;
    std::vector<Int4> bending_edges;

    std::vector<float> vert_mass;
    std::vector<float> vert_mass_inv;
    std::vector<uchar> is_fixed;
};

struct BaseTetMeshData
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
    std::string name;
    uint num_verts = 0;
    uint num_tets = 0;
    // uint num_edges;
    uint num_surface_verts = 0;
    uint num_surface_faces = 0;
    uint num_surface_tets = 0;
    uint num_surface_edges = 0;

public:
    std::vector<Float3> rest_x;
    std::vector<Float3> rest_v;

    std::vector<Int3> surface_faces;
    std::vector<Int2> surface_edges;
    std::vector<Int4> tets;

    std::vector<float> vert_mass;
    std::vector<float> vert_mass_inv;
    std::vector<uchar> is_fixed;


    std::vector<uint> vert_adj_verts_csr; 
    std::vector<uint> vert_adj_faces_csr; 
    std::vector<uint> vert_adj_tets_csr; 

    std::vector< std::vector<uint> > vert_adj_verts;
    std::vector< std::vector<uint> > vert_adj_faces;
    std::vector< std::vector<uint> > vert_adj_tets;
};

struct SceneObject
{
public:
    std::vector<BaseClothMeshData> cloth_data;
    std::vector<BaseTetMeshData> tet_data;

    bool is_cloth_valid() { return !cloth_data.empty(); }
    bool is_tet_valid() { return !tet_data.empty(); }

public:
    std::vector<uint> prefix_verts_cloth;
    std::vector<uint> prefix_verts_tet;

};
*/


} // namespace lcsv