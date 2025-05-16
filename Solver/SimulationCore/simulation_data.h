#pragma once

#include "SimulationCore/simulation_type.h"
#include <vector>
#include <string>
#include <luisa/luisa-compute.h>
#include <glm/glm.hpp>

namespace lcsv 
{

template<template<typename...> typename BufferType>
struct XpbdData : SimulationType
{
    // template<typename T>
    // using BufferType = Buffer<T>;
    BufferType<float3> sa_x_tilde;
    BufferType<float3> sa_x;
    BufferType<float3> sa_v;
    BufferType<float3> sa_v_start;
    BufferType<float3> sa_x_start; // For calculating velocity

    BufferType<uint2> sa_merged_edges; 
    BufferType<float> sa_merged_edges_rest_length;

    BufferType<uint4> sa_merged_bending_edges; 
    BufferType<float> sa_merged_bending_edges_angle;
    BufferType<float4x4> sa_merged_bending_edges_Q;

    uint num_clusters_stretch_mass_spring = 0;
    BufferType<uint> clusterd_constraint_stretch_mass_spring; 
    BufferType<uint> prefix_stretch_mass_spring;
    BufferType<float> sa_lambda_stretch_mass_spring;

    uint num_clusters_bending = 0;
    BufferType<uint> clusterd_constraint_bending; 
    BufferType<uint> prefix_bending; 
    BufferType<float> sa_lambda_bending;

    // VBD
    uint num_clusters_per_vertex_bending = 0; 
    BufferType<uint> prefix_per_vertex_bending; 
    BufferType<uint> clusterd_per_vertex_bending; 
    BufferType<uint> per_vertex_bending_cluster_id; // ubyte
    
    BufferType<float> sa_Hf; 
};


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