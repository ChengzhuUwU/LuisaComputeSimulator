#pragma once

#include "SimulationCore/simulation_type.h"
#include <vector>
#include <string>
#include <luisa/luisa-compute.h>
#include <glm/glm.hpp>

namespace lcsv
{

struct BaseClothMeshData
{    
    using uint = unsigned int;
    using uchar = unsigned char;

public:
    std::string mesh_name;
    uint num_verts = 0;
    uint num_faces = 0;
    uint num_edges = 0;
    uint num_bending_edges = 0;

public:
    std::vector<glm::vec<3, float>> rest_x;
    std::vector<glm::vec<3, float>> rest_v;
    std::vector<glm::vec<2, float>> rest_uv;

    std::vector<glm::vec<3, uint>> faces;
    std::vector<glm::vec<2, uint>> edges;
    std::vector<glm::vec<4, uint>> bending_edges;

    std::vector<float> vert_mass;
    std::vector<float> vert_mass_inv;
    std::vector<uchar> is_fixed;
};

template<template<typename...> typename BufferType>
struct MeshData : SimulationType
{
    uint num_verts;
    uint num_faces;
    uint num_edges;
    uint num_bending_edges;

    BufferType<float3> sa_rest_x;
    BufferType<float3> sa_rest_v;

    BufferType<uint3> sa_faces;
    BufferType<uint2> sa_edges;
    BufferType<uint4> sa_bending_edges;

    BufferType<float> sa_vert_mass;
    BufferType<float> sa_vert_mass_inv;
    BufferType<uint> sa_is_fixed; // TODO: uchar

    BufferType<float> sa_edges_rest_state_length;
    BufferType<float> sa_bending_edges_rest_angle;
    BufferType<float4x4> sa_bending_edges_Q;

    BufferType<uint> sa_vert_adj_verts_csr; 
    BufferType<uint> sa_vert_adj_verts_with_bending_csr; 
    BufferType<uint> sa_vert_adj_faces_csr; 
    BufferType<uint> sa_vert_adj_edges_csr; 
    BufferType<uint> sa_vert_adj_bending_edges_csr; 

    BufferType<float> sa_system_energy;

    // Host only
    std::vector<float3> sa_x_frame_start;
    std::vector<float3> sa_v_frame_start;
    std::vector<float3> sa_x_frame_saved;
    std::vector<float3> sa_v_frame_saved;
    std::vector<float3> sa_x_frame_end;
    std::vector<float3> sa_v_frame_end;

    std::vector< std::vector<uint> > vert_adj_verts;
    std::vector< std::vector<uint> > vert_adj_verts_with_bending;
    std::vector< std::vector<uint> > vert_adj_faces;
    std::vector< std::vector<uint> > vert_adj_edges;
    std::vector< std::vector<uint> > vert_adj_bending_edges;
};




/*


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