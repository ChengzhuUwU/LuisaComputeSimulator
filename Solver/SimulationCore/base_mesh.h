#pragma once

#include "SimulationCore/simulation_type.h"
#include "luisa/core/basic_types.h"
#include <vector>
#include <string>
#include <luisa/luisa-compute.h>
// #include <glm/glm.hpp>

namespace lcs
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
    std::vector<luisa::float3> rest_x;
    std::vector<luisa::float3> rest_v;
    std::vector<luisa::float2> rest_uv;

    std::vector<luisa::uint3> faces;
    std::vector<luisa::uint2> edges;
    std::vector<luisa::uint4> bending_edges;

    std::vector<float> vert_mass;
    std::vector<float> vert_mass_inv;
    std::vector<uchar> is_fixed;
};

template<template<typename...> typename BufferType>
struct MeshData : SimulationType
{
    uint num_meshes = 0;
    uint num_verts = 0;
    uint num_faces = 0;
    uint num_edges = 0;
    
    uint num_dihedral_edges = 0;
    uint num_tets = 0;

    // Input 
    BufferType<float3> sa_rest_x;
    BufferType<float3> sa_rest_v;
    BufferType<float3> sa_model_x;

    BufferType<uint3> sa_faces;
    BufferType<uint2> sa_edges;
    BufferType<uint4> sa_dihedral_edges;

    // Mesh attrubution
    BufferType<float> sa_vert_mass;
    BufferType<float> sa_vert_mass_inv;
    BufferType<uint> sa_is_fixed; // TODO: uchar
    BufferType<uint> sa_vert_mesh_id;
    BufferType<uint> sa_vert_mesh_type;
    
    BufferType<float> sa_rest_vert_area;
    BufferType<float> sa_rest_edge_area;
    BufferType<float> sa_rest_face_area;

    // Affine
    BufferType<float3> sa_rest_translate;
    BufferType<float3> sa_rest_scale;
    BufferType<float3> sa_rest_rotation;

    // Adjacent
    BufferType<uint> sa_vert_adj_verts_csr; 
    BufferType<uint> sa_vert_adj_faces_csr; 
    BufferType<uint> sa_vert_adj_edges_csr; 
    BufferType<uint> sa_vert_adj_dihedral_edges_csr; 
    BufferType<uint2> edge_adj_faces;
    BufferType<uint3> face_adj_edges;
    BufferType<uint3> face_adj_faces;

    // Other

    // Host only
    std::vector<float3> sa_x_frame_saved;
    std::vector<float3> sa_v_frame_saved;
    std::vector<float3> sa_x_frame_outer;
    std::vector<float3> sa_x_frame_outer_next;
    std::vector<float3> sa_v_frame_outer;

    std::vector<uint> prefix_num_verts;
    std::vector<uint> prefix_num_faces;
    std::vector<uint> prefix_num_edges;
    std::vector<uint> prefix_num_dihedral_edges;

    std::vector< std::vector<uint> > vert_adj_verts;
    std::vector< std::vector<uint> > vert_adj_faces;
    std::vector< std::vector<uint> > vert_adj_edges;
    std::vector< std::vector<uint> > vert_adj_dihedral_edges;
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


} // namespace lcs