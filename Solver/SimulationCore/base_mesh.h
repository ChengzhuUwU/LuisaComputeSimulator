#pragma once

#include "SimulationCore/simulation_type.h"
#include "luisa/core/basic_types.h"
#include <vector>
#include <string>
#include <luisa/luisa-compute.h>
// #include <glm/glm.hpp>

namespace lcs
{


namespace Animation
{
    struct PerVertexAnimation
    {
        uint                 vertex_id;
        std::array<float, 3> translation;
    };
    struct PerBodyAnimation
    {
        uint                 body_id;
        std::array<float, 3> translation;
        std::array<float, 4> rotation;

        void set_translation(const float x, const float y, const float z)
        {
            translation[0] = x;
            translation[1] = y;
            translation[2] = z;
        }
        void set_rotation(const float axis_x, const float axis_y, const float axis_z, const float angle_w)
        {
            rotation[0] = axis_x;
            rotation[1] = axis_y;
            rotation[2] = axis_z;
            rotation[3] = angle_w;
        }
        float4x4 to_transform_matrix() const
        {
            auto trans = luisa::translation(translation[0], translation[1], translation[2]);
            auto rot   = luisa::rotation(rotation[0], rotation[1], rotation[2], rotation[3]);
            // auto scale = identity();
            return trans * rot;
        }
        // std::array<float, 4> rotation;  // quaternion
        // std::array<float, 3> scale;
        // float4x4 to_transform_matrix() const
        // {
        //     float4x4 T = float4x4::eye(1.0f);
        //     T[3].xyz() = luisa::make_float3(translation[0], translation[1], translation[2]);
        //     float qw = rotation[3];
        //     float qx = rotation[0];
        //     float qy = rotation[1];
        //     float qz = rotation[2];
        //     float3x3 R;
        //     R[0][0] = 1 - 2 * (qy * qy + qz * qz);
        //     R[0][1] = 2 * (qx * qy - qz * qw);
        //     R[0][2] = 2 * (qx * qz + qy * qw);
        //     R[1][0] = 2 * (qx * qy + qz * qw);
        //     R[1][1] = 1 - 2 * (qx * qx + qz * qz);
        //     R[1][2] = 2 * (qy * qz - qx * qw);
        //     R[2][0] = 2 * (qx * qz - qy * qw);
        //     R[2][1] = 2 * (qy * qz + qx * qw);
        //     R[2][2] = 1 - 2 * (qx * qx + qy * qy);
        //     T[0].xyz() = R[0];
        //     T[1].xyz() = R[1];
        //     T[2].xyz() = R[2];
        //     return T;
        // }
    };
}  // namespace Animation


template <template <typename...> typename BufferType>
struct MeshData : SimulationType
{
    uint num_meshes = 0;
    uint num_verts  = 0;
    uint num_faces  = 0;
    uint num_edges  = 0;

    uint num_dihedral_edges = 0;
    uint num_tets           = 0;

    // Input
    BufferType<float3> sa_rest_x;
    BufferType<float3> sa_rest_v;
    BufferType<float3> sa_model_x;
    BufferType<float3> sa_scaled_model_x;  // TODO: Move to SimData

    BufferType<uint3> sa_faces;
    BufferType<uint2> sa_edges;
    BufferType<uint4> sa_dihedral_edges;
    BufferType<uint4> sa_tetrahedrons;

    // Mesh attrubution
    BufferType<float> sa_body_mass;
    BufferType<float> sa_vert_mass;
    BufferType<float> sa_vert_mass_inv;
    BufferType<uint>  sa_is_fixed;  // TODO: uchar
    BufferType<uint>  sa_vert_mesh_type;
    // BufferType<uint>  sa_vert_is_on_surface; // For collision detection on surface

    BufferType<uint> sa_vert_mesh_id;
    BufferType<uint> sa_face_mesh_id;
    BufferType<uint> sa_edge_mesh_id;
    BufferType<uint> sa_dihedral_edge_mesh_id;
    BufferType<uint> sa_tet_mesh_id;

    BufferType<float> sa_rest_body_area;
    BufferType<float> sa_rest_body_volume;

    BufferType<float> sa_rest_vert_area;
    BufferType<float> sa_rest_edge_area;
    BufferType<float> sa_rest_face_area;
    BufferType<float> sa_rest_tet_volume;
    BufferType<float> sa_rest_vert_volume;
    BufferType<float> sa_vert_thickness;  // For contact offset
    BufferType<float> sa_edge_thickness;
    BufferType<float> sa_face_thickness;

    // Affine
    BufferType<float3> sa_rest_translate;
    BufferType<float3> sa_rest_scale;
    BufferType<float3> sa_rest_rotation;

    // Adjacent
    BufferType<uint>  sa_vert_adj_verts_csr;
    BufferType<uint>  sa_vert_adj_faces_csr;
    BufferType<uint>  sa_vert_adj_edges_csr;
    BufferType<uint>  sa_vert_adj_dihedral_edges_csr;
    BufferType<uint>  sa_vert_adj_tets_csr;
    BufferType<uint2> edge_adj_faces;
    BufferType<uint3> face_adj_edges;
    BufferType<uint3> face_adj_faces;

    // Other

    // Host only
    // std::vector<float3> sa_x_frame_outer;
    // std::vector<float3> sa_v_frame_outer;

    std::vector<uint> prefix_num_verts;
    std::vector<uint> prefix_num_faces;
    std::vector<uint> prefix_num_edges;
    std::vector<uint> prefix_num_dihedral_edges;
    std::vector<uint> prefix_num_tets;

    std::vector<std::vector<uint>> vert_adj_verts;
    std::vector<std::vector<uint>> vert_adj_faces;
    std::vector<std::vector<uint>> vert_adj_edges;
    std::vector<std::vector<uint>> vert_adj_dihedral_edges;
    std::vector<std::vector<uint>> vert_adj_tets;

    std::vector<uint>              fixed_verts;
    std::vector<std::vector<uint>> fixed_verts_map;
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


}  // namespace lcs