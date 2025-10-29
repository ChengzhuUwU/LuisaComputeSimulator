#pragma once

#include "Core/float_nxn.h"
#include "Core/lc_to_eigen.h"
#include "Core/matrix_triplet.h"
#include "SimulationCore/simulation_type.h"
#include "Utils/buffer_allocator.h"
#include <vector>
#include <string>
#include <luisa/luisa-compute.h>
// #include <glm/glm.hpp>

namespace lcs
{
using ushort = uint16_t;
template <template <typename...> typename BufferType>
struct ColoredData : SimulationType
{
    // Merged constraints
    BufferType<uint2> sa_merged_stretch_springs;
    BufferType<float> sa_merged_stretch_spring_rest_length;

    BufferType<uint4>    sa_merged_bending_edges;
    BufferType<float>    sa_merged_bending_edges_angle;
    BufferType<float4x4> sa_merged_bending_edges_Q;

    // Coloring
    // Spring constraint
    uint              num_clusters_springs = 0;
    BufferType<uint>  sa_clusterd_springs;
    BufferType<uint>  sa_prefix_merged_springs;
    BufferType<float> sa_lambda_stretch_mass_spring;

    // Bending constraint
    uint              num_clusters_bending_edges = 0;
    BufferType<uint>  sa_clusterd_bending_edges;
    BufferType<uint>  sa_prefix_merged_bending_edges;
    BufferType<float> sa_lambda_bending;

    // VBD
    uint             num_clusters_per_vertex_with_material_constraints = 0;
    BufferType<uint> prefix_per_vertex_with_material_constraints;
    BufferType<uint> clusterd_per_vertex_with_material_constraints;
    BufferType<uint> per_vertex_bending_cluster_id;  // ubyte
};

template <template <typename...> typename BufferType>
struct AffineBodyData : SimulationType
{
    std::vector<EigenFloat12x12> sa_affine_bodies_mass_matrix_full;
    BufferType<uint>             sa_vert_affine_bodies_id;
    BufferType<uint>             sa_affine_bodies;
    BufferType<float>            sa_affine_bodies_volume;
    BufferType<float3x3>         sa_affine_bodies_mass_matrix_diag;
    BufferType<float3x3>         sa_affine_bodies_mass_matrix_compressed_offdiag;
    BufferType<float3>           sa_affine_bodies_rest_q;
    BufferType<float3>           sa_affine_bodies_rest_q_v;
    BufferType<float3>           sa_affine_bodies_gravity;
    BufferType<float3>           sa_affine_bodies_q;
    BufferType<float3>           sa_affine_bodies_q_v;
    BufferType<float3>           sa_affine_bodies_q_tilde;
    BufferType<float3>           sa_affine_bodies_q_iter_start;
    BufferType<float3>           sa_affine_bodies_q_step_start;

    BufferType<float3> sa_affine_bodies_q_outer;
    BufferType<float3> sa_affine_bodies_q_v_outer;
};

template <template <typename...> typename BufferType>
struct PcgData : SimulationType
{
    // PCG
    BufferType<float3>   sa_cgX;
    BufferType<float3>   sa_cgB;
    BufferType<float3x3> sa_cgA_diag;
    BufferType<float3x3> sa_cgMinv;
    BufferType<float3>   sa_cgP;
    BufferType<float3>   sa_cgQ;
    BufferType<float3>   sa_cgR;
    BufferType<float3>   sa_cgZ;
    BufferType<float>    sa_block_result;
    BufferType<float>    sa_convergence;
};

template <template <typename...> typename BufferType>
struct SimulationData : SimulationType
{
    // template<typename T>
    // using BufferType = Buffer<T>;
    BufferType<float3> sa_x_tilde;
    BufferType<float3> sa_x;
    BufferType<float3> sa_v;
    BufferType<float3> sa_x_step_start;
    BufferType<float3> sa_x_iter_start;

    BufferType<float3> sa_target_positions;

    // Energy
    uint              num_verts_soft    = 0;
    uint              num_verts_rigid   = 0;
    uint              num_affine_bodies = 0;
    uint              num_dof           = 0;  // Degree of freedom, actually is DOF / 3
    BufferType<uint>  sa_num_dof;
    BufferType<float> sa_system_energy;

    BufferType<uint2>    sa_stretch_springs;
    BufferType<float>    sa_stretch_spring_rest_state_length;
    BufferType<float>    sa_stretch_spring_stiffness;
    BufferType<ushort>   sa_stretch_springs_offsets_in_adjlist;
    BufferType<float3>   sa_stretch_springs_gradients;
    BufferType<float3x3> sa_stretch_springs_hessians;

    BufferType<uint3>    sa_stretch_faces;
    BufferType<float>    sa_stretch_faces_rest_area;
    BufferType<float2>   sa_stretch_faces_mu_lambda;  // scaled by thickness, thus only multiply by area
    BufferType<float2x2> sa_stretch_faces_Dm_inv;
    BufferType<ushort>   sa_stretch_faces_offsets_in_adjlist;
    BufferType<float3>   sa_stretch_faces_gradients;
    BufferType<float3x3> sa_stretch_faces_hessians;

    BufferType<uint4>    sa_bending_edges;
    BufferType<float>    sa_bending_edges_rest_angle;
    BufferType<float>    sa_bending_edges_stiffness;
    BufferType<float4x4> sa_bending_edges_Q;
    BufferType<float>    sa_bending_edges_rest_area;
    BufferType<ushort>   sa_bending_edges_offsets_in_adjlist;
    BufferType<float3>   sa_bending_edges_gradients;
    BufferType<float3x3> sa_bending_edges_hessians;

    std::vector<EigenFloat12x12> sa_affine_bodies_mass_matrix_full;
    BufferType<uint>             sa_vert_affine_bodies_id;
    BufferType<uint>             sa_affine_bodies_mesh_id;
    BufferType<uint>             sa_affine_bodies_is_fixed;
    BufferType<uint4>            sa_affine_bodies;
    BufferType<float>            sa_affine_bodies_kappa;
    BufferType<float>            sa_affine_bodies_volume;
    BufferType<float4x4>         sa_affine_bodies_mass_matrix;
    BufferType<float3>           sa_affine_bodies_rest_q;
    BufferType<float3>           sa_affine_bodies_rest_q_v;
    BufferType<float3>           sa_affine_bodies_gravity;
    BufferType<float3>           sa_affine_bodies_q;
    BufferType<float3>           sa_affine_bodies_q_v;
    BufferType<float3>           sa_affine_bodies_q_tilde;
    BufferType<float3>           sa_affine_bodies_q_iter_start;
    BufferType<float3>           sa_affine_bodies_q_step_start;
    BufferType<float3>           sa_affine_bodies_gradients;
    BufferType<float3x3>         sa_affine_bodies_hessians;
    BufferType<ushort>           sa_affine_bodies_offsets_in_adjlist;


    BufferType<float3> sa_affine_bodies_q_outer;
    BufferType<float3> sa_affine_bodies_q_v_outer;

    ColoredData<BufferType> colored_data;

    BufferType<float>    sa_Hf;
    BufferType<float4x3> sa_Hf1;

    // PCG
    BufferType<float3>           sa_cgX;
    BufferType<float3>           sa_cgB;
    BufferType<float3x3>         sa_cgA_diag;
    BufferType<MatrixTriplet3x3> sa_cgA_fixtopo_offdiag_triplet;
    BufferType<uint3>            sa_cgA_fixtopo_offdiag_triplet_info;

    BufferType<float3x3> sa_cgMinv;
    BufferType<float3>   sa_cgP;
    BufferType<float3>   sa_cgQ;
    BufferType<float3>   sa_cgR;
    BufferType<float3>   sa_cgZ;
    BufferType<float>    sa_block_result;
    BufferType<float>    sa_convergence;


    std::vector<std::vector<uint>> vert_adj_material_force_verts;
    std::vector<std::vector<uint>> vert_adj_stretch_springs;
    std::vector<std::vector<uint>> vert_adj_stretch_faces;
    std::vector<std::vector<uint>> vert_adj_bending_edges;
    std::vector<std::vector<uint>> vert_adj_affine_bodies;

    BufferType<uint> sa_vert_adj_material_force_verts_csr;
    BufferType<uint> sa_vert_adj_stretch_springs_csr;
    BufferType<uint> sa_vert_adj_stretch_faces_csr;
    BufferType<uint> sa_vert_adj_bending_edges_csr;
    BufferType<uint> sa_vert_adj_affine_bodies_csr;
};

}  // namespace lcs


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