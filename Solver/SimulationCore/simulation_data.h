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
struct VbdData : SimulationType
{
    BufferType<float>    sa_Hf;
    BufferType<float4x3> sa_Hf1;
};

namespace Constitutions
{
    enum class ConstraintType
    {
        StretchSpring,
        StretchFace,
        BendingEdge,
        StressTet,
        ElasticRod,
        AffineBody
    };

    template <template <typename...> typename BufferType, typename Derived>
    struct ConstitutionInterface : SimulationType
    {
        BufferType<ushort>   constraint_offsets_in_adjlist;
        BufferType<float3>   constraint_gradients;
        BufferType<float3x3> constraint_hessians;

        std::vector<std::vector<uint>> vert_adj_constraints;
        BufferType<uint>               vert_adj_constraints_csr;

        static constexpr size_t get_num_verts_per_constaint()
        {
            return Derived::get_num_verts_per_constaint();
        }
        static constexpr ConstraintType constraint_type() { return Derived::constraint_type(); }
        auto& get_indices() const { return static_cast<const Derived*>(this)->get_indices_impl(); }
        auto& get_constraint_offsets_in_adjlist() const { return constraint_offsets_in_adjlist; }
        auto& get_constraint_gradients() const { return constraint_gradients; }
        auto& get_constraint_hessians() const { return constraint_hessians; }
        auto& get_vert_adj_constraints() const { return vert_adj_constraints; }
        auto& get_vert_adj_constraints_csr() const { return vert_adj_constraints_csr; }


        template <typename T>
        static bool is_buffer_valid(const BufferType<T>& buffer)
        {
            if constexpr (requires { buffer.valid(); })
                return buffer.valid();
            else
                return !buffer.empty();
        }
        bool is_valid() const { return is_buffer_valid(get_indices()); }
        uint get_num_indices() const { return static_cast<uint>(get_indices().size()); }
    };

    template <template <typename...> typename BufferType>
    struct StretchSpring : ConstitutionInterface<BufferType, StretchSpring<BufferType>>
    {
        BufferType<uint2> sa_stretch_springs;
        BufferType<float> sa_stretch_spring_rest_state_length;
        BufferType<float> sa_stretch_spring_stiffness;

        static constexpr size_t         get_num_verts_per_constaint() { return 2; }
        static constexpr ConstraintType constraint_type() { return ConstraintType::StretchSpring; }
        auto&                           get_indices_impl() const { return sa_stretch_springs; }
    };

    template <template <typename...> typename BufferType>
    struct StretchFace : ConstitutionInterface<BufferType, StretchFace<BufferType>>
    {
        BufferType<uint3> sa_stretch_faces;
        BufferType<float> sa_stretch_faces_rest_area;
        BufferType<float2> sa_stretch_faces_mu_lambda;  // scaled by thickness, thus only multiply by area
        BufferType<float2x2> sa_stretch_faces_Dm_inv;

        static constexpr ConstraintType constraint_type() { return ConstraintType::StretchFace; }
        auto&                           get_indices_impl() const { return sa_stretch_faces; }
        static constexpr size_t         get_num_verts_per_constaint() { return 3; }
    };

    template <template <typename...> typename BufferType>
    struct BendingEdge : ConstitutionInterface<BufferType, BendingEdge<BufferType>>
    {
        BufferType<uint4>    sa_bending_edges;
        BufferType<float>    sa_bending_edges_rest_angle;
        BufferType<float>    sa_bending_edges_stiffness;
        BufferType<float4x4> sa_bending_edges_Q;
        BufferType<float>    sa_bending_edges_rest_area;

        static constexpr ConstraintType constraint_type() { return ConstraintType::BendingEdge; }
        auto&                           get_indices_impl() const { return sa_bending_edges; }
        static constexpr size_t         get_num_verts_per_constaint() { return 4; }
    };

    template <template <typename...> typename BufferType>
    struct StressTet : ConstitutionInterface<BufferType, StressTet<BufferType>>
    {
        BufferType<uint4>    sa_stress_tets;
        BufferType<float>    sa_stress_tets_rest_volume;
        BufferType<float2>   sa_stress_tets_mu_lambda;
        BufferType<float3x3> sa_stress_tets_Dm_inv;

        static constexpr ConstraintType constraint_type() { return ConstraintType::StressTet; }
        auto&                           get_indices_impl() const { return sa_stress_tets; }
        static constexpr size_t         get_num_verts_per_constaint() { return 4; }
    };

    template <template <typename...> typename BufferType>
    struct ElasticRod : ConstitutionInterface<BufferType, ElasticRod<BufferType>>
    {
        BufferType<uint2>    sa_elastic_rods;
        BufferType<float>    sa_elastic_rods_rest_volume;
        BufferType<float>    sa_elastic_rods_stiffness;
        BufferType<float2x2> sa_elastic_rods_Dm_inv;

        static constexpr ConstraintType constraint_type() { return ConstraintType::ElasticRod; }
        auto&                           get_indices_impl() const { return sa_elastic_rods; }
        static constexpr size_t         get_num_verts_per_constaint() { return 2; }
    };

    template <template <typename...> typename BufferType>
    struct AbdKinematics : ConstitutionInterface<BufferType, AbdKinematics<BufferType>>
    {
        BufferType<uint4>            sa_affine_bodies;
        BufferType<float>            sa_affine_bodies_kappa;
        BufferType<float>            sa_affine_bodies_volume;
        BufferType<float4x4>         sa_affine_bodies_mass_matrix;
        std::vector<EigenFloat12x12> sa_affine_bodies_mass_matrix_full;

        static constexpr ConstraintType constraint_type() { return ConstraintType::AffineBody; }
        auto&                           get_indices_impl() const { return sa_affine_bodies; }
        static constexpr size_t         get_num_verts_per_constaint() { return 4; }
    };

}  // namespace Constitutions

template <template <typename...> typename BufferType>
struct PcgInterfaceData : SimulationType
{
    // PCG
    BufferType<float3>           sa_cgX;
    BufferType<float3>           sa_cgB;
    BufferType<float3x3>         sa_cgA_diag;
    BufferType<MatrixTriplet3x3> sa_cgA_fixtopo_offdiag_triplet;
    BufferType<uint3>            sa_cgA_fixtopo_offdiag_triplet_info;
};

template <template <typename...> typename BufferType>
struct PcgInnerData : SimulationType
{
    // PCG
    BufferType<float3x3> sa_cgMinv;
    BufferType<float3>   sa_cgP;
    BufferType<float3>   sa_cgQ;
    BufferType<float3>   sa_cgR;
    BufferType<float3>   sa_cgZ;
    BufferType<float>    sa_block_result;
    BufferType<float>    sa_convergence;
};

// template <template <typename...> typename BufferType>
// struct AdjacentData : SimulationType
// {
//     std::vector<std::vector<uint>> vert_adj_material_force_verts;
//     BufferType<uint>               sa_vert_adj_material_force_verts_csr;
// };


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

  private:
    Constitutions::StretchSpring<BufferType> stretch_spring_constitution;
    Constitutions::StretchFace<BufferType>   stretch_face_constitution;
    Constitutions::BendingEdge<BufferType>   bending_edge_constitution;
    Constitutions::AbdKinematics<BufferType> affine_body_constitution;
    Constitutions::StressTet<BufferType>     stress_tet_constitution;
    Constitutions::ElasticRod<BufferType>    elastic_rod_constitution;


  public:
    Constitutions::StretchFace<BufferType>& get_stretch_face_data() { return stretch_face_constitution; }
    Constitutions::BendingEdge<BufferType>& get_bending_edge_data() { return bending_edge_constitution; }
    Constitutions::AbdKinematics<BufferType>& get_affine_body_data() { return affine_body_constitution; }
    Constitutions::StressTet<BufferType>&     get_stress_tet_data() { return stress_tet_constitution; }
    Constitutions::ElasticRod<BufferType>&    get_elastic_rod_data() { return elastic_rod_constitution; }

    const Constitutions::StretchSpring<BufferType>& get_stretch_spring_data() const
    {
        return stretch_spring_constitution;
    }
    Constitutions::StretchSpring<BufferType>& get_stretch_spring_data()
    {
        return stretch_spring_constitution;
    }
    const Constitutions::StretchFace<BufferType>& get_stretch_face_data() const
    {
        return stretch_face_constitution;
    }
    const Constitutions::BendingEdge<BufferType>& get_bending_edge_data() const
    {
        return bending_edge_constitution;
    }
    const Constitutions::AbdKinematics<BufferType>& get_affine_body_data() const
    {
        return affine_body_constitution;
    }
    const Constitutions::StressTet<BufferType>& get_stress_tet_data() const
    {
        return stress_tet_constitution;
    }
    const Constitutions::ElasticRod<BufferType>& get_elastic_rod_data() const
    {
        return elastic_rod_constitution;
    }


    BufferType<uint> sa_vert_affine_bodies_id;
    BufferType<uint> sa_affine_bodies_mesh_id;
    BufferType<uint> sa_affine_bodies_is_fixed;

    BufferType<float3> sa_affine_bodies_rest_q;
    BufferType<float3> sa_affine_bodies_rest_q_v;
    BufferType<float3> sa_affine_bodies_gravity;
    BufferType<float3> sa_affine_bodies_q;
    BufferType<float3> sa_affine_bodies_q_v;
    BufferType<float3> sa_affine_bodies_q_tilde;
    BufferType<float3> sa_affine_bodies_q_iter_start;
    BufferType<float3> sa_affine_bodies_q_step_start;

    // std::vector<EigenFloat12x12> sa_affine_bodies_mass_matrix_full;
    //
    // BufferType<uint4>    sa_affine_bodies;
    // BufferType<float>    sa_affine_bodies_kappa;
    // BufferType<float>    sa_affine_bodies_volume;
    // BufferType<float4x4> sa_affine_bodies_mass_matrix;
    // BufferType<float3>   sa_affine_bodies_gradients;
    // BufferType<float3x3> sa_affine_bodies_hessians;
    // BufferType<ushort>   sa_affine_bodies_offsets_in_adjlist;

    BufferType<uint>  sa_contact_active_verts;
    BufferType<uint>  sa_contact_active_edges;
    BufferType<uint>  sa_contact_active_faces;
    BufferType<float> sa_contact_active_verts_d_hat;
    BufferType<float> sa_contact_active_verts_offset;
    BufferType<float> sa_contact_active_verts_friction_coeff;


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
    BufferType<uint>               sa_vert_adj_material_force_verts_csr;
};

}  // namespace lcs

LUISA_BINDING_GROUP(lcs::Constitutions::StretchSpring<luisa::compute::Buffer>,
                    constraint_offsets_in_adjlist,
                    constraint_gradients,
                    constraint_hessians,
                    vert_adj_constraints_csr,
                    sa_stretch_springs,
                    sa_stretch_spring_rest_state_length,
                    sa_stretch_spring_stiffness){};

LUISA_BINDING_GROUP(lcs::Constitutions::StretchFace<luisa::compute::Buffer>,
                    constraint_offsets_in_adjlist,
                    constraint_gradients,
                    constraint_hessians,
                    vert_adj_constraints_csr,
                    sa_stretch_faces,
                    sa_stretch_faces_rest_area,
                    sa_stretch_faces_mu_lambda,
                    sa_stretch_faces_Dm_inv){};

LUISA_BINDING_GROUP(lcs::Constitutions::BendingEdge<luisa::compute::Buffer>,
                    constraint_offsets_in_adjlist,
                    constraint_gradients,
                    constraint_hessians,
                    vert_adj_constraints_csr,
                    sa_bending_edges,
                    sa_bending_edges_rest_angle,
                    sa_bending_edges_stiffness,
                    sa_bending_edges_Q,
                    sa_bending_edges_rest_area){};

LUISA_BINDING_GROUP(lcs::Constitutions::StressTet<luisa::compute::Buffer>,
                    constraint_offsets_in_adjlist,
                    constraint_gradients,
                    constraint_hessians,
                    vert_adj_constraints_csr,
                    sa_stress_tets,
                    sa_stress_tets_rest_volume,
                    sa_stress_tets_mu_lambda,
                    sa_stress_tets_Dm_inv){};

LUISA_BINDING_GROUP(lcs::Constitutions::AbdKinematics<luisa::compute::Buffer>,
                    constraint_offsets_in_adjlist,
                    constraint_gradients,
                    constraint_hessians,
                    vert_adj_constraints_csr,
                    sa_affine_bodies,
                    sa_affine_bodies_kappa,
                    sa_affine_bodies_volume,
                    sa_affine_bodies_mass_matrix){};

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