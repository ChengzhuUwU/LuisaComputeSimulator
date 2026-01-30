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

    BufferType<float>    sa_Hf;
    BufferType<float4x3> sa_Hf1;
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
        Orthogonality,
        SoftInertia,
        AbdInertia
    };
    static inline std::string_view to_string(ConstraintType type)
    {
        switch (type)
        {
            case ConstraintType::StretchSpring:
                return "Stretch Spring";
            case ConstraintType::StretchFace:
                return "Stretch Face";
            case ConstraintType::BendingEdge:
                return "Bending Edge";
            case ConstraintType::StressTet:
                return "Stress Tet";
            case ConstraintType::ElasticRod:
                return "Elastic Rod";
            case ConstraintType::Orthogonality:
                return "Affine Body Orthogonality";
            case ConstraintType::SoftInertia:
                return "Soft Body Inertia";
            case ConstraintType::AbdInertia:
                return "Affine Body Inertia";
            default:
                return "Unknown";
        }
    }

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
        static constexpr ConstraintType   constraint_type() { return Derived::constraint_type(); }
        static constexpr std::string_view get_constitution_name()
        {
            return to_string(constraint_type());
        }
        auto& get_indices() const { return static_cast<const Derived*>(this)->get_indices_impl(); }
        auto& get_constraint_offsets_in_adjlist() const { return constraint_offsets_in_adjlist; }
        auto& get_constraint_gradients() const { return constraint_gradients; }
        auto& get_constraint_hessians() const { return constraint_hessians; }
        auto& get_vert_adj_constraints() const { return vert_adj_constraints; }
        auto& get_vert_adj_constraints_csr() const { return vert_adj_constraints_csr; }

        auto& get_indices() { return static_cast<Derived*>(this)->get_indices_impl(); }
        auto& get_constraint_offsets_in_adjlist() { return constraint_offsets_in_adjlist; }
        auto& get_constraint_gradients() { return constraint_gradients; }
        auto& get_constraint_hessians() { return constraint_hessians; }
        auto& get_vert_adj_constraints() { return vert_adj_constraints; }
        auto& get_vert_adj_constraints_csr() { return vert_adj_constraints_csr; }


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
        BufferType<uint2> constraint_indices;
        BufferType<float> sa_stretch_spring_rest_state_length;
        BufferType<float> sa_stretch_spring_stiffness;

        static constexpr size_t         get_num_verts_per_constaint() { return 2; }
        static constexpr ConstraintType constraint_type() { return ConstraintType::StretchSpring; }
        auto&                           get_indices_impl() const { return constraint_indices; }
    };

    template <template <typename...> typename BufferType>
    struct StretchFace : ConstitutionInterface<BufferType, StretchFace<BufferType>>
    {
        BufferType<uint3> constraint_indices;
        BufferType<float> sa_stretch_faces_rest_area;
        BufferType<float2> sa_stretch_faces_mu_lambda;  // scaled by thickness, thus only multiply by area
        BufferType<float2x2> sa_stretch_faces_Dm_inv;

        static constexpr ConstraintType constraint_type() { return ConstraintType::StretchFace; }
        auto&                           get_indices_impl() const { return constraint_indices; }
        static constexpr size_t         get_num_verts_per_constaint() { return 3; }
    };

    template <template <typename...> typename BufferType>
    struct BendingEdge : ConstitutionInterface<BufferType, BendingEdge<BufferType>>
    {
        BufferType<uint4>    constraint_indices;
        BufferType<float>    sa_bending_edges_rest_angle;
        BufferType<float>    sa_bending_edges_stiffness;
        BufferType<float4x4> sa_bending_edges_Q;
        BufferType<float>    sa_bending_edges_rest_area;

        static constexpr ConstraintType constraint_type() { return ConstraintType::BendingEdge; }
        auto&                           get_indices_impl() const { return constraint_indices; }
        static constexpr size_t         get_num_verts_per_constaint() { return 4; }
    };

    template <template <typename...> typename BufferType>
    struct StressTet : ConstitutionInterface<BufferType, StressTet<BufferType>>
    {
        BufferType<uint4>    constraint_indices;
        BufferType<float>    sa_stress_tets_rest_volume;
        BufferType<float2>   sa_stress_tets_mu_lambda;
        BufferType<float3x3> sa_stress_tets_Dm_inv;

        static constexpr ConstraintType constraint_type() { return ConstraintType::StressTet; }
        auto&                           get_indices_impl() const { return constraint_indices; }
        static constexpr size_t         get_num_verts_per_constaint() { return 4; }
    };

    template <template <typename...> typename BufferType>
    struct ElasticRod : ConstitutionInterface<BufferType, ElasticRod<BufferType>>
    {
        BufferType<uint2>    constraint_indices;
        BufferType<float>    sa_elastic_rods_rest_volume;
        BufferType<float>    sa_elastic_rods_stiffness;
        BufferType<float2x2> sa_elastic_rods_Dm_inv;

        static constexpr ConstraintType constraint_type() { return ConstraintType::ElasticRod; }
        auto&                           get_indices_impl() const { return constraint_indices; }
        static constexpr size_t         get_num_verts_per_constaint() { return 2; }
    };

    template <template <typename...> typename BufferType>
    struct AbdOrthogonality : ConstitutionInterface<BufferType, AbdOrthogonality<BufferType>>
    {
        BufferType<uint3> constraint_indices;
        BufferType<float> abd_kappa;
        BufferType<float> abd_volume;

        static constexpr ConstraintType constraint_type() { return ConstraintType::Orthogonality; }
        auto&                           get_indices_impl() const { return constraint_indices; }
        static constexpr size_t         get_num_verts_per_constaint() { return 3; }
    };

    template <template <typename...> typename BufferType>
    struct SoftInertia : ConstitutionInterface<BufferType, SoftInertia<BufferType>>
    {
        // BufferType<uint>   sa_soft_vert_attributes;
        BufferType<uint>  constraint_indices;
        BufferType<float> sa_soft_vert_mass;
        BufferType<float> sa_stiffness_dirichlet;
        // BufferType<float3> sa_x_tilde;

        static auto vert_is_fixed(const auto vert_attribute) { return (vert_attribute & 0x1) == 0; }

        static constexpr ConstraintType constraint_type() { return ConstraintType::SoftInertia; }
        auto&                           get_indices_impl() const { return constraint_indices; }
        static constexpr size_t         get_num_verts_per_constaint() { return 1; }
    };

    template <template <typename...> typename BufferType>
    struct AbdInertia : ConstitutionInterface<BufferType, AbdInertia<BufferType>>
    {
        // BufferType<float3>           sa_q_tilde;
        BufferType<uint4>            constraint_indices;
        BufferType<float>            sa_stiffness_dirichlet;
        BufferType<float4x4>         sa_affine_bodies_mass_matrix;
        std::vector<EigenFloat12x12> sa_affine_bodies_mass_matrix_full;

        static constexpr ConstraintType constraint_type() { return ConstraintType::AbdInertia; }
        auto&                           get_indices_impl() const { return constraint_indices; }
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

namespace Attributions
{
    static constexpr uint32_t RIGID_BODY_FLAG = 1u << 31;
    static constexpr uint32_t RIGID_BODY_MASK = ~RIGID_BODY_FLAG;

    static constexpr uint32_t ABD_Is_Translation_DOF = 1u << 30;

}  // namespace Attributions


template <template <typename...> typename BufferType>
struct SimulationData : SimulationType
{
    // Simulation state
    BufferType<float3> sa_rest_q;        // Constant
    BufferType<float3> sa_rest_q_v;      // Constant
    BufferType<float3> sa_q;             // Re-calculate every frame
    BufferType<float3> sa_q_v;           // Re-calculate every frame
    BufferType<float3> sa_q_iter_start;  // Re-calculate every frame
    BufferType<float3> sa_q_step_start;  // Input
    BufferType<float3> sa_q_tilde;       // Re-calculate every frame
    BufferType<float3> sa_dq;            // Re-calculate every frame

    std::vector<float3> sa_q_outer;    // Input from outer
    std::vector<float3> sa_q_v_outer;  // Input from outer

    // Vert position and velocity
    BufferType<float3> sa_scaled_model_x;
    BufferType<float3> sa_rest_x;
    BufferType<float3> sa_rest_v;
    BufferType<float3> sa_x;
    BufferType<float3> sa_dx;
    BufferType<float3> sa_v;
    BufferType<float3> sa_x_step_start;
    BufferType<float3> sa_x_iter_start;

    // If value & (1<<31) == 0, then it's a soft body vert, map to dof directly;
    //                 else it's a rigid body vert, map to dof by affine body id
    BufferType<uint> sa_x_to_dof_map;
    BufferType<uint> sa_q_is_fixed;
    BufferType<uint> sa_q_property;  // Constant

    std::vector<float3> sa_x_outer;
    std::vector<float3> sa_v_outer;

    // Energy
    uint              num_verts_total   = 0;
    uint              num_verts_soft    = 0;
    uint              num_verts_rigid   = 0;
    uint              num_affine_bodies = 0;
    uint              num_dof           = 0;  // Degree of freedom, actually is DOF / 3
    BufferType<uint>  sa_num_dof;
    BufferType<float> sa_system_energy;

    BufferType<uint> sa_vert_affine_bodies_id;
    BufferType<uint> sa_affine_bodies_mesh_id;
    // BufferType<uint> sa_affine_bodies_is_fixed;

    // BufferType<float3> sa_affine_bodies_gravity;

    BufferType<uint>  sa_contact_active_verts;
    BufferType<uint>  sa_contact_active_edges;
    BufferType<uint>  sa_contact_active_faces;
    BufferType<float> sa_contact_active_verts_d_hat;
    BufferType<float> sa_contact_active_verts_offset;
    BufferType<float> sa_contact_active_verts_friction_coeff;

    ColoredData<BufferType> colored_data;


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

  public:
    Constitutions::StretchFace<BufferType>&      get_stretch_face_data() { return stretch_face; }
    Constitutions::BendingEdge<BufferType>&      get_bending_edge_data() { return bending_edge; }
    Constitutions::StressTet<BufferType>&        get_stress_tet_data() { return stress_tet; }
    Constitutions::ElasticRod<BufferType>&       get_elastic_rod_data() { return elastic_rod; }
    Constitutions::AbdInertia<BufferType>&       get_abd_inertia_data() { return abd_inertia; }
    Constitutions::SoftInertia<BufferType>&      get_soft_inertia_data() { return soft_inertia; }
    Constitutions::StretchSpring<BufferType>&    get_stretch_spring_data() { return stretch_spring; }
    Constitutions::AbdOrthogonality<BufferType>& get_abd_orthogonality_data()
    {
        return abd_orthogonality;
    }

    const Constitutions::StretchSpring<BufferType>& get_stretch_spring_data() const
    {
        return stretch_spring;
    }
    const Constitutions::StretchFace<BufferType>& get_stretch_face_data() const { return stretch_face; }
    const Constitutions::BendingEdge<BufferType>& get_bending_edge_data() const { return bending_edge; }
    const Constitutions::AbdOrthogonality<BufferType>& get_abd_orthogonality_data() const
    {
        return abd_orthogonality;
    }
    const Constitutions::StressTet<BufferType>&   get_stress_tet_data() const { return stress_tet; }
    const Constitutions::ElasticRod<BufferType>&  get_elastic_rod_data() const { return elastic_rod; }
    const Constitutions::AbdInertia<BufferType>&  get_abd_inertia_data() const { return abd_inertia; }
    const Constitutions::SoftInertia<BufferType>& get_soft_inertia_data() const { return soft_inertia; }

  private:
    Constitutions::StretchSpring<BufferType>    stretch_spring;
    Constitutions::StretchFace<BufferType>      stretch_face;
    Constitutions::BendingEdge<BufferType>      bending_edge;
    Constitutions::AbdOrthogonality<BufferType> abd_orthogonality;
    Constitutions::StressTet<BufferType>        stress_tet;
    Constitutions::ElasticRod<BufferType>       elastic_rod;
    Constitutions::AbdInertia<BufferType>       abd_inertia;
    Constitutions::SoftInertia<BufferType>      soft_inertia;
};

}  // namespace lcs

LUISA_BINDING_GROUP(lcs::Constitutions::StretchSpring<luisa::compute::Buffer>,
                    constraint_indices,
                    constraint_offsets_in_adjlist,
                    constraint_gradients,
                    constraint_hessians,
                    vert_adj_constraints_csr,
                    sa_stretch_spring_rest_state_length,
                    sa_stretch_spring_stiffness){};

LUISA_BINDING_GROUP(lcs::Constitutions::StretchFace<luisa::compute::Buffer>,
                    constraint_indices,
                    constraint_offsets_in_adjlist,
                    constraint_gradients,
                    constraint_hessians,
                    vert_adj_constraints_csr,
                    sa_stretch_faces_rest_area,
                    sa_stretch_faces_mu_lambda,
                    sa_stretch_faces_Dm_inv){};

LUISA_BINDING_GROUP(lcs::Constitutions::BendingEdge<luisa::compute::Buffer>,
                    constraint_indices,
                    constraint_offsets_in_adjlist,
                    constraint_gradients,
                    constraint_hessians,
                    vert_adj_constraints_csr,
                    sa_bending_edges_rest_angle,
                    sa_bending_edges_stiffness,
                    sa_bending_edges_Q,
                    sa_bending_edges_rest_area){};

LUISA_BINDING_GROUP(lcs::Constitutions::StressTet<luisa::compute::Buffer>,
                    constraint_offsets_in_adjlist,
                    constraint_gradients,
                    constraint_hessians,
                    vert_adj_constraints_csr,
                    constraint_indices,
                    sa_stress_tets_rest_volume,
                    sa_stress_tets_mu_lambda,
                    sa_stress_tets_Dm_inv){};

LUISA_BINDING_GROUP(lcs::Constitutions::AbdOrthogonality<luisa::compute::Buffer>,
                    constraint_indices,
                    constraint_offsets_in_adjlist,
                    constraint_gradients,
                    constraint_hessians,
                    vert_adj_constraints_csr,
                    abd_kappa,
                    abd_volume){};

LUISA_BINDING_GROUP(lcs::Constitutions::AbdInertia<luisa::compute::Buffer>,
                    constraint_indices,
                    constraint_offsets_in_adjlist,
                    constraint_gradients,
                    constraint_hessians,
                    vert_adj_constraints_csr,
                    sa_stiffness_dirichlet,
                    sa_affine_bodies_mass_matrix){};

LUISA_BINDING_GROUP(lcs::Constitutions::SoftInertia<luisa::compute::Buffer>,
                    constraint_indices,
                    constraint_offsets_in_adjlist,
                    constraint_gradients,
                    constraint_hessians,
                    vert_adj_constraints_csr,
                    sa_soft_vert_mass,
                    sa_stiffness_dirichlet){};


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