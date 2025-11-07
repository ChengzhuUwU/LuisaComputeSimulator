#include "Initializer/init_sim_data.h"
#include "Core/affine_position.h"
#include "Core/float_n.h"
#include "Core/float_nxn.h"
#include "Core/lc_to_eigen.h"
#include "Energy/bending_energy.h"
#include "Energy/stretch_energy.h"
#include "Initializer/init_mesh_data.h"
#include "MeshOperation/mesh_reader.h"
#include "Initializer/initializer_utils.h"
#include "luisa/core/logging.h"
#include "luisa/core/mathematics.h"


namespace lcs::Initializer
{

template <uint N>
std::array<luisa::ushort, N*(N - 1)> get_offsets_in_adjlist_from_adjacent_list(
    const std::vector<std::vector<uint>>& vert_adj_verts, const luisa::Vector<uint, N>& element)
{
    std::array<luisa::ushort, N*(N - 1)> offsets = {0};
    uint                                 idx     = 0;
    for (uint ii = 0; ii < N; ii++)
    {
        const uint               vid      = element[ii];
        const std::vector<uint>& adj_list = vert_adj_verts[vid];
        for (uint jj = 0; jj < N; jj++)
        {
            if (ii != jj)
            {
                const uint adj_vid = element[jj];
                const uint offset =
                    std::distance(adj_list.begin(), std::find(adj_list.begin(), adj_list.end(), adj_vid));
                if (offset >= adj_list.size())
                {
                    LUISA_ERROR("Offset in adjlist not found! vid = {}, adj_vid = {}, adj_list_size = {}",
                                vid,
                                adj_vid,
                                adj_list.size());
                }
                offsets[idx] = luisa::ushort(offset);
                idx += 1;
            }
        }
    }
    return offsets;
}

static void compute_trimesh_dyadic_mass(const std::vector<float3>& pos_view,
                                        const std::vector<uint3>&  tri_view,
                                        const uint                 prefix_face_start,
                                        const uint                 prefix_face_end,
                                        float                      rho,
                                        float&                     m,
                                        float3&                    m_x_bar,
                                        float3x3&                  m_x_bar_x_bar)
{
    m             = 0.0;
    m_x_bar       = float3(0.0f);
    m_x_bar_x_bar = float3x3::fill(0.0f);

    // Using Divergence theorem to compute the dyadic mass
    // by integrating on the surface of the trimesh
    // for(auto&& [i, F] : (tri_view))

    for (size_t fid = prefix_face_start; fid < prefix_face_end; fid++)
    {
        const auto& F = tri_view[fid];

        const auto& p0 = pos_view[F[0]];
        const auto& p1 = pos_view[F[1]];
        const auto& p2 = pos_view[F[2]];

        auto e1 = p1 - p0;
        auto e2 = p2 - p0;

        auto N = luisa::cross(e1, e2);  // e1.cross(e2);

        m += rho * luisa::dot(p0, N) / 6.0;

        {
            auto Q = [](const uint a, const float rho, const float3& N, const float3& p0, const float3& p1, const float3& p2)
            {
                float V = 0.0;

                V += p0[a] * p0[a] / 12;
                V += p0[a] * p1[a] / 12;
                V += p0[a] * p2[a] / 12;

                V += p1[a] * p1[a] / 12;
                V += p1[a] * p2[a] / 12;
                V += p2[a] * p2[a] / 12;

                return rho / 2 * N[a] * V;
            };

            for (uint a = 0; a < 3; a++)
            {
                m_x_bar[a] += Q(a, rho, N, p0, p1, p2);
            }
        }

        {
            auto Q = [](const uint a, float rho, const float3& N, const float3& p0, const float3& p1, const float3& p2)
            {
                float V = 0.0;

                float p0a_2 = p0[a] * p0[a];
                float p1a_2 = p1[a] * p1[a];
                float p2a_2 = p2[a] * p2[a];

                float p0a_3 = p0a_2 * p0[a];
                float p1a_3 = p1a_2 * p1[a];
                float p2a_3 = p2a_2 * p2[a];

                V += p0a_3 / 20;
                V += p0a_2 * p1[a] / 20;
                V += p0a_2 * p2[a] / 20;

                V += p0[a] * p1a_2 / 20;
                V += p0[a] * p1[a] * p2[a] / 20;
                V += p0[a] * p2a_2 / 20;

                V += p1a_3 / 20;
                V += p1a_2 * p2[a] / 20;
                V += p1[a] * p2a_2 / 20;

                V += p2a_3 / 20;

                return rho / 3 * N[a] * V;
            };

            for (uint j = 0; j < 3; j++)  // diagonal
                m_x_bar_x_bar[j][j] += Q(j, rho, N, p0, p1, p2);
        }

        {
            auto Q = [](const uint a, const uint b, float rho, const float3& N, const float3& p0, const float3& p1, const float3& p2)
            {
                float V = 0.0;

                float p0a_2 = p0[a] * p0[a];
                float p1a_2 = p1[a] * p1[a];
                float p2a_2 = p2[a] * p2[a];

                V += p0a_2 * p0[b] / 20;
                V += p0a_2 * p1[b] / 60;
                V += p0a_2 * p2[b] / 60;
                V += p0[a] * p0[b] * p1[a] / 30;

                V += p0[a] * p0[b] * p2[a] / 30;
                V += p0[a] * p1[a] * p1[b] / 30;
                V += p0[a] * p1[a] * p2[b] / 60;
                V += p0[a] * p1[b] * p2[a] / 60;

                V += p0[a] * p2[a] * p2[b] / 30;
                V += p0[b] * p1a_2 / 60;
                V += p0[b] * p1[a] * p2[a] / 60;
                V += p0[b] * p2a_2 / 60;

                V += p1a_2 * p1[b] / 20;
                V += p1a_2 * p2[b] / 60;
                V += p1[a] * p1[b] * p2[a] / 30;
                V += p1[a] * p2[a] * p2[b] / 30;

                V += p1[b] * p2a_2 / 60;
                V += p2a_2 * p2[b] / 20;

                return rho / 2 * N[a] * V;
            };

            m_x_bar_x_bar[0][1] += Q(0, 1, rho, N, p0, p1, p2);
            m_x_bar_x_bar[0][2] += Q(0, 2, rho, N, p0, p1, p2);
            m_x_bar_x_bar[1][2] += Q(1, 2, rho, N, p0, p1, p2);
        }

        // symmetric
        m_x_bar_x_bar[1][0] = m_x_bar_x_bar[0][1];
        m_x_bar_x_bar[2][0] = m_x_bar_x_bar[0][2];
        m_x_bar_x_bar[2][1] = m_x_bar_x_bar[1][2];
    }

    // m_x_bar_x_bar = luisa::transpose(m_x_bar_x_bar);
}


void init_sim_data(std::vector<lcs::Initializer::WorldData>& world_data,
                   lcs::MeshData<std::vector>*               mesh_data,
                   lcs::SimulationData<std::vector>*         sim_data)
{
    sim_data->sa_x_tilde.resize(mesh_data->num_verts);
    sim_data->sa_x.resize(mesh_data->num_verts);
    sim_data->sa_v.resize(mesh_data->num_verts);
    sim_data->sa_x_step_start.resize(mesh_data->num_verts);
    sim_data->sa_x_iter_start.resize(mesh_data->num_verts);

    // Init target positions
    {
        sim_data->sa_target_positions.resize(mesh_data->num_verts);
        CpuParallel::parallel_copy(mesh_data->sa_rest_x, sim_data->sa_target_positions);
    }

    // Calculate number of energy element
    constexpr bool cull_unused_constraints = true;

    std::vector<uint> stretch_spring_indices = fn_get_active_indices(
        [&](const uint eid)
        {
            const uint  mesh_idx   = mesh_data->sa_edge_mesh_id[eid];
            const auto& mesh_info  = world_data[mesh_idx];
            bool        use_spring = false;
            if (mesh_info.holds<ClothMaterial>())
            {
                use_spring = mesh_info.get_material<ClothMaterial>().stretch_model
                             == ConstitutiveStretchModelCloth::Spring;
            }
            // else if (mesh_info.holds<TetMaterial>())
            // {
            //     use_spring = mesh_info.get_material<TetMaterial>().model == ConstitutiveModelTet::Spring;
            // }
            // else if (shell_info.holds<RigidMaterial>())
            // {
            //     use_spring = shell_info.get<RigidMaterial>().model == ConstitutiveModelRigid::Spring;
            // }
            else if (mesh_info.holds<RodMaterial>())
            {
                use_spring = mesh_info.get_material<RodMaterial>().model == ConstitutiveModelRod::Spring;
            }
            // bool  is_cloth   = mesh_data->sa_vert_mesh_type[edge[0]] == uint(ShellTypeCloth);
            uint2 edge       = mesh_data->sa_edges[eid];
            bool  is_dynamic = cull_unused_constraints ?
                                   !mesh_data->sa_is_fixed[edge[0]] || !mesh_data->sa_is_fixed[edge[1]] :
                                   true;
            return (use_spring && is_dynamic) ? 1 : 0;
        },
        mesh_data->num_edges);

    std::vector<uint> stretch_face_indices = fn_get_active_indices(
        [&](const uint fid)
        {
            const uint  mesh_idx         = mesh_data->sa_face_mesh_id[fid];
            const auto& mesh_info        = world_data[mesh_idx];
            bool        use_stretch_face = mesh_info.holds<ClothMaterial>()
                                    && mesh_info.get_material<ClothMaterial>().stretch_model
                                           == ConstitutiveStretchModelCloth::FEM_BW98;
            uint3 face       = mesh_data->sa_faces[fid];
            bool  is_dynamic = cull_unused_constraints ?
                                   !mesh_data->sa_is_fixed[face[0]] || !mesh_data->sa_is_fixed[face[1]]
                                      || !mesh_data->sa_is_fixed[face[2]] :
                                   true;
            return (use_stretch_face && is_dynamic) ? 1 : 0;
        },
        mesh_data->num_faces);

    std::vector<uint> bending_edge_indices = fn_get_active_indices(
        [&](const uint eid)
        {
            const uint  mesh_idx    = mesh_data->sa_dihedral_edge_mesh_id[eid];
            const auto& mesh_info   = world_data[mesh_idx];
            bool        use_bending = mesh_info.holds<ClothMaterial>()
                               && mesh_info.get_material<ClothMaterial>().bending_model
                                      != ConstitutiveBendingModelCloth::None;
            uint4 edge       = mesh_data->sa_dihedral_edges[eid];
            bool  is_dynamic = cull_unused_constraints ?
                                   !mesh_data->sa_is_fixed[edge[0]] || !mesh_data->sa_is_fixed[edge[1]]
                                      || !mesh_data->sa_is_fixed[edge[2]] || !mesh_data->sa_is_fixed[edge[3]] :
                                   true;
            return (use_bending && is_dynamic) ? 1 : 0;
        },
        mesh_data->num_dihedral_edges);

    std::vector<uint> stress_tet_indices = fn_get_active_indices(
        [&](const uint tid)
        {
            const uint  mesh_idx   = mesh_data->sa_tet_mesh_id[tid];
            const auto& mesh_info  = world_data[mesh_idx];
            bool        use_stress = mesh_info.holds<TetMaterial>();
            uint4       tet        = mesh_data->sa_tetrahedrons[tid];
            bool        is_dynamic = cull_unused_constraints ?
                                         !mesh_data->sa_is_fixed[tet[0]] || !mesh_data->sa_is_fixed[tet[1]]
                                      || !mesh_data->sa_is_fixed[tet[2]] || !mesh_data->sa_is_fixed[tet[3]] :
                                         true;
            return (use_stress && is_dynamic) ? 1 : 0;
        },
        mesh_data->num_tets);

    std::vector<uint> affine_body_indices = fn_get_active_indices(
        [&](const uint meshIdx)
        {
            const uint curr_prefix       = mesh_data->prefix_num_verts[meshIdx];
            const uint first_vid         = curr_prefix;
            const bool has_boundary_edge =  // Unclosed
                (mesh_data->prefix_num_edges[meshIdx + 1] - mesh_data->prefix_num_edges[meshIdx])
                != (mesh_data->prefix_num_dihedral_edges[meshIdx + 1] - mesh_data->prefix_num_dihedral_edges[meshIdx]);
            // bool has_dynamic_vert = mesh_data->sa_is_fixed[first_vid];
            // bool is_rigid = (mesh_data->sa_vert_mesh_type[first_vid] == uint(ShellTypeRigid));  // ;&& !has_boundary_edge;
            bool is_rigid = world_data[meshIdx].holds<RigidMaterial>();
            // LUISA_INFO("Mesh {} is rigid = {}, has_boundary_edge = {}", meshIdx, is_rigid, has_boundary_edge);
            return (is_rigid) ? 1 : 0;  // has_dynamic_vert
        },
        mesh_data->num_meshes);

    std::vector<uint> soft_vert_indices =
        fn_get_active_indices([&](const uint vid)
                              { return mesh_data->sa_vert_mesh_type[vid] == SimulationTypeRigid ? 0 : 1; },
                              mesh_data->num_verts);

    const uint num_stretch_springs = static_cast<uint>(stretch_spring_indices.size());
    const uint num_stretch_faces   = static_cast<uint>(stretch_face_indices.size());
    const uint num_bending_edges   = static_cast<uint>(bending_edge_indices.size());
    const uint num_stress_tets     = static_cast<uint>(stress_tet_indices.size());
    const uint num_affine_bodies   = static_cast<uint>(affine_body_indices.size());
    const uint num_verts_soft      = static_cast<uint>(soft_vert_indices.size());
    const uint num_dof             = num_verts_soft + num_affine_bodies * 4;

    LUISA_INFO("Initialized energy element counts:");
    LUISA_INFO("      num Stretch Spring = {} (<{})", num_stretch_springs, mesh_data->num_edges);
    LUISA_INFO("      num Stretch Face   = {} (<{})", num_stretch_faces, mesh_data->num_faces);
    LUISA_INFO("      num Bending Edge   = {} (<{})", num_bending_edges, mesh_data->num_dihedral_edges);
    LUISA_INFO("      num Stress Tet     = {} (<{})", num_stress_tets, mesh_data->num_tets);
    LUISA_INFO("      num Affine Body    = {} (<{})", num_affine_bodies, mesh_data->num_meshes);
    LUISA_INFO("      num Soft Vert      = {} (<{})", num_verts_soft, mesh_data->num_verts);
    LUISA_INFO("      Total DOF = {}, NumVertSoft = {}, NumAffineBodies {}", num_dof, num_verts_soft, num_affine_bodies);


    sim_data->num_verts_soft    = num_verts_soft;
    sim_data->num_verts_rigid   = mesh_data->num_verts - num_verts_soft;
    sim_data->num_affine_bodies = num_affine_bodies;
    sim_data->num_dof           = num_dof;
    sim_data->sa_num_dof.resize(1);
    sim_data->sa_num_dof[0] = num_dof;


    // Init energy
    {
        sim_data->sa_system_energy.resize(10240);
        // Rest spring length
        sim_data->sa_stretch_springs.resize(num_stretch_springs);
        sim_data->sa_stretch_spring_rest_state_length.resize(num_stretch_springs);
        sim_data->sa_stretch_spring_stiffness.resize(num_stretch_springs);
        sim_data->sa_stretch_springs_gradients.resize(num_stretch_springs * 2);
        sim_data->sa_stretch_springs_hessians.resize(num_stretch_springs * 4);
        CpuParallel::parallel_for(0,
                                  num_stretch_springs,
                                  [&](const uint eid)
                                  {
                                      const uint orig_eid               = stretch_spring_indices[eid];
                                      uint2      edge                   = mesh_data->sa_edges[orig_eid];
                                      float3     x1                     = mesh_data->sa_rest_x[edge[0]];
                                      float3     x2                     = mesh_data->sa_rest_x[edge[1]];
                                      sim_data->sa_stretch_springs[eid] = edge;
                                      sim_data->sa_stretch_spring_rest_state_length[eid] =
                                          lcs::length_vec(x1 - x2);

                                      const auto& mesh_info = world_data[mesh_data->sa_edge_mesh_id[orig_eid]];
                                      const auto& material = mesh_info.get_material<ClothMaterial>();

                                      const float E     = material.youngs_modulus;
                                      const float nu    = material.poisson_ratio;
                                      auto [mu, lambda] = StretchEnergy::convert_prop(E, nu);
                                      mu                = mu * material.thickness;  // scale by thickness
                                      sim_data->sa_stretch_spring_stiffness[eid] = mu;
                                  });

        // Rest stretch face length
        sim_data->sa_stretch_faces.resize(num_stretch_faces);
        sim_data->sa_stretch_faces_mu_lambda.resize(num_stretch_faces);
        sim_data->sa_stretch_faces_rest_area.resize(num_stretch_faces);
        sim_data->sa_stretch_faces_Dm_inv.resize(num_stretch_faces);
        sim_data->sa_stretch_faces_gradients.resize(num_stretch_faces * 3);
        sim_data->sa_stretch_faces_hessians.resize(num_stretch_faces * 9);
        CpuParallel::parallel_for(0,
                                  num_stretch_faces,
                                  [&](const uint fid)
                                  {
                                      const uint    orig_fid    = stretch_face_indices[fid];
                                      uint3         face        = mesh_data->sa_faces[orig_fid];
                                      const float3  vert_pos[3] = {mesh_data->sa_rest_x[face[0]],
                                                                   mesh_data->sa_rest_x[face[1]],
                                                                   mesh_data->sa_rest_x[face[2]]};
                                      const float3& x_0         = vert_pos[0];
                                      const float3& x_1         = vert_pos[1];
                                      const float3& x_2         = vert_pos[2];

                                      const float2x2 inv_duv = StretchEnergy::get_Dm_inv(x_0, x_1, x_2);
                                      const float    area    = compute_face_area(x_0, x_1, x_2);

                                      const auto& mesh_info = world_data[mesh_data->sa_face_mesh_id[orig_fid]];
                                      const auto& material = mesh_info.get_material<ClothMaterial>();

                                      const float E  = material.youngs_modulus;
                                      const float nu = material.poisson_ratio;

                                      auto [mu, lambda] = StretchEnergy::convert_prop(E, nu);
                                      mu                = material.thickness * mu;  // scale by thickness
                                      lambda            = material.thickness * lambda;
                                      sim_data->sa_stretch_faces_mu_lambda[fid] = luisa::make_float2(mu, lambda);
                                      sim_data->sa_stretch_faces[fid]           = face;
                                      sim_data->sa_stretch_faces_rest_area[fid] = area;
                                      sim_data->sa_stretch_faces_Dm_inv[fid]    = inv_duv;
                                  });

        // Rest bending info
        sim_data->sa_bending_edges.resize(num_bending_edges);
        sim_data->sa_bending_edges_rest_area.resize(num_bending_edges);
        sim_data->sa_bending_edges_rest_angle.resize(num_bending_edges);
        sim_data->sa_bending_edges_stiffness.resize(num_bending_edges);
        sim_data->sa_bending_edges_Q.resize(num_bending_edges);
        sim_data->sa_bending_edges_gradients.resize(num_bending_edges * 4);
        sim_data->sa_bending_edges_hessians.resize(num_bending_edges * 16);
        CpuParallel::parallel_for(
            0,
            num_bending_edges,
            [&](const uint eid)
            {
                const uint orig_eid = bending_edge_indices[eid];
                uint4      edge     = mesh_data->sa_dihedral_edges[orig_eid];

                edge                     = luisa::make_uint4(edge[0], edge[1], edge[2], edge[3]);
                const float3 vert_pos[4] = {mesh_data->sa_rest_x[edge[0]],
                                            mesh_data->sa_rest_x[edge[1]],
                                            mesh_data->sa_rest_x[edge[2]],
                                            mesh_data->sa_rest_x[edge[3]]};

                // Rest state angle
                {
                    const float3& x0 = vert_pos[0];
                    const float3& x1 = vert_pos[1];
                    const float3& x2 = vert_pos[2];
                    const float3& x3 = vert_pos[3];

                    const float angle = lcs::BendingEnergy::compute_theta(x0, x1, x2, x3);

                    const float A1    = compute_face_area(x0, x1, x2);
                    const float A2    = compute_face_area(x0, x1, x3);
                    const float L0    = luisa::length(x0 - x1);
                    const float h_bar = (A1 + A2) / (3.0f * L0);

                    if (luisa::isnan(angle))
                        LUISA_ERROR("is nan rest angle {}", eid);

                    sim_data->sa_bending_edges_rest_area[eid]  = h_bar;
                    sim_data->sa_bending_edges[eid]            = edge;
                    sim_data->sa_bending_edges_rest_angle[eid] = angle;
                    sim_data->sa_bending_edges_stiffness[eid] =
                        world_data[mesh_data->sa_dihedral_edge_mesh_id[orig_eid]]
                            .get_material<ClothMaterial>()
                            .area_bending_stiffness;
                }

                // Rest state Q
                {
                    auto calculateCotTheta = [](const float3& x, const float3& y)
                    {
                        // const float scaled_cos_theta = dot_vec(x, y);
                        // const float scaled_sin_theta = (sqrt_scalar(1.0f - square_scalar(scaled_cos_theta)));
                        const float scaled_cos_theta = luisa::dot(x, y);
                        const float scaled_sin_theta = luisa::length(luisa::cross(x, y));
                        return scaled_cos_theta / scaled_sin_theta;
                    };

                    float3       e0     = vert_pos[1] - vert_pos[0];
                    float3       e1     = vert_pos[2] - vert_pos[0];
                    float3       e2     = vert_pos[3] - vert_pos[0];
                    float3       e3     = vert_pos[2] - vert_pos[1];
                    float3       e4     = vert_pos[3] - vert_pos[1];
                    const float  cot_01 = calculateCotTheta(e0, -e1);
                    const float  cot_02 = calculateCotTheta(e0, -e2);
                    const float  cot_03 = calculateCotTheta(e0, e3);
                    const float  cot_04 = calculateCotTheta(e0, e4);
                    const float4 K =
                        luisa::make_float4(cot_03 + cot_04, cot_01 + cot_02, -cot_01 - cot_03, -cot_02 - cot_04);
                    const float A_0 = 0.5f * luisa::length(luisa::cross(e0, e1));
                    const float A_1 = 0.5f * luisa::length(luisa::cross(e0, e2));
                    // if (is_nan_vec<float4>(K) || is_inf_vec<float4>(K)) fast_print_err("Q of Bending is Illigal");
                    const float4x4 m_Q = (3.f / (A_0 + A_1)) * lcs::outer_product(K, K);  // Q = 3 qq^T / (A0+A1) ==> Q is symmetric
                    sim_data->sa_bending_edges_Q[eid] = m_Q;  // See : A quadratic bending model for inextensible surfaces.
                }
            });

        // Rest tetrahedron info
        // BufferType<uint4>    sa_stress_tets;
        // BufferType<float>    sa_stress_tets_rest_volume;
        // BufferType<float>    sa_stress_tets_stiffness;
        // BufferType<float3x3> sa_stress_tets_Dm_inv;
        // BufferType<ushort>   sa_stress_tets_offsets_in_adjlist;
        // BufferType<float3>   sa_stress_tets_gradients;
        // BufferType<float3x3> sa_stress_tets_hessians;
        sim_data->sa_stress_tets.resize(num_stress_tets);
        sim_data->sa_stress_tets_rest_volume.resize(num_stress_tets);
        sim_data->sa_stress_tets_mu_lambda.resize(num_stress_tets);
        sim_data->sa_stress_tets_Dm_inv.resize(num_stress_tets);
        sim_data->sa_stress_tets_offsets_in_adjlist.resize(num_stress_tets);
        sim_data->sa_stress_tets_gradients.resize(num_stress_tets * 4);
        sim_data->sa_stress_tets_hessians.resize(num_stress_tets * 16);
        CpuParallel::parallel_for(0,
                                  num_stress_tets,
                                  [&](const uint tid)
                                  {
                                      const uint    orig_tid    = stress_tet_indices[tid];
                                      uint4         tet         = mesh_data->sa_tetrahedrons[orig_tid];
                                      const float3  vert_pos[4] = {mesh_data->sa_rest_x[tet[0]],
                                                                   mesh_data->sa_rest_x[tet[1]],
                                                                   mesh_data->sa_rest_x[tet[2]],
                                                                   mesh_data->sa_rest_x[tet[3]]};
                                      const float3& x0          = vert_pos[0];
                                      const float3& x1          = vert_pos[1];
                                      const float3& x2          = vert_pos[2];
                                      const float3& x3          = vert_pos[3];
                                      const float   volume      = compute_tet_volume(x0, x1, x2, x3);

                                      const float3x3 Dm = luisa::make_float3x3(x1 - x0, x2 - x0, x3 - x0);
                                      const float3x3 Dm_inv = luisa::inverse(Dm);
                                      const auto& mesh_info = world_data[mesh_data->sa_tet_mesh_id[orig_tid]];
                                      const auto& material = mesh_info.get_material<TetMaterial>();
                                      const float E        = material.youngs_modulus;
                                      const float nu       = material.poisson_ratio;
                                      auto [mu, lambda]    = StretchEnergy::convert_prop(E, nu);
                                      sim_data->sa_stress_tets[tid]             = tet;
                                      sim_data->sa_stress_tets_rest_volume[tid] = volume;
                                      sim_data->sa_stress_tets_Dm_inv[tid]      = Dm_inv;
                                      sim_data->sa_stress_tets_mu_lambda[tid] = luisa::make_float2(mu, lambda);
                                  });

        // Rest affine body info
        const uint num_blocks_affine_body = num_affine_bodies * 4;
        sim_data->sa_affine_bodies_mesh_id.resize(num_affine_bodies);
        sim_data->sa_affine_bodies.resize(num_affine_bodies);
        sim_data->sa_affine_bodies_is_fixed.resize(num_affine_bodies);

        sim_data->sa_affine_bodies_rest_q.resize(num_blocks_affine_body);
        sim_data->sa_affine_bodies_rest_q_v.resize(num_blocks_affine_body);
        sim_data->sa_affine_bodies_gravity.resize(num_blocks_affine_body);
        sim_data->sa_affine_bodies_q.resize(num_blocks_affine_body);
        sim_data->sa_affine_bodies_q_v.resize(num_blocks_affine_body);
        sim_data->sa_affine_bodies_q_tilde.resize(num_blocks_affine_body);
        sim_data->sa_affine_bodies_q_iter_start.resize(num_blocks_affine_body);
        sim_data->sa_affine_bodies_q_step_start.resize(num_blocks_affine_body);
        sim_data->sa_affine_bodies_q_outer.resize(num_blocks_affine_body);
        sim_data->sa_affine_bodies_q_v_outer.resize(num_blocks_affine_body);
        sim_data->sa_affine_bodies_volume.resize(num_blocks_affine_body);
        sim_data->sa_affine_bodies_kappa.resize(num_blocks_affine_body);

        sim_data->sa_affine_bodies_mass_matrix.resize(num_affine_bodies);
        sim_data->sa_affine_bodies_mass_matrix_full.resize(num_affine_bodies);

        sim_data->sa_affine_bodies_gradients.resize(num_affine_bodies * 4);
        sim_data->sa_affine_bodies_hessians.resize(num_affine_bodies * 16);

        sim_data->sa_vert_affine_bodies_id.resize(mesh_data->num_verts, -1u);

        CpuParallel::single_thread_for(
            0,
            num_affine_bodies,
            [&](const uint body_idx)
            {
                const uint  meshIdx   = affine_body_indices[body_idx];
                const auto& mesh_info = world_data[meshIdx];

                sim_data->sa_affine_bodies_mesh_id[body_idx] = meshIdx;
                sim_data->sa_affine_bodies[body_idx] = luisa::make_uint4(num_verts_soft + 4 * body_idx + 0,
                                                                         num_verts_soft + 4 * body_idx + 1,
                                                                         num_verts_soft + 4 * body_idx + 2,
                                                                         num_verts_soft + 4 * body_idx + 3);

                {
                    float3 init_translation = mesh_data->sa_rest_translate[meshIdx];
                    float3 init_rotation    = mesh_data->sa_rest_rotation[meshIdx];
                    // float3 init_scale = mesh_data->sa_rest_scale[meshIdx];
                    float3   init_scale = luisa::make_float3(1.0f);  // Since we use |AAT-I|
                    float4x4 init_transform_matrix =
                        lcs::make_model_matrix(init_translation, init_rotation, init_scale);
                    float4x3 rest_q = AffineBodyDynamics::extract_q_from_affine_matrix(init_transform_matrix);
                    sim_data->sa_affine_bodies_rest_q[4 * body_idx + 0] = rest_q[0];  // = init_transform_matrix[0].xyz()
                    sim_data->sa_affine_bodies_rest_q[4 * body_idx + 1] = rest_q[1];  // = init_transform_matrix[1].xyz()
                    sim_data->sa_affine_bodies_rest_q[4 * body_idx + 2] = rest_q[2];  // = init_transform_matrix[2].xyz()
                    sim_data->sa_affine_bodies_rest_q[4 * body_idx + 3] = rest_q[3];  // = init_transform_matrix[3].xyz()
                    // LUISA_INFO("Affine Body {} Rest q = \n{},\n{},\n{},\n{}",
                    //            body_idx,
                    //            rest_q[0],
                    //            rest_q[1],
                    //            rest_q[2],
                    //            rest_q[3]);
                    sim_data->sa_affine_bodies_rest_q_v[4 * body_idx + 0] = Zero3;
                    sim_data->sa_affine_bodies_rest_q_v[4 * body_idx + 1] = Zero3;
                    sim_data->sa_affine_bodies_rest_q_v[4 * body_idx + 2] = Zero3;
                    sim_data->sa_affine_bodies_rest_q_v[4 * body_idx + 3] = Zero3;
                    // LUISA_INFO("Affine Body {} Rest q = {}", body_idx, rest_q);
                }

                const uint curr_prefix    = mesh_data->prefix_num_verts[meshIdx];
                const uint next_prefix    = mesh_data->prefix_num_verts[meshIdx + 1];
                const uint num_verts_body = next_prefix - curr_prefix;

                EigenFloat12x12 body_mass = EigenFloat12x12::Zero();
                float4x4        compressed_mass_matrix;

                if (mesh_info.get_is_shell()
                    || (mesh_data->prefix_num_tets[meshIdx + 1] - mesh_data->prefix_num_tets[meshIdx]) > 0)
                {
                    // Shell or Tet mesh: integrate from vertices
                    CpuParallel::single_thread_for(curr_prefix,
                                                   next_prefix,
                                                   [&](const uint vid)
                                                   {
                                                       float mass = mesh_data->sa_vert_mass[vid];
                                                       float3 scaled_model_x = mesh_data->sa_scaled_model_x[vid];
                                                       auto J = AffineBodyDynamics::get_jacobian_dxdq(scaled_model_x);
                                                       // std::cout << "JtT of vert " << vid << " = \n" << J.transpose() * J << std::endl;
                                                       body_mass += mass * J.transpose() * J;
                                                   });
                }
                else  // Solid body: integrate from surface triangles
                {
                    float    rho = mesh_info.get_material<RigidMaterial>().density;
                    float    m;
                    float3   m_x_bar;
                    float3x3 m_x_bar_x_bar;

                    compute_trimesh_dyadic_mass(mesh_data->sa_scaled_model_x,
                                                mesh_data->sa_faces,
                                                mesh_data->prefix_num_faces[meshIdx],
                                                mesh_data->prefix_num_faces[meshIdx + 1],
                                                rho,
                                                m,
                                                m_x_bar,
                                                m_x_bar_x_bar);

                    EigenFloat3x3 I = EigenFloat3x3::Identity();
                    for (uint i = 0; i < 3; i++)
                        body_mass.block<3, 3>(3 + i * 3, 0) = body_mass.block<3, 3>(0, 3 + i * 3) =
                            m_x_bar[i] * I;
                    for (uint i = 0; i < 3; i++)
                        for (uint j = 0; j < 3; j++)
                            body_mass.block<3, 3>(3 + i * 3, 3 + j * 3) = m_x_bar_x_bar[i][j] * I;
                }

                body_mass.diagonal() = body_mass.diagonal().cwiseMax(Epsilon);

                for (uint i = 0; i < 4; i++)
                {
                    for (uint j = 0; j < 4; j++)
                    {
                        compressed_mass_matrix[j][i] = body_mass(i * 3 + 0, j * 3 + 0);
                    }
                }
                sim_data->sa_affine_bodies_mass_matrix[body_idx]      = compressed_mass_matrix;
                sim_data->sa_affine_bodies_mass_matrix_full[body_idx] = body_mass;

                // std::cout << "Mass Matrix = \n" << body_mass << std::endl;
                // LUISA_INFO("Affine Body {} Mass Matrix : ", body_idx);
                // LUISA_INFO("Affine Body {} Mass Matrix : {}", body_idx, compressed_mass_matrix[0]);
                // LUISA_INFO("Affine Body {} Mass Matrix : {}", body_idx, compressed_mass_matrix[1]);
                // LUISA_INFO("Affine Body {} Mass Matrix : {}", body_idx, compressed_mass_matrix[2]);
                // LUISA_INFO("Affine Body {} Mass Matrix : {}", body_idx, compressed_mass_matrix[3]);

                sim_data->sa_affine_bodies_is_fixed[body_idx] = false;
                sim_data->sa_affine_bodies_is_fixed[body_idx] =
                    std::any_of(mesh_data->sa_is_fixed.begin() + curr_prefix,
                                mesh_data->sa_is_fixed.begin() + next_prefix,
                                [](const bool is_fixed) { return is_fixed; });

                float area = std::reduce(mesh_data->sa_rest_vert_area.begin() + curr_prefix,
                                         mesh_data->sa_rest_vert_area.begin() + next_prefix,
                                         0.0f);

                sim_data->sa_affine_bodies_volume[body_idx] = mesh_data->sa_rest_body_volume[meshIdx];
                sim_data->sa_affine_bodies_kappa[body_idx] = mesh_info.get_material<RigidMaterial>().stiffness;

                EigenFloat12 gravity_sum = EigenFloat12::Zero();
                CpuParallel::single_thread_for(curr_prefix,
                                               next_prefix,
                                               [&](const uint vid)
                                               {
                                                   sim_data->sa_vert_affine_bodies_id[vid] = body_idx;
                                                   float  mass   = mesh_data->sa_vert_mass[vid];
                                                   float3 rest_x = mesh_data->sa_model_x[vid];
                                                   auto J = AffineBodyDynamics::get_jacobian_dxdq(rest_x);
                                                   gravity_sum +=
                                                       mass * J.transpose()
                                                       * float3_to_eigen3(luisa::make_float3(0, -9.8, 0));
                                               });  // / area_mass[1];

                EigenFloat12 body_gravity = body_mass.inverse() * gravity_sum;
                sim_data->sa_affine_bodies_gravity[4 * body_idx + 0] =
                    eigen3_to_float3(body_gravity.block<3, 1>(0, 0));
                sim_data->sa_affine_bodies_gravity[4 * body_idx + 1] =
                    eigen3_to_float3(body_gravity.block<3, 1>(3, 0));
                sim_data->sa_affine_bodies_gravity[4 * body_idx + 2] =
                    eigen3_to_float3(body_gravity.block<3, 1>(6, 0));
                sim_data->sa_affine_bodies_gravity[4 * body_idx + 3] =
                    eigen3_to_float3(body_gravity.block<3, 1>(9, 0));
                // LUISA_INFO("Affine body {} : Area = {}, Gravity = {}", body_idx, area, body_gravity);
            });

        CpuParallel::parallel_copy(sim_data->sa_affine_bodies_rest_q, sim_data->sa_affine_bodies_q_outer);
        CpuParallel::parallel_copy(sim_data->sa_affine_bodies_rest_q_v, sim_data->sa_affine_bodies_q_v_outer);
    }

    // Init Energy Adjacent List
    {
        // num_variables_in_system
        sim_data->vert_adj_material_force_verts.resize(num_dof);
        sim_data->vert_adj_stretch_springs.resize(num_dof);
        sim_data->vert_adj_stretch_faces.resize(num_dof);
        sim_data->vert_adj_bending_edges.resize(num_dof);
        sim_data->vert_adj_stress_tets.resize(num_dof);
        sim_data->vert_adj_affine_bodies.resize(num_dof);

        auto insert_adj_vert = [](std::vector<std::vector<uint>>& adj_map, const uint& vid1, const uint& vid2)
        {
            if (vid1 == vid2)
                std::cerr << "Try to build connection with self vertex";
            auto& inner_list  = adj_map[vid1];
            auto  find_result = std::find(inner_list.begin(), inner_list.end(), vid2);
            if (find_result == inner_list.end())
            {
                inner_list.push_back(vid2);
            }
        };

        // Vert adj faces
        for (uint fid = 0; fid < sim_data->sa_stretch_faces.size(); fid++)
        {
            auto face = sim_data->sa_stretch_faces[fid];

            for (uint j = 0; j < 3; j++)
            {
                sim_data->vert_adj_stretch_faces[face[j]].push_back(fid);
            }

            for (uint ii = 0; ii < 3; ii++)
            {
                for (uint jj = 0; jj < 3; jj++)
                {
                    if (ii != jj)
                    {
                        insert_adj_vert(sim_data->vert_adj_material_force_verts, face[ii], face[jj]);
                    }
                }
            }
        }
        upload_2d_csr_from(sim_data->sa_vert_adj_stretch_faces_csr, sim_data->vert_adj_stretch_faces);

        // Vert adj stretch springs
        for (uint eid = 0; eid < sim_data->sa_stretch_springs.size(); eid++)
        {
            auto edge = sim_data->sa_stretch_springs[eid];
            for (uint j = 0; j < 2; j++)
            {
                sim_data->vert_adj_stretch_springs[edge[j]].push_back(eid);
            }

            for (uint ii = 0; ii < 2; ii++)
            {
                for (uint jj = 0; jj < 2; jj++)
                {
                    if (ii != jj)
                    {
                        insert_adj_vert(sim_data->vert_adj_material_force_verts, edge[ii], edge[jj]);
                    }
                }
            }
        }
        upload_2d_csr_from(sim_data->sa_vert_adj_stretch_springs_csr, sim_data->vert_adj_stretch_springs);

        // Vert adj bending edges
        for (uint eid = 0; eid < sim_data->sa_bending_edges.size(); eid++)
        {
            auto edge = sim_data->sa_bending_edges[eid];
            for (uint j = 0; j < 4; j++)
            {
                sim_data->vert_adj_bending_edges[edge[j]].push_back(eid);
            }

            for (uint ii = 0; ii < 4; ii++)
            {
                for (uint jj = 0; jj < 4; jj++)
                {
                    if (ii != jj)
                    {
                        insert_adj_vert(sim_data->vert_adj_material_force_verts, edge[ii], edge[jj]);
                    }
                }
            }
        }
        upload_2d_csr_from(sim_data->sa_vert_adj_bending_edges_csr, sim_data->vert_adj_bending_edges);

        // Vert adj stress tets
        for (uint tid = 0; tid < sim_data->sa_stress_tets.size(); tid++)
        {
            auto tet = sim_data->sa_stress_tets[tid];
            for (uint j = 0; j < 4; j++)
            {
                sim_data->vert_adj_stress_tets[tet[j]].push_back(tid);
            }
            for (uint ii = 0; ii < 4; ii++)
            {
                for (uint jj = 0; jj < 4; jj++)
                {
                    if (ii != jj)
                    {
                        insert_adj_vert(sim_data->vert_adj_material_force_verts, tet[ii], tet[jj]);
                    }
                }
            }
        }
        upload_2d_csr_from(sim_data->sa_vert_adj_stress_tets_csr, sim_data->vert_adj_stress_tets);

        // Vert adj orthogonality energy
        for (uint body_idx = 0; body_idx < num_affine_bodies; body_idx++)
        {
            auto body = sim_data->sa_affine_bodies[body_idx];
            for (uint j = 0; j < 4; j++)
            {
                sim_data->vert_adj_affine_bodies[body[j]].push_back(body_idx);
            }

            for (uint ii = 0; ii < 4; ii++)
            {
                for (uint jj = 0; jj < 4; jj++)
                {
                    if (ii != jj)
                    {
                        insert_adj_vert(sim_data->vert_adj_material_force_verts, body[ii], body[jj]);
                    }
                }
            }
        }
        upload_2d_csr_from(sim_data->sa_vert_adj_affine_bodies_csr, sim_data->vert_adj_affine_bodies);

        // Sort adjacents
        CpuParallel::parallel_for(0,
                                  num_dof,
                                  [&](const uint vid)
                                  {
                                      std::vector<uint>& adj_list = sim_data->vert_adj_material_force_verts[vid];
                                      std::sort(adj_list.begin(), adj_list.end());
                                      if (adj_list.size() > 255)
                                          LUISA_ERROR("Adjacent count out of range {}");
                                  });
        upload_2d_csr_from(sim_data->sa_vert_adj_material_force_verts_csr, sim_data->vert_adj_material_force_verts);

        // Final off-diag hessian
        const uint orig_hessian_nnz = sim_data->sa_vert_adj_material_force_verts_csr.size();  //  - mesh_data->num_verts - 1

        constexpr bool use_block_scan = true;
        constexpr bool use_warp_scan  = false;
        constexpr uint segment_size   = use_block_scan ? 256 : use_warp_scan ? 32 : 1;
        const uint     alinged_nnz    = segment_size == 1 ?
                                            orig_hessian_nnz :
                                            (orig_hessian_nnz + segment_size - 1) / segment_size * segment_size;
        // const uint alinged_nnz = orig_hessian_nnz;
        sim_data->sa_cgA_fixtopo_offdiag_triplet.resize(alinged_nnz);
        sim_data->sa_cgA_fixtopo_offdiag_triplet_info.resize(alinged_nnz, make_matrix_triplet_info(-1u, -1u, 0));

        CpuParallel::parallel_for(
            0,
            num_dof,
            [&](const uint vid)
            {
                const uint curr_prefix = sim_data->sa_vert_adj_material_force_verts_csr[vid];
                const uint next_prefix = sim_data->sa_vert_adj_material_force_verts_csr[vid + 1];
                for (uint idx = curr_prefix; idx < next_prefix; idx++)  // Outer-of-range part will set to zero
                {
                    const uint adj_vid = sim_data->sa_vert_adj_material_force_verts_csr[idx];
                    const uint triplet_property =
                        MatrixTriplet::make_triplet_property_in_block(idx, curr_prefix, next_prefix - 1);

                    // LUISA_INFO("Hessian Triplet ({}, {}) at idx = {}, property = {:16b}", vid, adj_vid, idx, triplet_property);
                    sim_data->sa_cgA_fixtopo_offdiag_triplet_info[idx] =
                        make_matrix_triplet_info(vid, adj_vid, triplet_property);
                }
            });

        // Check
        if constexpr (use_block_scan)
        {
            std::vector<ushort> is_tirplet_accessed(alinged_nnz, false);
            CpuParallel::single_thread_for(
                0,
                alinged_nnz / 256,
                [&](const uint blockIdx)
                {
                    const uint blockPrefix = blockIdx * 256;
                    const uint blockEnd    = blockPrefix + 256;

                    uint              last_start = -1u;
                    uint              last_end   = -1u;
                    std::vector<bool> cache_used(32, false);
                    for (uint idx = blockPrefix; idx < blockEnd; idx++)
                    {
                        auto& info     = sim_data->sa_cgA_fixtopo_offdiag_triplet_info[idx];
                        auto  property = info[2];
                        if (MatrixTriplet::is_first_col_in_row(property))
                        {
                            if (last_end != -1u || last_start != -1u)
                            {
                                LUISA_ERROR("Last segment not closed");
                            }
                            last_start = idx;
                        }
                        if (MatrixTriplet::is_last_col_in_row(property))
                        {
                            if (last_start == -1u || last_end != -1u)
                            {
                                LUISA_ERROR("Last segment not started");
                            }
                            last_end = idx;
                        }
                        if (last_start != -1u && last_end != -1u)
                        {
                            // if (info[0] < 10)
                            // {
                            //     LUISA_INFO("Vert {}'s triplet (threadIdx from {:3} to {:3}), from {} to {} , same warp ? {}, try to find {} ({})",
                            //                info[0],
                            //                last_start % 256,
                            //                last_end % 256,
                            //                last_start,
                            //                last_end,
                            //                MatrixTriplet::is_first_and_last_col_in_same_warp(info[2]),
                            //                blockPrefix + MatrixTriplet::read_first_col_threadIdx(info[2]),
                            //                MatrixTriplet::read_first_col_threadIdx(info[2]));
                            // }
                            for (uint i = last_start; i <= last_end; i++)
                            {
                                if (is_tirplet_accessed[i])
                                {
                                    LUISA_ERROR("Triplet {} has been accessed", i);
                                }
                                is_tirplet_accessed[i] = true;
                            }
                            const uint start_threadIdx = last_start % 256;
                            const uint start_warpIdx   = start_threadIdx / 32;
                            const uint end_threadIdx   = last_end % 256;
                            const uint end_warpIdx     = end_threadIdx / 32;

                            uint3 info_start = sim_data->sa_cgA_fixtopo_offdiag_triplet_info[last_start];
                            uint3 info_end   = sim_data->sa_cgA_fixtopo_offdiag_triplet_info[last_end];

                            if (info_start[0] != info_end[0])
                            {
                                LUISA_ERROR("RowIdx not match");
                            }
                            const uint vert_prefix =
                                sim_data->sa_vert_adj_material_force_verts_csr[info_start[0]];
                            const uint vert_suffix =
                                sim_data->sa_vert_adj_material_force_verts_csr[info_start[0] + 1];
                            if (MatrixTriplet::is_first_and_last_col_in_same_warp(info_start[2])
                                != MatrixTriplet::is_first_and_last_col_in_same_warp(info_end[2]))
                            {
                                LUISA_ERROR("Reduce mode not match : {} - {} : (Triplet Range {} - {}) startTriplet = {} (threadIdx = {}, warpIdx = {}), endTriplet = {} (threadIdx = {}, warpIdx = {})",
                                            MatrixTriplet::is_first_and_last_col_in_same_warp(info_start[2]),
                                            MatrixTriplet::is_first_and_last_col_in_same_warp(info_end[2]),
                                            vert_prefix,
                                            vert_suffix - 1,
                                            last_start,
                                            start_threadIdx,
                                            start_warpIdx,
                                            last_end,
                                            end_threadIdx,
                                            end_warpIdx);
                            }
                            bool in_same_warp = MatrixTriplet::is_first_and_last_col_in_same_warp(info_start[2]);
                            if (in_same_warp)
                            {
                                if (last_start / 32 != last_end / 32)
                                {
                                    LUISA_ERROR("Not in the same warp");
                                }
                                const uint start_laneIdx = last_start % 32;
                                const uint desire_laneIdx = MatrixTriplet::read_first_col_info(info_end[2]);
                                if (start_laneIdx != desire_laneIdx)
                                {
                                    LUISA_ERROR("LaneIdx not match!!");
                                }
                            }
                            else
                            {
                                if (last_start / 32 == last_end / 32)
                                {
                                    LUISA_ERROR("In the same warp");
                                }
                                if (cache_used[start_warpIdx])
                                {
                                    LUISA_ERROR("Cache has been used");
                                }
                                const uint desire_warpIdx = MatrixTriplet::read_first_col_info(info_end[2]);
                                if (start_warpIdx != desire_warpIdx)
                                {
                                    LUISA_ERROR("WarpIdx not match: startTriplet = {} (threadIdx = {}, warpIdx = {}), endTriplet = {} (threadIdx = {}, Desire for {})",
                                                last_start,
                                                start_threadIdx,
                                                start_warpIdx,
                                                last_end,
                                                last_end % 256,
                                                desire_warpIdx);
                                }
                                cache_used[start_warpIdx] = true;
                            }
                            last_start = -1u;
                            last_end   = -1u;
                        }
                    }
                });
            for (uint triplet_idx = 0; triplet_idx < orig_hessian_nnz; triplet_idx++)
            {
                auto triplet_info = sim_data->sa_cgA_fixtopo_offdiag_triplet_info[triplet_idx];
                if (MatrixTriplet::is_valid(triplet_info[2]))
                {
                    if (!is_tirplet_accessed[triplet_idx])
                    {
                        LUISA_ERROR("Triplet {} has not been accessed", triplet_idx);
                    }
                }
            }
        }
    }

    // Find material-force-offset
    {
        const std::vector<std::vector<uint>>& reference_adj_list = sim_data->vert_adj_material_force_verts;

        // Spring energy
        sim_data->sa_stretch_springs_offsets_in_adjlist.resize(sim_data->sa_stretch_springs.size() * 2);
        CpuParallel::parallel_for(0,
                                  sim_data->sa_stretch_springs.size(),
                                  [&](const uint eid)
                                  {
                                      auto edge = sim_data->sa_stretch_springs[eid];
                                      auto mask = get_offsets_in_adjlist_from_adjacent_list<2>(reference_adj_list,
                                                                                               edge);  // size = 2
                                      std::memcpy(sim_data->sa_stretch_springs_offsets_in_adjlist.data() + eid * 2,
                                                  mask.data(),
                                                  sizeof(ushort) * 2);
                                  });
        for (uint i = 0; i < sim_data->sa_stretch_springs_offsets_in_adjlist.size(); i++)
        {
            auto adj_offset = sim_data->sa_stretch_springs_offsets_in_adjlist[i];
        }

        // Stretch face energy
        sim_data->sa_stretch_faces_offsets_in_adjlist.resize(sim_data->sa_stretch_faces.size() * 6);
        CpuParallel::parallel_for(0,
                                  sim_data->sa_stretch_faces.size(),
                                  [&](const uint fid)
                                  {
                                      auto face = sim_data->sa_stretch_faces[fid];
                                      auto mask = get_offsets_in_adjlist_from_adjacent_list<3>(reference_adj_list,
                                                                                               face);  // size = 6
                                      std::memcpy(sim_data->sa_stretch_faces_offsets_in_adjlist.data() + fid * 6,
                                                  mask.data(),
                                                  sizeof(ushort) * 6);
                                  });

        // Bending angle energy
        sim_data->sa_bending_edges_offsets_in_adjlist.resize(sim_data->sa_bending_edges.size() * 12);
        CpuParallel::parallel_for(0,
                                  sim_data->sa_bending_edges.size(),
                                  [&](const uint eid)
                                  {
                                      auto edge = sim_data->sa_bending_edges[eid];
                                      auto mask = get_offsets_in_adjlist_from_adjacent_list<4>(reference_adj_list,
                                                                                               edge);  // size = 12
                                      std::memcpy(sim_data->sa_bending_edges_offsets_in_adjlist.data() + eid * 12,
                                                  mask.data(),
                                                  sizeof(ushort) * 12);
                                  });

        // Stress tetrahedron energy
        sim_data->sa_stress_tets_offsets_in_adjlist.resize(sim_data->sa_stress_tets.size() * 12);
        CpuParallel::parallel_for(0,
                                  sim_data->sa_stress_tets.size(),
                                  [&](const uint tid)
                                  {
                                      auto tet = sim_data->sa_stress_tets[tid];
                                      auto mask = get_offsets_in_adjlist_from_adjacent_list<4>(reference_adj_list,
                                                                                               tet);  // size = 12
                                      std::memcpy(sim_data->sa_stress_tets_offsets_in_adjlist.data() + tid * 12,
                                                  mask.data(),
                                                  sizeof(ushort) * 12);
                                  });

        // Affine body inertia and orthogonality energy
        sim_data->sa_affine_bodies_offsets_in_adjlist.resize(num_affine_bodies * 12);
        CpuParallel::parallel_for(0,
                                  sim_data->sa_affine_bodies.size(),
                                  [&](const uint body_idx)
                                  {
                                      auto body = sim_data->sa_affine_bodies[body_idx];
                                      auto mask = get_offsets_in_adjlist_from_adjacent_list<4>(reference_adj_list,
                                                                                               body);  // size = 12
                                      std::memcpy(sim_data->sa_affine_bodies_offsets_in_adjlist.data() + body_idx * 12,
                                                  mask.data(),
                                                  sizeof(ushort) * 12);
                                  });
    }

    // Constraint Graph Coloring
    std::vector<std::vector<uint>> tmp_clusterd_constraint_stretch_mass_spring;
    std::vector<std::vector<uint>> tmp_clusterd_constraint_bending;
    auto*                          colored_data = &sim_data->colored_data;
    {
        fn_graph_coloring_per_constraint("Distance  Spring Constraint",
                                         tmp_clusterd_constraint_stretch_mass_spring,
                                         sim_data->vert_adj_stretch_springs,
                                         sim_data->sa_stretch_springs,
                                         2);

        fn_graph_coloring_per_constraint("Bending   Angle  Constraint",
                                         tmp_clusterd_constraint_bending,
                                         sim_data->vert_adj_bending_edges,
                                         sim_data->sa_bending_edges,
                                         4);

        colored_data->num_clusters_springs       = tmp_clusterd_constraint_stretch_mass_spring.size();
        colored_data->num_clusters_bending_edges = tmp_clusterd_constraint_bending.size();

        fn_get_prefix(colored_data->sa_prefix_merged_springs, tmp_clusterd_constraint_stretch_mass_spring);
        fn_get_prefix(colored_data->sa_prefix_merged_bending_edges, tmp_clusterd_constraint_bending);

        upload_2d_csr_from(colored_data->sa_clusterd_springs, tmp_clusterd_constraint_stretch_mass_spring);
        upload_2d_csr_from(colored_data->sa_clusterd_bending_edges, tmp_clusterd_constraint_bending);
    }

    // Vertex Block Descent Coloring
    {
        // Graph Coloring
        const uint num_verts_total = num_dof;
        sim_data->sa_Hf.resize(num_dof * 12);
        sim_data->sa_Hf1.resize(num_dof);

        const std::vector<std::vector<uint>>& vert_adj_verts = sim_data->vert_adj_material_force_verts;
        std::vector<std::vector<uint>>        clusterd_vertices_bending;
        std::vector<uint>                     prefix_vertices_bending;

        fn_graph_coloring_per_vertex(vert_adj_verts, clusterd_vertices_bending, prefix_vertices_bending);
        colored_data->num_clusters_per_vertex_with_material_constraints = clusterd_vertices_bending.size();
        upload_from(colored_data->prefix_per_vertex_with_material_constraints, prefix_vertices_bending);
        upload_2d_csr_from(colored_data->clusterd_per_vertex_with_material_constraints, clusterd_vertices_bending);

        // Reverse map
        colored_data->per_vertex_bending_cluster_id.resize(num_dof);
        for (uint cluster = 0; cluster < colored_data->num_clusters_per_vertex_with_material_constraints; cluster++)
        {
            const uint next_prefix = colored_data->clusterd_per_vertex_with_material_constraints[cluster + 1];
            const uint curr_prefix = colored_data->clusterd_per_vertex_with_material_constraints[cluster];
            const uint num_verts_cluster = next_prefix - curr_prefix;
            CpuParallel::parallel_for(0,
                                      num_verts_cluster,
                                      [&](const uint i)
                                      {
                                          const uint vid =
                                              colored_data->clusterd_per_vertex_with_material_constraints[curr_prefix + i];
                                          colored_data->per_vertex_bending_cluster_id[vid] = cluster;
                                      });
        }
    }

    // Colored contraint precomputation
    {
        // Spring Constraint
        {
            colored_data->sa_merged_stretch_springs.resize(num_stretch_springs);
            colored_data->sa_merged_stretch_spring_rest_length.resize(num_stretch_springs);
            colored_data->sa_lambda_stretch_mass_spring.resize(num_stretch_springs);

            uint prefix = 0;
            for (uint cluster = 0; cluster < tmp_clusterd_constraint_stretch_mass_spring.size(); cluster++)
            {
                const auto& curr_cluster = tmp_clusterd_constraint_stretch_mass_spring[cluster];
                CpuParallel::parallel_for(0,
                                          curr_cluster.size(),
                                          [&](const uint i)
                                          {
                                              const uint eid = curr_cluster[i];
                                              {
                                                  colored_data->sa_merged_stretch_springs[prefix + i] =
                                                      sim_data->sa_stretch_springs[eid];
                                                  colored_data->sa_merged_stretch_spring_rest_length[prefix + i] =
                                                      sim_data->sa_stretch_spring_rest_state_length[eid];
                                              }
                                          });
                prefix += curr_cluster.size();
            }
            if (prefix != sim_data->sa_stretch_springs.size())
                LUISA_ERROR("Sum of Mass Spring Cluster Is Not Equal  Than Orig");
        }

        // Bending Constraint
        {
            colored_data->sa_merged_bending_edges.resize(num_bending_edges);
            colored_data->sa_merged_bending_edges_angle.resize(num_bending_edges);
            colored_data->sa_merged_bending_edges_Q.resize(num_bending_edges);
            colored_data->sa_lambda_bending.resize(num_bending_edges);

            uint prefix = 0;
            for (uint cluster = 0; cluster < tmp_clusterd_constraint_bending.size(); cluster++)
            {
                const auto& curr_cluster = tmp_clusterd_constraint_bending[cluster];
                CpuParallel::parallel_for(0,
                                          curr_cluster.size(),
                                          [&](const uint i)
                                          {
                                              const uint eid = curr_cluster[i];
                                              {
                                                  colored_data->sa_merged_bending_edges[prefix + i] =
                                                      sim_data->sa_bending_edges[eid];
                                                  colored_data->sa_merged_bending_edges_angle[prefix + i] =
                                                      sim_data->sa_bending_edges_rest_angle[eid];
                                                  colored_data->sa_merged_bending_edges_Q[prefix + i] =
                                                      sim_data->sa_bending_edges_Q[eid];
                                              }
                                          });
                prefix += curr_cluster.size();
            }
            if (prefix != sim_data->sa_bending_edges.size())
                LUISA_ERROR("Sum of Bending Cluster Is Not Equal Than Orig");
        }
    }
}

void upload_sim_buffers(luisa::compute::Device&                      device,
                        luisa::compute::Stream&                      stream,
                        lcs::SimulationData<std::vector>*            input_data,
                        lcs::SimulationData<luisa::compute::Buffer>* output_data)
{
    output_data->num_dof                           = input_data->num_dof;
    output_data->num_affine_bodies                 = input_data->num_affine_bodies;
    output_data->num_verts_rigid                   = input_data->num_verts_rigid;
    output_data->num_verts_soft                    = input_data->num_verts_soft;
    output_data->colored_data.num_clusters_springs = input_data->colored_data.num_clusters_springs;
    output_data->colored_data.num_clusters_bending_edges = input_data->colored_data.num_clusters_bending_edges;
    output_data->colored_data.num_clusters_per_vertex_with_material_constraints =
        input_data->colored_data.num_clusters_per_vertex_with_material_constraints;

    stream << upload_buffer(device, output_data->sa_num_dof, input_data->sa_num_dof)
           << upload_buffer(device, output_data->sa_x_tilde, input_data->sa_x_tilde)
           << upload_buffer(device, output_data->sa_x, input_data->sa_x)
           << upload_buffer(device, output_data->sa_v, input_data->sa_v)
           << upload_buffer(device, output_data->sa_x_step_start, input_data->sa_x_step_start)
           << upload_buffer(device, output_data->sa_x_iter_start, input_data->sa_x_iter_start)

           << upload_buffer(device, output_data->sa_system_energy, input_data->sa_system_energy);

    stream << upload_buffer(device, output_data->sa_target_positions, input_data->sa_target_positions);

    if (input_data->sa_stretch_springs.size() > 0)
    {
        stream
            << upload_buffer(device, output_data->sa_stretch_springs, input_data->sa_stretch_springs)
            << upload_buffer(device, output_data->sa_stretch_spring_rest_state_length, input_data->sa_stretch_spring_rest_state_length)
            << upload_buffer(device, output_data->sa_stretch_spring_stiffness, input_data->sa_stretch_spring_stiffness)
            << upload_buffer(device, output_data->sa_stretch_springs_offsets_in_adjlist, input_data->sa_stretch_springs_offsets_in_adjlist)
            << upload_buffer(device, output_data->sa_stretch_springs_gradients, input_data->sa_stretch_springs_gradients)
            << upload_buffer(device, output_data->sa_stretch_springs_hessians, input_data->sa_stretch_springs_hessians)

            << upload_buffer(device,
                             output_data->colored_data.sa_merged_stretch_springs,
                             input_data->colored_data.sa_merged_stretch_springs)
            << upload_buffer(device,
                             output_data->colored_data.sa_merged_stretch_spring_rest_length,
                             input_data->colored_data.sa_merged_stretch_spring_rest_length)

            << upload_buffer(device,
                             output_data->colored_data.sa_clusterd_springs,
                             input_data->colored_data.sa_clusterd_springs)
            << upload_buffer(device,
                             output_data->colored_data.sa_prefix_merged_springs,
                             input_data->colored_data.sa_prefix_merged_springs)
            << upload_buffer(device,
                             output_data->colored_data.sa_lambda_stretch_mass_spring,
                             input_data->colored_data.sa_lambda_stretch_mass_spring);  // just resize
    }
    if (input_data->sa_stretch_faces.size() > 0)
    {
        stream
            << upload_buffer(device, output_data->sa_stretch_faces, input_data->sa_stretch_faces)
            << upload_buffer(device, output_data->sa_stretch_faces_mu_lambda, input_data->sa_stretch_faces_mu_lambda)
            << upload_buffer(device, output_data->sa_stretch_faces_rest_area, input_data->sa_stretch_faces_rest_area)
            << upload_buffer(device, output_data->sa_stretch_faces_Dm_inv, input_data->sa_stretch_faces_Dm_inv)
            << upload_buffer(device, output_data->sa_stretch_faces_offsets_in_adjlist, input_data->sa_stretch_faces_offsets_in_adjlist)
            << upload_buffer(device, output_data->sa_stretch_faces_gradients, input_data->sa_stretch_faces_gradients)
            << upload_buffer(device, output_data->sa_stretch_faces_hessians, input_data->sa_stretch_faces_hessians);
    }
    if (input_data->sa_bending_edges.size() > 0)
    {
        stream
            << upload_buffer(device, output_data->sa_bending_edges, input_data->sa_bending_edges)
            << upload_buffer(device, output_data->sa_bending_edges_rest_area, input_data->sa_bending_edges_rest_area)
            << upload_buffer(device, output_data->sa_bending_edges_rest_angle, input_data->sa_bending_edges_rest_angle)
            << upload_buffer(device, output_data->sa_bending_edges_stiffness, input_data->sa_bending_edges_stiffness)
            << upload_buffer(device, output_data->sa_bending_edges_Q, input_data->sa_bending_edges_Q)
            << upload_buffer(device, output_data->sa_bending_edges_offsets_in_adjlist, input_data->sa_bending_edges_offsets_in_adjlist)
            << upload_buffer(device, output_data->sa_bending_edges_gradients, input_data->sa_bending_edges_gradients)
            << upload_buffer(device, output_data->sa_bending_edges_hessians, input_data->sa_bending_edges_hessians)

            << upload_buffer(device,
                             output_data->colored_data.sa_merged_bending_edges,
                             input_data->colored_data.sa_merged_bending_edges)
            << upload_buffer(device,
                             output_data->colored_data.sa_merged_bending_edges_angle,
                             input_data->colored_data.sa_merged_bending_edges_angle)
            << upload_buffer(device,
                             output_data->colored_data.sa_merged_bending_edges_Q,
                             input_data->colored_data.sa_merged_bending_edges_Q)

            << upload_buffer(device,
                             output_data->colored_data.sa_clusterd_bending_edges,
                             input_data->colored_data.sa_clusterd_bending_edges)
            << upload_buffer(device,
                             output_data->colored_data.sa_prefix_merged_bending_edges,
                             input_data->colored_data.sa_prefix_merged_bending_edges)
            << upload_buffer(device,
                             output_data->colored_data.sa_lambda_bending,
                             input_data->colored_data.sa_lambda_bending)  // just resize
            ;
    }
    if (input_data->sa_stress_tets.size() > 0)
    {
        stream
            << upload_buffer(device, output_data->sa_stress_tets, input_data->sa_stress_tets)
            << upload_buffer(device, output_data->sa_stress_tets_mu_lambda, input_data->sa_stress_tets_mu_lambda)
            << upload_buffer(device, output_data->sa_stress_tets_rest_volume, input_data->sa_stress_tets_rest_volume)
            << upload_buffer(device, output_data->sa_stress_tets_Dm_inv, input_data->sa_stress_tets_Dm_inv)
            << upload_buffer(device, output_data->sa_stress_tets_offsets_in_adjlist, input_data->sa_stress_tets_offsets_in_adjlist)
            << upload_buffer(device, output_data->sa_stress_tets_gradients, input_data->sa_stress_tets_gradients)
            << upload_buffer(device, output_data->sa_stress_tets_hessians, input_data->sa_stress_tets_hessians);
    }
    if (input_data->sa_affine_bodies.size() > 0)
    {
        stream
            << upload_buffer(device, output_data->sa_affine_bodies, input_data->sa_affine_bodies)
            << upload_buffer(device, output_data->sa_affine_bodies_mesh_id, input_data->sa_affine_bodies_mesh_id)
            << upload_buffer(device, output_data->sa_affine_bodies_is_fixed, input_data->sa_affine_bodies_is_fixed)
            << upload_buffer(device, output_data->sa_affine_bodies_rest_q, input_data->sa_affine_bodies_rest_q)
            << upload_buffer(device, output_data->sa_affine_bodies_gravity, input_data->sa_affine_bodies_gravity)
            << upload_buffer(device, output_data->sa_affine_bodies_q, input_data->sa_affine_bodies_q)
            << upload_buffer(device, output_data->sa_affine_bodies_q_v, input_data->sa_affine_bodies_q_v)
            << upload_buffer(device, output_data->sa_affine_bodies_q_tilde, input_data->sa_affine_bodies_q_tilde)
            << upload_buffer(device, output_data->sa_affine_bodies_q_iter_start, input_data->sa_affine_bodies_q_iter_start)
            << upload_buffer(device, output_data->sa_affine_bodies_q_step_start, input_data->sa_affine_bodies_q_step_start)
            << upload_buffer(device, output_data->sa_affine_bodies_volume, input_data->sa_affine_bodies_volume)
            << upload_buffer(device, output_data->sa_affine_bodies_kappa, input_data->sa_affine_bodies_kappa)
            << upload_buffer(device, output_data->sa_affine_bodies_mass_matrix, input_data->sa_affine_bodies_mass_matrix)
            << upload_buffer(device, output_data->sa_affine_bodies_gradients, input_data->sa_affine_bodies_gradients)
            << upload_buffer(device, output_data->sa_affine_bodies_hessians, input_data->sa_affine_bodies_hessians)
            << upload_buffer(device, output_data->sa_affine_bodies_offsets_in_adjlist, input_data->sa_affine_bodies_offsets_in_adjlist);
    }
    stream << upload_buffer(device,
                            output_data->sa_vert_affine_bodies_id,
                            input_data->sa_vert_affine_bodies_id);  // Basic information

    stream
        << upload_buffer(device, output_data->sa_cgA_fixtopo_offdiag_triplet, input_data->sa_cgA_fixtopo_offdiag_triplet)
        << upload_buffer(device, output_data->sa_cgA_fixtopo_offdiag_triplet_info, input_data->sa_cgA_fixtopo_offdiag_triplet_info)
        << upload_buffer(device, output_data->sa_vert_adj_material_force_verts_csr, input_data->sa_vert_adj_material_force_verts_csr)
        << upload_buffer(device, output_data->sa_vert_adj_stretch_springs_csr, input_data->sa_vert_adj_stretch_springs_csr)
        << upload_buffer(device, output_data->sa_vert_adj_stretch_faces_csr, input_data->sa_vert_adj_stretch_faces_csr)
        << upload_buffer(device, output_data->sa_vert_adj_bending_edges_csr, input_data->sa_vert_adj_bending_edges_csr)
        << upload_buffer(device, output_data->sa_vert_adj_affine_bodies_csr, input_data->sa_vert_adj_affine_bodies_csr)

        << upload_buffer(device,
                         output_data->colored_data.prefix_per_vertex_with_material_constraints,
                         input_data->colored_data.prefix_per_vertex_with_material_constraints)
        << upload_buffer(device,
                         output_data->colored_data.clusterd_per_vertex_with_material_constraints,
                         input_data->colored_data.clusterd_per_vertex_with_material_constraints)

        << upload_buffer(device,
                         output_data->colored_data.per_vertex_bending_cluster_id,
                         input_data->colored_data.per_vertex_bending_cluster_id)
        << upload_buffer(device, output_data->sa_Hf, input_data->sa_Hf)
        << upload_buffer(device, output_data->sa_Hf1, input_data->sa_Hf1) << luisa::compute::synchronize();
}

void resize_pcg_data(luisa::compute::Device&                      device,
                     luisa::compute::Stream&                      stream,
                     lcs::MeshData<std::vector>*                  mesh_data,
                     lcs::SimulationData<std::vector>*            host_data,
                     lcs::SimulationData<luisa::compute::Buffer>* device_data)
{
    const uint num_springs       = host_data->sa_stretch_springs.size();
    const uint num_bending_edges = host_data->sa_bending_edges.size();
    const uint num_faces         = host_data->sa_stretch_faces.size();
    const uint num_affine_bodies = host_data->num_affine_bodies;
    const uint num_verts         = host_data->num_dof;

    // const uint off_diag_count = std::max(uint(device_data->sa_hessian_pairs.size()), num_springs * 2);

    resize_buffer(host_data->sa_cgX, num_verts);
    resize_buffer(host_data->sa_cgB, num_verts);
    resize_buffer(host_data->sa_cgA_diag, num_verts);

    resize_buffer(host_data->sa_cgMinv, num_verts);
    resize_buffer(host_data->sa_cgP, num_verts);
    resize_buffer(host_data->sa_cgQ, num_verts);
    resize_buffer(host_data->sa_cgR, num_verts);
    resize_buffer(host_data->sa_cgZ, num_verts);
    resize_buffer(host_data->sa_block_result, num_verts);
    resize_buffer(host_data->sa_convergence, 10240);

    resize_buffer(device, device_data->sa_cgX, num_verts);
    resize_buffer(device, device_data->sa_cgB, num_verts);
    resize_buffer(device, device_data->sa_cgA_diag, num_verts);
    resize_buffer(device, device_data->sa_cgMinv, num_verts);
    resize_buffer(device, device_data->sa_cgP, num_verts);
    resize_buffer(device, device_data->sa_cgQ, num_verts);
    resize_buffer(device, device_data->sa_cgR, num_verts);
    resize_buffer(device, device_data->sa_cgZ, num_verts);
    resize_buffer(device, device_data->sa_block_result, num_verts);
    resize_buffer(device, device_data->sa_convergence, 10240);
}

}  // namespace lcs::Initializer