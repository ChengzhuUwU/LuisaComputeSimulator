#include "Initializer/init_xpbd_data.h"
#include "Core/affine_position.h"
#include "Core/float_n.h"
#include "Core/float_nxn.h"
#include "Core/lc_to_eigen.h"
#include "Energy/bending_energy.h"
#include "Initializer/init_mesh_data.h"
#include "MeshOperation/mesh_reader.h"
#include "Initializer/initializer_utils.h"
#include "luisa/core/logging.h"
#include "luisa/core/mathematics.h"


namespace lcs::Initializer
{

void init_xpbd_data(lcs::MeshData<std::vector>* mesh_data, lcs::SimulationData<std::vector>* xpbd_data)
{
    xpbd_data->sa_x_tilde.resize(mesh_data->num_verts); 
    xpbd_data->sa_x.resize(mesh_data->num_verts);
    xpbd_data->sa_v.resize(mesh_data->num_verts);       CpuParallel::parallel_copy(mesh_data->sa_rest_v, xpbd_data->sa_v);
    xpbd_data->sa_v_step_start.resize(mesh_data->num_verts); CpuParallel::parallel_copy(mesh_data->sa_rest_v, xpbd_data->sa_v_step_start);
    xpbd_data->sa_x_step_start.resize(mesh_data->num_verts);
    xpbd_data->sa_x_iter_start.resize(mesh_data->num_verts);

    // Count for stretch springs, stretch faces, bending edges
    std::vector<uint> stretch_spring_indices(mesh_data->num_edges, -1u); uint num_stretch_springs = 0;
    std::vector<uint> stretch_face_indices(mesh_data->num_faces, -1u); uint num_stretch_faces = 0;
    std::vector<uint> bending_edge_indices(mesh_data->num_dihedral_edges, -1u); uint num_bending_edges = 0;

    // Calculate number of energy element
    constexpr bool cull_unused_constraints = true;
    CpuParallel::parallel_for_and_scan(0, mesh_data->num_edges, [&](const uint eid)
    {
        uint2 edge = mesh_data->sa_edges[eid];
        bool is_cloth = mesh_data->sa_vert_mesh_type[edge[0]] == uint(ShellTypeCloth);
        bool is_dynamic = cull_unused_constraints ? !mesh_data->sa_is_fixed[edge[0]] || !mesh_data->sa_is_fixed[edge[1]] : true;
        return (is_cloth && is_dynamic) ? 1 : 0;
    }, [&](const uint eid, const uint global_prefix, const uint parallel_result)
    {
        if (parallel_result == 1) stretch_spring_indices[global_prefix - 1] = eid;
        if (eid == mesh_data->num_edges - 1) num_stretch_springs = global_prefix;
    }, 0);

    CpuParallel::parallel_for_and_scan(0, mesh_data->num_faces, [&](const uint fid)
    {
        uint3 face = mesh_data->sa_faces[fid];
        bool is_cloth = mesh_data->sa_vert_mesh_type[face[0]] == uint(ShellTypeCloth);
        bool is_dynamic = cull_unused_constraints ? 
            !mesh_data->sa_is_fixed[face[0]] || 
            !mesh_data->sa_is_fixed[face[1]] || 
            !mesh_data->sa_is_fixed[face[2]] : true;
        return (is_cloth && is_dynamic) ? 1 : 0;
    }, [&](const uint fid, const uint global_prefix, const uint parallel_result)
    {
        if (parallel_result == 1) stretch_face_indices[global_prefix - 1] = fid;
        if (fid == mesh_data->num_faces - 1) num_stretch_faces = global_prefix;
    }, 0);

    CpuParallel::parallel_for_and_scan(0, mesh_data->num_dihedral_edges, [&](const uint eid)
    {
        uint4 edge = mesh_data->sa_dihedral_edges[eid];
        bool is_cloth = mesh_data->sa_vert_mesh_type[edge[0]] == uint(ShellTypeCloth);
        bool is_dynamic = cull_unused_constraints ?
            !mesh_data->sa_is_fixed[edge[0]] || 
            !mesh_data->sa_is_fixed[edge[1]] || 
            !mesh_data->sa_is_fixed[edge[2]] || 
            !mesh_data->sa_is_fixed[edge[3]] : true;
        return (is_cloth && is_dynamic) ? 1 : 0;
    }, [&](const uint eid, const uint global_prefix, const uint parallel_result)
    {
        if (parallel_result == 1) bending_edge_indices[global_prefix - 1] = eid;
        if (eid == mesh_data->num_dihedral_edges - 1) num_bending_edges = global_prefix;
    }, 0);

    LUISA_INFO("num_stretch_springs = {} (<-{}), num_stretch_faces = {}(<-{}), num_bending_edges = {}(<-{})", 
        num_stretch_springs, mesh_data->num_edges, num_stretch_faces, mesh_data->num_faces, num_bending_edges, mesh_data->num_dihedral_edges);

    const uint num_verts_soft = CpuParallel::parallel_for_and_reduce_sum<uint>(0, mesh_data->num_verts, [&](const uint vid)
    {
        return mesh_data->sa_vert_mesh_type[vid] == ShellTypeRigid ? 0 : 1;
    }); 

    std::vector<uint> affine_body_indices(mesh_data->num_meshes, -1u); uint num_affine_bodies = 0;
    CpuParallel::parallel_for_and_scan(0, mesh_data->num_meshes, [&](const uint meshIdx)
    {
        const uint curr_prefix = mesh_data->prefix_num_verts[meshIdx];
        const uint first_vid = curr_prefix;
        const bool has_boundary_edge = // Unclosed
            (mesh_data->prefix_num_edges[meshIdx + 1] - mesh_data->prefix_num_edges[meshIdx]) !=
            (mesh_data->prefix_num_dihedral_edges[meshIdx + 1] - mesh_data->prefix_num_dihedral_edges[meshIdx]);
        // bool has_dynamic_vert = mesh_data->sa_is_fixed[first_vid];
        bool is_rigid = (mesh_data->sa_vert_mesh_type[first_vid] == uint(ShellTypeRigid)) ;// ;&& !has_boundary_edge;
        return (is_rigid) ? 1 : 0; // has_dynamic_vert
    }, [&](const uint meshIdx, const uint global_prefix, const uint parallel_result)
    {
        if (parallel_result == 1) affine_body_indices[global_prefix - 1] = meshIdx;
        if (meshIdx == mesh_data->num_meshes - 1) num_affine_bodies = global_prefix;
    }, 0);

    xpbd_data->num_verts_soft = num_verts_soft;
    xpbd_data->num_verts_rigid = mesh_data->num_verts - num_verts_soft;
    xpbd_data->num_affine_bodies = num_affine_bodies;

    LUISA_INFO("NumVertSoft = {}, Num Affine Bodies {}", num_verts_soft, num_affine_bodies);

    // Init energy
    {
        xpbd_data->sa_system_energy.resize(10240);
        // Rest spring length
        xpbd_data->sa_stretch_springs.resize(num_stretch_springs);
        xpbd_data->sa_stretch_spring_rest_state_length.resize(num_stretch_springs);
        CpuParallel::parallel_for(0, num_stretch_springs, [&](const uint eid)
        {
            const uint orig_eid = stretch_spring_indices[eid];
            uint2 edge = mesh_data->sa_edges[orig_eid];
            float3 x1 = mesh_data->sa_rest_x[edge[0]];
            float3 x2 = mesh_data->sa_rest_x[edge[1]];
            xpbd_data->sa_stretch_springs[eid] = edge; /// 
            xpbd_data->sa_stretch_spring_rest_state_length[eid] = lcs::length_vec(x1 - x2); /// 
        });

        // Rest stretch face length
        xpbd_data->sa_stretch_faces.resize(num_stretch_faces);
        xpbd_data->sa_stretch_faces_Dm_inv.resize(num_stretch_faces);
        CpuParallel::parallel_for(0, num_stretch_faces, [&](const uint fid)
        {
            const uint orig_fid = stretch_face_indices[fid];
            uint3 face = mesh_data->sa_faces[orig_fid];
            const float3 vert_pos[3] = {
                mesh_data->sa_rest_x[face[0]], 
                mesh_data->sa_rest_x[face[1]], 
                mesh_data->sa_rest_x[face[2]]};
            const float3& x_0 = vert_pos[0];
            const float3& x_1 = vert_pos[1];
            const float3& x_2 = vert_pos[2];
            float3 r_1    = x_1 - x_0;
            float3 r_2    = x_2 - x_0;
            float3 cross  = cross_vec(r_1, r_2);
            float3 axis_1 = normalize_vec(r_1);
            float3 axis_2 = normalize_vec(cross_vec(cross, axis_1));
            float2 uv0 = float2(dot_vec(axis_1, x_0), dot_vec(axis_2, x_0));
            float2 uv1 = float2(dot_vec(axis_1, x_1), dot_vec(axis_2, x_1));
            float2 uv2 = float2(dot_vec(axis_1, x_2), dot_vec(axis_2, x_2));
            float2 duv0 = uv1 - uv0;
            float2 duv1 = uv2 - uv0;
            const float2x2 duv = float2x2(duv0, duv1);
            xpbd_data->sa_stretch_faces[fid] = face;
            xpbd_data->sa_stretch_faces_Dm_inv[fid] = luisa::inverse(duv);
        });

        // Rest bending info
        xpbd_data->sa_bending_edges.resize(num_bending_edges);
        xpbd_data->sa_bending_edges_rest_angle.resize(num_bending_edges);
        xpbd_data->sa_bending_edges_Q.resize(num_bending_edges);
        CpuParallel::parallel_for(0, num_bending_edges, [&](const uint eid)
        {
            const uint orig_eid = bending_edge_indices[eid];
            const uint4 edge = mesh_data->sa_dihedral_edges[orig_eid];
            const float3 vert_pos[4] = {
                mesh_data->sa_rest_x[edge[0]], 
                mesh_data->sa_rest_x[edge[1]], 
                mesh_data->sa_rest_x[edge[2]], 
                mesh_data->sa_rest_x[edge[3]]};
            
            // Rest state angle
            {
                const float3& x1 = vert_pos[2];
                const float3& x2 = vert_pos[3];
                const float3& x3 = vert_pos[0];
                const float3& x4 = vert_pos[1];
        
                float3 tmp;
                const float angle = lcs::BendingEnergy::CalcGradientsAndAngle(x1, x2, x3, x4, tmp, tmp, tmp, tmp);
                if (luisa::isnan(angle)) LUISA_ERROR("is nan rest angle {}", eid);
                
                xpbd_data->sa_bending_edges[eid] = edge;
                xpbd_data->sa_bending_edges_rest_angle[eid] = angle; 
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

                float3 e0 = vert_pos[1] - vert_pos[0];
                float3 e1 = vert_pos[2] - vert_pos[0];
                float3 e2 = vert_pos[3] - vert_pos[0];
                float3 e3 = vert_pos[2] - vert_pos[1];
                float3 e4 = vert_pos[3] - vert_pos[1];
                const float cot_01 = calculateCotTheta(e0, -e1);
                const float cot_02 = calculateCotTheta(e0, -e2);
                const float cot_03 = calculateCotTheta(e0, e3);
                const float cot_04 = calculateCotTheta(e0, e4);
                const float4 K = luisa::make_float4(
                    cot_03 + cot_04, 
                    cot_01 + cot_02, 
                    -cot_01 - cot_03, 
                    -cot_02 - cot_04);
                const float A_0 = 0.5f * luisa::length(luisa::cross(e0, e1));
                const float A_1 = 0.5f * luisa::length(luisa::cross(e0, e2));
                // if (is_nan_vec<float4>(K) || is_inf_vec<float4>(K)) fast_print_err("Q of Bending is Illigal");
                const float4x4 m_Q = (3.f / (A_0 + A_1)) * lcs::outer_product(K, K); // Q = 3 qq^T / (A0+A1) ==> Q is symmetric
                xpbd_data->sa_bending_edges_Q[eid] = m_Q; // See : A quadratic bending model for inextensible surfaces.
            }
        });

        // Rest affine body info
        const uint num_blocks_affine_body = num_affine_bodies * 4;
        xpbd_data->sa_affine_bodies.resize(num_affine_bodies);

        xpbd_data->sa_affine_bodies_rest_q.resize(num_blocks_affine_body);
        xpbd_data->sa_affine_bodies_rest_q_v.resize(num_blocks_affine_body);
        xpbd_data->sa_affine_bodies_gravity.resize(num_blocks_affine_body);
        xpbd_data->sa_affine_bodies_q.resize(num_blocks_affine_body);
        xpbd_data->sa_affine_bodies_q_v.resize(num_blocks_affine_body);
        xpbd_data->sa_affine_bodies_q_tilde.resize(num_blocks_affine_body);
        xpbd_data->sa_affine_bodies_q_iter_start.resize(num_blocks_affine_body);
        xpbd_data->sa_affine_bodies_q_step_start.resize(num_blocks_affine_body);
        xpbd_data->sa_affine_bodies_q_outer.resize(num_blocks_affine_body);
        xpbd_data->sa_affine_bodies_q_v_outer.resize(num_blocks_affine_body);
        xpbd_data->sa_affine_bodies_volume.resize(num_blocks_affine_body);

        xpbd_data->sa_affine_bodies_mass_matrix_diag.resize(num_affine_bodies * 4);
        xpbd_data->sa_affine_bodies_mass_matrix_compressed_offdiag.resize(num_affine_bodies);

        xpbd_data->sa_cgA_offdiag_affine_body.resize(num_affine_bodies * 6);
        xpbd_data->sa_vert_affine_bodies_id.resize(mesh_data->num_verts, -1u);

        CpuParallel::single_thread_for(0, num_affine_bodies, [&](const uint body_idx)
        {
            const uint meshIdx = affine_body_indices[body_idx];
            xpbd_data->sa_affine_bodies[body_idx] = meshIdx;

            {
                float3 init_translation = mesh_data->sa_rest_translate[meshIdx];
                float3 init_rotation = mesh_data->sa_rest_rotation[meshIdx];
                // float3 init_scale = mesh_data->sa_rest_scale[meshIdx];
                float3 init_scale = luisa::make_float3(1.0f); // Since we use |AAT-I|
                float4x4 init_transform_matrix = lcs::make_model_matrix(init_translation, init_rotation, init_scale);
                float4x3 rest_q = AffineBodyDynamics::extract_q_from_affine_matrix(init_transform_matrix);;
                float3x3 init_A; float3 init_p;
                AffineBodyDynamics::extract_Ap_from_q(rest_q.cols, init_A, init_p);
                // LUISA_INFO("init p = {}, ATA-I={}, |ATA-I| = {}", init_p, init_A * luisa::transpose(init_A) - Identity3x3, luisa::transpose(init_A));
                xpbd_data->sa_affine_bodies_rest_q[4 * body_idx + 0] = rest_q[0]; // = init_transform_matrix[0].xyz()
                xpbd_data->sa_affine_bodies_rest_q[4 * body_idx + 1] = rest_q[1]; // = init_transform_matrix[1].xyz()
                xpbd_data->sa_affine_bodies_rest_q[4 * body_idx + 2] = rest_q[2]; // = init_transform_matrix[2].xyz()
                xpbd_data->sa_affine_bodies_rest_q[4 * body_idx + 3] = rest_q[3]; // = init_transform_matrix[3].xyz()
                xpbd_data->sa_affine_bodies_rest_q_v[4 * body_idx + 0] = Zero3;
                xpbd_data->sa_affine_bodies_rest_q_v[4 * body_idx + 1] = Zero3;
                xpbd_data->sa_affine_bodies_rest_q_v[4 * body_idx + 2] = Zero3;
                xpbd_data->sa_affine_bodies_rest_q_v[4 * body_idx + 3] = Zero3;
                // LUISA_INFO("Affine Body {} Rest q = {}", body_idx, rest_q);
            }

            const uint curr_prefix = mesh_data->prefix_num_verts[meshIdx];
            const uint next_prefix = mesh_data->prefix_num_verts[meshIdx + 1];
            const uint num_verts_body = next_prefix - curr_prefix;

            EigenFloat12x12 body_mass = EigenFloat12x12::Zero();
            CpuParallel::single_thread_for(curr_prefix, next_prefix, [&](const uint vid)
            {
                float mass = mesh_data->sa_vert_mass[vid];
                float3 scaled_model_x = mesh_data->sa_scaled_model_x[vid];
                auto J = AffineBodyDynamics::get_jacobian_dxdq(scaled_model_x);
                // std::cout << "JtT of vert " << vid << " = \n" << J.transpose() * J << std::endl;
                body_mass += mass * J.transpose() * J;
            });
            // TODO: weighted squared sum in some dimension is zero => Mass matrix diagonal = 0 => Can not get inverse
            std::cout << "Mass Matrix = \n" << body_mass.block<3, 3>(0, 0) << std::endl;
            std::cout << "Sum of mass = \n" << std::reduce(&mesh_data->sa_vert_mass[curr_prefix], &mesh_data->sa_vert_mass[next_prefix], 0.0f) << std::endl;
            // std::cout << "Mass Matrix = \n" << body_mass << std::endl;
            // std::cout << "Inv Mass Matrix = \n" << body_mass.inverse() << std::endl;
            xpbd_data->sa_affine_bodies_mass_matrix_full.push_back(body_mass);
            xpbd_data->sa_affine_bodies_mass_matrix_diag[4 * body_idx + 0] = eigen3x3_to_float3x3(body_mass.block<3, 3>(0, 0));
            xpbd_data->sa_affine_bodies_mass_matrix_diag[4 * body_idx + 1] = eigen3x3_to_float3x3(body_mass.block<3, 3>(3, 3));
            xpbd_data->sa_affine_bodies_mass_matrix_diag[4 * body_idx + 2] = eigen3x3_to_float3x3(body_mass.block<3, 3>(6, 6));
            xpbd_data->sa_affine_bodies_mass_matrix_diag[4 * body_idx + 3] = eigen3x3_to_float3x3(body_mass.block<3, 3>(9, 9));
            xpbd_data->sa_affine_bodies_mass_matrix_compressed_offdiag[body_idx] = luisa::make_float3x3(
                eigen3_to_float3(body_mass.block<3, 1>(3, 0)),
                eigen3_to_float3(body_mass.block<3, 1>(6, 1)),
                eigen3_to_float3(body_mass.block<3, 1>(9, 2))
            );
            body_mass.diagonal() = body_mass.diagonal().cwiseMax(Epsilon);

            float area = CpuParallel::parallel_for_and_reduce_sum<float>(curr_prefix, next_prefix, [&](const uint vid)
            {
                xpbd_data->sa_vert_affine_bodies_id[vid] = body_idx;
                return (
                    mesh_data->sa_rest_vert_area[vid]
                );
            });
            
            const float defulat_density = 10.0f;
            xpbd_data->sa_affine_bodies_volume[body_idx] = area;

            EigenFloat12 gravity_sum = EigenFloat12::Zero(); 
            CpuParallel::single_thread_for(curr_prefix, next_prefix, [&](const uint vid)
            {
                float mass = mesh_data->sa_vert_mass[vid];
                float3 rest_x = mesh_data->sa_model_x[vid];
                auto J = AffineBodyDynamics::get_jacobian_dxdq(rest_x);
                gravity_sum += mass * J.transpose() * float3_to_eigen3(luisa::make_float3(0, -9.8, 0));
            }) ; // / area_mass[1];

            EigenFloat12 body_gravity = body_mass.inverse() * gravity_sum;
            xpbd_data->sa_affine_bodies_gravity[4 * body_idx + 0] = eigen3_to_float3(body_gravity.block<3, 1>(0, 0));
            xpbd_data->sa_affine_bodies_gravity[4 * body_idx + 1] = eigen3_to_float3(body_gravity.block<3, 1>(3, 0));
            xpbd_data->sa_affine_bodies_gravity[4 * body_idx + 2] = eigen3_to_float3(body_gravity.block<3, 1>(6, 0));
            xpbd_data->sa_affine_bodies_gravity[4 * body_idx + 3] = eigen3_to_float3(body_gravity.block<3, 1>(9, 0));
            // LUISA_INFO("Affine body {} : Area = {}, Gravity = {}", body_idx, area, body_gravity);
        });

        CpuParallel::parallel_copy(xpbd_data->sa_affine_bodies_rest_q, xpbd_data->sa_affine_bodies_q_outer);
        CpuParallel::parallel_copy(xpbd_data->sa_affine_bodies_rest_q_v, xpbd_data->sa_affine_bodies_q_v_outer);

    }

    // Init Energy Adjacent List
    {
        xpbd_data->vert_adj_material_force_verts.resize(mesh_data->num_verts);
        xpbd_data->vert_adj_stretch_springs.resize(mesh_data->num_verts);
        xpbd_data->vert_adj_stretch_faces.resize(mesh_data->num_verts);
        xpbd_data->vert_adj_bending_edges.resize(mesh_data->num_verts);

        auto insert_adj_vert = [](std::vector<std::vector<uint>>& adj_map, const uint& vid1, const uint& vid2) 
        {
            if (vid1 == vid2) std::cerr << "Try to build connection with self vertex";
            auto& inner_list = adj_map[vid1];
            auto find_result = std::find(inner_list.begin(), inner_list.end(), vid2);
            if (find_result == inner_list.end())
            {
                inner_list.push_back(vid2);
            }
        };

        // Vert adj faces
        for (uint fid = 0; fid < xpbd_data->sa_stretch_faces.size(); fid++)
        {
            auto face = xpbd_data->sa_stretch_faces[fid];

            for (uint j = 0; j < 3; j++)
            {
                xpbd_data->vert_adj_stretch_faces[face[j]].push_back(fid);
            }

            for (uint ii = 0; ii < 3; ii++)
            {
                for (uint jj = 0; jj < 3; jj++)
                {
                    if (ii != jj) { insert_adj_vert(xpbd_data->vert_adj_material_force_verts, face[ii], face[jj]); }
                }
            }
        } 
        upload_2d_csr_from(xpbd_data->sa_vert_adj_stretch_faces_csr, xpbd_data->vert_adj_stretch_faces); 

        // Vert adj stretch springs
        for (uint eid = 0; eid < xpbd_data->sa_stretch_springs.size(); eid++)
        {
            auto edge = xpbd_data->sa_stretch_springs[eid];
            for (uint j = 0; j < 2; j++)
            {
                xpbd_data->vert_adj_stretch_springs[edge[j]].push_back(eid);
            }

            for (uint ii = 0; ii < 2; ii++)
            {
                for (uint jj = 0; jj < 2; jj++)
                {
                    if (ii != jj) { insert_adj_vert(xpbd_data->vert_adj_material_force_verts, edge[ii], edge[jj]); }
                }
            }
        } 
        upload_2d_csr_from(xpbd_data->sa_vert_adj_stretch_springs_csr, xpbd_data->vert_adj_stretch_springs);

        // Vert adj bending edges
        for (uint eid = 0; eid < xpbd_data->sa_bending_edges.size(); eid++)
        {
            auto edge = xpbd_data->sa_bending_edges[eid];
            for (uint j = 0; j < 4; j++)
            {
                xpbd_data->vert_adj_bending_edges[edge[j]].push_back(eid);
            }

            for (uint ii = 0; ii < 4; ii++)
            {
                for (uint jj = 0; jj < 4; jj++)
                {
                    if (ii != jj) { insert_adj_vert(xpbd_data->vert_adj_material_force_verts, edge[ii], edge[jj]); }
                }
            }
        }  
        upload_2d_csr_from(xpbd_data->sa_vert_adj_bending_edges_csr, xpbd_data->vert_adj_bending_edges);

        // Vert adj material-force-verts
        CpuParallel::parallel_for(0, mesh_data->num_verts, [&](const uint vid)
        {
            std::vector<uint>& adj_list = xpbd_data->vert_adj_material_force_verts[vid];
            std::sort(adj_list.begin(), adj_list.end());
        });
        upload_2d_csr_from(xpbd_data->sa_vert_adj_material_force_verts_csr, xpbd_data->vert_adj_material_force_verts);
    }

    // Constraint Graph Coloring
    std::vector< std::vector<uint> > tmp_clusterd_constraint_stretch_mass_spring;
    std::vector< std::vector<uint> > tmp_clusterd_constraint_bending;
    {
        fn_graph_coloring_per_constraint(
            "Distance  Spring Constraint", 
            tmp_clusterd_constraint_stretch_mass_spring, 
            xpbd_data->vert_adj_stretch_springs, xpbd_data->sa_stretch_springs, 2);

        fn_graph_coloring_per_constraint(
            "Bending   Angle  Constraint", 
            tmp_clusterd_constraint_bending, 
            xpbd_data->vert_adj_bending_edges, xpbd_data->sa_bending_edges, 4);
            
        xpbd_data->num_clusters_springs = tmp_clusterd_constraint_stretch_mass_spring.size();
        xpbd_data->num_clusters_bending_edges = tmp_clusterd_constraint_bending.size();

        fn_get_prefix(xpbd_data->sa_prefix_merged_springs, tmp_clusterd_constraint_stretch_mass_spring);
        fn_get_prefix(xpbd_data->sa_prefix_merged_bending_edges, tmp_clusterd_constraint_bending);
        
        upload_2d_csr_from(xpbd_data->sa_clusterd_springs, tmp_clusterd_constraint_stretch_mass_spring);
        upload_2d_csr_from(xpbd_data->sa_clusterd_bending_edges, tmp_clusterd_constraint_bending);
    }

    // Init Newton Coloring
    {
        const uint num_verts = mesh_data->num_verts;
        const auto& vert_adj_verts = xpbd_data->vert_adj_material_force_verts; // Not considering Obstacle
        std::vector<uint2> upper_matrix_elements; // 
        upper_matrix_elements.reserve(num_verts * 10);
        std::vector<std::vector<uint>> vert_adj_upper_verts(num_verts);
        for (uint vid = 0; vid < num_verts; vid++)
        {
            for (const uint adj_vid : vert_adj_verts[vid])
            {
                if (vid < adj_vid)
                {
                    vert_adj_upper_verts[vid].push_back(adj_vid);
                    upper_matrix_elements.emplace_back(uint2(vid, adj_vid));
                }
            }
        }
        
        std::vector<std::vector<uint>> tmp_clusterd_hessian_set;
        fn_graph_coloring_per_constraint(
            "SpMV non-conlict set", 
            tmp_clusterd_hessian_set, 
            vert_adj_upper_verts, upper_matrix_elements, 2);
        
        upload_from(xpbd_data->sa_hessian_pairs, upper_matrix_elements);
        xpbd_data->num_clusters_hessian_pairs = tmp_clusterd_hessian_set.size();
        fn_get_prefix(xpbd_data->sa_prefix_merged_hessian_pairs, tmp_clusterd_hessian_set);
        upload_2d_csr_from(xpbd_data->sa_clusterd_hessian_pairs, tmp_clusterd_hessian_set);

        std::vector<std::vector<uint>> hessian_insert_indices(num_verts);
        CpuParallel::parallel_for(0, num_verts, [&](const uint vid)
        {
            hessian_insert_indices[vid].resize(vert_adj_verts[vid].size(), -1u);
        });
        CpuParallel::parallel_for(0, upper_matrix_elements.size(), [&](const uint pair_idx)
        {
            const uint2 pair = upper_matrix_elements[pair_idx];
            const uint left = pair[0];
            const uint right = pair[1];
            const auto& adj_list = vert_adj_verts[left];
            for (uint jj = 0; jj < adj_list.size(); jj++)
            {
                if (adj_list[jj] == right)
                {
                    hessian_insert_indices[left][jj] = pair_idx;
                    return;
                }
            }
            LUISA_ERROR("Can not find {} in adjacent list of {}", right, left); 
        });
        {
            const uint num_offdiag_upper = 1;
            xpbd_data->sa_hessian_slot_per_edge.resize(num_stretch_springs * num_offdiag_upper);
            CpuParallel::parallel_for(0, num_stretch_springs, [&](const uint eid)
            {
                auto edge = xpbd_data->sa_stretch_springs[eid];
                uint edge_offset = 0;
                for (uint ii = 0; ii < 2; ii++)
                {
                    for (uint jj = ii + 1; jj < 2; jj++)
                    {
                        const bool need_transpose = edge[ii] > edge[jj];
                        const uint left  = need_transpose ? edge[jj] : edge[ii];
                        const uint right = need_transpose ? edge[ii] : edge[jj];

                        const auto& adj_list = vert_adj_verts[left];
                        auto find = std::find(adj_list.begin(), adj_list.end(), right); 
                        if (find == adj_list.end()) { LUISA_ERROR("Can not find {} in adjacent list of {}", right, left); }
                        else 
                        {
                            uint offset = std::distance(adj_list.begin(), find);
                            const uint adj_index = hessian_insert_indices[left][offset];
                            xpbd_data->sa_hessian_slot_per_edge[num_offdiag_upper * eid + edge_offset] = adj_index;
                            edge_offset += 1;
                        }
                    }
                }
            });
        }
    }

    // Vertex Block Descent Coloring
    { 
        // Graph Coloring
        const uint num_verts_total = mesh_data->num_verts;
        xpbd_data->sa_Hf.resize(mesh_data->num_verts * 12);
        xpbd_data->sa_Hf1.resize(mesh_data->num_verts);

        const std::vector< std::vector<uint> >& vert_adj_verts = xpbd_data->vert_adj_material_force_verts;
        std::vector<std::vector<uint>> clusterd_vertices_bending; std::vector<uint> prefix_vertices_bending;

        fn_graph_coloring_per_vertex(vert_adj_verts, clusterd_vertices_bending, prefix_vertices_bending);
        xpbd_data->num_clusters_per_vertex_with_material_constraints = clusterd_vertices_bending.size();
        upload_from(xpbd_data->prefix_per_vertex_with_material_constraints, prefix_vertices_bending); 
        upload_2d_csr_from(xpbd_data->clusterd_per_vertex_with_material_constraints, clusterd_vertices_bending);

        // Reverse map
        xpbd_data->per_vertex_bending_cluster_id.resize(mesh_data->num_verts);
        for (uint cluster = 0; cluster < xpbd_data->num_clusters_per_vertex_with_material_constraints; cluster++)
        {
            const uint next_prefix = xpbd_data->clusterd_per_vertex_with_material_constraints[cluster + 1];
            const uint curr_prefix = xpbd_data->clusterd_per_vertex_with_material_constraints[cluster];
            const uint num_verts_cluster = next_prefix - curr_prefix;
            CpuParallel::parallel_for(0, num_verts_cluster, [&](const uint i)
            {
                const uint vid = xpbd_data->clusterd_per_vertex_with_material_constraints[curr_prefix + i];
                xpbd_data->per_vertex_bending_cluster_id[vid] = cluster;
            });
        }
        
    }

    // Precomputation
    {
        // Spring Constraint
        {
            xpbd_data->sa_merged_stretch_springs.resize(num_stretch_springs);
            xpbd_data->sa_merged_stretch_spring_rest_length.resize(num_stretch_springs);
            xpbd_data->sa_lambda_stretch_mass_spring.resize(num_stretch_springs);

            uint prefix = 0;
            for (uint cluster = 0; cluster < tmp_clusterd_constraint_stretch_mass_spring.size(); cluster++)
            {
                const auto& curr_cluster = tmp_clusterd_constraint_stretch_mass_spring[cluster];
                CpuParallel::parallel_for(0, curr_cluster.size(), [&](const uint i)
                {
                    const uint eid = curr_cluster[i];
                    {
                        xpbd_data->sa_merged_stretch_springs[prefix + i] = xpbd_data->sa_stretch_springs[eid];
                        xpbd_data->sa_merged_stretch_spring_rest_length[prefix + i] = xpbd_data->sa_stretch_spring_rest_state_length[eid];
                    }
                });
                prefix += curr_cluster.size();
            } if (prefix != xpbd_data->sa_stretch_springs.size()) LUISA_ERROR("Sum of Mass Spring Cluster Is Not Equal  Than Orig");
        }

        // Bending Constraint
        {
            xpbd_data->sa_merged_bending_edges.resize(num_bending_edges);
            xpbd_data->sa_merged_bending_edges_angle.resize(num_bending_edges);
            xpbd_data->sa_merged_bending_edges_Q.resize(num_bending_edges);
            xpbd_data->sa_lambda_bending.resize(num_bending_edges);

            uint prefix = 0;
            for (uint cluster = 0; cluster < tmp_clusterd_constraint_bending.size(); cluster++)
            {
                const auto& curr_cluster = tmp_clusterd_constraint_bending[cluster];
                CpuParallel::parallel_for(0, curr_cluster.size(), [&](const uint i)
                {
                    const uint eid = curr_cluster[i];
                    {
                        xpbd_data->sa_merged_bending_edges[prefix + i] = xpbd_data->sa_bending_edges[eid];
                        xpbd_data->sa_merged_bending_edges_angle[prefix + i] = xpbd_data->sa_bending_edges_rest_angle[eid];
                        xpbd_data->sa_merged_bending_edges_Q[prefix + i] = xpbd_data->sa_bending_edges_Q[eid];
                    }
                });
                prefix += curr_cluster.size();
            } if (prefix != xpbd_data->sa_bending_edges.size()) LUISA_ERROR("Sum of Bending Cluster Is Not Equal Than Orig");
        }
    }

}

void upload_xpbd_buffers(
    luisa::compute::Device& device, 
    luisa::compute::Stream& stream, 
    lcs::SimulationData<std::vector>* input_data, 
    lcs::SimulationData<luisa::compute::Buffer>* output_data)
{
    output_data->num_clusters_springs = input_data->num_clusters_springs;
    output_data->num_clusters_bending_edges = input_data->num_clusters_bending_edges;
    output_data->num_clusters_per_vertex_with_material_constraints = input_data->num_clusters_per_vertex_with_material_constraints;
    output_data->num_clusters_hessian_pairs= input_data->num_clusters_hessian_pairs;

    stream
        << upload_buffer(device, output_data->sa_x_tilde, input_data->sa_x_tilde)
        << upload_buffer(device, output_data->sa_x, input_data->sa_x)
        << upload_buffer(device, output_data->sa_v, input_data->sa_v)
        << upload_buffer(device, output_data->sa_v_step_start, input_data->sa_v_step_start)
        << upload_buffer(device, output_data->sa_x_step_start, input_data->sa_x_step_start)
        << upload_buffer(device, output_data->sa_x_iter_start, input_data->sa_x_iter_start)

        << upload_buffer(device, output_data->sa_system_energy, input_data->sa_system_energy)
        ;
    if (input_data->sa_stretch_springs.size() > 0)
    {
        stream 
            << upload_buffer(device, output_data->sa_stretch_springs, input_data->sa_stretch_springs)
            << upload_buffer(device, output_data->sa_stretch_spring_rest_state_length, input_data->sa_stretch_spring_rest_state_length)

            << upload_buffer(device, output_data->sa_merged_stretch_springs, input_data->sa_merged_stretch_springs)
            << upload_buffer(device, output_data->sa_merged_stretch_spring_rest_length, input_data->sa_merged_stretch_spring_rest_length)
            ;
    }
    if (input_data->sa_stretch_faces.size() > 0)
    {
        stream 
            << upload_buffer(device, output_data->sa_stretch_faces, input_data->sa_stretch_faces)
            << upload_buffer(device, output_data->sa_stretch_faces_Dm_inv, input_data->sa_stretch_faces_Dm_inv)

            << upload_buffer(device, output_data->sa_clusterd_springs, input_data->sa_clusterd_springs)
            << upload_buffer(device, output_data->sa_prefix_merged_springs, input_data->sa_prefix_merged_springs)
            << upload_buffer(device, output_data->sa_lambda_stretch_mass_spring, input_data->sa_lambda_stretch_mass_spring) // just resize
            ;
    }
    if (input_data->sa_bending_edges.size() > 0)
    {
        stream 
            << upload_buffer(device, output_data->sa_bending_edges, input_data->sa_bending_edges)
            << upload_buffer(device, output_data->sa_bending_edges_rest_angle, input_data->sa_bending_edges_rest_angle)
            << upload_buffer(device, output_data->sa_bending_edges_Q, input_data->sa_bending_edges_Q)

            << upload_buffer(device, output_data->sa_merged_bending_edges, input_data->sa_merged_bending_edges)
            << upload_buffer(device, output_data->sa_merged_bending_edges_angle, input_data->sa_merged_bending_edges_angle)
            << upload_buffer(device, output_data->sa_merged_bending_edges_Q, input_data->sa_merged_bending_edges_Q)

            << upload_buffer(device, output_data->sa_clusterd_bending_edges, input_data->sa_clusterd_bending_edges)
            << upload_buffer(device, output_data->sa_prefix_merged_bending_edges, input_data->sa_prefix_merged_bending_edges)
            << upload_buffer(device, output_data->sa_lambda_bending, input_data->sa_lambda_bending) // just resize
            ;
    }
    if (input_data->sa_affine_bodies.size() > 0)
    {
        stream  
            << upload_buffer(device, output_data->sa_affine_bodies, input_data->sa_affine_bodies)
            << upload_buffer(device, output_data->sa_affine_bodies_rest_q, input_data->sa_affine_bodies_rest_q)
            << upload_buffer(device, output_data->sa_affine_bodies_gravity, input_data->sa_affine_bodies_gravity)
            << upload_buffer(device, output_data->sa_affine_bodies_q, input_data->sa_affine_bodies_q)
            << upload_buffer(device, output_data->sa_affine_bodies_q_v, input_data->sa_affine_bodies_q_v)
            << upload_buffer(device, output_data->sa_affine_bodies_q_tilde, input_data->sa_affine_bodies_q_tilde)
            << upload_buffer(device, output_data->sa_affine_bodies_q_iter_start, input_data->sa_affine_bodies_q_iter_start)
            << upload_buffer(device, output_data->sa_affine_bodies_q_step_start, input_data->sa_affine_bodies_q_step_start)
            << upload_buffer(device, output_data->sa_affine_bodies_volume, input_data->sa_affine_bodies_volume)
            << upload_buffer(device, output_data->sa_affine_bodies_mass_matrix_diag, input_data->sa_affine_bodies_mass_matrix_diag)
            << upload_buffer(device, output_data->sa_affine_bodies_mass_matrix_compressed_offdiag, input_data->sa_affine_bodies_mass_matrix_compressed_offdiag)
            << upload_buffer(device, output_data->sa_cgA_offdiag_affine_body, input_data->sa_cgA_offdiag_affine_body)
        ;
    } 
    stream << upload_buffer(device, output_data->sa_vert_affine_bodies_id, input_data->sa_vert_affine_bodies_id); // Basic information
    if (input_data->sa_hessian_pairs.size() > 0)
    {
        stream 
            << upload_buffer(device, output_data->sa_prefix_merged_hessian_pairs, input_data->sa_prefix_merged_hessian_pairs)
            << upload_buffer(device, output_data->sa_clusterd_hessian_pairs, input_data->sa_clusterd_hessian_pairs)
            << upload_buffer(device, output_data->sa_hessian_pairs, input_data->sa_hessian_pairs)
            << upload_buffer(device, output_data->sa_hessian_slot_per_edge, input_data->sa_hessian_slot_per_edge)
        ;
    }
    stream
        << upload_buffer(device, output_data->sa_vert_adj_material_force_verts_csr, input_data->sa_vert_adj_material_force_verts_csr)
        << upload_buffer(device, output_data->sa_vert_adj_stretch_springs_csr, input_data->sa_vert_adj_stretch_springs_csr)
        << upload_buffer(device, output_data->sa_vert_adj_stretch_faces_csr, input_data->sa_vert_adj_stretch_faces_csr)
        << upload_buffer(device, output_data->sa_vert_adj_bending_edges_csr, input_data->sa_vert_adj_bending_edges_csr)

        << upload_buffer(device, output_data->prefix_per_vertex_with_material_constraints, input_data->prefix_per_vertex_with_material_constraints)
        << upload_buffer(device, output_data->clusterd_per_vertex_with_material_constraints, input_data->clusterd_per_vertex_with_material_constraints)

        << upload_buffer(device, output_data->per_vertex_bending_cluster_id, input_data->per_vertex_bending_cluster_id)
        << upload_buffer(device, output_data->sa_Hf, input_data->sa_Hf)
        << upload_buffer(device, output_data->sa_Hf1, input_data->sa_Hf1)
        << luisa::compute::synchronize();
}

void resize_pcg_data(
    luisa::compute::Device& device, 
    luisa::compute::Stream& stream, 
    lcs::MeshData<std::vector>* mesh_data, 
    lcs::SimulationData<std::vector>* host_data, 
    lcs::SimulationData<luisa::compute::Buffer>* device_data
)
{
    const uint num_springs = host_data->sa_stretch_springs.size();
    const uint num_bending_edges = host_data->sa_bending_edges.size();
    const uint num_faces = host_data->sa_stretch_faces.size();
    const uint num_affine_bodies = host_data->num_affine_bodies;
    const uint num_verts = host_data->num_verts_soft + num_affine_bodies * 4;

    // const uint off_diag_count = std::max(uint(device_data->sa_hessian_pairs.size()), num_springs * 2);

    resize_buffer(host_data->sa_cgX, num_verts);
    resize_buffer(host_data->sa_cgB, num_verts);
    resize_buffer(host_data->sa_cgA_diag, num_verts);
    if (num_springs > 0)        resize_buffer(host_data->sa_cgA_offdiag_stretch_spring, num_springs * 1);
    if (num_bending_edges > 0)  resize_buffer(host_data->sa_cgA_offdiag_bending, num_bending_edges * 6);
    if (num_affine_bodies > 0)   resize_buffer(host_data->sa_cgA_offdiag_affine_body, num_affine_bodies * 6);

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
    if (num_springs > 0)        resize_buffer(device, device_data->sa_cgA_offdiag_stretch_spring, num_springs * 1);
    if (num_bending_edges > 0)  resize_buffer(device, device_data->sa_cgA_offdiag_bending, num_bending_edges * 6);
    if (num_affine_bodies > 0)   resize_buffer(device, device_data->sa_cgA_offdiag_affine_body, num_affine_bodies * 6);
    resize_buffer(device, device_data->sa_cgMinv, num_verts);
    resize_buffer(device, device_data->sa_cgP, num_verts);
    resize_buffer(device, device_data->sa_cgQ, num_verts);
    resize_buffer(device, device_data->sa_cgR, num_verts);
    resize_buffer(device, device_data->sa_cgZ, num_verts);
    resize_buffer(device, device_data->sa_block_result, num_verts);
    resize_buffer(device, device_data->sa_convergence, 10240);


} 

} // namespace lcs::Initializer