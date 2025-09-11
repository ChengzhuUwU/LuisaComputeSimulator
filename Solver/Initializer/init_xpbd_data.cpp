#include "Initializer/init_xpbd_data.h"
#include "Core/affine_position.h"
#include "Core/float_n.h"
#include "Core/float_nxn.h"
#include "Energy/bending_energy.h"
#include "MeshOperation/mesh_reader.h"
#include "Initializer/initializer_utils.h"
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

    const uint2 range_verts_cloth = mesh_data->range_verts_cloth;
    const uint2 range_verts_tetrahedral = mesh_data->range_verts_tetrahedral;
    const uint2 range_edges_cloth = mesh_data->range_edges_cloth;
    const uint2 range_faces_cloth = mesh_data->range_faces_cloth;
    const uint2 range_bending_edges_cloth = mesh_data->range_bending_edges_cloth;

    const uint num_verts_cloth = range_verts_cloth[1] - range_verts_cloth[0];
    const uint num_verts_tet = range_verts_tetrahedral[1] - range_verts_tetrahedral[0];
    const uint num_stretch_springs = range_edges_cloth[1] - range_edges_cloth[0];
    const uint num_stretch_faces = range_faces_cloth[1] - range_faces_cloth[0];
    const uint num_bending_edges = range_bending_edges_cloth[1] - range_bending_edges_cloth[0];


    // Init energy
    {
        xpbd_data->sa_system_energy.resize(10240);
        // Rest spring length
        xpbd_data->sa_stretch_springs.resize(num_stretch_springs);
        xpbd_data->sa_stretch_spring_rest_state_length.resize(num_stretch_springs);
        CpuParallel::parallel_for(0, num_stretch_springs, [&](const uint eid)
        {
            uint2 edge = mesh_data->sa_edges[range_edges_cloth[0] + eid];
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
            uint3 face = mesh_data->sa_faces[range_faces_cloth[0] + fid];
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
            const uint4 edge = mesh_data->sa_bending_edges[range_bending_edges_cloth[0] + eid];
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
                if (luisa::isnan(angle)) luisa::log_error("is nan rest angle {}", eid);
                
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
    }

    // Constraint Graph Coloring
    std::vector< std::vector<uint> > tmp_clusterd_constraint_stretch_mass_spring;
    std::vector< std::vector<uint> > tmp_clusterd_constraint_bending;
    {
        fn_graph_coloring_per_constraint(
            "Distance  Spring Constraint", 
            tmp_clusterd_constraint_stretch_mass_spring, 
            mesh_data->vert_adj_edges, xpbd_data->sa_stretch_springs, 2);

        fn_graph_coloring_per_constraint(
            "Bending   Angle  Constraint", 
            tmp_clusterd_constraint_bending, 
            mesh_data->vert_adj_bending_edges, xpbd_data->sa_bending_edges, 4);
            
        xpbd_data->num_clusters_springs = tmp_clusterd_constraint_stretch_mass_spring.size();
        xpbd_data->num_clusters_bending_edges = tmp_clusterd_constraint_bending.size();

        fn_get_prefix(xpbd_data->sa_prefix_merged_springs, tmp_clusterd_constraint_stretch_mass_spring);
        fn_get_prefix(xpbd_data->sa_prefix_merged_bending_edges, tmp_clusterd_constraint_bending);
        
        upload_2d_csr_from(xpbd_data->sa_clusterd_springs, tmp_clusterd_constraint_stretch_mass_spring);
        upload_2d_csr_from(xpbd_data->sa_clusterd_bending_edges, tmp_clusterd_constraint_bending);
    }

    // Init newton coloring
    {
        const uint num_verts = num_verts_cloth + num_verts_tet;
        const auto& vert_adj_verts = mesh_data->vert_adj_verts_with_material_constraints; // Not considering Obstacle
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
            luisa::log_error("Can not find {} in adjacent list of {}", right, left); 
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
                        if (find == adj_list.end()) { luisa::log_error("Can not find {} in adjacent list of {}", right, left); }
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

        const std::vector< std::vector<uint> >& vert_adj_verts = mesh_data->vert_adj_verts_with_material_constraints;
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
            } if (prefix != mesh_data->num_edges) luisa::log_error("Sum of Mass Spring Cluster Is Not Equal  Than Orig");
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
            } if (prefix != mesh_data->num_bending_edges) luisa::log_error("Sum of Bending Cluster Is Not Equal Than Orig");
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
        << upload_buffer(device, output_data->sa_stretch_springs, input_data->sa_stretch_springs)
        << upload_buffer(device, output_data->sa_stretch_spring_rest_state_length, input_data->sa_stretch_spring_rest_state_length)
        << upload_buffer(device, output_data->sa_stretch_faces, input_data->sa_stretch_faces)
        << upload_buffer(device, output_data->sa_stretch_faces_Dm_inv, input_data->sa_stretch_faces_Dm_inv)
        << upload_buffer(device, output_data->sa_bending_edges, input_data->sa_bending_edges)
        << upload_buffer(device, output_data->sa_bending_edges_rest_angle, input_data->sa_bending_edges_rest_angle)
        << upload_buffer(device, output_data->sa_bending_edges_Q, input_data->sa_bending_edges_Q)

        << upload_buffer(device, output_data->sa_merged_stretch_springs, input_data->sa_merged_stretch_springs)
        << upload_buffer(device, output_data->sa_merged_stretch_spring_rest_length, input_data->sa_merged_stretch_spring_rest_length)
        << upload_buffer(device, output_data->sa_merged_bending_edges, input_data->sa_merged_bending_edges)
        << upload_buffer(device, output_data->sa_merged_bending_edges_angle, input_data->sa_merged_bending_edges_angle)
        << upload_buffer(device, output_data->sa_merged_bending_edges_Q, input_data->sa_merged_bending_edges_Q)

        << upload_buffer(device, output_data->sa_clusterd_springs, input_data->sa_clusterd_springs)
        << upload_buffer(device, output_data->sa_prefix_merged_springs, input_data->sa_prefix_merged_springs)
        << upload_buffer(device, output_data->sa_lambda_stretch_mass_spring, input_data->sa_lambda_stretch_mass_spring) // just resize

        << upload_buffer(device, output_data->sa_clusterd_bending_edges, input_data->sa_clusterd_bending_edges)
        << upload_buffer(device, output_data->sa_prefix_merged_bending_edges, input_data->sa_prefix_merged_bending_edges)
        << upload_buffer(device, output_data->sa_lambda_bending, input_data->sa_lambda_bending) // just resize
        
        << upload_buffer(device, output_data->sa_prefix_merged_hessian_pairs, input_data->sa_prefix_merged_hessian_pairs)
        << upload_buffer(device, output_data->sa_clusterd_hessian_pairs, input_data->sa_clusterd_hessian_pairs)
        << upload_buffer(device, output_data->sa_hessian_pairs, input_data->sa_hessian_pairs)
        << upload_buffer(device, output_data->sa_hessian_slot_per_edge, input_data->sa_hessian_slot_per_edge)

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
    const uint num_verts = mesh_data->num_verts;
    const uint num_edges = mesh_data->num_edges;
    const uint num_faces = mesh_data->num_faces;

    const uint off_diag_count = std::max(uint(device_data->sa_hessian_pairs.size()), num_edges * 2);

    resize_buffer(host_data->sa_cgX, num_verts);
    resize_buffer(host_data->sa_cgB, num_verts);
    resize_buffer(host_data->sa_cgA_diag, num_verts);
    resize_buffer(host_data->sa_cgA_offdiag, off_diag_count);
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
    resize_buffer(device, device_data->sa_cgA_offdiag, off_diag_count);
    resize_buffer(device, device_data->sa_cgMinv, num_verts);
    resize_buffer(device, device_data->sa_cgP, num_verts);
    resize_buffer(device, device_data->sa_cgQ, num_verts);
    resize_buffer(device, device_data->sa_cgR, num_verts);
    resize_buffer(device, device_data->sa_cgZ, num_verts);
    resize_buffer(device, device_data->sa_block_result, num_verts);
    resize_buffer(device, device_data->sa_convergence, 10240);


} 

} // namespace lcs::Initializer