#include "Initializer/init_xpbd_data.h"
#include "Core/affine_position.h"
#include "Core/float_nxn.h"
#include "Energy/bending_energy.h"
#include "MeshOperation/mesh_reader.h"
#include "Initializer/initializer_utils.h"


namespace lcsv::Initializer
{

void init_xpbd_data(lcsv::MeshData<std::vector>* mesh_data, lcsv::SimulationData<std::vector>* xpbd_data)
{
    xpbd_data->sa_x_tilde.resize(mesh_data->num_verts); 
    xpbd_data->sa_x.resize(mesh_data->num_verts);
    xpbd_data->sa_v.resize(mesh_data->num_verts);       CpuParallel::parallel_copy(mesh_data->sa_rest_v, xpbd_data->sa_v);
    xpbd_data->sa_v_step_start.resize(mesh_data->num_verts); CpuParallel::parallel_copy(mesh_data->sa_rest_v, xpbd_data->sa_v_step_start);
    xpbd_data->sa_x_step_start.resize(mesh_data->num_verts);
    xpbd_data->sa_x_iter_start.resize(mesh_data->num_verts);

    // Constraint Graph Coloring
    std::vector< std::vector<uint> > tmp_clusterd_constraint_stretch_mass_spring;
    std::vector< std::vector<uint> > tmp_clusterd_constraint_bending;
    {
        fn_graph_coloring_per_constraint(
            "Distance  Spring Constraint", 
            tmp_clusterd_constraint_stretch_mass_spring, 
            mesh_data->vert_adj_edges, mesh_data->sa_edges, 2);

        fn_graph_coloring_per_constraint(
            "Bending   Angle  Constraint", 
            tmp_clusterd_constraint_bending, 
            mesh_data->vert_adj_bending_edges, mesh_data->sa_bending_edges, 4);
            
        xpbd_data->num_clusters_springs = tmp_clusterd_constraint_stretch_mass_spring.size();
        xpbd_data->num_clusters_bending_edges = tmp_clusterd_constraint_bending.size();

        fn_get_prefix(xpbd_data->sa_prefix_merged_springs, tmp_clusterd_constraint_stretch_mass_spring);
        fn_get_prefix(xpbd_data->sa_prefix_merged_bending_edges, tmp_clusterd_constraint_bending);
        
        upload_2d_csr_from(xpbd_data->sa_clusterd_springs, tmp_clusterd_constraint_stretch_mass_spring);
        upload_2d_csr_from(xpbd_data->sa_clusterd_bending_edges, tmp_clusterd_constraint_bending);
    }

    // Init newton coloring
    {
        const uint num_verts = mesh_data->num_verts;
        const auto& vert_adj_verts = mesh_data->vert_adj_verts_with_bending;
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
            xpbd_data->sa_hessian_slot_per_edge.resize(mesh_data->num_edges * num_offdiag_upper);
            CpuParallel::parallel_for(0, mesh_data->num_edges, [&](const uint eid)
            {
                auto edge = mesh_data->sa_edges[eid];
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

        const std::vector< std::vector<uint> >& vert_adj_verts = mesh_data->vert_adj_verts_with_bending;
        std::vector<std::vector<uint>> clusterd_vertices_bending; std::vector<uint> prefix_vertices_bending;

        fn_graph_coloring_per_vertex(vert_adj_verts, clusterd_vertices_bending, prefix_vertices_bending);
        xpbd_data->num_clusters_per_vertex_bending = clusterd_vertices_bending.size();
        upload_from(xpbd_data->prefix_per_vertex_bending, prefix_vertices_bending); 
        upload_2d_csr_from(xpbd_data->clusterd_per_vertex_bending, clusterd_vertices_bending);

        // Reverse map
        xpbd_data->per_vertex_bending_cluster_id.resize(mesh_data->num_verts);
        for (uint cluster = 0; cluster < xpbd_data->num_clusters_per_vertex_bending; cluster++)
        {
            const uint next_prefix = xpbd_data->clusterd_per_vertex_bending[cluster + 1];
            const uint curr_prefix = xpbd_data->clusterd_per_vertex_bending[cluster];
            const uint num_verts_cluster = next_prefix - curr_prefix;
            CpuParallel::parallel_for(0, num_verts_cluster, [&](const uint i)
            {
                const uint vid = xpbd_data->clusterd_per_vertex_bending[curr_prefix + i];
                xpbd_data->per_vertex_bending_cluster_id[vid] = cluster;
            });
        }
        
    }

    // Precomputation
    {
        // Spring Constraint
        {
            xpbd_data->sa_merged_edges.resize(mesh_data->num_edges);
            xpbd_data->sa_merged_edges_rest_length.resize(mesh_data->num_edges);
            xpbd_data->sa_lambda_stretch_mass_spring.resize(mesh_data->num_edges);

            uint prefix = 0;
            for (uint cluster = 0; cluster < tmp_clusterd_constraint_stretch_mass_spring.size(); cluster++)
            {
                const auto& curr_cluster = tmp_clusterd_constraint_stretch_mass_spring[cluster];
                CpuParallel::parallel_for(0, curr_cluster.size(), [&](const uint i)
                {
                    const uint eid = curr_cluster[i];
                    {
                        xpbd_data->sa_merged_edges[prefix + i] = mesh_data->sa_edges[eid];
                        xpbd_data->sa_merged_edges_rest_length[prefix + i] = mesh_data->sa_edges_rest_state_length[eid];
                    }
                });
                prefix += curr_cluster.size();
            } if (prefix != mesh_data->num_edges) luisa::log_error("Sum of Mass Spring Cluster Is Not Equal  Than Orig");
        }

        // Bending Constraint
        {
            xpbd_data->sa_merged_bending_edges.resize(mesh_data->num_bending_edges);
            xpbd_data->sa_merged_bending_edges_angle.resize(mesh_data->num_bending_edges);
            xpbd_data->sa_merged_bending_edges_Q.resize(mesh_data->num_bending_edges);
            xpbd_data->sa_lambda_bending.resize(mesh_data->num_bending_edges);

            uint prefix = 0;
            for (uint cluster = 0; cluster < tmp_clusterd_constraint_bending.size(); cluster++)
            {
                const auto& curr_cluster = tmp_clusterd_constraint_bending[cluster];
                CpuParallel::parallel_for(0, curr_cluster.size(), [&](const uint i)
                {
                    const uint eid = curr_cluster[i];
                    {
                        xpbd_data->sa_merged_bending_edges[prefix + i] = mesh_data->sa_bending_edges[eid];
                        xpbd_data->sa_merged_bending_edges_angle[prefix + i] = mesh_data->sa_bending_edges_rest_angle[eid];
                        xpbd_data->sa_merged_bending_edges_Q[prefix + i] = mesh_data->sa_bending_edges_Q[eid];
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
    lcsv::SimulationData<std::vector>* input_data, 
    lcsv::SimulationData<luisa::compute::Buffer>* output_data)
{
    output_data->num_clusters_springs = input_data->num_clusters_springs;
    output_data->num_clusters_bending_edges = input_data->num_clusters_bending_edges;
    output_data->num_clusters_per_vertex_bending = input_data->num_clusters_per_vertex_bending;
    output_data->num_clusters_hessian_pairs= input_data->num_clusters_hessian_pairs;

    stream
        << upload_buffer(device, output_data->sa_x_tilde, input_data->sa_x_tilde)
        << upload_buffer(device, output_data->sa_x, input_data->sa_x)
        << upload_buffer(device, output_data->sa_v, input_data->sa_v)
        << upload_buffer(device, output_data->sa_v_step_start, input_data->sa_v_step_start)
        << upload_buffer(device, output_data->sa_x_step_start, input_data->sa_x_step_start)
        << upload_buffer(device, output_data->sa_x_iter_start, input_data->sa_x_iter_start)
        << upload_buffer(device, output_data->sa_merged_edges, input_data->sa_merged_edges)
        << upload_buffer(device, output_data->sa_merged_edges_rest_length, input_data->sa_merged_edges_rest_length)
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

        << upload_buffer(device, output_data->prefix_per_vertex_bending, input_data->prefix_per_vertex_bending)
        << upload_buffer(device, output_data->clusterd_per_vertex_bending, input_data->clusterd_per_vertex_bending)

        << upload_buffer(device, output_data->per_vertex_bending_cluster_id, input_data->per_vertex_bending_cluster_id)
        << upload_buffer(device, output_data->sa_Hf, input_data->sa_Hf)
        << upload_buffer(device, output_data->sa_Hf1, input_data->sa_Hf1)
        << luisa::compute::synchronize();
}

void resize_pcg_data(
    luisa::compute::Device& device, 
    luisa::compute::Stream& stream, 
    lcsv::MeshData<std::vector>* mesh_data, 
    lcsv::SimulationData<std::vector>* host_data, 
    lcsv::SimulationData<luisa::compute::Buffer>* device_data
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

} // namespace lcsv::Initializer