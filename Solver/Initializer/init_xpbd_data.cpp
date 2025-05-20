#include "Initializer/init_xpbd_data.h"
#include "Core/affine_position.h"
#include "Core/float_nxn.h"
#include "Energy/bending_energy.h"
#include "MeshOperation/mesh_reader.h"
#include "Initializer/initializer_utils.h"


namespace lcsv 
{

namespace Initializater
{

void init_xpbd_data(lcsv::MeshData<std::vector>* mesh_data, lcsv::XpbdData<std::vector>* xpbd_data)
{
    xpbd_data->sa_x_tilde.resize(mesh_data->num_verts); 
    xpbd_data->sa_x.resize(mesh_data->num_verts);
    xpbd_data->sa_v.resize(mesh_data->num_verts);       CpuParallel::parallel_copy(mesh_data->sa_rest_v, xpbd_data->sa_v);
    xpbd_data->sa_v_start.resize(mesh_data->num_verts); CpuParallel::parallel_copy(mesh_data->sa_rest_v, xpbd_data->sa_v_start);
    xpbd_data->sa_x_start.resize(mesh_data->num_verts);

    // Constraint Graph Coloring
    std::vector< std::vector<uint> > tmp_clusterd_constraint_stretch_mass_spring;
    std::vector< std::vector<uint> > tmp_clusterd_constraint_bending;
    {
        auto fn_graph_coloring_sequenced_constraint = [](const uint num_elements, const std::string& constraint_name, 
            std::vector< std::vector<uint> > & clusterd_constraint, 
            const std::vector< std::vector<uint> > & vert_adj_elements, const auto& element_indices, const uint nv)
        { 
            std::vector< bool > marked_constrains(num_elements, false);
            uint total_marked_count = 0;
        
        
            const uint color_threashold = 2000;
            std::vector<uint> rest_cluster;
        
            //
            // while there exist unmarked constraints
            //     create new set S
            //     clear all particle marks
            //     for all unmarked constraints C
            //      if no adjacent particle is marked
            //          add C to S
            //          mark C
            //          mark all adjacent particles
            //
            
            const bool merge_small_cluster = false;
        
            while (true) 
            {
                std::vector<uint> current_cluster;
                std::vector<bool> current_marked(marked_constrains);
                for (uint id = 0; id < num_elements; id++) 
                {
                    if (current_marked[id]) 
                    {
                        continue;
                    }
                    else 
                    {
                        // Add To Sets
                        marked_constrains[id] = true;
                        current_cluster.push_back(id);
        
                        // Mark
                        current_marked[id] = true;
                        auto element = element_indices[id];
                        for (uint j = 0; j < nv; j++) 
                        {
                            for (const uint& adj_eid : vert_adj_elements[element[j]]) 
                            { 
                                current_marked[adj_eid] = true; 
                            }
                        }
                    }
                }
                
                const uint cluster_size = static_cast<uint>(current_cluster.size());
                total_marked_count += cluster_size;
        
                
                if (merge_small_cluster && cluster_size < color_threashold) 
                {
                    rest_cluster.insert(rest_cluster.end(), current_cluster.begin(), current_cluster.end());
                }
                else 
                {
                    clusterd_constraint.push_back(current_cluster);
                }
                
                if (total_marked_count == num_elements) break;
            }
        
            if (merge_small_cluster && !rest_cluster.empty()) 
            {
                clusterd_constraint.push_back(rest_cluster);
            }
        
            luisa::log_info("Cluster Count of {} = {}", constraint_name, clusterd_constraint.size());
        };

        auto fn_get_prefix = [](auto& prefix_buffer, const std::vector< std::vector<uint> >& clusterd_constraint)
        {
            const uint num_cluster = clusterd_constraint.size();
            prefix_buffer.resize(num_cluster + 1);
            uint prefix = 0;
            for (uint cluster = 0; cluster < num_cluster; cluster++)
            {
                prefix_buffer[cluster] = prefix;
                prefix += clusterd_constraint[cluster].size();
            }
            prefix_buffer[num_cluster] = prefix;
        };

        
        fn_graph_coloring_sequenced_constraint(
            mesh_data->num_edges, 
            "Distance  Spring Constraint", 
            tmp_clusterd_constraint_stretch_mass_spring, 
            mesh_data->vert_adj_edges, mesh_data->sa_edges, 2);

        fn_graph_coloring_sequenced_constraint(
            mesh_data->num_bending_edges, 
            "Bending   Angle  Constraint", 
            tmp_clusterd_constraint_bending, 
            mesh_data->vert_adj_bending_edges, mesh_data->sa_bending_edges, 4);
            
        xpbd_data->num_clusters_stretch_mass_spring = tmp_clusterd_constraint_stretch_mass_spring.size();
        xpbd_data->num_clusters_bending = tmp_clusterd_constraint_bending.size();

        fn_get_prefix(xpbd_data->prefix_stretch_mass_spring, tmp_clusterd_constraint_stretch_mass_spring);
        fn_get_prefix(xpbd_data->prefix_bending, tmp_clusterd_constraint_bending);
        
        upload_2d_csr_from(xpbd_data->clusterd_constraint_stretch_mass_spring, tmp_clusterd_constraint_stretch_mass_spring);
        upload_2d_csr_from(xpbd_data->clusterd_constraint_bending, tmp_clusterd_constraint_bending);

    }

    // Vertex Block Descent Coloring
    {
        // Graph Coloring
        const uint num_verts_total = mesh_data->num_verts;
        xpbd_data->sa_Hf.resize(mesh_data->num_verts * 12);
        xpbd_data->sa_Hf1.resize(mesh_data->num_verts);

        const std::vector< std::vector<uint> >& vert_adj_verts = mesh_data->vert_adj_verts_with_bending;
        std::vector<std::vector<uint>> clusterd_vertices_bending; std::vector<uint> prefix_vertices_bending;

        auto fn_graph_coloring_pervertex = [&](const std::vector< std::vector<uint> >& vert_adj_, std::vector<std::vector<uint>>& clusterd_vertices, std::vector<uint>& prefix_vert)
        {
            std::vector<bool> marked_verts(num_verts_total, false);
            uint total_marked_count = 0;

            while (true) 
            {
                std::vector<uint> current_cluster;
                std::vector<bool> current_marked(marked_verts);

                for (uint vid = 0; vid < num_verts_total; vid++) 
                {
                    if (current_marked[vid]) 
                    {
                        continue;
                    }
                    else 
                    {
                        // Add To Sets
                        marked_verts[vid] = true;
                        current_cluster.push_back(vid);

                        // Mark
                        current_marked[vid] = true;
                        const auto& list = vert_adj_[vid];
                        for (const uint& adj_vid : list) 
                        {
                            current_marked[adj_vid] = true;
                        }
                    }
                }
                clusterd_vertices.push_back(current_cluster);
                total_marked_count += current_cluster.size();

                if (total_marked_count == num_verts_total) break;
            }
            

            prefix_vert.resize(clusterd_vertices.size() + 1); uint curr_prefix = 0;
            for (uint cluster = 0; cluster < clusterd_vertices.size(); cluster++)
            {
                prefix_vert[cluster] = curr_prefix;
                curr_prefix += clusterd_vertices[cluster].size();
            } prefix_vert[clusterd_vertices.size()] = curr_prefix;
        };

        fn_graph_coloring_pervertex(vert_adj_verts, clusterd_vertices_bending, prefix_vertices_bending);
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
    lcsv::XpbdData<std::vector>* input_data, 
    lcsv::XpbdData<luisa::compute::Buffer>* output_data)
{
    stream
        << upload_buffer(device, output_data->sa_x_tilde, input_data->sa_x_tilde)
        << upload_buffer(device, output_data->sa_x, input_data->sa_x)
        << upload_buffer(device, output_data->sa_v, input_data->sa_v)
        << upload_buffer(device, output_data->sa_v_start, input_data->sa_v_start)
        << upload_buffer(device, output_data->sa_x_start, input_data->sa_x_start)
        << upload_buffer(device, output_data->sa_merged_edges, input_data->sa_merged_edges)
        << upload_buffer(device, output_data->sa_merged_edges_rest_length, input_data->sa_merged_edges_rest_length)
        << upload_buffer(device, output_data->sa_merged_bending_edges, input_data->sa_merged_bending_edges)
        << upload_buffer(device, output_data->sa_merged_bending_edges_angle, input_data->sa_merged_bending_edges_angle)
        << upload_buffer(device, output_data->sa_merged_bending_edges_Q, input_data->sa_merged_bending_edges_Q)
        << upload_buffer(device, output_data->clusterd_constraint_stretch_mass_spring, input_data->clusterd_constraint_stretch_mass_spring)
        << upload_buffer(device, output_data->prefix_stretch_mass_spring, input_data->prefix_stretch_mass_spring)
        << upload_buffer(device, output_data->sa_lambda_stretch_mass_spring, input_data->sa_lambda_stretch_mass_spring) //
        << upload_buffer(device, output_data->clusterd_constraint_bending, input_data->clusterd_constraint_bending)
        << upload_buffer(device, output_data->prefix_bending, input_data->prefix_bending)
        << upload_buffer(device, output_data->sa_lambda_bending, input_data->sa_lambda_bending) //
        << upload_buffer(device, output_data->prefix_per_vertex_bending, input_data->prefix_per_vertex_bending)
        << upload_buffer(device, output_data->clusterd_per_vertex_bending, input_data->clusterd_per_vertex_bending)
        << upload_buffer(device, output_data->per_vertex_bending_cluster_id, input_data->per_vertex_bending_cluster_id)
        << upload_buffer(device, output_data->sa_Hf, input_data->sa_Hf)
        << upload_buffer(device, output_data->sa_Hf1, input_data->sa_Hf1)
        << luisa::compute::synchronize();
}

}


}