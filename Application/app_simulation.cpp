#include <iostream>
#include <luisa/luisa-compute.h>


#include "Core/sugar.h"
#include "Core/xbasic_types.h"
#include "Core/constant_value.h"
#include "Core/affine_position.h"
#include "Core/float_n.h"
#include "Core/float_nxn.h"
#include "Utils/cpu_parallel.h"
#include "Utils/device_parallel.h"
#include "Utils/buffer_filler.h"

#include "Energy/bending_energy.h"

#include "MeshOperation/mesh_reader.h"
#include "SimulationCore/scene_params.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/solver_interface.h"
#include "SimulationSolver/descent_solver.h"

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>

template<typename T>
using Buffer = luisa::compute::Buffer<T>;

using uint = unsigned int;
using float2 = luisa::float2;
using float3 = luisa::float3;
using float4 = luisa::float4;
using uint2 = luisa::uint2;
using uint3 = luisa::uint3;
using uint4 = luisa::uint4;
using uchar = luisa::uchar;
using float4x4 = luisa::float4x4;
using float4x3 = luisa::float4x3;


template<typename T>
inline void upload_from(std::vector<T>& dest, const std::vector<T>& input_data) 
{ 
    dest.resize(input_data.size());
    std::memcpy(dest.data(), input_data.data(), dest.size() * sizeof(T));  
}
inline uint upload_2d_csr_from(std::vector<uint>& dest, const std::vector<std::vector<uint>>& input_map) 
{
    uint num_outer = input_map.size();
    uint current_prefix = num_outer + 1;
    
    std::vector<uint> prefix_list(num_outer + 1);

    uint max_count = 0;
    for (uint i = 0; i < num_outer; i++) 
    {
        const auto& inner_list = input_map[i];
        uint num_inner = inner_list.size(); max_count = std::max(max_count, num_inner);
        prefix_list[i] = current_prefix;
        current_prefix += num_inner;
    }
    uint num_data = current_prefix;
    prefix_list[num_outer] = current_prefix;
    
    dest.resize(num_data);
    std::memcpy(dest.data(), prefix_list.data(), (num_outer + 1) * sizeof(uint));

    for (uint i = 0; i < num_outer; i++) 
    {
        const auto& inner_list = input_map[i];
        uint current_prefix = prefix_list[i];
        uint current_end = prefix_list[i + 1];
        for (uint j = current_prefix; j < current_end; j++) 
        {
            dest[j] = inner_list[j - current_prefix];
        }
    }
    return max_count;
}

/*
template<typename T>
inline void upload_from(luisa::compute::Buffer<T>& dest, const std::vector<T>& input_data, 
    luisa::compute::Device& device, luisa::compute::Stream& stream) 
{ 
    dest = device.create_buffer<T>(input_data.size());
    stream << dest.copy_from(input_data.data());
}
inline uint upload_2d_csr_from(luisa::compute::Buffer<uint>& dest, const std::vector<std::vector<uint>>& input_map, 
    luisa::compute::Device& device, luisa::compute::Stream& stream) 
{
    uint num_outer = input_map.size();
    uint current_prefix = num_outer + 1;
    std::vector<uint> prefix_list(num_outer + 1);

    uint max_count = 0;
    for (uint i = 0; i < num_outer; i++) 
    {
        const auto& inner_list = input_map[i];
        uint num_inner = inner_list.size(); max_count = std::max(max_count, num_inner);
        prefix_list[i] = current_prefix;
        current_prefix += num_inner;
    }
    uint num_data = current_prefix;
    prefix_list[num_outer] = current_prefix;
    
    dest = device.create_buffer<uint>(num_data);
    auto prefix_part = dest.view(0, num_outer + 1); 
    std::vector<uint> tmp_for_upload(num_data);

    for (uint i = 0; i < num_outer; i++) 
    {
        const auto& inner_list = input_map[i];
        uint current_prefix = prefix_list[i];
        uint current_end = prefix_list[i + 1];
        for (uint j = current_prefix; j < current_end; j++) 
        {
            tmp_for_upload[j] = inner_list[j - current_prefix];
        }
    }
    auto indices_part = dest.view(num_outer + 1, num_data - num_outer - 1);

    stream << prefix_part.copy_from(prefix_list.data()) 
           << indices_part.copy_from(tmp_for_upload.data() + num_outer + 1);

    return max_count;
}
*/

template<typename T>
static inline auto upload_buffer(luisa::compute::Device& device, Buffer<T>& dest, const std::vector<T>& src)
{
    dest = device.create_buffer<T>(src.size());
    return dest.copy_from(src.data());
};
template<typename T>
static inline auto resize_buffer(luisa::compute::Device& device, Buffer<T>& dest, const std::vector<T>& src)
{
    dest = device.create_buffer<T>(src.size());
};

namespace Initializater
{

// template<template<typename> typename BasicBuffer>
void init_mesh_data(lcsv::MeshData<std::vector>* mesh_data)
{
    std::string model_name = "square8K.obj";
    float3 transform = luisa::make_float3(0.0f);
    float3 rotation = luisa::make_float3(0.0f * lcsv::Pi);
    float3 scale = luisa::make_float3(1.0f);


    SimMesh::TriangleMeshData input_mesh;
    bool second_read = SimMesh::read_mesh_file(model_name, input_mesh, true);

    std::string obj_name = model_name;
    {
        std::filesystem::path path(obj_name);
        obj_name = path.stem().string();
    }

    const uint num_verts = input_mesh.model_positions.size();
    const uint num_faces = input_mesh.faces.size();
    const uint num_edges = input_mesh.edges.size();
    const uint num_bending_edges = input_mesh.bending_edges.size();

    luisa::log_info("Cloth : (numVerts : {}) (numFaces : {})  (numEdges : {}) (numBendingEdges : {})", 
        num_verts, num_faces, num_edges, num_bending_edges);

    // Constant scalar
    {
        mesh_data->num_verts = num_verts;
        mesh_data->num_faces = num_faces;
        mesh_data->num_edges = num_edges;  
        mesh_data->num_bending_edges = num_bending_edges;
    }
    
    // Core information
    {
        mesh_data->sa_rest_x.resize(num_verts);
        mesh_data->sa_faces.resize(num_faces);
        mesh_data->sa_edges.resize(num_edges);
        mesh_data->sa_bending_edges.resize(num_bending_edges);
        
        CpuParallel::parallel_for(0, mesh_data->num_verts, [&](const uint vid)
        {
            auto pos = input_mesh.model_positions[vid];
            mesh_data->sa_rest_x[vid] = luisa::make_float3(pos[0], pos[1], pos[2]);
        });
        CpuParallel::parallel_for(0, mesh_data->num_faces, [&](const uint vid)
        {
            auto face = input_mesh.faces[vid];
            mesh_data->sa_faces[vid] = luisa::make_uint3(face[0], face[1], face[2]);
        });
        CpuParallel::parallel_for(0, mesh_data->num_edges, [&](const uint vid)
        {
            auto edge = input_mesh.edges[vid];
            mesh_data->sa_edges[vid] = luisa::make_uint2(edge[0], edge[1]);
        });
        CpuParallel::parallel_for(0, mesh_data->num_bending_edges, [&](const uint vid)
        {
            auto pos = input_mesh.bending_edges[vid];
            mesh_data->sa_bending_edges[vid] = luisa::make_uint4(pos[0], pos[1], pos[2], pos[3]);
        });
    }
    
    // Init vert info
    {
        // Set rest position & velocity
        {
            mesh_data->sa_rest_v.resize(num_verts); 
            CpuParallel::parallel_for(0, num_verts, [&](const uint vid)
            {
                float3 model_position = mesh_data->sa_rest_x[vid];
                float4x4 model_matrix = lcsv::make_model_matrix(transform, rotation, scale);
                float3 world_position = lcsv::affine_position(model_matrix, model_position);
                mesh_data->sa_rest_x[vid] = world_position;
                mesh_data->sa_rest_v[vid] = luisa::make_float3(0.0f);
            });
        }

        // Set fixed-points
        {
            mesh_data->sa_is_fixed.resize(num_verts);

            struct AABB
            {
                float3 packed_min;
                float3 packed_max;
                AABB operator+(const AABB& input_aabb) const{
                    AABB tmp;
                    tmp.packed_min = lcsv::min_vec(packed_min, input_aabb.packed_min);
                    tmp.packed_max = lcsv::max_vec(packed_max, input_aabb.packed_max);
                    return tmp;
                }
            };
            
            AABB local_aabb = CpuParallel::parallel_for_and_reduce_sum<AABB>(0, mesh_data->sa_rest_x.size(), [&](const uint vid)
            {
                auto pos = mesh_data->sa_rest_x[vid];
                return AABB{
                    .packed_min = pos,
                    .packed_max = pos,
                };
            });

            auto pos_min = local_aabb.packed_min;
            auto pos_max = local_aabb.packed_max;
            auto pos_dim_inv = 1.0f / luisa::max(pos_max - pos_min, 0.0001f);

            CpuParallel::parallel_for(0, mesh_data->sa_rest_x.size(), [&](const uint vid)
            {
                float3 orig_pos = mesh_data->sa_rest_x[vid];
                float3 norm_pos = (orig_pos - pos_min) * pos_dim_inv;
                
                bool is_fixed = false;
                is_fixed = norm_pos.z < 0.01f && (norm_pos.x > 0.99f || norm_pos.x < 0.01f ) ;
                mesh_data->sa_is_fixed[vid] = is_fixed;
            });
        }

        // Set vert mass
        {
            mesh_data->sa_vert_mass.resize(num_verts); 
            mesh_data->sa_vert_mass_inv.resize(num_verts);

            const float defulat_density = 0.01f;
            const float default_mass = 1.0f;
            const float defulat_mass = defulat_density * default_mass;
            CpuParallel::parallel_for(0, num_verts, [&](const uint vid)
            {
                bool is_fixed = mesh_data->sa_is_fixed[vid] != 0;
                mesh_data->sa_vert_mass[vid] = (defulat_mass);
                mesh_data->sa_vert_mass_inv[vid] = is_fixed ? 0.0f : 1.0f / (defulat_mass);
            });
        }

    }

    // Init adjacent list
    {
        mesh_data->vert_adj_faces.resize(num_verts);
        mesh_data->vert_adj_edges.resize(num_verts);
        mesh_data->vert_adj_bending_edges.resize(num_verts);

        // Vert adj faces
        for (uint eid = 0; eid < num_faces; eid++)
        {
            auto edge = mesh_data->sa_faces[eid];
            for (uint j = 0; j < 3; j++)
                mesh_data->vert_adj_faces[edge[j]].push_back(eid);
        } 
        upload_2d_csr_from(mesh_data->sa_vert_adj_faces_csr, mesh_data->vert_adj_faces); 

        // Vert adj edges
        for (uint eid = 0; eid < num_edges; eid++)
        {
            auto edge = mesh_data->sa_edges[eid];
            for (uint j = 0; j < 2; j++)
                mesh_data->vert_adj_edges[edge[j]].push_back(eid);
        } 
        upload_2d_csr_from(mesh_data->sa_vert_adj_edges_csr, mesh_data->vert_adj_edges);

        // Vert adj bending-edges
        for (uint eid = 0; eid < num_bending_edges; eid++)
        {
            auto edge = mesh_data->sa_bending_edges[eid];
            for (uint j = 0; j < 4; j++)
                mesh_data->vert_adj_bending_edges[edge[j]].push_back(eid);
        }  
        upload_2d_csr_from(mesh_data->sa_vert_adj_bending_edges_csr, mesh_data->vert_adj_bending_edges);

        // Vert adj verts based on 1-order connection
        mesh_data->vert_adj_verts.resize(num_verts);
        for (uint eid = 0; eid < num_edges; eid++)
        {
            auto edge = mesh_data->sa_edges[eid];
            for (uint j = 0; j < 2; j++)
            {
                const uint left = edge[j];
                const uint right = edge[1 - j];
                mesh_data->vert_adj_verts[left].push_back(right);
            }
        } 
        upload_2d_csr_from(mesh_data->sa_vert_adj_verts_csr, mesh_data->vert_adj_verts);
        
        // Vert adj verts based on 1-order bending-connection
        auto insert_adj_vert = [](std::vector<std::vector<uint>>& adj_map, const uint& vid1, const uint& vid2) 
        {
            if (vid1 == vid2) std::cerr << "redudant!";
            auto& inner_list = adj_map[vid1];
            auto find_result = std::find(inner_list.begin(), inner_list.end(), vid2);
            if (find_result == inner_list.end())
            {
                inner_list.push_back(vid2);
            }
        };
        mesh_data->vert_adj_verts_with_bending = mesh_data->vert_adj_verts;
        for (uint eid = 0; eid < mesh_data->num_bending_edges; eid++)
        {
            const uint4 edge = mesh_data->sa_bending_edges[eid];
            for (size_t i = 0; i < 4; i++) 
            {
                for (size_t j = 0; j < 4; j++)
                {
                    if (i != j) { insert_adj_vert(mesh_data->vert_adj_verts_with_bending, edge[i], edge[j]); }
                    if (i != j) { if (edge[i] == edge[j])
                    {
                        luisa::log_info("Redundant Edge {} : {} & {}", eid, edge[i], edge[j]);
                    } }
                }
            }
        }
        upload_2d_csr_from(mesh_data->sa_vert_adj_verts_with_bending_csr, mesh_data->vert_adj_verts_with_bending);
    }

    // Init energy
    {
        // Rest spring length
        mesh_data->sa_edges_rest_state_length.resize(num_edges);
        CpuParallel::parallel_for(0, num_edges, [&](const uint eid)
        {
            uint2 edge = mesh_data->sa_edges[eid];
            float3 x1 = mesh_data->sa_rest_x[edge[0]];
            float3 x2 = mesh_data->sa_rest_x[edge[1]];
            mesh_data->sa_edges_rest_state_length[eid] = lcsv::length_vec(x1 - x2); /// 
        });

        // Rest bending info
        mesh_data->sa_bending_edges_rest_angle.resize(num_bending_edges);
        mesh_data->sa_bending_edges_Q.resize(num_bending_edges);
        CpuParallel::parallel_for(0, num_bending_edges, [&](const uint eid)
        {
            const uint4 edge = mesh_data->sa_bending_edges[eid];
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
                const float angle = lcsv::BendingEnergy::CalcGradientsAndAngle(x1, x2, x3, x4, tmp, tmp, tmp, tmp);
                if (luisa::isnan(angle)) luisa::log_error("is nan rest angle {}", eid);
    
                mesh_data->sa_bending_edges_rest_angle[eid] = angle; 
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
                const float4x4 m_Q = (3.f / (A_0 + A_1)) * lcsv::outer_product(K, K); // Q = 3 qq^T / (A0+A1) ==> Q is symmetric
                mesh_data->sa_bending_edges_Q[eid] = m_Q; // See : A quadratic bending model for inextensible surfaces.
            }
        });
    }

    // Init vert status
    {
        mesh_data->sa_x_frame_start.resize(num_verts); mesh_data->sa_x_frame_start = mesh_data->sa_rest_x;
        mesh_data->sa_v_frame_start.resize(num_verts); mesh_data->sa_v_frame_start = mesh_data->sa_rest_v;

        mesh_data->sa_x_frame_end.resize(num_verts); mesh_data->sa_x_frame_end = mesh_data->sa_rest_x;
        mesh_data->sa_v_frame_end.resize(num_verts); mesh_data->sa_v_frame_end = mesh_data->sa_rest_v;

        mesh_data->sa_x_frame_saved.resize(num_verts); mesh_data->sa_x_frame_saved = mesh_data->sa_rest_x;
        mesh_data->sa_v_frame_saved.resize(num_verts); mesh_data->sa_v_frame_saved = mesh_data->sa_rest_v;

        mesh_data->sa_system_energy.resize(10240);
    }
    
}
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

void upload_mesh_buffers(
    luisa::compute::Device& device, 
    luisa::compute::Stream& stream, 
    lcsv::MeshData<std::vector>* input_data, 
    lcsv::MeshData<luisa::compute::Buffer>* output_data)
{
    output_data->num_verts = input_data->num_verts;
    output_data->num_faces = input_data->num_faces;
    output_data->num_edges = input_data->num_edges;
    output_data->num_bending_edges = input_data->num_bending_edges;

    stream 
        << upload_buffer(device, output_data->sa_rest_x, input_data->sa_rest_x)
        << upload_buffer(device, output_data->sa_rest_v, input_data->sa_rest_v)
        << upload_buffer(device, output_data->sa_faces, input_data->sa_faces)
        << upload_buffer(device, output_data->sa_edges, input_data->sa_edges)
        << upload_buffer(device, output_data->sa_bending_edges, input_data->sa_bending_edges)
        << upload_buffer(device, output_data->sa_vert_mass, input_data->sa_vert_mass)
        << upload_buffer(device, output_data->sa_vert_mass_inv, input_data->sa_vert_mass_inv)
        << upload_buffer(device, output_data->sa_is_fixed, input_data->sa_is_fixed)
        << upload_buffer(device, output_data->sa_edges_rest_state_length, input_data->sa_edges_rest_state_length)
        << upload_buffer(device, output_data->sa_bending_edges_rest_angle, input_data->sa_bending_edges_rest_angle)
        << upload_buffer(device, output_data->sa_bending_edges_Q, input_data->sa_bending_edges_Q)
        // No std::vector<std::vector<uint>> vert_adj_verts info
        << upload_buffer(device, output_data->sa_vert_adj_verts_csr, input_data->sa_vert_adj_verts_csr) 
        << upload_buffer(device, output_data->sa_vert_adj_verts_with_bending_csr, input_data->sa_vert_adj_verts_with_bending_csr) 
        << upload_buffer(device, output_data->sa_vert_adj_faces_csr, input_data->sa_vert_adj_faces_csr) 
        << upload_buffer(device, output_data->sa_vert_adj_edges_csr, input_data->sa_vert_adj_edges_csr) 
        << upload_buffer(device, output_data->sa_vert_adj_bending_edges_csr, input_data->sa_vert_adj_bending_edges_csr) 
        << upload_buffer(device, output_data->sa_system_energy, input_data->sa_system_energy) 
        << luisa::compute::synchronize();
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
        << luisa::compute::synchronize();
}

void init_simulation_params()
{
    lcsv::get_scene_params().print_xpbd_convergence = false; // false true

    if (lcsv::get_scene_params().use_substep)
    {
        lcsv::get_scene_params().implicit_dt = 1.f / 60.f;
    }
    else 
    {
        lcsv::get_scene_params().num_substep = 1;
        lcsv::get_scene_params().constraint_iter_count = 200;
    }

    if (lcsv::get_scene_params().use_small_timestep) { lcsv::get_scene_params().implicit_dt = 0.001f; }
    
    lcsv::get_scene_params().num_iteration = lcsv::get_scene_params().num_substep * lcsv::get_scene_params().constraint_iter_count;
    lcsv::get_scene_params().collision_detection_frequece = 1;    

    // lcsv::get_scene_params().stiffness_stretch_spring = FEM::calcSecondLame(lcsv::get_scene_params().youngs_modulus_cloth, lcsv::get_scene_params().poisson_ratio_cloth); // mu;
    // lcsv::get_scene_params().stiffness_pressure = 1e6;
    
    {
        // lcsv::get_scene_params().stiffness_stretch_spring = 1e4;
        // lcsv::get_scene_params().xpbd_stiffness_collision = 1e7;
        lcsv::get_scene_params().stiffness_quadratic_bending = 5e-3;
        lcsv::get_scene_params().stiffness_DAB_bending = 5e-3;
    }

}


}

static uint energy_idx = 0; 





enum SolverType
{
    SolverTypeGaussNewton,
    SolverTypeXPBD_CPU,
    SolverTypeVBD_CPU,
    SolverTypeVBD_async,
};

#include <glm/glm.hpp>

int main(int argc, char** argv)
{
    luisa::log_level_info();
    std::cout << "Hello, LuisaComputeSimulation!" << std::endl;
    
    // Init GPU system
#if defined(__APPLE__)
    std::string    backend          = "metal";
#else
    std::string    backend          = "cuda";
#endif
    luisa::compute::Context context{ argv[0] };
    luisa::compute::Device device = context.create_device(backend);
    luisa::compute::Stream stream = device.create_stream(luisa::compute::StreamTag::COMPUTE);

    // Init data
    lcsv::MeshData<std::vector> cpu_mesh_data;
    lcsv::MeshData<luisa::compute::Buffer> mesh_data;
    {
        Initializater::init_mesh_data(&cpu_mesh_data);
        Initializater::upload_mesh_buffers(device, stream, &cpu_mesh_data, &mesh_data);
    }

    lcsv::XpbdData<std::vector> cpu_xpbd_data;
    lcsv::XpbdData<luisa::compute::Buffer> xpbd_data;
    {
        Initializater::init_xpbd_data(&cpu_mesh_data, &cpu_xpbd_data);
        Initializater::upload_xpbd_buffers(device, stream, &cpu_xpbd_data, &xpbd_data);
        Initializater::init_simulation_params();
    }

    // Init solver class
    lcsv::BufferFiller   buffer_filler;
    lcsv::DeviceParallel device_parallel;
    lcsv::DescentSolverCPU solver;
    {
        device_parallel.create(device);
        solver.lcsv::SolverInterface::set_data_pointer(
            &cpu_mesh_data, 
            &mesh_data, 
            &cpu_xpbd_data, 
            &xpbd_data, 
            &buffer_filler, 
            &device_parallel
        );
        solver.compile(device);
    }

    // Some params
    {
        lcsv::get_scene_params().use_substep = false;
        lcsv::get_scene_params().num_substep = 1;
        lcsv::get_scene_params().constraint_iter_count = 100; // 
        lcsv::get_scene_params().use_bending = true;
        lcsv::get_scene_params().use_quadratic_bending_model = true;
        lcsv::get_scene_params().print_xpbd_convergence = false;
        lcsv::get_scene_params().use_xpbd_solver = false;
        lcsv::get_scene_params().use_vbd_solver = true;
    }

    // Init GUI
    std::vector<glm::vec3> sa_rendering_vertices(cpu_mesh_data.num_verts);
    std::vector<std::vector<uint>> sa_rendering_faces(cpu_mesh_data.num_faces);
    {
        CpuParallel::parallel_for(0, cpu_mesh_data.num_verts, [&](const uint vid)
        {
            auto pos = cpu_mesh_data.sa_rest_x[vid];
            sa_rendering_vertices[vid] = glm::vec3(pos.x, pos.y, pos.z);
        });
        CpuParallel::parallel_for(0, cpu_mesh_data.num_faces, [&](const uint fid)
        {
            auto face = cpu_mesh_data.sa_faces[fid];
            sa_rendering_faces[fid] = {face[0], face[1], face[2]};
        });
    }
    // polyscope::init("openGL_mock");
    polyscope::init("openGL3_glfw");

    auto surface_mesh = polyscope::registerSurfaceMesh("cloth1", sa_rendering_vertices, sa_rendering_faces);
    // surface_mesh->setEnabled(false);

    polyscope::show();

    const uint num_frames = 30;
    
    // Synchronous Implementation
    {
        solver.lcsv::SolverInterface::restart_system();
        solver.lcsv::SolverInterface::save_mesh_to_obj(0, "_init"); 
        luisa::log_info("");
        luisa::log_info("");
        luisa::log_info("Sync part");
    }
    {   
        for (uint frame = 0; frame < num_frames; frame++)
        {   lcsv::get_scene_params().current_frame = frame; luisa::log_info("     Sync frame {}", frame);   

            solver.physics_step_vbd(device, stream);
        }
    }
    {
        solver.lcsv::SolverInterface::save_mesh_to_obj(lcsv::get_scene_params().current_frame, "_sync"); 
    }   


    return 0;
}