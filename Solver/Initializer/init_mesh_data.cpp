#include "Initializer/init_mesh_data.h"
#include "Core/affine_position.h"
#include "Core/float_nxn.h"
#include "Energy/bending_energy.h"
#include "MeshOperation/mesh_reader.h"
#include "Initializer/initializer_utils.h"
#include "Utils/cpu_parallel.h"
#include <algorithm>
#include <numeric>

namespace lcsv 
{

namespace Initializer
{



// template<template<typename> typename BasicBuffer>
void init_mesh_data(
    std::vector<lcsv::Initializer::ShellInfo>& shell_infos, 
    lcsv::MeshData<std::vector>* mesh_data)
{
    std::sort(shell_infos.begin(), shell_infos.end(), 
    [](const Initializer::ShellInfo& left, const Initializer::ShellInfo& right)
    {
        return int(left.shell_type) < int(right.shell_type);
    });
    const uint num_meshes = shell_infos.size();
    std::vector<SimMesh::TriangleMeshData> input_meshes(num_meshes);

    mesh_data->num_verts = 0;
    mesh_data->num_faces = 0;
    mesh_data->num_edges = 0;
    mesh_data->num_bending_edges = 0;

    mesh_data->prefix_num_verts.resize(1 + num_meshes, 0);
    mesh_data->prefix_num_faces.resize(1 + num_meshes, 0);
    mesh_data->prefix_num_edges.resize(1 + num_meshes, 0);
    mesh_data->prefix_num_bending_edges.resize(1 + num_meshes, 0);

    uint num_verts_cloth = 0;
    uint num_faces_cloth = 0;
    uint num_edges_cloth = 0;
    uint num_bending_edges_cloth = 0;
    uint num_verts_tetrahedral = 0;
    uint num_faces_tetrahedral = 0;
    uint num_edges_tetrahedral = 0;
    uint num_bending_edges_tetrahedral = 0;
    uint num_verts_obstacle = 0;
    uint num_faces_obstacle = 0;
    uint num_edges_obstacle = 0;
    uint num_bending_edges_obstacle = 0;

    // Constant scalar and init MeshData
    // TODO: Identity cloth, tet, rigid-body
    for (uint meshIdx = 0; meshIdx < num_meshes; meshIdx++)
    {
        const auto& shell_info = shell_infos[meshIdx];
        auto& input_mesh = input_meshes[meshIdx]; // TODO: Get (multiple) original mesh data from params
        bool second_read = SimMesh::read_mesh_file(shell_info.model_name, input_mesh);

        // std::string obj_name = model_name;
        // {
        //     std::filesystem::path path(obj_name);
        //     obj_name = path.stem().string();
        // }

        mesh_data->prefix_num_verts[meshIdx] = mesh_data->num_verts;
        mesh_data->prefix_num_faces[meshIdx] = mesh_data->num_faces;
        mesh_data->prefix_num_edges[meshIdx] = mesh_data->num_edges;
        mesh_data->prefix_num_bending_edges[meshIdx] = mesh_data->num_bending_edges;

        const uint curr_num_verts = input_mesh.model_positions.size();
        const uint curr_num_faces = input_mesh.faces.size();
        const uint curr_num_edges = input_mesh.edges.size();
        const uint curr_num_bending_edges = input_mesh.bending_edges.size();
        
        if (shell_info.shell_type == ShellTypeCloth)
        {
            num_verts_cloth += curr_num_verts;
            num_faces_cloth += curr_num_faces;
            num_edges_cloth += curr_num_edges;
            num_bending_edges_cloth += curr_num_bending_edges;
            
        }
        else if (shell_info.shell_type == ShellTypeTetrahedral)
        {
            num_verts_tetrahedral += curr_num_verts;
            num_faces_tetrahedral += curr_num_faces;
            num_edges_tetrahedral += curr_num_edges;
            num_bending_edges_tetrahedral += curr_num_bending_edges;
        }
        else if (shell_info.shell_type == ShellTypeObstacle)
        {
            num_verts_obstacle += curr_num_verts;
            num_faces_obstacle += curr_num_faces;
            num_edges_obstacle += curr_num_edges;
            num_bending_edges_obstacle += curr_num_bending_edges;
        }

        mesh_data->num_verts += curr_num_verts;
        mesh_data->num_faces += curr_num_faces;
        mesh_data->num_edges += curr_num_edges;
        mesh_data->num_bending_edges += curr_num_bending_edges;
        // mesh_data->num_tets += shell_info.shell_type == ShellTypeTetrahedral ? input_mesh.tets.size() : 0;
    }

    mesh_data->prefix_num_verts[num_meshes] = mesh_data->num_verts;
    mesh_data->prefix_num_faces[num_meshes] = mesh_data->num_faces;
    mesh_data->prefix_num_edges[num_meshes] = mesh_data->num_edges;
    mesh_data->prefix_num_bending_edges[num_meshes] = mesh_data->num_bending_edges;

    mesh_data->range_verts_cloth = {0, num_verts_cloth};
    mesh_data->range_faces_cloth = {0, num_faces_cloth};
    mesh_data->range_edges_cloth = {0, num_edges_cloth};
    mesh_data->range_bending_edges_cloth = {0, num_bending_edges_cloth};
    mesh_data->range_verts_tetrahedral = {num_verts_cloth, num_verts_cloth + num_verts_tetrahedral};
    mesh_data->range_faces_tetrahedral = {num_faces_cloth, num_faces_cloth + num_faces_tetrahedral};
    mesh_data->range_edges_tetrahedral = {num_edges_cloth, num_edges_cloth + num_edges_tetrahedral};
    mesh_data->range_bending_edges_tetrahedral = {num_bending_edges_cloth, num_bending_edges_cloth + num_bending_edges_tetrahedral};
    mesh_data->range_verts_obstacle = {num_verts_cloth + num_verts_tetrahedral, num_verts_cloth + num_verts_tetrahedral + num_verts_obstacle};
    mesh_data->range_faces_obstacle = {num_faces_cloth + num_faces_tetrahedral, num_faces_cloth + num_faces_tetrahedral + num_faces_obstacle};
    mesh_data->range_edges_obstacle = {num_edges_cloth + num_edges_tetrahedral, num_edges_cloth + num_edges_tetrahedral + num_edges_obstacle};
    mesh_data->range_bending_edges_obstacle = {num_bending_edges_cloth + num_bending_edges_tetrahedral, num_bending_edges_cloth + num_bending_edges_tetrahedral + num_bending_edges_obstacle};

    uint num_verts = mesh_data->num_verts;
    uint num_faces = mesh_data->num_faces;
    uint num_edges = mesh_data->num_edges;
    uint num_bending_edges = mesh_data->num_bending_edges;

    luisa::log_info("Cloth : (numVerts : {}) (numFaces : {})  (numEdges : {}) (numBendingEdges : {})", 
        num_verts, num_faces, num_edges, num_bending_edges);
    
    // Read information
    {
        mesh_data->sa_rest_x.resize(num_verts);
        mesh_data->sa_faces.resize(num_faces);
        mesh_data->sa_edges.resize(num_edges);
        mesh_data->sa_bending_edges.resize(num_bending_edges);

        mesh_data->sa_rest_v.resize(num_verts);
        mesh_data->sa_is_fixed.resize(num_verts);

        uint prefix_num_verts = 0;
        uint prefix_num_faces = 0;
        uint prefix_num_edges = 0;
        uint prefix_num_bending_edges = 0;

        for (uint meshIdx = 0; meshIdx < num_meshes; meshIdx++)
        {
            auto& curr_shell_info = shell_infos[meshIdx];
            const auto& curr_input_mesh = input_meshes[meshIdx];

            const uint curr_num_verts = curr_input_mesh.model_positions.size();
            const uint curr_num_faces = curr_input_mesh.faces.size();
            const uint curr_num_edges = curr_input_mesh.edges.size();
            const uint curr_num_bending_edges = curr_input_mesh.bending_edges.size();

            // Read position with affine
            CpuParallel::parallel_for(0, curr_num_verts, [&](const uint vid)
            {
                std::array<float, 3> read_pos = curr_input_mesh.model_positions[vid];
                float3 model_position = luisa::make_float3(read_pos[0], read_pos[1], read_pos[2]);
                float4x4 model_matrix = lcsv::make_model_matrix(curr_shell_info.transform, curr_shell_info.rotation, curr_shell_info.scale);
                float3 world_position = lcsv::affine_position(model_matrix, model_position);
                mesh_data->sa_rest_x[prefix_num_verts + vid] = world_position;
                mesh_data->sa_rest_v[prefix_num_verts + vid] = luisa::make_float3(0.0f);
            });
            // Read triangle face
            CpuParallel::parallel_for(0, curr_num_faces, [&](const uint fid)
            {
                auto face = curr_input_mesh.faces[fid];
                mesh_data->sa_faces[prefix_num_faces + fid] = prefix_num_verts + luisa::make_uint3(face[0], face[1], face[2]);
            });
            // Read edge
            CpuParallel::parallel_for(0, curr_num_edges, [&](const uint eid)
            {
                auto edge = curr_input_mesh.edges[eid];
                mesh_data->sa_edges[prefix_num_edges + eid] = prefix_num_verts + luisa::make_uint2(edge[0], edge[1]);
            });
            // Read bending edge
            CpuParallel::parallel_for(0, curr_num_bending_edges, [&](const uint eid)
            {
                auto bending_edge = curr_input_mesh.bending_edges[eid];
                mesh_data->sa_bending_edges[prefix_num_bending_edges + eid] = prefix_num_verts + luisa::make_uint4(bending_edge[0], bending_edge[1], bending_edge[2], bending_edge[3]);
            });

            // Set fixed-points
            {
                struct AABB 
                {
                    float3 packed_min; float3 packed_max;
                    AABB operator+(const AABB& input_aabb) const 
                    {
                        AABB tmp;
                        tmp.packed_min = lcsv::min_vec(packed_min, input_aabb.packed_min);
                        tmp.packed_max = lcsv::max_vec(packed_max, input_aabb.packed_max);
                        return tmp;
                    }
                    AABB() : packed_min(float3(Float_max)), packed_max(float3(-Float_max)) {}
                    AABB(const float3& pos) : packed_min(pos), packed_max(pos) {}
                };

                AABB local_aabb = CpuParallel::parallel_for_and_reduce_sum<AABB>(0, curr_num_verts, [&](const uint vid)
                {
                    auto pos = mesh_data->sa_rest_x[prefix_num_verts + vid];
                    return AABB(pos);
                });

                auto pos_min = local_aabb.packed_min;
                auto pos_max = local_aabb.packed_max;
                auto pos_dim_inv = 1.0f / luisa::max(pos_max - pos_min, 0.0001f);

                float avg_spring_length = CpuParallel::parallel_for_and_reduce_sum<float>(0, curr_num_edges, [&](const uint eid)
                {
                    auto edge = mesh_data->sa_edges[prefix_num_edges + eid];
                    return length_vec(mesh_data->sa_rest_x[edge[0]] - mesh_data->sa_rest_x[edge[1]]);
                }) / float(curr_num_edges);
                luisa::log_info("Mesh {:<2} : numVerts = {:<5}, numFaces = {:<5}, numEdges = {:<5}, avgEdgeLength = {:2.4f}, AABB range = {}", meshIdx, 
                    curr_num_verts, curr_num_faces, curr_num_edges, avg_spring_length,
                    pos_max - pos_min);


                CpuParallel::single_thread_for(0, curr_num_verts, [&](const uint local_vid)
                {
                    const uint global_vid = prefix_num_verts + local_vid;
                    float3 orig_pos = mesh_data->sa_rest_x[global_vid];
                    float3 norm_pos = (orig_pos - pos_min) * pos_dim_inv;
                    
                    bool is_fixed = false;
                    for (auto& fixed_point_info : curr_shell_info.fixed_point_list)
                    {
                        if (fixed_point_info.is_fixed_point_func(norm_pos))
                        {
                            is_fixed = true;
                            fixed_point_info.fixed_point_verts.push_back(global_vid);
                            break;
                        }
                    }
                    // is_fixed = norm_pos.z < 0.001f && (norm_pos.x > 0.999f || norm_pos.x < 0.001f ) ;
                    mesh_data->sa_is_fixed[global_vid] = is_fixed;
                });
            }

            prefix_num_verts += curr_num_verts;
            prefix_num_faces += curr_num_faces;
            prefix_num_edges += curr_num_edges;
            prefix_num_bending_edges += curr_num_bending_edges;
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
        mesh_data->vert_adj_verts_with_material_constraints = mesh_data->vert_adj_verts;
        for (uint eid = 0; eid < mesh_data->num_bending_edges; eid++)
        {
            const uint4 edge = mesh_data->sa_bending_edges[eid];
            for (size_t i = 0; i < 4; i++) 
            {
                for (size_t j = 0; j < 4; j++)
                {
                    if (i != j) { insert_adj_vert(mesh_data->vert_adj_verts_with_material_constraints, edge[i], edge[j]); }
                    if (i != j) { if (edge[i] == edge[j])
                    {
                        luisa::log_info("Redundant Edge {} : {} & {}", eid, edge[i], edge[j]);
                    } }
                }
            }
        }
        upload_2d_csr_from(mesh_data->sa_vert_adj_verts_with_material_constraints_csr, mesh_data->vert_adj_verts_with_material_constraints);

        // face_adj_edges
        // Face adj edges
        mesh_data->face_adj_edges.resize(num_faces);
        mesh_data->face_adj_faces.resize(num_faces);
        mesh_data->edge_adj_faces.resize(num_edges, makeUint2(-1u));
        auto fn_vert_in_face = [](const uint& vid, const uint3& face) { return vid == face[0] || vid == face[1] || vid == face[2]; };
        CpuParallel::parallel_for(0, num_faces, [&](const uint fid)
        {
            std::unordered_set<uint> adj_edges_set; adj_edges_set.reserve(3);
            const auto face = mesh_data->sa_faces[fid];
            uint face_sum_indices = face[0] + face[1] + face[2];
            for (uint j = 0; j < 3; j++)
            {
                const uint vid = face[j];
                const auto& vert_adj_edges = mesh_data->vert_adj_edges[vid];
                for (const uint& adj_eid : vert_adj_edges)
                {
                    const auto adj_edge = mesh_data->sa_edges[adj_eid];
                    if (fn_vert_in_face(adj_edge[0], face) && fn_vert_in_face(adj_edge[1], face))
                    {
                        adj_edges_set.insert(adj_eid);
                    }
                }
            }
            if (adj_edges_set.size() != 3) luisa::log_error("Face {} adj edge count {} != 3", fid, adj_edges_set.size());
            uint3 face_adj_edges; uint idx = 0;
            for (const auto& adj_eid : adj_edges_set) { face_adj_edges[idx++] = adj_eid; }
            mesh_data->face_adj_edges[fid] = face_adj_edges;
        });
        std::vector<uint> edge_adj_face_count(num_edges, 0);
        CpuParallel::single_thread_for(0, num_faces, [&](const uint fid)
        {
            uint3 face_adj_edges = mesh_data->face_adj_edges[fid];
            for (uint j = 0; j < 3; j++)
            {
                uint adj_eid = face_adj_edges[j];
                uint& offset = edge_adj_face_count[adj_eid];
                mesh_data->edge_adj_faces[adj_eid][offset++] = fid;
            }
        });
        CpuParallel::parallel_for(0, num_faces, [&](const uint fid)
        {
            const uint3 face_adj_edges = mesh_data->face_adj_edges[fid];
            uint3 face_adj_faces;
            for (uint j = 0; j < 3; j++)
            {
                uint adj_eid = face_adj_edges[j];
                uint2 edge_adj_faces = mesh_data->edge_adj_faces[adj_eid];
                if (edge_adj_faces[0] != -1u && edge_adj_faces[0] != fid) face_adj_faces[j] = edge_adj_faces[0];
                if (edge_adj_faces[1] != -1u && edge_adj_faces[1] != fid) face_adj_faces[j] = edge_adj_faces[1];
            }
            mesh_data->face_adj_faces[fid] = face_adj_faces;
        });
    }

    // Compute rest area
    {
        mesh_data->sa_rest_vert_area.resize(num_verts);
        mesh_data->sa_rest_edge_area.resize(num_edges);
        mesh_data->sa_rest_face_area.resize(num_faces);
        
        CpuParallel::parallel_for(0, num_faces, [&](const uint fid)
        {
            const uint3 face = mesh_data->sa_faces[fid]; 
            float area = compute_face_area(
                mesh_data->sa_rest_x[face[0]],
                mesh_data->sa_rest_x[face[1]],
                mesh_data->sa_rest_x[face[2]]
            );
            mesh_data->sa_rest_face_area[fid] = area;
        });
        CpuParallel::parallel_for(0, num_verts, [&](const uint vid)
        {
            const auto& adj_faces = mesh_data->vert_adj_faces[vid];
            double area = 0.0;
            for (const uint& adj_fid : adj_faces) area += mesh_data->sa_rest_face_area[adj_fid] / 3.0;
            mesh_data->sa_rest_vert_area[vid] = area;
        });
        CpuParallel::parallel_for(0, num_edges, [&](const uint eid)
        {
            uint2 adj_faces = mesh_data->edge_adj_faces[eid];
            double area = 0.0;
            for (uint j = 0; j < 2; j++)
            {
                uint adj_fid = adj_faces[j];
                if (adj_fid != -1u)
                {
                    area += mesh_data->sa_rest_face_area[adj_fid] / 3.0;
                }
            }
            mesh_data->sa_rest_edge_area[eid] = area;
        });
        float sum_face_area = CpuParallel::parallel_reduce_sum(mesh_data->sa_rest_face_area);
        float sum_edge_area = CpuParallel::parallel_reduce_sum(mesh_data->sa_rest_edge_area);
        float sum_vert_area = CpuParallel::parallel_reduce_sum(mesh_data->sa_rest_vert_area);
        // luisa::log_info("Summary areas : face = {}, edge = {}, vert = {}", sum_face_area, sum_edge_area, sum_vert_area);
        luisa::log_info("Average areas : face = {}, edge = {}, vert = {}", sum_face_area / double(num_faces), sum_edge_area / double(num_edges), sum_vert_area / double(num_verts));
    }

    // Init vert info
    {
        // Set vert mass
        {
            mesh_data->sa_vert_mass.resize(num_verts); 
            mesh_data->sa_vert_mass_inv.resize(num_verts);

            const float defulat_density = 10.0f;
            CpuParallel::parallel_for(0, num_verts, [&](const uint vid)
            {
                bool is_fixed = mesh_data->sa_is_fixed[vid] != 0;
                const float mass = defulat_density * mesh_data->sa_rest_vert_area[vid];
                mesh_data->sa_vert_mass[vid] = mass;
                mesh_data->sa_vert_mass_inv[vid] = is_fixed ? 0.0f : 1.0f / (mass);
            });
        }

    }

    // Init vert status
    {
        mesh_data->sa_x_frame_outer_next.resize(num_verts);
        mesh_data->sa_x_frame_outer.resize(num_verts);
        mesh_data->sa_v_frame_outer.resize(num_verts);
        
        mesh_data->sa_x_frame_saved.resize(num_verts); mesh_data->sa_x_frame_saved = mesh_data->sa_rest_x;
        mesh_data->sa_v_frame_saved.resize(num_verts); mesh_data->sa_v_frame_saved = mesh_data->sa_rest_v;

        CpuParallel::parallel_for(0, num_verts, [&](const uint vid)
        {
            const float3 rest_x = mesh_data->sa_rest_x[vid];
            const float3 rest_v = mesh_data->sa_rest_v[vid];

            mesh_data->sa_x_frame_outer_next[vid] = rest_x;
            mesh_data->sa_x_frame_outer[vid] = rest_x;
            mesh_data->sa_v_frame_outer[vid] = rest_v;
            mesh_data->sa_x_frame_saved[vid] = rest_x;
            mesh_data->sa_v_frame_saved[vid] = rest_v;
        });
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

    output_data->num_tets = input_data->num_tets;
    output_data->range_verts_cloth = input_data->range_verts_cloth;
    output_data->range_faces_cloth = input_data->range_faces_cloth;
    output_data->range_edges_cloth = input_data->range_edges_cloth;
    output_data->range_bending_edges_cloth = input_data->range_bending_edges_cloth;
    output_data->range_verts_tetrahedral = input_data->range_verts_tetrahedral;
    output_data->range_faces_tetrahedral = input_data->range_faces_tetrahedral;
    output_data->range_edges_tetrahedral = input_data->range_edges_tetrahedral;
    output_data->range_bending_edges_tetrahedral = input_data->range_bending_edges_tetrahedral;
    output_data->range_verts_obstacle = input_data->range_verts_obstacle;
    output_data->range_faces_obstacle = input_data->range_faces_obstacle;
    output_data->range_edges_obstacle = input_data->range_edges_obstacle;
    output_data->range_bending_edges_obstacle = input_data->range_bending_edges_obstacle;

    stream 
        << upload_buffer(device, output_data->sa_rest_x, input_data->sa_rest_x)
        << upload_buffer(device, output_data->sa_rest_v, input_data->sa_rest_v)
        << upload_buffer(device, output_data->sa_faces, input_data->sa_faces)
        << upload_buffer(device, output_data->sa_edges, input_data->sa_edges)
        << upload_buffer(device, output_data->sa_bending_edges, input_data->sa_bending_edges)
        << upload_buffer(device, output_data->sa_vert_mass, input_data->sa_vert_mass)
        << upload_buffer(device, output_data->sa_vert_mass_inv, input_data->sa_vert_mass_inv)
        << upload_buffer(device, output_data->sa_is_fixed, input_data->sa_is_fixed)

        << upload_buffer(device, output_data->sa_rest_vert_area, input_data->sa_rest_vert_area)
        << upload_buffer(device, output_data->sa_rest_edge_area, input_data->sa_rest_edge_area)
        << upload_buffer(device, output_data->sa_rest_face_area, input_data->sa_rest_face_area)

        // No std::vector<std::vector<uint>> vert_adj_verts info
        << upload_buffer(device, output_data->sa_vert_adj_verts_csr, input_data->sa_vert_adj_verts_csr) 
        << upload_buffer(device, output_data->sa_vert_adj_verts_with_material_constraints_csr, input_data->sa_vert_adj_verts_with_material_constraints_csr) 
        << upload_buffer(device, output_data->sa_vert_adj_faces_csr, input_data->sa_vert_adj_faces_csr) 
        << upload_buffer(device, output_data->sa_vert_adj_edges_csr, input_data->sa_vert_adj_edges_csr) 
        << upload_buffer(device, output_data->sa_vert_adj_bending_edges_csr, input_data->sa_vert_adj_bending_edges_csr) 
        << upload_buffer(device, output_data->edge_adj_faces, input_data->edge_adj_faces) 
        << upload_buffer(device, output_data->face_adj_edges, input_data->face_adj_edges) 
        << upload_buffer(device, output_data->face_adj_faces, input_data->face_adj_faces) 
        << luisa::compute::synchronize();
}

} // namespace Initializer


} // namespace lcsv 