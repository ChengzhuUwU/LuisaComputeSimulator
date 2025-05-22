#include "Initializer/init_mesh_data.h"
#include "Core/affine_position.h"
#include "Core/float_nxn.h"
#include "Energy/bending_energy.h"
#include "MeshOperation/mesh_reader.h"
#include "Initializer/initializer_utils.h"

namespace lcsv 
{

namespace Initializater
{



// template<template<typename> typename BasicBuffer>
void init_mesh_data(lcsv::MeshData<std::vector>* mesh_data)
{
    std::string model_name = "square8K.obj";
    float3 transform = luisa::make_float3(0.0f, 0.5f, 0.0f);
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
                is_fixed = norm_pos.z < 0.001f && (norm_pos.x > 0.999f || norm_pos.x < 0.001f ) ;
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
        mesh_data->sa_pcg_convergence.resize(10240);
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
        << upload_buffer(device, output_data->sa_pcg_convergence, input_data->sa_pcg_convergence) 
        << luisa::compute::synchronize();
}

}


}