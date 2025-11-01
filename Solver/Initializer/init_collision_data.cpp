#include "Initializer/init_collision_data.h"
#include "CollisionDetector/distance.hpp"
#include <algorithm>

namespace lcs::Initializer
{

void init_collision_data(std::vector<lcs::Initializer::ShellInfo>& shell_infos,
                         lcs::MeshData<std::vector>*               mesh_data,
                         lcs::SimulationData<std::vector>*         sim_data,
                         lcs::CollisionData<std::vector>*          collision_data)
{
    std::vector<uint>& surface_vert_indices = sim_data->sa_contact_active_verts;
    std::vector<uint>& surface_face_indices = sim_data->sa_contact_active_faces;
    std::vector<uint>& surface_edge_indices = sim_data->sa_contact_active_edges;

    surface_vert_indices = fn_get_active_indices(
        [&](const uint vid)
        {
            const uint  mesh_idx   = mesh_data->sa_vert_mesh_id[vid];
            const auto& shell_info = shell_infos[mesh_idx];
            if (shell_info.shell_type == ShellTypeTetrahedral)
            {
                const auto& adj_faces = mesh_data->vert_adj_faces[vid];
                if (!adj_faces.empty())
                    return 1;  // Surface vertex
                return 0;      // Internal vertex
            }
            else
            {
                return 1;  // Surface vertex
            }
        },
        mesh_data->num_verts);
    const uint num_surface_verts = static_cast<uint>(surface_vert_indices.size());

    std::vector<bool> is_surface_vert(mesh_data->num_verts, false);
    for (const auto vid : surface_vert_indices)
    {
        is_surface_vert[vid] = true;
    }

    surface_face_indices = fn_get_active_indices(
        [&](const uint fid)
        {
            const uint  mesh_idx   = mesh_data->sa_face_mesh_id[fid];
            const auto& shell_info = shell_infos[mesh_idx];
            return 1;  // All faces are surface faces
        },
        mesh_data->num_faces);
    const uint num_surface_faces = static_cast<uint>(surface_face_indices.size());

    surface_edge_indices = fn_get_active_indices(
        [&](const uint eid)
        {
            const uint2 edge = mesh_data->sa_edges[eid];
            if (!is_surface_vert[edge[0]] || !is_surface_vert[edge[1]])
                return 0;  // Internal edge
            return 1;      // Surface edge
        },
        mesh_data->num_edges);
    const uint num_surface_edges = static_cast<uint>(surface_edge_indices.size());

    // TODO: Replace contact list with active contact lists


    std::vector<float> mesh_min_dist(mesh_data->num_meshes, 1e10f);
    CpuParallel::parallel_for(0,
                              mesh_data->num_edges,
                              [&](const uint eid)
                              {
                                  const uint2 edge        = mesh_data->sa_edges[eid];
                                  const float rest_length = luisa::length(mesh_data->sa_rest_x[edge[0]]
                                                                          - mesh_data->sa_rest_x[edge[1]]);
                                  const uint  mesh_idx    = mesh_data->sa_edge_mesh_id[eid];
                                  CpuParallel::spin_atomic<float>::fetch_min(mesh_min_dist[mesh_idx], rest_length);
                              });
    CpuParallel::parallel_for(0,
                              mesh_data->num_faces,
                              [&](const uint fid)
                              {
                                  const uint3 face        = mesh_data->sa_faces[fid];
                                  float3      vert_pos[3] = {mesh_data->sa_rest_x[face[0]],
                                                             mesh_data->sa_rest_x[face[1]],
                                                             mesh_data->sa_rest_x[face[2]]};

                                  auto rest_dist = 10000.0f;
                                  for (int i = 0; i < 3; i++)
                                  {
                                      const float3& v0 = vert_pos[i];
                                      const float3& v1 = vert_pos[(i + 1) % 3];
                                      const float3& v2 = vert_pos[(i + 2) % 3];

                                      auto bary = host_distance::point_edge_distance_coeff(
                                          float3_to_eigen3(v0), float3_to_eigen3(v1), float3_to_eigen3(v2));
                                      const float curr_dist = length_vec(bary[0] * v1 + bary[1] * v2 - v0);
                                      rest_dist = min_scalar(rest_dist, curr_dist);
                                  }
                                  const uint mesh_idx = mesh_data->sa_face_mesh_id[fid];
                                  CpuParallel::spin_atomic<float>::fetch_min(mesh_min_dist[mesh_idx], rest_dist);
                              });

    std::vector<float> mesh_scaled_offset(mesh_data->num_meshes);
    std::vector<float> mesh_scaled_d_hat(mesh_data->num_meshes);
    for (uint mesh_idx = 0; mesh_idx < mesh_data->num_meshes; mesh_idx++)
    {
        float thickness = shell_infos[mesh_idx].get_thickness();
        float d_hat     = shell_infos[mesh_idx].get_d_hat();

        float scaled_offset = 0.5f * thickness;
        float scaled_d_hat  = d_hat;

        float       min_dist  = mesh_min_dist[mesh_idx];
        const float safe_dist = 0.9f * min_dist;
        if (scaled_offset + d_hat < min_dist)
        {
        }
        else
        {
            if (scaled_offset > 0.5f * safe_dist)
            {
                LUISA_INFO("For mesh {}: Mesh offset scaled from {} to the safe distance {}",
                           mesh_idx,
                           scaled_offset,
                           0.5f * safe_dist);
                scaled_offset = 0.5f * safe_dist;
            }
            if (scaled_d_hat > safe_dist - scaled_offset)
            {
                LUISA_INFO("For mesh {}: Mesh d_hat scaled from {} to the safe distance {}",
                           mesh_idx,
                           scaled_d_hat,
                           safe_dist - scaled_offset);
                scaled_d_hat = safe_dist - scaled_offset;
            }
        }
        mesh_scaled_offset[mesh_idx] = scaled_offset;
        mesh_scaled_d_hat[mesh_idx]  = scaled_d_hat;
        // mesh_scaled_offset[mesh_idx] = 0.000f;
        // mesh_scaled_d_hat[mesh_idx]  = 0.003f;
    }

    sim_data->sa_contact_active_verts_d_hat.resize(mesh_data->num_verts);
    sim_data->sa_contact_active_verts_offset.resize(mesh_data->num_verts);
    CpuParallel::parallel_for(0,
                              mesh_data->num_verts,
                              [&](const uint vid)
                              {
                                  const uint mesh_idx = mesh_data->sa_vert_mesh_id[vid];
                                  sim_data->sa_contact_active_verts_d_hat[vid] = mesh_scaled_d_hat[mesh_idx];
                                  sim_data->sa_contact_active_verts_offset[vid] = mesh_scaled_offset[mesh_idx];
                                  //   LUISA_INFO("Vertex {}: d_hat = {}, offset = {}",
                                  //              vid,
                                  //              sim_data->sa_contact_active_verts_d_hat[vid],
                                  //              sim_data->sa_contact_active_verts_offset[vid]);
                              });
}
void upload_collision_buffers(luisa::compute::Device&                      device,
                              luisa::compute::Stream&                      stream,
                              lcs::SimulationData<std::vector>*            input_sim_data,
                              lcs::SimulationData<luisa::compute::Buffer>* output_sim_data,
                              lcs::CollisionData<std::vector>*             input_collision_data,
                              lcs::CollisionData<luisa::compute::Buffer>*  output_collision_data)
{
    stream << upload_buffer(device, output_sim_data->sa_contact_active_verts, input_sim_data->sa_contact_active_verts)
           << upload_buffer(device, output_sim_data->sa_contact_active_faces, input_sim_data->sa_contact_active_faces)
           << upload_buffer(device, output_sim_data->sa_contact_active_edges, input_sim_data->sa_contact_active_edges);

    stream << upload_buffer(device, output_sim_data->sa_contact_active_verts_d_hat, input_sim_data->sa_contact_active_verts_d_hat)
           << upload_buffer(device, output_sim_data->sa_contact_active_verts_offset, input_sim_data->sa_contact_active_verts_offset)
           << luisa::compute::synchronize();
}


}  // namespace lcs::Initializer