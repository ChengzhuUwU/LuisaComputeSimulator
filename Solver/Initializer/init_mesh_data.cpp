#include "Initializer/init_mesh_data.h"
#include "Core/affine_position.h"
#include "Core/float_nxn.h"
#include "Energy/bending_energy.h"
#include "MeshOperation/mesh_reader.h"
#include "Initializer/initializer_utils.h"
#include "Utils/cpu_parallel.h"
#include <algorithm>
#include <numeric>

namespace lcs
{

namespace Initializer
{
    struct AABB
    {
        float3 packed_min;
        float3 packed_max;
        AABB   operator+(const AABB& input_aabb) const
        {
            AABB tmp;
            tmp.packed_min = lcs::min_vec(packed_min, input_aabb.packed_min);
            tmp.packed_max = lcs::max_vec(packed_max, input_aabb.packed_max);
            return tmp;
        }
        AABB()
            : packed_min(float3(Float_max))
            , packed_max(float3(-Float_max))
        {
        }
        AABB(const float3& pos)
            : packed_min(pos)
            , packed_max(pos)
        {
        }
    };
    void ShellInfo::set_pinned_verts_from_functions(const std::function<bool(uint)>& func,
                                                    const FixedPointAnimationInfo&   fixed_info)
    {
        if (input_mesh.model_positions.size() == 0)
        {
            load_mesh_data();
        }

        for (uint vid = 0; vid < input_mesh.model_positions.size(); vid++)
        {
            if (func(vid))
            {
                auto   read_pos = input_mesh.model_positions[vid];
                float3 pos      = luisa::make_float3(read_pos[0], read_pos[1], read_pos[2]);
                // LUISA_INFO("Found fixed point vert : local_vid {}, pos {}", vid, pos);
                auto affine_pos = FixedPointAnimationInfo::fn_affine_position(fixed_info, 0.0f, pos);
                fixed_point_indices.emplace_back(vid);
                fixed_point_target_positions.emplace_back(affine_pos);
                fixed_point_animations.push_back(fixed_info);
            }
        }
    }
    void ShellInfo::set_pinned_verts_from_norm_position(const std::function<bool(const float3&)>& func,
                                                        const FixedPointAnimationInfo& fixed_info)
    {
        if (input_mesh.model_positions.size() == 0)
        {
            load_mesh_data();
            // LUISA_INFO("ShellInfo::set_pinned_verts_from_norm_position() : auto load mesh data for shell {}", model_name);
        }
        AABB local_aabb = CpuParallel::parallel_for_and_reduce_sum<AABB>(
            0,
            input_mesh.model_positions.size(),
            [&](const uint vid)
            {
                auto   read_pos = input_mesh.model_positions[vid];
                float3 pos      = luisa::make_float3(read_pos[0], read_pos[1], read_pos[2]);
                return AABB(pos);
            });

        auto pos_min     = local_aabb.packed_min;
        auto pos_max     = local_aabb.packed_max;
        auto pos_dim_inv = 1.0f / luisa::max(pos_max - pos_min, 0.0001f);

        for (uint vid = 0; vid < input_mesh.model_positions.size(); vid++)
        {
            auto   read_pos = input_mesh.model_positions[vid];
            float3 pos      = luisa::make_float3(read_pos[0], read_pos[1], read_pos[2]);
            float3 norm_pos = (pos - pos_min) * pos_dim_inv;

            if (func(norm_pos))
            {
                // LUISA_INFO("Found fixed point vert : local_vid {}, pos {}", vid, pos);
                auto affine_pos = FixedPointAnimationInfo::fn_affine_position(fixed_info, 0.0f, pos);
                fixed_point_indices.emplace_back(vid);
                fixed_point_target_positions.emplace_back(affine_pos);
                fixed_point_animations.push_back(fixed_info);
            }
        }
    }
    void ShellInfo::set_pinned_verts_from_indices(const std::vector<uint>&       indices,
                                                  const FixedPointAnimationInfo& fixed_info)
    {
        if (input_mesh.model_positions.size() == 0)
        {
            load_mesh_data();
        }

        for (const uint vid : indices)
        {
            auto   read_pos   = input_mesh.model_positions[vid];
            float3 pos        = luisa::make_float3(read_pos[0], read_pos[1], read_pos[2]);
            auto   affine_pos = FixedPointAnimationInfo::fn_affine_position(fixed_info, 0.0f, pos);

            fixed_point_indices.emplace_back(vid);
            fixed_point_target_positions.emplace_back(affine_pos);
            fixed_point_animations.push_back(fixed_info);
        }
    }
    ShellInfo& ShellInfo::load_fixed_points()
    {
        if (input_mesh.model_positions.size() == 0)
        {
            load_mesh_data();
        }

        for (const auto& fixed_point_func : fixed_point_range_info)
        {
            const auto& range = fixed_point_func.range;
            if (fixed_point_func.method == FixedPointsType::All)
            {
                from_norm_position([](const float3& norm_pos) { return true; }, fixed_point_func.fixed_info);
            }
            else if (fixed_point_func.method == FixedPointsType::Left)
            {
                from_norm_position([range = fixed_point_func.range.front()](const float3& norm_pos)
                                   { return norm_pos.x < range; },
                                   fixed_point_func.fixed_info);
            }
            else if (fixed_point_func.method == FixedPointsType::Right)
            {
                from_norm_position([range = fixed_point_func.range.front()](const float3& norm_pos)
                                   { return norm_pos.x > 1.0f - range; },
                                   fixed_point_func.fixed_info);
            }
            else if (fixed_point_func.method == FixedPointsType::Front)
            {
                from_norm_position([range = fixed_point_func.range.front()](const float3& norm_pos)
                                   { return norm_pos.z < range; },
                                   fixed_point_func.fixed_info);
            }
            else if (fixed_point_func.method == FixedPointsType::Back)
            {
                from_norm_position([range = fixed_point_func.range.front()](const float3& norm_pos)
                                   { return norm_pos.z > 1.0f - range; },
                                   fixed_point_func.fixed_info);
            }
            else if (fixed_point_func.method == FixedPointsType::Up)
            {
                from_norm_position([range = fixed_point_func.range.front()](const float3& norm_pos)
                                   { return norm_pos.y > 1.0f - range; },
                                   fixed_point_func.fixed_info);
            }
            else if (fixed_point_func.method == FixedPointsType::Down)
            {
                from_norm_position([range = fixed_point_func.range.front()](const float3& norm_pos)
                                   { return norm_pos.y < range; },
                                   fixed_point_func.fixed_info);
            }
            else if (fixed_point_func.method == FixedPointsType::LeftUp)
            {
                from_norm_position([range = fixed_point_func.range.front()](const float3& norm_pos)
                                   { return norm_pos.x < range && norm_pos.y > 1.0f - range; },
                                   fixed_point_func.fixed_info);
            }
            else if (fixed_point_func.method == FixedPointsType::LeftDown)
            {
                from_norm_position([range = fixed_point_func.range.front()](const float3& norm_pos)
                                   { return norm_pos.x < range && norm_pos.y < range; },
                                   fixed_point_func.fixed_info);
            }
            else if (fixed_point_func.method == FixedPointsType::LeftFront)
            {
                from_norm_position([range = fixed_point_func.range.front()](const float3& norm_pos)
                                   { return norm_pos.x < range && norm_pos.z > 1.0f - range; },
                                   fixed_point_func.fixed_info);
            }
            else if (fixed_point_func.method == FixedPointsType::LeftBack)
            {
                from_norm_position([range = fixed_point_func.range.front()](const float3& norm_pos)
                                   { return norm_pos.x < range && norm_pos.z < range; },
                                   fixed_point_func.fixed_info);
            }
            else if (fixed_point_func.method == FixedPointsType::RightUp)
            {
                from_norm_position([range = fixed_point_func.range.front()](const float3& norm_pos)
                                   { return norm_pos.x > 1.0f - range && norm_pos.y > 1.0f - range; },
                                   fixed_point_func.fixed_info);
            }
            else if (fixed_point_func.method == FixedPointsType::RightDown)
            {
                from_norm_position([range = fixed_point_func.range.front()](const float3& norm_pos)
                                   { return norm_pos.x > 1.0f - range && norm_pos.y < range; },
                                   fixed_point_func.fixed_info);
            }
            else if (fixed_point_func.method == FixedPointsType::RightFront)
            {
                from_norm_position([range = fixed_point_func.range.front()](const float3& norm_pos)
                                   { return norm_pos.x > 1.0f - range && norm_pos.z > 1.0f - range; },
                                   fixed_point_func.fixed_info);
            }
            else if (fixed_point_func.method == FixedPointsType::RightBack)
            {
                from_norm_position([range = fixed_point_func.range.front()](const float3& norm_pos)
                                   { return norm_pos.x > 1.0f - range && norm_pos.z < range; },
                                   fixed_point_func.fixed_info);
            }
            else if (fixed_point_func.method == FixedPointsType::FrontUp)
            {
                from_norm_position([range = fixed_point_func.range.front()](const float3& norm_pos)
                                   { return norm_pos.z < range && norm_pos.y > 1.0f - range; },
                                   fixed_point_func.fixed_info);
            }
            else if (fixed_point_func.method == FixedPointsType::FrontDown)
            {
                from_norm_position([range = fixed_point_func.range.front()](const float3& norm_pos)
                                   { return norm_pos.z < range && norm_pos.y < range; },
                                   fixed_point_func.fixed_info);
            }
            else if (fixed_point_func.method == FixedPointsType::BackUp)
            {
                from_norm_position([range = fixed_point_func.range.front()](const float3& norm_pos)
                                   { return norm_pos.z > 1.0f - range && norm_pos.y > 1.0f - range; },
                                   fixed_point_func.fixed_info);
            }
            else if (fixed_point_func.method == FixedPointsType::BackDown)
            {
                from_norm_position([range = fixed_point_func.range.front()](const float3& norm_pos)
                                   { return norm_pos.z > 1.0f - range && norm_pos.y < range; },
                                   fixed_point_func.fixed_info);
            }
            else if (fixed_point_func.method == FixedPointsType::FromIndices)
            {
                auto indices = *((std::vector<uint>*)fixed_point_func.data_ptr);
                set_pinned_verts_from_indices(indices, fixed_point_func.fixed_info);
            }
            else if (fixed_point_func.method == FixedPointsType::FromFunction)
            {
                auto func = *((std::function<bool(uint)>*)fixed_point_func.data_ptr);
                set_pinned_verts_from_functions(func, fixed_point_func.fixed_info);
            }
            else
            {
                LUISA_ERROR("Unsupported FixedPointsType {} in ShellInfo::load_fixed_points().",
                            int(fixed_point_func.method));
            }
        }

        return *this;

        // fixed_point_list.insert(
        //     fixed_point_list.end(), curr_fixed_point_verts.begin(), curr_fixed_point_verts.end());
        // fixed_point_target_positions.insert(fixed_point_target_positions.end(),
        //                                     curr_fixed_point_target_positions.begin(),
        //                                     curr_fixed_point_target_positions.end());
        // fixed_point_info.insert(
        //     fixed_point_info.end(), curr_fixed_point_info.begin(), curr_fixed_point_info.end());
        // return curr_fixed_point_verts;
    }

    std::vector<float3> ShellInfo::get_fixed_point_target_positions(const float time)
    {
        CpuParallel::parallel_for(
            0,
            fixed_point_indices.size(),
            [&](const uint index)
            {
                const uint  local_vid        = fixed_point_indices[index];
                const auto& fixed_info       = fixed_point_animations[index];
                const auto  model_pos        = input_mesh.model_positions[local_vid];
                auto        transform_matrix = lcs::make_model_matrix(translation, rotation, scale);
                const auto  rest_pos =
                    (transform_matrix * luisa::make_float4(model_pos[0], model_pos[1], model_pos[2], 1.0f))
                        .xyz();

                auto target = FixedPointAnimationInfo::fn_affine_position(fixed_info, time, rest_pos);
                auto orig   = fixed_point_target_positions[index];
                fixed_point_target_positions[index] = target;
                // LUISA_INFO("For FixedVert {}: local vid = {} try to push delta {} : from {} to {}",
                //            index,
                //            local_vid,
                //            target - orig,
                //            rest_pos,
                //            target);
            });
        return fixed_point_target_positions;
    }
    void ShellInfo::update_pinned_verts(const std::vector<float3>& new_positions)
    {
        CpuParallel::parallel_copy(new_positions, fixed_point_target_positions);
    }
    // template <typename T>
    void ShellInfo::get_rest_positions(std::vector<std::array<float, 3>>& rest_positions)
    {
        rest_positions.resize(input_mesh.model_positions.size());
        auto transform_matrix = lcs::make_model_matrix(translation, rotation, scale);
        CpuParallel::parallel_for(0,
                                  input_mesh.model_positions.size(),
                                  [&](const uint vid)
                                  {
                                      const auto model_pos = input_mesh.model_positions[vid];
                                      auto       rest_pos =
                                          (transform_matrix
                                           * luisa::make_float4(model_pos[0], model_pos[1], model_pos[2], 1.0f))
                                              .xyz();
                                      std::array<float, 3> output;
                                      output[0]           = rest_pos.x;
                                      output[1]           = rest_pos.y;
                                      output[2]           = rest_pos.z;
                                      rest_positions[vid] = output;
                                  });
    }

    // template<template<typename> typename BasicBuffer>
    void init_mesh_data(std::vector<lcs::Initializer::ShellInfo>& shell_infos, lcs::MeshData<std::vector>* mesh_data)
    {
        std::sort(shell_infos.begin(),
                  shell_infos.end(),
                  [](const Initializer::ShellInfo& left, const Initializer::ShellInfo& right)
                  { return int(left.shell_type) < int(right.shell_type); });
        const uint num_meshes = shell_infos.size();
        // std::vector<SimMesh::TriangleMeshData> input_meshes(num_meshes);

        mesh_data->num_meshes = num_meshes;

        mesh_data->num_verts          = 0;
        mesh_data->num_faces          = 0;
        mesh_data->num_edges          = 0;
        mesh_data->num_dihedral_edges = 0;
        mesh_data->num_tets           = 0;

        mesh_data->prefix_num_verts.resize(1 + num_meshes, 0);
        mesh_data->prefix_num_faces.resize(1 + num_meshes, 0);
        mesh_data->prefix_num_edges.resize(1 + num_meshes, 0);
        mesh_data->prefix_num_dihedral_edges.resize(1 + num_meshes, 0);
        mesh_data->prefix_num_tets.resize(1 + num_meshes, 0);

        mesh_data->sa_rest_translate.resize(num_meshes);
        mesh_data->sa_rest_rotation.resize(num_meshes);
        mesh_data->sa_rest_scale.resize(num_meshes);

        mesh_data->fixed_verts_map.resize(num_meshes);

        // Pre-process materials
        for (uint meshIdx = 0; meshIdx < num_meshes; meshIdx++)
        {
            auto& shell_info = shell_infos[meshIdx];
            auto& input_mesh = shell_info.input_mesh;
            if (input_mesh.model_positions.empty())
            {
                shell_info.load_mesh_data();
            }

            if (shell_info.shell_type == ShellTypeCloth)
            {
                if (!shell_info.holds<ClothMaterial>())
                {
                    shell_info.physics_material = ClothMaterial();
                }
                auto& mat    = shell_info.get_material<ClothMaterial>();
                mat.is_shell = true;  // Cloth material must be shell
            }
            else if (shell_info.shell_type == ShellTypeTetrahedral)
            {
                if (!shell_info.holds<TetMaterial>())
                {
                    shell_info.physics_material = TetMaterial();
                }
                auto& mat    = shell_info.get_material<TetMaterial>();
                mat.is_shell = false;
            }
            else if (shell_info.shell_type == ShellTypeRigid)
            {
                if (!shell_info.holds<RigidMaterial>())
                {
                    shell_info.physics_material = RigidMaterial();
                }
                auto& mat = shell_info.get_material<RigidMaterial>();
                if (mat.is_shell)
                {
                }
                else
                {
                    mat.thickness = 0.0f;
                }
            }
            else if (shell_info.shell_type == ShellTypeRod)
            {
                if (!shell_info.holds<RodMaterial>())
                {
                    shell_info.physics_material = RodMaterial();
                }
                auto& mat    = shell_info.get_material<RodMaterial>();
                mat.is_shell = true;
            }
        }

        // Constant scalar and init MeshData
        for (uint meshIdx = 0; meshIdx < num_meshes; meshIdx++)
        {
            auto& shell_info = shell_infos[meshIdx];
            auto& input_mesh = shell_info.input_mesh;

            mesh_data->prefix_num_verts[meshIdx]          = mesh_data->num_verts;
            mesh_data->prefix_num_faces[meshIdx]          = mesh_data->num_faces;
            mesh_data->prefix_num_edges[meshIdx]          = mesh_data->num_edges;
            mesh_data->prefix_num_dihedral_edges[meshIdx] = mesh_data->num_dihedral_edges;
            mesh_data->prefix_num_tets[meshIdx]           = mesh_data->num_tets;

            const uint curr_num_verts          = input_mesh.model_positions.size();
            const uint curr_num_faces          = input_mesh.faces.size();
            const uint curr_num_edges          = input_mesh.edges.size();
            const uint curr_num_dihedral_edges = input_mesh.dihedral_edges.size();
            const uint curr_num_tets           = input_mesh.tetrahedrons.size();

            mesh_data->num_verts += curr_num_verts;
            mesh_data->num_faces += curr_num_faces;
            mesh_data->num_edges += curr_num_edges;
            mesh_data->num_dihedral_edges += curr_num_dihedral_edges;
            mesh_data->num_tets += curr_num_tets;
        }

        mesh_data->prefix_num_verts[num_meshes]          = mesh_data->num_verts;
        mesh_data->prefix_num_faces[num_meshes]          = mesh_data->num_faces;
        mesh_data->prefix_num_edges[num_meshes]          = mesh_data->num_edges;
        mesh_data->prefix_num_dihedral_edges[num_meshes] = mesh_data->num_dihedral_edges;
        mesh_data->prefix_num_tets[num_meshes]           = mesh_data->num_tets;

        uint num_verts          = mesh_data->num_verts;
        uint num_faces          = mesh_data->num_faces;
        uint num_edges          = mesh_data->num_edges;
        uint num_dihedral_edges = mesh_data->num_dihedral_edges;
        uint num_tets           = mesh_data->num_tets;

        LUISA_INFO("Mesh : (numVerts : {}) (numFaces : {})  (numEdges : {}) (numDihedralEdges : {}), (numTets : {})",
                   num_verts,
                   num_faces,
                   num_edges,
                   num_dihedral_edges,
                   num_tets);

        // Read information
        {
            mesh_data->sa_rest_x.resize(num_verts);
            mesh_data->sa_model_x.resize(num_verts);
            mesh_data->sa_scaled_model_x.resize(num_verts);
            mesh_data->sa_faces.resize(num_faces);
            mesh_data->sa_edges.resize(num_edges);
            mesh_data->sa_dihedral_edges.resize(num_dihedral_edges);
            mesh_data->sa_tetrahedrons.resize(num_tets);

            mesh_data->sa_rest_v.resize(num_verts);
            mesh_data->sa_is_fixed.resize(num_verts);
            mesh_data->sa_vert_mesh_type.resize(num_verts);

            mesh_data->sa_vert_mesh_id.resize(num_verts);
            mesh_data->sa_face_mesh_id.resize(num_faces);
            mesh_data->sa_edge_mesh_id.resize(num_edges);
            mesh_data->sa_dihedral_edge_mesh_id.resize(num_dihedral_edges);
            mesh_data->sa_tet_mesh_id.resize(num_tets);

            uint prefix_num_verts          = 0;
            uint prefix_num_faces          = 0;
            uint prefix_num_edges          = 0;
            uint prefix_num_dihedral_edges = 0;
            uint prefix_num_tets           = 0;

            for (uint meshIdx = 0; meshIdx < num_meshes; meshIdx++)
            {
                auto&       curr_shell_info = shell_infos[meshIdx];
                const auto& curr_input_mesh = curr_shell_info.input_mesh;

                // Model info
                {
                    mesh_data->sa_rest_translate[meshIdx] = curr_shell_info.translation;
                    mesh_data->sa_rest_rotation[meshIdx]  = curr_shell_info.rotation;
                    mesh_data->sa_rest_scale[meshIdx]     = curr_shell_info.scale;
                }

                const uint curr_num_verts          = curr_input_mesh.model_positions.size();
                const uint curr_num_faces          = curr_input_mesh.faces.size();
                const uint curr_num_edges          = curr_input_mesh.edges.size();
                const uint curr_num_dihedral_edges = curr_input_mesh.dihedral_edges.size();
                const uint curr_num_tets           = curr_input_mesh.tetrahedrons.size();

                // Read position with affine
                CpuParallel::parallel_for(
                    0,
                    curr_num_verts,
                    [&](const uint vid)
                    {
                        std::array<float, 3> read_pos = curr_input_mesh.model_positions[vid];
                        float3 model_position = luisa::make_float3(read_pos[0], read_pos[1], read_pos[2]);
                        float4x4 model_matrix = lcs::make_model_matrix(
                            curr_shell_info.translation, curr_shell_info.rotation, curr_shell_info.scale);
                        float3 world_position = lcs::affine_position(model_matrix, model_position);
                        mesh_data->sa_model_x[prefix_num_verts + vid] = model_position;
                        mesh_data->sa_scaled_model_x[prefix_num_verts + vid] = curr_shell_info.scale * model_position;
                        mesh_data->sa_rest_x[prefix_num_verts + vid]       = world_position;
                        mesh_data->sa_rest_v[prefix_num_verts + vid]       = luisa::make_float3(0.0f);
                        mesh_data->sa_vert_mesh_id[prefix_num_verts + vid] = meshIdx;
                        mesh_data->sa_vert_mesh_type[prefix_num_verts + vid] = uint(curr_shell_info.shell_type);
                    });
                // Read triangle face
                CpuParallel::parallel_for(0,
                                          curr_num_faces,
                                          [&](const uint fid)
                                          {
                                              auto face = curr_input_mesh.faces[fid];
                                              mesh_data->sa_faces[prefix_num_faces + fid] =
                                                  prefix_num_verts + luisa::make_uint3(face[0], face[1], face[2]);
                                              mesh_data->sa_face_mesh_id[prefix_num_faces + fid] = meshIdx;
                                          });
                // Read edge
                CpuParallel::parallel_for(0,
                                          curr_num_edges,
                                          [&](const uint eid)
                                          {
                                              auto edge = curr_input_mesh.edges[eid];
                                              mesh_data->sa_edges[prefix_num_edges + eid] =
                                                  prefix_num_verts + luisa::make_uint2(edge[0], edge[1]);
                                              mesh_data->sa_edge_mesh_id[prefix_num_edges + eid] = meshIdx;
                                          });
                // Read bending edge
                CpuParallel::parallel_for(
                    0,
                    curr_num_dihedral_edges,
                    [&](const uint eid)
                    {
                        auto bending_edge = curr_input_mesh.dihedral_edges[eid];
                        mesh_data->sa_dihedral_edges[prefix_num_dihedral_edges + eid] =
                            prefix_num_verts
                            + luisa::make_uint4(bending_edge[0], bending_edge[1], bending_edge[2], bending_edge[3]);
                        mesh_data->sa_dihedral_edge_mesh_id[prefix_num_dihedral_edges + eid] = meshIdx;
                    });

                // Read tetrahedrons
                CpuParallel::parallel_for(0,
                                          curr_num_tets,
                                          [&](const uint tid)
                                          {
                                              auto tet = curr_input_mesh.tetrahedrons[tid];
                                              mesh_data->sa_tetrahedrons[prefix_num_tets + tid] =
                                                  prefix_num_verts
                                                  + luisa::make_uint4(tet[0], tet[1], tet[2], tet[3]);
                                              mesh_data->sa_tet_mesh_id[prefix_num_tets + tid] = meshIdx;
                                          });

                // Read fixed points
                mesh_data->fixed_verts_map[meshIdx].resize(curr_shell_info.fixed_point_indices.size());
                CpuParallel::parallel_for(0,
                                          curr_shell_info.fixed_point_indices.size(),
                                          [&](const uint index)
                                          {
                                              const uint local_vid = curr_shell_info.fixed_point_indices[index];
                                              const uint global_vid = prefix_num_verts + local_vid;
                                              mesh_data->sa_is_fixed[global_vid] = true;
                                              mesh_data->fixed_verts.push_back(global_vid);
                                              mesh_data->fixed_verts_map[meshIdx][index] = global_vid;
                                          });
                // Set fixed-points
                {
                    AABB local_aabb = CpuParallel::parallel_for_and_reduce_sum<AABB>(
                        0,
                        curr_num_verts,
                        [&](const uint vid)
                        {
                            auto   read_pos = mesh_data->sa_rest_x[prefix_num_verts + vid];
                            float3 pos      = luisa::make_float3(read_pos[0], read_pos[1], read_pos[2]);
                            return AABB(pos);
                        });
                    auto pos_min = local_aabb.packed_min;
                    auto pos_max = local_aabb.packed_max;

                    float avg_spring_length =
                        CpuParallel::parallel_for_and_reduce_sum<float>(
                            0,
                            curr_num_edges,
                            [&](const uint eid)
                            {
                                auto edge = mesh_data->sa_edges[prefix_num_edges + eid];
                                return length_vec(mesh_data->sa_rest_x[edge[0]] - mesh_data->sa_rest_x[edge[1]]);
                            })
                        / float(curr_num_edges);

                    LUISA_INFO("Mesh {:<2} : numVerts = {:<5}, numFaces = {:<5}, numEdges = {:<5}, numTets = {:5} avgEdgeLength = {:2.4f}, AABB range = {}",
                               meshIdx,
                               curr_num_verts,
                               curr_num_faces,
                               curr_num_edges,
                               curr_num_tets,
                               avg_spring_length,
                               pos_max - pos_min);
                }

                prefix_num_verts += curr_num_verts;
                prefix_num_faces += curr_num_faces;
                prefix_num_edges += curr_num_edges;
                prefix_num_dihedral_edges += curr_num_dihedral_edges;
                prefix_num_tets += curr_num_tets;
            }
        }

        // Init topology adjacent list
        {
            mesh_data->vert_adj_faces.resize(num_verts);
            mesh_data->vert_adj_edges.resize(num_verts);
            mesh_data->vert_adj_dihedral_edges.resize(num_verts);
            mesh_data->vert_adj_verts.resize(num_verts);
            mesh_data->vert_adj_tets.resize(num_verts);

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
            for (uint eid = 0; eid < num_dihedral_edges; eid++)
            {
                auto edge = mesh_data->sa_dihedral_edges[eid];
                for (uint j = 0; j < 4; j++)
                    mesh_data->vert_adj_dihedral_edges[edge[j]].push_back(eid);
            }
            upload_2d_csr_from(mesh_data->sa_vert_adj_dihedral_edges_csr, mesh_data->vert_adj_dihedral_edges);

            // Vert adj tets
            for (uint tid = 0; tid < num_tets; tid++)
            {
                auto tet = mesh_data->sa_tetrahedrons[tid];
                for (uint j = 0; j < 4; j++)
                    mesh_data->vert_adj_tets[tet[j]].push_back(tid);
            }
            upload_2d_csr_from(mesh_data->sa_vert_adj_tets_csr, mesh_data->vert_adj_tets);

            // Vert adj verts based on 1-order connection
            for (uint eid = 0; eid < num_edges; eid++)
            {
                auto edge = mesh_data->sa_edges[eid];
                for (uint j = 0; j < 2; j++)
                {
                    const uint left  = edge[j];
                    const uint right = edge[1 - j];
                    mesh_data->vert_adj_verts[left].push_back(right);
                }
            }
            upload_2d_csr_from(mesh_data->sa_vert_adj_verts_csr, mesh_data->vert_adj_verts);

            // Vert adj verts based on 1-order bending-connection
            auto insert_adj_vert = [](std::vector<std::vector<uint>>& adj_map, const uint& vid1, const uint& vid2)
            {
                if (vid1 == vid2)
                    std::cerr << "redudant!";
                auto& inner_list  = adj_map[vid1];
                auto  find_result = std::find(inner_list.begin(), inner_list.end(), vid2);
                if (find_result == inner_list.end())
                {
                    inner_list.push_back(vid2);
                }
            };


            // face_adj_edges
            // Face adj edges
            mesh_data->face_adj_edges.resize(num_faces);
            mesh_data->face_adj_faces.resize(num_faces);
            mesh_data->edge_adj_faces.resize(num_edges, luisa::make_uint2(-1u));
            auto fn_vert_in_face = [](const uint& vid, const uint3& face)
            { return vid == face[0] || vid == face[1] || vid == face[2]; };
            CpuParallel::parallel_for(
                0,
                num_faces,
                [&](const uint fid)
                {
                    std::unordered_set<uint> adj_edges_set;
                    adj_edges_set.reserve(3);
                    const auto face             = mesh_data->sa_faces[fid];
                    uint       face_sum_indices = face[0] + face[1] + face[2];
                    for (uint j = 0; j < 3; j++)
                    {
                        const uint  vid            = face[j];
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
                    if (adj_edges_set.size() != 3)
                        LUISA_ERROR("Face {} adj edge count {} != 3", fid, adj_edges_set.size());
                    uint3 face_adj_edges;
                    uint  idx = 0;
                    for (const auto& adj_eid : adj_edges_set)
                    {
                        face_adj_edges[idx++] = adj_eid;
                    }
                    mesh_data->face_adj_edges[fid] = face_adj_edges;
                });
            std::vector<uint> edge_adj_face_count(num_edges, 0);
            CpuParallel::single_thread_for(0,
                                           num_faces,
                                           [&](const uint fid)
                                           {
                                               uint3 face_adj_edges = mesh_data->face_adj_edges[fid];
                                               for (uint j = 0; j < 3; j++)
                                               {
                                                   uint  adj_eid = face_adj_edges[j];
                                                   uint& offset  = edge_adj_face_count[adj_eid];
                                                   mesh_data->edge_adj_faces[adj_eid][offset++] = fid;
                                               }
                                           });
            CpuParallel::parallel_for(0,
                                      num_faces,
                                      [&](const uint fid)
                                      {
                                          const uint3 face_adj_edges = mesh_data->face_adj_edges[fid];
                                          uint3       face_adj_faces;
                                          for (uint j = 0; j < 3; j++)
                                          {
                                              uint  adj_eid        = face_adj_edges[j];
                                              uint2 edge_adj_faces = mesh_data->edge_adj_faces[adj_eid];
                                              if (edge_adj_faces[0] != -1u && edge_adj_faces[0] != fid)
                                                  face_adj_faces[j] = edge_adj_faces[0];
                                              if (edge_adj_faces[1] != -1u && edge_adj_faces[1] != fid)
                                                  face_adj_faces[j] = edge_adj_faces[1];
                                          }
                                          mesh_data->face_adj_faces[fid] = face_adj_faces;
                                      });
        }

        // Compute rest area
        {
            mesh_data->sa_rest_vert_area.resize(num_verts);
            mesh_data->sa_rest_edge_area.resize(num_edges);
            mesh_data->sa_rest_face_area.resize(num_faces);
            mesh_data->sa_rest_tet_volume.resize(num_tets);
            mesh_data->sa_rest_vert_volume.resize(num_verts);
            mesh_data->sa_vert_thickness.resize(num_verts);
            mesh_data->sa_edge_thickness.resize(num_edges);
            mesh_data->sa_face_thickness.resize(num_faces);

            CpuParallel::parallel_for(0,
                                      num_faces,
                                      [&](const uint fid)
                                      {
                                          const uint3 face = mesh_data->sa_faces[fid];
                                          float area = compute_face_area(mesh_data->sa_rest_x[face[0]],
                                                                         mesh_data->sa_rest_x[face[1]],
                                                                         mesh_data->sa_rest_x[face[2]]);
                                          mesh_data->sa_rest_face_area[fid] = area;

                                          const uint mesh_idx = mesh_data->sa_face_mesh_id[fid];
                                          mesh_data->sa_face_thickness[fid] =
                                              shell_infos[mesh_idx].get_thickness();
                                      });
            CpuParallel::parallel_for(0,
                                      num_tets,
                                      [&](const uint tid)
                                      {
                                          const uint4 tet = mesh_data->sa_tetrahedrons[tid];
                                          float volume = compute_tet_volume(mesh_data->sa_rest_x[tet[0]],
                                                                            mesh_data->sa_rest_x[tet[1]],
                                                                            mesh_data->sa_rest_x[tet[2]],
                                                                            mesh_data->sa_rest_x[tet[3]]);
                                          mesh_data->sa_rest_tet_volume[tid] = volume;
                                      });

            CpuParallel::parallel_for(0,
                                      num_verts,
                                      [&](const uint vid)
                                      {
                                          const auto& adj_faces = mesh_data->vert_adj_faces[vid];
                                          double      area      = 0.0;
                                          for (const uint& adj_fid : adj_faces)
                                              area += mesh_data->sa_rest_face_area[adj_fid] / 3.0;
                                          mesh_data->sa_rest_vert_area[vid] = area;

                                          const uint  mesh_idx   = mesh_data->sa_vert_mesh_id[vid];
                                          const auto& shell_info = shell_infos[mesh_idx];
                                          mesh_data->sa_vert_thickness[vid] = shell_info.get_thickness();

                                          const auto& adj_tets = mesh_data->vert_adj_tets[vid];
                                          if (shell_info.get_is_shell() || adj_tets.empty())
                                          {
                                              mesh_data->sa_rest_vert_volume[vid] =
                                                  area * shell_info.get_thickness();
                                          }
                                          else
                                          {
                                              double volume = 0.0;
                                              for (const uint& adj_tid : adj_tets)
                                                  volume += mesh_data->sa_rest_tet_volume[adj_tid] / 4.0;
                                              mesh_data->sa_rest_vert_volume[vid] = volume;
                                          }
                                      });
            CpuParallel::parallel_for(0,
                                      num_edges,
                                      [&](const uint eid)
                                      {
                                          uint2  adj_faces = mesh_data->edge_adj_faces[eid];
                                          double area      = 0.0;
                                          for (uint j = 0; j < 2; j++)
                                          {
                                              uint adj_fid = adj_faces[j];
                                              if (adj_fid != -1u)
                                              {
                                                  area += mesh_data->sa_rest_face_area[adj_fid] / 3.0;
                                              }
                                          }
                                          mesh_data->sa_rest_edge_area[eid] = area;

                                          const uint mesh_idx = mesh_data->sa_edge_mesh_id[eid];
                                          mesh_data->sa_edge_thickness[eid] =
                                              shell_infos[mesh_idx].get_thickness();
                                      });

            // float sum_face_area = CpuParallel::parallel_reduce_sum(mesh_data->sa_rest_face_area);
            // float sum_edge_area = CpuParallel::parallel_reduce_sum(mesh_data->sa_rest_edge_area);
            // float sum_vert_area = CpuParallel::parallel_reduce_sum(mesh_data->sa_rest_vert_area);
            // // LUISA_INFO("Summary areas : face = {}, edge = {}, vert = {}", sum_face_area, sum_edge_area, sum_vert_area);
            // LUISA_INFO("Average areas : face = {}, edge = {}, vert = {}",
            //            sum_face_area / double(num_faces),
            //            sum_edge_area / double(num_edges),
            //            sum_vert_area / double(num_verts));
        }

        // Init mass info
        {
            mesh_data->sa_body_mass.resize(num_meshes);
            mesh_data->sa_rest_body_volume.resize(num_meshes, 0.0f);
            mesh_data->sa_rest_body_area.resize(num_meshes, 0.0f);
            for (uint meshIdx = 0; meshIdx < num_meshes; meshIdx++)
            {
                const auto& shell_info = shell_infos[meshIdx];

                float sum_volume = 0.0f;
                if (shell_info.get_is_shell())  // Shell volume = area * thickness
                {
                    sum_volume = CpuParallel::parallel_for_and_reduce_sum<float>(
                        0,
                        mesh_data->prefix_num_faces[meshIdx + 1] - mesh_data->prefix_num_faces[meshIdx],
                        [&](const uint fid)
                        {
                            float face_area =
                                mesh_data->sa_rest_face_area[mesh_data->prefix_num_faces[meshIdx] + fid];
                            return face_area * shell_info.get_thickness();
                        });
                }
                else  // For solid body, compute volume from tets or faces integration
                {
                    if (shell_info.input_mesh.tetrahedrons.empty())
                    {
                        if (shell_info.shell_type == ShellTypeTetrahedral)
                        {
                            LUISA_ERROR("Mesh {} is set as Tetrahedral type but has no tetrahedron elements!", meshIdx);
                        }
                        else  // Use face integration
                        {
                            sum_volume = CpuParallel::parallel_for_and_reduce_sum<float>(
                                0,
                                mesh_data->prefix_num_faces[meshIdx + 1] - mesh_data->prefix_num_faces[meshIdx],
                                [&](const uint fid)
                                {
                                    uint3 face = mesh_data->sa_faces[mesh_data->prefix_num_faces[meshIdx] + fid];
                                    float3 v0 = mesh_data->sa_rest_x[face[0]];
                                    float3 v1 = mesh_data->sa_rest_x[face[1]];
                                    float3 v2 = mesh_data->sa_rest_x[face[2]];

                                    float3 e1 = v1 - v0;
                                    float3 e2 = v2 - v0;
                                    float3 N  = cross(e1, e2);
                                    return luisa::dot(v0, N) / 6.0f;
                                });
                        }
                    }
                    else
                    {
                        sum_volume = CpuParallel::parallel_for_and_reduce_sum<float>(
                            0,
                            mesh_data->prefix_num_tets[meshIdx + 1] - mesh_data->prefix_num_tets[meshIdx],
                            [&](const uint tid) {
                                return mesh_data->sa_rest_tet_volume[mesh_data->prefix_num_tets[meshIdx] + tid];
                            });
                    }
                }
                mesh_data->sa_rest_body_volume[meshIdx] = sum_volume;

                float sum_surface_area = CpuParallel::parallel_for_and_reduce_sum<float>(
                    0,
                    mesh_data->prefix_num_faces[meshIdx + 1] - mesh_data->prefix_num_faces[meshIdx],
                    [&](const uint fid)
                    { return mesh_data->sa_rest_face_area[mesh_data->prefix_num_faces[meshIdx] + fid]; });
                mesh_data->sa_rest_body_area[meshIdx] = sum_surface_area;


                const float input_mass    = shell_info.get_mass();
                const float input_density = shell_info.get_density();
                mesh_data->sa_body_mass[meshIdx] = input_mass != 0.0f ? input_mass : sum_volume * input_density;

                LUISA_INFO("Mesh {}'s volume = {}{}, body mass = {}, avg vert mass = {}",
                           meshIdx,
                           sum_volume,
                           shell_info.get_is_shell() ? luisa::format(", surface area = {}", sum_surface_area) : "",
                           mesh_data->sa_body_mass[meshIdx],
                           mesh_data->sa_body_mass[meshIdx]
                               / float(mesh_data->prefix_num_verts[meshIdx + 1] - mesh_data->prefix_num_verts[meshIdx]));
            }
            // {
            //     uint  prefix_num_faces = mesh_data->prefix_num_faces[meshIdx];
            //     uint  curr_num_faces   = mesh_data->prefix_num_faces[meshIdx + 1] - prefix_num_faces;
            //     float mesh_area        = CpuParallel::parallel_for_and_reduce_sum<float>(
            //         0,
            //         curr_num_faces,
            //         [&](const uint fid) { return mesh_data->sa_rest_face_area[prefix_num_faces + fid]; });
            //     body_areas[meshIdx]              = mesh_area;
            //     mesh_data->sa_body_mass[meshIdx] = shell_infos[meshIdx].mass != 0.0f ?
            //                                            shell_infos[meshIdx].mass :
            //                                            mesh_area * shell_infos[meshIdx].density;
            //     LUISA_INFO("Mesh {}'s area = {}, total mass = {}", meshIdx, mesh_area, mesh_data->sa_body_mass[meshIdx]);
            // }

            // Set vert mass
            mesh_data->sa_vert_mass.resize(num_verts);
            mesh_data->sa_vert_mass_inv.resize(num_verts);

            CpuParallel::parallel_for(0,
                                      num_verts,
                                      [&](const uint vid)
                                      {
                                          bool        is_fixed    = mesh_data->sa_is_fixed[vid] != 0;
                                          const uint  mesh_id     = mesh_data->sa_vert_mesh_id[vid];
                                          const float vert_volume = mesh_data->sa_rest_vert_volume[vid];
                                          const float body_volume = mesh_data->sa_rest_body_volume[mesh_id];
                                          const float weight = vert_volume / body_volume;
                                          //   const float vert_area = mesh_data->sa_rest_vert_area[vid];
                                          //   const float mesh_area = body_areas[mesh_id];
                                          //   const float weight    = vert_area / mesh_area;
                                          const float mass = weight * mesh_data->sa_body_mass[mesh_id];
                                          mesh_data->sa_vert_mass[vid] = mass;
                                          mesh_data->sa_vert_mass_inv[vid] = is_fixed ? 0.0f : 1.0f / (mass);
                                      });
        }

        // Init vert status
        {
            mesh_data->sa_x_frame_outer.resize(num_verts);
            mesh_data->sa_v_frame_outer.resize(num_verts);

            CpuParallel::parallel_for(0,
                                      num_verts,
                                      [&](const uint vid)
                                      {
                                          const float3 rest_x = mesh_data->sa_rest_x[vid];
                                          const float3 rest_v = mesh_data->sa_rest_v[vid];

                                          mesh_data->sa_x_frame_outer[vid] = rest_x;
                                          mesh_data->sa_v_frame_outer[vid] = rest_v;
                                      });
        }
    }


    void upload_mesh_buffers(luisa::compute::Device&                device,
                             luisa::compute::Stream&                stream,
                             lcs::MeshData<std::vector>*            input_data,
                             lcs::MeshData<luisa::compute::Buffer>* output_data)
    {
        output_data->num_meshes         = input_data->num_meshes;
        output_data->num_verts          = input_data->num_verts;
        output_data->num_faces          = input_data->num_faces;
        output_data->num_edges          = input_data->num_edges;
        output_data->num_dihedral_edges = input_data->num_dihedral_edges;
        output_data->num_tets           = input_data->num_tets;

        stream << upload_buffer(device, output_data->sa_rest_translate, input_data->sa_rest_translate)
               << upload_buffer(device, output_data->sa_rest_rotation, input_data->sa_rest_rotation)
               << upload_buffer(device, output_data->sa_rest_scale, input_data->sa_rest_scale)
               << upload_buffer(device, output_data->sa_model_x, input_data->sa_model_x)
               << upload_buffer(device, output_data->sa_scaled_model_x, input_data->sa_scaled_model_x)

               << upload_buffer(device, output_data->sa_rest_x, input_data->sa_rest_x)
               << upload_buffer(device, output_data->sa_rest_v, input_data->sa_rest_v)
               << upload_buffer(device, output_data->sa_faces, input_data->sa_faces)
               << upload_buffer(device, output_data->sa_edges, input_data->sa_edges);

        if (input_data->num_dihedral_edges > 0)
            stream << upload_buffer(device, output_data->sa_dihedral_edges, input_data->sa_dihedral_edges)
                   << upload_buffer(device, output_data->sa_dihedral_edge_mesh_id, input_data->sa_dihedral_edge_mesh_id);

        if (!input_data->sa_tetrahedrons.empty())
            stream << upload_buffer(device, output_data->sa_tetrahedrons, input_data->sa_tetrahedrons)
                   << upload_buffer(device, output_data->sa_tet_mesh_id, input_data->sa_tet_mesh_id)
                   << upload_buffer(device, output_data->sa_rest_tet_volume, input_data->sa_rest_tet_volume);

        // TODO: We may not have face
        stream
            << upload_buffer(device, output_data->sa_body_mass, input_data->sa_body_mass)
            << upload_buffer(device, output_data->sa_vert_mass, input_data->sa_vert_mass)
            << upload_buffer(device, output_data->sa_vert_mass_inv, input_data->sa_vert_mass_inv)
            << upload_buffer(device, output_data->sa_is_fixed, input_data->sa_is_fixed)
            << upload_buffer(device, output_data->sa_vert_mesh_id, input_data->sa_vert_mesh_id)
            << upload_buffer(device, output_data->sa_edge_mesh_id, input_data->sa_edge_mesh_id)
            << upload_buffer(device, output_data->sa_face_mesh_id, input_data->sa_face_mesh_id)
            << upload_buffer(device, output_data->sa_vert_mesh_type, input_data->sa_vert_mesh_type)

            << upload_buffer(device, output_data->sa_rest_body_area, input_data->sa_rest_body_area)
            << upload_buffer(device, output_data->sa_rest_body_volume, input_data->sa_rest_body_volume)

            << upload_buffer(device, output_data->sa_rest_vert_area, input_data->sa_rest_vert_area)
            << upload_buffer(device, output_data->sa_rest_edge_area, input_data->sa_rest_edge_area)
            << upload_buffer(device, output_data->sa_rest_face_area, input_data->sa_rest_face_area)

            << upload_buffer(device, output_data->sa_rest_vert_volume, input_data->sa_rest_vert_volume)
            << upload_buffer(device, output_data->sa_vert_thickness, input_data->sa_vert_thickness)
            << upload_buffer(device, output_data->sa_edge_thickness, input_data->sa_edge_thickness)
            << upload_buffer(device, output_data->sa_face_thickness, input_data->sa_face_thickness)

            // No std::vector<std::vector<uint>> vert_adj_verts info
            << upload_buffer(device, output_data->sa_vert_adj_verts_csr, input_data->sa_vert_adj_verts_csr)
            << upload_buffer(device, output_data->sa_vert_adj_faces_csr, input_data->sa_vert_adj_faces_csr)
            << upload_buffer(device, output_data->sa_vert_adj_edges_csr, input_data->sa_vert_adj_edges_csr)
            << upload_buffer(device, output_data->sa_vert_adj_dihedral_edges_csr, input_data->sa_vert_adj_dihedral_edges_csr)
            << upload_buffer(device, output_data->sa_vert_adj_tets_csr, input_data->sa_vert_adj_tets_csr)
            << upload_buffer(device, output_data->edge_adj_faces, input_data->edge_adj_faces)
            << upload_buffer(device, output_data->face_adj_edges, input_data->face_adj_edges)
            << upload_buffer(device, output_data->face_adj_faces, input_data->face_adj_faces)
            << luisa::compute::synchronize();
    }

}  // namespace Initializer


}  // namespace lcs