#include <iostream>
#include <luisa/luisa-compute.h>

#include "CollisionDetector/lbvh.h"
#include "CollisionDetector/narrow_phase.h"
#include "Core/constant_value.h"
#include "Initializer/init_collision_data.h"
#include "MeshOperation/default_mesh.h"
#include "MeshOperation/mesh_reader.h"
#include "SimulationSolver/newton_solver.h"
#include "Utils/cpu_parallel.h"
#include "Utils/device_parallel.h"
#include "Utils/buffer_filler.h"

#include "SimulationCore/scene_params.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationSolver/solver_interface.h"
#include "SimulationSolver/descent_solver.h"

#include "Initializer/init_mesh_data.h"
#include "Initializer/init_sim_data.h"
#include "app_simulation_demo_config.h"
#include "luisa/core/basic_types.h"

#if defined(SIMULATION_APP_USE_GUI)
#include "polyscope/volume_grid.h"
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <Eigen/Dense>
#endif

template <typename T>
using Buffer = luisa::compute::Buffer<T>;

namespace lcs::Initializer
{


void init_simulation_params()
{
    // if (lcs::get_scene_params().use_small_timestep) { lcs::get_scene_params().implicit_dt = 0.001f; }

    // lcs::get_scene_params().num_iteration = lcs::get_scene_params().num_substep * lcs::get_scene_params().nonlinear_iter_count;
    // lcs::get_scene_params().collision_detection_frequece = 1;

    // lcs::get_scene_params().stiffness_stretch_spring = FEM::calcSecondLame(lcs::get_scene_params().youngs_modulus_cloth, lcs::get_scene_params().poisson_ratio_cloth); // mu;
    // lcs::get_scene_params().stiffness_pressure = 1e6;

    {
        // lcs::get_scene_params().stiffness_stretch_spring = 1e4;
        // lcs::get_scene_params().xpbd_stiffness_collision = 1e7;
        // lcs::get_scene_params().stiffness_quadratic_bending = 5e-3;
        // lcs::get_scene_params().stiffness_DAB_bending = 5e-3;
    }
}


}  // namespace lcs::Initializer

static uint energy_idx = 0;


enum SolverType
{
    SolverTypeGaussNewton,
    SolverTypeXPBD_CPU,
    SolverTypeVBD_CPU,
    SolverTypeVBD_async,
};

int main(int argc, char** argv)
{

#ifndef NDEBUG
    luisa::log_level_info();  // log_level_verbose
#else
    luisa::log_level_verbose();
#endif

    luisa::fiber::scheduler scheduler;  // Initialize the fiber scheduler, which is also need in HostParallel functions
    LUISA_INFO("Hello, LuisaComputeSimulation!");

    // Init GPU system
#if defined(__APPLE__)
    std::string backend = "metal";
#else
    std::string backend = "cuda";
#endif
    const std::string            binary_path(argv[0]);
    luisa::compute::Context      context{binary_path};
    luisa::vector<luisa::string> device_names = context.backend_device_names(backend);
    if (device_names.empty())
    {
        LUISA_WARNING("No haredware device found.");
        exit(1);
    }
    for (size_t i = 0; i < device_names.size(); ++i)
    {
        LUISA_INFO("Device {}: {}", i, device_names[i]);
    }
    if (argc >= 2)
    {
        backend = argv[1];
    }
    luisa::compute::Device device = context.create_device(backend,
                                                          nullptr,
#ifndef NDEBUG
                                                          false
#else
                                                          true
#endif
    );
    luisa::compute::Stream stream = device.create_stream(luisa::compute::StreamTag::COMPUTE);

    lcs::get_scene_params().solver_type = lcs::SolverTypeNewton;

    // Some params
    {
        lcs::get_scene_params().implicit_dt    = 1.0f / 60.f;
        lcs::get_scene_params().num_substep    = 1;
        lcs::get_scene_params().pcg_iter_count = 100;
        // lcs::get_scene_params().use_bending = false;
        // lcs::get_scene_params().use_quadratic_bending_model = true;
        // lcs::get_scene_params().use_xpbd_solver = false;
        // lcs::get_scene_params().use_vbd_solver = false;
        // lcs::get_scene_params().use_newton_solver = true;
        lcs::get_scene_params().use_gpu = false;  // true
    }

    // Read Mesh
    std::vector<lcs::Initializer::ShellInfo> shell_list;
    Demo::Simulation::load_default_scene(shell_list);

    // Init Solver
    lcs::NewtonSolver solver;
    solver.init_solver(device, stream, shell_list);

    // for (auto edge : host_mesh_data.sa_edges)
    // {
    //     LUISA_INFO("edge = {}", edge);
    // }
    // for (auto bendingedge : host_mesh_data.sa_bending_edges)
    // {
    //     LUISA_INFO("edge = {}", bendingedge);
    // }
    // for (auto face : host_mesh_data.sa_faces)
    // {
    //     LUISA_INFO("face = {}", face);
    // }
    // for (auto mass : host_mesh_data.sa_vert_mass)
    // {
    //     LUISA_INFO("mass = {}", mass);
    // }

    auto fn_physics_step = [&]()
    {
        auto fn_affine_position =
            [](const lcs::Initializer::FixedPointInfo& fixed_point, const float time, const lcs::float3& pos)
        {
            auto fn_scale =
                [](const lcs::Initializer::FixedPointInfo& fixed_point, const float time, const lcs::float3& pos)
            { return (luisa::scaling(fixed_point.scale * time) * luisa::make_float4(pos, 1.0f)).xyz(); };
            auto fn_rotate =
                [](const lcs::Initializer::FixedPointInfo& fixed_point, const float time, const lcs::float3& pos)
            {
                const float rotAngRad    = time * fixed_point.rotAngVelDeg / 180.0f * float(lcs::Pi);
                const auto  relative_vec = pos - fixed_point.rotCenter;
                auto        matrix       = luisa::rotation(fixed_point.rotAxis, rotAngRad);
                const auto  rotated_pos  = matrix * luisa::make_float4(relative_vec, 1.0f);
                return fixed_point.rotCenter + rotated_pos.xyz();
            };
            auto fn_translate = [](const lcs::Initializer::FixedPointInfo& fixed_point,
                                   const float                             time,
                                   const lcs::float3&                      pos) {
                return (luisa::translation(fixed_point.translate * time) * luisa::make_float4(pos, 1.0f)).xyz();
            };
            auto new_pos = pos;
            if (fixed_point.use_scale)
                new_pos = fn_scale(fixed_point, time, new_pos);
            if (fixed_point.use_rotate)
                new_pos = fn_rotate(fixed_point, time, new_pos);
            if (fixed_point.use_translate)
                new_pos = fn_translate(fixed_point, time, new_pos);
            return new_pos;
        };

        auto fn_fixed_point_animation = [&](const uint curr_frame)
        {
            const float h = lcs::get_scene_params().implicit_dt;

            auto& host_mesh_data = solver.get_host_mesh_data();
            CpuParallel::parallel_for(0,
                                      host_mesh_data.num_verts,
                                      [&](const uint vid)
                                      {
                                          if (host_mesh_data.sa_is_fixed[vid])
                                          {
                                              host_mesh_data.sa_x_frame_outer[vid] = host_mesh_data.sa_rest_x[vid];
                                              host_mesh_data.sa_x_frame_outer_next[vid] =
                                                  host_mesh_data.sa_rest_x[vid];
                                              host_mesh_data.sa_v_frame_outer[vid] = luisa::make_float3(0.0f);
                                          }
                                      });

            // Animation for fixed points
            for (uint clothIdx = 0; clothIdx < shell_list.size(); clothIdx++)
            {
                if (shell_list[clothIdx].fixed_point_list.empty())
                    continue;

                CpuParallel::parallel_for(
                    0,
                    shell_list[clothIdx].fixed_point_list.size(),
                    [&](const uint index)
                    {
                        const auto& vid        = shell_list[clothIdx].fixed_point_list[index];
                        const auto& fixed_info = shell_list[clothIdx].fixed_point_info[index];
                        if (fixed_info.use_rotate || fixed_info.use_scale || fixed_info.use_translate)
                        {
                            auto& orig_pos = host_mesh_data.sa_rest_x[vid];
                            {
                                // Rotate
                                const float rotAngRad  = curr_frame * h;
                                const float start_time = curr_frame == 0 ? 0 : (curr_frame - 1) * h;
                                const float end_time   = curr_frame * h;
                                auto        bg = fn_affine_position(fixed_info, start_time, orig_pos);
                                auto        ed = fn_affine_position(fixed_info, end_time, orig_pos);
                                host_mesh_data.sa_x_frame_outer[vid]      = bg;
                                host_mesh_data.sa_x_frame_outer_next[vid] = ed;
                                host_mesh_data.sa_v_frame_outer[vid]      = (ed - bg) / h;
                                // LUISA_INFO("Fix point desire from {} to {} (vel = {})", bg, ed, (ed - bg) / h);
                                // host_mesh_data.sa_x_frame_outer[vid] = ed;
                                // host_mesh_data.sa_x_frame_outer_next[vid] = ed;
                                // host_mesh_data.sa_v_frame_outer[vid] = luisa::make_float3(0.0f);
                            }
                        }
                    });
            }
        };
        fn_fixed_point_animation(lcs::get_scene_params().current_frame);

        if (lcs::get_scene_params().use_gpu)
            solver.physics_step_GPU(device, stream);
        else
            solver.physics_step_CPU(device, stream);

        auto& host_mesh_data = solver.get_host_mesh_data();
        CpuParallel::parallel_for(0,
                                  host_mesh_data.num_verts,
                                  [&](const uint vid)
                                  {
                                      if (host_mesh_data.sa_is_fixed[vid])
                                      {
                                          host_mesh_data.sa_x_frame_outer[vid] =
                                              host_mesh_data.sa_x_frame_outer_next[vid];
                                      }
                                  });

        lcs::get_scene_params().current_frame += 1;
    };


    uint           max_frame         = 0;
    uint           optimize_frames   = 20;
    constexpr bool draw_bounding_box = false;
    constexpr bool use_ui            = true;

    // Init rendering data
    std::vector<std::vector<std::array<float, 3>>> sa_rendering_vertices(shell_list.size() + 0 + 0);
    std::vector<std::vector<std::array<uint, 3>>>  sa_rendering_faces(shell_list.size() + 0 + 0);
    std::vector<std::array<float, 3>> sa_global_aabb_vertices(SimMesh::BoundingBox::get_num_vertices(),
                                                              std::array<float, 3>({0.0f, 0.0f, 0.0f}));
    std::vector<std::array<uint, 3>>  sa_global_aabb_faces = SimMesh::BoundingBox::get_box_faces();
    std::vector<std::vector<std::array<float, 3>>> face_color(shell_list.size());
    {
        const auto& host_mesh_data = solver.get_host_mesh_data();
        for (uint meshIdx = 0; meshIdx < shell_list.size(); meshIdx++)
        {
            sa_rendering_vertices[meshIdx].resize(host_mesh_data.prefix_num_verts[meshIdx + 1]
                                                  - host_mesh_data.prefix_num_verts[meshIdx]);
            sa_rendering_faces[meshIdx].resize(host_mesh_data.prefix_num_faces[meshIdx + 1]
                                               - host_mesh_data.prefix_num_faces[meshIdx]);
            const uint curr_prefix_num_verts = host_mesh_data.prefix_num_verts[meshIdx];
            const uint next_prefix_num_verts = host_mesh_data.prefix_num_verts[meshIdx + 1];
            const uint curr_prefix_num_faces = host_mesh_data.prefix_num_faces[meshIdx];
            const uint next_prefix_num_faces = host_mesh_data.prefix_num_faces[meshIdx + 1];
            CpuParallel::parallel_for(0,
                                      next_prefix_num_verts - curr_prefix_num_verts,
                                      [&](const uint vid)
                                      {
                                          auto pos = host_mesh_data.sa_rest_x[curr_prefix_num_verts + vid];
                                          // sa_rendering_vertices[i][vid] = glm::vec3(pos.x, pos.y, pos.z);
                                          sa_rendering_vertices[meshIdx][vid] = {pos.x, pos.y, pos.z};
                                      });
            CpuParallel::parallel_for(0,
                                      next_prefix_num_faces - curr_prefix_num_faces,
                                      [&](const uint fid)
                                      {
                                          auto face = host_mesh_data.sa_faces[curr_prefix_num_faces + fid];
                                          sa_rendering_faces[meshIdx][fid] = {face[0] - curr_prefix_num_verts,
                                                                              face[1] - curr_prefix_num_verts,
                                                                              face[2] - curr_prefix_num_verts};
                                      });
            face_color[meshIdx].resize(host_mesh_data.prefix_num_faces[meshIdx + 1]
                                           - host_mesh_data.prefix_num_faces[meshIdx],
                                       {0.7, 0.2, 0.3});
        }
    }
    auto fn_update_rendering_vertices = [&]()
    {
        const auto& host_mesh_data = solver.get_host_mesh_data();
        for (uint clothIdx = 0; clothIdx < shell_list.size(); clothIdx++)
        {
            CpuParallel::parallel_for(
                0,
                host_mesh_data.prefix_num_verts[clothIdx + 1] - host_mesh_data.prefix_num_verts[clothIdx],
                [&](const uint vid)
                {
                    auto pos = host_mesh_data.sa_x_frame_outer[vid + host_mesh_data.prefix_num_verts[clothIdx]];
                    sa_rendering_vertices[clothIdx][vid] = {pos.x, pos.y, pos.z};
                });
        }
    };
    auto fn_save_frame_to_obj = [&](const std::string& additional_info = "")
    {
        SimMesh::saveToOBJ_combined(sa_rendering_vertices,
                                    sa_rendering_faces,
                                    std::format("0{}", lcs::get_scene_params().scene_id),
                                    additional_info,
                                    lcs::get_scene_params().current_frame);
    };

#if !defined(SIMULATION_APP_USE_GUI)
    {
        auto fn_single_step_without_ui = [&]()
        {
            LUISA_INFO("     Newton solver frame {}", lcs::get_scene_params().current_frame);

            fn_physics_step();
        };

        // solver.lcs::SolverInterface::restart_system();

        fn_save_frame_to_obj("_init");
        for (uint frame = 0; frame < 20; frame++)
        {
            fn_single_step_without_ui();

            fn_update_rendering_vertices();
            fn_save_frame_to_obj();
        }
        // SimMesh::saveToOBJ_combined(
        //     sa_rendering_vertices, sa_rendering_faces, "", "", lcs::get_scene_params().current_frame);
        // solver.lcs::SolverInterface::save_mesh_to_obj(lcs::get_scene_params().current_frame, "");
    }
#else
    {
        // Init Polyscope
        polyscope::init("openGL3_glfw");
        std::vector<polyscope::SurfaceMesh*> surface_meshes;
        std::vector<polyscope::SurfaceMesh*> bounding_boxes;


        for (uint meshIdx = 0; meshIdx < shell_list.size(); meshIdx++)
        {
            const std::string& curr_mesh_name = shell_list[meshIdx].model_name + std::to_string(meshIdx);
            polyscope::SurfaceMesh* curr_mesh_ptr = polyscope::registerSurfaceMesh(
                curr_mesh_name, sa_rendering_vertices[meshIdx], sa_rendering_faces[meshIdx]);
            curr_mesh_ptr->setEnabled(true);
            curr_mesh_ptr->addFaceColorQuantity("Collision Count", face_color[meshIdx]);
            surface_meshes.push_back(curr_mesh_ptr);
        }

        auto fn_update_GUI_vertices = [&]()
        {
            for (uint clothIdx = 0; clothIdx < shell_list.size(); clothIdx++)
            {
                surface_meshes[clothIdx]->updateVertexPositions(sa_rendering_vertices[clothIdx]);
            }
        };
        auto fn_single_step_with_ui = [&]()
        {
            // LUISA_INFO("     Sync frame {}", lcs::get_scene_params().current_frame);
            fn_physics_step();

            fn_update_rendering_vertices();
            fn_update_GUI_vertices();
        };

        bool is_simulate_frame = false;

        polyscope::state::userCallback = [&]()
        {
            if (ImGui::IsKeyPressed(ImGuiKey_Escape))
                polyscope::unshow();

            // Selection
            {
                static polyscope::PickResult prev_selection;
                if (polyscope::haveSelection())
                {
                    const auto& host_mesh_data = solver.get_host_mesh_data();

                    polyscope::PickResult selection = polyscope::getSelection();
                    if (selection.isHit
                        && !(selection.screenCoords.x == prev_selection.screenCoords.x
                             && selection.screenCoords.y == prev_selection.screenCoords.y))
                    {
                        prev_selection = selection;
                        for (uint meshIdx = 0; meshIdx < surface_meshes.size(); meshIdx++)
                        {
                            polyscope::SurfaceMesh* mesh = surface_meshes[meshIdx];
                            if (mesh == selection.structure)
                            {
                                polyscope::SurfaceMeshPickResult meshPickResult =
                                    mesh->interpretPickResult(selection);
                                if (meshPickResult.elementType == polyscope::MeshElement::VERTEX)
                                {
                                    uint prefix = host_mesh_data.prefix_num_verts[meshIdx];
                                    uint vid    = prefix + meshPickResult.index;
                                    LUISA_INFO("Select Vert {:3} on mesh {}", vid, meshIdx);
                                }
                                else if (meshPickResult.elementType == polyscope::MeshElement::FACE)
                                {
                                    uint prefix = host_mesh_data.prefix_num_faces[meshIdx];
                                    uint vid    = prefix + meshPickResult.index;
                                    LUISA_INFO("Select Face {:3} on mesh {}", vid, meshIdx);
                                }
                                else if (meshPickResult.elementType == polyscope::MeshElement::EDGE)
                                {
                                    uint prefix = host_mesh_data.prefix_num_edges[meshIdx];
                                    uint vid    = prefix + meshPickResult.index;
                                    LUISA_INFO("Select Edge {:3} on mesh {}", vid, meshIdx);
                                }
                            }
                        }
                    }
                }
            }

            if (ImGui::CollapsingHeader("Parameters", ImGuiTreeNodeFlags_DefaultOpen))
            {
                // ImGui::InputScalar("Num Substep", ImGuiDataType_U32, &lcs::get_scene_params().num_substep);
                ImGui::InputScalar("Num Nonliear-Iteration", ImGuiDataType_U32, &lcs::get_scene_params().nonlinear_iter_count);
                ImGui::InputScalar("Num PCG-Iteration", ImGuiDataType_U32, &lcs::get_scene_params().pcg_iter_count);
                ImGui::SliderFloat("Implicit Timestep", &lcs::get_scene_params().implicit_dt, 0.0001f, 0.2f);
                ImGui::Checkbox("Use Energy LineSearch", &lcs::get_scene_params().use_energy_linesearch);
                ImGui::Checkbox("Use CCD LineSearch", &lcs::get_scene_params().use_ccd_linesearch);
                if (lcs::get_scene_params().contact_energy_type == uint(lcs::ContactEnergyType::Barrier))
                    lcs::get_scene_params().use_ccd_linesearch = true;


                // ImGui::Checkbox("Use Bending", &lcs::get_scene_params().use_bending);
                // ImGui::Checkbox("Use Quadratic Bending", &lcs::get_scene_params().use_quadratic_bending_model);
                ImGui::SliderFloat("Bending Stiffness", &lcs::get_scene_params().stiffness_bending_ui, 0.0f, 10.0f);
                // static int stiffness_bending_exp = 0;
                // ImGui::InputInt("Bending Stiffness's Exp", &stiffness_bending_exp);
                // lcs::get_scene_params().stiffness_bending_ui = pow(10.0f, (float)stiffness_bending_exp);

                static uint stiffness_spring_exp = 4;
                ImGui::InputScalar("Stretch Stiffness's Exp", ImGuiDataType_U32, &stiffness_spring_exp);
                lcs::get_scene_params().stiffness_spring = pow(10.0f, (float)stiffness_spring_exp);
                // ImGui::Checkbox("Print Convergence", &lcs::get_scene_params().print_xpbd_convergence);
                ImGui::Checkbox("Print Energy", &lcs::get_scene_params().print_system_energy);
                ImGui::Checkbox("Use GPU Solver", &lcs::get_scene_params().use_gpu);
                ImGui::Checkbox("Use Self-Collision", &lcs::get_scene_params().use_self_collision);
                // ImGui::Checkbox("Print PCG Convergence", &lcs::get_scene_params().print_pcg_convergence);

                // static const char* items[] = { "A", "B", "C" };
                // static int current_item = 0;
                // ImGui::Combo("Combo", &current_item, items, IM_ARRAYSIZE(items));
            }

            if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::Text("Frame %d", lcs::get_scene_params().current_frame);
                if (ImGui::Button("Reset", ImVec2(-1, 0)))
                {
                    lcs::get_scene_params().current_frame = 0;
                    max_frame                             = 0;
                    solver.lcs::SolverInterface::restart_system();
                    fn_update_rendering_vertices();
                    fn_update_GUI_vertices();
                }
                if (ImGui::Button("Advance Single Frame", ImVec2(-1, 0)))
                {
                    fn_single_step_with_ui();
                }
                if (ImGui::Button("Advance Some Frame", ImVec2(-1, 0)))
                {
                    is_simulate_frame = true;
                    max_frame         = lcs::get_scene_params().current_frame + optimize_frames;
                }
                ImGui::InputScalar("Advance Frame Count", ImGuiDataType_U32, &optimize_frames);
                if (ImGui::Button("Start Simulation", ImVec2(-1, 0)))
                {
                    is_simulate_frame = true;
                    max_frame         = 10000;
                }
                if (ImGui::Button("End Simulation", ImVec2(-1, 0)))
                {
                    is_simulate_frame = false;
                    max_frame         = lcs::get_scene_params().current_frame;
                }
            }

            if (ImGui::CollapsingHeader("Collision", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::Checkbox("Use Ground Collision", &lcs::get_scene_params().use_floor);
                if (lcs::get_scene_params().use_floor)
                {
                    ImGui::SliderFloat("Floor Y", &lcs::get_scene_params().floor.y, -1.0f, 1.0f);
                    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::TileReflection;
                    polyscope::options::groundPlaneHeightMode = polyscope::GroundPlaneHeightMode::Manual;
                    polyscope::options::groundPlaneHeight     = lcs::get_scene_params().floor.y;
                }
                else
                {
                    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;
                }
                const auto& host_collision_data = solver.get_host_collision_data();
                const auto& collision_data      = solver.get_device_collision_data();
                const uint  offset_pairs = lcs::CollisionPair::CollisionCount::total_adj_pairs_offset();
                const uint  offset_verts = lcs::CollisionPair::CollisionCount::total_adj_verts_offset();
                ImGui::Text("Broad VF/EE = %u / %u Narrow = %u , Triplet = %u",
                            host_collision_data.broad_phase_collision_count[2],
                            host_collision_data.broad_phase_collision_count[3],
                            host_collision_data.narrow_phase_collision_count[offset_pairs],
                            host_collision_data.narrow_phase_collision_count[offset_verts]);
                ImGui::Text("MaxCount = %zu / %zu ,  Narrow %zu , Triplet = %zu)",
                            host_collision_data.broad_phase_list_vf.size() / 2,
                            host_collision_data.broad_phase_list_ee.size() / 2,
                            host_collision_data.narrow_phase_list.size(),
                            collision_data.sa_cgA_contact_offdiag_triplet.size());
            }

            if (ImGui::CollapsingHeader("Data IO", ImGuiTreeNodeFlags_DefaultOpen))
            {
                if (ImGui::Button("Save mesh", ImVec2(-1, 0)))
                {
                    SimMesh::saveToOBJ_combined(
                        sa_rendering_vertices, sa_rendering_faces, "", "", lcs::get_scene_params().current_frame);
                }
                ImGui::Checkbox("Output Each Frame", &lcs::get_scene_params().output_per_frame);
                if (ImGui::Button("Save State", ImVec2(-1, 0)))
                {
                    solver.lcs::SolverInterface::save_current_frame_state_to_host(lcs::get_scene_params().current_frame,
                                                                                  "");
                }
                uint& state_frame = lcs::get_scene_params().load_state_frame;
                ImGui::InputScalar("Load State Frame", ImGuiDataType_U32, &state_frame);
                if (ImGui::Button("Load State", ImVec2(-1, 0)))
                {
                    solver.lcs::SolverInterface::load_saved_state_from_host(state_frame, "");
                    lcs::get_scene_params().current_frame = state_frame;
                    ;
                    fn_update_rendering_vertices();
                    fn_update_GUI_vertices();
                }
            }

            if (is_simulate_frame)
            {
                if (lcs::get_scene_params().output_per_frame && lcs::get_scene_params().current_frame == 0)
                {
                    // First frame
                    SimMesh::saveToOBJ_combined(sa_rendering_vertices,
                                                sa_rendering_faces,
                                                std::format("0{}", lcs::get_scene_params().scene_id),
                                                "start",
                                                lcs::get_scene_params().current_frame);
                }

                fn_single_step_with_ui();

                const float animation_fps = 60.0f;
                const uint  output_freq   = (1.0f / animation_fps) / lcs::get_scene_params().implicit_dt;
                if (lcs::get_scene_params().output_per_frame && lcs::get_scene_params().current_frame % output_freq == 0)
                {
                    SimMesh::saveToOBJ_combined(sa_rendering_vertices,
                                                sa_rendering_faces,
                                                std::format("0{}", lcs::get_scene_params().scene_id),
                                                "",
                                                lcs::get_scene_params().current_frame);
                }
                if (lcs::get_scene_params().current_frame >= max_frame)
                {
                    is_simulate_frame = false;
                }
            }
        };
        polyscope::show();
    }
#endif
    return 0;
}