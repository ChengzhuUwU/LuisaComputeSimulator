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
#include "SimulationCore/solver_interface.h"
#include "SimulationSolver/descent_solver.h"

#include "Initializer/init_mesh_data.h"
#include "Initializer/init_xpbd_data.h"
#include "app_simulation_demo_config.h"
#include "polyscope/volume_grid.h"

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <Eigen/Dense>

template<typename T>
using Buffer = luisa::compute::Buffer<T>;

namespace lcsv::Initializer
{


void init_simulation_params()
{
    // if (lcsv::get_scene_params().use_small_timestep) { lcsv::get_scene_params().implicit_dt = 0.001f; }
    
    // lcsv::get_scene_params().num_iteration = lcsv::get_scene_params().num_substep * lcsv::get_scene_params().nonlinear_iter_count;
    // lcsv::get_scene_params().collision_detection_frequece = 1;    

    // lcsv::get_scene_params().stiffness_stretch_spring = FEM::calcSecondLame(lcsv::get_scene_params().youngs_modulus_cloth, lcsv::get_scene_params().poisson_ratio_cloth); // mu;
    // lcsv::get_scene_params().stiffness_pressure = 1e6;
    
    {
        // lcsv::get_scene_params().stiffness_stretch_spring = 1e4;
        // lcsv::get_scene_params().xpbd_stiffness_collision = 1e7;
        // lcsv::get_scene_params().stiffness_quadratic_bending = 5e-3;
        // lcsv::get_scene_params().stiffness_DAB_bending = 5e-3;
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
    const std::string binary_path(argv[0]);
    luisa::compute::Context context{ binary_path };
    luisa::vector<luisa::string> device_names = context.backend_device_names(backend);
    if (device_names.empty()) { LUISA_WARNING("No haredware device found."); exit(1); }
    for (size_t i = 0; i < device_names.size(); ++i) { luisa::log_info("Device {}: {}", i, device_names[i]); }
    luisa::compute::Device device = context.create_device(backend);
    luisa::compute::Stream stream = device.create_stream(luisa::compute::StreamTag::COMPUTE);

    lcsv::get_scene_params().solver_type = lcsv::SolverTypeNewton;

    // Some params
    {
        lcsv::get_scene_params().implicit_dt = 0.05;
        lcsv::get_scene_params().num_substep = 1;
        lcsv::get_scene_params().nonlinear_iter_count = 50;
        lcsv::get_scene_params().pcg_iter_count = 2000;
        // lcsv::get_scene_params().use_bending = false;
        // lcsv::get_scene_params().use_quadratic_bending_model = true;
        // lcsv::get_scene_params().use_xpbd_solver = false;
        // lcsv::get_scene_params().use_vbd_solver = false;
        // lcsv::get_scene_params().use_newton_solver = true;
        lcsv::get_scene_params().use_gpu = false; // true
    }

    // Read Mesh
    std::vector<lcsv::Initializer::ShellInfo> shell_list;
    Demo::Simulation::load_scene(shell_list);
    

    // Init data
    lcsv::MeshData<std::vector>             host_mesh_data;
    lcsv::MeshData<luisa::compute::Buffer>  mesh_data;
    {
        lcsv::Initializer::init_mesh_data(shell_list, &host_mesh_data);
        lcsv::Initializer::upload_mesh_buffers(device, stream, &host_mesh_data, &mesh_data);
    }

    lcsv::SimulationData<std::vector>               host_xpbd_data;
    lcsv::SimulationData<luisa::compute::Buffer>    xpbd_data;
    {
        lcsv::Initializer::init_xpbd_data(&host_mesh_data, &host_xpbd_data);
        lcsv::Initializer::upload_xpbd_buffers(device, stream, &host_xpbd_data, &xpbd_data);
        lcsv::Initializer::resize_pcg_data(device, stream, &host_mesh_data, &host_xpbd_data, &xpbd_data);
        lcsv::Initializer::init_simulation_params();
    }

    lcsv::LbvhData<luisa::compute::Buffer>  lbvh_data_face;
    lcsv::LbvhData<luisa::compute::Buffer>  lbvh_data_edge;
    {
        lbvh_data_face.allocate(device, host_mesh_data.num_faces, lcsv::LBVHTreeTypeFace, lcsv::LBVHUpdateTypeCloth);
        lbvh_data_edge.allocate(device, host_mesh_data.num_edges, lcsv::LBVHTreeTypeEdge, lcsv::LBVHUpdateTypeCloth);
        // lbvh_cloth_vert.unit_test(device, stream);
    }
    
    lcsv::CollisionData<std::vector>             host_collision_data;
    lcsv::CollisionData<luisa::compute::Buffer>  collision_data;
    {
        host_collision_data.resize_collision_data(device, host_mesh_data.num_verts, host_mesh_data.num_faces, host_mesh_data.num_edges);
        collision_data.resize_collision_data(device, host_mesh_data.num_verts, host_mesh_data.num_faces, host_mesh_data.num_edges);
    }

    // Init solver class
    lcsv::BufferFiller   buffer_filler;
    lcsv::DeviceParallel device_parallel;

    lcsv::LBVH           lbvh_face;
    lcsv::LBVH           lbvh_edge;
    {
        lbvh_face.set_lbvh_data(&lbvh_data_face);
        lbvh_edge.set_lbvh_data(&lbvh_data_edge);
        lbvh_face.compile(device);
        lbvh_edge.compile(device);
    }

    lcsv::NarrowPhasesDetector narrow_phase_detector;
    {
        narrow_phase_detector.set_collision_data(&host_collision_data, &collision_data);
        narrow_phase_detector.compile(device);
        narrow_phase_detector.unit_test(device, stream);
    }
    
    lcsv::ConjugateGradientSolver pcg_solver;
    {
        pcg_solver.set_data(
            &host_mesh_data, 
            &mesh_data, 
            &host_xpbd_data, 
            &xpbd_data
        );
        pcg_solver.compile(device);
    }

    // lcsv::DescentSolver  solver;
    lcsv::NewtonSolver      solver;
    {
        // device_parallel.create(device); // TODO: Check CUDA backend on windows's debug mode
        solver.lcsv::SolverInterface::set_data_pointer(
            &host_mesh_data, 
            &mesh_data, 
            &host_xpbd_data, 
            &xpbd_data, 
            &host_collision_data,
            &collision_data,
            &lbvh_face,
            &lbvh_edge,
            &buffer_filler, 
            &device_parallel,
            &narrow_phase_detector,
            &pcg_solver
        );
        solver.compile(device);
    }

    // Define Simulation
    {
        solver.lcsv::SolverInterface::restart_system();
        luisa::log_info("Simulation begin...");
    }

    auto fn_physics_step = [&]()
    {
        auto fn_fixed_point_animation = [&](const uint curr_frame)
        {
            // Animation for fixed points
            for (uint clothIdx = 0; clothIdx < shell_list.size(); clothIdx++)
            {
                const auto& fixed_point_info = shell_list[clothIdx].fixed_point_list;
                if (fixed_point_info.empty()) continue;
                
                for (const auto& fixed_point : fixed_point_info)
                {
                    if (fixed_point.use_rotate)
                    {
                        const std::vector<uint>& fixed_point_verts = fixed_point.fixed_point_verts;
                        CpuParallel::parallel_for(0, fixed_point_verts.size(), [&](const uint index)
                        {
                            const uint vid = fixed_point_verts[index];
                            auto& pos = host_mesh_data.sa_x_frame_outer[vid];
                            {
                                // Rotate
                                const float h = lcsv::get_scene_params().implicit_dt;
                                const float rotAngRad = curr_frame * fixed_point.rotAngVelDeg / 180.0f * float(lcsv::Pi) * h;
                                const Eigen::Vector3d rotAxis(
                                    fixed_point.rotAxis[0],
                                    fixed_point.rotAxis[1],
                                    fixed_point.rotAxis[2]);
                                const Eigen::Vector3d rotCenter(
                                    fixed_point.rotCenter[0],
                                    fixed_point.rotCenter[1],
                                    fixed_point.rotCenter[2]);
                                const Eigen::Matrix3d rotMtr = Eigen::AngleAxisd(rotAngRad, rotAxis.normalized()).toRotationMatrix();
                                Eigen::Vector3d relative_vec(
                                    pos[0] - rotCenter[0],
                                    pos[1] - rotCenter[1],
                                    pos[2] - rotCenter[2]);
                                Eigen::Vector3d rotx = rotMtr * relative_vec;
                                pos[0] = rotx[0] + rotCenter[0];
                                pos[1] = rotx[1] + rotCenter[1];
                                pos[2] = rotx[2] + rotCenter[2];
                            }
                        });
                    }
                }
            }
        };
        fn_fixed_point_animation(lcsv::get_scene_params().current_frame);

        if (lcsv::get_scene_params().use_gpu)
            solver.physics_step_GPU(device, stream);
        else
            solver.physics_step_CPU(device, stream);

        lcsv::get_scene_params().current_frame += 1; 
    };


    uint max_frame = 20; 
    constexpr bool draw_bounding_box = true;
    constexpr bool use_ui = true; 
    
    // Init rendering data
    std::vector<std::vector<std::array<float, 3>>> sa_rendering_vertices(shell_list.size() + 0 + 0);
    std::vector<std::vector<std::array<uint, 3>>> sa_rendering_faces(shell_list.size() + 0 + 0);
    std::vector<std::array<float, 3>> sa_global_aabb_vertices(SimMesh::BoundingBox::get_num_vertices(), std::array<float, 3>({0.0f, 0.0f, 0.0f}));
    std::vector<std::array<uint, 3>> sa_global_aabb_faces = SimMesh::BoundingBox::get_box_faces();
    std::vector<std::vector<std::array<float, 3>>> face_color(shell_list.size());
    {
        for (uint meshIdx = 0; meshIdx < shell_list.size(); meshIdx++)
        {
            sa_rendering_vertices[meshIdx].resize(host_mesh_data.prefix_num_verts[meshIdx + 1] - host_mesh_data.prefix_num_verts[meshIdx]);
            sa_rendering_faces[meshIdx].resize(host_mesh_data.prefix_num_faces[meshIdx + 1] - host_mesh_data.prefix_num_faces[meshIdx]);
            const uint curr_prefix_num_verts = host_mesh_data.prefix_num_verts[meshIdx];
            const uint next_prefix_num_verts = host_mesh_data.prefix_num_verts[meshIdx + 1];
            const uint curr_prefix_num_faces = host_mesh_data.prefix_num_faces[meshIdx];
            const uint next_prefix_num_faces = host_mesh_data.prefix_num_faces[meshIdx + 1];
            CpuParallel::parallel_for(0, next_prefix_num_verts - curr_prefix_num_verts, [&](const uint vid)
            {
                auto pos = host_mesh_data.sa_rest_x[curr_prefix_num_verts + vid];
                // sa_rendering_vertices[i][vid] = glm::vec3(pos.x, pos.y, pos.z);
                sa_rendering_vertices[meshIdx][vid] = {pos.x, pos.y, pos.z};
            });
            CpuParallel::parallel_for(0, next_prefix_num_faces - curr_prefix_num_faces, [&](const uint fid)
            {
                auto face = host_mesh_data.sa_faces[curr_prefix_num_faces + fid];
                sa_rendering_faces[meshIdx][fid] = {
                    face[0] - curr_prefix_num_verts, 
                    face[1] - curr_prefix_num_verts, 
                    face[2] - curr_prefix_num_verts};
            });
            face_color[meshIdx].resize(host_mesh_data.prefix_num_faces[meshIdx + 1] - host_mesh_data.prefix_num_faces[meshIdx], {0.7, 0.2, 0.3});
        }

        if constexpr (draw_bounding_box)
        {
            std::array<float, 3> min_pos = { -0.01f, -0.01f, -0.01f };; std::array<float, 3> max_pos = {  0.01f,  0.01f,  0.01f };
            SimMesh::BoundingBox::update_vertices(sa_global_aabb_vertices, min_pos, max_pos);
        }
    }
    auto fn_update_rendering_vertices = [&]()
    {
        for (uint clothIdx = 0; clothIdx < shell_list.size(); clothIdx++)
        {
            CpuParallel::parallel_for(0, host_mesh_data.prefix_num_verts[clothIdx + 1] - host_mesh_data.prefix_num_verts[clothIdx], [&](const uint vid)
            {
                auto pos = host_mesh_data.sa_x_frame_outer[vid + host_mesh_data.prefix_num_verts[clothIdx]];
                sa_rendering_vertices[clothIdx][vid] = {pos.x, pos.y, pos.z};
            });
        }
        if constexpr (draw_bounding_box)
        {
            lcsv::float2x3 global_aabb; std::array<float, 3> min_pos; std::array<float, 3> max_pos; 
            // stream << lbvh_data_face.sa_node_aabb.view(0, 1).copy_to(&global_aabb) << luisa::compute::synchronize();
            // stream << lbvh_data_face.sa_block_aabb.view(0, 1).copy_to(&global_aabb) << luisa::compute::synchronize();
            global_aabb = lbvh_data_face.host_node_aabb[0];
            min_pos = { global_aabb[0][0], global_aabb[0][1], global_aabb[0][2] };
            max_pos = { global_aabb[1][0], global_aabb[1][1], global_aabb[1][2] };
            SimMesh::BoundingBox::update_vertices(sa_global_aabb_vertices, min_pos, max_pos);
        }
    };

    // SimMesh::saveToOBJ_combined(sa_rendering_vertices, sa_rendering_faces, "_init", 0);

    if constexpr (!use_ui)
    {
        auto fn_single_step_without_ui = [&]()
        {
            luisa::log_info("     Newton solver frame {}", lcsv::get_scene_params().current_frame);   

            fn_physics_step();
        };

        // solver.lcsv::SolverInterface::restart_system();

        // for (uint frame = 0; frame < max_frame; frame++)
        {
            fn_single_step_without_ui();
        }
        fn_update_rendering_vertices();
        SimMesh::saveToOBJ_combined(sa_rendering_vertices, sa_rendering_faces, "", lcsv::get_scene_params().current_frame);
        // solver.lcsv::SolverInterface::save_mesh_to_obj(lcsv::get_scene_params().current_frame, ""); 
    }
    else
    {
        // Init Polyscope
        polyscope::init("openGL3_glfw");
        std::vector<polyscope::SurfaceMesh*> surface_meshes;
        std::vector<polyscope::SurfaceMesh*> bounding_boxes;
        
        
        for (uint meshIdx = 0; meshIdx < shell_list.size(); meshIdx++)
        {
            const std::string& curr_mesh_name = shell_list[meshIdx].model_name + std::to_string(meshIdx);
            polyscope::SurfaceMesh* curr_mesh_ptr = polyscope::registerSurfaceMesh(
                curr_mesh_name, 
                sa_rendering_vertices[meshIdx], 
                sa_rendering_faces[meshIdx]
            );
            curr_mesh_ptr->setEnabled(true);
            curr_mesh_ptr->addFaceColorQuantity("Collision Count", face_color[meshIdx]);
            surface_meshes.push_back(curr_mesh_ptr);
        }
        
        if constexpr (draw_bounding_box)
        {
            polyscope::SurfaceMesh* bounding_box_ptr = polyscope::registerSurfaceMesh("Global Bounding Box", sa_global_aabb_vertices, sa_global_aabb_faces);
            bounding_box_ptr->setTransparency(0.25f);
            bounding_boxes.push_back(bounding_box_ptr);
        }

        polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;
        
        
        auto fn_update_GUI_vertices = [&]()
        {
            for (uint clothIdx = 0; clothIdx < shell_list.size(); clothIdx++)
            {
                surface_meshes[clothIdx]->updateVertexPositions(sa_rendering_vertices[clothIdx]);
                
                
                // surface_meshes[clothIdx]->getFloatingQuantity("Collision Count");
            }
            if constexpr (draw_bounding_box) bounding_boxes.back()->updateVertexPositions(sa_global_aabb_vertices);
        };
        auto fn_single_step_with_ui = [&]()
        {
            // luisa::log_info("     Sync frame {}", lcsv::get_scene_params().current_frame);   
            fn_physics_step();

            fn_update_rendering_vertices();
            fn_update_GUI_vertices();
        };
        
        bool is_simulate_frame = false;

        polyscope::state::userCallback = [&]()
        {
            if (ImGui::IsKeyPressed(ImGuiKey_Escape)) polyscope::unshow();

            if (ImGui::CollapsingHeader("Parameters", ImGuiTreeNodeFlags_DefaultOpen)) 
            {
                ImGui::InputScalar("Max Frame", ImGuiDataType_U32, &max_frame);
                ImGui::InputScalar("Num Substep", ImGuiDataType_U32, &lcsv::get_scene_params().num_substep);
                ImGui::InputScalar("Num Nonliear-Iteration", ImGuiDataType_U32, &lcsv::get_scene_params().nonlinear_iter_count);
                ImGui::InputScalar("Num PCG-Iteration", ImGuiDataType_U32, &lcsv::get_scene_params().pcg_iter_count);
                ImGui::SliderFloat("Implicit Timestep", &lcsv::get_scene_params().implicit_dt, 0.0001f, 0.2f); 
                ImGui::Checkbox("Use Bending", &lcsv::get_scene_params().use_bending);
                ImGui::Checkbox("Use Quadratic Bending", &lcsv::get_scene_params().use_quadratic_bending_model);
                ImGui::SliderFloat("Bending Stiffness", &lcsv::get_scene_params().stiffness_bending_ui, 0.0f, 1.0f); 
                // ImGui::Checkbox("Print Convergence", &lcsv::get_scene_params().print_xpbd_convergence);
                ImGui::Checkbox("Print Energy", &lcsv::get_scene_params().print_system_energy);
                ImGui::Checkbox("Use GPU Solver", &lcsv::get_scene_params().use_gpu);
                // ImGui::Checkbox("Print PCG Convergence", &lcsv::get_scene_params().print_pcg_convergence);

                // static const char* items[] = { "A", "B", "C" };
                // static int current_item = 0;
                // ImGui::Combo("Combo", &current_item, items, IM_ARRAYSIZE(items));
            }

            if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen)) 
            {
                ImGui::Text("Frame %d", lcsv::get_scene_params().current_frame);
                if (ImGui::Button("Reset", ImVec2(-1, 0))) 
                {
                    lcsv::get_scene_params().current_frame = 0;
                    solver.lcsv::SolverInterface::restart_system();
                    fn_update_rendering_vertices();
                    fn_update_GUI_vertices();
                }
                if (ImGui::Button("Optimize Single Step", ImVec2(-1, 0)))
                {
                    fn_single_step_with_ui();
                }
                if (ImGui::Button("Optimize Some Step", ImVec2(-1, 0)))
                {
                    is_simulate_frame = true;
                }
            }

            if (ImGui::CollapsingHeader("Data IO", ImGuiTreeNodeFlags_DefaultOpen)) 
            {
                if (ImGui::Button("Save mesh", ImVec2(-1, 0)))
                {
                    SimMesh::saveToOBJ_combined(sa_rendering_vertices, sa_rendering_faces, "", lcsv::get_scene_params().current_frame);
                }
                if (ImGui::Button("Save State", ImVec2(-1, 0)))
                {
                    solver.lcsv::SolverInterface::save_current_frame_state_to_host(lcsv::get_scene_params().current_frame, "");
                }
                uint& state_frame = lcsv::get_scene_params().load_state_frame;
                ImGui::InputScalar("Load State Frame", ImGuiDataType_U32, &state_frame);
                if (ImGui::Button("Load State", ImVec2(-1, 0)))
                {
                    solver.lcsv::SolverInterface::load_saved_state_from_host(state_frame, "");
                    lcsv::get_scene_params().current_frame = state_frame;;
                    fn_update_rendering_vertices();
                    fn_update_GUI_vertices();
                }
            }
            
            if (is_simulate_frame)
            {
                fn_single_step_with_ui();
                if (lcsv::get_scene_params().current_frame >= max_frame)
                {
                    is_simulate_frame = false;
                    SimMesh::saveToOBJ_combined(sa_rendering_vertices, sa_rendering_faces, "", lcsv::get_scene_params().current_frame);
                }
            }
        };
        polyscope::show();
    }

    return 0;
}