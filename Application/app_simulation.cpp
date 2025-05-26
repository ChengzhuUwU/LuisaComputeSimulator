#include <iostream>
#include <luisa/luisa-compute.h>

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

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>

template<typename T>
using Buffer = luisa::compute::Buffer<T>;

namespace lcsv::Initializater
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
    luisa::compute::Context context{ argv[0] };
    luisa::compute::Device device = context.create_device(backend);
    luisa::compute::Stream stream = device.create_stream(luisa::compute::StreamTag::COMPUTE);

    lcsv::get_scene_params().solver_type = lcsv::SolverTypeNewton;

    // Read Mesh
    std::vector<lcsv::Initializater::ShellInfo> shell_list;
    const std::string obj_mesh_path = std::string(LCSV_RESOURCE_PATH) + "/InputMesh/";
    const std::string tet_mesh_path = std::string(LCSV_RESOURCE_PATH) + "/InputMesh/vtks/";
    shell_list.push_back({
        .model_name = obj_mesh_path + "Cylinder/cylinder7K.obj",
        .fixed_point_info = {
            lcsv::Initializater::FixedPointInfo{
                // .is_fixed_point_func = [](const luisa::float3& norm_pos) { return norm_pos.z < 0.001f && (norm_pos.x > 0.999f || norm_pos.x < 0.001f ); }
                .is_fixed_point_func = [](const luisa::float3& norm_pos) { return (norm_pos.x > 0.999f || norm_pos.x < 0.001f ); }
            }
        }
    });
    shell_list.push_back({
        .model_name = obj_mesh_path + "square8K.obj",
        .transform = luisa::make_float3(0.0f, 1.0f, 0.0f),
        .fixed_point_info = {
            lcsv::Initializater::FixedPointInfo{
                .is_fixed_point_func = [](const luisa::float3& norm_pos) { return (norm_pos.x > 0.999f || norm_pos.x < 0.001f ); }
            }
        }
    });

    // Init data
    lcsv::MeshData<std::vector> cpu_mesh_data;
    lcsv::MeshData<luisa::compute::Buffer> mesh_data;
    {
        lcsv::Initializater::init_mesh_data(shell_list, &cpu_mesh_data);
        lcsv::Initializater::upload_mesh_buffers(device, stream, &cpu_mesh_data, &mesh_data);
    }

    lcsv::SimulationData<std::vector> cpu_xpbd_data;
    lcsv::SimulationData<luisa::compute::Buffer> xpbd_data;
    {
        lcsv::Initializater::init_xpbd_data(&cpu_mesh_data, &cpu_xpbd_data);
        lcsv::Initializater::upload_xpbd_buffers(device, stream, &cpu_xpbd_data, &xpbd_data);
        lcsv::Initializater::init_simulation_params();
    }

    // Init solver class
    lcsv::BufferFiller   buffer_filler;
    lcsv::DeviceParallel device_parallel;
    // lcsv::DescentSolver solver;
    lcsv::NewtonSolver solver;
    {
        // device_parallel.create(device); // TODO: Check CUDA backend on windows's debug mode
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
        lcsv::get_scene_params().implicit_dt = 0.05;
        lcsv::get_scene_params().num_substep = 1;
        lcsv::get_scene_params().nonlinear_iter_count = 20;   
        lcsv::get_scene_params().pcg_iter_count = 2000; 
        lcsv::get_scene_params().use_bending = false;
        lcsv::get_scene_params().use_quadratic_bending_model = true;
        lcsv::get_scene_params().use_xpbd_solver = false;
        lcsv::get_scene_params().use_vbd_solver = true;
    }

    // Define Simulation
    {
        solver.lcsv::SolverInterface::restart_system();
        luisa::log_info("");
        luisa::log_info("");
    }
    auto fn_physics_step = [&]()
    {
        solver.physics_step_CPU(device, stream);
    };


    uint max_frame = 20; 
    
    // Init rendering data
    std::vector<std::vector<std::array<float, 3>>> sa_rendering_vertices(shell_list.size() + 0 + 0);
    std::vector<std::vector<std::array<uint, 3>>> sa_rendering_faces(shell_list.size() + 0 + 0);
    for (uint i = 0; i < shell_list.size(); i++)
    {
        sa_rendering_vertices[i].resize(cpu_mesh_data.prefix_num_verts[i + 1] - cpu_mesh_data.prefix_num_verts[i]);
        sa_rendering_faces[i].resize(cpu_mesh_data.prefix_num_faces[i + 1] - cpu_mesh_data.prefix_num_faces[i]);
        const uint curr_prefix_num_verts = cpu_mesh_data.prefix_num_verts[i];
        const uint next_prefix_num_verts = cpu_mesh_data.prefix_num_verts[i + 1];
        const uint curr_prefix_num_faces = cpu_mesh_data.prefix_num_faces[i];
        const uint next_prefix_num_faces = cpu_mesh_data.prefix_num_faces[i + 1];
        CpuParallel::parallel_for(0, next_prefix_num_verts - curr_prefix_num_verts, [&](const uint vid)
        {
            auto pos = cpu_mesh_data.sa_rest_x[curr_prefix_num_verts + vid];
            // sa_rendering_vertices[i][vid] = glm::vec3(pos.x, pos.y, pos.z);
            sa_rendering_vertices[i][vid] = {pos.x, pos.y, pos.z};
        });
        CpuParallel::parallel_for(0, next_prefix_num_faces - curr_prefix_num_faces, [&](const uint fid)
        {
            auto face = cpu_mesh_data.sa_faces[curr_prefix_num_faces + fid];
            sa_rendering_faces[i][fid] = {
                face[0] - curr_prefix_num_verts, 
                face[1] - curr_prefix_num_verts, 
                face[2] - curr_prefix_num_verts};
        });
    }
    SimMesh::saveToOBJ_combined(sa_rendering_vertices, sa_rendering_faces, "_init", 0);

    constexpr bool use_ui = true; 
    if constexpr (!use_ui)
    {
        auto fn_single_step_without_ui = [&]()
        {
            luisa::log_info("     Newton solver frame {}", lcsv::get_scene_params().current_frame);   

            fn_physics_step();

            lcsv::get_scene_params().current_frame += 1; 
        };

        auto fn_update_rendering_vertices = [&]()
        {
            for (uint clothIdx = 0; clothIdx < shell_list.size(); clothIdx++)
            {
                CpuParallel::parallel_for(0, cpu_mesh_data.prefix_num_verts[clothIdx + 1] - cpu_mesh_data.prefix_num_verts[clothIdx], [&](const uint vid)
                {
                    auto pos = cpu_mesh_data.sa_x_frame_outer[vid + cpu_mesh_data.prefix_num_verts[clothIdx]];
                    sa_rendering_vertices[clothIdx][vid] = {pos.x, pos.y, pos.z};
                });
            }
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
        for (uint clothIdx = 0; clothIdx < shell_list.size(); clothIdx++)
        {
            const std::string& curr_mesh_name = shell_list[clothIdx].model_name + std::to_string(clothIdx);
            polyscope::SurfaceMesh* curr_mesh_ptr = polyscope::registerSurfaceMesh(
                curr_mesh_name, 
                sa_rendering_vertices[clothIdx], 
                sa_rendering_faces[clothIdx]
            );
            curr_mesh_ptr->setEnabled(true);
            surface_meshes.push_back(curr_mesh_ptr);
        }
        polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

        // Define single step in GUI
        auto fn_update_rendering_vertices = [&]()
        {
            for (uint clothIdx = 0; clothIdx < shell_list.size(); clothIdx++)
            {
                CpuParallel::parallel_for(0, cpu_mesh_data.prefix_num_verts[clothIdx + 1] - cpu_mesh_data.prefix_num_verts[clothIdx], [&](const uint vid)
                {
                    auto pos = cpu_mesh_data.sa_x_frame_outer[vid + cpu_mesh_data.prefix_num_verts[clothIdx]];
                    // sa_rendering_vertices[clothIdx][vid] = glm::vec3(pos.x, pos.y, pos.z);
                    sa_rendering_vertices[clothIdx][vid] = {pos.x, pos.y, pos.z};
                });
                surface_meshes[clothIdx]->updateVertexPositions(sa_rendering_vertices[clothIdx]);
            }
        };
        auto fn_single_step_with_ui = [&]()
        {
            // luisa::log_info("     Sync frame {}", lcsv::get_scene_params().current_frame);   
            fn_physics_step();

            lcsv::get_scene_params().current_frame += 1; 
            fn_update_rendering_vertices();
        };
        
        bool is_simulate_frame = false;

        polyscope::state::userCallback = [&]()
        {
            if (ImGui::IsKeyPressed(ImGuiKey_Escape)) polyscope::unshow();

            if (ImGui::CollapsingHeader("Parameters", ImGuiTreeNodeFlags_DefaultOpen)) 
            {
                ImGui::InputScalar("Num Substep", ImGuiDataType_U32, &lcsv::get_scene_params().num_substep);
                ImGui::InputScalar("Num Nonliear-Iteration", ImGuiDataType_U32, &lcsv::get_scene_params().nonlinear_iter_count);
                ImGui::SliderFloat("Implicit Timestep", &lcsv::get_scene_params().implicit_dt, 0.0001f, 0.2f); 
                ImGui::Checkbox("Use Bending", &lcsv::get_scene_params().use_bending);
                ImGui::Checkbox("Use Quadratic Bending", &lcsv::get_scene_params().use_quadratic_bending_model);
                ImGui::SliderFloat("Bending Stiffness", &lcsv::get_scene_params().stiffness_bending_ui, 0.0f, 1.0f); 
                // ImGui::Checkbox("Print Convergence", &lcsv::get_scene_params().print_xpbd_convergence);
                ImGui::Checkbox("Print Energy", &lcsv::get_scene_params().print_system_energy);
                // ImGui::Checkbox("Print PCG Convergence", &lcsv::get_scene_params().print_pcg_convergence);

                // static const char* items[] = { "A", "B", "C" };
                // static int current_item = 0;
                // ImGui::Combo("Combo", &current_item, items, IM_ARRAYSIZE(items));

                // ImGui::InputDouble("Thickness", &thickness);
                // ImGui::InputDouble("Poisson's Ration", &poisson);
                // ImGui::Combo("Material Model", &matid, "NeoHookean\0StVK\0\0");
                // ImGui::Combo("Second Fundamental Form", &sffid,
                //             "TanTheta\0SinTheta\0Average\0\0");
            }

            if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen)) 
            {
                ImGui::Text("Frame %d", lcsv::get_scene_params().current_frame);
                if (ImGui::Button("Reset", ImVec2(-1, 0))) 
                {
                    lcsv::get_scene_params().current_frame = 0;
                    solver.lcsv::SolverInterface::restart_system();
                    fn_update_rendering_vertices();
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
                static uint state_frame = 8;
                ImGui::InputScalar("Load State Frame", ImGuiDataType_U32, &state_frame);
                if (ImGui::Button("Load State", ImVec2(-1, 0)))
                {
                    solver.lcsv::SolverInterface::load_saved_state_from_host(state_frame, "");
                    fn_update_rendering_vertices();
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