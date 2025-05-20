#include <iostream>
#include <luisa/luisa-compute.h>

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
    lcsv::get_scene_params().print_xpbd_convergence = false; // false true

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
    const std::string binary_path(argv[0]);
    luisa::compute::Context context{ argv[0] };
    luisa::compute::Device device = context.create_device(backend);
    luisa::compute::Stream stream = device.create_stream(luisa::compute::StreamTag::COMPUTE);

    // Init data
    lcsv::MeshData<std::vector> cpu_mesh_data;
    lcsv::MeshData<luisa::compute::Buffer> mesh_data;
    {
        lcsv::Initializater::init_mesh_data(&cpu_mesh_data);
        lcsv::Initializater::upload_mesh_buffers(device, stream, &cpu_mesh_data, &mesh_data);
    }

    lcsv::XpbdData<std::vector> cpu_xpbd_data;
    lcsv::XpbdData<luisa::compute::Buffer> xpbd_data;
    {
        lcsv::Initializater::init_xpbd_data(&cpu_mesh_data, &cpu_xpbd_data);
        lcsv::Initializater::upload_xpbd_buffers(device, stream, &cpu_xpbd_data, &xpbd_data);
        lcsv::Initializater::init_simulation_params();
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

    solver.test_luisa();

    // Some params
    {
        lcsv::get_scene_params().use_substep = false;
        lcsv::get_scene_params().num_substep = 10;
        lcsv::get_scene_params().constraint_iter_count = 1; // 
        lcsv::get_scene_params().use_bending = true;
        lcsv::get_scene_params().use_quadratic_bending_model = true;
        lcsv::get_scene_params().print_xpbd_convergence = false;
        lcsv::get_scene_params().use_xpbd_solver = false;
        lcsv::get_scene_params().use_vbd_solver = true;
    }

    // Init GUI
    std::vector<glm::vec3> sa_rendering_vertices(cpu_mesh_data.num_verts);
    std::vector<std::vector<uint>> sa_rendering_faces(cpu_mesh_data.num_faces);
    auto fn_update_rendering_vertices = [&]()
    {
        CpuParallel::parallel_for(0, cpu_mesh_data.num_verts, [&](const uint vid)
        {
            auto pos = cpu_mesh_data.sa_x_frame_end[vid];
            sa_rendering_vertices[vid] = glm::vec3(pos.x, pos.y, pos.z);
        });
    };
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

    constexpr bool use_ui = true; polyscope::SurfaceMesh* surface_mesh;
    if constexpr (use_ui) 
    {
        polyscope::init("openGL3_glfw");
        polyscope::registerSurfaceMesh("cloth1", sa_rendering_vertices, sa_rendering_faces);
        surface_mesh = polyscope::getSurfaceMesh("cloth1"); // surface_mesh->setEnabled(false);
        polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;
        polyscope::screenshot();
    }
    

    // Define Simulation
    {
        solver.lcsv::SolverInterface::restart_system();
        solver.lcsv::SolverInterface::save_mesh_to_obj(0, "_init"); 
        luisa::log_info("");
        luisa::log_info("");
    }
    auto fn_physics_step = [&]()
    {
        solver.physics_step_vbd_GPU(device, stream);
    };
    auto fn_single_step = [&]()
    {
        lcsv::get_scene_params().current_frame += 1; 
        luisa::log_info("     Sync frame {}", lcsv::get_scene_params().current_frame);   

        fn_physics_step();

        fn_update_rendering_vertices();
        surface_mesh->updateVertexPositions(sa_rendering_vertices);
    };
    auto fn_step_several_step = [&](const uint frames)
    {
        for (uint frame = 0; frame < frames; frame++)
        {   
            lcsv::get_scene_params().current_frame = frame; 
            luisa::log_info("     Sync frame {}", frame);   

            fn_physics_step();

            fn_update_rendering_vertices();
            surface_mesh->updateVertexPositions(sa_rendering_vertices);
        }
        solver.lcsv::SolverInterface::save_mesh_to_obj(lcsv::get_scene_params().current_frame, ""); 
    };
    const uint num_frames = 20;

    
    polyscope::state::userCallback = [&]()
    {
        if (ImGui::Button("Reset", ImVec2(-1, 0))) 
        {
            solver.lcsv::SolverInterface::restart_system();
            fn_update_rendering_vertices();
            surface_mesh->updateVertexPositions(sa_rendering_vertices);
        }

        if (ImGui::CollapsingHeader("Parameters", ImGuiTreeNodeFlags_DefaultOpen)) 
        {
            ImGui::InputScalar("Num Substep", ImGuiDataType_U32, &lcsv::get_scene_params().num_substep);
            ImGui::InputScalar("Num Iteration", ImGuiDataType_U32, &lcsv::get_scene_params().constraint_iter_count);
            ImGui::Checkbox("Use Bending", &lcsv::get_scene_params().use_bending);
            ImGui::Checkbox("Use Quadratic Bending", &lcsv::get_scene_params().use_quadratic_bending_model);
            ImGui::SliderFloat("Bending Stiffness", &lcsv::get_scene_params().stiffness_bending_ui, 0, 100); 
            ImGui::Checkbox("Print Convergence", &lcsv::get_scene_params().print_xpbd_convergence);

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
            if (ImGui::Button("Optimize Single Step", ImVec2(-1, 0)))
            {
                fn_single_step();
            }
            if (ImGui::Button("Optimize Some Step", ImVec2(-1, 0)))
            {
                fn_step_several_step(num_frames);
            }
        }
    };
    if constexpr (use_ui) polyscope::show();

    return 0;
}