#include "app_simulation_demo_config.h"
#include "SimulationCore/scene_params.h"

namespace Demo::Simulation
{

const std::string obj_mesh_path = std::string(LCSV_RESOURCE_PATH) + "/InputMesh/";
const std::string tet_mesh_path = std::string(LCSV_RESOURCE_PATH) + "/InputMesh/vtks/";

void ccd_vf_unit_case(std::vector<lcsv::Initializer::ShellInfo>& shell_list)
{
    shell_list.push_back({
        .model_name = obj_mesh_path + "square2.obj",
        .fixed_point_list = {
            lcsv::Initializer::FixedPointInfo{
                .is_fixed_point_func = [](const luisa::float3& norm_pos) { return norm_pos.z > 0.999f  && norm_pos.x < 0.001f; },
            },
        }
    });
    shell_list.push_back({
        .model_name = obj_mesh_path + "square2.obj",
        .transform = luisa::make_float3(0.1, -0.3, 0),
        .fixed_point_list = {
            lcsv::Initializer::FixedPointInfo{
                .is_fixed_point_func = [](const luisa::float3& norm_pos) { return norm_pos.x < 0.001f || norm_pos.x > 0.999; },
            },
        }
    });
    lcsv::get_scene_params().load_state_frame = 6;
    lcsv::get_scene_params().implicit_dt = 0.05;;
    lcsv::get_scene_params().num_substep = 1;
    lcsv::get_scene_params().nonlinear_iter_count = 1;
    lcsv::get_scene_params().pcg_iter_count = 200;
}
void ccd_ee_unit_case(std::vector<lcsv::Initializer::ShellInfo>& shell_list)
{
    
    shell_list.push_back({
        // .model_name = obj_mesh_path + "square8K.obj",
        .model_name = obj_mesh_path + "square2.obj",
        .fixed_point_list = {
            lcsv::Initializer::FixedPointInfo{
                .is_fixed_point_func = [](const luisa::float3& norm_pos) { return norm_pos.z < 0.001; },
            },
        }
    });
    shell_list.push_back({
        // .model_name = obj_mesh_path + "Cylinder/cylinder7K.obj",
        .model_name = obj_mesh_path + "square2.obj",
        .transform = luisa::make_float3(0.1, -0.3, 0),
        .fixed_point_list = {
            lcsv::Initializer::FixedPointInfo{
                .is_fixed_point_func = [](const luisa::float3& norm_pos) { return norm_pos.x < 0.001f || norm_pos.x > 0.999; },
            },
        }
    });
    lcsv::get_scene_params().load_state_frame = 4;
    lcsv::get_scene_params().implicit_dt = 0.05;
    lcsv::get_scene_params().num_substep = 1;
    lcsv::get_scene_params().nonlinear_iter_count = 1;
    lcsv::get_scene_params().pcg_iter_count = 200;
}
void dcd_proximity_repulsion_unit_case(std::vector<lcsv::Initializer::ShellInfo>& shell_list)
{
    
    shell_list.push_back({
        // .model_name = obj_mesh_path + "square8K.obj",
        .model_name = obj_mesh_path + "square2.obj",
        .fixed_point_list = {
            lcsv::Initializer::FixedPointInfo{
                .is_fixed_point_func = [](const luisa::float3& norm_pos) { return norm_pos.z < 0.001; },
            },
        }
    });
    shell_list.push_back({
        // .model_name = obj_mesh_path + "Cylinder/cylinder7K.obj",
        .model_name = obj_mesh_path + "square2.obj",
        .transform = luisa::make_float3(0.1, -0.3, 0),
        .fixed_point_list = {
            lcsv::Initializer::FixedPointInfo{
                .is_fixed_point_func = [](const luisa::float3& norm_pos) { return norm_pos.x < 0.001f || norm_pos.x > 0.999; },
            },
        }
    });
    lcsv::get_scene_params().load_state_frame = 32;
    lcsv::get_scene_params().implicit_dt = 1.0/240.0;
    lcsv::get_scene_params().num_substep = 1;
    lcsv::get_scene_params().nonlinear_iter_count = 1;
    lcsv::get_scene_params().pcg_iter_count = 200;
}
void ccd_ipc_unit_case(std::vector<lcsv::Initializer::ShellInfo>& shell_list)
{
    
    shell_list.push_back({
        // .model_name = obj_mesh_path + "square8K.obj",
        .model_name = obj_mesh_path + "square2.obj",
        .fixed_point_list = {
            lcsv::Initializer::FixedPointInfo{
                .is_fixed_point_func = [](const luisa::float3& norm_pos) { return norm_pos.z < 0.001; },
            },
        }
    });
    shell_list.push_back({
        // .model_name = obj_mesh_path + "Cylinder/cylinder7K.obj",
        .model_name = obj_mesh_path + "square2.obj",
        .transform = luisa::make_float3(0.1, -0.3, 0),
        .fixed_point_list = {
            lcsv::Initializer::FixedPointInfo{
                .is_fixed_point_func = [](const luisa::float3& norm_pos) { return norm_pos.x < 0.001f || norm_pos.x > 0.999; },
            },
        }
    });
    lcsv::get_scene_params().load_state_frame = 6;
    lcsv::get_scene_params().implicit_dt = 0.05;
    lcsv::get_scene_params().num_substep = 1;
    lcsv::get_scene_params().nonlinear_iter_count = 20;
    lcsv::get_scene_params().pcg_iter_count = 1000;
}
void ccd_cloth_cylinder(std::vector<lcsv::Initializer::ShellInfo>& shell_list)
{
    shell_list.push_back({
        .model_name = obj_mesh_path + "square8K.obj",
        .fixed_point_list = {
            lcsv::Initializer::FixedPointInfo{
                .is_fixed_point_func = [](const luisa::float3& norm_pos) { return norm_pos.z > 0.999f  && norm_pos.x < 0.001f; },
            },
        }
    });
    shell_list.push_back({
        .model_name = obj_mesh_path + "Cylinder/cylinder7K.obj",
        .transform = luisa::make_float3(0.1, -0.3, 0),
        .fixed_point_list = {
            lcsv::Initializer::FixedPointInfo{
                .is_fixed_point_func = [](const luisa::float3& norm_pos) { return norm_pos.x < 0.001f || norm_pos.x > 0.999; },
            },
        }
    });
    lcsv::get_scene_params().load_state_frame = 4;

    // lcsv::get_scene_params().implicit_dt = 1.0/500.0;
    lcsv::get_scene_params().num_substep = 1;
    lcsv::get_scene_params().nonlinear_iter_count = 20;
    lcsv::get_scene_params().pcg_iter_count = 2000;
}
void ccd_rotation_cylinder(std::vector<lcsv::Initializer::ShellInfo>& shell_list)
{
    shell_list.push_back({
        .model_name = obj_mesh_path + "Cylinder/cylinder7K.obj",
        .fixed_point_list = {
            lcsv::Initializer::FixedPointInfo{
                .is_fixed_point_func = [](const luisa::float3& norm_pos) { return (norm_pos.x < 0.001f ); },
                .use_rotate = true,
                .rotCenter = luisa::make_float3(0.005, 0, 0),
                .rotAxis = luisa::make_float3(1, 0, 0),
                .rotAngVelDeg = -72, 
            },
            lcsv::Initializer::FixedPointInfo{
                .is_fixed_point_func = [](const luisa::float3& norm_pos) { return (norm_pos.x > 0.999f); },
                .use_rotate = true,
                .rotCenter = luisa::make_float3(-0.005, 0, 0),
                .rotAxis = luisa::make_float3(1, 0, 0),
                .rotAngVelDeg = 72, 
            }
        }
    });
}
void load_scene(std::vector<lcsv::Initializer::ShellInfo>& shell_list)
{
    const uint case_number = 
        2
    ;

    switch (case_number)
    {
        case 0: { ccd_vf_unit_case(shell_list); break; };
        case 1: { ccd_ee_unit_case(shell_list); break; };
        case 2: { ccd_cloth_cylinder(shell_list); break; };
        case 3: { ccd_rotation_cylinder(shell_list); break; };
        case 4: { load_scene(shell_list); break; };
        case 5: { ccd_ipc_unit_case(shell_list); break; };
        case 6: { dcd_proximity_repulsion_unit_case(shell_list); break; };
        default: ccd_vf_unit_case(shell_list); break;
    };

    
}



}