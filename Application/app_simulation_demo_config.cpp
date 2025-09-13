#include "app_simulation_demo_config.h"
#include "Core/constant_value.h"
#include "Core/float_n.h"
#include "SimulationCore/scene_params.h"
#include "luisa/core/basic_types.h"

namespace Demo::Simulation
{

const std::string obj_mesh_path = std::string(LCSV_RESOURCE_PATH) + "/InputMesh/";
const std::string tet_mesh_path = std::string(LCSV_RESOURCE_PATH) + "/InputMesh/vtks/";

void ccd_vf_unit_case(std::vector<lcs::Initializer::ShellInfo>& shell_list)
{
    shell_list.push_back({
        .model_name = obj_mesh_path + "square2.obj",
        .fixed_point_list = {
            lcs::Initializer::FixedPointInfo{
                .is_fixed_point_func = [](const luisa::float3& norm_pos) { return norm_pos.z > 0.999f  && norm_pos.x < 0.001f; },
            },
        }
    });
    shell_list.push_back({
        .model_name = obj_mesh_path + "square2.obj",
        .transform = luisa::make_float3(0.1, -0.3, 0),
        .fixed_point_list = {
            lcs::Initializer::FixedPointInfo{
                .is_fixed_point_func = [](const luisa::float3& norm_pos) { return norm_pos.x < 0.001f || norm_pos.x > 0.999; },
            },
        }
    });

    
    lcs::get_scene_params().use_floor = false;
    lcs::get_scene_params().load_state_frame = 10;
    lcs::get_scene_params().implicit_dt = 0.05;;
    lcs::get_scene_params().num_substep = 1;
    lcs::get_scene_params().nonlinear_iter_count = 1;
    lcs::get_scene_params().pcg_iter_count = 200;
    lcs::get_scene_params().use_ccd_linesearch = true;
    lcs::get_scene_params().use_energy_linesearch = true;
}
void moving_vf_unit(std::vector<lcs::Initializer::ShellInfo>& shell_list)
{
    shell_list.push_back({
        .model_name = obj_mesh_path + "square2.obj",
    });
    shell_list.push_back({
        .model_name = obj_mesh_path + "square2.obj",
        .transform = luisa::make_float3(0.1, -0.3, 0),
        .fixed_point_list = {
            lcs::Initializer::FixedPointInfo{
                .is_fixed_point_func = [](const luisa::float3& norm_pos) { return norm_pos.x < 0.001f || norm_pos.x > 0.999; },
                .use_translate = true,
                .translate = luisa::make_float3(0, 1, 0),
            },
        }
    });
    lcs::get_scene_params().use_floor = false;
    lcs::get_scene_params().implicit_dt = 0.05;;
    lcs::get_scene_params().nonlinear_iter_count = 1;
    // lcs::get_scene_params().use_ccd_linesearch = true;
    lcs::get_scene_params().use_energy_linesearch = true;
    lcs::get_scene_params().pcg_iter_count = 200;
}
void ccd_ee_unit_case(std::vector<lcs::Initializer::ShellInfo>& shell_list)
{
    
    shell_list.push_back({
        // .model_name = obj_mesh_path + "square8K.obj",
        .model_name = obj_mesh_path + "square2.obj",
        .fixed_point_list = {
            lcs::Initializer::FixedPointInfo{
                .is_fixed_point_func = [](const luisa::float3& norm_pos) { return norm_pos.z < 0.001; },
            },
        }
    });
    shell_list.push_back({
        // .model_name = obj_mesh_path + "Cylinder/cylinder7K.obj",
        .model_name = obj_mesh_path + "square2.obj",
        .transform = luisa::make_float3(0.1, -0.3, 0),
        .fixed_point_list = {
            lcs::Initializer::FixedPointInfo{
                .is_fixed_point_func = [](const luisa::float3& norm_pos) { return norm_pos.x < 0.001f || norm_pos.x > 0.999; },
            },
        }
    });
    lcs::get_scene_params().load_state_frame = 29;
    lcs::get_scene_params().implicit_dt = 0.05;
    lcs::get_scene_params().num_substep = 1;
    lcs::get_scene_params().nonlinear_iter_count = 1;
    lcs::get_scene_params().pcg_iter_count = 200;
}
void dcd_cloth_cylinder_repulsion(std::vector<lcs::Initializer::ShellInfo>& shell_list)
{
    shell_list.push_back({
        .model_name = obj_mesh_path + "square8K.obj",
        .fixed_point_list = {
            lcs::Initializer::FixedPointInfo{
                .is_fixed_point_func = [](const luisa::float3& norm_pos) { return norm_pos.z > 0.999f  && norm_pos.x < 0.001f; },
            },
        }
    });
    shell_list.push_back({
        .model_name = obj_mesh_path + "Cylinder/cylinder7K.obj",
        .transform = luisa::make_float3(0.1, -0.3, 0),
        .fixed_point_list = {
            lcs::Initializer::FixedPointInfo{
                .is_fixed_point_func = [](const luisa::float3& norm_pos) { return norm_pos.x < 0.001f || norm_pos.x > 0.999; },
            },
        }
    });
    // lcs::get_scene_params().load_state_frame = 4;

    lcs::get_scene_params().implicit_dt = 0.003f;
    lcs::get_scene_params().num_substep = 1;
    lcs::get_scene_params().nonlinear_iter_count = 1;
    lcs::get_scene_params().pcg_iter_count = 2000;
}
void dcd_cloth_ball(std::vector<lcs::Initializer::ShellInfo>& shell_list)
{
    shell_list.push_back({
        .model_name = obj_mesh_path + "square2K.obj",
        .transform = luisa::make_float3(0.0, 0.1, 0),
        .scale = luisa::make_float3(0.2f),
        .fixed_point_list = {
            // lcs::Initializer::FixedPointInfo{
            //     .is_fixed_point_func = [](const luisa::float3& norm_pos) { return norm_pos.z > 0.999f  && norm_pos.x < 0.001f; },
            // },
        }
    });
    shell_list.push_back({
        .model_name = obj_mesh_path + "square2K.obj",
        .transform = luisa::make_float3(0.0, 0.12, 0),
        .rotation = luisa::make_float3(0, lcs::Pi / 6.0f, 0),
        .scale = luisa::make_float3(0.2f),
        .fixed_point_list = {
        }
    });
    shell_list.push_back({
        .model_name = obj_mesh_path + "square2K.obj",
        .transform = luisa::make_float3(0.0, 0.14, 0),
        .rotation = luisa::make_float3(0, lcs::Pi / 6.0f * 2, 0),
        .scale = luisa::make_float3(0.2f),
        .fixed_point_list = {
        }
    });
    // shell_list.push_back({
    //     .model_name = obj_mesh_path + "sphere1K.obj",
    //     .transform = luisa::make_float3(0.0, 0.02, 0),
    //     .scale = luisa::make_float3(0.1f),
    //     .fixed_point_list = {
    //         lcs::Initializer::FixedPointInfo{
    //             .is_fixed_point_func = [](const luisa::float3& norm_pos) { return true; },
    //         },
    //     }
    // });
    shell_list.push_back({
        .model_name = obj_mesh_path + "bowl/bowl.obj",
        .transform = luisa::make_float3(0.0, 0.02, 0),
        .scale = luisa::make_float3(0.3f),
        .fixed_point_list = {
            lcs::Initializer::FixedPointInfo{
                .is_fixed_point_func = [](const luisa::float3& norm_pos) { return true; },
            },
        }
    });
    // lcs::get_scene_params().load_state_frame = 4;

    lcs::get_scene_params().d_hat = 1e-3f;
    lcs::get_scene_params().thickness = 0.0f;
    lcs::get_scene_params().implicit_dt = 0.01f;
    lcs::get_scene_params().nonlinear_iter_count = 5;
    lcs::get_scene_params().pcg_iter_count = 200;
}
void cloth_bottle4(std::vector<lcs::Initializer::ShellInfo>& shell_list)
{
    shell_list.push_back({
        .model_name = obj_mesh_path + "square2K.obj",
        .transform = luisa::make_float3(0.0, 0.1, 0),
        .scale = luisa::make_float3(0.2f),
        .fixed_point_list = {
            // lcs::Initializer::FixedPointInfo{
            //     .is_fixed_point_func = [](const luisa::float3& norm_pos) { return norm_pos.z > 0.999f  && norm_pos.x < 0.001f; },
            // },
        }
    });
    shell_list.push_back({
        .model_name = obj_mesh_path + "square2K.obj",
        .transform = luisa::make_float3(0.0, 0.12, 0),
        .rotation = luisa::make_float3(0, lcs::Pi / 6.0f, 0),
        .scale = luisa::make_float3(0.2f),
        .fixed_point_list = {
        }
    });
    shell_list.push_back({
        .model_name = obj_mesh_path + "square2K.obj",
        .transform = luisa::make_float3(0.0, 0.14, 0),
        .rotation = luisa::make_float3(0, lcs::Pi / 6.0f * 2, 0),
        .scale = luisa::make_float3(0.2f),
        .fixed_point_list = {
        }
    });
    shell_list.push_back({
        .model_name = obj_mesh_path + "bowl/bottle4.obj",
        .transform = luisa::make_float3(0.0, -0.4, 0),
        .scale = luisa::make_float3(0.6f),
        .fixed_point_list = {
            lcs::Initializer::FixedPointInfo{
                .is_fixed_point_func = [](const luisa::float3& norm_pos) { return true; },
            },
        }
    });
    // lcs::get_scene_params().load_state_frame = 4;

    lcs::get_scene_params().d_hat = 1e-3f;
    lcs::get_scene_params().thickness = 0.0f;
    lcs::get_scene_params().implicit_dt = 0.01f;
    lcs::get_scene_params().nonlinear_iter_count = 5;
    lcs::get_scene_params().pcg_iter_count = 200;
    lcs::get_scene_params().use_floor = false;
}
void ccd_rotation_cylinder(std::vector<lcs::Initializer::ShellInfo>& shell_list)
{
    shell_list.push_back({
        .model_name = obj_mesh_path + "Cylinder/cylinder7K.obj",
        .fixed_point_list = {
            lcs::Initializer::FixedPointInfo{
                .is_fixed_point_func = [](const luisa::float3& norm_pos) { return (norm_pos.x < 0.001f ); },
                .use_rotate = true,
                .rotCenter = luisa::make_float3(0.005, 0, 0),
                .rotAxis = luisa::make_float3(1, 0, 0),
                .rotAngVelDeg = -72, 
            },
            lcs::Initializer::FixedPointInfo{
                .is_fixed_point_func = [](const luisa::float3& norm_pos) { return (norm_pos.x > 0.999f); },
                .use_rotate = true,
                .rotCenter = luisa::make_float3(-0.005, 0, 0),
                .rotAxis = luisa::make_float3(1, 0, 0),
                .rotAngVelDeg = 72, 
            }
        }
    });
    lcs::get_scene_params().pcg_iter_count = 500;;
    lcs::get_scene_params().nonlinear_iter_count = 6;
    lcs::get_scene_params().use_ccd_linesearch = true;
    lcs::get_scene_params().use_energy_linesearch = true;
    lcs::get_scene_params().gravity = luisa::make_float3(0.0f);
    lcs::get_scene_params().use_floor = false;
}
void load_scene(std::vector<lcs::Initializer::ShellInfo>& shell_list)
{
    const uint case_number = 
        0
    ;

    switch (case_number)
    {
        case 0: { ccd_vf_unit_case(shell_list); break; };
        case 1: { ccd_ee_unit_case(shell_list); break; };
        case 2: { moving_vf_unit(shell_list); break; };
        case 3: { ccd_rotation_cylinder(shell_list); break; };
        case 8: { dcd_cloth_ball(shell_list); break; };
        case 9: { cloth_bottle4(shell_list); break; };

        default: ccd_vf_unit_case(shell_list); break;
    };

    
}



}