#include "app_simulation_demo_config.h"
#include "CollisionDetector/narrow_phase.h"
#include "Core/constant_value.h"
#include "Core/float_n.h"
#include "MeshOperation/mesh_reader.h"
#include "SimulationCore/scene_params.h"
#include "luisa/core/basic_types.h"
#include <fstream>
#include <sstream>
#include <string>
#include <yyjson.h>

namespace Demo::Simulation
{

const std::string obj_mesh_path = std::string(LCSV_RESOURCE_PATH) + "/InputMesh/";
const std::string tet_mesh_path = std::string(LCSV_RESOURCE_PATH) + "/InputMesh/vtks/";

using namespace lcs::Initializer;

void ccd_vf_unit_case(std::vector<ShellInfo>& shell_list)
{
    auto& up = shell_list
                   .emplace_back(ShellInfo{
                       .model_name = obj_mesh_path + "square2.obj",
                   })
                   .load_mesh_data();
    up.set_pinned_verts_from_norm_position([](const luisa::float3& norm_pos)
                                           { return norm_pos.z > 0.999f && norm_pos.x < 0.001f; });

    auto& down = shell_list
                     .emplace_back(ShellInfo{
                         .model_name  = obj_mesh_path + "square2.obj",
                         .translation = luisa::make_float3(0.1, -0.3, 0),
                     })
                     .load_mesh_data();
    down.set_pinned_verts_from_norm_position([](const luisa::float3& norm_pos)
                                             { return norm_pos.x < 0.001f || norm_pos.x > 0.999; });

    lcs::get_scene_params().use_floor             = false;
    lcs::get_scene_params().implicit_dt           = 0.2;
    lcs::get_scene_params().use_energy_linesearch = true;
    lcs::get_scene_params().stiffness_DAB_bending = 200.0f;
    lcs::get_scene_params().use_gpu               = false;
}
void rigid_body_cube_unit_case(std::vector<ShellInfo>& shell_list)
{
    shell_list.emplace_back(ShellInfo{// .model_name = obj_mesh_path + "square2.obj",
                                      .model_name = obj_mesh_path + "cube.obj",
                                      //   .translation = luisa::make_float3(0, -0.1, 0),
                                      .rotation = luisa::make_float3(lcs::Pi / 6, 0, lcs::Pi / 6),
                                      //   .rotation = luisa::make_float3(0.0, 0, 0.0),
                                      .scale      = luisa::make_float3(0.1f),
                                      .shell_type = ShellTypeRigid});
    shell_list.emplace_back(ShellInfo{// .model_name = obj_mesh_path + "square2.obj",
                                      .model_name  = obj_mesh_path + "cube.obj",
                                      .translation = luisa::make_float3(0.6, 0.4, 0),
                                      .rotation    = luisa::make_float3(lcs::Pi / 6, 0, lcs::Pi / 3),
                                      .scale       = luisa::make_float3(0.2),
                                      .shell_type  = ShellTypeRigid});
    shell_list.emplace_back(ShellInfo{// .model_name = obj_mesh_path + "square2.obj",
                                      .model_name  = obj_mesh_path + "cube.obj",
                                      .translation = luisa::make_float3(0.9, 0.6, 0),
                                      .rotation    = luisa::make_float3(lcs::Pi / 3, lcs::Pi / 6, 0),
                                      .scale       = luisa::make_float3(0.5),
                                      .shell_type  = ShellTypeRigid});

    for (auto& shell : shell_list)
    {
        shell.load_mesh_data();
    }
    // lcs::get_scene_params().use_gpu     = false;
    lcs::get_scene_params().nonlinear_iter_count = 2;
    lcs::get_scene_params().use_self_collision   = false;
    lcs::get_scene_params().use_ccd_linesearch   = false;
}
void rigid_body_folding_cube_case(std::vector<ShellInfo>& shell_list)
{
    shell_list.emplace_back(ShellInfo{.model_name  = obj_mesh_path + "cube.obj",
                                      .translation = luisa::make_float3(0, 1.0, 0),
                                      .scale       = luisa::make_float3(0.1),
                                      .shell_type  = ShellTypeRigid});
    shell_list.emplace_back(ShellInfo{.model_name  = obj_mesh_path + "cube.obj",
                                      .translation = luisa::make_float3(0, 0.7, 0),
                                      //   .translation = luisa::make_float3(0.1, 0.511, 0.2),
                                      //   .rotation    = luisa::make_float3(lcs::Pi / 6, 0, lcs::Pi / 6),
                                      .scale      = luisa::make_float3(0.2),
                                      .shell_type = ShellTypeRigid});
    shell_list.emplace_back(ShellInfo{.model_name  = obj_mesh_path + "cube.obj",
                                      .translation = luisa::make_float3(0, 0.1, 0),
                                      .scale       = luisa::make_float3(0.5),
                                      .shell_type  = ShellTypeRigid});
    for (auto& shell : shell_list)
    {
        shell.load_mesh_data();
    }
    lcs::get_scene_params().use_floor = true;
    // lcs::get_scene_params().implicit_dt          = 0.003;
    // lcs::get_scene_params().nonlinear_iter_count = 1;
}
void moving_dirichlet_unit(std::vector<ShellInfo>& shell_list)
{
    shell_list
        .emplace_back(ShellInfo{
            .model_name = obj_mesh_path + "square2.obj",
        })
        .load_mesh_data();

    auto& down = shell_list
                     .emplace_back(ShellInfo{
                         .model_name  = obj_mesh_path + "square2.obj",
                         .translation = luisa::make_float3(0.1, -0.3, 0),
                     })
                     .load_mesh_data();

    down.set_pinned_verts_from_norm_position([](const luisa::float3& norm_pos)
                                             { return norm_pos.x < 0.001f || norm_pos.x > 0.999; },
                                             FixedPointInfo{
                                                 .use_translate = true,
                                                 .translate     = luisa::make_float3(0, 1, 0),
                                             });

    lcs::get_scene_params().use_floor             = false;
    lcs::get_scene_params().implicit_dt           = 0.05;
    lcs::get_scene_params().nonlinear_iter_count  = 1;
    lcs::get_scene_params().use_energy_linesearch = true;
}
void dcd_cloth_cylinder_repulsion(std::vector<ShellInfo>& shell_list)
{
    auto& up = shell_list
                   .emplace_back(ShellInfo{
                       .model_name = obj_mesh_path + "square8K.obj",
                   })
                   .load_mesh_data();
    up.set_pinned_verts_from_norm_position([](const luisa::float3& norm_pos)
                                           { return norm_pos.z > 0.999f && norm_pos.x < 0.001f; });

    auto& down = shell_list
                     .emplace_back(ShellInfo{
                         .model_name = obj_mesh_path + "Cylinder/cylinder7K.obj",
                     })
                     .load_mesh_data();
    down.set_pinned_verts_from_norm_position([](const luisa::float3& norm_pos)
                                             { return norm_pos.x < 0.001f || norm_pos.x > 0.999; });


    lcs::get_scene_params().implicit_dt          = 0.003f;
    lcs::get_scene_params().nonlinear_iter_count = 1;
}
void cloth_ball(std::vector<ShellInfo>& shell_list)
{
    shell_list
        .emplace_back(ShellInfo{.model_name  = obj_mesh_path + "square26K.obj",
                                .translation = luisa::make_float3(0.0, 0.22, 0),
                                .rotation    = luisa::make_float3(0, lcs::Pi / 6.0f, 0),
                                .scale       = luisa::make_float3(1.0f)})
        .load_mesh_data();
    // shell_list.push_back({.model_name       = obj_mesh_path + "square26K.obj",
    //                       .translation      = luisa::make_float3(0.0, 0.24, 0),
    //                       .rotation         = luisa::make_float3(0, lcs::Pi / 6.0f * 2, 0),
    //                       .scale            = luisa::make_float3(0.2f),
    //                       .fixed_point_list = {}});
    // shell_list.push_back(
    //     {.model_name       = obj_mesh_path + "sphere1K.obj",
    //      .translation      = luisa::make_float3(0.0, 0.15, 0),
    //      .scale            = luisa::make_float3(0.1f),
    //      .fixed_point_list = {
    //          FixedPointInfo{
    //              .is_fixed_point_func = [](const luisa::float3& norm_pos) { return true; },
    //          },
    //      }});
    // shell_list.push_back(
    //     {.model_name       = obj_mesh_path + "bowl/bowl.obj",
    //      .translation      = luisa::make_float3(0.0, 0.02, 0),
    //      .scale            = luisa::make_float3(0.3f),
    //      .fixed_point_list = {
    //          FixedPointInfo{
    //              .is_fixed_point_func = [](const luisa::float3& norm_pos) { return true; },
    //          },
    //      }});

    lcs::get_scene_params().implicit_dt          = 0.01f;
    lcs::get_scene_params().nonlinear_iter_count = 1;
    lcs::get_scene_params().pcg_iter_count       = 50;
}
void cloth_bottle4(std::vector<ShellInfo>& shell_list)
{
    shell_list.emplace_back(ShellInfo{
        .model_name  = obj_mesh_path + "square2K.obj",
        .translation = luisa::make_float3(0.0, 0.1, 0),
        .scale       = luisa::make_float3(0.2f),
    });
    shell_list.emplace_back(ShellInfo{
        .model_name  = obj_mesh_path + "square2K.obj",
        .translation = luisa::make_float3(0.0, 0.12, 0),
        .rotation    = luisa::make_float3(0, lcs::Pi / 6.0f, 0),
        .scale       = luisa::make_float3(0.2f),
    });
    shell_list.emplace_back(ShellInfo{
        .model_name  = obj_mesh_path + "square2K.obj",
        .translation = luisa::make_float3(0.0, 0.14, 0),
        .rotation    = luisa::make_float3(0, lcs::Pi / 6.0f * 2, 0),
        .scale       = luisa::make_float3(0.2f),
    });
    for (auto& shell : shell_list)
    {
        shell.load_mesh_data();
    }

    auto& bottle = shell_list
                       .emplace_back(ShellInfo{
                           .model_name  = obj_mesh_path + "bowl/bottle4.obj",
                           .translation = luisa::make_float3(0.0, -0.4, 0),
                           .scale       = luisa::make_float3(0.6f),
                       })
                       .load_mesh_data();
    bottle.set_pinned_verts_from_norm_position([](const luisa::float3& norm_pos) { return true; });

    lcs::get_scene_params().implicit_dt          = 0.01f;
    lcs::get_scene_params().nonlinear_iter_count = 5;
    lcs::get_scene_params().pcg_iter_count       = 200;
    lcs::get_scene_params().use_floor            = false;
}
void ccd_rotation_cylinder_7K_quadratic(std::vector<ShellInfo>& shell_list)
{
    auto& shell = shell_list
                      .emplace_back(ShellInfo{
                          .model_name = obj_mesh_path + "Cylinder/cylinder7K.obj",
                      })
                      .load_mesh_data();

    shell.set_pinned_verts_from_norm_position([](const luisa::float3& norm_pos)
                                              { return (norm_pos.x < 0.001f); },
                                              {
                                                  .use_rotate   = true,
                                                  .rotCenter    = luisa::make_float3(0.005, 0, 0),
                                                  .rotAxis      = luisa::make_float3(1, 0, 0),
                                                  .rotAngVelDeg = -72,
                                              });
    shell.set_pinned_verts_from_norm_position([](const luisa::float3& norm_pos)
                                              { return (norm_pos.x > 0.999f); },
                                              {
                                                  .use_rotate   = true,
                                                  .rotCenter    = luisa::make_float3(-0.005, 0, 0),
                                                  .rotAxis      = luisa::make_float3(1, 0, 0),
                                                  .rotAngVelDeg = 72,
                                              });

    // lcs::get_scene_params().use_ccd_linesearch    = false;
    lcs::get_scene_params().pcg_iter_count       = 50;
    lcs::get_scene_params().nonlinear_iter_count = 1;
    lcs::get_scene_params().gravity              = luisa::make_float3(0.0f);
    lcs::get_scene_params().use_floor            = false;
    lcs::get_scene_params().contact_energy_type  = uint(lcs::ContactEnergyType::Barrier);
}
void ccd_rotation_cylinder_7K_ipc(std::vector<ShellInfo>& shell_list)
{
    auto& shell = shell_list
                      .emplace_back(ShellInfo{
                          .model_name = obj_mesh_path + "Cylinder/cylinder7K.obj",
                      })
                      .load_mesh_data();

    shell.set_pinned_verts_from_norm_position([](const luisa::float3& norm_pos)
                                              { return (norm_pos.x < 0.001f); },
                                              {
                                                  .use_rotate   = true,
                                                  .rotCenter    = luisa::make_float3(0.005, 0, 0),
                                                  .rotAxis      = luisa::make_float3(1, 0, 0),
                                                  .rotAngVelDeg = -72,
                                              });
    shell.set_pinned_verts_from_norm_position([](const luisa::float3& norm_pos)
                                              { return (norm_pos.x > 0.999f); },
                                              {
                                                  .use_rotate   = true,
                                                  .rotCenter    = luisa::make_float3(-0.005, 0, 0),
                                                  .rotAxis      = luisa::make_float3(1, 0, 0),
                                                  .rotAngVelDeg = 72,
                                              });
    lcs::get_scene_params().use_energy_linesearch = false;
    lcs::get_scene_params().gravity               = luisa::make_float3(0.0f);
    lcs::get_scene_params().use_floor             = false;
    lcs::get_scene_params().contact_energy_type   = uint(lcs::ContactEnergyType::Barrier);
}
void ccd_rotation_square(std::vector<ShellInfo>& shell_list)
{
    auto& shell = shell_list
                      .emplace_back(ShellInfo{
                          .model_name = obj_mesh_path + "square2K.obj",
                      })
                      .load_mesh_data();

    shell.set_pinned_verts_from_norm_position([](const luisa::float3& norm_pos)
                                              { return (norm_pos.x < 0.001f); },
                                              {
                                                  .use_rotate   = true,
                                                  .rotCenter    = luisa::make_float3(0.005, 0, 0),
                                                  .rotAxis      = luisa::make_float3(1, 0, 0),
                                                  .rotAngVelDeg = -72,
                                              });
    shell.set_pinned_verts_from_norm_position([](const luisa::float3& norm_pos)
                                              { return (norm_pos.x > 0.999f); },
                                              {
                                                  .use_rotate   = true,
                                                  .rotCenter    = luisa::make_float3(-0.005, 0, 0),
                                                  .rotAxis      = luisa::make_float3(1, 0, 0),
                                                  .rotAngVelDeg = 72,
                                              });

    lcs::get_scene_params().pcg_iter_count       = 50;
    lcs::get_scene_params().nonlinear_iter_count = 1;
    lcs::get_scene_params().gravity              = luisa::make_float3(0.0f);
    lcs::get_scene_params().use_floor            = false;
    lcs::get_scene_params().contact_energy_type  = uint(lcs::ContactEnergyType::Barrier);
}
void ccd_rotation_cylinder_highres(std::vector<ShellInfo>& shell_list)
{
    auto& shell = shell_list
                      .emplace_back(ShellInfo{
                          .model_name = obj_mesh_path + "Cylinder/cylinder88K.obj",
                      })
                      .load_mesh_data();

    shell.set_pinned_verts_from_norm_position([](const luisa::float3& norm_pos)
                                              { return (norm_pos.x < 0.001f); },
                                              {
                                                  .use_rotate   = true,
                                                  .rotCenter    = luisa::make_float3(0.005, 0, 0),
                                                  .rotAxis      = luisa::make_float3(1, 0, 0),
                                                  .rotAngVelDeg = -72,
                                              });
    shell.set_pinned_verts_from_norm_position([](const luisa::float3& norm_pos)
                                              { return (norm_pos.x > 0.999f); },
                                              {
                                                  .use_rotate   = true,
                                                  .rotCenter    = luisa::make_float3(-0.005, 0, 0),
                                                  .rotAxis      = luisa::make_float3(1, 0, 0),
                                                  .rotAngVelDeg = 72,
                                              });

    lcs::get_scene_params().nonlinear_iter_count = 3;
    lcs::get_scene_params().gravity              = luisa::make_float3(0.0f);
    lcs::get_scene_params().use_floor            = false;
    lcs::get_scene_params().contact_energy_type  = uint(lcs::ContactEnergyType::Barrier);
}
void pinned_cloth(std::vector<ShellInfo>& shell_list)
{
    auto& shell = shell_list
                      .emplace_back(ShellInfo{
                          .model_name  = obj_mesh_path + "square2.obj",
                          .translation = luisa::make_float3(0, 0.2, 0),
                          .scale       = luisa::make_float3(0.2f),
                      })
                      .load_mesh_data();
    shell.set_pinned_verts_from_norm_position([](const luisa::float3& norm_pos)
                                              { return (norm_pos.x < 0.001f) && (norm_pos.z < 0.001f); });
    lcs::get_scene_params().use_floor            = false;
    lcs::get_scene_params().use_self_collision   = false;
    lcs::get_scene_params().use_gpu              = false;
    lcs::get_scene_params().nonlinear_iter_count = 1;
    lcs::get_scene_params().use_ccd_linesearch   = false;
    lcs::get_scene_params().pcg_iter_count       = 50;
}
void cloth_rigid_coupling(std::vector<ShellInfo>& shell_list)
{

    shell_list.push_back({.model_name  = obj_mesh_path + "square26K.obj",
                          .translation = luisa::make_float3(0, 0.22, 0),
                          .scale       = luisa::make_float3(0.2f)});
    shell_list.back().load_mesh_data();
    shell_list.back().set_pinned_verts_from_norm_position([](const auto& pos)
                                                          { return pos.x < 0.001f | pos.x > 0.999f; });
    // shell_list.push_back({.model_name  = obj_mesh_path + "square2K.obj",
    //                       .translation = luisa::make_float3(0, 0.24, 0),
    //                       .scale       = luisa::make_float3(0.1f)});
    shell_list.push_back({.model_name  = obj_mesh_path + "cube.obj",
                          .translation = luisa::make_float3(0, 0.24, 0),
                          .rotation    = luisa::make_float3(lcs::Pi / 6, 0, lcs::Pi / 6),
                          .scale       = luisa::make_float3(0.1),
                          .shell_type  = ShellTypeRigid});

    for (auto& shell : shell_list)
        shell.load_mesh_data();
}
void load_default_scene(std::vector<ShellInfo>& shell_list)
{
    lcs::get_scene_params().scene_id = 1;
    const uint case_number           = lcs::get_scene_params().scene_id;

    switch (case_number)
    {
        case 0: {
            ccd_vf_unit_case(shell_list);
            break;
        };
        case 1: {
            pinned_cloth(shell_list);
            break;
        };
        case 2: {
            moving_dirichlet_unit(shell_list);
            break;
        };
        case 3: {
            ccd_rotation_cylinder_7K_quadratic(shell_list);
            break;
        };
        case 4: {
            ccd_rotation_cylinder_7K_ipc(shell_list);
            break;
        };
        case 5: {
            ccd_rotation_cylinder_highres(shell_list);
            break;
        };
        case 6: {
            ccd_rotation_square(shell_list);
            break;
        };
        // case 7: {
        //     energy_linesearch_vf_unit_case(shell_list);
        //     break;
        // };
        case 8: {
            rigid_body_cube_unit_case(shell_list);
            break;
        };
        case 9: {
            rigid_body_folding_cube_case(shell_list);
            break;
        };
        case 10: {
            cloth_rigid_coupling(shell_list);
            break;
        };
        case 11: {
            cloth_bottle4(shell_list);
            break;
        };

        default:
            ccd_vf_unit_case(shell_list);
            break;
    };

    for (auto& shell_info : shell_list)
    {
        if (shell_info.input_mesh.model_positions.empty())
        {
            bool second_read = SimMesh::read_mesh_file(shell_info.model_name, shell_info.input_mesh);
        }
    }
}

void load_scene_params_from_json(std::vector<ShellInfo>& shell_list)
{
    const std::string json_path = std::string(LCSV_RESOURCE_PATH) + "/scene_config.json";
    std::ifstream     ifs(json_path);
    if (!ifs.is_open())
    {
        LUISA_WARNING("Cannot open json file: {}, using default scene params", json_path);
        return;
    }
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    std::string content = buffer.str();
    ifs.close();
    yyjson_doc* doc = yyjson_read(content.c_str(), content.size(), 0);
    if (!doc)
    {
        LUISA_WARNING("Cannot parse json file: {}, using default scene params", json_path);
        return;
    }
}


}  // namespace Demo::Simulation