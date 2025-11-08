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
#include <filesystem>
#include <yyjson.h>
#include <cassert>

namespace Demo::Simulation
{

const std::string obj_mesh_path = std::string(LCSV_RESOURCE_PATH) + "/InputMesh/";
const std::string tet_mesh_path = std::string(LCSV_RESOURCE_PATH) + "/InputMesh/vtks/";

using namespace lcs::Initializer;

void ccd_vf_unit_case(std::vector<WorldData>& shell_list)
{
    WorldData& up = shell_list.emplace_back(WorldData())
                        .set_name("upper square")
                        .set_simulation_type(lcs::Initializer::SimulationTypeCloth)
                        .load_mesh_from_path(obj_mesh_path + "square2.obj")
                        .set_physics_material(ClothMaterial{.thickness = 0.1f})
                        .add_fixed_point_info({.method = lcs::Initializer::FixedPointsType::LeftBack})
                        .load_fixed_points();

    WorldData& down = shell_list.emplace_back(WorldData())
                          .set_name("lower square")
                          .set_simulation_type(lcs::Initializer::SimulationTypeCloth)
                          .load_mesh_from_path(obj_mesh_path + "square2.obj")
                          .set_physics_material(ClothMaterial{.thickness = 0.1f})
                          .add_fixed_point_info({.method = lcs::Initializer::FixedPointsType::Left})
                          .add_fixed_point_info({.method = lcs::Initializer::FixedPointsType::Right})
                          .load_fixed_points();

    lcs::get_scene_params().use_floor             = false;
    lcs::get_scene_params().implicit_dt           = 0.2;
    lcs::get_scene_params().use_energy_linesearch = true;
    lcs::get_scene_params().stiffness_DAB_bending = 200.0f;
    lcs::get_scene_params().use_gpu               = false;
}

void load_scene_params_from_json(std::vector<WorldData>& shell_list, const std::string& json_path)
{
    // Determine which path to open:
    // 1) If user provided an absolute path and it exists, use it.
    // 2) Otherwise, try "LCSV_RESOURCE_PATH/Scenes/" + json_path and then + basename(json_path).
    // 3) If none exists, log a warning and return (use default scene params).

    auto file_exists_safe = [](const std::string& p)
    {
        try
        {
            return std::filesystem::exists(p);
        }
        catch (...)
        {
            return false;
        }
    };

    std::string           path_to_open;
    std::filesystem::path user_path(json_path);
    const std::string     resource_scenes = std::string(LCSV_RESOURCE_PATH) + "/Scenes/";

    // Prefer user absolute path only when it's absolute and exists
    if (user_path.is_absolute() && file_exists_safe(user_path.string()))
    {
        path_to_open = user_path.string();
    }
    else
    {
        // Try resource_scenes + json_path (keep any subdirs user provided)
        std::string candidate = resource_scenes + json_path;
        if (file_exists_safe(candidate))
        {
            path_to_open = candidate;
        }
        else
        {
            // Try basename only
            std::string base = user_path.filename().string();
            candidate        = resource_scenes + base;
            if (file_exists_safe(candidate))
            {
                path_to_open = candidate;
            }
        }
    }

    if (path_to_open.empty())
    {
        LUISA_WARNING("Cannot find json file at provided path '{}' nor in '{}', using default scene params",
                      json_path,
                      resource_scenes);
        return;
    }

    std::ifstream ifs(path_to_open);
    if (!ifs.is_open())
    {
        LUISA_WARNING("Found json candidate '{}' but failed to open it, using default scene params", path_to_open);
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
    yyjson_val* root = yyjson_doc_get_root(doc);
    if (!root)
    {
        LUISA_WARNING("Empty json root: {}, using default scene params", json_path);
        yyjson_doc_free(doc);
        return;
    }

    // Helper lambdas
    auto get_bool = [&](const char* key, bool& out)
    {
        yyjson_val* v = yyjson_obj_get(root, key);
        if (v && yyjson_is_bool(v))
            out = yyjson_get_bool(v);
    };
    auto get_uint = [&](const char* key, uint& out)
    {
        yyjson_val* v = yyjson_obj_get(root, key);
        if (v && yyjson_is_uint(v))
            out = static_cast<uint>(yyjson_get_uint(v));
    };
    auto get_int = [&](const char* key, int& out)
    {
        yyjson_val* v = yyjson_obj_get(root, key);
        if (v && yyjson_is_int(v))
            out = yyjson_get_int(v);
    };
    auto get_real = [&](const char* key, float& out)
    {
        yyjson_val* v = yyjson_obj_get(root, key);
        if (v && yyjson_is_num(v))
            out = static_cast<float>(yyjson_get_num(v));
    };
    auto get_vec3 = [&](const char* key, luisa::float3& out)
    {
        yyjson_val* v = yyjson_obj_get(root, key);
        if (v && yyjson_is_arr(v) && yyjson_get_len(v) >= 3)
        {
            yyjson_val* elem;
            size_t      idx, max;
            float       vals[3] = {0.f, 0.f, 0.f};
            yyjson_arr_foreach(v, idx, max, elem)
            {
                if (idx < 3 && yyjson_is_num(elem))
                    vals[idx] = static_cast<float>(yyjson_get_num(elem));
            }
            out = luisa::make_float3(vals[0], vals[1], vals[2]);
        }
    };

    // Scene-level params (common ones used in demo code)
    yyjson_val* val = nullptr;
    val             = yyjson_obj_get(root, "scene_id");
    if (val && yyjson_is_uint(val))
        lcs::get_scene_params().scene_id = static_cast<uint>(yyjson_get_uint(val));

    get_bool("use_floor", lcs::get_scene_params().use_floor);
    get_bool("use_gpu", lcs::get_scene_params().use_gpu);
    get_bool("use_self_collision", lcs::get_scene_params().use_self_collision);
    get_bool("use_ccd_linesearch", lcs::get_scene_params().use_ccd_linesearch);

    get_real("implicit_dt", lcs::get_scene_params().implicit_dt);
    get_real("stiffness_collision", lcs::get_scene_params().stiffness_collision);
    get_real("stiffness_dirichlet", lcs::get_scene_params().stiffness_dirichlet);

    get_int("nonlinear_iter_count", reinterpret_cast<int&>(lcs::get_scene_params().nonlinear_iter_count));
    get_int("pcg_iter_count", reinterpret_cast<int&>(lcs::get_scene_params().pcg_iter_count));

    get_vec3("gravity", lcs::get_scene_params().gravity);

    // contact_energy_type may be provided as uint
    val = yyjson_obj_get(root, "contact_energy_type");
    if (val && yyjson_is_uint(val))
        lcs::get_scene_params().contact_energy_type = static_cast<uint>(yyjson_get_uint(val));

    if (lcs::get_scene_params().contact_energy_type == uint(lcs::ContactEnergyType::Barrier)
        && (lcs::get_scene_params().use_self_collision || lcs::get_scene_params().use_floor))
        lcs::get_scene_params().use_ccd_linesearch = true;

    // Helper to parse FixedPointsType from string
    auto parse_fixed_method = [](const char* s)
    {
        using namespace lcs::Initializer;
        if (!s)
            return FixedPointsType::All;
        if (strcmp(s, "None") == 0)
            return FixedPointsType::None;
        if (strcmp(s, "FromIndices") == 0)
            return FixedPointsType::FromIndices;
        if (strcmp(s, "FromFunction") == 0)
            return FixedPointsType::FromFunction;
        if (strcmp(s, "Left") == 0)
            return FixedPointsType::Left;
        if (strcmp(s, "Right") == 0)
            return FixedPointsType::Right;
        if (strcmp(s, "Front") == 0)
            return FixedPointsType::Front;
        if (strcmp(s, "Back") == 0)
            return FixedPointsType::Back;
        if (strcmp(s, "Up") == 0)
            return FixedPointsType::Up;
        if (strcmp(s, "Down") == 0)
            return FixedPointsType::Down;
        if (strcmp(s, "LeftBack") == 0)
            return FixedPointsType::LeftBack;
        if (strcmp(s, "LeftFront") == 0)
            return FixedPointsType::LeftFront;
        if (strcmp(s, "RightBack") == 0)
            return FixedPointsType::RightBack;
        if (strcmp(s, "RightFront") == 0)
            return FixedPointsType::RightFront;
        if (strcmp(s, "All") == 0)
            return FixedPointsType::All;
        return FixedPointsType::All;
    };

    // Parse shells array
    yyjson_val* shells = yyjson_obj_get(root, "shells");
    if (shells && yyjson_is_arr(shells))
    {
        yyjson_val* shell_val;
        size_t      i, n;
        yyjson_arr_foreach(shells, i, n, shell_val)
        {
            if (!yyjson_is_obj(shell_val))
                continue;
            lcs::Initializer::WorldData info;

            // model_name
            yyjson_val* m = yyjson_obj_get(shell_val, "model_name");
            if (m && yyjson_is_str(m))
            {
                const char* s           = yyjson_get_str(m);
                std::string model_str   = s ? s : "";
                auto        file_exists = [](const std::string& p)
                {
                    try
                    {
                        return std::filesystem::exists(p);
                    }
                    catch (...)  // in case filesystem is not available or path is invalid
                    {
                        return false;
                    }
                };

                if (model_str.empty())
                {
                    info.set_name(model_str);
                }
                else if (file_exists(model_str))
                {
                    // user provided a path that exists (absolute or relative)
                    info.set_name(model_str);
                }
                else
                {
                    // try Resources/InputMesh/<model_str>
                    std::string candidate = obj_mesh_path + model_str;
                    if (file_exists(candidate))
                    {
                        info.set_name(candidate);
                    }
                    else
                    {
                        // try basename in InputMesh (user may give subpath or just name)
                        size_t pos = model_str.find_last_of("/\\");
                        std::string base = (pos == std::string::npos) ? model_str : model_str.substr(pos + 1);
                        candidate = obj_mesh_path + base;
                        if (file_exists(candidate))
                        {
                            info.set_name(candidate);
                        }
                        else
                        {
                            // try tet mesh path (vtks)
                            candidate = tet_mesh_path + model_str;
                            if (file_exists(candidate))
                            {
                                info.set_name(candidate);
                            }
                            else
                            {
                                // fallback: keep as provided (load_mesh_data will try to read it)
                                info.set_name(model_str);
                            }
                        }
                    }
                }
            }

            // translation / rotation / scale
            yyjson_val* t = yyjson_obj_get(shell_val, "translation");
            if (t && yyjson_is_arr(t) && yyjson_get_len(t) >= 3)
            {
                yyjson_val* e;
                size_t      idx, max;
                float       tv[3] = {0};
                yyjson_arr_foreach(t, idx, max, e)
                {
                    if (idx < 3 && yyjson_is_num(e))
                        tv[idx] = static_cast<float>(yyjson_get_num(e));
                }
                info.set_translation(tv[0], tv[1], tv[2]);
            }
            yyjson_val* r = yyjson_obj_get(shell_val, "rotation");
            if (r && yyjson_is_arr(r) && yyjson_get_len(r) >= 3)
            {
                yyjson_val* e;
                size_t      idx, max;
                float       rv[3] = {0};
                yyjson_arr_foreach(r, idx, max, e)
                {
                    if (idx < 3 && yyjson_is_num(e))
                        rv[idx] = static_cast<float>(yyjson_get_num(e));
                }
                info.set_rotation(rv[0], rv[1], rv[2]);
            }
            yyjson_val* sc = yyjson_obj_get(shell_val, "scale");
            if (sc)
            {
                if (yyjson_is_arr(sc) && yyjson_get_len(sc) >= 3)
                {
                    yyjson_val* e;
                    size_t      idx, max;
                    float       sv[3] = {1.f, 1.f, 1.f};
                    yyjson_arr_foreach(sc, idx, max, e)
                    {
                        if (idx < 3 && yyjson_is_num(e))
                            sv[idx] = static_cast<float>(yyjson_get_num(e));
                    }
                    info.set_scale(sv[0], sv[1], sv[2]);
                }
                else if (yyjson_is_num(sc))
                {
                    float sval = static_cast<float>(yyjson_get_num(sc));
                    info.set_scale(sval);  // Scale x,y,z with same range
                }
            }

            // set simulation type
            yyjson_val* stype = yyjson_obj_get(shell_val, "shell_type");
            if (stype && yyjson_is_str(stype))
            {
                const char* ss = yyjson_get_str(stype);
                if (strcmp(ss, "Rigid") == 0)
                    info.set_simulation_type(lcs::Initializer::SimulationTypeRigid);
                else if (strcmp(ss, "Tetrahedral") == 0)
                    info.set_simulation_type(lcs::Initializer::SimulationTypeTetrahedral);
                else if (strcmp(ss, "Rod") == 0)
                    info.set_simulation_type(lcs::Initializer::SimulationTypeRod);
                else
                    info.set_simulation_type(lcs::Initializer::SimulationTypeCloth);
            }

            // set physical material
            yyjson_val* pm = yyjson_obj_get(shell_val, "material");
            if (pm)
            {
                yyjson_val* mat_obj = nullptr;
                if (yyjson_is_arr(pm))
                {
                    yyjson_val* e;
                    size_t      idx, max;
                    yyjson_arr_foreach(pm, idx, max, e)
                    {
                        if (yyjson_is_obj(e))
                        {
                            mat_obj = e;
                            break;
                        }
                    }
                }
                else if (yyjson_is_obj(pm))
                {
                    mat_obj = pm;
                }

                if (mat_obj)
                {
                    // Determine material type: explicit "type" field preferred
                    const char* mtype_str = nullptr;
                    yyjson_val* mtv       = yyjson_obj_get(mat_obj, "type");
                    if (mtv && yyjson_is_str(mtv))
                        mtype_str = yyjson_get_str(mtv);

                    std::string material_type = mtype_str ? std::string(mtype_str) :
                                                stype && yyjson_is_str(stype) ?
                                                            std::string(yyjson_get_str(stype)) :
                                                            "Cloth";

                    // Helper to map material type -> shell type
                    auto material_to_shell = [&](const std::string& s)
                    {
                        if (s == "Rigid")
                            return lcs::Initializer::SimulationTypeRigid;
                        else if (s == "Tetrahedral" || s == "Tet" || s == "TetMaterial")
                            return lcs::Initializer::SimulationTypeTetrahedral;
                        else if (s == "Rod")
                            return lcs::Initializer::SimulationTypeRod;
                        return lcs::Initializer::SimulationTypeCloth;
                    };

                    // Parse and fill material struct based on material_type
                    if (material_type == "Cloth")
                    {
                        lcs::Initializer::ClothMaterial mat;
                        yyjson_val*                     v = nullptr;
                        v                                 = yyjson_obj_get(mat_obj, "thickness");
                        if (v && yyjson_is_num(v))
                            mat.thickness = static_cast<float>(yyjson_get_num(v));
                        v = yyjson_obj_get(mat_obj, "youngs_modulus");
                        if (v && yyjson_is_num(v))
                            mat.youngs_modulus = static_cast<float>(yyjson_get_num(v));
                        v = yyjson_obj_get(mat_obj, "poisson_ratio");
                        if (v && yyjson_is_num(v))
                            mat.poisson_ratio = static_cast<float>(yyjson_get_num(v));
                        v = yyjson_obj_get(mat_obj, "area_bending_stiffness");
                        if (v && yyjson_is_num(v))
                            mat.area_bending_stiffness = static_cast<float>(yyjson_get_num(v));
                        v = yyjson_obj_get(mat_obj, "mass");
                        if (v && yyjson_is_num(v))
                            mat.mass = static_cast<float>(yyjson_get_num(v));
                        v = yyjson_obj_get(mat_obj, "density");
                        if (v && yyjson_is_num(v))
                            mat.density = static_cast<float>(yyjson_get_num(v));
                        v = yyjson_obj_get(mat_obj, "d_hat");
                        if (v && yyjson_is_num(v))
                            mat.d_hat = static_cast<float>(yyjson_get_num(v));
                        v = yyjson_obj_get(mat_obj, "thickness");
                        if (v && yyjson_is_num(v))
                            mat.thickness = static_cast<float>(yyjson_get_num(v));

                        // stretch_model may be string or integer
                        v = yyjson_obj_get(mat_obj, "stretch_model");
                        if (v)
                        {
                            if (yyjson_is_str(v))
                            {
                                const char* s = yyjson_get_str(v);
                                if (strcmp(s, "Spring") == 0)
                                    mat.stretch_model = lcs::Initializer::ConstitutiveStretchModelCloth::Spring;
                                else
                                    mat.stretch_model = lcs::Initializer::ConstitutiveStretchModelCloth::FEM_BW98;
                            }
                            else if (yyjson_is_int(v) || yyjson_is_uint(v) || yyjson_is_num(v))
                            {
                                int iv = yyjson_get_int(v);
                                mat.stretch_model =
                                    static_cast<lcs::Initializer::ConstitutiveStretchModelCloth>(iv);
                            }
                        }

                        v = yyjson_obj_get(mat_obj, "bending_model");
                        if (v)
                        {
                            if (yyjson_is_str(v))
                            {
                                const char* s = yyjson_get_str(v);
                                if (strcmp(s, "None") == 0)
                                    mat.bending_model = lcs::Initializer::ConstitutiveBendingModelCloth::None;
                                else if (strcmp(s, "QuadraticBending") == 0)
                                    mat.bending_model = lcs::Initializer::ConstitutiveBendingModelCloth::QuadraticBending;
                                else
                                    mat.bending_model = lcs::Initializer::ConstitutiveBendingModelCloth::DihedralAngle;
                            }
                            else if (yyjson_is_int(v) || yyjson_is_uint(v) || yyjson_is_num(v))
                            {
                                int iv = yyjson_get_int(v);
                                mat.bending_model =
                                    static_cast<lcs::Initializer::ConstitutiveBendingModelCloth>(iv);
                            }
                        }

                        info.set_physics_material(mat);
                        // if shell_type not provided explicitly, set from material
                        if (stype == nullptr)
                        {
                            info.set_simulation_type(lcs::Initializer::SimulationTypeCloth);
                        }
                        else
                        {
                            // ensure consistency when both provided
                            auto mt = material_to_shell(material_type);
                            LUISA_ASSERT(mt == info.simulation_type && "shell_type and material.type mismatch");
                        }
                    }
                    else if (material_type == "Tetrahedral" || material_type == "Tet" || material_type == "TetMaterial")
                    {
                        lcs::Initializer::TetMaterial mat;
                        yyjson_val*                   v = nullptr;
                        v                               = yyjson_obj_get(mat_obj, "youngs_modulus");
                        if (v && yyjson_is_num(v))
                            mat.youngs_modulus = static_cast<float>(yyjson_get_num(v));
                        v = yyjson_obj_get(mat_obj, "poisson_ratio");
                        if (v && yyjson_is_num(v))
                            mat.poisson_ratio = static_cast<float>(yyjson_get_num(v));
                        v = yyjson_obj_get(mat_obj, "mass");
                        if (v && yyjson_is_num(v))
                            mat.mass = static_cast<float>(yyjson_get_num(v));
                        v = yyjson_obj_get(mat_obj, "density");
                        if (v && yyjson_is_num(v))
                            mat.density = static_cast<float>(yyjson_get_num(v));
                        v = yyjson_obj_get(mat_obj, "d_hat");
                        if (v && yyjson_is_num(v))
                            mat.d_hat = static_cast<float>(yyjson_get_num(v));

                        info.set_physics_material(mat);
                        if (stype == nullptr)
                        {
                            info.set_simulation_type(lcs::Initializer::SimulationTypeTetrahedral);
                        }
                        else
                        {
                            auto mt = material_to_shell(material_type);
                            LUISA_ASSERT(mt == info.simulation_type && "shell_type and material.type mismatch");
                        }
                    }
                    else if (material_type == "Rigid")
                    {
                        lcs::Initializer::RigidMaterial mat;
                        yyjson_val*                     v = nullptr;
                        v                                 = yyjson_obj_get(mat_obj, "thickness");
                        if (v && yyjson_is_num(v))
                            mat.thickness = static_cast<float>(yyjson_get_num(v));
                        v = yyjson_obj_get(mat_obj, "stiffness");
                        if (v && yyjson_is_num(v))
                            mat.stiffness = static_cast<float>(yyjson_get_num(v));
                        v = yyjson_obj_get(mat_obj, "mass");
                        if (v && yyjson_is_num(v))
                            mat.mass = static_cast<float>(yyjson_get_num(v));
                        v = yyjson_obj_get(mat_obj, "density");
                        if (v && yyjson_is_num(v))
                            mat.density = static_cast<float>(yyjson_get_num(v));
                        v = yyjson_obj_get(mat_obj, "d_hat");
                        if (v && yyjson_is_num(v))
                            mat.d_hat = static_cast<float>(yyjson_get_num(v));
                        v = yyjson_obj_get(mat_obj, "thickness");
                        if (v && yyjson_is_num(v))
                            mat.thickness = static_cast<float>(yyjson_get_num(v));
                        v = yyjson_obj_get(mat_obj, "is_shell");
                        if (v && yyjson_is_bool(v))
                            mat.is_shell = static_cast<bool>(yyjson_get_bool(v));

                        info.set_physics_material(mat);
                        if (stype == nullptr)
                        {
                            info.set_simulation_type(lcs::Initializer::SimulationTypeRigid);
                        }
                        else
                        {
                            auto mt = material_to_shell(material_type);
                            LUISA_ASSERT(mt == info.simulation_type && "shell_type and material.type mismatch");
                        }
                    }
                    else if (material_type == "Rod")
                    {
                        lcs::Initializer::RodMaterial mat;
                        yyjson_val*                   v = nullptr;
                        v                               = yyjson_obj_get(mat_obj, "radius");
                        if (v && yyjson_is_num(v))
                            mat.radius = static_cast<float>(yyjson_get_num(v));
                        v = yyjson_obj_get(mat_obj, "bending_stiffness");
                        if (v && yyjson_is_num(v))
                            mat.bending_stiffness = static_cast<float>(yyjson_get_num(v));
                        v = yyjson_obj_get(mat_obj, "twisting_stiffness");
                        if (v && yyjson_is_num(v))
                            mat.twisting_stiffness = static_cast<float>(yyjson_get_num(v));
                        v = yyjson_obj_get(mat_obj, "mass");
                        if (v && yyjson_is_num(v))
                            mat.mass = static_cast<float>(yyjson_get_num(v));
                        v = yyjson_obj_get(mat_obj, "density");
                        if (v && yyjson_is_num(v))
                            mat.density = static_cast<float>(yyjson_get_num(v));
                        v = yyjson_obj_get(mat_obj, "d_hat");
                        if (v && yyjson_is_num(v))
                            mat.d_hat = static_cast<float>(yyjson_get_num(v));
                        v = yyjson_obj_get(mat_obj, "radius");
                        if (v && yyjson_is_num(v))
                            mat.radius = static_cast<float>(yyjson_get_num(v));

                        info.set_physics_material(mat);
                        if (stype == nullptr)
                        {
                            info.set_simulation_type(lcs::Initializer::SimulationTypeRod);
                        }
                        else
                        {
                            auto mt = material_to_shell(material_type);
                            LUISA_ASSERT(mt == info.simulation_type && "shell_type and material.type mismatch");
                        }
                    }
                    else
                    {
                        // fallback: treat as cloth
                        lcs::Initializer::ClothMaterial mat;
                        yyjson_val*                     v = yyjson_obj_get(mat_obj, "thickness");
                        if (v && yyjson_is_num(v))
                            mat.thickness = static_cast<float>(yyjson_get_num(v));
                        v = yyjson_obj_get(mat_obj, "d_hat");
                        if (v && yyjson_is_num(v))
                            mat.d_hat = static_cast<float>(yyjson_get_num(v));

                        info.set_physics_material(mat);
                        if (stype == nullptr)
                            info.set_simulation_type(lcs::Initializer::SimulationTypeCloth);
                    }
                }
            }

            // fixed_points
            yyjson_val* fp_arr = yyjson_obj_get(shell_val, "fixed_points");
            if (fp_arr && yyjson_is_arr(fp_arr))
            {
                yyjson_val* fpv;
                size_t      j, jm;
                yyjson_arr_foreach(fp_arr, j, jm, fpv)
                {
                    if (!yyjson_is_obj(fpv))
                        continue;
                    lcs::Initializer::MakeFixedPointsInterface mfp;
                    // method
                    yyjson_val* mth = yyjson_obj_get(fpv, "method");
                    if (mth && yyjson_is_str(mth))
                        mfp.method = parse_fixed_method(yyjson_get_str(mth));
                    // range
                    yyjson_val* rng = yyjson_obj_get(fpv, "range");
                    if (rng && yyjson_is_arr(rng))
                    {
                        yyjson_val* ev;
                        size_t      ri, rmax;
                        mfp.range.clear();
                        yyjson_arr_foreach(rng, ri, rmax, ev)
                        {
                            if (yyjson_is_num(ev))
                                mfp.range.push_back(static_cast<float>(yyjson_get_num(ev)));
                        }
                    }
                    // fixed_info (animation)
                    yyjson_val* fin = yyjson_obj_get(fpv, "fixed_info");
                    if (fin && yyjson_is_obj(fin))
                    {
                        auto&       fi = mfp.fixed_info;
                        yyjson_val* b;
                        b = yyjson_obj_get(fin, "use_translate");
                        if (b && yyjson_is_bool(b))
                            fi.use_translate = yyjson_get_bool(b);
                        b = yyjson_obj_get(fin, "translate");
                        if (b && yyjson_is_arr(b) && yyjson_get_len(b) >= 3)
                        {
                            yyjson_val* e;
                            size_t      idx, max;
                            float       tv2[3] = {0};
                            yyjson_arr_foreach(b, idx, max, e)
                            {
                                if (idx < 3 && yyjson_is_num(e))
                                    tv2[idx] = static_cast<float>(yyjson_get_num(e));
                            }
                            fi.translate = luisa::make_float3(tv2[0], tv2[1], tv2[2]);
                        }
                        b = yyjson_obj_get(fin, "use_rotate");
                        if (b && yyjson_is_bool(b))
                            fi.use_rotate = yyjson_get_bool(b);
                        b = yyjson_obj_get(fin, "rotCenter");
                        if (b && yyjson_is_arr(b) && yyjson_get_len(b) >= 3)
                        {
                            yyjson_val* e;
                            size_t      idx, max;
                            float       rv2[3] = {0};
                            yyjson_arr_foreach(b, idx, max, e)
                            {
                                if (idx < 3 && yyjson_is_num(e))
                                    rv2[idx] = static_cast<float>(yyjson_get_num(e));
                            }
                            fi.rotCenter = luisa::make_float3(rv2[0], rv2[1], rv2[2]);
                        }
                        b = yyjson_obj_get(fin, "rotAxis");
                        if (b && yyjson_is_arr(b) && yyjson_get_len(b) >= 3)
                        {
                            yyjson_val* e;
                            size_t      idx, max;
                            float       av[3] = {0};
                            yyjson_arr_foreach(b, idx, max, e)
                            {
                                if (idx < 3 && yyjson_is_num(e))
                                    av[idx] = static_cast<float>(yyjson_get_num(e));
                            }
                            fi.rotAxis = luisa::make_float3(av[0], av[1], av[2]);
                        }
                        b = yyjson_obj_get(fin, "rotAngVelDeg");
                        if (b && yyjson_is_num(b))
                            fi.rotAngVelDeg = static_cast<float>(yyjson_get_num(b));
                        b = yyjson_obj_get(fin, "use_scale");
                        if (b && yyjson_is_bool(b))
                            fi.use_scale = yyjson_get_bool(b);
                        b = yyjson_obj_get(fin, "scale");
                        if (b && yyjson_is_arr(b) && yyjson_get_len(b) >= 3)
                        {
                            yyjson_val* e;
                            size_t      idx, max;
                            float       sv2[3] = {1.f, 1.f, 1.f};
                            yyjson_arr_foreach(b, idx, max, e)
                            {
                                if (idx < 3 && yyjson_is_num(e))
                                    sv2[idx] = static_cast<float>(yyjson_get_num(e));
                            }
                            fi.scale     = luisa::make_float3(sv2[0], sv2[1], sv2[2]);
                            fi.use_scale = true;
                        }
                        b = yyjson_obj_get(fin, "use_setting_position");
                        if (b && yyjson_is_bool(b))
                            fi.use_setting_position = yyjson_get_bool(b);
                        b = yyjson_obj_get(fin, "setting_position");
                        if (b && yyjson_is_arr(b) && yyjson_get_len(b) >= 3)
                        {
                            yyjson_val* e;
                            size_t      idx, max;
                            float       sv3[3] = {0};
                            yyjson_arr_foreach(b, idx, max, e)
                            {
                                if (idx < 3 && yyjson_is_num(e))
                                    sv3[idx] = static_cast<float>(yyjson_get_num(e));
                            }
                            fi.setting_position     = luisa::make_float3(sv3[0], sv3[1], sv3[2]);
                            fi.use_setting_position = true;
                        }
                    }

                    info.fixed_point_range_info.push_back(mfp);
                }
            }

            // finally add to shell_list and load mesh (and fixed points if provided)
            shell_list.emplace_back(info);
            if (!shell_list.back().input_mesh.model_positions.empty())
                shell_list.back().load_mesh_data();
            if (!shell_list.back().fixed_point_range_info.empty())
                shell_list.back().load_fixed_points();
        }
    }

    yyjson_doc_free(doc);
}


}  // namespace Demo::Simulation