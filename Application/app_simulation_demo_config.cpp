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

	bool is_match(const std::string& str, const char*& target)
	{
		return (strcmp(str.c_str(), target) == 0);
	}
	bool is_match(const std::string& str, const std::vector<const char*>& targets)
	{
		for (const char* target : targets)
		{
			if (strcmp(str.c_str(), target) == 0)
				return true;
		}
		return false;
	}

	void load_scene_params_from_json(const std::function<void(const lcs::Initializer::WorldData&)>& fn_register_mesh, const std::string& json_path)
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

		std::string			  path_to_open;
		std::filesystem::path user_path(json_path);
		const std::string	  resource_scenes = std::string(LCSV_RESOURCE_PATH) + "/Scenes/";

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
				candidate = resource_scenes + base;
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
				size_t		idx, max;
				float		vals[3] = { 0.f, 0.f, 0.f };
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
		val = yyjson_obj_get(root, "scene_id");
		if (val && yyjson_is_uint(val))
			lcs::get_scene_params().scene_id = static_cast<uint>(yyjson_get_uint(val));

		get_bool("use_floor", lcs::get_scene_params().use_floor);
		get_bool("use_gpu", lcs::get_scene_params().use_gpu);
		get_bool("use_self_collision", lcs::get_scene_params().use_self_collision);
		get_bool("use_ccd_linesearch", lcs::get_scene_params().use_ccd_linesearch);
		get_bool("use_energy_linesearch", lcs::get_scene_params().use_energy_linesearch);

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
			size_t		i, n;
			yyjson_arr_foreach(shells, i, n, shell_val)
			{
				if (!yyjson_is_obj(shell_val))
					continue;

				// Resolve model path using absolute_dir / relative_dir rules.
				std::string resolved_model_path;

				yyjson_val* abs_dir = yyjson_obj_get(shell_val, "absolute_path");
				if (abs_dir && yyjson_is_str(abs_dir))
				{
					const char* abs_s = yyjson_get_str(abs_dir);
					if (abs_s && file_exists_safe(abs_s))
					{
						resolved_model_path = abs_s;
					}
					else
					{
						LUISA_WARNING("absolute_dir '{}' not found", abs_s ? abs_s : "");
					}
				}

				if (resolved_model_path.empty())
				{
					yyjson_val* rel_dir = yyjson_obj_get(shell_val, "relative_path");
					if (rel_dir && yyjson_is_str(rel_dir))
					{
						const char* rel_s = yyjson_get_str(rel_dir);
						if (rel_s)
						{
							std::string candidate = obj_mesh_path + std::string(rel_s);
							if (file_exists_safe(candidate))
							{
								resolved_model_path = candidate;
							}
							else
							{
								LUISA_WARNING("relative_path '{}' (checked '{}') not found", rel_s, candidate);
							}
						}
					}
				}

				if (resolved_model_path.empty())
				{
					LUISA_WARNING("Neither 'absolute_path' nor 'relative_path' exist or point to a valid file for this shell; skipping shell");
					continue;
				}

				lcs::Initializer::WorldData info;

				// set display name: prefer provided model_name, otherwise use filename of resolved path
				yyjson_val* m = yyjson_obj_get(shell_val, "model_name");
				if (m && yyjson_is_str(m))
				{
					const char* s = yyjson_get_str(m);
					info.set_name(s ? s : "");
				}
				else
				{
					info.set_name(std::filesystem::path(resolved_model_path).filename().string());
				}

				// load mesh from resolved path
				info.load_mesh_from_path(resolved_model_path);

				// translation / rotation / scale
				yyjson_val* t = yyjson_obj_get(shell_val, "translation");
				if (t && yyjson_is_arr(t) && yyjson_get_len(t) >= 3)
				{
					yyjson_val* e;
					size_t		idx, max;
					float		tv[3] = { 0 };
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
					size_t		idx, max;
					float		rv[3] = { 0 };
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
						size_t		idx, max;
						float		sv[3] = { 1.f, 1.f, 1.f };
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
						info.set_scale(sval); // Scale x,y,z with same range
					}
				}

				// set simulation type
				yyjson_val* stype = yyjson_obj_get(shell_val, "material_type");
				if (stype && yyjson_is_str(stype))
				{
					const char* ss = yyjson_get_str(stype);
					if (strcmp(ss, "Rigid") == 0)
						info.set_material_type(lcs::Material::MaterialType::Rigid);
					else if (strcmp(ss, "Tetrahedral") == 0)
						info.set_material_type(lcs::Material::MaterialType::Tetrahedral);
					else if (strcmp(ss, "Rod") == 0)
						info.set_material_type(lcs::Material::MaterialType::Rod);
					else
						info.set_material_type(lcs::Material::MaterialType::Cloth);
				}

				// set physical material
				yyjson_val* pm = yyjson_obj_get(shell_val, "material");
				if (pm)
				{
					yyjson_val* mat_obj = nullptr;
					if (yyjson_is_arr(pm))
					{
						yyjson_val* e;
						size_t		idx, max;
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
						yyjson_val* mtv = yyjson_obj_get(mat_obj, "type");
						if (mtv && yyjson_is_str(mtv))
							mtype_str = yyjson_get_str(mtv);

						std::string material_type = mtype_str ? std::string(mtype_str) : stype && yyjson_is_str(stype) ? std::string(yyjson_get_str(stype))
																													   : "Cloth";

						// Helper to map material type -> shell type
						auto material_to_shell = [&](const std::string& s)
						{
							if (s == "Rigid")
								return lcs::Material::MaterialType::Rigid;
							else if (s == "Tetrahedral" || s == "Tet" || s == "TetMaterial")
								return lcs::Material::MaterialType::Tetrahedral;
							else if (s == "Rod")
								return lcs::Material::MaterialType::Rod;
							return lcs::Material::MaterialType::Cloth;
						};

						// Parse and fill material struct based on material_type
						if (material_type == "Cloth")
						{
							lcs::Material::ClothMaterial mat;
							yyjson_val*					 v = nullptr;
							v = yyjson_obj_get(mat_obj, "thickness");
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
							v = yyjson_obj_get(mat_obj, "contact_offset");
							if (v && yyjson_is_num(v))
								mat.contact_offset = static_cast<float>(yyjson_get_num(v));
							v = yyjson_obj_get(mat_obj, "friction_mu");
							if (v && yyjson_is_num(v))
								mat.friction_mu = static_cast<float>(yyjson_get_num(v));

							// stretch_model may be string or integer
							v = yyjson_obj_get(mat_obj, "stretch_model");
							if (v)
							{
								if (yyjson_is_str(v))
								{
									const char* s = yyjson_get_str(v);
									mat.stretch_model = lcs::Material::parse_cloth_stretch_model(s);
								}
							}

							v = yyjson_obj_get(mat_obj, "bending_model");
							if (v)
							{
								if (yyjson_is_str(v))
								{
									const char* s = yyjson_get_str(v);
									mat.bending_model = lcs::Material::parse_cloth_bending_model(s);
								}
							}

							info.set_physics_material(mat);
							// if material_type not provided explicitly, set from material
							if (stype == nullptr)
							{
								info.set_material_type(lcs::Material::MaterialType::Cloth);
							}
							else
							{
								// ensure consistency when both provided
								auto mt = material_to_shell(material_type);
								LUISA_ASSERT(mt == info.material_type && "material_type and material.type mismatch");
							}
						}
						else if (material_type == "Tetrahedral" || material_type == "Tet" || material_type == "TetMaterial")
						{
							lcs::Material::TetMaterial mat;
							yyjson_val*				   v = nullptr;
							v = yyjson_obj_get(mat_obj, "youngs_modulus");
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
							v = yyjson_obj_get(mat_obj, "contact_offset");
							if (v && yyjson_is_num(v))
								mat.contact_offset = static_cast<float>(yyjson_get_num(v));
							v = yyjson_obj_get(mat_obj, "friction_mu");
							if (v && yyjson_is_num(v))
								mat.friction_mu = static_cast<float>(yyjson_get_num(v));

							v = yyjson_obj_get(mat_obj, "model");
							if (v)
							{
								if (yyjson_is_str(v))
								{
									const char* s = yyjson_get_str(v);
									mat.model = lcs::Material::parse_tet_model(s);
								}
							}

							info.set_physics_material(mat);
							if (stype == nullptr)
							{
								info.set_material_type(lcs::Material::MaterialType::Tetrahedral);
							}
							else
							{
								auto mt = material_to_shell(material_type);
								LUISA_ASSERT(mt == info.material_type && "material_type and material.type mismatch");
							}
						}
						else if (material_type == "Rigid")
						{
							lcs::Material::RigidMaterial mat;
							yyjson_val*					 v = nullptr;
							v = yyjson_obj_get(mat_obj, "thickness");
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
							v = yyjson_obj_get(mat_obj, "contact_offset");
							if (v && yyjson_is_num(v))
								mat.contact_offset = static_cast<float>(yyjson_get_num(v));
							v = yyjson_obj_get(mat_obj, "friction_mu");
							if (v && yyjson_is_num(v))
								mat.friction_mu = static_cast<float>(yyjson_get_num(v));
							v = yyjson_obj_get(mat_obj, "thickness");
							if (v && yyjson_is_num(v))
								mat.thickness = static_cast<float>(yyjson_get_num(v));
							v = yyjson_obj_get(mat_obj, "is_shell");
							if (v && yyjson_is_bool(v))
								mat.is_shell = static_cast<bool>(yyjson_get_bool(v));

							v = yyjson_obj_get(mat_obj, "model");
							if (v)
							{
								if (yyjson_is_str(v))
								{
									const char* s = yyjson_get_str(v);
									mat.model = lcs::Material::parse_rigid_model(s);
								}
							}

							info.set_physics_material(mat);
							if (stype == nullptr)
							{
								info.set_material_type(lcs::Material::MaterialType::Rigid);
							}
							else
							{
								auto mt = material_to_shell(material_type);
								LUISA_ASSERT(mt == info.material_type && "material_type and material.type mismatch");
							}
						}
						else if (material_type == "Rod")
						{
							lcs::Material::RodMaterial mat;
							yyjson_val*				   v = nullptr;
							v = yyjson_obj_get(mat_obj, "radius");
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
							v = yyjson_obj_get(mat_obj, "contact_offset");
							if (v && yyjson_is_num(v))
								mat.contact_offset = static_cast<float>(yyjson_get_num(v));
							v = yyjson_obj_get(mat_obj, "radius");
							if (v && yyjson_is_num(v))
								mat.radius = static_cast<float>(yyjson_get_num(v));
							v = yyjson_obj_get(mat_obj, "friction_mu");
							if (v && yyjson_is_num(v))
								mat.friction_mu = static_cast<float>(yyjson_get_num(v));

							v = yyjson_obj_get(mat_obj, "model");
							if (v)
							{
								if (yyjson_is_str(v))
								{
									const char* s = yyjson_get_str(v);
									mat.model = lcs::Material::parse_rod_model(s);
								}
							}

							info.set_physics_material(mat);
							if (stype == nullptr)
							{
								info.set_material_type(lcs::Material::MaterialType::Rod);
							}
							else
							{
								auto mt = material_to_shell(material_type);
								LUISA_ASSERT(mt == info.material_type && "material_type and material.type mismatch");
							}
						}
						else
						{
							// fallback: treat as cloth
							lcs::Material::ClothMaterial mat;
							yyjson_val*					 v = yyjson_obj_get(mat_obj, "thickness");
							if (v && yyjson_is_num(v))
								mat.thickness = static_cast<float>(yyjson_get_num(v));
							v = yyjson_obj_get(mat_obj, "d_hat");
							if (v && yyjson_is_num(v))
								mat.d_hat = static_cast<float>(yyjson_get_num(v));
							v = yyjson_obj_get(mat_obj, "contact_offset");
							if (v && yyjson_is_num(v))
								mat.contact_offset = static_cast<float>(yyjson_get_num(v));
							info.set_physics_material(mat);
							if (stype == nullptr)
								info.set_material_type(lcs::Material::MaterialType::Cloth);
						}
					}
				}

				// fixed_points
				yyjson_val* fp_arr = yyjson_obj_get(shell_val, "fixed_points");
				if (fp_arr && yyjson_is_arr(fp_arr))
				{
					yyjson_val* fpv;
					size_t		j, jm;
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
						if (rng && yyjson_is_num(rng))
							mfp.range = static_cast<float>(yyjson_get_num(rng));

						// fixed_info (animation)
						yyjson_val* fin = yyjson_obj_get(fpv, "fixed_info");
						if (fin && yyjson_is_obj(fin))
						{
							auto&		fi = mfp.fixed_info;
							yyjson_val* b;
							b = yyjson_obj_get(fin, "use_translate");
							if (b && yyjson_is_bool(b))
								fi.use_translate = yyjson_get_bool(b);
							b = yyjson_obj_get(fin, "translate");
							if (b && yyjson_is_arr(b) && yyjson_get_len(b) >= 3)
							{
								yyjson_val* e;
								size_t		idx, max;
								float		tv2[3] = { 0 };
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
								size_t		idx, max;
								float		rv2[3] = { 0 };
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
								size_t		idx, max;
								float		av[3] = { 0 };
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
								size_t		idx, max;
								float		sv2[3] = { 1.f, 1.f, 1.f };
								yyjson_arr_foreach(b, idx, max, e)
								{
									if (idx < 3 && yyjson_is_num(e))
										sv2[idx] = static_cast<float>(yyjson_get_num(e));
								}
								fi.scale = luisa::make_float3(sv2[0], sv2[1], sv2[2]);
								fi.use_scale = true;
							}
							b = yyjson_obj_get(fin, "use_setting_position");
							if (b && yyjson_is_bool(b))
								fi.use_setting_position = yyjson_get_bool(b);
							b = yyjson_obj_get(fin, "setting_position");
							if (b && yyjson_is_arr(b) && yyjson_get_len(b) >= 3)
							{
								yyjson_val* e;
								size_t		idx, max;
								float		sv3[3] = { 0 };
								yyjson_arr_foreach(b, idx, max, e)
								{
									if (idx < 3 && yyjson_is_num(e))
										sv3[idx] = static_cast<float>(yyjson_get_num(e));
								}
								fi.setting_position = luisa::make_float3(sv3[0], sv3[1], sv3[2]);
								fi.use_setting_position = true;
							}
						}
						info.add_fixed_point_from_method(mfp);
					}
				}
				fn_register_mesh(info);
			}
		}

		yyjson_doc_free(doc);
	}

} // namespace Demo::Simulation