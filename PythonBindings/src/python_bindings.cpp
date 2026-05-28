#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <array>
#include <memory>
#include <string>
#include <string_view>

#include <luisa/luisa-compute.h>

#include "SimulationCore/world_data.h"
#include "SimulationSolver/newton_solver.h"
#include "SimulationCore/scene_params.h"
#include "app_simulation_demo_config.h"

namespace py = pybind11;
using namespace lcs;
using namespace lcs::Initializer;
using namespace lcs::Material;

// Helper wrapper to hold WorldData pointer and expose chainable methods
struct WorldDataWrapper
{
	std::shared_ptr<WorldData> wd;
	WorldDataWrapper(std::shared_ptr<WorldData> w)
		: wd(std::move(w))
	{
	}

	WorldDataWrapper& set_name(const std::string_view& name)
	{
		wd->set_name(name);
		return *this;
	}
	WorldDataWrapper& set_simulation_type(lcs::Material::MaterialType t)
	{
		wd->set_material_type(t);
		return *this;
	}

	// Expose cloth material setter by accepting keyword args
	WorldDataWrapper& set_physics_material_cloth(
		const std::string_view& stretch_model = cloth_stretch_model_to_string(ClothMaterial::default_stretch_model()),
		const std::string_view& bending_model = cloth_bending_model_to_string(ClothMaterial::default_bending_model()),
		float					thickness = ClothMaterial::default_thickness(),
		float					youngs_modulus = ClothMaterial::default_youngs_modulus(),
		float					poisson_ratio = ClothMaterial::default_poisson_ratio(),
		float					area_bending_stiffness = ClothMaterial::default_area_bending_stiffness(),
		float					d_hat = ClothMaterial::default_d_hat(),
		float					contact_offset = ClothMaterial::default_contact_offset())
	{
		wd->set_material_type(lcs::Material::MaterialType::Cloth);
		ClothMaterial mat;
		mat.stretch_model = parse_cloth_stretch_model(stretch_model);
		mat.bending_model = parse_cloth_bending_model(bending_model);
		mat.thickness = thickness;
		mat.youngs_modulus = youngs_modulus;
		mat.poisson_ratio = poisson_ratio;
		mat.area_bending_stiffness = area_bending_stiffness;
		mat.d_hat = d_hat;
		mat.contact_offset = contact_offset;
		wd->set_physics_material(mat);
		return *this;
	}

	// Expose tetrahedral material setter
	WorldDataWrapper& set_physics_material_tet(
		const std::string_view& model = tet_model_to_string(TetMaterial::default_model()),
		float					youngs_modulus = TetMaterial::default_youngs_modulus(),
		float					poisson_ratio = TetMaterial::default_poisson_ratio(),
		float					density = TetMaterial::default_density(),
		float					mass = TetMaterial::default_mass(),
		float					d_hat = ClothMaterial::default_d_hat(),
		float					contact_offset = ClothMaterial::default_contact_offset())
	{
		wd->set_material_type(lcs::Material::MaterialType::Tetrahedral);
		TetMaterial mat;
		mat.model = parse_tet_model(model);
		mat.youngs_modulus = youngs_modulus;
		mat.poisson_ratio = poisson_ratio;
		mat.density = density;
		mat.mass = mass;
		mat.is_shell = false;
		mat.d_hat = d_hat;
		mat.contact_offset = contact_offset;
		wd->set_physics_material(mat);
		return *this;
	}

	// Expose rigid material setter
	WorldDataWrapper& set_physics_material_rigid(
		const std::string_view& model = rigid_model_to_string(RigidMaterial::default_model()),
		float					thickness = RigidMaterial::default_thickness(),
		float					stiffness = RigidMaterial::default_stiffness(),
		float					density = RigidMaterial::default_density(),
		float					mass = RigidMaterial::default_mass(),
		float					d_hat = ClothMaterial::default_d_hat(),
		float					contact_offset = ClothMaterial::default_contact_offset())
	{
		wd->set_material_type(lcs::Material::MaterialType::Rigid);
		RigidMaterial mat;
		mat.model = parse_rigid_model(model);
		mat.thickness = thickness;
		mat.stiffness = stiffness;
		mat.density = density;
		mat.mass = mass;
		mat.d_hat = d_hat;
		mat.contact_offset = contact_offset;
		wd->set_physics_material(mat);
		return *this;
	}

	// Expose rod material setter
	WorldDataWrapper& set_physics_material_rod(
		const std::string_view& model = rod_model_to_string(RodMaterial::default_model()),
		float					radius = RodMaterial::default_radius(),
		float					bending_stiffness = RodMaterial::default_bending_stiffness(),
		float					twisting_stiffness = RodMaterial::default_twisting_stiffness(),
		float					density = RodMaterial::default_density(),
		float					mass = RodMaterial::default_mass(),
		float					d_hat = ClothMaterial::default_d_hat(),
		float					contact_offset = ClothMaterial::default_contact_offset())
	{
		wd->set_material_type(lcs::Material::MaterialType::Rod);
		RodMaterial mat;
		mat.model = parse_rod_model(model);
		mat.radius = radius;
		mat.bending_stiffness = bending_stiffness;
		mat.twisting_stiffness = twisting_stiffness;
		mat.density = density;
		mat.mass = mass;
		wd->set_physics_material(mat);
		return *this;
	}

	// Convenience: add fixed-point rule by name and optional numeric range/list
	WorldDataWrapper& add_fixed_point_by_method(const std::string_view& method, float range)
	{
		MakeFixedPointsInterface mfp;
		mfp.method = parse_fixed_method_py(method);
		mfp.range = range;

		wd->add_fixed_point_from_method(mfp);
		return *this;
	}

	// Convenience: add explicit vertex indices as fixed points
	WorldDataWrapper& add_fixed_point_by_indices(py::array_t<int, py::array::c_style | py::array::forcecast> indices)
	{
		if (indices.ndim() != 1)
			throw std::runtime_error("indices must be a 1-D array of ints");
		auto			  buf = indices.unchecked<1>();
		const size_t	  n = indices.shape(0);
		std::vector<uint> idx_vec(n);
		for (size_t i = 0; i < n; ++i)
		{
			int idx = buf(i);
			if (idx < 0)
				throw std::runtime_error("indices must be non-negative");
			idx_vec[i] = static_cast<uint>(idx);
		}
		wd->add_fixed_point_from_indices(idx_vec);
		return *this;
	}

	WorldDataWrapper& set_translation(float x, float y, float z)
	{
		wd->set_translation(x, y, z);
		return *this;
	}

	WorldDataWrapper& set_rotation(float x, float y, float z)
	{
		wd->set_rotation(x, y, z);
		return *this;
	}

	WorldDataWrapper& set_scale(float s)
	{
		wd->set_scale(s);
		return *this;
	}

	std::array<float, 3> get_rest_translation() const
	{
		auto t = wd->translation;
		return std::array<float, 3>{ t.x, t.y, t.z };
	}

	std::array<float, 3> get_rest_rotation() const
	{
		auto r = wd->rotation;
		return std::array<float, 3>{ r.x, r.y, r.z };
	}

	std::array<float, 3> get_rest_scale() const
	{
		auto s = wd->scale;
		return std::array<float, 3>{ s.x, s.y, s.z };
	}

	std::string get_name() const
	{
		return wd->get_model_name();
	}
	uint get_registration_index() const
	{
		return wd->get_registration_index();
	}

	py::list get_fixed_point_indices() const
	{
		const auto& indices = wd->fixed_point_indices;
		py::list	out;
		for (auto idx : indices)
			out.append(static_cast<uint32_t>(idx));
		return out;
	}

	py::array_t<float> get_rest_positions() const
	{
		const auto		   rest = wd->get_rest_positions();
		py::array_t<float> out({ rest.size(), static_cast<size_t>(3) });
		auto			   buf = out.mutable_unchecked<2>();
		for (size_t i = 0; i < rest.size(); ++i)
		{
			buf(i, 0) = rest[i][0];
			buf(i, 1) = rest[i][1];
			buf(i, 2) = rest[i][2];
		}
		return out;
	}
};

// Read-only wrapper for APIs that return const WorldData&.
struct ConstWorldDataWrapper
{
	const WorldData* wd;
	ConstWorldDataWrapper(const WorldData* w)
		: wd(w)
	{
	}

	std::string get_name() const
	{
		return wd->get_model_name();
	}
	uint get_registration_index() const
	{
		return wd->get_registration_index();
	}

	py::list get_fixed_point_indices() const
	{
		const auto& indices = wd->fixed_point_indices;
		py::list	out;
		for (auto idx : indices)
			out.append(static_cast<uint32_t>(idx));
		return out;
	}

	py::array_t<float> get_rest_positions() const
	{
		const auto		   rest = wd->get_rest_positions();
		py::array_t<float> out({ rest.size(), static_cast<size_t>(3) });
		auto			   buf = out.mutable_unchecked<2>();
		for (size_t i = 0; i < rest.size(); ++i)
		{
			buf(i, 0) = rest[i][0];
			buf(i, 1) = rest[i][1];
			buf(i, 2) = rest[i][2];
		}
		return out;
	}

	std::array<float, 3> get_rest_translation() const
	{
		auto t = wd->translation;
		return std::array<float, 3>{ t.x, t.y, t.z };
	}

	std::array<float, 3> get_rest_rotation() const
	{
		auto r = wd->rotation;
		return std::array<float, 3>{ r.x, r.y, r.z };
	}

	std::array<float, 3> get_rest_scale() const
	{
		auto s = wd->scale;
		return std::array<float, 3>{ s.x, s.y, s.z };
	}
};

// Python-facing Newton-like builder that stores a vector<WorldData>
struct PyNewtonBuilder
{
	std::unique_ptr<lcs::NewtonSolver> solver_ptr;

	PyNewtonBuilder()
		: solver_ptr(std::make_unique<lcs::NewtonSolver>())
	{
	}

	// register_mesh accepts numpy arrays (vertices Nx3, triangles Mx3)
	WorldDataWrapper create_world_data_from_array(const std::string_view& name,
		py::array_t<double, py::array::c_style | py::array::forcecast>	  vertices,
		py::array_t<int, py::array::c_style | py::array::forcecast>		  triangles)
	{
		// Validate shapes
		if (vertices.ndim() != 2 || vertices.shape(1) != 3)
			throw std::runtime_error("vertices must be a (N,3) array of floats");
		if (triangles.ndim() != 2 || triangles.shape(1) != 3)
			throw std::runtime_error("triangles must be a (M,3) array of ints");

		using InputVertexType = std::array<float, 3>;
		using InputFaceType = std::array<uint32_t, 3>;

		const size_t nverts = vertices.shape(0);
		const size_t nfaces = triangles.shape(0);

		std::vector<InputVertexType> input_vertices(nverts);
		std::vector<InputFaceType>	 input_triangles(nfaces);

		auto buf_v = vertices.unchecked<2>();
		auto buf_t = triangles.unchecked<2>();
		for (size_t i = 0; i < nverts; ++i)
		{
			InputVertexType p;
			p[0] = static_cast<float>(buf_v(i, 0));
			p[1] = static_cast<float>(buf_v(i, 1));
			p[2] = static_cast<float>(buf_v(i, 2));
			input_vertices[i] = p;
		}
		for (size_t i = 0; i < nfaces; ++i)
		{
			InputFaceType f;
			f[0] = static_cast<uint32_t>(buf_t(i, 0));
			f[1] = static_cast<uint32_t>(buf_t(i, 1));
			f[2] = static_cast<uint32_t>(buf_t(i, 2));
			input_triangles[i] = f;
		}

		auto world_data = std::make_shared<WorldData>();
		world_data->set_name(name);
		world_data->load_mesh_from_array(input_vertices, input_triangles);
		return WorldDataWrapper(world_data);
	}
	// register_tet_mesh accepts numpy arrays (vertices Nx3, tets Mx4)
	WorldDataWrapper create_world_data_from_tet_array(const std::string_view& name,
		py::array_t<double, py::array::c_style | py::array::forcecast>		  vertices,
		py::array_t<int, py::array::c_style | py::array::forcecast>			  tets)
	{
		// Validate shapes
		if (vertices.ndim() != 2 || vertices.shape(1) != 3)
			throw std::runtime_error("vertices must be a (N,3) array of floats");
		if (tets.ndim() != 2 || tets.shape(1) != 4)
			throw std::runtime_error("tets must be a (M,4) array of ints");

		using InputVertexType = std::array<float, 3>;
		using InputTetType = std::array<uint32_t, 4>;

		const size_t nverts = vertices.shape(0);
		const size_t ntets = tets.shape(0);

		std::vector<InputVertexType> input_vertices(nverts);
		std::vector<InputTetType>	 input_tets(ntets);

		auto buf_v = vertices.unchecked<2>();
		auto buf_t = tets.unchecked<2>();
		for (size_t i = 0; i < nverts; ++i)
		{
			InputVertexType p;
			p[0] = static_cast<float>(buf_v(i, 0));
			p[1] = static_cast<float>(buf_v(i, 1));
			p[2] = static_cast<float>(buf_v(i, 2));
			input_vertices[i] = p;
		}
		for (size_t i = 0; i < ntets; ++i)
		{
			InputTetType t;
			t[0] = static_cast<uint32_t>(buf_t(i, 0));
			t[1] = static_cast<uint32_t>(buf_t(i, 1));
			t[2] = static_cast<uint32_t>(buf_t(i, 2));
			t[3] = static_cast<uint32_t>(buf_t(i, 3));
			input_tets[i] = t;
		}

		auto world_data = std::make_shared<WorldData>();
		world_data->set_name(name);
		world_data->load_tet_mesh_from_array(input_vertices, input_tets);
		// Automatically mark as Tetrahedral type; user can still call set_physics_material_tet()
		// to override the material parameters before registering.
		world_data->set_material_type(lcs::Material::MaterialType::Tetrahedral);
		return WorldDataWrapper(world_data);
	}

	// create world data from an obj file path; call register_world_data() to add into solver
	WorldDataWrapper create_world_data_from_file_path(const std::string_view& name, const std::string_view& obj_file_path)
	{
		auto world_data = std::make_shared<WorldData>();
		world_data->set_name(name);
		world_data->load_mesh_from_path(obj_file_path);
		return WorldDataWrapper(world_data);
	}

	uint register_world_data(const WorldDataWrapper& world_data)
	{
		if (!world_data.wd)
			throw std::runtime_error("Invalid world data handle.");
		return solver_ptr->register_world_data(*world_data.wd);
	}

	// expose method to get number of registered meshes
	size_t num_meshes() const { return solver_ptr->get_sorted_world_data().size(); }

	// expose a method to export registered meshes as python lists (simple)
	py::list get_mesh_names() const
	{
		py::list				 out;
		const auto&				 world_data = solver_ptr->get_sorted_world_data();
		std::vector<std::string> names(world_data.size());
		// for (const auto& w : world_data)
		// {
		// 	names[w.get_registration_index()] = w.get_model_name();
		// }
		for (uint registration_id = 0; registration_id < world_data.size(); ++registration_id)
		{
			const uint	sorted_idx = solver_ptr->query_sorted_index_by_registration_id(registration_id);
			const auto& w = world_data[sorted_idx];
			names[registration_id] = w.get_model_name();
		}
		for (const auto& name : names)
			out.append(name);
		return out;
	}

	// Debug helper to print registered meshes info in C++ logs
	void print_registered_meshes_info() const
	{
		auto& world_data = solver_ptr->get_sorted_world_data();
		for (auto& w : world_data)
		{
			auto& mesh = w.get_mesh();
			LUISA_INFO("Mesh '{}': registration_id={}, num_verts={}, num_faces={}",
				w.get_model_name(), w.get_registration_index(), mesh.model_positions.size(), mesh.faces.size());
		}
	}

	// Initialize underlying NewtonSolver using the device previously set via init_device()/set_device().
	void init_solver()
	{
		solver_ptr->init_solver();
		LUISA_INFO("Solver initialized.");
	}

	// Load a full scene from JSON, including world_data and scene params.
	// This should be called before init_solver().
	void load_scene_from_json(const std::string_view& json_path)
	{
		Demo::Simulation::load_scene_params_from_json(
			[&](const lcs::Initializer::WorldData& wd)
			{
				solver_ptr->register_world_data(wd);
			},
			std::string(json_path));
	}

	void physics_step_cpu()
	{
		solver_ptr->physics_step_CPU();
	}

	void physics_step_gpu()
	{
		solver_ptr->physics_step_GPU();
	}

	void restart_system()
	{
		solver_ptr->lcs::SolverInterface::restart_system();
	}

	// Update a pinned vertex position on the solver (mesh local vertex id, target position)
	void update_per_vertex_animation(const unsigned int				  mesh_idx,
		const unsigned int											  local_vid,
		py::array_t<float, py::array::c_style | py::array::forcecast> target_pos)
	{
		if (target_pos.ndim() != 1 || target_pos.shape(0) != 3)
			throw std::runtime_error("target_pos must be a 1-D array of length 3 (x,y,z)");

		auto				 buf = target_pos.unchecked<1>();
		std::array<float, 3> tp{ buf(0), buf(1), buf(2) };
		solver_ptr->update_per_vertex_animation(mesh_idx, local_vid, tp);
	}

	// Update a pinned body state on the solver (body id, target translation and rotation)
	void update_per_body_animation(const unsigned int				  mesh_idx,
		py::array_t<float, py::array::c_style | py::array::forcecast> target_translation,
		py::array_t<float, py::array::c_style | py::array::forcecast> target_rotation)
	{
		if (target_translation.ndim() != 1 || target_translation.shape(0) != 3)
			throw std::runtime_error("target_translation must be a 1-D array of length 3 (x,y,z)");
		if (target_rotation.ndim() != 1 || target_rotation.shape(0) != 3)
			throw std::runtime_error("target_rotation must be a 1-D array of length 3 (x,y,z)");

		auto				 buf_t = target_translation.unchecked<1>();
		std::array<float, 3> tt{ buf_t(0), buf_t(1), buf_t(2) };
		auto				 buf_r = target_rotation.unchecked<1>();
		std::array<float, 3> tr{ buf_r(0), buf_r(1), buf_r(2) };
		solver_ptr->update_per_body_animation(mesh_idx, tt, tr);
	}

	std::array<float, 3> get_rigid_body_translation(uint registration_id) const
	{
		return solver_ptr->get_rigid_body_translation(registration_id);
	}

	std::array<float, 3> get_rigid_body_scaling(uint registration_id) const
	{
		return solver_ptr->get_rigid_body_scaling(registration_id);
	}

	std::array<float, 4> get_rigid_body_rotation_quaternion(uint registration_id) const
	{
		return solver_ptr->get_rigid_body_rotation_quaternion(registration_id);
	}

	std::array<float, 3> get_rigid_body_rotation_axis_angle(uint registration_id) const
	{
		return solver_ptr->get_rigid_body_rotation_axis_angle(registration_id);
	}

	void add_fixed_joint(const unsigned int							  body_a_registration,
		const unsigned int											  body_b_registration,
		py::array_t<float, py::array::c_style | py::array::forcecast> anchor_a_local,
		py::array_t<float, py::array::c_style | py::array::forcecast> anchor_b_local,
		float														  stiffness_pos,
		float														  stiffness_rot)
	{
		if (anchor_a_local.ndim() != 1 || anchor_a_local.shape(0) != 3 || anchor_b_local.ndim() != 1 || anchor_b_local.shape(0) != 3)
			throw std::runtime_error("anchor_a_local/anchor_b_local must be 1-D arrays of length 3.");
		auto a = anchor_a_local.unchecked<1>();
		auto b = anchor_b_local.unchecked<1>();

		FixedJointConstraintDesc desc;
		desc.body_a_registration = body_a_registration;
		desc.body_b_registration = body_b_registration;
		desc.anchor_a_local = luisa::make_float3(a(0), a(1), a(2));
		desc.anchor_b_local = luisa::make_float3(b(0), b(1), b(2));
		desc.stiffness_pos = stiffness_pos;
		desc.stiffness_rot = stiffness_rot;
		solver_ptr->add_fixed_joint(desc);
	}

	void add_prismatic_joint(const unsigned int						  body_a_registration,
		const unsigned int											  body_b_registration,
		py::array_t<float, py::array::c_style | py::array::forcecast> anchor_a_local,
		py::array_t<float, py::array::c_style | py::array::forcecast> anchor_b_local,
		py::array_t<float, py::array::c_style | py::array::forcecast> axis_world,
		float														  stiffness_pos,
		float														  stiffness_rot,
		float														  slide_min,
		float														  slide_max)
	{
		if (anchor_a_local.ndim() != 1 || anchor_a_local.shape(0) != 3 || anchor_b_local.ndim() != 1 || anchor_b_local.shape(0) != 3 || axis_world.ndim() != 1 || axis_world.shape(0) != 3)
			throw std::runtime_error("anchor_a_local/anchor_b_local/axis_world must be 1-D arrays of length 3.");
		auto a = anchor_a_local.unchecked<1>();
		auto b = anchor_b_local.unchecked<1>();
		auto w = axis_world.unchecked<1>();

		PrismaticJointConstraintDesc desc;
		desc.body_a_registration = body_a_registration;
		desc.body_b_registration = body_b_registration;
		desc.anchor_a_local = luisa::make_float3(a(0), a(1), a(2));
		desc.anchor_b_local = luisa::make_float3(b(0), b(1), b(2));
		desc.axis_world = luisa::make_float3(w(0), w(1), w(2));
		// desc.axis_world = luisa::normalize(desc.axis_world);
		desc.stiffness_pos = stiffness_pos;
		desc.stiffness_rot = stiffness_rot;
		desc.slide_min = slide_min;
		desc.slide_max = slide_max;
		solver_ptr->add_prismatic_joint(desc);
	}

	void add_revolute_joint(const unsigned int						  body_a_registration,
		const unsigned int											  body_b_registration,
		py::array_t<float, py::array::c_style | py::array::forcecast> anchor_a_local,
		py::array_t<float, py::array::c_style | py::array::forcecast> anchor_b_local,
		py::array_t<float, py::array::c_style | py::array::forcecast> axis_world,
		py::array_t<float, py::array::c_style | py::array::forcecast> axis_a_local,
		py::array_t<float, py::array::c_style | py::array::forcecast> axis_b_local,
		float														  stiffness_pos,
		float														  stiffness_axis)
	{
		if (anchor_a_local.ndim() != 1 || anchor_a_local.shape(0) != 3 || anchor_b_local.ndim() != 1 || anchor_b_local.shape(0) != 3 || axis_world.ndim() != 1 || axis_world.shape(0) != 3 || axis_a_local.ndim() != 1 || axis_a_local.shape(0) != 3 || axis_b_local.ndim() != 1 || axis_b_local.shape(0) != 3)
			throw std::runtime_error("All joint vectors must be 1-D arrays of length 3.");
		auto a = anchor_a_local.unchecked<1>();
		auto b = anchor_b_local.unchecked<1>();
		auto w = axis_world.unchecked<1>();
		auto ua = axis_a_local.unchecked<1>();
		auto ub = axis_b_local.unchecked<1>();

		RevoluteJointConstraintDesc desc;
		desc.body_a_registration = body_a_registration;
		desc.body_b_registration = body_b_registration;
		desc.anchor_a_local = luisa::make_float3(a(0), a(1), a(2));
		desc.anchor_b_local = luisa::make_float3(b(0), b(1), b(2));
		desc.axis_world = luisa::make_float3(w(0), w(1), w(2));
		// desc.axis_world = luisa::normalize(desc.axis_world);
		desc.axis_a_local = luisa::make_float3(ua(0), ua(1), ua(2));
		desc.axis_b_local = luisa::make_float3(ub(0), ub(1), ub(2));
		desc.stiffness_pos = stiffness_pos;
		desc.stiffness_axis = stiffness_axis;
		solver_ptr->add_revolute_joint(desc);
	}

	// Return simulation results as a tuple of (vertices_list, faces_list) of numpy arrays.
	// Uses memcpy for efficient data transfer.
	py::tuple get_sim_result()
	{
		const uint num_meshes = solver_ptr->get_host_mesh_data().num_meshes;

		std::vector<std::vector<std::array<float, 3>>> sa_rendering_vertices(num_meshes);
		std::vector<std::vector<std::array<uint, 3>>>  sa_rendering_triangles(num_meshes);
		solver_ptr->get_curr_vertices_to_host(sa_rendering_vertices);
		solver_ptr->get_triangles_to_host(sa_rendering_triangles);

		py::list py_verts;
		py::list py_faces;
		for (uint i = 0; i < num_meshes; ++i)
		{
			// vertices – contiguous std::array<float,3>, safe to memcpy
			const auto&		   mesh_verts = sa_rendering_vertices[i];
			py::array_t<float> v_arr({ (size_t)mesh_verts.size(), (size_t)3 });
			std::memcpy(v_arr.mutable_data(), mesh_verts.data(), mesh_verts.size() * 3 * sizeof(float));
			py_verts.append(v_arr);

			// faces
			const auto&			  mesh_faces = sa_rendering_triangles[i];
			py::array_t<uint32_t> f_arr({ (size_t)mesh_faces.size(), (size_t)3 });
			std::memcpy(f_arr.mutable_data(), mesh_faces.data(), mesh_faces.size() * 3 * sizeof(uint32_t));
			py_faces.append(f_arr);
		}
		return py::make_tuple(py_verts, py_faces);
	}

	uint query_local_vid_from_global_vid(const uint global_vid) const
	{
		const auto& mapping = solver_ptr->get_host_mesh_data().sa_global_vid_to_local_vid;
		return (global_vid < mapping.size()) ? mapping[global_vid] : std::numeric_limits<uint>::max();
	}

	uint query_registration_vid_from_global_vid(const uint global_vid) const
	{
		const std::vector<uint>& mapping1 = solver_ptr->get_host_mesh_data().sa_vert_mesh_id;
		const uint				 sorted_idx = (global_vid < mapping1.size()) ? mapping1[global_vid] : std::numeric_limits<uint>::max();
		const uint				 registration_id = solver_ptr->query_registration_id_by_sorted_index(sorted_idx);
		return registration_id;
	}

	py::tuple get_object_sim_result_by_registration_id(uint registration_id)
	{
		std::vector<std::array<float, 3>> object_vertices;
		std::vector<std::array<uint, 3>>  object_triangles;
		solver_ptr->get_object_sim_result_by_registration_id(registration_id, object_vertices, object_triangles);

		py::array_t<float> v_arr({ (size_t)object_vertices.size(), (size_t)3 });
		if (!object_vertices.empty())
			std::memcpy(v_arr.mutable_data(), object_vertices.data(), object_vertices.size() * 3 * sizeof(float));

		py::array_t<uint32_t> f_arr({ (size_t)object_triangles.size(), (size_t)3 });
		if (!object_triangles.empty())
			std::memcpy(f_arr.mutable_data(), object_triangles.data(), object_triangles.size() * 3 * sizeof(uint32_t));

		return py::make_tuple(v_arr, f_arr);
	}

	ConstWorldDataWrapper get_object_by_registration_id(uint registration_id) const
	{
		return ConstWorldDataWrapper(&solver_ptr->get_object_by_registration_id(registration_id));
	}

	void save_sim_result(const std::string_view& full_path)
	{
		solver_ptr->save_mesh_to_obj(full_path);
	}

	float get_vert_mass(uint global_vid) const
	{
		const auto& mesh_data = solver_ptr->get_host_mesh_data();
		const auto& src_masses = mesh_data.sa_vert_mass;

		return (global_vid < src_masses.size()) ? src_masses[global_vid] : 0.f;
	}

	// ---------------------------------------------------------------------------
	// Device management (mirrors lcs::SolverInterface device methods)
	// ---------------------------------------------------------------------------

	// Create and own a luisa device/stream.
	// backend_name: e.g. "metal", "cuda", "dx". None = platform default.
	// binary_path : path used by luisa::compute::Context. None = auto-detect from lcs_py module file.
	void init_device(py::object backend_name_obj = py::none(), py::object binary_path_obj = py::none())
	{
		// Resolve binary path
		std::string binary_path;
		if (!binary_path_obj.is_none())
		{
			binary_path = binary_path_obj.cast<std::string>();
		}
		else
		{
			try
			{
				py::module_ self = py::module_::import("lcs_py");
				if (py::hasattr(self, "__file__"))
				{
					binary_path = self.attr("__file__").cast<std::string>();
					auto parent = std::filesystem::path(binary_path).parent_path();
					if (std::filesystem::exists(parent / "bin"))
						binary_path = (parent / "bin").string();
					else
						binary_path = parent.string();
				}
			}
			catch (...)
			{
			}
		}

		// Resolve backend name
		std::string backend;
		if (!backend_name_obj.is_none())
			backend = backend_name_obj.cast<std::string>();

		solver_ptr->create_device(binary_path, backend);
	}

	// Borrow an external device/stream (non-owning). The caller must ensure they outlive this solver.
	void set_device(uintptr_t device_ptr, uintptr_t stream_ptr)
	{
		solver_ptr->set_device_from_pointers(device_ptr, stream_ptr);
	}

	// Release owned device resources.
	void cleanup_device()
	{
		solver_ptr->cleanup_device();
	}

	// Return raw pointer (as int) to the active luisa::compute::Device.
	uintptr_t get_device_ptr() const
	{
		return solver_ptr->get_device_ptr();
	}

	// Return raw pointer (as int) to the active luisa::compute::Stream.
	uintptr_t get_stream_ptr() const
	{
		return solver_ptr->get_stream_ptr();
	}

	lcs::SceneParams& get_config() const
	{
		return solver_ptr->get_config();
	}
};

PYBIND11_MODULE(lcs_py, m)
{
	py::enum_<lcs::Material::MaterialType>(m, "MaterialType")
		.value("Particle", lcs::Material::MaterialType::Particle)
		.value("Cloth", lcs::Material::MaterialType::Cloth)
		.value("Tetrahedral", lcs::Material::MaterialType::Tetrahedral)
		.value("Rigid", lcs::Material::MaterialType::Rigid)
		.value("Rod", lcs::Material::MaterialType::Rod)
		.export_values();

	// py::class_<ClothMaterial>(m, "ClothMaterial")
	// 	.def(py::init<>())
	// 	.def_readwrite("thickness", &ClothMaterial::thickness)
	// 	.def_readwrite("youngs_modulus", &ClothMaterial::youngs_modulus)
	// 	.def_readwrite("poisson_ratio", &ClothMaterial::poisson_ratio)
	// 	.def_readwrite("area_bending_stiffness", &ClothMaterial::area_bending_stiffness)
	// 	.def_readwrite("mass", &ClothMaterial::mass)
	// 	.def_readwrite("density", &ClothMaterial::density);

	// py::class_<TetMaterial>(m, "TetMaterial")
	// 	.def(py::init<>())
	// 	.def_readwrite("youngs_modulus", &TetMaterial::youngs_modulus)
	// 	.def_readwrite("poisson_ratio", &TetMaterial::poisson_ratio)
	// 	.def_readwrite("mass", &TetMaterial::mass)
	// 	.def_readwrite("density", &TetMaterial::density)
	// 	.def_readwrite("d_hat", &TetMaterial::d_hat)
	// 	.def_readwrite("friction_mu", &TetMaterial::friction_mu);

	// py::class_<RigidMaterial>(m, "RigidMaterial")
	// 	.def(py::init<>())
	// 	.def_readwrite("thickness", &RigidMaterial::thickness)
	// 	.def_readwrite("stiffness", &RigidMaterial::stiffness)
	// 	.def_readwrite("is_solid", &RigidMaterial::is_solid)
	// 	.def_readwrite("mass", &RigidMaterial::mass)
	// 	.def_readwrite("density", &RigidMaterial::density)
	// 	.def_readwrite("d_hat", &RigidMaterial::d_hat)
	// 	.def_readwrite("friction_mu", &RigidMaterial::friction_mu);

	// py::class_<RodMaterial>(m, "RodMaterial")
	// 	.def(py::init<>())
	// 	.def_readwrite("radius", &RodMaterial::radius)
	// 	.def_readwrite("bending_stiffness", &RodMaterial::bending_stiffness)
	// 	.def_readwrite("twisting_stiffness", &RodMaterial::twisting_stiffness)
	// 	.def_readwrite("mass", &RodMaterial::mass)
	// 	.def_readwrite("density", &RodMaterial::density)
	// 	.def_readwrite("d_hat", &RodMaterial::d_hat)
	// 	.def_readwrite("friction_mu", &RodMaterial::friction_mu);

	// Expose FixedPointsType so MakeFixedPointsInterface.method has a real
	// type in the generated stubs (pybind11-stubgen otherwise prints `...`,
	// which costs editor completion). Names mirror parse_fixed_method_py().
	py::enum_<FixedPointsType>(m, "FixedPointsType")
		.value("None_", FixedPointsType::None)
		.value("FromIndices", FixedPointsType::FromIndices)
		.value("FromFunction", FixedPointsType::FromFunction)
		.value("Left", FixedPointsType::Left)
		.value("Right", FixedPointsType::Right)
		.value("Front", FixedPointsType::Front)
		.value("Back", FixedPointsType::Back)
		.value("Up", FixedPointsType::Up)
		.value("Down", FixedPointsType::Down)
		.value("LeftUp", FixedPointsType::LeftUp)
		.value("LeftDown", FixedPointsType::LeftDown)
		.value("LeftFront", FixedPointsType::LeftFront)
		.value("LeftBack", FixedPointsType::LeftBack)
		.value("RightUp", FixedPointsType::RightUp)
		.value("RightDown", FixedPointsType::RightDown)
		.value("RightFront", FixedPointsType::RightFront)
		.value("RightBack", FixedPointsType::RightBack)
		.value("FrontUp", FixedPointsType::FrontUp)
		.value("FrontDown", FixedPointsType::FrontDown)
		.value("BackUp", FixedPointsType::BackUp)
		.value("BackDown", FixedPointsType::BackDown)
		.value("All", FixedPointsType::All);

	py::class_<MakeFixedPointsInterface>(m, "MakeFixedPointsInterface")
		.def(py::init<>())
		.def_readwrite("method", &MakeFixedPointsInterface::method,
			"Region selector for fixed points (FixedPointsType enum).")
		.def_readwrite("range", &MakeFixedPointsInterface::range,
			"Tolerance used by region-based selectors (e.g. Left/Right).");

	py::class_<WorldDataWrapper>(m, "WorldData")
		.def("set_name", &WorldDataWrapper::set_name, py::arg("name"), "Set the object name used in solver logs and exported results.")
		.def("set_simulation_type", &WorldDataWrapper::set_simulation_type, py::arg("material_type"), "Set the object simulation/material category.")
		.def("set_physics_material_cloth",
			&WorldDataWrapper::set_physics_material_cloth,
			py::arg("stretch_model") = std::string(cloth_stretch_model_to_string(ClothMaterial::default_stretch_model())),
			py::arg("bending_model") = std::string(cloth_bending_model_to_string(ClothMaterial::default_bending_model())),
			py::arg("thickness") = ClothMaterial::default_thickness(),
			py::arg("youngs_modulus") = ClothMaterial::default_youngs_modulus(),
			py::arg("poisson_ratio") = ClothMaterial::default_poisson_ratio(),
			py::arg("area_bending_stiffness") = ClothMaterial::default_area_bending_stiffness(),
			py::arg("d_hat") = ClothMaterial::default_d_hat(),
			py::arg("contact_offset") = ClothMaterial::default_contact_offset())
		.def("set_physics_material_tet",
			&WorldDataWrapper::set_physics_material_tet,
			py::arg("model") = std::string(tet_model_to_string(TetMaterial::default_model())),
			py::arg("youngs_modulus") = TetMaterial::default_youngs_modulus(),
			py::arg("poisson_ratio") = TetMaterial::default_poisson_ratio(),
			py::arg("density") = TetMaterial::default_density(),
			py::arg("mass") = TetMaterial::default_mass(),
			py::arg("d_hat") = ClothMaterial::default_d_hat(),
			py::arg("contact_offset") = ClothMaterial::default_contact_offset())
		.def("set_physics_material_rigid",
			&WorldDataWrapper::set_physics_material_rigid,
			py::arg("model") = std::string(rigid_model_to_string(RigidMaterial::default_model())),
			py::arg("thickness") = RigidMaterial::default_thickness(),
			py::arg("stiffness") = RigidMaterial::default_stiffness(),
			py::arg("density") = RigidMaterial::default_density(),
			py::arg("mass") = RigidMaterial::default_mass(),
			py::arg("d_hat") = ClothMaterial::default_d_hat(),
			py::arg("contact_offset") = ClothMaterial::default_contact_offset())
		.def("set_physics_material_rod",
			&WorldDataWrapper::set_physics_material_rod,
			py::arg("model") = std::string(rod_model_to_string(RodMaterial::default_model())),
			py::arg("radius") = RodMaterial::default_radius(),
			py::arg("bending_stiffness") = RodMaterial::default_bending_stiffness(),
			py::arg("twisting_stiffness") = RodMaterial::default_twisting_stiffness(),
			py::arg("density") = RodMaterial::default_density(),
			py::arg("mass") = RodMaterial::default_mass(),
			py::arg("d_hat") = ClothMaterial::default_d_hat(),
			py::arg("contact_offset") = ClothMaterial::default_contact_offset())
		.def("add_fixed_point_by_method", &WorldDataWrapper::add_fixed_point_by_method, py::arg("method"), py::arg("range") = 0.001f)
		.def("add_fixed_point_by_indices", &WorldDataWrapper::add_fixed_point_by_indices, py::arg("indices"))
		.def("set_translation", &WorldDataWrapper::set_translation, py::arg("x"), py::arg("y"), py::arg("z"), "Set rest translation in world coordinates.")
		.def("set_rotation", &WorldDataWrapper::set_rotation, py::arg("x"), py::arg("y"), py::arg("z"), "Set rest Euler rotation in radians.")
		.def("set_scale", &WorldDataWrapper::set_scale, py::arg("scale"), "Set uniform rest scale.")
		.def("get_rest_translation", &WorldDataWrapper::get_rest_translation, "Return rest translation as [x, y, z].")
		.def("get_rest_rotation", &WorldDataWrapper::get_rest_rotation, "Return rest Euler rotation as [x, y, z].")
		.def("get_rest_scale", &WorldDataWrapper::get_rest_scale, "Return rest scale as [x, y, z].")
		.def("get_name", &WorldDataWrapper::get_name, "Return object name.")
		.def("get_id", &WorldDataWrapper::get_registration_index, "Return object registration id.")
		.def("get_registration_index", &WorldDataWrapper::get_registration_index, "Return object registration id.")
		.def("get_fixed_point_indices", &WorldDataWrapper::get_fixed_point_indices,
			"Return currently registered fixed-point local vertex indices as a Python list")
		.def("get_rest_positions", &WorldDataWrapper::get_rest_positions,
			"Return rest positions (after object transform) as an (N,3) float32 numpy array");

	py::class_<ConstWorldDataWrapper>(m, "ConstWorldData")
		.def("get_name", &ConstWorldDataWrapper::get_name, "Return object name.")
		.def("get_registration_index", &ConstWorldDataWrapper::get_registration_index, "Return object registration id.")
		.def("get_fixed_point_indices", &ConstWorldDataWrapper::get_fixed_point_indices,
			"Return currently registered fixed-point local vertex indices as a Python list")
		.def("get_rest_positions", &ConstWorldDataWrapper::get_rest_positions,
			"Return rest positions (after object transform) as an (N,3) float32 numpy array")
		.def("get_rest_translation", &ConstWorldDataWrapper::get_rest_translation, "Return rest translation as [x, y, z].")
		.def("get_rest_rotation", &ConstWorldDataWrapper::get_rest_rotation, "Return rest Euler rotation as [x, y, z].")
		.def("get_rest_scale", &ConstWorldDataWrapper::get_rest_scale, "Return rest scale as [x, y, z].");

	// disambiguate overloaded world_data creation signatures
	using VertArr = py::array_t<double, py::array::c_style | py::array::forcecast>;
	using TriArr = py::array_t<int, py::array::c_style | py::array::forcecast>;

	py::class_<PyNewtonBuilder>(m, "NewtonSolver")
		.def(py::init<>())
		.def("create_world_data_from_array", &PyNewtonBuilder::create_world_data_from_array, py::arg("name"), py::arg("vertices"), py::arg("triangles"))
		.def("create_world_data_from_tet_array",
			&PyNewtonBuilder::create_world_data_from_tet_array,
			py::arg("name"),
			py::arg("vertices"),
			py::arg("tets"),
			"Create a tetrahedral-mesh WorldData from numpy arrays.\n\n"
			"vertices: (N,3) float array of rest-pose positions\n"
			"tets:     (M,4) int array of tetrahedron vertex indices\n"
			"Surface topology (faces, edges, surface_verts) is extracted automatically.\n"
			"Call set_physics_material_tet() on the returned object to set material params,\n"
			"then register_world_data() to add it to the solver.")
		.def("create_world_data_from_file_path", &PyNewtonBuilder::create_world_data_from_file_path, py::arg("name"), py::arg("obj_file_path"))
		.def("register_world_data", &PyNewtonBuilder::register_world_data, py::arg("world_data"), "Register configured WorldData and return object registration id")
		.def("load_scene_from_json",
			&PyNewtonBuilder::load_scene_from_json,
			py::arg("json_path"),
			"Load world_data and scene params from a JSON scene file (same format as app_simulation).")
		.def("num_meshes", &PyNewtonBuilder::num_meshes, "Return the number of registered mesh/world objects.")
		.def("get_mesh_names", &PyNewtonBuilder::get_mesh_names, "Return registered mesh names ordered by registration id.")
		.def("print_registered_meshes_info", &PyNewtonBuilder::print_registered_meshes_info, "Print registered meshes info")
		.def("init_device",
			&PyNewtonBuilder::init_device,
			py::arg("backend_name") = py::none(),
			py::arg("binary_path") = py::none(),
			"Create and own a luisa compute device/stream.\n\n"
			"backend_name: optional backend string (e.g. 'cuda','metal','dx')\n"
			"binary_path: optional binary path passed to luisa::compute::Context")
		.def("set_device",
			&PyNewtonBuilder::set_device,
			py::arg("device_ptr"),
			py::arg("stream_ptr"),
			"Borrow an existing luisa Device/Stream (non-owning).\n\n"
			"device_ptr: integer address of a luisa::compute::Device object\n"
			"stream_ptr: integer address of a luisa::compute::Stream object\n"
			"The caller must ensure these objects outlive this solver.")
		.def("cleanup_device", &PyNewtonBuilder::cleanup_device, "Release owned device resources (no-op for borrowed device).")
		.def("get_device_ptr", &PyNewtonBuilder::get_device_ptr, "Return the raw pointer (as int) to the active luisa::compute::Device.")
		.def("get_stream_ptr", &PyNewtonBuilder::get_stream_ptr, "Return the raw pointer (as int) to the active luisa::compute::Stream.")
		.def("get_config",
			&PyNewtonBuilder::get_config,
			py::return_value_policy::reference_internal,
			"Return reference to solver-owned SceneParams config")
		.def("init_solver", &PyNewtonBuilder::init_solver, "Initialize the underlying solver using the device set via init_device()/set_device()")
		.def("physics_step_cpu", &PyNewtonBuilder::physics_step_cpu, "Advance one simulation frame using the CPU solver path.")
		.def("physics_step_gpu", &PyNewtonBuilder::physics_step_gpu, "Advance one simulation frame using the GPU solver path.")
		.def("restart_system", &PyNewtonBuilder::restart_system, "Reset positions/velocities to initial rest state")
		.def("update_per_vertex_animation", &PyNewtonBuilder::update_per_vertex_animation, py::arg("mesh_idx"), py::arg("local_vid"), py::arg("target_pos"),
			"Update one animated vertex target position for a registered object.")
		.def("update_per_body_animation", &PyNewtonBuilder::update_per_body_animation, py::arg("mesh_idx"), py::arg("target_translation"), py::arg("target_rotation"),
			"Update animated rigid body target translation and rotation for a registered object.")
		.def("get_rigid_body_translation", &PyNewtonBuilder::get_rigid_body_translation, py::arg("registration_id"),
			"Return rigid body translation as (tx, ty, tz).")
		.def("get_rigid_body_scaling", &PyNewtonBuilder::get_rigid_body_scaling, py::arg("registration_id"),
			"Return rigid body scaling extracted from ABD affine matrix as (sx, sy, sz).")
		.def("get_rigid_body_rotation_quaternion", &PyNewtonBuilder::get_rigid_body_rotation_quaternion, py::arg("registration_id"),
			"Return rigid body rotation quaternion as (qx, qy, qz, qw).")
		.def("get_rigid_body_rotation_axis_angle", &PyNewtonBuilder::get_rigid_body_rotation_axis_angle, py::arg("registration_id"),
			"Return rigid body rotation vector (axis * angle_rad) as (rx, ry, rz).")
		.def("get_sim_result", &PyNewtonBuilder::get_sim_result, "Return simulation results as a tuple (vertices_list, faces_list) of numpy arrays")
		.def("add_fixed_joint",
			&PyNewtonBuilder::add_fixed_joint,
			py::arg("body_a_registration"),
			py::arg("body_b_registration"),
			py::arg("anchor_a_local"),
			py::arg("anchor_b_local"),
			py::arg("stiffness_pos") = 1.0e4f,
			py::arg("stiffness_rot") = 1.0e3f,
			"Add a fixed joint between two rigid bodies.\n\n"
			"The local anchors are expressed in each body's rest local frame. The joint constrains\n"
			"both anchor coincidence and relative orientation.")
		.def("add_prismatic_joint",
			&PyNewtonBuilder::add_prismatic_joint,
			py::arg("body_a_registration"),
			py::arg("body_b_registration"),
			py::arg("anchor_a_local"),
			py::arg("anchor_b_local"),
			py::arg("axis_world"),
			py::arg("stiffness_pos") = 1.0e4f,
			py::arg("stiffness_rot") = 1.0e3f,
			py::arg("slide_min") = -std::numeric_limits<float>::infinity(),
			py::arg("slide_max") = std::numeric_limits<float>::infinity(),
			"Add a prismatic joint between two rigid bodies.\n\n"
			"axis_world defines the free sliding axis in the rest pose. The joint constrains the\n"
			"relative offset perpendicular to the axis and locks relative orientation; slide_min\n"
			"and slide_max bound the scalar coordinate along the axis.")
		.def("add_revolute_joint",
			&PyNewtonBuilder::add_revolute_joint,
			py::arg("body_a_registration"),
			py::arg("body_b_registration"),
			py::arg("anchor_a_local"),
			py::arg("anchor_b_local"),
			py::arg("axis_world"),
			py::arg("axis_a_local"),
			py::arg("axis_b_local"),
			py::arg("stiffness_pos") = 1.0e4f,
			py::arg("stiffness_axis") = 1.0e3f,
			"Add a revolute/hinge joint between two rigid bodies.\n\n"
			"The anchors are kept coincident while axis_a_local and axis_b_local are aligned,\n"
			"leaving rotation around the hinge axis free.")
		.def("query_local_vid_from_global_vid",
			&PyNewtonBuilder::query_local_vid_from_global_vid,
			py::arg("global_vid"),
			"Look up the local vertex index for a given global vertex id.")
		.def("query_registration_vid_from_global_vid",
			&PyNewtonBuilder::query_registration_vid_from_global_vid,
			py::arg("global_vid"),
			"Look up the world_data (mesh) registration index for a given global vertex id.")
		.def("get_object_sim_result_by_registration_id",
			&PyNewtonBuilder::get_object_sim_result_by_registration_id,
			py::arg("registration_id"),
			"Return one object simulation result as tuple (vertices, faces) by registration id")
		.def("get_object_by_registration_id", &PyNewtonBuilder::get_object_by_registration_id, py::arg("registration_id"))
		.def("get_vert_mass", &PyNewtonBuilder::get_vert_mass, py::arg("global_vid"), "Return mass of a vertex by global vertex id")
		.def("save_sim_result", &PyNewtonBuilder::save_sim_result, py::arg("obj_path"), "Save current simulation result to an OBJ file.");

	// Expose luisa::float3 so Python can access .x/.y/.z on floor, gravity, etc.
	py::class_<luisa::float3>(m, "Float3")
		.def(py::init<>())
		.def(py::init<float, float, float>(), py::arg("x"), py::arg("y"), py::arg("z"))
		.def_readwrite("x", &luisa::float3::x)
		.def_readwrite("y", &luisa::float3::y)
		.def_readwrite("z", &luisa::float3::z)
		.def("__repr__", [](const luisa::float3& v)
			{ return "Float3(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ")"; });

	// Expose SceneParams and accessors so Python can read/modify global scene settings
	py::class_<lcs::SceneParams>(m, "SceneParams")
		.def("update_dt", &lcs::SceneParams::update_dt, py::arg("dt"), "Update the frame time step and derived substep time step.")
		.def_readwrite("use_gpu", &lcs::SceneParams::use_gpu)
		.def_readwrite("fix_scene", &lcs::SceneParams::fix_scene)
		.def_readwrite("use_energy_linesearch", &lcs::SceneParams::use_energy_linesearch)
		.def_readwrite("use_ccd_linesearch", &lcs::SceneParams::use_ccd_linesearch)
		.def_readwrite("print_system_energy", &lcs::SceneParams::print_system_energy)
		.def_readwrite("print_pcg_info", &lcs::SceneParams::print_pcg_info)
		.def_readwrite("print_collision_info", &lcs::SceneParams::print_collision_info)
		.def_readwrite("use_floor", &lcs::SceneParams::use_floor)
		.def_readwrite("use_self_collision", &lcs::SceneParams::use_self_collision)
		.def_readwrite("output_per_frame", &lcs::SceneParams::output_per_frame)
		.def_readwrite("output_per_iteration", &lcs::SceneParams::output_per_iteration)
		.def_readwrite("scene_id", &lcs::SceneParams::scene_id)
		.def_readonly("num_substep", &lcs::SceneParams::num_substep)
		.def_readwrite("nonlinear_iter_count", &lcs::SceneParams::nonlinear_iter_count)
		.def_readwrite("pcg_iter_count", &lcs::SceneParams::pcg_iter_count)
		.def_readwrite("current_frame", &lcs::SceneParams::current_frame)
		.def_readonly("current_nonlinear_iter", &lcs::SceneParams::current_nonlinear_iter)
		.def_readonly("current_pcg_it", &lcs::SceneParams::current_pcg_it)
		.def_readonly("current_substep", &lcs::SceneParams::current_substep)
		.def_readwrite("collision_detection_frequece", &lcs::SceneParams::collision_detection_frequece)
		.def_readwrite("contact_energy_type", &lcs::SceneParams::contact_energy_type)
		.def_readwrite("implicit_dt", &lcs::SceneParams::implicit_dt)
		.def_readwrite("explicit_dt", &lcs::SceneParams::explicit_dt)
		.def_readonly("dt", &lcs::SceneParams::dt)
		.def_readwrite("floor", &lcs::SceneParams::floor)
		.def_readwrite("gravity", &lcs::SceneParams::gravity)
		.def_readwrite("stiffness_bending_ui", &lcs::SceneParams::stiffness_bending_ui)
		.def_readwrite("stiffness_collision", &lcs::SceneParams::stiffness_collision)
		.def_readwrite("stiffness_dirichlet", &lcs::SceneParams::stiffness_dirichlet)
		.def_readwrite("damping_rate", &lcs::SceneParams::damping_rate)
		.def_readwrite("d_hat", &lcs::SceneParams::d_hat)
		.def("get_substep_dt", &lcs::SceneParams::get_substep_dt, "Return the current substep time step.")
		.def("get_bending_stiffness_scaling", &lcs::SceneParams::get_bending_stiffness_scaling, "Return the bending stiffness scaling factor for current settings.");

	m.doc() = "Python bindings for basic NewtonSolver scene building (lightweight)";
}
