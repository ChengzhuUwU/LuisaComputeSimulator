#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <luisa/luisa-compute.h>

#include "Initializer/init_mesh_data.h"
#include "MeshOperation/mesh_reader.h"
#include "SimulationSolver/newton_solver.h"
#include "SimulationCore/scene_params.h"

namespace py = pybind11;
using namespace lcs;
using namespace lcs::Initializer;

// Helper wrapper to hold WorldData pointer and expose chainable methods
struct WorldDataWrapper
{
	WorldData* wd;
	WorldDataWrapper(WorldData* w)
		: wd(w)
	{
	}

	// parse FixedPointsType from string (same names used in JSON/config)
	static FixedPointsType parse_fixed_method_py(const std::string& s)
	{
		if (s == "None")
			return FixedPointsType::None;
		if (s == "FromIndices")
			return FixedPointsType::FromIndices;
		if (s == "FromFunction")
			return FixedPointsType::FromFunction;
		if (s == "Left")
			return FixedPointsType::Left;
		if (s == "Right")
			return FixedPointsType::Right;
		if (s == "Front")
			return FixedPointsType::Front;
		if (s == "Back")
			return FixedPointsType::Back;
		if (s == "Up")
			return FixedPointsType::Up;
		if (s == "Down")
			return FixedPointsType::Down;
		if (s == "LeftBack")
			return FixedPointsType::LeftBack;
		if (s == "LeftFront")
			return FixedPointsType::LeftFront;
		if (s == "RightBack")
			return FixedPointsType::RightBack;
		if (s == "RightFront")
			return FixedPointsType::RightFront;
		if (s == "All")
			return FixedPointsType::All;
		return FixedPointsType::All;
	}

	WorldDataWrapper& set_name(const std::string& name)
	{
		wd->set_name(name);
		return *this;
	}
	WorldDataWrapper& set_simulation_type(lcs::Initializer::MaterialType t)
	{
		wd->set_material_type(t);
		return *this;
	}

	// Expose cloth material setter by accepting keyword args
	WorldDataWrapper& set_physics_material_cloth(
		float thickness = ClothMaterial::default_thickness(),
		float youngs_modulus = ClothMaterial::default_youngs_modulus(),
		float poisson_ratio = ClothMaterial::default_poisson_ratio(),
		float area_bending_stiffness = ClothMaterial::default_area_bending_stiffness())
	{
		wd->set_material_type(lcs::Initializer::MaterialType::Cloth);
		ClothMaterial mat;
		mat.thickness = thickness;
		mat.youngs_modulus = youngs_modulus;
		mat.poisson_ratio = poisson_ratio;
		mat.area_bending_stiffness = area_bending_stiffness;
		wd->set_physics_material(mat);
		return *this;
	}

	// Expose tetrahedral material setter
	WorldDataWrapper& set_physics_material_tet(
		float youngs_modulus = TetMaterial::default_youngs_modulus(),
		float poisson_ratio = TetMaterial::default_poisson_ratio(),
		float density = TetMaterial::default_density(),
		float mass = TetMaterial::default_mass())
	{
		wd->set_material_type(lcs::Initializer::MaterialType::Tetrahedral);
		TetMaterial mat;
		mat.youngs_modulus = youngs_modulus;
		mat.poisson_ratio = poisson_ratio;
		mat.density = density;
		mat.mass = mass;
		mat.is_shell = false;
		wd->set_physics_material(mat);
		return *this;
	}

	// Expose rigid material setter
	WorldDataWrapper& set_physics_material_rigid(
		float thickness = RigidMaterial::default_thickness(),
		float stiffness = RigidMaterial::default_stiffness(),
		float density = RigidMaterial::default_density(),
		float mass = RigidMaterial::default_mass())
	{
		wd->set_material_type(lcs::Initializer::MaterialType::Rigid);
		RigidMaterial mat;
		mat.thickness = thickness;
		mat.stiffness = stiffness;
		mat.density = density;
		mat.mass = mass;
		wd->set_physics_material(mat);
		return *this;
	}

	// Expose rod material setter
	WorldDataWrapper& set_physics_material_rod(
		float radius = RodMaterial::default_radius(),
		float bending_stiffness = RodMaterial::default_bending_stiffness(),
		float twisting_stiffness = RodMaterial::default_twisting_stiffness(),
		float density = RodMaterial::default_density(),
		float mass = RodMaterial::default_mass())
	{
		wd->set_material_type(lcs::Initializer::MaterialType::Rod);
		RodMaterial mat;
		mat.radius = radius;
		mat.bending_stiffness = bending_stiffness;
		mat.twisting_stiffness = twisting_stiffness;
		mat.density = density;
		mat.mass = mass;
		wd->set_physics_material(mat);
		return *this;
	}

	WorldDataWrapper& add_fixed_point_info(const MakeFixedPointsInterface& info)
	{
		wd->add_fixed_point_info(info);
		return *this;
	}

	// Convenience: add fixed-point rule by name and optional numeric range/list
	WorldDataWrapper& add_fixed_point_by_method(const std::string& method, float range)
	{
		MakeFixedPointsInterface mfp;
		mfp.method = parse_fixed_method_py(method);
		mfp.range = range;

		wd->add_fixed_point_info(mfp);
		return *this;
	}

	// Convenience: add explicit vertex indices as fixed points
	WorldDataWrapper& add_fixed_point_indices(py::array_t<int, py::array::c_style | py::array::forcecast> indices)
	{
		if (indices.ndim() != 1)
			throw std::runtime_error("indices must be a 1-D array of ints");
		auto		 buf = indices.unchecked<1>();
		const size_t n = indices.shape(0);
		for (size_t i = 0; i < n; ++i)
		{
			int v = buf(i);
			if (v >= 0)
				wd->fixed_point_indices.push_back(static_cast<uint>(v));
		}
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

	std::string get_name() const
	{
		return wd->get_model_name();
	}
	uint get_registration_index() const
	{
		return wd->get_registration_index();
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
	WorldDataWrapper register_mesh_from_array(const std::string&	   name,
		py::array_t<double, py::array::c_style | py::array::forcecast> vertices,
		py::array_t<int, py::array::c_style | py::array::forcecast>	   triangles)
	{
		// Validate shapes
		if (vertices.ndim() != 2 || vertices.shape(1) != 3)
			throw std::runtime_error("vertices must be a (N,3) array of floats");
		if (triangles.ndim() != 2 || triangles.shape(1) != 3)
			throw std::runtime_error("triangles must be a (M,3) array of ints");

		using InputVertexType = std::array<float32_t, 3>;
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
			p[0] = static_cast<float32_t>(buf_v(i, 0));
			p[1] = static_cast<float32_t>(buf_v(i, 1));
			p[2] = static_cast<float32_t>(buf_v(i, 2));
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

		return WorldDataWrapper(&solver_ptr->register_world_data_from_array(name, input_vertices, input_triangles));
	}
	// register mesh from an obj file path: read file then compute auxiliary topology
	WorldDataWrapper register_mesh_from_file_path(const std::string& name, const std::string& obj_file_path)
	{
		return WorldDataWrapper(&solver_ptr->register_world_data_from_file_path(name, obj_file_path));
	}

	// expose method to get number of registered meshes
	size_t num_meshes() const { return solver_ptr->get_world_data().size(); }

	// expose a method to export registered meshes as python lists (simple)
	py::list get_mesh_names() const
	{
		py::list				 out;
		const auto&				 world_data = solver_ptr->get_world_data();
		std::vector<std::string> names(world_data.size());
		for (const auto& w : world_data)
		{
			names[w.get_registration_index()] = w.get_model_name();
		}
		for (const auto& name : names)
			out.append(name);
		return out;
	}

	// Initialize underlying NewtonSolver using the device previously set via init_device()/set_device().
	void init_solver()
	{
		solver_ptr->init_solver();
		LUISA_INFO("Solver initialized.");
	}

	void physics_step_cpu()
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");
		solver_ptr->physics_step_CPU();
	}

	void physics_step_gpu()
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");
		solver_ptr->physics_step_GPU();
	}

	void restart_system()
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");
		solver_ptr->lcs::SolverInterface::restart_system();
	}

	// Update a pinned vertex position on the solver (mesh local vertex id, target position)
	void update_pinned_verts_position(const unsigned int			  mesh_idx,
		const unsigned int											  local_vid,
		py::array_t<float, py::array::c_style | py::array::forcecast> target_pos)
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

		if (target_pos.ndim() != 1 || target_pos.shape(0) != 3)
			throw std::runtime_error("target_pos must be a 1-D array of length 3 (x,y,z)");

		auto				 buf = target_pos.unchecked<1>();
		std::array<float, 3> tp{ buf(0), buf(1), buf(2) };
		solver_ptr->update_pinned_verts_position(mesh_idx, local_vid, tp);
	}

	// Return simulation results as a tuple of (vertices_list, faces_list) of numpy arrays.
	// Uses memcpy for efficient data transfer.
	py::tuple get_sim_result()
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

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

	py::tuple get_object_sim_result_by_registration_id(uint registration_id)
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

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

	py::tuple get_object_sim_result_by_unique_name(const std::string& unique_name)
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

		std::vector<std::array<float, 3>> object_vertices;
		std::vector<std::array<uint, 3>>  object_triangles;
		solver_ptr->get_object_sim_result_by_unique_name(unique_name, object_vertices, object_triangles);

		py::array_t<float> v_arr({ (size_t)object_vertices.size(), (size_t)3 });
		if (!object_vertices.empty())
			std::memcpy(v_arr.mutable_data(), object_vertices.data(), object_vertices.size() * 3 * sizeof(float));

		py::array_t<uint32_t> f_arr({ (size_t)object_triangles.size(), (size_t)3 });
		if (!object_triangles.empty())
			std::memcpy(f_arr.mutable_data(), object_triangles.data(), object_triangles.size() * 3 * sizeof(uint32_t));

		return py::make_tuple(v_arr, f_arr);
	}

	WorldDataWrapper get_object_by_registration_id(uint registration_id) const
	{
		const uint sorted_idx = solver_ptr->query_object_index_by_registration_id(registration_id);
		return WorldDataWrapper(&solver_ptr->get_world_data()[sorted_idx]);
	}

	WorldDataWrapper get_object_by_unique_name(const std::string& unique_name) const
	{
		const uint sorted_idx = solver_ptr->query_object_index_by_unique_name(unique_name);
		return WorldDataWrapper(&solver_ptr->get_world_data()[sorted_idx]);
	}

	void save_sim_result(const std::string& full_path)
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

		solver_ptr->save_mesh_to_obj(full_path);
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
					binary_path = self.attr("__file__").cast<std::string>();
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
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized.");
		return solver_ptr->get_config();
	}
};

PYBIND11_MODULE(lcs_py, m)
{
	py::enum_<lcs::Initializer::MaterialType>(m, "MaterialType")
		.value("Cloth", lcs::Initializer::MaterialType::Cloth)
		.value("Tetrahedral", lcs::Initializer::MaterialType::Tetrahedral)
		.value("Rigid", lcs::Initializer::MaterialType::Rigid)
		.value("Rod", lcs::Initializer::MaterialType::Rod)
		.export_values();

	py::class_<ClothMaterial>(m, "ClothMaterial")
		.def(py::init<>())
		.def_readwrite("thickness", &ClothMaterial::thickness)
		.def_readwrite("youngs_modulus", &ClothMaterial::youngs_modulus)
		.def_readwrite("poisson_ratio", &ClothMaterial::poisson_ratio)
		.def_readwrite("area_bending_stiffness", &ClothMaterial::area_bending_stiffness)
		.def_readwrite("mass", &ClothMaterial::mass)
		.def_readwrite("density", &ClothMaterial::density);

	py::class_<TetMaterial>(m, "TetMaterial")
		.def(py::init<>())
		.def_readwrite("youngs_modulus", &TetMaterial::youngs_modulus)
		.def_readwrite("poisson_ratio", &TetMaterial::poisson_ratio)
		.def_readwrite("mass", &TetMaterial::mass)
		.def_readwrite("density", &TetMaterial::density)
		.def_readwrite("d_hat", &TetMaterial::d_hat)
		.def_readwrite("friction_mu", &TetMaterial::friction_mu);

	py::class_<RigidMaterial>(m, "RigidMaterial")
		.def(py::init<>())
		.def_readwrite("thickness", &RigidMaterial::thickness)
		.def_readwrite("stiffness", &RigidMaterial::stiffness)
		.def_readwrite("is_solid", &RigidMaterial::is_solid)
		.def_readwrite("mass", &RigidMaterial::mass)
		.def_readwrite("density", &RigidMaterial::density)
		.def_readwrite("d_hat", &RigidMaterial::d_hat)
		.def_readwrite("friction_mu", &RigidMaterial::friction_mu);

	py::class_<RodMaterial>(m, "RodMaterial")
		.def(py::init<>())
		.def_readwrite("radius", &RodMaterial::radius)
		.def_readwrite("bending_stiffness", &RodMaterial::bending_stiffness)
		.def_readwrite("twisting_stiffness", &RodMaterial::twisting_stiffness)
		.def_readwrite("mass", &RodMaterial::mass)
		.def_readwrite("density", &RodMaterial::density)
		.def_readwrite("d_hat", &RodMaterial::d_hat)
		.def_readwrite("friction_mu", &RodMaterial::friction_mu);

	py::class_<MakeFixedPointsInterface>(m, "MakeFixedPointsInterface")
		.def(py::init<>())
		.def_readwrite("method", &MakeFixedPointsInterface::method)
		.def_readwrite("range", &MakeFixedPointsInterface::range);

	py::class_<WorldDataWrapper>(m, "WorldData")
		.def("set_name", &WorldDataWrapper::set_name)
		.def("set_simulation_type", &WorldDataWrapper::set_simulation_type)
		.def("set_physics_material_cloth",
			&WorldDataWrapper::set_physics_material_cloth,
			py::arg("thickness") = ClothMaterial::default_thickness(),
			py::arg("youngs_modulus") = ClothMaterial::default_youngs_modulus(),
			py::arg("poisson_ratio") = ClothMaterial::default_poisson_ratio(),
			py::arg("area_bending_stiffness") = ClothMaterial::default_area_bending_stiffness())
		.def("set_physics_material_tet",
			&WorldDataWrapper::set_physics_material_tet,
			py::arg("youngs_modulus") = TetMaterial::default_youngs_modulus(),
			py::arg("poisson_ratio") = TetMaterial::default_poisson_ratio(),
			py::arg("density") = TetMaterial::default_density(),
			py::arg("mass") = TetMaterial::default_mass())
		.def("set_physics_material_rigid",
			&WorldDataWrapper::set_physics_material_rigid,
			py::arg("thickness") = RigidMaterial::default_thickness(),
			py::arg("stiffness") = RigidMaterial::default_stiffness(),
			py::arg("density") = RigidMaterial::default_density(),
			py::arg("mass") = RigidMaterial::default_mass())
		.def("set_physics_material_rod",
			&WorldDataWrapper::set_physics_material_rod,
			py::arg("radius") = RodMaterial::default_radius(),
			py::arg("bending_stiffness") = RodMaterial::default_bending_stiffness(),
			py::arg("twisting_stiffness") = RodMaterial::default_twisting_stiffness(),
			py::arg("density") = RodMaterial::default_density(),
			py::arg("mass") = RodMaterial::default_mass())
		.def("add_fixed_point_info", &WorldDataWrapper::add_fixed_point_info)
		.def("add_fixed_point_by_method", &WorldDataWrapper::add_fixed_point_by_method, py::arg("method"), py::arg("range") = 0.001f)
		.def("add_fixed_point_indices", &WorldDataWrapper::add_fixed_point_indices, py::arg("indices"))
		.def("set_translation", &WorldDataWrapper::set_translation)
		.def("set_rotation", &WorldDataWrapper::set_rotation)
		.def("set_scale", &WorldDataWrapper::set_scale)
		.def("get_name", &WorldDataWrapper::get_name)
		.def("get_registration_index", &WorldDataWrapper::get_registration_index);

	// disambiguate overloaded register_mesh signatures
	using VertArr = py::array_t<double, py::array::c_style | py::array::forcecast>;
	using TriArr = py::array_t<int, py::array::c_style | py::array::forcecast>;

	py::class_<PyNewtonBuilder>(m, "NewtonSolver")
		.def(py::init<>())
		.def("register_mesh_from_array", &PyNewtonBuilder::register_mesh_from_array, py::arg("name"), py::arg("vertices"), py::arg("triangles"))
		.def("register_mesh_from_file_path", &PyNewtonBuilder::register_mesh_from_file_path, py::arg("name"), py::arg("obj_file_path"))
		.def("num_meshes", &PyNewtonBuilder::num_meshes)
		.def("get_mesh_names", &PyNewtonBuilder::get_mesh_names)
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
		.def("physics_step_cpu", &PyNewtonBuilder::physics_step_cpu)
		.def("physics_step_gpu", &PyNewtonBuilder::physics_step_gpu)
		.def("restart_system", &PyNewtonBuilder::restart_system, "Reset positions/velocities to initial rest state")
		.def("update_pinned_verts_position", &PyNewtonBuilder::update_pinned_verts_position, py::arg("mesh_idx"), py::arg("local_vid"), py::arg("target_pos"))
		.def("get_sim_result", &PyNewtonBuilder::get_sim_result, "Return simulation results as a tuple (vertices_list, faces_list) of numpy arrays")
		.def("get_object_sim_result_by_registration_id",
			&PyNewtonBuilder::get_object_sim_result_by_registration_id,
			py::arg("registration_id"),
			"Return one object simulation result as tuple (vertices, faces) by registration id")
		.def("get_object_sim_result_by_unique_name",
			&PyNewtonBuilder::get_object_sim_result_by_unique_name,
			py::arg("unique_name"),
			"Return one object simulation result as tuple (vertices, faces) by unique object name")
		.def("get_object_by_registration_id", &PyNewtonBuilder::get_object_by_registration_id, py::arg("registration_id"))
		.def("get_object_by_unique_name", &PyNewtonBuilder::get_object_by_unique_name, py::arg("unique_name"))
		.def("save_sim_result", &PyNewtonBuilder::save_sim_result, py::arg("obj_path"));

	// Expose luisa::float3 so Python can access .x/.y/.z on floor, gravity, etc.
	py::class_<luisa::float3>(m, "Float3")
		.def(py::init<>())
		.def(py::init<float, float, float>())
		.def_readwrite("x", &luisa::float3::x)
		.def_readwrite("y", &luisa::float3::y)
		.def_readwrite("z", &luisa::float3::z)
		.def("__repr__", [](const luisa::float3& v)
			{ return "Float3(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ")"; });

	// Expose SceneParams and accessors so Python can read/modify global scene settings
	py::class_<lcs::SceneParams>(m, "SceneParams")
		.def("update_dt", &lcs::SceneParams::update_dt)
		.def_readwrite("use_gpu", &lcs::SceneParams::use_gpu)
		.def_readwrite("fix_scene", &lcs::SceneParams::fix_scene)
		.def_readwrite("use_energy_linesearch", &lcs::SceneParams::use_energy_linesearch)
		.def_readwrite("use_ccd_linesearch", &lcs::SceneParams::use_ccd_linesearch)
		.def_readwrite("print_system_energy", &lcs::SceneParams::print_system_energy)
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
		.def("get_substep_dt", &lcs::SceneParams::get_substep_dt)
		.def("get_bending_stiffness_scaling", &lcs::SceneParams::get_bending_stiffness_scaling);

	m.doc() = "Python bindings for basic NewtonSolver scene building (lightweight)";
}
