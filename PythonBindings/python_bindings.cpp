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

// Global state for luisa device/context created from Python.
// Supports two modes:
//   - Owned:    created by device_init(), resources released by cleanup()
//   - Borrowed: set by device_set(), caller retains ownership
struct GlobalState
{
	// Owned resources (only populated when we create them ourselves)
	std::unique_ptr<luisa::compute::Context> owned_context;
	std::unique_ptr<luisa::compute::Device>	 owned_device;
	std::unique_ptr<luisa::compute::Stream>	 owned_stream;

	// Active pointers (always valid when initialized == true;
	// point to owned_* or external objects depending on mode)
	luisa::compute::Device* device = nullptr;
	luisa::compute::Stream* stream = nullptr;

	bool initialized = false;
	bool owns_resources = false;

	void cleanup()
	{
		device = nullptr;
		stream = nullptr;
		if (owns_resources)
		{
			owned_stream.reset();
			owned_device.reset();
			owned_context.reset();
		}
		initialized = false;
		owns_resources = false;
	}
} g_state;

static void global_set_device(luisa::compute::Device* device, luisa::compute::Stream* stream)
{
	if (g_state.initialized)
		g_state.cleanup();

	g_state.device = device;
	g_state.stream = stream;
	g_state.initialized = true;
}

// Create a luisa device/stream owned by this module, with optional backend and binary path arguments.
static void global_create_device(py::object backend_name_obj = py::none(), py::object binary_path_obj = py::none())
{
	if (g_state.initialized)
		return;

	// determine binary path
	std::string binary_path;
	if (!binary_path_obj.is_none())
	{
		binary_path = binary_path_obj.cast<std::string>();
	}
	else
	{
		try
		{
			// try to get this module file path
			py::module_ self = py::module_::import("lcs_py");
			if (py::hasattr(self, "__file__"))
			{
				binary_path = self.attr("__file__").cast<std::string>();
			}
		}
		catch (...)
		{
		}
		if (binary_path.empty())
		{
#if defined(__APPLE__)
			// fallback to empty; luisa::compute may still accept an empty path
			binary_path = "";
#else
			binary_path = "";
#endif
		}
	}

	// create context
	LUISA_INFO("Creating luisa compute context/device/stream from Python...");
	LUISA_INFO("  binary path: {}", binary_path);
	g_state.owned_context = std::make_unique<luisa::compute::Context>(binary_path);

	// choose backend
	std::string backend;
	if (!backend_name_obj.is_none())
		backend = backend_name_obj.cast<std::string>();
	else
	{
#if defined(__APPLE__)
		backend = "metal";
#elif defined(_WIN32)
		backend = "dx";
#else
		backend = "cuda";
#endif
	}

	luisa::vector<luisa::string> device_names = g_state.owned_context->backend_device_names(backend);
	if (device_names.empty())
	{
		LUISA_WARNING("No hardware device found.");
		exit(1);
	}
	for (size_t i = 0; i < device_names.size(); ++i)
	{
		LUISA_INFO("Device {}: {}", i, device_names[i]);
	}

	// create device (use default flags)
	try
	{
		auto dev = g_state.owned_context->create_device(backend, nullptr, true);
		g_state.owned_device = std::make_unique<luisa::compute::Device>(std::move(dev));
		auto st = g_state.owned_device->create_stream(luisa::compute::StreamTag::COMPUTE);
		g_state.owned_stream = std::make_unique<luisa::compute::Stream>(std::move(st));

		global_set_device(g_state.owned_device.get(), g_state.owned_stream.get());
		g_state.owns_resources = true;
	}
	catch (const std::exception& e)
	{
		throw std::runtime_error(std::string("Failed to create luisa device: ") + e.what());
	}
}

// Use an existing device/stream from another module (non-owning).
// The caller must ensure that the passed-in objects outlive this module's usage.
static void global_set_device_from_pointers(uintptr_t device_ptr, uintptr_t stream_ptr)
{
	if (device_ptr == 0 || stream_ptr == 0)
		throw std::runtime_error("device_ptr and stream_ptr must be non-null");

	auto* device = reinterpret_cast<luisa::compute::Device*>(device_ptr);
	auto* stream = reinterpret_cast<luisa::compute::Stream*>(stream_ptr);
	global_set_device(device, stream);
	g_state.owns_resources = false;
}

static void global_free_device()
{
	g_state.cleanup();
}

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
	WorldDataWrapper& set_simulation_type(lcs::Initializer::SimulationType t)
	{
		wd->set_simulation_type(t);
		return *this;
	}

	// Expose cloth material setter by accepting keyword args
	WorldDataWrapper& set_physics_material_cloth(float thickness = 1e-3f, float youngs_modulus = 1e6f, float poisson_ratio = 0.35f, float area_bending_stiffness = 5e-3f)
	{
		wd->set_simulation_type(lcs::Initializer::SimulationType::Cloth);
		ClothMaterial mat;
		mat.thickness = thickness;
		mat.youngs_modulus = youngs_modulus;
		mat.poisson_ratio = poisson_ratio;
		mat.area_bending_stiffness = area_bending_stiffness;
		wd->set_physics_material(mat);
		return *this;
	}

	// Expose tetrahedral material setter
	WorldDataWrapper& set_physics_material_tet(float youngs_modulus = 1e6f, float poisson_ratio = 0.35f, float density = 1e3f, float mass = 0.0f)
	{
		wd->set_simulation_type(lcs::Initializer::SimulationType::Tetrahedral);
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
	WorldDataWrapper& set_physics_material_rigid(float thickness = 1e-3f, float stiffness = 1e6f, float density = 1e3f, float mass = 0.0f)
	{
		wd->set_simulation_type(lcs::Initializer::SimulationType::Rigid);
		RigidMaterial mat;
		mat.thickness = thickness;
		mat.stiffness = stiffness;
		mat.density = density;
		mat.mass = mass;
		wd->set_physics_material(mat);
		return *this;
	}

	// Expose rod material setter
	WorldDataWrapper& set_physics_material_rod(float radius = 1e-3f, float bending_stiffness = 1e4f, float twisting_stiffness = 1e4f, float density = 1e3f, float mass = 0.0f)
	{
		wd->set_simulation_type(lcs::Initializer::SimulationType::Rod);
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
};

// Python-facing Newton-like builder that stores a vector<WorldData>
struct PyNewtonBuilder
{
	std::vector<WorldData>			   shell_list;
	std::unique_ptr<lcs::NewtonSolver> solver_ptr;

	PyNewtonBuilder()
		: solver_ptr(std::make_unique<lcs::NewtonSolver>())
	{
	}

	// register_mesh accepts numpy arrays (vertices Nx3, triangles Mx3)
	WorldDataWrapper register_mesh(const std::string&				   name,
		py::array_t<double, py::array::c_style | py::array::forcecast> vertices,
		py::array_t<int, py::array::c_style | py::array::forcecast>	   triangles)
	{
		// Validate shapes
		if (vertices.ndim() != 2 || vertices.shape(1) != 3)
			throw std::runtime_error("vertices must be a (N,3) array of floats");
		if (triangles.ndim() != 2 || triangles.shape(1) != 3)
			throw std::runtime_error("triangles must be a (M,3) array of ints");

		WorldData info;
		info.set_name(name);

		const size_t nverts = vertices.shape(0);
		const size_t nfaces = triangles.shape(0);

		// copy vertices
		info.input_mesh.model_positions.resize(nverts);
		auto buf_v = vertices.unchecked<2>();
		for (size_t i = 0; i < nverts; ++i)
		{
			SimMesh::Float3 p;
			p[0] = static_cast<float>(buf_v(i, 0));
			p[1] = static_cast<float>(buf_v(i, 1));
			p[2] = static_cast<float>(buf_v(i, 2));
			info.input_mesh.model_positions[i] = p;
		}

		// copy faces
		info.input_mesh.faces.resize(nfaces);
		auto buf_t = triangles.unchecked<2>();
		for (size_t i = 0; i < nfaces; ++i)
		{
			SimMesh::Int3 f;
			f[0] = static_cast<unsigned int>(buf_t(i, 0));
			f[1] = static_cast<unsigned int>(buf_t(i, 1));
			f[2] = static_cast<unsigned int>(buf_t(i, 2));
			info.input_mesh.faces[i] = f;
		}

		SimMesh::extract_edges_from_surface(info.input_mesh.faces, info.input_mesh.edges, info.input_mesh.dihedral_edges, true);

		shell_list.emplace_back(std::move(info));
		return WorldDataWrapper(&shell_list.back());
	}

	// register mesh from an obj file path: read file then compute auxiliary topology
	WorldDataWrapper register_mesh(const std::string& name, const std::string& obj_file_path)
	{
		WorldData info;
		info.set_name(name);
		SimMesh::read_mesh_file(obj_file_path, info.input_mesh);

		shell_list.emplace_back(std::move(info));
		return WorldDataWrapper(&shell_list.back());
	}
	// expose method to get number of registered meshes
	size_t num_meshes() const { return shell_list.size(); }

	// expose a method to export registered meshes as python lists (simple)
	py::list get_mesh_names() const
	{
		py::list out;
		for (const auto& w : shell_list)
			out.append(w.get_model_name());
		return out;
	}

	// Initialize underlying NewtonSolver using the global device/context created by py_init
	void init_solver()
	{
		if (!g_state.initialized)
			throw std::runtime_error("Global luisa context/device not initialized. Call init(...) first.");

		solver_ptr->init_solver(*g_state.device, *g_state.stream, shell_list);
		LUISA_INFO("Solver initialized (device owned={}).", g_state.owns_resources);
	}

	void physics_step_cpu()
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");
		solver_ptr->physics_step_CPU(*g_state.device, *g_state.stream);
	}

	void physics_step_gpu()
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");
		solver_ptr->physics_step_GPU(*g_state.device, *g_state.stream);
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
	py::tuple get_sim_result_to()
	{
		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

		const uint num_meshes = solver_ptr->get_host_mesh_data().num_meshes;

		std::vector<std::vector<std::array<float, 3>>> sa_rendering_vertices(num_meshes);
		solver_ptr->get_simulation_results_to_host(sa_rendering_vertices);

		py::list py_verts;
		py::list py_faces;
		for (uint i = 0; i < num_meshes; ++i)
		{
			// vertices – contiguous std::array<float,3>, safe to memcpy
			const auto&		   mesh_verts = sa_rendering_vertices[i];
			py::array_t<float> v_arr({ (ssize_t)mesh_verts.size(), (ssize_t)3 });
			if (!mesh_verts.empty())
				std::memcpy(v_arr.mutable_data(), mesh_verts.data(), mesh_verts.size() * 3 * sizeof(float));
			py_verts.append(v_arr);

			// faces
			const auto&			  mesh_faces = shell_list[i].input_mesh.faces;
			py::array_t<uint32_t> f_arr({ (ssize_t)mesh_faces.size(), (ssize_t)3 });
			if (!mesh_faces.empty())
				std::memcpy(f_arr.mutable_data(), mesh_faces.data(), mesh_faces.size() * 3 * sizeof(uint32_t));
			py_faces.append(f_arr);
		}
		return py::make_tuple(py_verts, py_faces);
	}

	void save_to(const std::string& full_path)
	{
		std::vector<std::vector<std::array<float, 3>>> sa_rendering_vertices;
		std::vector<std::vector<std::array<uint, 3>>>  sa_rendering_faces;

		if (!solver_ptr)
			throw std::runtime_error("Solver not initialized. Call init_solver() first.");

		const uint num_meshes = solver_ptr->get_host_mesh_data().num_meshes;

		sa_rendering_vertices.resize(num_meshes);
		sa_rendering_faces.resize(num_meshes);
		for (uint i = 0; i < num_meshes; ++i)
		{
			sa_rendering_faces[i] = shell_list[i].input_mesh.faces;
		}
		solver_ptr->get_simulation_results_to_host(sa_rendering_vertices);
		SimMesh::saveToOBJ_combined(sa_rendering_vertices, sa_rendering_faces, full_path);
	}
};

PYBIND11_MODULE(lcs_py, m)
{
	py::enum_<lcs::Initializer::SimulationType>(m, "SimulationType")
		.value("Cloth", lcs::Initializer::SimulationType::Cloth)
		.value("Tetrahedral", lcs::Initializer::SimulationType::Tetrahedral)
		.value("Rigid", lcs::Initializer::SimulationType::Rigid)
		.value("Rod", lcs::Initializer::SimulationType::Rod)
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
			py::arg("thickness") = 1e-3f,
			py::arg("youngs_modulus") = 1e6f,
			py::arg("poisson_ratio") = 0.35f,
			py::arg("area_bending_stiffness") = 5e-3f)
		.def("set_physics_material_tet",
			&WorldDataWrapper::set_physics_material_tet,
			py::arg("youngs_modulus") = 1e6f,
			py::arg("poisson_ratio") = 0.35f,
			py::arg("density") = 1e3f,
			py::arg("mass") = 0.0f)
		.def("set_physics_material_rigid",
			&WorldDataWrapper::set_physics_material_rigid,
			py::arg("thickness") = 1e-3f,
			py::arg("stiffness") = 1e6f,
			py::arg("density") = 1e3f,
			py::arg("mass") = 0.0f)
		.def("set_physics_material_rod",
			&WorldDataWrapper::set_physics_material_rod,
			py::arg("radius") = 1e-3f,
			py::arg("bending_stiffness") = 1e4f,
			py::arg("twisting_stiffness") = 1e4f,
			py::arg("density") = 1e3f,
			py::arg("mass") = 0.0f)
		.def("add_fixed_point_info", &WorldDataWrapper::add_fixed_point_info)
		.def("add_fixed_point_by_method", &WorldDataWrapper::add_fixed_point_by_method, py::arg("method"), py::arg("range") = 0.001f)
		.def("add_fixed_point_indices", &WorldDataWrapper::add_fixed_point_indices, py::arg("indices"))
		.def("set_translation", &WorldDataWrapper::set_translation)
		.def("set_rotation", &WorldDataWrapper::set_rotation)
		.def("set_scale", &WorldDataWrapper::set_scale);

	// disambiguate overloaded register_mesh signatures
	using VertArr = py::array_t<double, py::array::c_style | py::array::forcecast>;
	using TriArr = py::array_t<int, py::array::c_style | py::array::forcecast>;

	py::class_<PyNewtonBuilder>(m, "NewtonSolver")
		.def(py::init<>())
		.def("register_mesh",
			(WorldDataWrapper(PyNewtonBuilder::*)(const std::string&, VertArr, TriArr)) & PyNewtonBuilder::register_mesh,
			py::arg("name"), py::arg("vertices"), py::arg("triangles"))
		.def("register_mesh",
			(WorldDataWrapper(PyNewtonBuilder::*)(const std::string&, const std::string&)) & PyNewtonBuilder::register_mesh,
			py::arg("name"), py::arg("obj_file_path"))
		.def("num_meshes", &PyNewtonBuilder::num_meshes)
		.def("get_mesh_names", &PyNewtonBuilder::get_mesh_names)
		.def("init_solver", &PyNewtonBuilder::init_solver, "Initialize the underlying solver using previously created device/context")
		.def("physics_step_cpu", &PyNewtonBuilder::physics_step_cpu)
		.def("physics_step_gpu", &PyNewtonBuilder::physics_step_gpu)
		.def("restart_system", &PyNewtonBuilder::restart_system, "Reset positions/velocities to initial rest state")
		.def("update_pinned_verts_position", &PyNewtonBuilder::update_pinned_verts_position,
			py::arg("mesh_idx"), py::arg("local_vid"), py::arg("target_pos"))
		.def("get_sim_result", &PyNewtonBuilder::get_sim_result_to, "Return simulation results as a tuple (vertices_list, faces_list) of numpy arrays")
		.def("save_sim_result", &PyNewtonBuilder::save_to, py::arg("obj_path"));

	m.def("device_init",
		&global_create_device,
		py::arg("backend_name") = py::none(),
		py::arg("binary_path") = py::none(),
		"Initialize luisa compute context/device/stream from Python.\n\n"
		"backend_name: optional backend string (e.g. 'cuda','metal','dx')\n"
		"binary_path: optional binary path (argv[0]) to pass to luisa::compute::Context");

	m.def("device_set",
		&global_set_device_from_pointers,
		py::arg("device_ptr"),
		py::arg("stream_ptr"),
		"Share an existing luisa Device/Stream from another module (non-owning).\n\n"
		"device_ptr: integer address of a luisa::compute::Device object\n"
		"stream_ptr: integer address of a luisa::compute::Stream object\n"
		"The caller must ensure these objects outlive this module's usage.");

	m.def("device_cleanup", &global_free_device, "Clean up luisa compute resources created from Python (optional; will also be called automatically at Python shutdown)");

	// Expose getters so other modules can retrieve our device/stream pointers
	m.def(
		"get_device_ptr",
		[]() -> uintptr_t
		{
			if (!g_state.initialized)
				throw std::runtime_error("Device not initialized.");
			return reinterpret_cast<uintptr_t>(g_state.device);
		},
		"Return the raw pointer (as int) to the active luisa::compute::Device.");

	m.def(
		"get_stream_ptr",
		[]() -> uintptr_t
		{
			if (!g_state.initialized)
				throw std::runtime_error("Device not initialized.");
			return reinterpret_cast<uintptr_t>(g_state.stream);
		},
		"Return the raw pointer (as int) to the active luisa::compute::Stream.");

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
		.def_readwrite("num_substep", &lcs::SceneParams::num_substep)
		.def_readwrite("nonlinear_iter_count", &lcs::SceneParams::nonlinear_iter_count)
		.def_readwrite("pcg_iter_count", &lcs::SceneParams::pcg_iter_count)
		.def_readwrite("current_frame", &lcs::SceneParams::current_frame)
		.def_readwrite("current_nonlinear_iter", &lcs::SceneParams::current_nonlinear_iter)
		.def_readwrite("current_pcg_it", &lcs::SceneParams::current_pcg_it)
		.def_readwrite("current_substep", &lcs::SceneParams::current_substep)
		.def_readwrite("collision_detection_frequece", &lcs::SceneParams::collision_detection_frequece)
		.def_readwrite("contact_energy_type", &lcs::SceneParams::contact_energy_type)
		.def_readwrite("implicit_dt", &lcs::SceneParams::implicit_dt)
		.def_readwrite("explicit_dt", &lcs::SceneParams::explicit_dt)
		.def_readwrite("dt", &lcs::SceneParams::dt)
		.def_readwrite("floor", &lcs::SceneParams::floor)
		.def_readwrite("gravity", &lcs::SceneParams::gravity)
		.def_readwrite("stiffness_bending_ui", &lcs::SceneParams::stiffness_bending_ui)
		.def_readwrite("stiffness_collision", &lcs::SceneParams::stiffness_collision)
		.def_readwrite("stiffness_dirichlet", &lcs::SceneParams::stiffness_dirichlet)
		.def_readwrite("damping_rate", &lcs::SceneParams::damping_rate)
		.def_readwrite("d_hat", &lcs::SceneParams::d_hat)
		.def("get_substep_dt", &lcs::SceneParams::get_substep_dt)
		.def("get_bending_stiffness_scaling", &lcs::SceneParams::get_bending_stiffness_scaling);

	m.def(
		"get_scene_params",
		[]() -> lcs::SceneParams&
		{ return lcs::get_scene_params(); },
		py::return_value_policy::reference,
		"Return reference to global SceneParams singleton");

	m.def("init_scene_params", &lcs::init_scene_params,
		"Initialize scene params (no-op if already initialized)");

	// Capsule destructor fires during Python interpreter shutdown (before C++
	// static destruction), ensuring owned resources are released while luisa
	// internals are still alive.  In borrowed mode this is a harmless no-op.
	py::capsule cleanup(new int(0), [](void* p)
		{
		delete static_cast<int*>(p);
		g_state.cleanup(); });
	m.add_object("_cleanup", cleanup);

	m.doc() = "Python bindings for basic NewtonSolver scene building (lightweight)";
}
