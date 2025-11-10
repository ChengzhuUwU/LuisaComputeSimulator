#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <memory>
#include <optional>
#include <string>

#include <luisa/luisa-compute.h>

#include "Initializer/init_mesh_data.h"
#include "MeshOperation/mesh_reader.h"
#include "SimulationSolver/newton_solver.h"

namespace py = pybind11;
using namespace lcs;
using namespace lcs::Initializer;

// Global state for luisa device/context created from Python
struct GlobalState
{
    std::unique_ptr<luisa::compute::Context> context;
    std::unique_ptr<luisa::compute::Device>  device;
    std::unique_ptr<luisa::compute::Stream>  stream;
    bool                                     initialized = false;
} g_state;

static void py_init(py::object backend_name_obj = py::none(), py::object binary_path_obj = py::none())
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
            py::module_ self = py::module_::import("lcs_solver_py");
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
    g_state.context = std::make_unique<luisa::compute::Context>(binary_path);

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

    luisa::vector<luisa::string> device_names = g_state.context->backend_device_names(backend);
    if (device_names.empty())
    {
        LUISA_WARNING("No haredware device found.");
        exit(1);
    }
    for (size_t i = 0; i < device_names.size(); ++i)
    {
        LUISA_INFO("Device {}: {}", i, device_names[i]);
    }

    // create device (use default flags)
    try
    {
        auto dev       = g_state.context->create_device(backend, nullptr, true);
        g_state.device = std::make_unique<luisa::compute::Device>(std::move(dev));
        auto st        = g_state.device->create_stream(luisa::compute::StreamTag::COMPUTE);
        g_state.stream = std::make_unique<luisa::compute::Stream>(std::move(st));
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error(std::string("Failed to create luisa device: ") + e.what());
    }

    // shader path (optional) - Context::set_shader_path is not available in this Luisa version,
    // so we currently ignore shader_path parameter. Keep parameter for API compatibility.

    g_state.initialized = true;
}

// Helper wrapper to hold WorldData pointer and expose chainable methods
struct WorldDataWrapper
{
    WorldData* wd;
    WorldDataWrapper(WorldData* w)
        : wd(w)
    {
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
    WorldDataWrapper& set_physics_material_cloth(float thickness = 0.002f, float youngs_modulus = 1e5f, float poisson_ratio = 0.25f)
    {
        ClothMaterial mat;
        mat.thickness      = thickness;
        mat.youngs_modulus = youngs_modulus;
        mat.poisson_ratio  = poisson_ratio;
        wd->set_physics_material(mat);
        return *this;
    }

    WorldDataWrapper& add_fixed_point_info(const MakeFixedPointsInterface& info)
    {
        wd->add_fixed_point_info(info);
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

    // Optionally call load_mesh_data to compute auxiliary topology
    WorldDataWrapper& load_mesh_data()
    {
        wd->load_mesh_data();
        return *this;
    }
};

// Python-facing Newton-like builder that stores a vector<WorldData>
struct PyNewtonBuilder
{
    std::vector<WorldData>             shell_list;
    std::unique_ptr<lcs::NewtonSolver> solver_ptr;

    PyNewtonBuilder() = default;

    // register_mesh accepts numpy arrays (vertices Nx3, triangles Mx3)
    WorldDataWrapper register_mesh(const std::string& name,
                                   py::array_t<double, py::array::c_style | py::array::forcecast> vertices,
                                   py::array_t<int, py::array::c_style | py::array::forcecast> triangles)
    {
        // Validate shapes
        if (vertices.ndim() != 2 || vertices.shape(1) != 3)
            throw std::runtime_error("vertices must be a (N,3) array of floats");
        if (triangles.ndim() != 2 || triangles.shape(1) != 3)
            throw std::runtime_error("triangles must be a (M,3) array of ints");

        WorldData info;
        info.set_name(name);

        const ssize_t nverts = vertices.shape(0);
        const ssize_t nfaces = triangles.shape(0);

        // copy vertices
        info.input_mesh.model_positions.resize(nverts);
        auto buf_v = vertices.unchecked<2>();
        for (ssize_t i = 0; i < nverts; ++i)
        {
            SimMesh::Float3 p;
            p[0]                               = static_cast<float>(buf_v(i, 0));
            p[1]                               = static_cast<float>(buf_v(i, 1));
            p[2]                               = static_cast<float>(buf_v(i, 2));
            info.input_mesh.model_positions[i] = p;
        }

        // copy faces
        info.input_mesh.faces.resize(nfaces);
        auto buf_t = triangles.unchecked<2>();
        for (ssize_t i = 0; i < nfaces; ++i)
        {
            SimMesh::Int3 f;
            f[0]                     = static_cast<unsigned int>(buf_t(i, 0));
            f[1]                     = static_cast<unsigned int>(buf_t(i, 1));
            f[2]                     = static_cast<unsigned int>(buf_t(i, 2));
            info.input_mesh.faces[i] = f;
        }

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

        solver_ptr = std::make_unique<lcs::NewtonSolver>();
        solver_ptr->init_solver(*g_state.device, *g_state.stream, shell_list);
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

    py::list get_simulation_results()
    {
        if (!solver_ptr)
            throw std::runtime_error("Solver not initialized. Call init_solver() first.");

        auto&                                          mesh_data  = solver_ptr->get_host_mesh_data();
        const uint                                     num_meshes = mesh_data.num_meshes;
        std::vector<std::vector<std::array<float, 3>>> out_positions(num_meshes);
        // allocate per-mesh sizes
        for (uint i = 0; i < num_meshes; ++i)
        {
            const uint start = mesh_data.prefix_num_verts[i];
            const uint end   = mesh_data.prefix_num_verts[i + 1];
            out_positions[i].resize(end - start);
        }

        solver_ptr->get_simulation_results_to_host(out_positions);

        py::list py_out;
        for (uint i = 0; i < num_meshes; ++i)
        {
            const auto&          mesh  = out_positions[i];
            std::vector<ssize_t> shape = {(ssize_t)mesh.size(), 3};
            py::array_t<float>   arr(shape);
            auto                 buf = arr.mutable_unchecked<2>();
            for (ssize_t r = 0; r < (ssize_t)mesh.size(); ++r)
            {
                buf(r, 0) = mesh[r][0];
                buf(r, 1) = mesh[r][1];
                buf(r, 2) = mesh[r][2];
            }
            py_out.append(arr);
        }
        return py_out;
    }
};

PYBIND11_MODULE(lcs_solver_py, m)
{
    py::enum_<lcs::Initializer::SimulationType>(m, "SimulationType")
        .value("Cloth", lcs::Initializer::SimulationType::SimulationTypeCloth)
        .value("Tetrahedral", lcs::Initializer::SimulationType::SimulationTypeTetrahedral)
        .value("Rigid", lcs::Initializer::SimulationType::SimulationTypeRigid)
        .value("Rod", lcs::Initializer::SimulationType::SimulationTypeRod)
        .export_values();

    py::class_<ClothMaterial>(m, "ClothMaterial")
        .def(py::init<>())
        .def_readwrite("thickness", &ClothMaterial::thickness)
        .def_readwrite("youngs_modulus", &ClothMaterial::youngs_modulus)
        .def_readwrite("poisson_ratio", &ClothMaterial::poisson_ratio)
        .def_readwrite("area_bending_stiffness", &ClothMaterial::area_bending_stiffness)
        .def_readwrite("mass", &ClothMaterial::mass)
        .def_readwrite("density", &ClothMaterial::density);

    py::class_<MakeFixedPointsInterface>(m, "MakeFixedPointsInterface")
        .def(py::init<>())
        .def_readwrite("method", &MakeFixedPointsInterface::method)
        .def_readwrite("range", &MakeFixedPointsInterface::range);

    py::class_<WorldDataWrapper>(m, "WorldData")
        .def("set_name", &WorldDataWrapper::set_name)
        .def("set_simulation_type", &WorldDataWrapper::set_simulation_type)
        .def("set_physics_material_cloth",
             &WorldDataWrapper::set_physics_material_cloth,
             py::arg("thickness")      = 0.002f,
             py::arg("youngs_modulus") = 1e5f,
             py::arg("poisson_ratio")  = 0.25f)
        .def("add_fixed_point_info", &WorldDataWrapper::add_fixed_point_info)
        .def("set_translation", &WorldDataWrapper::set_translation)
        .def("set_rotation", &WorldDataWrapper::set_rotation)
        .def("set_scale", &WorldDataWrapper::set_scale)
        .def("load_mesh_data", &WorldDataWrapper::load_mesh_data);

    py::class_<PyNewtonBuilder>(m, "NewtonSolver")
        .def(py::init<>())
        .def("register_mesh", &PyNewtonBuilder::register_mesh, py::arg("name"), py::arg("vertices"), py::arg("triangles"))
        .def("num_meshes", &PyNewtonBuilder::num_meshes)
        .def("get_mesh_names", &PyNewtonBuilder::get_mesh_names)
        .def("init_solver", &PyNewtonBuilder::init_solver, "Initialize the underlying solver using previously created device/context")
        .def("physics_step_cpu", &PyNewtonBuilder::physics_step_cpu)
        .def("physics_step_gpu", &PyNewtonBuilder::physics_step_gpu)
        .def("get_simulation_results", &PyNewtonBuilder::get_simulation_results);

    m.def("init",
          &py_init,
          py::arg("backend_name") = py::none(),
          py::arg("binary_path")  = py::none(),
          "Initialize luisa compute context/device/stream from Python.\n\n"
          "backend_name: optional backend string (e.g. 'cuda','metal','dx')\n"
          "binary_path: optional binary path (argv[0]) to pass to luisa::compute::Context");

    m.doc() = "Python bindings for basic NewtonSolver scene building (lightweight)";
}
