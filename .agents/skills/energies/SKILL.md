## Adding a New Energy to the System

This guide walks through all the steps needed to introduce a new energy term — from the mathematical implementation to system-level integration and testing.

---

### Step 1 — Implement the Energy in `Solver/Energies/detail/`

Add your energy implementation in `Solver/Energies/detail/*_energy.hpp` (or `*_constraint.hpp`).

Because the system has both **device (GPU shader)** and **host (CPU)** execution paths, all functions must be written in **template form** so they work with both LuisaCompute types (`luisa::compute::Var<float>`, `luisa::compute::Var<float3>`, `luisa::compute::Var<float3x3>`) and plain C++ types (`float`, `float3`, `float3x3`).

Key conventions:
- Define an `Input<ScalarT, Vec3T>` struct that holds the per-element inputs (positions, stiffness, rest state, etc.).
- Provide a `compute_energy(...)` function for the scalar energy (used in the energy-only shader).
- Provide an `evaluate(input, identity)` function that returns an `EnergyEvalResult<GradCount, HessCount, GradT, HessT>` (or its aliases `EdgeEvalResult`, `SingleVertexEvalResult`) with `.gradients` and `.hessians` arrays.
- Store the result in the type aliases from [`energy_detail_common.hpp`](/Solver/Energies/detail/energy_detail_common.hpp):
  - `SingleVertexEvalResult<GradT, HessT>` — 1 gradient, 1 hessian
  - `EdgeEvalResult<GradT, HessT>` — 2 gradients, 4 hessians (2 vertices)
  - Or define your own `EnergyEvalResult<GradCount, HessCount, GradT, HessT>` for larger systems

Reference implementations:
- [`hookean_spring_energy.hpp`](/Solver/Energies/detail/hookean_spring_energy.hpp) — spring between two vertices; $6 \times 1$ gradient (2 `Float3`) and $6 \times 6$ hessian (4 `Float3x3`).
- [`fixed_joint_constaint.hpp`](/Solver/Energies/detail/fixed_joint_constaint.hpp) — fixed joint between two rigid (affine) bodies; $24 \times 1$ gradient (8 `Float3`) and $24 \times 24$ hessian (64 `Float3x3`), since each affine body has 12 DOF.

For joint-like rigid constraints, follow the current body-local convention:
- Store positional rest relation in body-A local frame (`rest_position_delta`), then enforce runtime target as `A * rest_position_delta`.
- If relative orientation is locked (fixed/prismatic), store rest relative rotation columns (`rest_rot_col0/1/2_a_to_b`) and use `B - A * R_ab0` residuals.
- Keep `compute_energy(...)` and `evaluate(...)` mathematically consistent (same residual definition) across device and host paths.

---

### Step 2 — Unit-Test Gradient & Hessian

Add a test case in [`UnitTest/test_gradient_hessian.cpp`](/UnitTest/test_gradient_hessian.cpp) to verify that your symbolic gradient and Hessian match the central-difference approximation.

> **Important:** You must implement the gradient/hessian symbolically — do **not** use central differences as the production implementation.

Enable tests at build time:

```bash
cmake -S . -B build -D LCS_ENABLE_TEST=ON
```

---

### Step 3 — Declare the Constitution Data Struct

In [`Solver/SimulationCore/simulation_data.h`](/Solver/SimulationCore/simulation_data.h), inside `namespace lcs::Constitutions`:

**3a.** Add an entry to the `ConstraintType` enum and its `to_string()` switch:

```cpp
enum class ConstraintType { ..., MyEnergy };

static inline std::string_view to_string(ConstraintType type) {
    case ConstraintType::MyEnergy: return "My Energy";
}
```

**3b.** Define a new `struct` templated on `BufferType`, inheriting from `ConstitutionInterface`. You must implement:
- `constraint_indices` — the per-element DOF indices
- Any per-element parameters (rest-state values, stiffness, etc.)
- `static constexpr size_t get_num_verts_per_constaint()` — number of DOF nodes per element (N)
- `static constexpr ConstraintType constraint_type()` — the enum value from 3a
- `get_indices_impl()` — returns a reference to `constraint_indices`

```cpp
template <template <typename...> typename BufferType>
struct MyEnergy : ConstitutionInterface<BufferType, MyEnergy<BufferType>>
{
    BufferType<uint2> constraint_indices;       // e.g., edge (2 vertices)
    BufferType<float> sa_rest_length;
    BufferType<float> sa_stiffness;

    static constexpr size_t         get_num_verts_per_constaint() { return 2; }
    static constexpr ConstraintType constraint_type() { return ConstraintType::MyEnergy; }
    auto& get_indices_impl() const { return constraint_indices; }
    auto& get_indices_impl()       { return constraint_indices; }
};
```

The base class `ConstitutionInterface` automatically provides:
- `constraint_gradients` — `N` `float3` per element
- `constraint_hessians` — `N*N` `float3x3` per element
- `constraint_offsets_in_adjlist` — triplet offsets for off-diagonal Hessian assembly
- `vert_adj_constraints` / `vert_adj_constraints_csr` — adjacency list

**3c.** Add a private member and public getter to `SimulationData<BufferType>`:

```cpp
// in private section:
Constitutions::MyEnergy<BufferType> my_energy;

// in public section:
Constitutions::MyEnergy<BufferType>&       get_my_energy_data()       { return my_energy; }
const Constitutions::MyEnergy<BufferType>& get_my_energy_data() const { return my_energy; }
```

---

### Step 4 — Add an Energy Offset Slot

In [`Solver/Energies/energy_offsets.h`](/Solver/Energies/energy_offsets.h), add a new unique index and update `num_energy_slots`:

```cpp
constexpr uint32_t offset_my_energy = 10; // next available index
constexpr uint32_t num_energy_slots = 11; // must be >= all offsets + 1
```

---

### Step 5 — Create the Energy Class (`.h` + `.cpp`)

Create `Solver/Energies/my_energy.h` and `my_energy.cpp` with a class inheriting from `Energy`:

**Header (`my_energy.h`)**:
```cpp
class MyEnergy : public Energy
{
public:
    MyEnergy(luisa::compute::BufferView<float> sa_system_energy) noexcept;
    void   compile(AsyncCompiler& compiler) override;
    void   device_compute_energy(luisa::compute::Stream& stream) override; // no-op base
    void   device_compute_energy(luisa::compute::Stream& stream,
               const Constitutions::MyEnergy<luisa::compute::Buffer>& data,
               size_t dispatch_count);
    void   device_evaluate(luisa::compute::Stream& stream,
               const Constitutions::MyEnergy<luisa::compute::Buffer>& data,
               size_t dispatch_count);
    double host_evaluate(const std::vector<float>& host_energy) override;
    void   host_evaluate(lcs::SimulationData<std::vector>& host_sim_data,
                         lcs::MeshData<std::vector>& host_mesh_data);

private:
    luisa::compute::BufferView<float> _sa_system_energy;
    luisa::compute::Shader<1, Constitutions::MyEnergy<luisa::compute::Buffer>> _shader;
    luisa::compute::Shader<1, Constitutions::MyEnergy<luisa::compute::Buffer>> _eval_shader;
};
```

**Implementation (`my_energy.cpp`)**:

`compile()` must compile **two shaders**:

1. **`_shader`** (energy-only): reads per-element data, computes scalar energy, atomically accumulates into `sa_system_energy[offset_my_energy]` using warp-reduce.
2. **`_eval_shader`** (gradient/hessian): reads per-element data, calls `detail::my_energy::evaluate(...)`, writes results into `constraint.constraint_gradients` and `constraint.constraint_hessians`.

`host_evaluate(host_sim_data, host_mesh_data)` must mirror the eval shader on CPU using `CpuParallel::parallel_for`, filling the same `constraint_gradients` / `constraint_hessians` vectors.

`host_evaluate(host_energy)` just returns `host_energy[offset_my_energy]`.

See [`spring_energy.cpp`](/Solver/Energies/spring_energy.cpp) and [`joint_constraint_energy.cpp`](/Solver/Energies/joint_constraint_energy.cpp) for complete reference implementations.

---

### Step 6 — Initialize Data in `init_sim_data.cpp`

In [`Solver/Initializer/init_sim_data.cpp`](/Solver/Initializer/init_sim_data.cpp), there are three initialization phases where the new energy must be registered:

**6a. Buffer allocation** — resize all per-element buffers based on count:
```cpp
auto& my_data = sim_data->get_my_energy_data();
my_data.constraint_indices.resize(num_elements);
my_data.sa_rest_length.resize(num_elements);
my_data.sa_stiffness.resize(num_elements);
// fill values...
```

**6b. Adjacency list + gradient/hessian pre-allocation** — call `build_adj_list_and_init_grad_hess`, which:
- Builds `vert_adj_constraints` (which constraints reference each DOF vertex)
- Pre-allocates `constraint_gradients` (N float3/element) and `constraint_hessians` (N² float3x3/element)

```cpp
auto& my_data = sim_data->get_my_energy_data();
build_adj_list_and_init_grad_hess(adj_map, my_data);
```

**6c. Triplet offsets** — call `init_constitution_offsets_in_adjlist`, which fills `constraint_offsets_in_adjlist` (N*(N-1) offsets per element) used by the GPU assembly shader to locate off-diagonal Hessian positions:
```cpp
auto& my_data = sim_data->get_my_energy_data();
init_constitution_offsets_in_adjlist(adj_list, csr, my_data);
```

**6d. GPU upload** — upload all CPU buffers to GPU buffers using `upload_buffer(device, gpu_buf, cpu_buf)`:
```cpp
auto& my_I = input_data->get_my_energy_data();
auto& my_O = output_data->get_my_energy_data();
if (my_I.is_valid()) {
    stream
        << upload_buffer(device, my_O.constraint_indices, my_I.constraint_indices)
        << upload_buffer(device, my_O.sa_rest_length, my_I.sa_rest_length)
        << upload_buffer(device, my_O.sa_stiffness, my_I.sa_stiffness)
        << upload_buffer(device, my_O.constraint_gradients, my_I.constraint_gradients)
        << upload_buffer(device, my_O.constraint_hessians, my_I.constraint_hessians)
        << upload_buffer(device, my_O.constraint_offsets_in_adjlist, my_I.constraint_offsets_in_adjlist)
        << upload_buffer(device, my_O.vert_adj_constraints_csr, my_I.vert_adj_constraints_csr);
}
```

For joint constraints specifically, ensure all body-local rest fields are uploaded:
- `rest_position_delta`
- `rest_rot_col0_a_to_b`
- `rest_rot_col1_a_to_b`
- `rest_rot_col2_a_to_b`

---

### Step 7 — Register in SolverInterface

In [`Solver/SimulationSolver/solver_interface.h`](/Solver/SimulationSolver/solver_interface.h):
- Add `std::unique_ptr<MyEnergy> my_energy;` to private members
- Add `MyEnergy* get_my_energy() const { return my_energy.get(); }` accessor

In [`Solver/SimulationSolver/solver_interface.cpp`](/Solver/SimulationSolver/solver_interface.cpp), inside `compile_energies()`:
```cpp
my_energy = std::make_unique<MyEnergy>(sim_data->sa_system_energy.view());
my_energy->compile(compiler);
```

Inside `device_compute_elastic_energy()`:
```cpp
const auto& my_data = sim_data->get_my_energy_data();
if (my_data.is_valid())
    my_energy->device_compute_energy(stream, my_data, my_data.get_num_indices());
```

---

### Step 8 — Register in Newton Solver Assembly

In [`Solver/SimulationSolver/newton_solver.h`](/Solver/SimulationSolver/newton_solver.h), add the assembly shader member:
```cpp
luisa::compute::Shader<1, Constitutions::MyEnergy<luisa::compute::Buffer>, uint>
    fn_material_energy_assembly_my_energy;
```

In [`Solver/SimulationSolver/newton_solver.cpp`](/Solver/SimulationSolver/newton_solver.cpp), inside `compile()`:
```cpp
compiler.compile(fn_material_energy_assembly_my_energy,
    [perform_assembly_interface](Var<Constitutions::MyEnergy<luisa::compute::Buffer>> constraint,
                                 const Uint prefix)
    { perform_assembly_interface(constraint, prefix); });
```

In the **device Newton step** (gradient/hessian eval + assembly):
```cpp
const auto& my_data = sim_data->get_my_energy_data();
if (my_data.is_valid()) {
    get_my_energy()->device_evaluate(stream, my_data, my_data.get_num_indices());
    stream << fn_material_energy_assembly_my_energy(my_data, dof_prefix)
                .dispatch(my_data.constraint_offsets_in_adjlist.size());
}
```

In the **host Newton step** (CPU fallback path):
```cpp
auto& my_data = host_sim_data->get_my_energy_data();
if (my_data.is_valid())
    CpuParallel::parallel_for(0, num_dof, [&](const uint vid) {
        assembly_template2(vid, my_data, adj_verts, sa_cgB, sa_cgA_diag, sa_cgA_offdiag_triplet);
    });
```

> **Note:** Before CPU assembly, `host_evaluate(host_sim_data, host_mesh_data)` must be called to fill `constraint_gradients` and `constraint_hessians`. This is handled automatically by the existing `host_evaluate_all_energies()` call in the Newton loop.

---

### Step 9 — Expose to Python Bindings and Write a Test

**9a. Expose the API in Python bindings**

If the new energy needs to be configurable from Python (e.g., a user-specified constraint like joints), add a binding method to the `NewtonSolver` Python wrapper. Follow the pattern of `add_fixed_joint` / `add_prismatic_joint` / `add_revolute_joint` for how constraint descriptors are passed in.

**9b. Write a Python integration test**

Add a test in `PythonBindings/tests/test_my_energy.py`. The standard structure is:

```python
import os, sys
import numpy as np
import trimesh

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(root, "build", "bin"))
import lcs_py as lcs
import utils.arg_parser

args = utils.arg_parser.parse_args()
solver = lcs.NewtonSolver()
solver.init_device(backend_name=args.backend)

# --- Scene setup ---
# Load meshes, create and register bodies, add your new energy/constraint, e.g.:
#   solver.add_my_constraint(body_a_id, body_b_id, ...)

solver.init_solver()

config_ref = solver.get_config()
config_ref.use_floor = True  # adjust as needed

output_dir = os.path.join(root, "Resources", "OutputMesh")
os.makedirs(output_dir, exist_ok=True)

# --- Headless or GUI ---
if args.headless:
    solver.save_sim_result(os.path.join(output_dir, "my_energy_init.obj"))
    for frame in range(args.advance_frames):
        if config_ref.use_gpu:
            solver.physics_step_gpu()
        else:
            solver.physics_step_cpu()
    solver.save_sim_result(os.path.join(output_dir, "my_energy_result.obj"))
else:
    import utils.polyscope_gui
    gui = utils.polyscope_gui.SimulationGUI(solver, config_ref, output_dir)
    gui.show()

solver.cleanup_device()
```

Run headless:

```bash
<LCS_PYTHON_EXECUTABLE> PythonBindings/tests/test_my_energy.py --headless
```

Key conventions (from existing tests):
- Always call `solver.init_device()` before any scene setup and `solver.cleanup_device()` at the end.
- Always call `solver.init_solver()` after all bodies and constraints are registered.
- Use `config_ref = solver.get_config()` to access/modify simulation parameters after `init_solver()`.
- Use `config_ref.use_gpu` to branch between `physics_step_gpu()` and `physics_step_cpu()`.
- Use `args = utils.arg_parser.parse_args()` for consistent `--backend`, `--headless`, and `--advance_frames` CLI args.
- Save the initial state **before** the simulation loop to capture the rest configuration.

Reference: [`test_rigid_joint_animation.py`](/PythonBindings/tests/test_rigid_joint_animation.py) for a complete example with multiple joint types, per-frame animation callbacks, and both headless and GUI modes.
