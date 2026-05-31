---
name: overview
description: Use when onboarding to LuisaComputeSimulator, answering project architecture questions, locating files, building/running the simulator, or writing Python frontend scripts.
---

# LuisaComputeSimulator Project Overview

Use this skill first when you need a fast mental model of the repository. LuisaComputeSimulator is a C++20 physics simulator built on LuisaCompute with pybind11 Python bindings. It simulates cloth, soft bodies, tetrahedral bodies, rigid affine-body dynamics, joints, collision, friction, and IPC-style penetration-free contact across CUDA, Metal, DirectX, Vulkan, and fallback/CPU paths.

## Architecture At A Glance

The project is organized around a small set of layers:

- **Frontend layer:** Python scripts in `PythonBindings/tests/` and C++ demos in `Application/` construct scenes, configure `SceneParams`, and call `NewtonSolver`.
- **Binding layer:** `PythonBindings/src/python_bindings.cpp` exposes the C++ solver, world-data builder, material setters, animation updates, joints, config, and result extraction as `lcs_py`.
- **Solver orchestration layer:** `Solver/SimulationSolver/solver_interface.*` owns scene data, device state, initialization, energy objects, collision helpers, animation state, and save/query APIs. `Solver/SimulationSolver/newton_solver.*` implements time stepping, Newton iteration, assembly, PCG solve, collision update, CCD/energy line search, and CPU/GPU step variants.
- **Initialization layer:** `Solver/Initializer/` converts registered `WorldData` into contiguous mesh, simulation, collision, DOF, adjacency, and GPU-buffer data.
- **Physics data layer:** `Solver/SimulationCore/` defines `WorldData`, `SceneParams`, material types, simulation buffers, collision buffers, fixed/joint descriptors, and topology-independent simulation state.
- **Energy layer:** `Solver/Energies/` computes material, inertia, bending, tet, ABD, ground, and joint energies plus gradients/Hessians for Newton assembly.
- **Collision layer:** `Solver/CollisionDetector/` provides LBVH broad phase, narrow phase CCD/DCD, contact/friction energy evaluation, and contact Hessian triplet assembly.
- **Linear algebra/runtime layer:** `Solver/Core/`, `Solver/LinearSolver/`, and `Solver/Utils/` hold math types, matrix triplets, SVD helpers, PCG, parallel loops, buffer utilities, profiler, and Luisa runtime helpers.

## Simulation Pipeline

Python and C++ frontends follow the same lifecycle:

1. Create `NewtonSolver`.
2. Initialize the compute backend with `init_device(backend_name="cuda" | "metal" | "dx" | "vk" | "fallback")`.
3. Create one or more `WorldData` objects from OBJ files, NumPy arrays, or tetrahedral arrays.
4. Set material type and material parameters, transforms, fixed points, default animations, and optional joints.
5. Register each `WorldData` with `register_world_data()`.
6. Edit `config = solver.get_config()` for solver parameters such as `use_gpu`, `use_floor`, `use_self_collision`, `implicit_dt`, `nonlinear_iter_count`, `pcg_iter_count`, and line-search flags.
7. Call `init_solver()` once after all bodies and constraints are registered. This sorts world data, initializes mesh/simulation/collision buffers, uploads GPU buffers, builds animation metadata, compiles Luisa shaders, compiles energy kernels, compiles LBVH/narrow phase/PCG kernels, and restarts the system.
8. Advance simulation with `physics_step_gpu()` or `physics_step_cpu()`.
9. Extract results with `get_sim_result()`, `get_object_sim_result_by_registration_id()`, rigid-body query methods, or `save_sim_result(path)`.
10. Call `cleanup_device()` when a script owns the device and is done.

Inside a simulation step:

- `SolverInterface::physics_step_prev_operation()` updates pinned vertex/body animation targets and copies outer state to step-start state.
- `NewtonSolver::predict_position()` computes predicted positions/DOFs from velocity, gravity, and substep time.
- Collision data is reset and, when self-collision is enabled, LBVH broad phase and narrow phase update active contact sets.
- Energy objects evaluate gradients and Hessians into per-constraint buffers.
- Material and contact Hessians are assembled into diagonal `sa_cgA_diag`, off-diagonal triplets, and RHS `sa_cgB`.
- PCG solves the Newton system, producing `sa_cgX`/search direction.
- CCD line search clamps motion to avoid new penetrations; energy line search optionally enforces descent.
- Positions, velocities, and outer state are updated in `physics_step_post_operation()`.

Important implementation files for the pipeline:

- `Solver/SimulationSolver/solver_interface.cpp`: data initialization, world sorting, animation preparation, device lifecycle, result extraction, state save/load.
- `Solver/SimulationSolver/newton_solver.cpp`: CPU/GPU physics steps, Newton loop, material assembly, contact update, line search, PCG SpMV.
- `Solver/Initializer/init_mesh_data.*`: mesh/topology preprocessing.
- `Solver/Initializer/init_sim_data.*`: DOF layout, energies/constraints, adjacency lists, rest data, upload of simulation buffers.
- `Solver/Initializer/init_collision_data.*`: active contact vertices/faces/edges and collision data setup.
- `Solver/Energies/energy.h` and individual energy files: energy object interface and concrete kernels.
- `Solver/CollisionDetector/narrow_phase.*` and `lbvh.*`: collision queries and contact contribution assembly.

## Directory And File Guide

- `.Agents/skills/`: agent onboarding skills. Existing topics include build, energies, and Python stubs. This file is the architecture overview skill.
- `.agents/skills/`: lowercase mirror/alternate agent-skill location with similar existing skills.
- `.github/workflows/`: CI for CMake, xmake, and wheel builds.
- `.vscode/`: local editor settings and compile-command references.
- `Application/`: C++ executable frontends. `app_simulation.cpp` runs JSON scene demos. `app_integration.cpp` is the C++ integration example. `app_simulation_demo_config.*` loads scene JSON/config into solver/world data.
- `Document/`: human docs. `Build.md` is the primary build guide. `PythonAPI.md` documents Python bindings. `CppAPI.md` documents C++ usage. `Energies.md` documents energy models. `dfdx_codegen_utils.py` supports derivative/codegen experiments.
- `PythonBindings/`: pybind11 module and Python frontend tests. `src/python_bindings.cpp` defines the `lcs_py` module. `python/lcs_py/__init__.pyi` is generated stub output. `cmake/normalize_stub.cmake` normalizes stubgen output. `tests/` contains Python scenes and smoke tests.
- `PythonBindings/tests/utils/`: Python frontend helpers. `arg_parser.py` standardizes `--backend`, `--headless`, and `--advance_frames`. `polyscope_gui.py` implements interactive Polyscope GUI. Animation helpers include `vertex_animator.py`, `body_animator.py`, `smpl_animator.py`, and `animation_transform.py`. `mesh_proc.py` contains mesh-processing helpers.
- `Resources/`: assets and outputs. `InputMesh/` contains OBJ meshes. `Scenes/` contains JSON demo scenes for the C++ application and Python `load_scene_from_json()`. `OutputMesh/` and `SimulationState/` hold generated outputs/state snapshots.
- `Solver/`: core simulator implementation.
- `Solver/CollisionDetector/`: LBVH, AABB, narrow phase, CCD/DCD, contact/friction kernels.
- `Solver/Core/`: Luisa/Eigen-compatible math types, matrix triplets, scalar helpers, affine-body helpers, SVD utilities.
- `Solver/Energies/`: energy classes and kernels. `detail/` contains template math implementations shared by host and device paths. Key files include `spring_energy.*`, `stretch_face_energy.*`, `bending_energy_kernel.*`, `tet_elastic_energy.*`, `abd_inertia_energy.*`, `abd_ortho_energy.*`, `soft_inertia_energy.*`, `ground_collision_energy.*`, and `joint_constraint_energy.*`.
- `Solver/Initializer/`: converts high-level `WorldData` into mesh, simulation, constraint, adjacency, collision, and GPU buffer data.
- `Solver/LinearSolver/`: preconditioned conjugate-gradient solver.
- `Solver/MeshOperation/`: OBJ/tet mesh loading and default mesh helpers.
- `Solver/SimulationCore/`: `WorldData`, `SceneParams`, materials, base mesh, simulation type enums, simulation/collision data, and joint descriptors.
- `Solver/SimulationSolver/`: solver interfaces and concrete `NewtonSolver` implementation.
- `Solver/Scripts/`: profiling analysis utilities.
- `Solver/Utils/`: async shader compiler wrapper, buffer allocation/fill/add helpers, CPU/device parallel utilities, profiler, runtime helpers.
- `UnitTest/`: C++ tests for collision, LBVH, energy assembly, gradients/Hessians, integration, PCG, JIT, profiling, and solver behavior.
- `ext/`: third-party dependencies and submodules, including LuisaCompute, Eigen, and lcpp depending on setup.
- `build/`: generated build output, usually including `build/bin/lcs_py.*` and app binaries.
- `output/`: additional generated output location.
- `CMakeLists.txt`: top-level CMake build.
- `xmake.lua` and `setup.lua`: xmake build and dependency setup.
- `pyproject.toml`: scikit-build Python package metadata for editable installs/wheels.
- `lcs_config_template.ini` and `lcs_config.ini`: local build configuration hints used by agent build workflows.

## Build And Configure Guidance

Prefer the existing build skill for exact local conventions, but the common workflows are below.

For CMake C++ build (Recomended):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

For CMake with Python bindings:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install scikit-build-core pybind11 ninja numpy pybind11-stubgen trimesh polyscope
cmake -S . -B build -DLCS_BUILD_PYBINDINGS=ON -DLCS_PYTHON_EXECUTABLE="$(pwd)/.venv/bin/python"
cmake --build build -j --target stubs
.venv/bin/python -m pip install -e .
```

For xmake with Python bindings:

```bash
xmake lua setup.lua
xmake f -c -m release --lcs_build_pybindings=y --lcs_python_executable="$(pwd)/.venv/bin/python"
xmake build lcs_py
xmake build stubs
```

Backend notes:

- Runtime backend strings are usually `cuda`, `metal`, `dx`, `vk`, and `fallback`.
- macOS normally uses `metal`; CUDA is not available there.
- CMake Python bindings require `-DLCS_BUILD_PYBINDINGS=ON`; otherwise `lcs_py` and `stubs` are not built.
- After editing `PythonBindings/src/python_bindings.cpp`, rebuild `stubs` so `PythonBindings/python/lcs_py/__init__.pyi` stays current.
- If Python import fails after xmake builds, run with `PYTHONPATH=build/bin` or install editable with CMake/scikit-build.
- The project-local `.venv` interpreter should match `LCS_PYTHON_EXECUTABLE`.

Useful verification commands:

```bash
PYTHONPATH=build/bin .venv/bin/python -c "import lcs_py; print('OK')"
.venv/bin/python PythonBindings/tests/test_rigid_joint_animation.py --headless --advance_frames 30
PYTHONPATH=build/bin .venv/bin/python PythonBindings/tests/test_rigid_joint_animation.py --headless --advance_frames 30
```

## Python Frontend Usage

Most active frontend examples live in `PythonBindings/tests/`. They are not only tests; they are practical scene scripts.

Representative scripts:

- `test_cloth_animation.py`: cloth setup and animation.
- `test_cloth_large_deform.py`: larger cloth deformation.
- `test_cloth_soft_rigid_coupling.py`: coupled cloth/soft/rigid setup.
- `test_cloth_garment_animation.py`: garment/SMPL-style animation utilities.
- `test_rigid_animation.py`: animated rigid bodies.
- `test_rigid_folding.py`: rigid folding scene.
- `test_rigid_joint_animation.py`: fixed, prismatic, and revolute joint integration example.
- `test_rigid_joint_animation2.py`: larger joint showcase.
- `test_soft_folding.py`: soft-body scene.
- `test_tet_unit.py` and `test_tet_drop.py`: tetrahedral smoke tests.
- `test_load_from_json.py`: Python entry point for JSON scene loading.

Standard Python script pattern:

```python
import os, sys
import platform
import numpy as np
import argparse
import trimesh
import lcs_py as lcs

parser = argparse.ArgumentParser(description="LuisaCompute Python example")
parser.add_argument("--backend",type=str,default="metal" if platform.system() == "Darwin" else "cuda")
parser.add_argument("--headless",action="store_true")
parser.add_argument("--advance_frames",type=int,default=30)
args = parser.parse_args()

solver = lcs.NewtonSolver()
solver.init_device(backend_name=args.backend)

cube_vertices = np.array([
    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0], 
    [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]  
], dtype=np.float32)
cube_triangles = np.array([
    [0, 1, 2],[0, 2, 3], [0, 4, 5], [0, 5, 1],
    [1, 5, 6], [1, 6, 2], [3, 2, 6], [3, 6, 7],
    [0, 7, 4],[0, 3, 7], [4, 7, 6], [4, 6, 5] 
], dtype=np.int32)

# Or just from trimesh
# cube_mesh = trimesh.load(cube_mesh_path, process=False)
# cube_mesh = trimesh.Trimesh(vertices=cube_vertices, faces=cube_triangles, process=False)
# cube_vertices = cube_mesh.vertices
# cube_triangles = cube_mesh.faces

count = 0
for i in range(5):
    for j in range(5):
        for k in range(5):
            name = f'cube_{i}_{j}_{k}'
            cube = solver.create_world_data_from_array(name, cube_vertices, cube_triangles)
            # cube = solver.create_world_data_from_file_path(name, your_path)
            cube.set_simulation_type(lcs.MaterialType.Rigid)
            cube.set_scale(0.1)
            delta = 0.102
            cube.set_translation(i * delta, 0.1 + j * delta, k * delta)
            solver.register_world_data(cube)
            count += 1
solver.init_solver()
solver.print_registered_meshes_info()
config_ref = solver.get_config()

output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)

if args.headless:
	solver.save_sim_result(obj_path=os.path.join(output_dir, "init.obj"))
	for frame in range(0, args.advance_frames):
		solver.physics_step_gpu()
	solver.save_sim_result(obj_path=os.path.join(output_dir, "result.obj"))
else:
	from lcs_gui import SimulationGUI 
	gui = SimulationGUI(solver, config_ref, output_dir)
	gui.show()

solver.cleanup_device()
```

Python frontend rules that prevent common failures:

- Call `solver.init_device()` before creating/running device-dependent solver work.
- Register all objects and joints before `solver.init_solver()`.
- Call `solver.init_solver()` exactly after scene setup; it sorts world data, so use registration IDs rather than sorted indices in Python APIs.
- Use `config = solver.get_config()` and branch with `config.use_gpu` when deciding between `physics_step_gpu()` and `physics_step_cpu()`.
- For GUI mode, use `utils.polyscope_gui.SimulationGUI`; for automation and CI, pass `--headless --advance_frames N`.
- Use `sys.path.insert(0, os.path.join(root, "build", "bin"))` or `PYTHONPATH=build/bin` when using an uninstalled build.
- Use `.venv/bin/python -m pip install -e .` after CMake/scikit-build if you want the package available in the active venv without manual `PYTHONPATH`.
- Save initial output before stepping if you need rest-state debugging.
- Prefer `get_object_sim_result_by_registration_id(registration_id)` for per-object validation and `get_sim_result()` for all meshes.

## JSON Scene Frontends

`Application/app_simulation.cpp` runs C++ JSON scenes from `Resources/Scenes/`. Python can load the same scene format through `solver.load_scene_from_json(json_path)` before `init_solver()`.

Example commands:

```bash
./build/bin/app_simulation metal Resources/Scenes/default_scene.json
./build/bin/app_simulation cuda Resources/Scenes/cloth_rigid_coupling_high_res.json
```

For Python JSON loading, inspect `PythonBindings/tests/test_load_from_json.py` and `Application/app_simulation_demo_config.*`.

## Where To Make Common Changes

- Add/modify Python API: edit `PythonBindings/src/python_bindings.cpp`, rebuild `lcs_py`, run `stubs`, update `Document/PythonAPI.md` if needed.
- Add a Python example/test: add a script in `PythonBindings/tests/`, use `utils.arg_parser.parse_args()`, support headless mode, save outputs under `Resources/OutputMesh/`.
- Change scene parameters: edit `Solver/SimulationCore/scene_params.*`, bindings in `python_bindings.cpp`, and JSON loader in `Application/app_simulation_demo_config.*` if the setting should load from scenes.
- Add a material/energy/constraint: follow `.Agents/skills/energies/SKILL.md`; changes usually touch `Solver/Energies/`, `Solver/SimulationCore/simulation_data.h`, `Solver/Initializer/init_sim_data.cpp`, `Solver/SimulationSolver/solver_interface.*`, `Solver/SimulationSolver/newton_solver.*`, bindings, and tests.
- Change collision behavior: start in `Solver/CollisionDetector/narrow_phase.*`, then inspect collision use in `newton_solver.cpp` line-search/contact update paths.
- Change build flags or dependency setup: start with `CMakeLists.txt`, `PythonBindings/CMakeLists.txt`, `xmake.lua`, `setup.lua`, and `.Agents/skills/build/SKILL.md`.

## Fast Code-Reading Entry Points

- Start with `PythonBindings/tests/test_rigid_joint_animation.py` to understand Python frontend conventions.
- Then read `PythonBindings/src/python_bindings.cpp` to map Python calls to C++ methods.
- Then read `Solver/SimulationSolver/solver_interface.h` for public solver API and owned data.
- Then read `Solver/SimulationSolver/solver_interface.cpp` for initialization and result extraction.
- Then read `Solver/SimulationSolver/newton_solver.cpp` for the actual time-step algorithm.
- Use `Solver/SimulationCore/simulation_data.h` to understand packed buffers and constraint data layouts.
- Use `Solver/Initializer/init_sim_data.cpp` to understand how `WorldData` becomes solver buffers.
