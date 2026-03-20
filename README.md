# LuisaComputeSimulator 🧮⚡

<p align="center">
  <img src="Document/README1.png" alt="Teaser" width="600"/>
</p>

[![linux](https://github.com/ChengzhuUwU/LuisaComputeSimulator/actions/workflows/cmake_linux.yml/badge.svg?branch=main)](https://github.com/ChengzhuUwU/LuisaComputeSimulator/actions/workflows/cmake_linux.yml)
[![windows](https://github.com/ChengzhuUwU/LuisaComputeSimulator/actions/workflows/cmake_windows.yml/badge.svg?branch=main)](https://github.com/ChengzhuUwU/LuisaComputeSimulator/actions/workflows/cmake_windows.yml)
[![macos](https://github.com/ChengzhuUwU/LuisaComputeSimulator/actions/workflows/cmake_macos.yml/badge.svg?branch=main)](https://github.com/ChengzhuUwU/LuisaComputeSimulator/actions/workflows/cmake_macos.yml)
[![License](https://img.shields.io/github/license/ChengzhuUwU/LuisaComputeSimulator)](LICENSE)

LuisaComputeSimulator is a **high-performance cross-platform physics simulator** built on [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute). It provides real-time simulation of **soft bodies, cloth, and rigid bodies** with **penetration-free contact handling**, accelerated by GPU/CPU backends.

> **Performance Demo:** 88K vertices, 174K triangles, 3M+ collision pairs → **~3 FPS on RTX 3090 (CUDA)**, **~2 FPS on M2 Max (Metal)**

---

## ✨ Features

| Feature                            | Description                                                                           |
| ---------------------------------- | ------------------------------------------------------------------------------------- |
| **Soft Body / Cloth Simulation**   | High-resolution soft-body and cloth simulation with FEM-based constitutive models |
| **Rigid Body Dynamics**            | Rigid body simulation with collision and friction                                     |
| **Soft-Cloth-Rigid Coupling**      | Seamless interaction among soft bodies, cloth, and rigid bodies in one solver        |
| **Penetration-Free Contact (IPC)** | Robust collision handling using barrier functions                                     |
| **Affine Body Dynamics (ABD)**     | Efficient reduced-space simulation for rigid bodies                                   |
| **Multi-Backend Support**          | CUDA, DirectX 12, Vulkan, Metal, CPU (Fallback)                                       |
| **Python & C++ APIs**              | Flexible programming interfaces for different use cases                               |
| **Interactive GUI**                | Real-time visualization with Polyscope                                                |

### Supported Physics

- ✅ Soft Body / Cloth Simulation (Spring + ARAP FEM)
- ✅ Rigid Body Simulation  
- ✅ Soft-Rigid Body Coupling
- ✅ Cloth-Soft-Rigid Coupling
- ✅ Ground Collision
- ✅ Frictional Contact
- ✅ Continuous Collision Detection (CCD)
- ✅ Fixed Point / Pinned Constraints
- ✅ Tetrahedral Mesh (In Development)
- 🔄 Joint Constraints (Planned)
- 🔄 OpenUSD Scene Support

## Usage

### Python Frontend

Sample Python-frontend code can be found at:

- [test_cloth_animation](PythonBindings/tests/test_cloth_animation.py) (cloth twisting animation)
- [test_soft_folding.py](PythonBindings/tests/test_soft_folding.py) (soft-body tetrahedral simulation)
- [test_cloth_soft_rigid_coupling.py](PythonBindings/tests/test_cloth_soft_rigid_coupling.py) (cloth-soft-rigid coupling)

```python
    import os
    import sys
    import trimesh
    import numpy as np

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.insert(0, os.path.join(root, "build", "bin"))
    import lcs_py as lcs

    # Initialize solver with backend
    solver = lcs.NewtonSolver()
    solver.init_device(backend_name="metal")  # or cuda, dx, vk

    # Create rigid body from mesh file
    cube_mesh = trimesh.load("cube.obj", process=False)
    cube = solver.create_world_data_from_array("cube", cube_mesh.vertices, cube_mesh.faces)
    cube.set_simulation_type(lcs.MaterialType.Rigid)
    cube.set_translation(0.0, 0.34, 0.0)
    cube.set_rotation(0.5235988, 0.0, 0.5235988)
    cube.set_scale(0.1)
    cube_id = solver.register_world_data(cube)

    # Create cloth from file with different stretch models
    cloth = solver.create_world_data_from_file_path("cloth", "square2K.obj")
    cloth.set_simulation_type(lcs.MaterialType.Cloth)
    
    # Configure cloth material with different constitutive models:
    # - Spring: Basic linear spring energy
    # - FEM_BW98: Finite element method (stable for large deformations)
    cloth.set_physics_material_cloth(
        thickness=0.001, 
        youngs_modulus=1e6,
        poisson_ratio=0.3,
        stretch_model="FEM_BW98",  # or "Spring"
        bending_model="QuadraticBending"
    )
    cloth.set_scale(0.75)
    
    # Add fixed points
    cloth.add_fixed_point_by_method("LeftBack")
    cloth.add_fixed_point_by_method("RightBack")
    cloth.add_fixed_point_by_method("LeftFront")
    cloth.add_fixed_point_by_method("RightFront")
    cloth_id = solver.register_world_data(cloth)

    # Configure simulation
    config = solver.get_config()
    config.use_floor = False
    config.use_self_collision = False
    config.implicit_dt = 1/60

    # Initialize solver (compiles shaders, allocates buffers)
    solver.init_solver()

    # Run simulation
    for frame in range(100):
        solver.physics_step_gpu()
        solver.save_sim_result(f"output/frame_{frame}.obj")
    
    solver.cleanup_device()
```

### Cpp Frontend

Sample Cpp-frontend code can be found at [app_integration.cpp](Application/app_integration.cpp).

```C++
    #include <string>
    #include <vector>
    #include "SimulationSolver/newton_solver.h"

    int main(int argc, char** argv)
    {
        // Set log level
        luisa::log_level_info();

        // Parse backend from command line
        std::string backend = (argc >= 2) ? argv[1] : "";

        // Create solver instance
        lcs::NewtonSolver solver;
        solver.create_device(argv[0], backend);

        // ========================================================================
        // Cloth with FEM_BW98 (Finite Element Method) - large deformations
        // ========================================================================
        auto upper_square = lcs::Initializer::WorldData()
                                .set_name("upper square")
                                .load_mesh_from_path(std::string(LCSV_RESOURCE_PATH) + "/InputMesh/square2.obj")
                                .set_material_type(lcs::Material::MaterialType::Cloth)
                                .set_physics_material(lcs::Material::ClothMaterial{
                                    .stretch_model = lcs::Material::ConstitutiveStretchModelCloth::FEM_BW98,
                                    .bending_model = lcs::Material::ConstitutiveBendingModelCloth::DihedralAngle,
                                    .thickness = 0.001f,
                                    .youngs_modulus = 1e6f,
                                    .poisson_ratio = 0.3f,
                                })
                                .set_translation({ 0.0f, 0.4f, 0.0f });
        uint upper_square_id = solver.register_world_data(upper_square);

        // ========================================================================
        // Cloth with Spring model - linear, fast
        // ========================================================================
        std::vector<std::array<float, 3>> square_mesh_vertices{ { -0.5, 0, -0.5 }, { 0.5, 0, -0.5 }, { -0.5, 0, 0.5 }, { 0.5, 0, 0.5 } };
        std::vector<std::array<uint, 3>>  square_mesh_faces{ { 0, 3, 1 }, { 0, 2, 3 } };
        auto lower_square = lcs::Initializer::WorldData()
                                .set_name("lower square")
                                .load_mesh_from_array(square_mesh_vertices, square_mesh_faces)
                                .set_material_type(lcs::Material::MaterialType::Cloth)
                                .set_physics_material(lcs::Material::ClothMaterial{
                                    .stretch_model = lcs::Material::ConstitutiveStretchModelCloth::Spring
                                })
                                .set_scale(0.8f)
                                .set_translation({ 0.1f, 0.2f, 0.0f })
                                .add_fixed_point_from_method({ .method = lcs::Initializer::FixedPointsType::Left })
                                .add_fixed_point_from_method({ .method = lcs::Initializer::FixedPointsType::Right });
        uint lower_square_id = solver.register_world_data(lower_square);

        // Configure solver
        auto& config = solver.get_config();
        config.use_floor = false;
        config.use_self_collision = true;
        config.implicit_dt = 1.0 / 60.0;
        config.use_energy_linesearch = true;
        config.nonlinear_iter_count = 10;
        config.pcg_iter_count = 100;

        // Initialize solver (compiles shaders, allocates buffers)
        solver.init_solver();

        // Get initial vertices
        std::vector<std::vector<std::array<float, 3>>> rendering_vertices;
        solver.get_curr_vertices_to_host(rendering_vertices);

        // Main simulation loop
        for (uint ii = 0; ii < 20; ii++)
        {
            solver.physics_step_GPU();  // or solver.physics_step_CPU()
            solver.get_curr_vertices_to_host(rendering_vertices);
            // Render or process vertices here
        }

        // Save result
        solver.save_mesh_to_obj(luisa::format("{}/OutputMesh/sample.obj", LCSV_RESOURCE_PATH));

        return 0;
    }
```

## 🚀 Quick Start

### 1. Clone & Build

```bash
# Clone the repository
git clone https://github.com/ChengzhuUwU/LuisaComputeSimulator.git
cd LuisaComputeSimulator

# Configure and build (CMake)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Or use Xmake
xmake lua setup.lua
xmake build
```

### 2. Run a Demo

#### C++ Application
```bash
# Run with default scene
./build/bin/app-simulation

# Specify backend and scene
./build/bin/app-simulation cuda Resources/Scenes/cloth_rigid_coupling_high_res.json
```

#### Python Application
```bash
# With GUI (requires polyscope)
python PythonBindings/example_usage.py --backend cuda

# Headless mode (batch processing)
python PythonBindings/example_usage.py --backend cuda --headless --advance_frames 60
```

> **Note:** Most used backends in LuisaCompute : `cuda`, `dx` (DirectX), `vk` (Vulkan), `metal` (macOS)

More compiling details can be found at [BUILD.md](/Document/Build.md)

## 🖥️ Supported Backends

| Backend   | Windows   | Linux        | MacOS      | Description                                                                                                                                                       |
| --------- | --------- | ------------ | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CUDA      | Supported | Supported    |            | Requires [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) (CUDA > 12.0)                                                                          |
| Vulkan    | Supported | Experimental | Developing | Requires [vulkan SDK](https://vulkan.lunarg.com/). Linux (currently for x86_64 only) and Macos is in development                                                  |
| DirectX12 | Supported |              |            |                                                                                                                                                                   |
| Metal     |           |              | Supported  |                                                                                                                                                                   |
| Fallback  | Supported | Supported    | Supported  | CPU fallback via TBB/Embree. Requires [llvm](https://llvm.org/), [TBB](https://github.com/uxlfoundation/oneTBB) and [Embree](https://github.com/RenderKit/embree) |

---

## 📊 Gallery

### Rotation Cylinder Demo
| 88K vertices, 174K triangles, 3M+ collision pairs           |
| ----------------------------------------------------------- |
| ![Rotation](Document/Images/RotationCylinder60s.gif)        |
| **~3 FPS on RTX 3090 (CUDA)**, **~2 FPS on M2 Max (Metal)** |

### More Examples

| Scene                                                                       | Preview                       | Description                   |
| --------------------------------------------------------------------------- | ----------------------------- | ----------------------------- |
| [Cloth-Rigid Coupling](Resources/Scenes/cloth_rigid_coupling_high_res.json) | ![](Document/Images/4.png)    | High-res cloth on rigid cube  |
| [Rotation Cylinder 7K](Resources/Scenes/cloth_rotation_cylinder_7K.json)    | ![](Document/Images/5.png)    | Cloth wrapping cylinder       |
| [Pinned Cloth](Resources/Scenes/cloth_pinned.json)                          | ![](Document/Images/1.png)    | Different material properties |
| [Moving Boundary](Resources/Scenes/cloth_moving_boundary.json)              | ![](Document/Images/0_ed.png) | Dynamic Dirichlet boundary    |
| [Rigid Bucket](Resources/Scenes/rigid_bucket.json)                          | ![](Document/Images/11.png)   | Multi-body collision          |
| [Folding Cubes](Resources/Scenes/rigid_multi_folding_cubes.json)            | ![](Document/Images/13.png)   | Self-collision folding        |
| [Friction Test](Resources/Scenes/rigid_frictional_test.json)                | ![](Document/Images/18.png)   | Frictional contact            |

---

## 🛤️ Roadmap

### Completed ✅
- [x] Python Bindings
- [x] Cloth Simulation
- [x] Rigid Body Simulation
- [x] Soft Body (Spring + ARAP FEM)
- [x] Soft-Cloth-Rigid Coupling
- [x] Penetration-Free Contact (IPC)
- [x] Affine Body Dynamics
- [x] Frictional Modeling
- [x] C++ Integration API

### In Progress 🔄
- [ ] Tetrahedral Mesh Support
- [ ] Joint Constraints

### Planned 📋
- [ ] UV Mapping Package
- [ ] Strain Limiting
- [ ] Consistent Solve
- [ ] Thin Shell Rigid-Body
- [ ] Matrix Assembly Optimization
- [ ] Better Preconditioners

---

## 📚 References

- **Constitutions:** [libuipc](https://github.com/spiriMirror/libuipc), [GAMES 103](https://www.bilibili.com/video/BV12Q4y1S73g), [PNCG-IPC](https://github.com/Xingbaji/PNCG_IPC), [HOBAK](https://github.com/theodorekim/HOBAKv1), [solid-sim-tutorial](https://github.com/phys-sim-book/solid-sim-tutorial), [Codim-IPC](https://github.com/ipc-sim/Codim-IPC), [ZOZO's Contact Solver](https://github.com/st-tech/ppf-contact-solver)
- **DCD & CCD:** ZOZO's Contact Solver, libuipc.
- **PCG (Linear Equation Solver):** [MAS](https://wanghmin.github.io/publication/wu-2022-gbm/), [AMGCL](https://github.com/ddemidov/amgcl), libuipc.
- **Framework:** [libshell](https://github.com/legionus/libshell), [LuisaComputeGaussSplatting](https://github.com/LuisaGroup/LuisaComputeGaussianSplatting).
- **GPU Intrinsic:** LuisaComputeGaussSplatting.
- **Collision Energy:** MAS, [PNCG-IPC](https://github.com/Xingbaji/PNCG_IPC)
- **Affine Body Dynamics:** [abd-warp](https://github.com/Luke-Skycrawler/abd-warp), libuipc ([documentation](https://spirimirror.github.io/libuipc-doc/specification/constitutions/affine_body/), [theory derivation](https://github.com/spiriMirror/libuipc/blob/main/scripts/symbol_calculation/affine_body_quantity.ipynb)).

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Thanks to the [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute) and [libuipc](https://github.com/spiriMirror/libuipc) communities for their open-source contributions to physically-based simulation.

---

<p align="center">
  <b>LuisaComputeSimulator</b> — High-Performance Physics Simulation 📈
</p>