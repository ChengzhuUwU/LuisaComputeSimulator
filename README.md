# LuisaComputeSolver: Physics Simulator Based on LuisaCompute

![Teaser](Document/README1.png)

[![linux](https://github.com/ChengzhuUwU/LuisaComputeSimulator/actions/workflows/cmake_linux.yml/badge.svg?branch=main)](https://github.com/ChengzhuUwU/LuisaComputeSimulator/actions/workflows/cmake_linux.yml)
[![windows](https://github.com/ChengzhuUwU/LuisaComputeSimulator/actions/workflows/cmake_windows.yml/badge.svg?branch=main)](https://github.com/ChengzhuUwU/LuisaComputeSimulator/actions/workflows/cmake_windows.yml)
[![macos](https://github.com/ChengzhuUwU/LuisaComputeSimulator/actions/workflows/cmake_macos.yml/badge.svg?branch=main)](https://github.com/ChengzhuUwU/LuisaComputeSimulator/actions/workflows/cmake_macos.yml)


LuisaComputeSimulator is a high-performance cross-platform **Physics Simulator** based on [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute), providing support for **Cloth and Rigid-Body** simulations and support for **Penetration-Free** contact handling, accelated by variant GPU/CPU backends(e.g., CUDA, DirectX12, Vulkan, Metal, Fallback).

> Teasor figure: 88K vertices & 174K triangles, over 3,000,000 collision pairs, about 3 fps on RTX3090 (CUDA backend) and 2 fps on M2 Max (Metal Backend).

## Usage

### Python Frontend

Sample Python-frontend code can be found at [test_cloth_rigid_coupling.py](PythonBindings/tests/test_cloth_rigid_coupling.py):

```python
    from sim_utils import parse_args
    import lcs_py as lcs
    args = parse_args()

    solver = lcs.NewtonSolver()
    solver.init_device(backend_name=args.backend, binary_path=None)

    # Build 2 world_data objects first: a rigid cube and a soft cloth
    cube_mesh_path = os.path.join(root, 'Resources', 'InputMesh', 'cube.obj')
    cube_mesh = trimesh.load(cube_mesh_path, process=False)
    cube_wd = solver.create_world_data_from_array('cube', cube_mesh.vertices, cube_mesh.faces)
    cube_wd.set_simulation_type(lcs.MaterialType.Rigid)
    cube_wd.set_translation(0.0, 0.34, 0.0)
    cube_wd.set_rotation(0.5235988, 0.0, 0.5235988)
    cube_wd.set_scale(0.1)
    cube_id = solver.register_world_data(cube_wd)

    cloth_mesh_path = os.path.join(root, 'Resources', 'InputMesh', 'square2K.obj')
    cloth_wd = solver.create_world_data_from_file_path('cloth', cloth_mesh_path)
    cloth_wd.set_simulation_type(lcs.MaterialType.Cloth)
    cloth_wd.set_physics_material_cloth(thickness=0.001, youngs_modulus=1e6)
    cloth_wd.set_scale(0.75)
    cloth_wd.add_fixed_point_by_method("LeftBack")
    cloth_id = solver.register_world_data(cloth_wd)

    # register_world_data(...) returns object id.
    # After registration, query objects via const access APIs, e.g.:
    cube_const = solver.get_object_by_registration_id(cube_id)

    # Initialize the solver
    solver.init_solver()

    config_ref = solver.get_config()
    config_ref.use_floor = False
    config_ref.implicit_dt = 1/60

    output_dir = os.path.join(root, "Resources", "OutputMesh")

    # Launch simulation
    if args.headless:
        solver.save_sim_result(obj_path=os.path.join(output_dir, "init.obj"))
        for frame in range(0, args.advance_frames):
            solver.physics_step_gpu() # or solver.physics_step_cpu()
        solver.save_sim_result(obj_path=os.path.join(output_dir, "result.obj"))
    else:
        from polyscope_gui import SimulationGUI
        gui = SimulationGUI(solver, config_ref, output_dir)
        gui.show()
```

### Cpp Frontend

Sample Cpp-frontend code can be found at [app_integration.cpp](Application/app_integration.cpp).

```C++
    #include "SimulationSolver/newton_solver.h"

    int main(int argc, char** argv)
    {
        lcs::NewtonSolver solver;
        solver.create_device(/*binary_path =*/argv[0], /*backend =*/ "cuda");

        // Build world_data using file path, then register
        auto upper_square = lcs::Initializer::WorldData()
                                .set_name("upper square")
                                .load_mesh_from_path(std::string(LCSV_RESOURCE_PATH) + "/InputMesh/square2.obj")
                                .set_material_type(lcs::Material::MaterialType::Cloth)
                                .set_physics_material(lcs::Material::ClothMaterial{
                                    .stretch_model = lcs::Initializer::ConstitutiveStretchModelCloth::Spring,
                                })
                                .set_translation({ 0.0f, 0.4f, 0.0f });
        uint upper_square_id = solver.register_world_data(upper_square);

        // Build world_data using array, then register
        std::vector<std::array<float, 3>> square_mesh_vertices{ { -0.5, 0, -0.5 }, { 0.5, 0, -0.5 }, { -0.5, 0, 0.5 }, { 0.5, 0, 0.5 } };
        std::vector<std::array<uint, 3>>  square_mesh_faces{ { 0, 3, 1 }, { 0, 2, 3 } };
        auto lower_square = lcs::Initializer::WorldData()
                                .set_name("lower square")
                                .load_mesh_from_array(square_mesh_vertices, square_mesh_faces)
                                .set_physics_material(lcs::Material::ClothMaterial{}) 
                                .set_scale(0.8f)
                                .set_translation({ 0.1f, 0.2f, 0.0f })
                                .add_fixed_point_info({ .method = lcs::Initializer::FixedPointsType::Left })
                                .add_fixed_point_info({ .method = lcs::Initializer::FixedPointsType::Right });
        uint lower_square_id = solver.register_world_data(lower_square);

        // Scene configs
        auto config = solver.get_config();
        config.use_floor = false;
        config.implicit_dt = 0.2;
        config.use_energy_linesearch = true;

        solver.init_solver();

        // Init rendering data
        std::vector<std::vector<std::array<float, 3>>> sa_rendering_vertices;
        solver.get_curr_vertices_to_host(sa_rendering_vertices);

        // Main application
        for (uint ii = 0; ii < 20; ii++)
        {
            solver.physics_step_GPU();
            solver.get_curr_vertices_to_host(sa_rendering_vertices);
            // Display or other processing
        }

        return 0;
    }
```

## Getting Started

- **Clone the repository:**
    ```Bash
    git clone https://github.com/ChengzhuUwU/LuisaComputeSimulator.git
    cd LuisaComputeSimulator
    ```

- **Install required packages:**  
    - Cmake > 3.26

    - For Linux users: 
		- You need to install required packages:
      <!-- sudo apt-get update && sudo apt-get upgrade -y
      sudo apt-get install -y software-properties-common
      sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
      sudo apt-get update -->
      ```bash
      sudo apt-get install -y wget uuid-dev ninja-build libvulkan-dev libeigen3-dev libx11-dev cmake
      ```
      Clang is recommended as the compiler:`sudo apt-get install -y clang-15 `
      
    - For Linux and Windows users:
      - If you want to use CUDA backend, you need to install NVIDIA CUDA Toolkit (required: `CUDA >= 12.0`). Check the maximum supported CUDA version using `nvidia-smi`.

	<!-- - For MacOS users:
      - [Xcode](https://developer.apple.com/cn/xcode/) is required for the support of Metal Backend. -->


- **You can build with Cmake:**  
  - Configure: ```cmake -S . -B build```
  - Build   : ```cmake --build build -j```
  - Configure Params: You can specify your favorite generators, compilers, or build types by adding parameters after `cmake -S . -B build`:
	- Specify **generators**: `-G Ninja`
	- Specify **compilers**: `-D CMAKE_C_COMPILER=clang-15 -D CMAKE_CXX_COMPILER=clang++-15`
	- Specify **compilers by path**: `-D CMAKE_C_COMPILER=/usr/bin/gcc-13, -D CMAKE_CXX_COMPILER=/usr/bin/g++-13`
	- Specify **build type**: `-D CMAKE_BUILD_TYPE=Release`
    - Enable/disable **computing backends**: `-D LUISA_COMPUTE_ENABLE_VULKAN=ON`
	- Enable/disable **python binding**: Set `-D LCS_BUILD_PYBINDINGS=ON`, and specify the python executable path `-D LCS_PYTHON_EXECUTABLE=/usr/bin/python3`
	- Enable/disable **GUI**: `-D LCS_ENABLE_GUI=ON` (based on [polyscope](https://github.com/nmwsharp/polyscope))

- **You can also build with Xmake:**  
  - Configure: ```xmake lua setup.lua```
  	- If you are root-user, you may need `xmake lua --root setup.lua`
  - Build   : ```xmake build```

- **Launch the sample Cpp application:**  
    - Launch on Linux/macOS: `build/bin/app-simulation <backend-name> <scene-json-file>`
    - Launch on Windows: `build/bin/app-simulation.exe  <backend-name> <scene-json-file>` 

    - Note that: The launching parameters `<backend-name> <scene-json-file>` is optional, you can specify your favorite backend in `<backend-name>` (e.g., `metal/cuda/dx/vulkan`) and choose a simulation scenario in `<scene-json-file>` (e.g., `cloth_rigid_coupling_high_res.json`, we provide several example scenarios in [Resources/Scenes](Resources/Scenes) directory).

- **Launch the sample Python application**
    - Launch with GUI: `path_to_python PythonBindings/example_usage.py --backend <backend-name>` 
    - Launch without GUI: `path_to_python PythonBindings/example_usage.py --backend <backend-name> --headless --advance_frames <N>`
    - Note that: `path_to_python` should corresponds with `LCS_PYTHON_EXECUTABLE` in configuration progress.
    - `<backend-name>` can be `cuda/dx/vk/metal`; `--headless` and `--advance_frames` are optional (`advance_frames` default is `30`).
    - `numpy` and `trimesh` are required; `polyscope` is required when not using `--headless`.

More building guidance about computing backend can be found in [the document of LuisaCompute](https://github.com/LuisaGroup/LuisaCompute/blob/stable/BUILD.md) and [Build.md](Document/Build.md).


### Other Configuration

1. The default backend is `dx` (DirectX12) on Windows, `cuda` on Linux, and `metal` on macOS. 
   To enable other backends such as `vulkan`, or `fallback (TBB)`, update the building options in CMake/Xmake (e.g., set `LUISA_COMPUTE_ENABLE_VULKAN` to `ON`), and specify the target backend by passing `<backend-name>` in the launching parameters.

2. Check the generated shader using `echo 'export LUISA_DUMP_SOURCE=0' >> ~/.zshrc` (Shader files will be saved in `build/bin/.cache/`)

## Supported Backends (of LuisaCompute)

|   Backend |  Windows   | Linux     |  MacOS  | Description |
|  -----    |  ------    |  ------   |  ------ |      ------ |
| CUDA      | Supported  | Supported |         | Requires [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) (CUDA > 12.0) | 
| Vulkan    | Supported  | Experimental | Developing  | Requires [vulkan SDK](https://vulkan.lunarg.com/). Linux (currently for x86_64 only) and Macos is in development | 
| DirectX12 | Supported  |           |           |   | 
| Metal     |            |           | Supported |   | 
| Fallback  | Supported  | Supported | Supported | Launch kernels on the CPU. Requires [llvm](https://llvm.org/), [TBB](https://github.com/uxlfoundation/oneTBB) and [Embree](https://github.com/RenderKit/embree) |


## Examples

|   [Rotation Cylinder](Resources/Scenes/cloth_rotation_cylinder_88K.json)  |
|  -----   |
| ![Case6](Document/Images/RotationCylinder60s.gif)  |
| About 3 fps on RTX3090 (CUDA backend), about 2 fps on M2 Max (Metal Backend) |

|       |   |
|  -----   |------|
|          |      |
| Moving Dirichlet Case |  |
| ![Case0Bg](Document/Images/0_bg.png) [File](Resources/Scenes/cloth_moving_boundary.json) |  ![Teaser](Document/Images/0_ed.png) (The velocity of red plane is 3m/s )  |    
|  Different Material Properties | Cloth-Rigid Coupling  Case 1 |
| ![Case1](Document/Images/1.png) [File](Resources/Scenes/cloth_pinned.json)  |  ![Case3](Document/Images/3.png) [File](Resources/Scenes/cloth_rigid_coupling_drop.json) |
|   Cloth-Rigid Coupling Case 2 | Rotation Cylinder (21K DOF) |
| ![Case4](Document/Images/4.png) [File](Resources/Scenes/cloth_rigid_coupling_high_res.json)  |  ![Case5](Document/Images/5.png) [File](Resources/Scenes/cloth_rotation_cylinder_7K.json) |
|   Rotation Cylinder (260K DOF) | Large Thickness Case |
| ![Case6](Document/Images/6.png) [File](Resources/Scenes/cloth_rotation_cylinder_88K.json)   |  ![Case9](Document/Images/9.png) [File](Resources/Scenes/cloth_unit_test_square2.json) |
|   Multi-Rigid-Body Case 1 | Multi-Rigid-Body Case 2 |
| ![Case11](Document/Images/11.png)  [File](Resources/Scenes/rigid_bucket.json)  |  ![Case13](Document/Images/13.png) [File](Resources/Scenes/rigid_multi_folding_cubes.json) |
|  Friciontal Test |  |
| ![Case11](Document/Images/18.png)  [File](Resources/Scenes/rigid_frictional_test.json)  |  |

## TODOLIST

- [ ] Joint Constraint
- [x] Python Binding
- [ ] UV Package
- [ ] Deformable Body Energy (And atomatic tet mesh generation)
- [ ] Elastic Rod Energy
- [ ] Strain Limiting
- [ ] Consistent Solve
- [x] Replace All Constraint With Binding Group for AOT
- [ ] Thin Shell Rigid-Body Simulation
- [ ] Upper/Lower-Triangle of System Matrix Optimization
- [ ] GPU-based Global Triplet Sorting (For Matrix Assembly)
- [ ] Mesh Split and Simplification
- [x] Accurate Frictional Modeling
- [ ] Better Numerical Preconditioners


## References

- **Constitutions:** [libuipc](https://github.com/spiriMirror/libuipc), [GAMES 103](https://www.bilibili.com/video/BV12Q4y1S73g), [PNCG-IPC](https://github.com/Xingbaji/PNCG_IPC), [HOBAK](https://github.com/theodorekim/HOBAKv1), [solid-sim-tutorial](https://github.com/phys-sim-book/solid-sim-tutorial), [Codim-IPC](https://github.com/ipc-sim/Codim-IPC), [ZOZO's Contact Solver](https://github.com/st-tech/ppf-contact-solver)
- **DCD & CCD:** ZOZO's Contact Solver, libuipc.
- **PCG (Linear Equation Solver):** [MAS](https://wanghmin.github.io/publication/wu-2022-gbm/), [AMGCL](https://github.com/ddemidov/amgcl), libuipc.
- **Framework:** [libshell](https://github.com/legionus/libshell), [LuisaComputeGaussSplatting](https://github.com/LuisaGroup/LuisaComputeGaussianSplatting).
- **GPU Intrinsic:** LuisaComputeGaussSplatting.
- **Collision Energy:** MAS, [PNCG-IPC](https://github.com/Xingbaji/PNCG_IPC)
- **Affine Body Dynamics:** [abd-warp](https://github.com/Luke-Skycrawler/abd-warp), libuipc ([documentation](https://spirimirror.github.io/libuipc-doc/specification/constitutions/affine_body/), [theory derivation](https://github.com/spiriMirror/libuipc/blob/main/scripts/symbol_calculation/affine_body_quantity.ipynb)).

## Others

Thanks to LuisaCompute and Libuipc community, their open-source spirit has propelled the advancement of the reality.