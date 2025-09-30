# LuisaComputeSolver: Physics Simulation Based on LuisaCompute

![Teaser](Document/README1.png)

[![linux](https://github.com/ChengzhuUwU/LuisaComputeSimulator/actions/workflows/cmake_linux.yml/badge.svg?branch=main)](https://github.com/ChengzhuUwU/LuisaComputeSimulator/actions/workflows/cmake_linux.yml)
[![windows](https://github.com/ChengzhuUwU/LuisaComputeSimulator/actions/workflows/cmake_windows.yml/badge.svg?branch=main)](https://github.com/ChengzhuUwU/LuisaComputeSimulator/actions/workflows/cmake_windows.yml)
[![macos](https://github.com/ChengzhuUwU/LuisaComputeSimulator/actions/workflows/cmake_macos.yml/badge.svg?branch=main)](https://github.com/ChengzhuUwU/LuisaComputeSimulator/actions/workflows/cmake_macos.yml)

## Getting Started

- **Clone the repository:**
    ```git clone https://github.com/ChengzhuUwU/LuisaComputeSimulator.git```

- **Install required packages:**  
    - For Linux users:  
      ```sudo apt-get -y install build-essential uuid-dev```
    - For Linux and Windows users: If you want to use cuda backend, you need to install cuda-toolkit. Otherwise you need to enable `LUISA_COMPUTE_ENABLE_DX` option or other backend in Cmake or Xmake.

- **You can build with Cmake:**  
  - Congiure: ```cmake -S . -B build```
  - Build   : ```cmake --build build -j```
  - Optionally, you can specify your favorite generators, compilers, or build types like `-G Ninja`, `-D CMAKE_C_COMPILER=clang -D CMAKE_CXX_COMPILER=clang++` (or `-D CMAKE_C_COMPILER=/usr/bin/clang-15`) and `-D CMAKE_BUILD_TYPE=Release`.

- **You can also build with Xmake:**  
  - Congiure: ```xmake l setup.lua```
  - Build   : ```xmake build```

- **Run the application:**  
    `build/bin/app-simulation` (Linux/macOS)  
    `build/bin/app-simulation.exe` (Windows)

- You can select other scenarios:
    Edit `Application/app_simulation_demo_config.cpp` in the `load_scene()` function and modify the `case_number` value.

### Other Configuration

1. The default backend is `cuda` on Windows and Linux, and `metal` on macOS.  
    To use other backends such as `dx`, `vulkan`, or `fallback (TBB)`, update the compile options in the main CMake file and set the desired backend in `Application/app_simulation.cpp` (`main() > backend`).  
    *(Dynamic backend selection may be supported in the future.)*

2. The GUI is disabled by default for broader platform compatibility.  
    To enable the GUI (using [polyscope](https://github.com/nmwsharp/polyscope)), set the `LUISA_COMPUTE_SOLVER_USE_GUI` option in CMake.

## References

- **Contact energy:**  
  [PNCG-IPC](https://github.com/Xingbaji/PNCG_IPC), [libuipc](https://github.com/spiriMirror/libuipc), [HOBAK](https://github.com/theodorekim/HOBAKv1), [solid-sim-tutorial](https://github.com/phys-sim-book/solid-sim-tutorial), [C-IPC](https://github.com/ipc-sim/Codim-IPC)
- **DCD & CCD:**  
  [ZOZO's Contact Solver](https://github.com/st-tech/ppf-contact-solver)
- **PCG (Linear equation solver):**  
  [MAS](https://wanghmin.github.io/publication/wu-2022-gbm/), [AMGCL](https://github.com/ddemidov/amgcl)
- **Framework:**  
  [libshell](https://github.com/legionus/libshell), [LuisaComputeGaussSplatting](https://github.com/LuisaGroup/LuisaComputeGaussianSplatting)
- **LBVH:**  
  libuipc
- **Dirichlet boundary energy:**  
  solid-sim-tutorial
- **GPU Intrinsic:**  
  LuisaComputeGaussSplatting
- **Affine body dynamics:**  
  [abd-warp](https://github.com/Luke-Skycrawler/abd-warp), libuipc ([documentation](https://spirimirror.github.io/libuipc-doc/specification/constitutions/affine_body/), [theory derivation](https://github.com/spiriMirror/libuipc/blob/main/scripts/symbol_calculation/affine_body_quantity.ipynb))

## Other

Thanks to...