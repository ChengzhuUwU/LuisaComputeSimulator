# LuisaComputeSolver: Physics Simulation Based on LuisaCompute

![Teaser](Document/README1.png)

<!-- [![actions status linux](https://github.com/ChengzhuUwU/LuisaComputeSimulator/workflows/linux/cmake.svg)](https://github.com/nmwsharp/polyscope/actions)
[![actions status macOS](https://github.com/ChengzhuUwU/LuisaComputeSimulator/workflows/macOS/cmake.svg)](https://github.com/nmwsharp/polyscope/actions)
[![actions status windows](https://github.com/ChengzhuUwU/LuisaComputeSimulator/workflows/windows/cmake.svg)](https://github.com/nmwsharp/polyscope/actions) -->

## Getting Started

### Cmake

1. **Clone the repository:**  
    `git clone https://github.com/ChengzhuUwU/LuisaComputeSimulator.git`

2. **Install required packages:**  
    - For Linux users:  
      `sudo apt-get -y install build-essential uuid-dev`
    - For Linux and Windows users: If you want to use cuda backend, you need to install cuda-toolkit. Otherwise you need to enable `LUISA_COMPUTE_ENABLE_DX` option or other backend in CMake.

3. **Configure the project:**  
    `cmake -S . -B build`  
    - Optionally, specify the compiler:  
      `-D CMAKE_C_COMPILER=clang -D CMAKE_CXX_COMPILER=clang++`  
      or  
      `-D CMAKE_C_COMPILER=/usr/bin/clang`  
    - Specify the generator:  
      `-G Ninja`

4. **Build the project:**  
    `cmake --build build -j`

5. **Run the application:**  
    `build/bin/app-simulation` (Linux/macOS)  
    `build/bin/app-simulation.exe` (Windows)

6. **Select a scene:**  
    Edit `Application/app_simulation_demo_config.cpp` in the `load_scene()` function and modify the `case_number` value.

### Xmake



## Additional Configuration

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