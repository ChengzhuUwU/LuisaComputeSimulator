# Build Guide

This guide covers building LuisaComputeSimulator on different platforms and configurations.

## Requirements

| Requirement           | Minimum Version | Notes                              |
| --------------------- | --------------- | ---------------------------------- |
| CMake                 | 3.26+           |                                    |
| Xmake                 | 3.0.6+          | Required by LuisaCompute submodule |
| C++ Compiler          | C++20 capable   | Clang 15+, GCC 13+                 |
| CUDA (optional)       | 12.0+           | For CUDA backend                   |
| Vulkan SDK (optional) | 1.3+            | For Vulkan backend                 |
| Python 3.8+           |                 | For Python bindings                |

### Platform-Specific Requirements

#### Linux
```bash
sudo apt-get update
sudo apt-get install -y \
    cmake \
    ninja-build \
    libvulkan-dev \
    libeigen3-dev \
    libx11-dev \
    uuid-dev \
    clang-15
```

#### macOS
- Homebrew dependencies:
```bash
brew install cmake eigen
```

#### Windows
- Visual Studio 2022+
- Windows SDK 10.0.19041.0+

---

## Quick Build

### CMake (Recommended)

```bash
# Clone and enter directory
git clone https://github.com/ChengzhuUwU/LuisaComputeSimulator.git
cd LuisaComputeSimulator

# Configure
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j
```

### Xmake

```bash
# Clone dependencies first
xmake lua setup.lua

# Configure for your platform
# On macOS with Metal:
xmake f -m release --lc_cuda_backend=n --lc_metal_backend=y --lc_fallback_backend=y

# On Linux with CUDA:
xmake f -m release --lc_cuda_backend=y --lc_vk_backend=n

# Build
xmake build
```

> **Note:** In this project, xmake backend switches use `lc_*_backend` names. On macOS, CUDA is forced off (`lc_cuda_backend=false`) and Metal is only enabled on macOS (`lc_metal_backend=true` only on macOS), regardless of manual override.

---

## Build Options

### CMake Flags

| Option                           | Description                   | Default        |
| -------------------------------- | ----------------------------- | -------------- |
| `-DCMAKE_BUILD_TYPE`             | Build type: `Release`/`Debug` | `Release`      |
| `-DCMAKE_C_COMPILER`             | C compiler                    | auto-detect    |
| `-DCMAKE_CXX_COMPILER`           | C++ compiler                  | auto-detect    |
| `-G Ninja`                       | Use Ninja generator           |                |
| `-D LUISA_COMPUTE_ENABLE_CUDA`   | Enable CUDA backend           | `ON`           |
| `-D LUISA_COMPUTE_ENABLE_VULKAN` | Enable Vulkan backend         | `OFF`          |
| `-D LUISA_COMPUTE_ENABLE_METAL`  | Enable Metal backend          | `ON` (macOS)   |
| `-D LUISA_COMPUTE_ENABLE_DX`     | Enable DirectX backend        | `ON` (Windows) |
| `-D LCS_BUILD_PYBINDINGS`        | Build Python bindings         | `OFF`          |
| `-D LCS_PYTHON_EXECUTABLE`       | Python interpreter path       |                |
| `-D LCS_ENABLE_GUI`              | Enable GUI (Polyscope)        | `OFF`          |
| `-D LCS_ENABLE_TEST`             | Enable Unit Test              | `OFF`          |

### Xmake Flags

| Option                                | Description                 | Default     |
| ------------------------------------- | --------------------------- | ----------- |
| `--mode=release/debug`                | Build type                  | `release`   |
| `--lc_cuda_backend=y/n`               | Enable CUDA backend         | `y` (non-macOS effective) |
| `--lc_vk_backend=y/n`                 | Enable Vulkan backend       | `n`         |
| `--lc_metal_backend=y/n`              | Enable Metal backend        | `y` (macOS effective) |
| `--lc_dx_backend=y/n`                 | Enable DirectX backend      | `n`         |
| `--lc_fallback_backend=y/n`           | Enable CPU fallback backend | `n`         |
| `--lcs_enable_gui=y/n`                | Enable GUI (Polyscope)      | `n`         |
| `--lcs_enable_test=y/n`               | Build unit tests            | `n`         |
| `--lcs_build_pybindings=y/n`          | Build Python bindings       | `n`         |
| `--lcs_python_executable=<path>`      | Python interpreter path     | empty       |

### Build Examples

#### Linux with CUDA
```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang-15 \
    -DCMAKE_CXX_COMPILER=clang++-15

cmake --build build -j$(nproc)
```

#### macOS with Metal
```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++

cmake --build build -j$(sysctl -n hw.ncpu)
```

#### With Python Bindings (CMake)

One-time setup:
```bash
# Create project venv
python3 -m venv .venv

# Install build/dev tooling
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install scikit-build-core pybind11 ninja numpy pybind11-stubgen trimesh
```

Configure, build, and install:
```bash
# Configure — MUST include -DLCS_BUILD_PYBINDINGS=ON
cmake -S . -B build \
    -DLCS_BUILD_PYBINDINGS=ON \
    -DLCS_PYTHON_EXECUTABLE="$(pwd)/.venv/bin/python"

# Build + regenerate stubs
cmake --build build -j --target stubs

# Editable install of the lcs package into the venv
.venv/bin/python -m pip install -e .
```

After C++ binding changes (`PythonBindings/src/python_bindings.cpp`), rerun:
```bash
cmake --build build -j --target stubs
```

Run tests:
```bash
.venv/bin/python PythonBindings/tests/test_rigid_joint_animation.py --headless --advance_frames 30
```

#### With Vulkan Backend (Linux)
```bash
cmake -S . -B build \
    -DLUISA_COMPUTE_ENABLE_VULKAN=ON

cmake --build build -j
```

#### Xmake with Python Bindings (Windows example)
```bash
xmake lua setup.lua
xmake f -c -m release \
    --lcs_build_pybindings=y \
    --lcs_python_executable="C:/Applications/Python/python.exe" \
    --lcs_enable_gui=n \
    --lcs_enable_test=n \
    --lc_cuda_backend=y \
    --lc_dx_backend=n \
    --lc_vk_backend=n \
    --lc_fallback_backend=n

xmake build lcs_py
```

#### Xmake with Python Bindings (macOS example)
```bash
xmake f -c -m release \
    --lcs_build_pybindings=y \
    --lcs_python_executable=/opt/homebrew/bin/python3 \
    --lcpp_test=n

xmake build lcs_py

# Verify import
PYTHONPATH=build/bin python3 -c "import lcs_py; print('OK')"

# Run tests
PYTHONPATH=build/bin python3 PythonBindings/tests/test_rigid_joint_animation.py --headless --advance_frames 30
```

#### Xmake build only Python module
```bash
xmake build lcs_py
```

#### Xmake generate stubs
```bash
xmake build stubs
```

---

## Platform-Specific Notes

### Linux

#### Installing CMake 3.26+

If your package manager provides an older version:

```bash
# Download latest CMake
wget https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-linux-x86_64.tar.gz
tar -xf cmake-3.28.1-linux-x86_64.tar.gz
sudo ln -sf $(pwd)/cmake-3.28.1-linux-x86_64/bin/cmake /usr/local/bin/cmake
sudo ln -sf $(pwd)/cmake-3.28.1-linux-x86_64/bin/ctest /usr/local/bin/ctest
```

#### NVIDIA Driver & CUDA

Ensure you have compatible drivers:

```bash
# Check driver version
nvidia-smi

# Check CUDA version
nvcc --version
```

### macOS

#### Metal Backend

Metal is supported by default on Apple Silicon and Intel Macs with Metal-capable GPUs.

#### Build on Apple Silicon (M1/M2/M3)

```bash
cmake -S . -B build \
    -DCMAKE_OSX_ARCHITECTURE=arm64 \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build -j$(sysctl -n hw.ncpu)
```

### Windows

#### Visual Studio

Open "Developer Command Prompt for VS 2022" and run:

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

#### DirectX 12

DirectX 12 is enabled by default on Windows. Ensure you have:

- Windows 10 version 1809+ or Windows 11
- DirectX 12 compatible GPU
- Windows SDK installed via Visual Studio

---

## Backend Selection

### At Runtime

#### C++ Application

```bash
# CUDA (Linux)
./build/bin/app_simulation cuda

# Metal (macOS)
./build/bin/app_simulation metal

# DirectX (Windows)
./build/bin/app_simulation dx

# Vulkan
./build/bin/app_simulation vk

# CPU (Fallback)
./build/bin/app_simulation fallback
```

#### Python Application

```python
solver.init_device(backend_name="cuda")  # or "metal", "dx", "vk", "fallback"
```

---

## Troubleshooting

### Shader Compilation Errors

Set environment variable to dump shaders for debugging:

```bash
export LUISA_DUMP_SOURCE=1
./build/bin/app_simulation cuda
```

Shader files will be saved to `build/bin/.cache/`.

### Out of Memory

Reduce simulation size or enable memory optimizations:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

### CUDA Out of Date

Ensure CUDA toolkit matches your GPU driver:

```bash
# Check driver
nvidia-smi

# Check CUDA version
nvcc --version
```

### Metal Not Available on macOS

Ensure you're running on a Metal-capable device and macOS 12+:

```bash
# Check Metal device
system_profiler SPDisplaysDataType | grep Metal
```

### Python Bindings Import Error

Ensure Python path is set correctly:

```bash
# For CMake builds (or after .venv/bin/python -m pip install -e .)
export PYTHONPATH=$PYTHONPATH:/path/to/LuisaComputeSimulator/build/bin
python -c "import lcs_py; print('OK')"

# For Xmake builds
PYTHONPATH=build/bin python -c "import lcs_py; print('OK')"
```

---

## Building Documentation

### API Documentation

API docs are included in the `Document/` folder:

- `PythonAPI.md` - Python API Reference
- `CppAPI.md` - C++ API Reference  
- `Energies.md` - Energy Model Documentation
- `Build.md` - This file

---

## Clean Build

If you encounter build issues:

### CMake
```bash
# Remove build directory
rm -rf build

# Clean and rebuild
cmake -S . -B build
cmake --build build -j
```

For a completely fresh start:

```bash
rm -rf build .cache
git clean -fd
cmake -S . -B build
cmake --build build -j
```

### Xmake
```bash
# Clean build artifacts
xmake clean

# Full clean (removes config cache too)
xmake clean --all
xmake f -c -m release
xmake build
```

