# Build Guide

This guide covers building LuisaComputeSimulator on different platforms and configurations.

## Requirements

| Requirement           | Minimum Version | Notes               |
| --------------------- | --------------- | ------------------- |
| CMake                 | 3.26+           |                     |
| C++ Compiler          | C++20 capable   | Clang 15+, GCC 13+  |
| CUDA (optional)       | 12.0+           | For CUDA backend    |
| Vulkan SDK (optional) | 1.3+            | For Vulkan backend  |
| Python 3.8+           |                 | For Python bindings |

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
- Xcode 15+ (for Metal backend)
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
xmake lua setup.lua
xmake build
```

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
| `-D LCS_ENABLE_GUI`              | Enable GUI (Polyscope)        | `ON`           |

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

#### With Python Bindings
```bash
cmake -S . -B build \
    -DLCS_BUILD_PYBINDINGS=ON \
    -DLCS_PYTHON_EXECUTABLE=/usr/bin/python3

cmake --build build -j
```

#### With Vulkan Backend (Linux)
```bash
cmake -S . -B build \
    -DLUISA_COMPUTE_ENABLE_VULKAN=ON

cmake --build build -j
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
./build/bin/app-simulation cuda

# Metal (macOS)
./build/bin/app-simulation metal

# DirectX (Windows)
./build/bin/app-simulation dx

# Vulkan
./build/bin/app-simulation vulkan

# CPU (Fallback)
./build/bin/app-simulation fallback
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
./build/bin/app-simulation cuda
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
export PYTHONPATH=$PYTHONPATH:/path/to/LuisaComputeSimulator/build/bin
python -c "import lcs_py; print('OK')"
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

