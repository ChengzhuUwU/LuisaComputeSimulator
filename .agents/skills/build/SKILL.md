## Basic Building Operation

If you are coding agent, before configuration, you need to read the configuration params in [lcs_config.ini](/lcs_config.ini) (If the file not exists, you need to create one from copying [lcs_config_template.ini](/lcs_config_template.ini)). If value of the key is not empty, you need to apply the value in configuration:
- You can ignore the param if value is empty: `CMAKE_BUILD_TYPE = `
- You need to apply the param if value is not empty: `LCS_PYTHON_EXECUTABLE = /opt/homebrew/bin/python3`

The Python bindings (`lcs` / `lcs_py`) are built and installed against a project-local `.venv`. The same interpreter is used for the C++ build, the stub generator, and the editable install — keep them in sync.

### One-time setup

```bash
# Create the project venv (only once; reuse afterwards)
python3 -m venv .venv

# Install build/dev tooling for the bindings
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install scikit-build-core pybind11 ninja numpy pybind11-stubgen trimesh
```

If `lcs_config.ini` sets `LCS_PYTHON_EXECUTABLE`, point that key to `<repo>/.venv/bin/python` so cmake/xmake configures and installs against the same interpreter.

---

## CMake Build

### Build (Dev Mode)

With the venv activated, configure and build:

```bash
# Configure — MUST include -DLCS_BUILD_PYBINDINGS=ON
cmake -S . -B build \
      -DLCS_BUILD_PYBINDINGS=ON \
      -DLCS_PYTHON_EXECUTABLE="$(pwd)/.venv/bin/python"

# Build + regenerate stubs (requires pybind11-stubgen)
cmake --build build -j --target stubs

# Editable install of the lcs package into the venv
.venv/bin/python -m pip install -e .
```

After C++ binding changes (anything in `PythonBindings/src/python_bindings.cpp`), rerun:
```bash
cmake --build build -j --target stubs
```

---

## XMake Build

### One-time dependency setup

Eigen must be present in `ext/eigen`. If not already cloned:
```bash
git clone --depth 1 -b 3.4 https://gitlab.com/libeigen/eigen.git ext/eigen
```

The LuisaCompute submodule (`ext/LuisaCompute`) and lcpp (`ext/lcpp`) must also be present. Use `xmake lua setup.lua` or clone them manually if needed.

### Configure

```bash
xmake f --lcs_build_pybindings=y \
        --lcs_python_executable="$(pwd)/.venv/bin/python" \
        --lcpp_test=n \
        -c
```

Other useful options:
- `--lcs_enable_gui=y` — enable GUI (polyscope)
- `--lcs_enable_test=y` — enable unit tests
- `--lc_metal_backend=y` / `--lc_cuda_backend=y` — backend selection
- `-m release` / `-m debug` / `-m releasedbg` — build mode

### Build

```bash
# Build the Python bindings module
xmake build lcs_py

# Build + regenerate stubs
xmake build stubs

# Build the main simulation application
xmake build app_simulation

# Build all targets
xmake
```

Output binaries go to `build/bin/`.

### Running tests (xmake)

For Python tests, set `PYTHONPATH` to the xmake output directory:

```bash
PYTHONPATH=build/bin python PythonBindings/tests/test_rigid_joint_animation.py --headless --advance_frames 30
```

Or with the venv interpreter from outside the activated shell:
```bash
PYTHONPATH=build/bin .venv/bin/python PythonBindings/tests/test_rigid_joint_animation.py --headless --advance_frames 30
```

For C++ unit tests (requires `--lcs_enable_test=y`):
```bash
xmake run test_lbvh
xmake run test_narrow_phase
# etc.
```

---

## Running tests (general)

For python tests, most of the time you need to add the launching param `--headless`. With the venv you can call the interpreter directly:

```bash
.venv/bin/python PythonBindings/tests/test_rigid_joint_animation.py --headless --advance_frames 30
```

If you need to invoke the venv interpreter from outside the activated shell, use `<repo>/.venv/bin/python` (this matches `LCS_PYTHON_EXECUTABLE`).

## Critical Flags & Settings

### `-DLCS_BUILD_PYBINDINGS=ON` / `--lcs_build_pybindings=y` (REQUIRED)
Default is OFF. Without this, the `stubs` target does not exist,
and the Python bindings module (`lcs_py`) is not built.

### `LUISA_COMPUTE_USE_SYSTEM_STL=ON` (macOS/Metal, CMake only)
Set in `ext/CMakeLists.txt`. Must be ON for ALL build paths (plain cmake,
SKBUILD, editable install). Setting it OFF for SKBUILD produces Metal
binaries that load but fail device probing with
"No hardware device found for backend 'metal'".
(The xmake build handles this automatically via `ext/xmake.lua`.)

## Build Paths Explained

| Build System | Path | Used For |
|------|---------|----------|
| cmake | `cmake --build build` | Daily dev, stubs |
| cmake (SKBUILD) | `.venv/bin/python -m pip install -e .` | Editable install |
| cmake (SKBUILD) | `pip wheel .` | Distribution wheel |
| xmake | `xmake build lcs_py` | Daily dev |
| xmake | `xmake build stubs` | Stub generation |

Both cmake and xmake output to `build/bin/`.

## Common Errors

### "No module named 'lcs'"
Missing `.venv/bin/python -m pip install -e .`. The `lcs` package is only on `sys.path` after editable install.
For xmake builds, set `PYTHONPATH=build/bin` instead.

### "stubs target not found" (cmake)
Missing `-DLCS_BUILD_PYBINDINGS=ON` in cmake configure. Fix: reconfigure with the flag.

### "No hardware device found for backend 'metal'"
`LUISA_COMPUTE_USE_SYSTEM_STL` was OFF during the build. Ensure it's ON in `ext/CMakeLists.txt` and rebuild.

### "pybind11_stubgen: command not found" or "No module named pybind11_stubgen"
Install pybind11-stubgen on the Python used by `LCS_PYTHON_EXECUTABLE`:
```bash
.venv/bin/python -m pip install pybind11-stubgen
```
Or use `.venv/bin/python -m pip install -e .[dev]` if `pyproject.toml` has the dev extra.

### "Eigen/Dense file not found" (xmake)
Eigen is not cloned. Run:
```bash
git clone --depth 1 -b 3.4 https://gitlab.com/libeigen/eigen.git ext/eigen
```

### "lcpp/device/device_radix_sort.h file not found" (xmake)
The `ext/lcpp` submodule is missing. Clone it:
```bash
git clone https://github.com/Ligo04/lc_parallel_primitive.git ext/lcpp
```
