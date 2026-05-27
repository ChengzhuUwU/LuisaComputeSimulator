## Basic Building Operation

If you are coding agent, before configuration, you need to read the configuration params in [lcs_config.ini](/lcs_config.ini) (If the file not exists, you need to create one from copying [lcs_config_template.ini](/lcs_config_template.ini)). If value of the key is not empty, you need to apply the value in configuration:
- You can ignore the param if value is empty: `CMAKE_BUILD_TYPE = `
- You need to apply the param if value is not empty: `LCS_PYTHON_EXECUTABLE = /opt/homebrew/bin/python3`

The Python bindings (`lcs` / `lcs_py`) are built and installed against a project-local `.venv`. The same interpreter is used for the C++ build, the stub generator, and the editable install — keep them in sync.

### One-time setup

```bash
# Create the project venv (only once; reuse afterwards)
python3 -m venv .venv
source .venv/bin/activate

# Install build/dev tooling for the bindings
pip install --upgrade pip
pip install scikit-build-core pybind11 ninja numpy pybind11-stubgen trimesh
```

If `lcs_config.ini` sets `LCS_PYTHON_EXECUTABLE`, point that key to `<repo>/.venv/bin/python` so cmake configures and installs against the same interpreter.

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
pip install -e .
```

After C++ binding changes (anything in `PythonBindings/src/python_bindings.cpp`), rerun:
```bash
cmake --build build -j --target stubs
```

### Running tests

For python tests, most of the time you need to add the launching param `--headless`. With the venv active you can call the interpreter directly:

```bash
python PythonBindings/tests/test_rigid_joint_animation.py --headless --advance_frames 30
```

If you need to invoke the venv interpreter from outside the activated shell, use `<repo>/.venv/bin/python` (this matches `LCS_PYTHON_EXECUTABLE`).

## Critical Flags & Settings

### `-DLCS_BUILD_PYBINDINGS=ON` (REQUIRED)
Default is OFF. Without this, the `stubs` cmake target does not exist,
and the Python bindings module (`lcs_py`) is not built.

### `LUISA_COMPUTE_USE_SYSTEM_STL=ON` (macOS/Metal)
Set in `ext/CMakeLists.txt`. Must be ON for ALL build paths (plain cmake,
SKBUILD, editable install). Setting it OFF for SKBUILD produces Metal
binaries that load but fail device probing with
"No hardware device found for backend 'metal'".

## Build Paths Explained

| Path | SKBUILD | USE_SYSTEM_STL | Used For |
|------|---------|----------------|----------|
| `cmake --build build` | 0 | ON | Daily dev, stubs |
| `pip install -e .` | 1 | ON | Editable install |
| `pip wheel .` | 1 | ON | Distribution wheel |

The `build/` directory (plain cmake) outputs to `build/bin/`.
The `build/{wheel_tag}/` directory (SKBUILD) is scikit-build-core's
isolated build tree for wheels.

## Common Errors

### "No module named 'lcs'"
Missing `pip install -e .`. The `lcs` package is only on `sys.path` after editable install.

### "stubs target not found"
Missing `-DLCS_BUILD_PYBINDINGS=ON` in cmake configure. Fix: reconfigure with the flag.

### "No hardware device found for backend 'metal'"
`LUISA_COMPUTE_USE_SYSTEM_STL` was OFF during the build. Ensure it's ON in `ext/CMakeLists.txt` and rebuild.

### "pybind11_stubgen: command not found" or "No module named pybind11_stubgen"
Install pybind11-stubgen on the Python used by `LCS_PYTHON_EXECUTABLE`:
```bash
pip install pybind11-stubgen
```
Or use `pip install -e .[dev]` if `pyproject.toml` has the dev extra.
