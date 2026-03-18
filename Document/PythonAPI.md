# Python API Reference

This document provides a complete reference for the Python bindings of LuisaComputeSimulator.

## Installation

```bash
# Build with Python bindings
cmake -S . -B build \
  -DLCS_BUILD_PYBINDINGS=ON \
  -DLCS_PYTHON_EXECUTABLE=/usr/bin/python3

cmake --build build -j

# Add to Python path
export PYTHONPATH=$PYTHONPATH:/path/to/LuisaComputeSimulator/build/bin
```

## Dependencies

```bash
pip install numpy trimesh polyscope
```

---

## Core Classes

### `lcs.NewtonSolver`

The main solver class for physics simulation.

#### Constructor

```python
solver = lcs.NewtonSolver()
```

#### Device Initialization

```python
solver.init_device(backend_name: str, binary_path: str = None)
```

**Parameters:**
- `backend_name` (str): Backend to use. Options: `"cuda"`, `"dx"`, `"vk"`, `"metal"`, `"fallback"`
- `binary_path` (str, optional): Path to the binary. Defaults to None.

**Example:**
```python
solver.init_device(backend_name="cuda")
solver.init_device(backend_name="metal")  # macOS
```

---

### World Data Creation

#### From NumPy Array

```python
world_data = solver.create_world_data_from_array(
    name: str,
    vertices: np.ndarray,    # Shape: (N, 3)
    triangles: np.ndarray    # Shape: (M, 3)
)
```

**Example:**
```python
import numpy as np
import trimesh

# Load mesh from file
cube = trimesh.load("cube.obj", process=False)
cube_data = solver.create_world_data_from_array(
    "cube",
    cube.vertices,
    cube.faces
)
```

#### From OBJ File

```python
world_data = solver.create_world_data_from_file_path(
    name: str,
    obj_file_path: str
)
```

**Example:**
```python
cloth = solver.create_world_data_from_file_path(
    "cloth",
    "Resources/InputMesh/square2K.obj"
)
```

#### From Tetrahedral Mesh

```python
world_data = solver.create_world_data_from_tet_array(
    name: str,
    vertices: np.ndarray,    # Shape: (N, 3)
    tets: np.ndarray        # Shape: (M, 4)
)
```

---

### World Data Methods

Chainable methods to configure simulation objects:

#### Basic Properties

```python
world_data.set_name(name: str) -> WorldData
```

```python
world_data.set_translation(x: float, y: float, z: float) -> WorldData
```

```python
world_data.set_rotation(x: float, y: float, z: float) -> WorldData  # Euler angles (radians)
```

```python
world_data.set_scale(scale: float) -> WorldData
```

#### Simulation Type

```python
world_data.set_simulation_type(material_type: lcs.MaterialType)
```

**Material Types:**
- `lcs.MaterialType.Cloth` - Cloth/soft body
- `lcs.MaterialType.Rigid` - Rigid body
- `lcs.MaterialType.Tetrahedral` - Tetrahedral mesh
- `lcs.MaterialType.Rod` - Elastic rod

---

### Material Configuration

#### Cloth Material

```python
world_data.set_physics_material_cloth(
    stretch_model: str = "Spring",      # "Spring", "PBD", "Corotated"
    bending_model: str = "None",       # "None", "Bending"
    thickness: float = 0.001,
    youngs_modulus: float = 1e6,
    poisson_ratio: float = 0.0,
    area_bending_stiffness: float = 1e-5,
    d_hat: float = 0.001,
    contact_offset: float = 0.001
)
```

**Example:**
```python
cloth.set_physics_material_cloth(
    thickness=0.001,
    youngs_modulus=1e6,
    stretch_model="Spring"
)
```

#### Rigid Material

```python
world_data.set_physics_material_rigid(
    model: str = "Fixed",               # "Fixed", "Kinematic", "Dynamic"
    thickness: float = 0.001,
    stiffness: float = 1e10,
    density: float = 1000.0,
    mass: float = 1.0,
    d_hat: float = 0.001,
    contact_offset: float = 0.001
)
```

#### Tetrahedral Material

```python
world_data.set_physics_material_tet(
    model: str = "Corotated",           # "Corotated", "NeoHookean", "StableNeoHookean"
    youngs_modulus: float = 1e6,
    poisson_ratio: float = 0.4,
    density: float = 1000.0,
    mass: float = 1.0,
    d_hat: float = 0.001,
    contact_offset: float = 0.001
)
```

#### Rod Material

```python
world_data.set_physics_material_rod(
    model: str = "Elastic",
    radius: float = 0.001,
    bending_stiffness: float = 1e-6,
    twisting_stiffness: float = 1e-6,
    density: float = 1000.0,
    mass: float = 1.0,
    d_hat: float = 0.001,
    contact_offset: float = 0.001
)
```

---

### Constraints

#### Fixed Points (Pinned Vertices)

```python
# Pin vertices by method
world_data.add_fixed_point_by_method(
    method: str,    # "Left", "Right", "Top", "Bottom", 
                    # "LeftTop", "RightTop", "LeftBottom", "RightBottom",
                    # "Center", "All", or vertex index
    range: float = 0.0    # Optional range for pinning
)

# Pin specific vertex indices
world_data.add_fixed_point_by_indices(indices: np.ndarray)
```

**Example:**
```python
# Pin corners of cloth
cloth.add_fixed_point_by_method("LeftBack")
cloth.add_fixed_point_by_method("RightBack")

# Pin specific vertices
cloth.add_fixed_point_by_indices(np.array([0, 1, 2, 3]))
```

---

### Registration

```python
registration_id = solver.register_world_data(world_data: WorldData) -> int
```

**Example:**
```python
cube_id = solver.register_world_data(cube)
cloth_id = solver.register_world_data(cloth)
```

---

### Solver Configuration

#### Get Config Reference

```python
config = solver.get_config()
```

**Configurable Properties:**

| Property                | Type  | Default        | Description                |
| ----------------------- | ----- | -------------- | -------------------------- |
| `use_floor`             | bool  | `True`         | Enable ground collision    |
| `implicit_dt`           | float | `1/60`         | Time step size (seconds)   |
| `use_energy_linesearch` | bool  | `True`         | Enable line search         |
| `gravity`               | tuple | `(0, -9.8, 0)` | Gravity vector             |
| `contact_stiffness`     | float | `1e2`          | Contact barrier stiffness  |
| `dhat`                  | float | `0.001`        | Barrier function parameter |
| `max_iterations`        | int   | `100`          | Newton solver iterations   |

**Example:**
```python
config = solver.get_config()
config.use_floor = False
config.implicit_dt = 1/60
config.gravity = (0, -9.8, 0)
config.max_iterations = 50
```

---

### Solver Operations

#### Initialize Solver

```python
solver.init_solver()
```

Compiles shaders and initializes internal data structures. Must be called after registering all world data.

#### Physics Step

```python
# GPU simulation
solver.physics_step_gpu()

# CPU simulation  
solver.physics_step_cpu()
```

#### Restart Simulation

```python
solver.restart_system()
```

Resets the simulation to initial state.

---

### Animation Updates

#### Per-Vertex Animation

```python
solver.update_per_vertex_animation(
    mesh_index: int,
    local_vertex_id: int,
    target_position: np.ndarray  # Shape: (3,)
)
```

#### Per-Body Animation

```python
solver.update_per_body_animation(
    mesh_index: int,
    target_translation: np.ndarray,  # Shape: (3,)
    target_rotation: np.ndarray       # Shape: (3,), Euler angles
)
```

---

### Result Extraction

#### Get All Results

```python
vertices_list, faces_list = solver.get_sim_result()
```

**Returns:**
- `vertices_list`: List of vertex arrays, each shape (N, 3)
- `faces_list`: List of face arrays, each shape (M, 3)

#### Get Result by Registration ID

```python
vertices, faces = solver.get_object_sim_result_by_registration_id(registration_id: int)
```

#### Save to OBJ

```python
solver.save_sim_result(obj_path: str, registration_id: int = None)
```

**Parameters:**
- `obj_path` (str): Output file path
- `registration_id` (int, optional): Specific object to save. Saves all if None.

---

### Query Methods

```python
# Get all registered mesh names
names = solver.get_mesh_names()

# Get object by registration ID
obj = solver.get_object_by_registration_id(registration_id: int)

# Print debug info
solver.print_registered_meshes_info()

# Get number of meshes
num = solver.num_meshes()
```

---

## Complete Example

```python
import numpy as np
import trimesh
import lcs_py as lcs

# Initialize
solver = lcs.NewtonSolver()
solver.init_device(backend_name="cuda")

# Create rigid cube
cube_mesh = trimesh.load("Resources/InputMesh/cube.obj", process=False)
cube = solver.create_world_data_from_array(
    "cube",
    cube_mesh.vertices,
    cube_mesh.faces
)
cube.set_simulation_type(lcs.MaterialType.Rigid)
cube.set_translation(0.0, 0.34, 0.0)
cube.set_rotation(0.5, 0.0, 0.5)
cube.set_scale(0.1)
cube_id = solver.register_world_data(cube)

# Create cloth
cloth = solver.create_world_data_from_file_path(
    "cloth",
    "Resources/InputMesh/square2K.obj"
)
cloth.set_simulation_type(lcs.MaterialType.Cloth)
cloth.set_physics_material_cloth(
    thickness=0.001,
    youngs_modulus=1e6
)
cloth.set_scale(0.75)
cloth.add_fixed_point_by_method("LeftBack")
cloth.add_fixed_point_by_method("RightBack")
cloth_id = solver.register_world_data(cloth)

# Configure
config = solver.get_config()
config.use_floor = False
config.implicit_dt = 1/60

# Initialize and run
solver.init_solver()

# Save initial state
solver.save_sim_result("output/init.obj")

# Run simulation
for frame in range(100):
    solver.physics_step_gpu()

# Save final result
solver.save_sim_result("output/result.obj")

# Or get results programmatically
vertices, faces = solver.get_object_sim_result_by_registration_id(cube_id)
```

---

## Command Line Arguments

When running Python scripts, the following arguments are supported:

```bash
python script.py --backend cuda --headless --advance_frames 60
```

| Argument           | Description                                  | Default |
| ------------------ | -------------------------------------------- | ------- |
| `--backend`        | Backend to use (`cuda`, `dx`, `vk`, `metal`) | `cuda`  |
| `--headless`       | Run without GUI                              | `False` |
| `--advance_frames` | Number of frames to simulate                 | `30`    |
