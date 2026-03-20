# C++ API Reference

This document provides a complete reference for the C++ API of LuisaComputeSimulator.

## Overview

The C++ API provides low-level access to the simulator with maximum performance and flexibility. It uses the builder pattern for intuitive object creation.

## Include Headers

```cpp
#include "SimulationSolver/newton_solver.h"
#include "SimulationCore/world_data.h"
```

---

## Core Classes

### `lcs::NewtonSolver`

The main solver class for physics simulation.

#### Header
```cpp
#include "SimulationSolver/newton_solver.h"
```

#### Constructor
```cpp
lcs::NewtonSolver solver;
```

---

### Device Initialization

```cpp
void create_device(const std::string& binary_path, const std::string& backend)
```

**Parameters:**
- `binary_path`: Path to the executable (usually `argv[0]`)
- `backend`: Backend string (`"cuda"`, `"dx"`, `"vk"`, `"metal"`, `""`)

**Example:**
```cpp
lcs::NewtonSolver solver;
solver.create_device(argv[0], "cuda");  // Use CUDA backend
solver.create_device(argv[0], "");      // Use default backend
```

---

## World Data Creation

### Builder Pattern

Use `lcs::Initializer::WorldData()` to create simulation objects:

```cpp
auto world_data = lcs::Initializer::WorldData()
    .set_name("object_name")
    .load_mesh_from_path("path/to/mesh.obj")
    .set_material_type(lcs::Material::MaterialType::Cloth)
    .set_physics_material(lcs::Material::ClothMaterial{...})
    .set_translation({x, y, z})
    .set_rotation({rx, ry, rz})
    .set_scale(s)
    .add_fixed_point_from_method({.method = lcs::Initializer::FixedPointsType::Left});
```

---

### Loading Meshes

#### From File

```cpp
WorldData& load_mesh_from_path(const std::string_view& path)
```

**Example:**
```cpp
auto cloth = lcs::Initializer::WorldData()
    .set_name("cloth")
    .load_mesh_from_path("Resources/InputMesh/square2.obj");
```

#### From Array

```cpp
WorldData& load_mesh_from_array(
    const std::vector<std::array<float, 3>>& vertices,
    const std::vector<std::array<uint, 3>>& faces)
```

**Example:**
```cpp
std::vector<std::array<float, 3>> vertices{
    {-0.5f, 0.0f, -0.5f},
    { 0.5f, 0.0f, -0.5f},
    {-0.5f, 0.0f,  0.5f},
    { 0.5f, 0.0f,  0.5f}
};
std::vector<std::array<uint, 3>> faces{
    {0, 3, 1},
    {0, 2, 3}
};

auto square = lcs::Initializer::WorldData()
    .set_name("square")
    .load_mesh_from_array(vertices, faces);
```

#### From Tetrahedral Mesh

```cpp
WorldData& load_tet_mesh_from_array(
    const std::vector<std::array<float, 3>>& vertices,
    const std::vector<std::array<uint, 4>>& tets)
```

---

## Material Types

### Enum: `lcs::Material::MaterialType`

| Type          | Description                 |
| ------------- | --------------------------- |
| `Cloth`       | Cloth/soft body simulation  |
| `Rigid`       | Rigid body simulation       |
| `Tetrahedral` | Tetrahedral volumetric mesh |
| `Rod`         | Elastic rod simulation      |

**Usage:**
```cpp
.set_material_type(lcs::Material::MaterialType::Cloth)
```

---

## Material Configuration

### Cloth Material

```cpp
struct ClothMaterial {
    ConstitutiveStretchModelCloth stretch_model;  // Spring, PBD, Corotated
    ConstitutiveBendingModelCloth bending_model;   // None, Bending
    float thickness = 0.001f;
    float youngs_modulus = 1e6f;
    float poisson_ratio = 0.0f;
    float area_bending_stiffness = 1e-5f;
    float d_hat = 0.001f;
    float contact_offset = 0.001f;
};
```

**Usage:**
```cpp
.set_physics_material(lcs::Material::ClothMaterial{
    .stretch_model = lcs::Material::ConstitutiveStretchModelCloth::Spring,
    .bending_model = lcs::Material::ConstitutiveBendingModelCloth::None,
    .thickness = 0.001f,
    .youngs_modulus = 1e6f
})
```

### Rigid Material

```cpp
struct RigidMaterial {
    ConstitutiveModelRigid model;      // Fixed, Kinematic, Dynamic
    float thickness = 0.001f;
    float stiffness = 1e10f;
    float density = 1000.0f;
    float mass = 1.0f;
    float d_hat = 0.001f;
    float contact_offset = 0.001f;
};
```

### Tetrahedral Material

```cpp
struct TetMaterial {
    ConstitutiveModelTet model;        // Corotated, NeoHookean, StableNeoHookean
    float youngs_modulus = 1e6f;
    float poisson_ratio = 0.4f;
    float density = 1000.0f;
    float mass = 1.0f;
    float d_hat = 0.001f;
    float contact_offset = 0.001f;
    bool is_shell = false;
};
```

---

## Transform Configuration

### Translation

```cpp
// Method 1: Float3
.set_translation(float3{ x, y, z })

// Method 2: Individual components
.set_translation(float x, float y, float z)
```

### Rotation

```cpp
// Rotation is specified as Euler angles (in radians) along each axis
.set_rotation(float3{ rx, ry, rz })  // Radians
.set_rotation(float x, float y, float z)
```

### Scale

```cpp
.set_scale(float s)  // Uniform scale
```

---

## Fixed Points (Constraints)

### FixedPointsType Enum

```cpp
enum struct FixedPointsType {
    None,
    FromIndices,
    FromFunction,
    Left,           // Left edge
    Right,          // Right edge
    Front,          // Front edge
    Back,           // Back edge
    Up,             // Top edge
    Down,           // Bottom edge
    LeftUp,
    LeftDown,
    LeftFront,
    LeftBack,
    RightUp,
    RightDown,
    RightFront,
    RightBack,
    All,
    Center
};
```

### Adding Fixed Points

```cpp
.add_fixed_point_from_method({.method = lcs::Initializer::FixedPointsType::Left})
.add_fixed_point_from_method({.method = lcs::Initializer::FixedPointsType::LeftBack})
.add_fixed_point_from_method({.method = lcs::Initializer::FixedPointsType::Right})
```

**Example - Pin both corners:**
```cpp
.add_fixed_point_from_method({.method = lcs::Initializer::FixedPointsType::Left})
.add_fixed_point_from_method({.method = lcs::Initializer::FixedPointsType::Right})
```

---

## Solver Configuration

### Get Config Reference

```cpp
auto config = solver.get_config();
```

### Configurable Properties

| Property                | Type   | Default        | Description             |
| ----------------------- | ------ | -------------- | ----------------------- |
| `use_floor`             | bool   | `true`         | Enable ground collision |
| `implicit_dt`           | float  | `1/60`         | Time step (seconds)     |
| `use_energy_linesearch` | bool   | `true`         | Enable line search      |
| `gravity`               | float3 | `{0, -9.8, 0}` | Gravity vector          |
| `contact_stiffness`     | float  | `1e2`          | Contact stiffness       |
| `dhat`                  | float  | `0.001`        | Barrier parameter       |
| `max_iterations`        | int    | `100`          | Newton iterations       |

**Example:**
```cpp
auto config = solver.get_config();
config.use_floor = false;
config.implicit_dt = 1.0f / 60.0f;
config.gravity = make_float3(0.0f, -9.8f, 0.0f);
```

---

## Registration

```cpp
uint register_world_data(WorldData& world_data)
```

**Returns:** Registration ID for the object

**Example:**
```cpp
uint cloth_id = solver.register_world_data(cloth);
uint cube_id = solver.register_world_data(cube);
```

---

## Solver Operations

### Initialize

```cpp
solver.init_solver()
```

Must be called after all world data is registered. Compiles shaders and initializes data structures.

### Physics Step

```cpp
// GPU simulation
solver.physics_step_GPU();

// CPU simulation
solver.physics_step_CPU();
```

### Restart

```cpp
solver.restart_system();
```

Resets simulation to initial state.

---

## Result Extraction

### Get Current Vertices

```cpp
void get_curr_vertices_to_host(std::vector<std::vector<std::array<float, 3>>>& vertices)
```

**Example:**
```cpp
std::vector<std::vector<std::array<float, 3>>> rendering_vertices;
solver.get_curr_vertices_to_host(rendering_vertices);

// Access vertices for each mesh
for (const auto& mesh_verts : rendering_vertices) {
    // mesh_verts[i] is {x, y, z} for vertex i
}
```

### Get Triangles

```cpp
void get_triangles_to_host(std::vector<std::vector<std::array<uint, 3>>>& triangles)
```

### Save to OBJ

```cpp
void save_mesh_to_obj(const std::string& path)
```

---

## Complete Example

```cpp
#include <string>
#include <vector>
#include "SimulationSolver/newton_solver.h"

int main(int argc, char** argv) {
    // Initialize logging
    luisa::log_level_info();

    // Create solver
    lcs::NewtonSolver solver;
    solver.create_device(argv[0], "cuda");

    // === Create first cloth (upper square) ===
    auto upper_square = lcs::Initializer::WorldData()
        .set_name("upper square")
        .load_mesh_from_path("Resources/InputMesh/square2.obj")
        .set_material_type(lcs::Material::MaterialType::Cloth)
        .set_physics_material(lcs::Material::ClothMaterial{
            .stretch_model = lcs::Material::ConstitutiveStretchModelCloth::Spring,
        })
        .set_translation({0.0f, 0.4f, 0.0f})
        .add_fixed_point_from_method({
            .method = lcs::Initializer::FixedPointsType::LeftBack
        });
    uint upper_id = solver.register_world_data(upper_square);

    // === Create second cloth (lower square) ===
    std::vector<std::array<float, 3>> vertices{
        {-0.5f, 0.0f, -0.5f},
        { 0.5f, 0.0f, -0.5f},
        {-0.5f, 0.0f,  0.5f},
        { 0.5f, 0.0f,  0.5f}
    };
    std::vector<std::array<uint, 3>> faces{
        {0, 3, 1},
        {0, 2, 3}
    };
    
    auto lower_square = lcs::Initializer::WorldData()
        .set_name("lower square")
        .load_mesh_from_array(vertices, faces)
        .set_material_type(lcs::Material::MaterialType::Cloth)
        .set_physics_material(lcs::Material::ClothMaterial{})
        .set_scale(0.8f)
        .set_translation({0.1f, 0.2f, 0.0f})
        .add_fixed_point_from_method({.method = lcs::Initializer::FixedPointsType::Left})
        .add_fixed_point_from_method({.method = lcs::Initializer::FixedPointsType::Right});
    uint lower_id = solver.register_world_data(lower_square);

    // === Configure solver ===
    auto config = solver.get_config();
    config.use_floor = false;
    config.implicit_dt = 0.2f;
    config.use_energy_linesearch = true;

    // === Initialize ===
    solver.init_solver();

    // === Run simulation ===
    std::vector<std::vector<std::array<float, 3>>> rendering_vertices;
    
    for (uint frame = 0; frame < 20; ++frame) {
        solver.physics_step_GPU();
        solver.get_curr_vertices_to_host(rendering_vertices);
        
        // Process vertices...
    }

    // === Save result ===
    solver.save_mesh_to_obj("output/result.obj");

    return 0;
}
```

---

## Building with CMake

```cmake
# CMakeLists.txt
add_executable(my_simulation main.cpp)

target_include_directories(my_simulation PRIVATE 
    ${CMAKE_SOURCE_DIR}/Solver/SimulationSolver
    ${CMAKE_SOURCE_DIR}/Solver/SimulationCore
    ${CMAKE_SOURCE_DIR}/ext/LuisaCompute/include
)

target_link_libraries(my_simulation PRIVATE 
    luisa-compute-solver-lib
)
```

---

## Resource Path Macro

Use `LCSV_RESOURCE_PATH` to access bundled resources:

```cpp
std::string mesh_path = std::string(LCSV_RESOURCE_PATH) + "/InputMesh/square2.obj";
```

This macro points to the `Resources` directory in the build tree.
