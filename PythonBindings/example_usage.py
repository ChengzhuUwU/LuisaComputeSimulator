# Example usage of the Python bindings (requires trimesh and the built lcs_solver_py module)
# Install trimesh via: pip install trimesh
import trimesh
import numpy as np

# import the generated module
import lcs_solver_py as lcs

backend = "metal"  # or "cuda", "cpu", etc.

# initialize luisa compute context/device (optional args)
lcs.init(backend_name=backend, binary_path=None)

# Load a mesh using trimesh
mesh = trimesh.load('cube.obj', process=False)
verts = np.asarray(mesh.vertices, dtype=np.double)
faces = np.asarray(mesh.faces, dtype=np.int32)

# build solver and register meshes
solver = lcs.NewtonSolver()
wd = solver.register_mesh('cube', verts, faces)
# chain calls
wd.set_simulation_type(lcs.SimulationType.Cloth).set_physics_material_cloth(thickness=0.01, youngs_modulus=1e4)

# initialize underlying C++ NewtonSolver (creates shaders / internal data)
solver.init_solver()

print('Registered meshes:', solver.get_mesh_names())
print('Num meshes:', solver.num_meshes())

# step once on CPU
solver.physics_step_cpu()

# get vertex positions for each mesh as numpy arrays
results = solver.get_simulation_results()
for i, arr in enumerate(results):
	print(f'mesh {i} positions shape: {arr.shape}')
