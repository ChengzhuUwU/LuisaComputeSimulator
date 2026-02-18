import trimesh
import numpy as np
import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(root, 'build', 'bin'))

# import the generated module
import lcs_py as lcs

backend = "cuda"  # backends: cuda, dx, vk, metal (if supported on the platform)

# initialize luisa compute context/device (optional args)
lcs.init(backend_name=backend, binary_path=None)

# Load a mesh using trimesh
mesh_path = os.path.join(root, 'Resources', 'InputMesh', 'cube.obj')
mesh = trimesh.load(mesh_path, process=False)
verts = np.asarray(mesh.vertices, dtype=np.double)
faces = np.asarray(mesh.faces, dtype=np.int32)

# "translation": [
# 	0.0,
# 	0.34,
# 	0.0
# ],
# "rotation": [
# 	0.5235988,
# 	0.0,
# 	0.5235988
# ],
# "scale": 0.1,
# "shell_type": "Rigid"

# build solver and register meshes
solver = lcs.NewtonSolver()
cube = solver.register_mesh('cube', verts, faces)
cube.set_simulation_type(lcs.SimulationType.Rigid)
cube.set_physics_material_cloth(thickness=0.01, youngs_modulus=1e4)
cube.set_translation(0.0, 0.34, 0.0)
cube.set_rotation(0.5235988, 0.0, 0.5235988)
cube.set_scale(0.1)



# cube = solver.register_mesh('cube', mesh_path)

# initialize the solver (builds internal data structures, compiles shaders, etc.)
solver.init_solver()

print('Registered meshes:', solver.get_mesh_names())
print('Num meshes:', solver.num_meshes())

# step once on CPU
solver.physics_step_cpu()

# get vertex positions for each mesh as numpy arrays
results = solver.get_simulation_results()
for i, arr in enumerate(results):
	print(f'mesh {i} positions shape: {arr.shape}')
