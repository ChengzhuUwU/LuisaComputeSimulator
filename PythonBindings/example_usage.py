import trimesh
import numpy as np
import os, sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(root, 'build', 'bin'))
import lcs_py as lcs


# Initialize luisa compute context/device (optional args)
backend = "cuda"  # backends: cuda, dx, vk, metal (if supported on the platform)
lcs.init(backend_name=backend, binary_path=None)

# Load a mesh using trimesh
mesh_path = os.path.join(root, 'Resources', 'InputMesh', 'cube.obj')
mesh = trimesh.load(mesh_path, process=False)
verts = np.asarray(mesh.vertices, dtype=np.double)
faces = np.asarray(mesh.faces, dtype=np.int32)

# Register meshes
solver = lcs.NewtonSolver()
cube = solver.register_mesh('cube', verts, faces)
cube.set_simulation_type(lcs.SimulationType.Rigid)
cube.set_physics_material_cloth(thickness=0.01, youngs_modulus=1e4)
cube.set_translation(0.0, 0.34, 0.0)
cube.set_rotation(0.5235988, 0.0, 0.5235988)
cube.set_scale(0.1)

# Initialize the solver (builds internal data structures, compiles shaders, etc.)
solver.init_solver()

print('Registered meshes:', solver.get_mesh_names())
print('Num meshes:', solver.num_meshes())


# Simulate loop
output_dir = os.path.join(root, "Resources", "OutputMesh")
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

solver.save_to(full_path=os.path.join(output_dir, "init.obj"))

def on_draw():
	pass

def update():
	solver.physics_step_gpu() # cpu is invalid???

for frame in range(0, 50):
	update()

# results = solver.get_simulation_results()
solver.save_to(full_path=os.path.join(output_dir, "result.obj"))