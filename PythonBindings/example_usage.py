import trimesh
import numpy as np
import os, sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(root, 'build', 'bin'))
import lcs_py as lcs


# Initialize luisa compute context/device (optional args)
backend = "cuda"  # backends: cuda, dx, vk, metal (if supported on the platform)
lcs.init(backend_name=backend, binary_path=None)
solver = lcs.NewtonSolver()

# Register meshes

# Load a mesh using trimesh
cube_mesh_path = os.path.join(root, 'Resources', 'InputMesh', 'cube.obj')
cube_mesh = trimesh.load(cube_mesh_path, process=False)
cube_verts = np.asarray(cube_mesh.vertices, dtype=np.double)
cube_faces = np.asarray(cube_mesh.faces, dtype=np.int32)
cube = solver.register_mesh('cube', cube_verts, cube_faces)
cube.set_simulation_type(lcs.SimulationType.Rigid)
cube.set_translation(0.0, 0.34, 0.0)
cube.set_rotation(0.5235988, 0.0, 0.5235988)
cube.set_scale(0.1)

# Load a mesh using obj file path directly 
cloth_mesh_path = os.path.join(root, 'Resources', 'InputMesh', 'square2K.obj')
cloth = solver.register_mesh('cloth', cloth_mesh_path)
cloth.set_simulation_type(lcs.SimulationType.Cloth)
cloth.set_physics_material_cloth(thickness=0.001, youngs_modulus=1e4)
cloth.set_scale(0.75)
cloth.add_fixed_point_by_method("LeftBack")
cloth.add_fixed_point_by_method("RightBack")
cloth.add_fixed_point_by_method("LeftFront")
cloth.add_fixed_point_by_method("RightFront")

# Initialize the solver (builds internal data structures, compiles shaders, etc.)
solver.init_solver()

print('Registered meshes:', solver.get_mesh_names())
print('Num meshes:', solver.num_meshes())

# Set scene parameters
sp = lcs.get_scene_params()
sp.use_floor = False

# Simulate loop
output_dir = os.path.join(root, "Resources", "OutputMesh")
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

solver.save_to(full_path=os.path.join(output_dir, "init.obj"))

def update():
	# solver.update_pinned_verts_position(mesh_idx=1, local_vid=0, target_pos=np.array([0.0, 0.5, 0.0], dtype=_np.float32))
	solver.physics_step_gpu() # cpu is invalid???

for frame in range(0, 1):
	update()

# results = solver.get_simulation_results()
solver.save_to(full_path=os.path.join(output_dir, "result.obj"))