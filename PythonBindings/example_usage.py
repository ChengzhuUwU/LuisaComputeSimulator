import trimesh
import numpy as np
import os, sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(root, 'build', 'bin'))
import lcs_py as lcs

# Initialize LuisaCompute device
backend = "metal"  # backends: cuda, dx, vk, metal (if supported on the platform)
lcs.device_init(backend_name=backend, binary_path=None)
solver = lcs.NewtonSolver()

# Register meshes

# Load a mesh by providing vertices and triangles array directly
cube_mesh_path = os.path.join(root, 'Resources', 'InputMesh', 'cube.obj')
cube_mesh = trimesh.load(cube_mesh_path, process=False)
cube_verts = np.asarray(cube_mesh.vertices, dtype=np.double)
cube_faces = np.asarray(cube_mesh.faces, dtype=np.int32)
cube_ref = solver.register_mesh('cube', cube_verts, cube_faces)
cube_ref.set_simulation_type(lcs.SimulationType.Rigid)
cube_ref.set_physics_material_rigid(thickness=1e-3, stiffness=1e6)
cube_ref.set_translation(0.0, 0.34, 0.0)
cube_ref.set_rotation(0.5235988, 0.0, 0.5235988)
cube_ref.set_scale(0.1)

# Load a mesh by providing the path to the obj file
cloth_mesh_path = os.path.join(root, 'Resources', 'InputMesh', 'square2K.obj')
cloth_ref = solver.register_mesh('cloth', cloth_mesh_path)
cloth_ref.set_simulation_type(lcs.SimulationType.Cloth)
cloth_ref.set_physics_material_cloth(thickness=0.001, youngs_modulus=1e6)
cloth_ref.set_scale(0.75)
cloth_ref.add_fixed_point_by_method("LeftBack")
cloth_ref.add_fixed_point_by_method("RightBack")
cloth_ref.add_fixed_point_by_method("LeftFront")
cloth_ref.add_fixed_point_by_method("RightFront")

# Initialize the solver (builds internal data structures, compiles shaders, etc.)
solver.init_solver()

print('Registered meshes:', solver.get_mesh_names())

# Set scene parameters
config_ref = lcs.get_scene_params()
config_ref.use_floor = False

# Output directory (for optional file saving)
output_dir = os.path.join(root, "Resources", "OutputMesh")
os.makedirs(output_dir, exist_ok=True)

# Launch polyscope GUI
from polyscope_gui import SimulationGUI
gui = SimulationGUI(solver, config_ref, output_dir)
gui.show()

# Or run the simulation without GUI
# solver.save_to(full_path=os.path.join(output_dir, "init.obj"))
# for frame in range(0, 30):
# 	solver.physics_step_gpu() # or solver.physics_step_cpu() 
# solver.save_to(full_path=os.path.join(output_dir, "result.obj"))

lcs.device_cleanup()