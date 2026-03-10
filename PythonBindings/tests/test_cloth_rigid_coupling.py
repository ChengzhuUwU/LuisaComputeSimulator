import trimesh
import numpy as np
import os, sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(root, 'build', 'bin'))
import lcs_py as lcs

import utils.arg_parser
args = utils.arg_parser.parse_args()

# Initialize LuisaCompute device
backend = args.backend  # backends: cuda, dx, vk, metal (if supported on the platform)
solver = lcs.NewtonSolver()
solver.init_device(backend_name=backend)

# Register meshes

# Load a mesh by providing vertices and triangles array directly
cube_mesh_path = os.path.join(root, 'Resources', 'InputMesh', 'cube.obj')
cube_mesh = trimesh.load(cube_mesh_path, process=False)
cube = solver.create_world_data_from_array('cube', cube_mesh.vertices, cube_mesh.faces)
cube.set_simulation_type(lcs.MaterialType.Rigid)
cube.set_translation(0.0, 0.34, 0.0)
cube.set_rotation(0.5235988, 0.0, 0.5235988)
cube.set_scale(0.1)
cube_id = solver.register_world_data(cube)

# Load a mesh by providing the path to the obj file
cloth_mesh_path = os.path.join(root, 'Resources', 'InputMesh', 'square2K.obj')
cloth = solver.create_world_data_from_file_path('cloth', cloth_mesh_path)
cloth.set_simulation_type(lcs.MaterialType.Cloth)
cloth.set_physics_material_cloth(thickness=0.001, youngs_modulus=1e6)
cloth.set_scale(0.75)
cloth.add_fixed_point_by_method("LeftBack")
cloth.add_fixed_point_by_method("RightBack")
cloth.add_fixed_point_by_method("LeftFront")
cloth.add_fixed_point_by_method("RightFront")
cloth_id = solver.register_world_data(cloth)

# Initialize the solver (builds internal data structures, compiles shaders, etc.)
solver.init_solver()

# Get mesh info
solver.print_registered_meshes_info()
cube_get_1 = solver.get_object_by_registration_id(cube_id)

# Set scene parameters
config_ref = solver.get_config()
config_ref.use_floor = False

# Output directory (for optional file saving)
output_dir = os.path.join(root, "Resources", "OutputMesh")
os.makedirs(output_dir, exist_ok=True)

# Launch polyscope GUI or run headless
if args.headless:
	solver.save_sim_result(obj_path=os.path.join(output_dir, "init.obj"))
	for frame in range(0, args.advance_frames):
		solver.physics_step_gpu()
	solver.save_sim_result(obj_path=os.path.join(output_dir, "result.obj"))
else:
	import utils.polyscope_gui 
	gui = utils.polyscope_gui.SimulationGUI(solver, config_ref, output_dir)
	gui.show()

solver.cleanup_device()