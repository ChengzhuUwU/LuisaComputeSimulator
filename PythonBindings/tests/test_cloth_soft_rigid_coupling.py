from utils.test_script_path import PROJECT_ROOT
import argparse
import trimesh
import numpy as np
import os, sys

import lcs_py as lcs

import utils.arg_parser
args = utils.arg_parser.parse_args()

# Initialize LuisaCompute device
import platform
backend = "metal" if platform.system() == "Darwin" else "cuda" # backends: cuda, dx, vk, metal (if supported on the platform)
solver = lcs.NewtonSolver()
solver.init_device(backend_name=backend)

# Register meshes

# Load a mesh by providing vertices and triangles array directly
rigid_mesh_path = os.path.join(PROJECT_ROOT, 'Resources', 'InputMesh', 'sphere63.obj')
rigid_mesh = trimesh.load(rigid_mesh_path, process=False)
rigid = solver.create_world_data_from_array('cube', rigid_mesh.vertices, rigid_mesh.faces)
rigid.set_simulation_type(lcs.MaterialType.Rigid)
rigid.set_translation(0.0, 0.34, 0.0)
rigid.set_rotation(0.5235988, 0.0, 0.5235988)
rigid.set_scale(0.1)
rigid_id = solver.register_world_data(rigid)

# Load a mesh by providing the path to the obj file
cloth_mesh_path = os.path.join(PROJECT_ROOT, 'Resources', 'InputMesh', 'square2K.obj')
cloth = solver.create_world_data_from_file_path('cloth', cloth_mesh_path)
cloth.set_simulation_type(lcs.MaterialType.Cloth)
cloth.set_physics_material_cloth(thickness=0.001, youngs_modulus=1e6)
cloth.set_scale(0.75)
cloth.add_fixed_point_by_method("LeftBack")
cloth.add_fixed_point_by_method("RightBack")
cloth.add_fixed_point_by_method("LeftFront")
cloth.add_fixed_point_by_method("RightFront")
cloth_id = solver.register_world_data(cloth)

# Load a tet mesh by providing vertices and tets array directly
from utils.mesh_proc import get_sample_tet_grid
tet_verts, tet_indices = get_sample_tet_grid(origin=(-0.2, 0.3, -0.2), size=(0.5, 0.5, 2.0), resolution=(5, 5, 20))
tet = solver.create_world_data_from_tet_array("tet", tet_verts, tet_indices)
tet.set_simulation_type(lcs.MaterialType.Tetrahedral)
tet.set_physics_material_tet(
	model="ARAP",
	youngs_modulus=1e5,
	poisson_ratio=0.4,
)
tet.set_translation(0.2, 0.5, 0.0)
tet.set_scale(0.2)
tet_id = solver.register_world_data(tet)


# Initialize the solver (builds internal data structures, compiles shaders, etc.)
solver.init_solver()

# Get mesh info
solver.print_registered_meshes_info()

# Set scene parameters
config_ref = solver.get_config()
config_ref.use_floor = False

# Output directory (for optional file saving)
output_dir = os.path.join(PROJECT_ROOT, "Resources", "OutputMesh")
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