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
from utils.mesh_proc import get_sample_tet_grid
tet_verts, tet_indices = get_sample_tet_grid(origin=(-0.2, 0.3, -0.2), size=(1, 1, 1), resolution=(10, 10, 10))

delta = 0.102
count = 0
for i in range(5):
    for j in range(5):
        for k in range(5):
            name = f'soft_grid_{i}_{j}_{k}'
            tet = solver.create_world_data_from_tet_array("tet", tet_verts, tet_indices)
            tet.set_simulation_type(lcs.MaterialType.Tetrahedral)
            tet.set_physics_material_tet(
                model="ARAP",
                youngs_modulus=1e5,
                poisson_ratio=0.4,
            )
            tet.set_scale(0.1)
            tet.set_translation(i * delta, 0.1 + j * delta, k * delta)
            tet_id = solver.register_world_data(tet)
            count += 1
# Initialize the solver (builds internal data structures, compiles shaders, etc.)
solver.init_solver()

# Get mesh info
solver.print_registered_meshes_info()

# Set scene parameters
config_ref = solver.get_config()
config_ref.nonlinear_iter_count = 8

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