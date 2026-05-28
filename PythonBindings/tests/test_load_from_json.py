from utils.test_script_path import PROJECT_ROOT
import trimesh
import numpy as np
import os, sys

import lcs_py as lcs

# import utils.arg_parser
# args = utils.arg_parser.parse_args()
import argparse
parser = argparse.ArgumentParser(description="Load from json example")
parser.add_argument("--backend", type=str, default="metal", choices=["cuda", "dx", "metal", "vk", "fallback", "cpu", "remote"], help="Compute backend to use (default: metal)")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--advance_frames", type=int, default=1, help="Number of simulation frames to advance in headless mode (default: 30)")
args = parser.parse_args()

# Initialize LuisaCompute device
backend = args.backend  # backends: cuda, dx, vk, metal (if supported on the platform)
solver = lcs.NewtonSolver()
solver.init_device(backend_name=backend)

# Register meshes
input_dir = os.path.join(PROJECT_ROOT, "Resources", "Scenes", "default_scene.json")
solver.load_scene_from_json(input_dir)

# Initialize the solver (builds internal data structures, compiles shaders, etc.)
solver.init_solver()

# Get mesh info
solver.print_registered_meshes_info()

# Set scene parameters
config_ref = solver.get_config()

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