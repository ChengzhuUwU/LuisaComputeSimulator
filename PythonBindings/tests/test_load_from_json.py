from utils.test_script_path import PROJECT_ROOT
import trimesh
import numpy as np
import os, sys

import lcs_py as lcs

# import utils.arg_parser
# args = utils.arg_parser.parse_args()
from run_tests import create_default_parser

parser = create_default_parser()
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
from utils.test_runner import TestRunner
runner = TestRunner(solver, config_ref, output_dir, headless=args.headless)
runner.run(advance_frames=args.advance_frames)

solver.cleanup_device()