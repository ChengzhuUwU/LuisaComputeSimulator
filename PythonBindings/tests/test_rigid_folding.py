from utils.test_script_path import PROJECT_ROOT
import trimesh
import numpy as np
import os, sys

import lcs_py as lcs

from run_tests import create_default_parser
args = create_default_parser().parse_args()

# Initialize LuisaCompute device
backend = args.backend  # backends: cuda, dx, vk, metal (if supported on the platform)
solver = lcs.NewtonSolver()
solver.init_device(backend_name=backend)

# Register meshes
import trimesh
cube_mesh_path = os.path.join(PROJECT_ROOT, 'Resources', 'InputMesh', 'cube.obj')
cube_mesh = trimesh.load(cube_mesh_path, process=False)
count = 0
for i in range(5):
    for j in range(5):
        for k in range(5):
            name = f'cube_{i}_{j}_{k}'
            cube = solver.create_world_data_from_array(name, cube_mesh.vertices, cube_mesh.faces)
            cube.set_simulation_type(lcs.MaterialType.Rigid)
            cube.set_scale(0.1)
            delta = 0.102
            cube.set_translation(i * delta, 0.1 + j * delta, k * delta)
            solver.register_world_data(cube)
            count += 1
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