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
config_ref.set_nonlinear_iter_count(8)

# Output directory (for optional file saving)
output_dir = os.path.join(PROJECT_ROOT, "Resources", "OutputMesh")
os.makedirs(output_dir, exist_ok=True)

# Launch polyscope GUI or run headless
from utils.test_runner import TestRunner
runner = TestRunner(solver, config_ref, output_dir, headless=args.headless)
runner.run(advance_frames=args.advance_frames)

solver.cleanup_device()