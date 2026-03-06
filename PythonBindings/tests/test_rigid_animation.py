import numpy as np
import os, sys
from dataclasses import dataclass

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(root, 'build', 'bin'))
import lcs_py as lcs

import utils.arg_parser
args = utils.arg_parser.parse_args()

# Initialize LuisaCompute device
backend = args.backend  # backends: cuda, dx, vk, metal (if supported on the platform)
solver = lcs.NewtonSolver()
solver.init_device(backend_name=backend)

# Load a mesh by providing the path to the obj file
cube_mesh_path = os.path.join(root, 'Resources', 'InputMesh', 'cube.obj')

cube_top_ref = solver.register_mesh_from_file_path('cube', cube_mesh_path)
cube_top_ref.set_simulation_type(lcs.MaterialType.Rigid)
cube_top_ref.set_scale(0.1)
cube_top_ref.set_translation(0.0, 0.14, 0.0)

cube_bottom_ref = solver.register_mesh_from_file_path('cube', cube_mesh_path)
cube_bottom_ref.set_simulation_type(lcs.MaterialType.Rigid)
cube_bottom_ref.set_scale(0.1)
cube_bottom_ref.set_translation(0.0, 0.01, 0.0)
cube_bottom_ref.add_fixed_point_by_method("All")

# from utils.animation_transform import FixedPointTransform
# from utils.vertex_animation import PinnedVertexAnimator
# animator = PinnedVertexAnimator(cloth_ref)

# animator.add_rule_by_method(
# 	"Left",
# 	FixedPointTransform(
# 		use_translate=True,
# 		translate=[0.000, 0.1, 0.0]
#     )
# )
# animator.add_rule_by_method(
# 	"Right",
# 	FixedPointTransform(
# 		use_translate=True,
# 		translate=[0.000, 0.1, 0.0]
#     )
# )

# Initialize the solver (builds internal data structures, compiles shaders, etc.)
solver.init_solver()

# Set scene parameters
config_ref = solver.get_config()
config_ref.nonlinear_iter_count = 2
config_ref.use_floor = True
config_ref.use_self_collision = True
config_ref.use_ccd_linesearch = True
config_ref.use_energy_linesearch = False
config_ref.implicit_dt = 1.0 / 60.0
config_ref.gravity = lcs.Float3(0.0, -9.8, 0.0)
# config_ref.pcg_iter_count = 50
# config_ref.gravity = lcs.Float3(0.0, 0.0, 0.0)
# config_ref.use_floor = False


# Output directory (for optional file saving)
output_dir = os.path.join(root, "Resources", "OutputMesh")
os.makedirs(output_dir, exist_ok=True)

# Launch polyscope GUI or run headless
if args.headless:
	solver.save_sim_result(obj_path=os.path.join(output_dir, "init.obj"))
	for _ in range(0, args.advance_frames):
		if config_ref.use_gpu:
			solver.physics_step_gpu()
		else:
			solver.physics_step_cpu()
	solver.save_sim_result(obj_path=os.path.join(output_dir, "result.obj"))
else:
	import utils.polyscope_gui

	class AnimatedSimulationGUI(utils.polyscope_gui.SimulationGUI):
		def __init__(self, solver_ref, cfg_ref, out_dir, pinned_animator):
			super().__init__(solver_ref, cfg_ref, out_dir)
			# self._pinned_animator = pinned_animator

		def _physics_step(self):
			# self._pinned_animator.update_pinned_vertices(self._solver, self._config.current_frame, self._config.implicit_dt)
			super()._physics_step()

	gui = AnimatedSimulationGUI(solver, config_ref, output_dir, None)
	gui.show()

solver.cleanup_device()