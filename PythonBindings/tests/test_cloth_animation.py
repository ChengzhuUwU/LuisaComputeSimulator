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
cloth_mesh_path = os.path.join(root, 'Resources', 'InputMesh', 'Cylinder', 'cylinder7K.obj')
cloth = solver.register_mesh_from_file_path('cylinder7K', cloth_mesh_path)
cloth.set_simulation_type(lcs.MaterialType.Cloth)

from utils.animation_transform import FixedPointTransform
from utils.vertex_animator import VertexAnimator
animator = VertexAnimator(cloth)
animator.add_rule_by_method(
    "Left",
    FixedPointTransform(
        use_rotate=True,
        rot_center=[0.0, 0.0, 0.005],
        rot_axis=[1.0, 0.0, 0.0],
        rot_ang_vel_deg=-72.0
    )
)
animator.add_rule_by_method(
    "Right",
    FixedPointTransform(
        use_rotate=True,
        rot_center=[0.0, 0.0, -0.005],
        rot_axis=[1.0, 0.0, 0.0],
        rot_ang_vel_deg=72.0
    )
)

# Initialize the solver (builds internal data structures, compiles shaders, etc.)
solver.init_solver()

# Set scene parameters
config_ref = solver.get_config()
config_ref.nonlinear_iter_count = 1
config_ref.pcg_iter_count = 50
config_ref.gravity = lcs.Float3(0.0, 0.0, 0.0)
config_ref.use_floor = False
# config_ref.use_self_collision = False
# config_ref.contact_energy_type = 0

# Output directory (for optional file saving)
output_dir = os.path.join(root, "Resources", "OutputMesh")
os.makedirs(output_dir, exist_ok=True)

# Launch polyscope GUI or run headless
if args.headless:
	solver.save_sim_result(obj_path=os.path.join(output_dir, "init.obj"))
	for _ in range(0, args.advance_frames):
		animator.update_vertex_animation(solver, config_ref.current_frame, config_ref.implicit_dt)
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
			self._animator = pinned_animator

		def _physics_step(self):
			self._animator.update_vertex_animation(self._solver, self._config.current_frame, self._config.implicit_dt)
			super()._physics_step()

	gui = AnimatedSimulationGUI(solver, config_ref, output_dir, animator)
	gui.show()

solver.cleanup_device()