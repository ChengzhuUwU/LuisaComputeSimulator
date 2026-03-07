import numpy as np
import os, sys
from dataclasses import dataclass

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(root, 'build', 'bin'))
import lcs_py as lcs

import utils.arg_parser
args = utils.arg_parser.parse_args()

from utils.animation_transform import FixedPointTransform
from utils.body_animator import BodyAnimator

# Initialize LuisaCompute device
backend = args.backend  # backends: cuda, dx, vk, metal (if supported on the platform)
solver = lcs.NewtonSolver()
solver.init_device(backend_name=backend)

# Load a mesh by providing the path to the obj file
import trimesh
cube_mesh_path = os.path.join(root, 'Resources', 'InputMesh', 'cube.obj')
cube_mesh = trimesh.load(cube_mesh_path, process=False)

def load_top_cube():
	cube_top = solver.register_mesh_from_array('cube1', cube_mesh.vertices, cube_mesh.faces)
	cube_top.set_simulation_type(lcs.MaterialType.Rigid)
	cube_top.set_scale(0.1)
	cube_top.set_translation(0.0, 0.14, 0.0)

def load_bottom_cube():
	cube_bottom = solver.register_mesh_from_array('cube2', cube_mesh.vertices, cube_mesh.faces)
	cube_bottom.set_simulation_type(lcs.MaterialType.Rigid)
	cube_bottom.set_scale(0.1)
	cube_bottom.set_translation(0.0, 0.01, 0.0)

	body_animator = BodyAnimator(
		world_data = cube_bottom, 
		initial_translation=cube_bottom.get_rest_translation(), 
		initial_rotation=cube_bottom.get_rest_rotation())
	body_animator.add_rule_by_method(
		"All",
		FixedPointTransform(
			use_translate=True,
			translate=[0.0, 0.02, 0.0],
			use_rotate=True,
			rot_axis=[1.0, 0.0, 0.0],
			rot_ang_vel_deg=45.0,
		),
	)
	return body_animator

load_top_cube()
body_animator = load_bottom_cube()




# Initialize the solver (builds internal data structures, compiles shaders, etc.)
solver.init_solver()

# Set scene parameters
config_ref = solver.get_config()

# Output directory (for optional file saving)
output_dir = os.path.join(root, "Resources", "OutputMesh")
os.makedirs(output_dir, exist_ok=True)

# Launch polyscope GUI or run headless
if args.headless:
	solver.save_sim_result(obj_path=os.path.join(output_dir, "init.obj"))
	for _ in range(0, args.advance_frames):
		body_animator.update_body_animation(solver, config_ref.current_frame, config_ref.implicit_dt)
		if config_ref.use_gpu:
			solver.physics_step_gpu()
		else:
			solver.physics_step_cpu()
	solver.save_sim_result(obj_path=os.path.join(output_dir, "result.obj"))
else:
	import utils.polyscope_gui

	class AnimatedSimulationGUI(utils.polyscope_gui.SimulationGUI):
		def __init__(self, solver_ref, cfg_ref, out_dir, body_animator_ref):
			super().__init__(solver_ref, cfg_ref, out_dir)
			self._body_animator = body_animator_ref

		def _physics_step(self):
			self._body_animator.update_body_animation(self._solver, self._config.current_frame, self._config.implicit_dt)
			super()._physics_step()

	gui = AnimatedSimulationGUI(solver, config_ref, output_dir, body_animator)
	gui.show()

solver.cleanup_device()