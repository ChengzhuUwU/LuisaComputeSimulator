from utils.test_script_path import PROJECT_ROOT
import os, sys

import lcs_py as lcs

import utils.arg_parser
args = utils.arg_parser.parse_args()

from utils.animation_transform import DefaultTransformAnimation
from utils.body_animator import BodyAnimator

# Initialize LuisaCompute device
backend = args.backend  # backends: cuda, dx, vk, metal (if supported on the platform)
solver = lcs.NewtonSolver()
solver.init_device(backend_name=backend)

# Load a mesh by providing the path to the obj file
import trimesh
cube_mesh_path = os.path.join(PROJECT_ROOT, 'Resources', 'InputMesh', 'cube.obj')
cube_mesh = trimesh.load(cube_mesh_path, process=False)

def load_top_cube():
	cube_top = solver.create_world_data_from_array('cube1', cube_mesh.vertices, cube_mesh.faces)
	cube_top.set_simulation_type(lcs.MaterialType.Rigid)
	cube_top.set_scale(0.1)
	cube_top.set_translation(0.0, 0.34, 0.0)

	body_animator = None
	# body_animator = BodyAnimator(cube_top)
	# body_animator.add_rule_by_method(
	# 	cube_top,
	# 	"All",
	# 	DefaultTransformAnimation(
	# 		use_rotate=True,
	# 		rot_axis=[1.0, 0.0, 0.0],
	# 		rot_ang_vel_deg=-45.0,
	# 	),
	# )
	top_cube_id = solver.register_world_data(cube_top)
	# body_animator.set_mesh_index(top_cube_id)
	return body_animator

def load_bottom_cube():
	cube_bottom = solver.create_world_data_from_array('cube2', cube_mesh.vertices, cube_mesh.faces)
	cube_bottom.set_simulation_type(lcs.MaterialType.Rigid)
	cube_bottom.set_scale(0.1)
	cube_bottom.set_translation(0.0, 0.01, 0.0)

	body_animator = None
	body_animator = BodyAnimator(cube_bottom)
	body_animator.add_rule_by_method(
		cube_bottom,
		"All",
		DefaultTransformAnimation(
			use_rotate=True,
			rot_axis=[1.0, 0.0, 0.0],
			rot_ang_vel_deg=45.0,
		),
	)
	bottom_cube_id = solver.register_world_data(cube_bottom)
	body_animator.set_mesh_index(bottom_cube_id)
	
	return body_animator

# if args.order == 0:
# 	load_top_cube()
# 	buttom_cube_animator = load_bottom_cube()
# else:
top_cube_animator = load_top_cube()
buttom_cube_animator = load_bottom_cube()
animators = [buttom_cube_animator, top_cube_animator]

# Initialize the solver (builds internal data structures, compiles shaders, etc.)
solver.init_solver()

# Set scene parameters
config_ref = solver.get_config()

# config_ref.use_floor = False
# config_ref.nonlinear_iter_count = 1
# config_ref.use_self_collision = False

# Output directory (for optional file saving)
output_dir = os.path.join(PROJECT_ROOT, "Resources", "OutputMesh")
os.makedirs(output_dir, exist_ok=True)

def update_animation():
	for animator in animators:
		if animator is not None:
			animator.update_animation(solver, config_ref.current_frame, config_ref.implicit_dt)

# Launch polyscope GUI or run headless
if args.headless:
	solver.save_sim_result(obj_path=os.path.join(output_dir, "init.obj"))
	for _ in range(0, args.advance_frames):
		update_animation()
		if config_ref.use_gpu:
			solver.physics_step_gpu()
		else:
			solver.physics_step_cpu()
	solver.save_sim_result(obj_path=os.path.join(output_dir, "result.obj"))
else:
	import utils.polyscope_gui

	class AnimatedSimulationGUI(utils.polyscope_gui.SimulationGUI):
		def _physics_step(self):
			update_animation()
			super()._physics_step()

	gui = AnimatedSimulationGUI(solver, config_ref, output_dir)
	gui.show()

solver.cleanup_device()