from utils.test_script_path import PROJECT_ROOT
import os, sys

import lcs_py as lcs

import utils.arg_parser
args = utils.arg_parser.parse_args()

# Initialize LuisaCompute device
backend = args.backend  # backends: cuda, dx, vk, metal (if supported on the platform)
solver = lcs.NewtonSolver()
solver.init_device(backend_name=backend)

# Load a mesh by providing the path to the obj file
from utils.animation_transform import DefaultTransformAnimation
from utils.vertex_animator import VertexAnimator
def load_cloth_with_vertex_animation():
	cloth_mesh_path = os.path.join(PROJECT_ROOT, 'Resources', 'InputMesh', 'Cylinder', 'cylinder7K.obj')
	cloth = solver.create_world_data_from_file_path('cylinder7K', cloth_mesh_path)
	cloth.set_simulation_type(lcs.MaterialType.Cloth)
	cloth.set_physics_material_cloth(thickness=0.001, youngs_modulus=1e6)
	cloth.set_scale(0.75)

	animator = VertexAnimator(cloth)
	animator.add_rule_by_method(
		"Left",
		DefaultTransformAnimation(
			use_rotate=True,
			rot_center=[0.0, 0.0, 0.005],
			rot_axis=[1.0, 0.0, 0.0],
			rot_ang_vel_deg=-72.0
		)
	)
	animator.add_rule_by_method(
		"Right",
		DefaultTransformAnimation(
			use_rotate=True,
			rot_center=[0.0, 0.0, -0.005],
			rot_axis=[1.0, 0.0, 0.0],
			rot_ang_vel_deg=72.0
		)
	)
	cloth_id = solver.register_world_data(cloth)
	animator.set_mesh_index(cloth_id)
	return animator

animator = load_cloth_with_vertex_animation()

# Initialize the solver (builds internal data structures, compiles shaders, etc.)
solver.init_solver()

# Set scene parameters
config_ref = solver.get_config()
config_ref.set_nonlinear_iter_count(1)
config_ref.set_pcg_iter_count(50)
config_ref.set_gravity(lcs.Float3(0.0, 0.0, 0.0))
config_ref.set_use_floor(False)
# config_ref.use_self_collision = False
# config_ref.contact_energy_type = 0

# Output directory (for optional file saving)
output_dir = os.path.join(PROJECT_ROOT, "Resources", "OutputMesh")
os.makedirs(output_dir, exist_ok=True)

# Launch polyscope GUI or run headless
def update_animation():
	animator.update_animation(solver, config_ref.current_frame, config_ref.implicit_dt)

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