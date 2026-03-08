import os, sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(root, 'build', 'bin'))
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
from utils.animation_transform import DefaultTransformAnimation
from utils.vertex_animator import VertexAnimator
def load_garment():
	cloth_mesh_path = os.path.join(root, 'Resources', 'InputMesh', 'Cylinder', 'cylinder7K.obj')
	cloth = solver.create_world_data_from_file_path('cylinder7K', cloth_mesh_path)
	cloth.set_simulation_type(lcs.MaterialType.Cloth)
	cloth.set_physics_material_cloth(thickness=0.001, youngs_modulus=1e6, stretch_model="Empty", bending_model="Empty")
	cloth.set_scale(1.0)
	cloth_id = solver.register_world_data(cloth)

def load_smpl():
	cloth_mesh_path = os.path.join(root, 'Resources', 'InputMesh', 'Cylinder', 'cylinder7K.obj')
	cloth = solver.create_world_data_from_file_path('cylinder7K', cloth_mesh_path)
	cloth.set_simulation_type(lcs.MaterialType.Cloth)
	cloth.set_physics_material_cloth(thickness=0.001, youngs_modulus=1e6)
	cloth_id = solver.register_world_data(cloth)

load_garment()

animators = []

# Initialize the solver (builds internal data structures, compiles shaders, etc.)
solver.init_solver()

# Set scene parameters
config_ref = solver.get_config()

# config_ref.use_floor = False
# config_ref.nonlinear_iter_count = 1
# config_ref.use_self_collision = False

# Output directory (for optional file saving)
output_dir = os.path.join(root, "Resources", "OutputMesh")
os.makedirs(output_dir, exist_ok=True)

def update_animation():
	for animator in animators:
		if animator is not None:
			animator.update_body_animation(solver, config_ref.current_frame, config_ref.implicit_dt)

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