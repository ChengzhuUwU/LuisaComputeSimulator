from utils.test_script_path import PROJECT_ROOT
import inspect
import os
import pickle
import tempfile
import urllib.request

import numpy as np

import lcs_py as lcs

def parse_args2():
	import argparse
	parser = argparse.ArgumentParser(description="LuisaCompute Python example")
	parser.add_argument(
		"--backend",
		type=str,
		default="metal",
		choices=["cuda", "dx", "metal", "vk", "fallback", "cpu", "remote"],
		help="Compute backend to use (default: metal)",
	)
	parser.add_argument(
		"--headless",
		action="store_true",
		help="Run without GUI",
	)
	parser.add_argument(
		"--advance_frames",
		type=int,
		default=30,
		help="Number of simulation frames to advance in headless mode (default: 30)",
	)
	parser.add_argument(
		"--smpl_model_path",
		type=str,
		help="Path to the SMPL model file (pickle format)",
	)
	parser.add_argument(
		"--sequence_path",
		type=str,
		help="Path to the SMPL sequence file (pickle format)",
	)
	return parser.parse_args()
args = parse_args2()

from utils.vertex_animator import VertexAnimator
from utils.body_animator import BodyAnimator

# Initialize LuisaCompute device
backend = args.backend  # backends: cuda, dx, vk, metal (if supported on the platform)
solver = lcs.NewtonSolver()
solver.init_device(backend_name=backend)

# Output directory (for optional file saving)
output_dir = os.path.join(PROJECT_ROOT, "Resources", "OutputMesh")
os.makedirs(output_dir, exist_ok=True)


# Load a mesh by providing the path to the obj file
def load_garment():
	tshirt_mesh_path = os.path.join(PROJECT_ROOT, 'Resources', 'InputMesh', 'SMPL', 'tshirt.obj')
	tshirt = solver.create_world_data_from_file_path('tshirt', tshirt_mesh_path)
	tshirt.set_simulation_type(lcs.MaterialType.Cloth)
	tshirt.set_physics_material_cloth(stretch_model="Spring")
	solver.register_world_data(tshirt)

	pants_mesh_path = os.path.join(PROJECT_ROOT, 'Resources', 'InputMesh', 'SMPL', 'pants.obj')
	pants = solver.create_world_data_from_file_path('pants', pants_mesh_path)
	pants.set_simulation_type(lcs.MaterialType.Cloth)
	pants.set_physics_material_cloth(stretch_model="Spring")
	solver.register_world_data(pants)

	# smpl2_mesh_path = os.path.join(PROJECT_ROOT, 'Resources', 'InputMesh', 'SMPL', 'smpl.obj')
	# smpl2 = solver.create_world_data_from_file_path('smpl', smpl2_mesh_path)
	# smpl2.set_simulation_type(lcs.MaterialType.Cloth)
	# solver.register_world_data(smpl2)


from utils.smpl_animator import SMPLSequenceAnimator, _maybe_download_smpl_model, _maybe_download_sequence_model
from utils.mesh_proc import write_obj
def load_smpl():
	if not hasattr(inspect, "getargspec"):
		inspect.getargspec = inspect.getfullargspec
	try:
		import smplx
	except ImportError as exc:
		raise RuntimeError("SMPL animation requires smplx. Install with: pip install smplx") from exc

	smpl_model_path = args.smpl_model_path
	sequence_path = args.sequence_path
	if not smpl_model_path:
		smpl_model_path = os.path.join(PROJECT_ROOT, "build", "models", "SMPL_FEMALE.pkl")
	if not sequence_path:
		sequence_path = os.path.join(PROJECT_ROOT, "build", "models", "SEQUENCE.npz")
	if not os.path.isfile(smpl_model_path):
		_maybe_download_smpl_model(smpl_model_path)
	if not os.path.isfile(sequence_path):
		_maybe_download_sequence_model(sequence_path)

	smpl_model = smplx.SMPL(smpl_model_path)
	sequence_data = np.load(sequence_path, allow_pickle=True)
	
	animator = SMPLSequenceAnimator(smpl_model, sequence_data, loop=True, smooth_start_frame=15, smooth_transition_frames=100)

	smpl_faces = np.asarray(smpl_model.faces, dtype=np.int32)
	# smpl_verts = smpl_model.v_template.detach().cpu().numpy().astype(np.float32)
	smpl_verts = animator.get_rest_pose_vertices() 
	write_obj(os.path.join(output_dir, "smpl_rest_pose.obj"), smpl_verts, smpl_faces)

	obstacle = solver.create_world_data_from_array("smpl_body", smpl_verts, smpl_faces)
	obstacle.set_simulation_type(lcs.MaterialType.Cloth)
	obstacle.set_physics_material_cloth(stretch_model="Empty", bending_model="Empty")
	obstacle.add_fixed_point_by_method("All")
	obstacle_id = solver.register_world_data(obstacle)

	animator.set_mesh_index(obstacle_id)
	return animator

load_garment()
animator = load_smpl()

animators = [animator]

# Initialize the solver (builds internal data structures, compiles shaders, etc.)
solver.init_solver()

# Set scene parameters
config_ref = solver.get_config()

# config_ref.use_floor = False
config_ref.set_floor(lcs.Float3(0.0, 0.0, 0.0)) 
# config_ref.contact_energy_type = 0 # 0: quadratic, 1: barrier
# config_ref.print_pcg_info = True
# config_ref.print_collision_info = True
config_ref.set_nonlinear_iter_count(0)
# config_ref.use_self_collision = False

def update_animation():
	for animator in animators:
		if animator is not None:
			if isinstance(animator, VertexAnimator):
				animator.update_animation(solver, config_ref.current_frame, config_ref.implicit_dt)
			elif isinstance(animator, BodyAnimator):
				animator.update_animation(solver, config_ref.current_frame, config_ref.implicit_dt)
			elif isinstance(animator, SMPLSequenceAnimator):
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
