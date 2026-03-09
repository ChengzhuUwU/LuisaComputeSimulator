import inspect
import os
import pickle
import sys
import tempfile
import urllib.request

import numpy as np

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(root, 'build', 'bin'))
import lcs_py as lcs

def parse_args2():
	import argparse
	parser = argparse.ArgumentParser(description="LuisaCompute Python example")
	parser.add_argument(
		"--backend",
		type=str,
		default="metal",
		choices=["cuda", "dx", "vk", "metal"],
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
output_dir = os.path.join(root, "Resources", "OutputMesh")
os.makedirs(output_dir, exist_ok=True)


def _write_obj(file_path: str, vertices: np.ndarray, faces: np.ndarray):
	with open(file_path, "w", encoding="utf-8") as f:
		for v in vertices:
			f.write(f"v {float(v[0])} {float(v[1])} {float(v[2])}\n")
		for tri in faces:
			# OBJ uses 1-based indexing.
			f.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")


# Load a mesh by providing the path to the obj file
def load_garment():
	cloth_mesh_path = os.path.join(root, 'Resources', 'InputMesh', 'Cylinder', 'cylinder7K.obj')
	cloth = solver.create_world_data_from_file_path('cylinder7K', cloth_mesh_path)
	cloth.set_simulation_type(lcs.MaterialType.Cloth)
	cloth.set_physics_material_cloth(thickness=0.001, youngs_modulus=1e6)
	cloth.set_scale(1.0)
	cloth.set_translation(0.0, 1.0, 0.0)
	solver.register_world_data(cloth)


from utils.smpl_animator import SMPLSequenceAnimator, _maybe_download_smpl_model, _maybe_download_sequence_model
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
		smpl_model_path = os.path.join(root, "build", "models", "SMPL_FEMALE.pkl")
	if not sequence_path:
		sequence_path = os.path.join(root, "build", "models", "SEQUENCE.npz")
	if not os.path.isfile(smpl_model_path):
		_maybe_download_smpl_model(smpl_model_path)
	if not os.path.isfile(sequence_path):
		_maybe_download_sequence_model(sequence_path)
	if not os.path.isfile(smpl_model_path):
		raise FileNotFoundError(f"SMPL model not found after download attempt: {smpl_model_path}.")
	if not os.path.isfile(sequence_path):
		raise FileNotFoundError(f"SMPL sequence not found after download attemp: {sequence_path}.")

	smpl_model = smplx.SMPL(smpl_model_path)
	sequence_data = np.load(sequence_path, allow_pickle=True)
	
	# Convert AMASS format to SMPL format
	# AMASS: [global_orient (3), body_pose (63), hand_pose (90)] = 156 total
	# SMPL expects: [global_orient (3), body_pose (69 for 23 joints)] = 72 total
	full_poses = sequence_data["poses"].astype(np.float32)  # (N, 156)
	
	# Extract SMPL-compatible pose parameters (first 72 dimensions)
	# This includes global_orient (3) + body_pose (69 for 23 joints)
	smpl_poses = full_poses[:, :72]  
	
	global_orient = smpl_poses[:, :3]
	body_pose = smpl_poses[:, 3:72]
	
	# Body pose is a sequence of rotation vectors for each joint (69/3 = 23 joints)
	body_pose_reshaped = body_pose.reshape(-1, 23, 3)  # (N, 23, 3)
	body_pose = body_pose_reshaped.reshape(-1, 69)  # (N, 69)
	
	transl = sequence_data["trans"].astype(np.float32)  # (N, 3)
	
	# Ensure betas has the correct size (SMPL requires 10 params, AMASS may have 16)
	betas = sequence_data["betas"].astype(np.float32)  # (16,)
	if betas.shape[0] > 10:
		betas = betas[:10]  # Use only first 10 components for SMPL
	
	sequence = {
		"body_pose": body_pose,  # (N, 69)
		"global_orient": global_orient,  # (N, 3)
		"transl": transl,  # (N, 3)
		"betas": betas,  # (10,) for SMPL
	}
	animator = SMPLSequenceAnimator(smpl_model, sequence, loop=True, smooth_transition_frames=100)

	faces = np.asarray(smpl_model.faces, dtype=np.int32)
	verts = animator.get_rest_pose_vertices()  # (V, 3) in T-pose
	# verts = smpl_model.v_template.detach().cpu().numpy().astype(np.float32)

	obstacle = solver.create_world_data_from_array("smpl_body", verts, faces)
	obstacle.set_simulation_type(lcs.MaterialType.Cloth)
	obstacle.set_physics_material_cloth(stretch_model="Empty", bending_model="Empty")
	obstacle.add_fixed_point_by_method("All", range=0.001)
	obstacle_id = solver.register_world_data(obstacle)

	animator.set_mesh_index(obstacle_id)
	return animator

# load_garment()
animator = load_smpl()

animators = [animator]

# Initialize the solver (builds internal data structures, compiles shaders, etc.)
solver.init_solver()

# Set scene parameters
config_ref = solver.get_config()

# config_ref.use_floor = False
config_ref.floor = lcs.Float3(0.0, -1.6, 0.0) 
# config_ref.print_pcg_info = True
# config_ref.print_collision_info = True
# config_ref.nonlinear_iter_count = 1
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
		
		def restart_system(self):
			"""Override restart to reset animators."""
			# Reset all animators to start from frame 0 with smooth transition
			for animator in animators:
				if animator is not None and hasattr(animator, 'reset'):
					animator.reset()
			# Call parent's restart_system
			super().restart_system()

	gui = AnimatedSimulationGUI(solver, config_ref, output_dir)
	gui.show()

solver.cleanup_device()
