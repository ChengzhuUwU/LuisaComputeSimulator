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


def _ask_yes_no_dialog(title: str, message: str) -> bool:
	"""Ask user with a GUI dialog when possible, then fallback to terminal input."""
	try:
		import tkinter as tk
		from tkinter import messagebox

		root_tk = tk.Tk()
		root_tk.withdraw()
		result = bool(messagebox.askyesno(title, message))
		root_tk.destroy()
		return result
	except Exception:
		answer = input(f"{message} [y/N]: ").strip().lower()
		return answer in {"y", "yes"}

def _maybe_download_smpl_model(smpl_model_path: str) -> None:
	smpl_url = "https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPL_FEMALE.pkl"
	message = (
		f"SMPL model not found:\\n{smpl_model_path}\\n\\n"
		f"Download from Hugging Face now?\\n{smpl_url}"
	)
	if not _ask_yes_no_dialog("SMPL Model Missing", message):
		raise FileNotFoundError(f"SMPL model not found: {smpl_model_path}")

	os.makedirs(os.path.dirname(os.path.abspath(smpl_model_path)), exist_ok=True)
	try:
		print(f"Downloading SMPL model to: {smpl_model_path}")
		_download_with_progress(smpl_url, smpl_model_path)
	except Exception as exc:
		raise RuntimeError(f"Failed to download SMPL model from {smpl_url}") from exc

def _maybe_download_sequence_model(sequence_path: str) -> None:
	sequence_url = "https://huggingface.co/datasets/realdream-ai/AMASS/resolve/main/raw/CMU/01/01_01_poses.npz"
	message = (
		f"Sequence not found:\\n{sequence_path}\\n\\n"
		f"Download from Hugging Face now?\\n{sequence_url}"
	)
	if not _ask_yes_no_dialog("Sequence Missing", message):
		raise FileNotFoundError(f"Sequence not found: {sequence_path}")

	os.makedirs(os.path.dirname(os.path.abspath(sequence_path)), exist_ok=True)
	try:
		print(f"Downloading sequence to: {sequence_path}")
		_download_with_progress(sequence_url, sequence_path)
	except Exception as exc:
		raise RuntimeError(f"Failed to download sequence from {sequence_url}") from exc

def _download_with_progress(url: str, file_path: str) -> None:
	"""Download file with progress bar."""
	try:
		from tqdm import tqdm
	except ImportError:
		urllib.request.urlretrieve(url, file_path)
		return

	def reporthook(blocknum, blocksize, totalsize):
		if totalsize <= 0:
			return
		downloaded = blocknum * blocksize
		percent = min(downloaded * 100 // totalsize, 100)
		sys.stdout.write(f"\rProgress: {percent}%")
		sys.stdout.flush()

	urllib.request.urlretrieve(url, file_path, reporthook=reporthook)
	print()



class SMPLSequenceAnimator:
	"""Evaluate SMPL every frame and push per-vertex animation targets to the solver."""

	def __init__(self, mesh_idx: int, smpl_model, sequence: dict, loop: bool = True, smooth_transition_frames: int = 100):
		self.mesh_idx = int(mesh_idx)
		self.smpl_model = smpl_model
		self.loop = bool(loop)
		self.smooth_transition_frames = int(smooth_transition_frames)  # Frames to smooth from T-pose to first frame

		self.body_pose = self._as_frame_tensor(sequence["body_pose"])
		self.global_orient = self._as_frame_tensor(sequence["global_orient"])
		self.transl = self._as_frame_tensor(sequence["transl"])
		self.betas = np.asarray(sequence["betas"], dtype=np.float32)
		if self.betas.ndim == 1:
			self.betas = self.betas[None, :]

		self.total_frame = int(self.body_pose.shape[0])
		if self.total_frame <= 0:
			raise RuntimeError("SMPL sequence has no frame data.")
		
		# Cache the first frame pose for smooth transition
		self.first_body_pose = self.body_pose[0:1].copy()
		self.first_global_orient = self.global_orient[0:1].copy()
		self.first_transl = self.transl[0:1].copy()
		
		# Zero pose for T-pose (all zeros)
		self.zero_body_pose = np.zeros_like(self.first_body_pose)
		self.zero_global_orient = np.zeros_like(self.first_global_orient)
		self.zero_transl = np.zeros_like(self.first_transl)
		
		# Track the starting frame for smooth transition
		self.start_frame = 0

	@staticmethod
	def _as_frame_tensor(data) -> np.ndarray:
		arr = np.asarray(data, dtype=np.float32)
		if arr.ndim == 1:
			arr = arr[None, :]
		return arr

	def _pick_frame(self, curr_frame: int) -> int:
		if self.loop:
			return int(curr_frame) % self.total_frame
		return min(int(curr_frame), self.total_frame - 1)

	def _eval_smpl_vertices(self, frame_idx: int, transition_factor: float = 1.0) -> np.ndarray:
		"""
		Evaluate SMPL vertices at the given frame with optional smooth transition.
		
		Args:
			frame_idx: The frame index in the sequence
			transition_factor: Blend factor between [0, 1]. 0 = T-pose, 1 = actual pose
		"""
		try:
			import torch
		except ImportError as exc:
			raise RuntimeError("SMPL animation requires torch. Install with: pip install torch") from exc

		betas = self.betas[frame_idx : frame_idx + 1] if self.betas.shape[0] == self.total_frame else self.betas[:1]
		
		# Blend between zero pose and actual pose for smooth transition
		if transition_factor < 1.0:
			body_pose = self.zero_body_pose * (1 - transition_factor) + self.body_pose[frame_idx : frame_idx + 1] * transition_factor
			global_orient = self.zero_global_orient * (1 - transition_factor) + self.global_orient[frame_idx : frame_idx + 1] * transition_factor
			transl = self.zero_transl * (1 - transition_factor) + self.transl[frame_idx : frame_idx + 1] * transition_factor
		else:
			body_pose = self.body_pose[frame_idx : frame_idx + 1]
			global_orient = self.global_orient[frame_idx : frame_idx + 1]
			transl = self.transl[frame_idx : frame_idx + 1]
		
		with torch.no_grad():
			out = self.smpl_model(
				betas=torch.from_numpy(betas),
				body_pose=torch.from_numpy(body_pose),
				global_orient=torch.from_numpy(global_orient),
				transl=torch.from_numpy(transl),
			)
		verts = out.vertices[0].detach().cpu().numpy().astype(np.float32)
		# SMPL with AMASS parameters outputs vertices in Z-up space.
		# Transform to Y-up: (x, y, z)_Zup -> (x, z, -y)_Yup
		verts = np.stack([verts[:, 0], verts[:, 2], -verts[:, 1]], axis=-1)
		return verts

	def reset(self):
		"""Reset the animator to start from frame 0 with smooth transition."""
		self.start_frame = 0

	def update_animation(self, solver, curr_frame: int, dt: float):
		"""
		Update animation for current frame.
		
		Args:
			solver: The physics solver
			curr_frame: Current frame number (from the simulation)
			dt: Time step
		"""
		# Calculate frames since animation started
		frames_elapsed = curr_frame - self.start_frame
		
		# Calculate smooth transition factor (0 to 1 over smooth_transition_frames)
		if frames_elapsed < self.smooth_transition_frames:
			transition_factor = float(frames_elapsed) / float(self.smooth_transition_frames)
		else:
			transition_factor = 1.0
		
		# Get the appropriate frame from the sequence
		frame_idx = self._pick_frame(curr_frame)
		target_vertices = self._eval_smpl_vertices(frame_idx, transition_factor)
		for local_vid, target_pos in enumerate(target_vertices):
			solver.update_per_vertex_animation(self.mesh_idx, int(local_vid), target_pos)

# Load a mesh by providing the path to the obj file
def load_garment():
	cloth_mesh_path = os.path.join(root, 'Resources', 'InputMesh', 'Cylinder', 'cylinder7K.obj')
	cloth = solver.create_world_data_from_file_path('cylinder7K', cloth_mesh_path)
	cloth.set_simulation_type(lcs.MaterialType.Cloth)
	cloth.set_physics_material_cloth(thickness=0.001, youngs_modulus=1e6)
	cloth.set_scale(1.0)
	cloth.set_translation(0.0, 1.0, 0.0)
	solver.register_world_data(cloth)

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
	faces = np.asarray(smpl_model.faces, dtype=np.int32)
	verts = smpl_model.v_template.detach().cpu().numpy().astype(np.float32)

	generated_obj = os.path.join(output_dir, "generated_smpl.obj")
	_write_obj(generated_obj, verts, faces)

	obstacle = solver.create_world_data_from_file_path("smpl_body", generated_obj)
	obstacle.set_simulation_type(lcs.MaterialType.Cloth)
	obstacle.set_physics_material_cloth(stretch_model="Empty", bending_model="Empty")
	obstacle.add_fixed_point_by_method("All", range=0.001)

	obstacle_id = solver.register_world_data(obstacle)

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

	animator = SMPLSequenceAnimator(obstacle_id, smpl_model, sequence, loop=False, smooth_transition_frames=0)
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
