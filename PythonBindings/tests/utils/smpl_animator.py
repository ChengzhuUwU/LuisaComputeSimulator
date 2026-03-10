import numpy as np
try:
	import smplx
except ImportError as exc:
	raise RuntimeError("SMPL animation requires smplx. Install with: pip install smplx") from exc

class SMPLSequenceAnimator:
	"""Evaluate SMPL every frame and push per-vertex animation targets to the solver."""

	def __init__(self, smpl_model: smplx.SMPL, sequence_data , loop: bool = True, smooth_transition_frames: int = 100):

		for key_val in sequence_data.items():
			print(f"  => Amass Sequence data key: {key_val[0]}, shape: {key_val[1].shape}, dtype: {key_val[1].dtype}")

		# Poses and Orientations
		full_poses = sequence_data["poses"].astype(np.float32)  # (N, 156)
		global_orient = full_poses[:, :3] # shape (N, 3) for root joint global orientation (rotation vector)
		body_pose = full_poses[:, 3:72] # shape (N, 69) for 23 joints * 3 values each (rotation vector)
		body_pose = body_pose.reshape(-1, 69)  # (N, 69)
		
		# Translation
		transl = sequence_data["trans"].astype(np.float32)  # (N, 3)
		
		# Shape coefficients (betas)
		betas = sequence_data["betas"].astype(np.float32)  # (16,)
		if betas.shape[0] > 10: # (SMPL requires 10 params, AMASS may have 16)
			betas = betas[:10]  # Use only first 10 components for SMPL
		
		sequence: dict = {
			"body_pose": body_pose,  # (N, 69)
			"global_orient": global_orient,  # (N, 3)
			"transl": transl,  # (N, 3)
			"betas": betas,  # (10,) for SMPL
		}
		
		# self.mesh_idx = int(mesh_idx)
		self.smpl_model = smpl_model
		self.loop = bool(loop)
		self.smooth_transition_frames = int(smooth_transition_frames)  # Frames to smooth from T-pose to first frame

		self.body_pose = self._as_frame_tensor(sequence["body_pose"])  # Expects (N, 69)
		self.global_orient = self._as_frame_tensor(sequence["global_orient"]) # Expects (N, 3)
		self.transl = self._as_frame_tensor(sequence["transl"]) # Expects (N, 3)
		self.betas = np.asarray(sequence["betas"], dtype=np.float32) # Expects (10,) or (N, 10)
		if self.betas.ndim == 1:
			self.betas = self.betas[None, :]

		self.total_frame = int(self.body_pose.shape[0])
		if self.total_frame <= 0:
			raise RuntimeError("SMPL sequence has no frame data.")
		
		# Cache the first frame pose for smooth transition
		self.first_body_pose = self.body_pose[0:1].copy()
		self.first_global_orient = self.global_orient[0:1].copy()
		self.first_transl = self.transl[0:1].copy()

		# T-pose parameters: keep first-frame root transform, but zero-out articulated body joints.
		init_verts = smpl_model.v_template.detach().cpu().numpy().astype(np.float32)
		# self.tpose_global_verts = smpl_model.v_template.detach().cpu().numpy().astype(np.float32)
		self.tpose_global_verts = self._run_smpl( 
			self.betas[:1], 
			np.zeros_like(self.first_body_pose), 
			self.first_global_orient, 
			self.first_transl)
		
		# Track the starting frame for smooth transition
		self.start_frame = 0

	def get_rest_pose_vertices(self) -> np.ndarray:
		"""Get the rest pose vertices (T-pose) from the SMPL model."""
		return (self.tpose_global_verts)
	
	def set_mesh_index(self, mesh_idx: int):
		self.mesh_idx = int(mesh_idx)

	def _run_smpl(self, betas: np.ndarray, body_pose: np.ndarray, global_orient: np.ndarray, transl: np.ndarray) -> np.ndarray:
		try:
			import torch
		except ImportError as exc:
			raise RuntimeError("SMPL animation requires torch. Install with: pip install torch") from exc

		with torch.no_grad():
			out = self.smpl_model(
				betas=torch.from_numpy(betas),
				body_pose=torch.from_numpy(body_pose),
				global_orient=torch.from_numpy(global_orient),
				transl=torch.from_numpy(transl),
			)
		verts = out.vertices[0].detach().cpu().numpy().astype(np.float32)
		return self._transform_AMASS_axis(verts)

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

	def _eval_vertices_in_frame(self, frame_idx: int, transition_factor: float = 1.0) -> np.ndarray:
		"""
		Evaluate SMPL vertices at the given frame with optional smooth transition.
		
		Args:
			frame_idx: The frame index in the sequence
			transition_factor: Blend factor between [0, 1]. 0 = T-pose, 1 = actual pose
		"""
		betas = self.betas[frame_idx : frame_idx + 1] if self.betas.shape[0] == self.total_frame else self.betas[:1]
		body_pose = self.body_pose[frame_idx : frame_idx + 1]
		global_orient = self.global_orient[frame_idx : frame_idx + 1]
		transl = self.transl[frame_idx : frame_idx + 1]

		actual_verts = self._run_smpl(betas, body_pose, global_orient, transl)
		if transition_factor >= 1.0:
			return actual_verts
		else:
			return self.tpose_global_verts * (1.0 - transition_factor) + actual_verts * transition_factor

	def _transform_AMASS_axis(self, verts):
		# AMASS with SMPL parameters outputs vertices in Z-up space.
		# Transform to Y-up: (x, y, z)_Zup -> (x, z, -y)_Yup
		return np.stack([verts[:, 0], verts[:, 2], -verts[:, 1]], axis=-1)

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
		target_vertices = self._eval_vertices_in_frame(frame_idx, transition_factor)
		for local_vid, target_pos in enumerate(target_vertices):
			solver.update_per_vertex_animation(self.mesh_idx, int(local_vid), target_pos)



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
		f"Download from Hugging Face now? (About 200 MB) \\n{smpl_url}"
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
		f"Download from Hugging Face now? (About 5 MB) \\n{sequence_url}"
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

