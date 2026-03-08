import numpy as np

from utils.animation_transform import DefaultTransformAnimation

class BodyAnimator:
	"""Manage fixed rigid body selection and Python-driven per-frame body updates."""

	def __init__(self, world_data):
		self._transform = None
		self._initial_translation = np.asarray(
			world_data.get_rest_translation(), 
			dtype=np.float32,
		)
		self._initial_rotation = np.asarray(
			world_data.get_rest_rotation(),
			dtype=np.float32,
		)

	def set_mesh_index(self, mesh_idx: int):
		self.body_id = int(mesh_idx)

	def add_rule_by_method(self, world_data, method: str, transform: DefaultTransformAnimation, range_value: float = 0.001):
		before_fixed_indices = np.asarray(world_data.get_fixed_point_indices(), dtype=np.uint32)
		world_data.add_fixed_point_by_method(method, range=range_value)
		after_fixed_indices = np.asarray(world_data.get_fixed_point_indices(), dtype=np.uint32)
		self._transform = transform
		added_fixed_indices = after_fixed_indices[before_fixed_indices.size :]
		return added_fixed_indices

	def update_animation(self, solver, curr_frame: int, dt: float):
		if self.body_id is None:
			raise RuntimeError("mesh_idx is not set.")
		if self._transform is None:
			return

		curr_time = float(curr_frame) * float(dt)
		transform = self._transform

		translation = self._initial_translation.copy()
		if transform.use_setting_position:
			translation = np.asarray(transform.setting_position, dtype=np.float32).copy()
			
		if transform.use_translate:
			translation = translation + np.asarray(transform.translate, dtype=np.float32) * np.float32(curr_time)

		rotation = self._initial_rotation.copy()
		if transform.use_rotate:
			axis = np.asarray(transform.rot_axis, dtype=np.float32)
			axis_norm = np.linalg.norm(axis)
			if axis_norm > 1e-8:
				axis = axis / axis_norm
				angle_rad = np.deg2rad(np.float32(curr_time) * np.float32(transform.rot_ang_vel_deg))
				# Feed incremental xyz-angle style rotation expected by solver update API.
				rotation = rotation + axis * angle_rad

		solver.update_per_body_animation(self.body_id, translation.astype(np.float32), rotation.astype(np.float32))
