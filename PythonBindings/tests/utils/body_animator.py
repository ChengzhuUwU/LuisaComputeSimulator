import numpy as np

from utils.animation_transform import FixedPointTransform


class PinnedBodyAnimator:
	"""Manage fixed rigid body selection and Python-driven per-frame body updates."""

	def __init__(self, world_data, initial_translation=None, initial_rotation=None):
		self.world_data = world_data
		self.mesh_idx = world_data.get_registration_index()
		self._transform = None
		self._initial_translation = np.asarray(
			[0.0, 0.0, 0.0] if initial_translation is None else initial_translation,
			dtype=np.float32,
		)
		self._initial_rotation = np.asarray(
			[0.0, 0.0, 0.0] if initial_rotation is None else initial_rotation,
			dtype=np.float32,
		)

	def add_rule_by_method(self, method: str, transform: FixedPointTransform, range_value: float = 0.001):
		before = np.asarray(self.world_data.get_fixed_point_indices(), dtype=np.uint32)
		self.world_data.add_fixed_point_by_method(method, range=range_value)
		after = np.asarray(self.world_data.get_fixed_point_indices(), dtype=np.uint32)
		self._transform = transform
		return after[before.size :]

	def update_pinned_body(self, solver, curr_frame: int, dt: float):
		if self.mesh_idx is None:
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

		solver.update_per_body_animation(self.mesh_idx, translation.astype(np.float32), rotation.astype(np.float32))
