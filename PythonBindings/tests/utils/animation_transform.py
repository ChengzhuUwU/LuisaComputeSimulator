import numpy as np
import os, sys
from dataclasses import dataclass

@dataclass
class DefaultTransformAnimation:
	"""Python-side counterpart of C++ FixedPointAnimationInfo."""

	use_translate: bool = False
	translate: np.ndarray = None

	use_scale: bool = False
	scale: np.ndarray = None

	use_rotate: bool = False
	rot_center: np.ndarray = None
	rot_axis: np.ndarray = None
	rot_ang_vel_deg: float = 0.0

	use_setting_position: bool = False
	setting_position: np.ndarray = None

	def __post_init__(self):
		self.translate = np.array([0.0, 0.0, 0.0], dtype=np.float32) if self.translate is None else np.asarray(self.translate, dtype=np.float32)
		self.scale = np.array([1.0, 1.0, 1.0], dtype=np.float32) if self.scale is None else np.asarray(self.scale, dtype=np.float32)
		self.rot_center = np.array([0.0, 0.0, 0.0], dtype=np.float32) if self.rot_center is None else np.asarray(self.rot_center, dtype=np.float32)
		self.rot_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32) if self.rot_axis is None else np.asarray(self.rot_axis, dtype=np.float32)
		self.setting_position = np.array([0.0, 0.0, 0.0], dtype=np.float32) if self.setting_position is None else np.asarray(self.setting_position, dtype=np.float32)

	@staticmethod
	def _rotate_with_axis_angle(pos: np.ndarray, center: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
		axis_norm = np.linalg.norm(axis)
		if axis_norm < 1e-8:
			return pos
		k = axis / axis_norm
		v = pos - center
		cos_t = np.cos(angle_rad)
		sin_t = np.sin(angle_rad)
		rotated = v * cos_t + np.cross(k, v) * sin_t + k * np.dot(k, v) * (1.0 - cos_t)
		return center + rotated

	def apply(self, time: float, rest_pos: np.ndarray) -> np.ndarray:
		"""
		Match C++ order in FixedPointAnimationInfo::fn_affine_position:
		setting_position -> scale -> rotate -> translate.
		"""
		new_pos = np.asarray(rest_pos, dtype=np.float32)
		if self.use_setting_position:
			new_pos = self.setting_position.copy()
		if self.use_scale:
			new_pos = (self.scale * np.float32(time)) * new_pos
		if self.use_rotate:
			angle_rad = np.deg2rad(np.float32(time) * np.float32(self.rot_ang_vel_deg))
			new_pos = self._rotate_with_axis_angle(new_pos, self.rot_center, self.rot_axis, angle_rad)
		if self.use_translate:
			new_pos = new_pos + self.translate * np.float32(time)
		return new_pos.astype(np.float32)


