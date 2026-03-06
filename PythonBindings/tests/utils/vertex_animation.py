import numpy as np
import os, sys
from utils.animation_transform import FixedPointTransform

class PinnedVertexAnimator:
	"""Manage fixed-point selection and Python-driven per-frame pinned vertex updates."""

	def __init__(self, world_data):
		self.world_data = world_data
		self.mesh_idx = None
		self._vertex_transform_map = {}
		self._rest_positions = None

	def set_mesh_index(self, mesh_idx: int):
		self.mesh_idx = int(mesh_idx)

	def add_rule_by_method(self, method: str, transform: FixedPointTransform, range_value: float = 0.001):
		before = np.asarray(self.world_data.get_fixed_point_indices(), dtype=np.uint32)
		self.world_data.add_fixed_point_by_method(method, range=range_value)
		after = np.asarray(self.world_data.get_fixed_point_indices(), dtype=np.uint32)
		new_ids = after[before.size :]
		for vid in new_ids.tolist():
			# If rules overlap, latest rule wins for that vertex.
			self._vertex_transform_map[int(vid)] = transform
		return new_ids

	def capture_rest_positions(self):
		self._rest_positions = np.asarray(self.world_data.get_rest_positions(), dtype=np.float32)

	def update_pinned_vertices(self, solver, curr_frame: int, dt: float):
		if self.mesh_idx is None:
			raise RuntimeError("mesh_idx is not set. Call set_mesh_index() first.")
		if self._rest_positions is None:
			raise RuntimeError("rest positions are not cached. Call capture_rest_positions() first.")

		curr_time = float(curr_frame) * float(dt)
		for local_vid, transform in self._vertex_transform_map.items():
			rest_pos = self._rest_positions[local_vid]
			target_pos = transform.apply(curr_time, rest_pos)
			solver.update_pinned_verts_position(self.mesh_idx, local_vid, target_pos)