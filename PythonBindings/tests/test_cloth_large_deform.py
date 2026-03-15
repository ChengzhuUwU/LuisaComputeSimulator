"""
Compare cloth stretch models (FEM_BW98 vs Spring) under symmetric pulling.

This test builds a minimal cloth mesh with 4 vertices and 2 triangles, creates two
cloth objects with different stretch models, and pulls two fixed points to opposite
sides frame-by-frame via update_per_vertex_animation.
"""

import argparse
import os
import sys
from typing import Dict

import numpy as np

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(root, "build", "bin"))
import lcs_py as lcs


def parse_args():
    parser = argparse.ArgumentParser(description="Cloth stretch model comparison test")
    parser.add_argument("--backend", type=str, default="metal", choices=["metal", "cuda", "dx", "vk"])
    parser.add_argument("--advance_frames", type=int, default=40)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()

def get_fixed_indices():
    return np.array([0, 3], dtype=np.int32)

def make_simple_cloth_mesh():
    """Return a 4-vertex cloth quad split into 2 triangles."""
    vertices = np.array(
        [
            [-0.5, 0.0, 0.0],  # 0: bottom-left (fixed)
            [0.5, 0.0, 0.0],   # 1: bottom-right (fixed)
            [-0.5, 1.0, 0.0],  # 2: top-left (free)
            [0.5, 1.0, 0.0],   # 3: top-right (free)
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [0, 1, 2],
            [2, 1, 3],
        ],
        dtype=np.int32,
    )
    return vertices, triangles


def apply_fixed_point_stretch(
    solver,
    mesh_idx: int,
    rest_positions: np.ndarray,
    curr_frame: int,
    dt: float,
    fixed_dirs: Dict[int, np.ndarray],
    pull_speed: float,
):
    """Apply linear-in-time target positions to fixed points, same order as VertexAnimator.update_animation."""
    curr_time = float(curr_frame) * float(dt)
    for local_vid, direction in fixed_dirs.items():
        rest_pos = rest_positions[local_vid]
        target_pos = rest_pos + np.asarray(direction, dtype=np.float32) * np.float32(pull_speed * curr_time)
        solver.update_per_vertex_animation(mesh_idx, int(local_vid), target_pos)


def register_cloth_object(solver, name: str, stretch_model: str, z_offset: float):
    vertices, triangles = make_simple_cloth_mesh()

    # Prefer explicit world-data mesh loading API requested by this test.
    if hasattr(lcs, "WorldData"):
        try:
            cloth = lcs.WorldData()
            cloth.set_name(name)
            cloth.load_mesh_from_array(vertices, triangles)
        except TypeError:
            cloth = solver.create_world_data_from_array(name, vertices, triangles)
    else:
        cloth = solver.create_world_data_from_array(name, vertices, triangles)

    if hasattr(cloth, "set_simulation_type"):
        cloth.set_simulation_type(lcs.MaterialType.Cloth)
    else:
        cloth.set_material_type(lcs.MaterialType.Cloth)
    cloth.set_physics_material_cloth(
        stretch_model=stretch_model,
        bending_model="Empty",
        thickness=0.001,
        youngs_modulus=1e5,
        poisson_ratio=0.3,
    )
    cloth.add_fixed_point_by_indices(np.array(get_fixed_indices(), dtype=np.int32))
    cloth.set_translation(0.0, 0.0, z_offset)

    mesh_idx = solver.register_world_data(cloth)
    rest_positions = np.asarray(cloth.get_rest_positions(), dtype=np.float32)
    return mesh_idx, rest_positions


def test_cloth_stretching_models(backend: str = "metal", advance_frames: int = 40, headless: bool = False):
    solver = lcs.NewtonSolver()
    solver.init_device(backend_name=backend)

    fem_id, fem_rest = register_cloth_object(solver, "cloth_fem_bw98", "FEM_BW98", z_offset=-0.25)
    spring_id, spring_rest = register_cloth_object(solver, "cloth_spring", "Spring", z_offset=0.25)

    config = solver.get_config()
    config.use_floor = False
    config.use_self_collision = False
    config.nonlinear_iter_count = 20
    config.pcg_iter_count = 3
    config.use_ccd_linesearch = False
    config.use_gpu = False
    config.gravity = lcs.Float3(0.0, 0.0, 0.0)
    config.implicit_dt = 0.01

    output_dir = os.path.join(root, "Resources", "OutputMesh")
    os.makedirs(output_dir, exist_ok=True)

    solver.init_solver()
    print("Running 2-model cloth stretching test on 4-vertex mesh...")

    # Pull the two pinned points apart along +/-X.
    # fixed_dirs = {
    #     0: np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    #     2: np.array([1.0, 0.0, 0.0], dtype=np.float32),
    # }

    # _vertices, _ = make_simple_cloth_mesh()
    fixed_dirs = dict()
    for local_vid in get_fixed_indices():
        if local_vid == 0:
            fixed_dirs[local_vid] = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        elif local_vid == 3:
            fixed_dirs[local_vid] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    pull_speed = 0.2

    if headless:
        for frame in range(advance_frames):
            apply_fixed_point_stretch(solver, fem_id, fem_rest, frame, config.implicit_dt, fixed_dirs, pull_speed)
            apply_fixed_point_stretch(solver, spring_id, spring_rest, frame, config.implicit_dt, fixed_dirs, pull_speed)
            solver.physics_step_gpu()
        solver.save_sim_result(obj_path=os.path.join(output_dir, "result.obj"))
    else:
        import utils.polyscope_gui
        class AnimatedSimulationGUI(utils.polyscope_gui.SimulationGUI):
            def _physics_step(self):
                frame = config.current_frame
                apply_fixed_point_stretch(solver, fem_id, fem_rest, frame, config.implicit_dt, fixed_dirs, pull_speed)
                apply_fixed_point_stretch(solver, spring_id, spring_rest, frame, config.implicit_dt, fixed_dirs, pull_speed)
                super()._physics_step()
        gui = AnimatedSimulationGUI(solver, config, output_dir)
        gui.show()

    fem_verts, _ = solver.get_object_sim_result_by_registration_id(fem_id)
    spring_verts, _ = solver.get_object_sim_result_by_registration_id(spring_id)

    fem_verts = np.asarray(fem_verts, dtype=np.float32)
    spring_verts = np.asarray(spring_verts, dtype=np.float32)

    # Direction check: free vertices should follow outward stretch direction in X.
    fem_dx = fem_verts[:, 0] - fem_rest[:, 0]
    spring_dx = spring_verts[:, 0] - spring_rest[:, 0]

    print(
        f"FEM_BW98 dx: fixed_left={fem_dx[0]:.6f}, fixed_right={fem_dx[1]:.6f}, "
        f"free_left={fem_dx[2]:.6f}, free_right={fem_dx[3]:.6f}"
    )
    print(
        f"Spring   dx: fixed_left={spring_dx[0]:.6f}, fixed_right={spring_dx[1]:.6f}, "
        f"free_left={spring_dx[2]:.6f}, free_right={spring_dx[3]:.6f}"
    )

    fem_width_rest = float(np.max(fem_rest[:, 0]) - np.min(fem_rest[:, 0]))
    fem_width_final = float(np.max(fem_verts[:, 0]) - np.min(fem_verts[:, 0]))
    spring_width_rest = float(np.max(spring_rest[:, 0]) - np.min(spring_rest[:, 0]))
    spring_width_final = float(np.max(spring_verts[:, 0]) - np.min(spring_verts[:, 0]))
    fixed_indices = get_fixed_indices()
    free_indices = np.array([i for i in range(len(fem_rest)) if i not in fixed_indices], dtype=np.int32)
    fem_free_move =  float(np.linalg.norm(fem_verts[free_indices, :] - fem_rest[free_indices, :], axis=1).mean())
    spring_free_move = float(np.linalg.norm(spring_verts[free_indices, :] - spring_rest[free_indices, :], axis=1).mean())

    print(
        f"FEM_BW98 width x: rest={fem_width_rest:.6f}, final={fem_width_final:.6f}, free_move={fem_free_move:.6f}; "
        f"Spring width x: rest={spring_width_rest:.6f}, final={spring_width_final:.6f}, free_move={spring_free_move:.6f}"
    )

    for local_vid, direction in fixed_dirs.items():
        assert fem_dx[local_vid] * direction[0] > 0.0, f"FEM_BW98 fixed vertex {local_vid} did not move in correct stretch direction."
        assert spring_dx[local_vid] * direction[0] > 0.0, f"Spring fixed vertex {local_vid} did not move in correct stretch direction."
    assert spring_width_final > spring_width_rest, "Spring cloth width did not increase in pulling direction."
    assert fem_free_move > 1e-6, "FEM_BW98 free vertices did not deform."
    assert spring_free_move > 1e-6, "Spring free vertices did not deform."

    output_dir = os.path.join(root, "Resources", "OutputMesh")
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "cloth_fem_bw98_vertices.npy"), fem_verts)
    np.save(os.path.join(output_dir, "cloth_spring_vertices.npy"), spring_verts)
    print(f"Saved final vertices to {output_dir}")

    solver.cleanup_device()
    print("2-model stretching test completed.")


if __name__ == "__main__":
    cli_args = parse_args()
    test_cloth_stretching_models(backend=cli_args.backend, advance_frames=cli_args.advance_frames, headless=cli_args.headless)
