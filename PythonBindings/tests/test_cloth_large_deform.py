from utils.test_script_path import PROJECT_ROOT
"""
Compare cloth stretch models (FEM_BW98 vs Spring) under symmetric pulling.

This test builds a minimal cloth mesh with 4 vertices and 2 triangles, creates two
cloth objects with different stretch models, and pulls two fixed points to opposite
sides frame-by-frame via update_per_vertex_animation.
"""

import argparse
import os
from typing import Dict

import numpy as np

import lcs_py as lcs


def parse_args():
    parser = argparse.ArgumentParser(description="Cloth stretch model comparison test")
    parser.add_argument("--backend", type=str, default="metal", choices=["cuda", "dx", "metal", "vk", "fallback", "cpu", "remote"])
    parser.add_argument("--advance_frames", type=int, default=40)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()

def get_fixed_indices():
    return np.array([0, 3], dtype=np.int32)

def get_fixed_dirs():
    return {
        0: np.array([-1.0, -1.0, 0.0], dtype=np.float32),  # bottom-left vertex pulled away (up-left direction)
        # 1: np.array([1.0, -1.0, 0.0], dtype=np.float32),   # bottom-right vertex pulled up-right
        # 2: np.array([-1.0, 1.0, 0.0], dtype=np.float32), # top-left vertex pulled down-left
        3: np.array([1.0, 1.0, 0.0], dtype=np.float32),  # top-right vertex pulled away (down-right direction)
    }

def make_simple_cloth_mesh():
    """Return a 4-vertex cloth quad split into 2 triangles with symmetric orientation.
    
    Mesh layout:
      2(−0.5, 1) -------- 3(0.5, 1)
      |                /  |
      |             /     |
      |          /        |
      |       /           |
      0(−0.5, 0) -------- 1(0.5, 0)
    
    Triangle orders are chosen to ensure symmetric local coordinate systems:
    - Triangle 0: [0, 1, 2] - standard left-bottom origin
    - Triangle 1: [1, 3, 2] - right-bottom origin (mirrors triangle 0's structure)
    """
    vertices = np.array(
        [
            [-0.5, 0.0, 0.0],  # 0: bottom-left
            [0.5, 0.0, 0.0],   # 1: bottom-right
            [-0.5, 1.0, 0.0],  # 2: top-left
            [0.5, 1.0, 0.0],   # 3: top-right
        ],
        dtype=np.float64,
    )
    # FIXED: Changed triangle 1 from [2, 1, 3] to [1, 3, 2]
    # This ensures both triangles use symmetric local coordinate systems for FEM_BW98
    triangles = np.array(
        [
            [0, 1, 2],         # Triangle 0: origin at vertex 0
            # [1, 3, 2],         # Triangle 1: origin at vertex 1 (FIXED from [2,1,3])
            [2, 1, 3] # Original order with vertex 2 as origin, leads to asymmetric local coordinates and energy metrics
        ],
        dtype=np.int32,
    )
    return vertices, triangles


def _compute_dm_inv_from_rest_triangle(x0: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Match StretchEnergy::get_Dm_inv with a local 2D basis built from rest positions."""
    r1 = x1 - x0
    r2 = x2 - x0
    cross = np.cross(r1, r2)
    axis1 = r1 / (np.linalg.norm(r1) + 1e-12)
    axis2 = np.cross(cross, axis1)
    axis2 /= (np.linalg.norm(axis2) + 1e-12)

    uv0 = np.array([np.dot(axis1, x0), np.dot(axis2, x0)], dtype=np.float64)
    uv1 = np.array([np.dot(axis1, x1), np.dot(axis2, x1)], dtype=np.float64)
    uv2 = np.array([np.dot(axis1, x2), np.dot(axis2, x2)], dtype=np.float64)

    duv0 = uv1 - uv0
    duv1 = uv2 - uv0
    duv = np.column_stack([duv0, duv1])  # 2x2
    return np.linalg.inv(duv)


def print_fem_bw98_stretch_shear_breakdown(
    label: str,
    rest_positions: np.ndarray,
    curr_positions: np.ndarray,
    triangles: np.ndarray,
    youngs_modulus: float,
    poisson_ratio: float,
    thickness: float,
    lambda_override: float | None = None,
):
    """Print per-face and total FEM_BW98 stretch/shear energy components."""
    mu = thickness * (youngs_modulus / (2.0 * (1.0 + poisson_ratio)))
    lam = thickness * (youngs_modulus * poisson_ratio / (1.0 - poisson_ratio * poisson_ratio))
    if lambda_override is not None:
        lam = float(lambda_override)

    total_stretch = 0.0
    total_shear = 0.0
    total_area = 0.0

    print(f"\n{label} FEM_BW98 energy decomposition:")
    print(" fid | stretch_energy | shear_energy | shear_ratio | I6=dot(Fu,Fv)")
    print("-----+----------------+--------------+-------------+----------------")

    for fid, tri in enumerate(triangles):
        i0, i1, i2 = (int(tri[0]), int(tri[1]), int(tri[2]))
        x0r, x1r, x2r = rest_positions[i0], rest_positions[i1], rest_positions[i2]
        x0, x1, x2 = curr_positions[i0], curr_positions[i1], curr_positions[i2]

        dm_inv = _compute_dm_inv_from_rest_triangle(x0r, x1r, x2r)
        ds = np.column_stack([x1 - x0, x2 - x0])  # 3x2
        f_mat = ds @ dm_inv  # 3x2
        fu = f_mat[:, 0]
        fv = f_mat[:, 1]

        i5u = float(np.dot(fu, fu))
        i5v = float(np.dot(fv, fv))
        i6 = float(np.dot(fu, fv))

        stretch = 0.5 * mu * ((np.sqrt(i5u) - 1.0) ** 2 + (np.sqrt(i5v) - 1.0) ** 2)
        shear = 0.5 * lam * (i6 ** 2)

        area = 0.5 * np.linalg.norm(np.cross(x1r - x0r, x2r - x0r))
        stretch *= area
        shear *= area

        total_stretch += stretch
        total_shear += shear
        total_area += area

        ratio = shear / (stretch + 1e-12)
        print(f"{fid:>4d} | {stretch:>14.6e} | {shear:>12.6e} | {ratio:>11.6f} | {i6:>14.6e}")

    total_ratio = total_shear / (total_stretch + 1e-12)
    print(
        f"total area={total_area:.6f}, total_stretch={total_stretch:.6e}, "
        f"total_shear={total_shear:.6e}, total_shear/stretch={total_ratio:.6f}"
    )


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
        direction = np.asarray(direction, dtype=np.float32)
        direction /= np.linalg.norm(direction) + 1e-8  # Normalize to ensure consistent pull speed
        rest_pos = rest_positions[local_vid]
        target_pos = rest_pos + direction * np.float32(pull_speed * curr_time)
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
    before = np.asarray(cloth.get_fixed_point_indices(), dtype=np.uint32)
    cloth.add_fixed_point_by_indices(np.array(get_fixed_indices(), dtype=np.int32))
    after = np.asarray(cloth.get_fixed_point_indices(), dtype=np.uint32)
    print(f"{name}: Added fixed points {after.tolist()}, total fixed points now {len(after)}")

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
    config.nonlinear_iter_count = 1
    config.pcg_iter_count = 50
    config.use_ccd_linesearch = False
    config.use_gpu = False
    config.gravity = lcs.Float3(0.0, 0.0, 0.0)
    config.implicit_dt = 0.01

    output_dir = os.path.join(PROJECT_ROOT, "Resources", "OutputMesh")
    os.makedirs(output_dir, exist_ok=True)

    solver.init_solver()
    print("Running 2-model cloth stretching test on 4-vertex mesh...")

    # Pull the two pinned points apart along +/-X.
    # fixed_dirs = {
    #     0: np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    #     2: np.array([1.0, 0.0, 0.0], dtype=np.float32),
    # }

    num_verts = fem_rest.shape[0] + spring_rest.shape[0]
    vert_masses = [solver.get_vert_mass(idx) for idx in range(num_verts)]
    print(f"Vertex masses (first 4 are FEM cloth, next 4 are Spring cloth): {vert_masses}")

    fixed_indices = get_fixed_indices()
    fixed_dict = get_fixed_dirs()

    pull_speed = 10.0

    if headless:
        for frame in range(advance_frames):
            apply_fixed_point_stretch(solver, fem_id, fem_rest, frame, config.implicit_dt, fixed_dict, pull_speed)
            apply_fixed_point_stretch(solver, spring_id, spring_rest, frame, config.implicit_dt, fixed_dict, pull_speed)
            solver.physics_step_gpu()
        solver.save_sim_result(obj_path=os.path.join(output_dir, "result.obj"))
    else:
        import utils.polyscope_gui
        class AnimatedSimulationGUI(utils.polyscope_gui.SimulationGUI):
            def _physics_step(self):
                frame = config.current_frame
                apply_fixed_point_stretch(solver, fem_id, fem_rest, frame, config.implicit_dt, fixed_dict, pull_speed)
                apply_fixed_point_stretch(solver, spring_id, spring_rest, frame, config.implicit_dt, fixed_dict, pull_speed)
                super()._physics_step()
        gui = AnimatedSimulationGUI(solver, config, output_dir)
        gui.show()

    fem_verts, _ = solver.get_object_sim_result_by_registration_id(fem_id)
    spring_verts, _ = solver.get_object_sim_result_by_registration_id(spring_id)

    fem_verts = np.asarray(fem_verts, dtype=np.float32)
    spring_verts = np.asarray(spring_verts, dtype=np.float32)

    # Direction check: free vertices should follow outward stretch direction in X.
    valid_axis = range(0, 2)  # Only check X and Y, ignore Z
    fem_dx = fem_verts[:, valid_axis] - fem_rest[:, valid_axis]
    spring_dx = spring_verts[:, valid_axis] - spring_rest[:, valid_axis]

    fixed_indices = get_fixed_indices()
    free_indices = np.array([i for i in range(len(fem_rest)) if i not in fixed_indices], dtype=np.int32)

    def _print_vertex_rows(vertex_ids: np.ndarray, tag: str):
        for local_vid in vertex_ids:
            fem_xy = fem_dx[local_vid]
            spring_xy = spring_dx[local_vid]
            print(
                f"{int(local_vid):>3d} | {tag:<5s} | "
                f"({fem_xy[0]:>+9.6f}, {fem_xy[1]:>+9.6f}) | "
                f"({spring_xy[0]:>+9.6f}, {spring_xy[1]:>+9.6f})"
            )

    print("\nVertex displacement summary (XY only):")
    print(" vid | type  | FEM_BW98 (dx, dy)         | Spring (dx, dy)")
    print("-----+-------+----------------------------+----------------------------")
    _print_vertex_rows(fixed_indices, "fixed")
    _print_vertex_rows(free_indices, "free")

    _, triangles = make_simple_cloth_mesh()
    print_fem_bw98_stretch_shear_breakdown(
        label="FEM cloth",
        rest_positions=fem_rest,
        curr_positions=fem_verts,
        triangles=triangles,
        youngs_modulus=1e5,
        poisson_ratio=0.3,
        thickness=0.001,
        # lambda_override=0.0,
    )
    print_fem_bw98_stretch_shear_breakdown(
        label="Spring cloth (projected on FEM metrics)",
        rest_positions=spring_rest,
        curr_positions=spring_verts,
        triangles=triangles,
        youngs_modulus=1e5,
        poisson_ratio=0.3,
        thickness=0.001,
    )
    

    output_dir = os.path.join(PROJECT_ROOT, "Resources", "OutputMesh")
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "cloth_fem_bw98_vertices.npy"), fem_verts)
    np.save(os.path.join(output_dir, "cloth_spring_vertices.npy"), spring_verts)
    print(f"Saved final vertices to {output_dir}")

    solver.cleanup_device()
    print("2-model stretching test completed.")


if __name__ == "__main__":
    cli_args = parse_args()
    test_cloth_stretching_models(backend=cli_args.backend, advance_frames=cli_args.advance_frames, headless=cli_args.headless)
