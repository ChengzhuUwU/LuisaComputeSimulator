"""
test_tet_simulation.py
======================
Minimal headless smoke-test for tetrahedral body simulation.

Usage:
    python test_tet_simulation.py [--backend metal|cuda|dx|vk]
                                  [--advance_frames N]
                                  [--headless]

The script creates a small tet cube (2x2x2 = 8 vertices, 5 tets),
drops it under gravity onto the floor and runs N frames.
"""
import argparse
import os
import sys

import numpy as np

# -- locate the built lcs_py module ----------------------------------------
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(root, 'build', 'bin'))
import lcs_py as lcs


def parse_grid_resolution(text):
    """Parse resolution string like '10,10,20' into a positive integer triplet."""
    try:
        parts = [int(v.strip()) for v in text.split(',')]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"grid_resolution must be three integers like '10,10,20', got '{text}'"
        ) from e

    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"grid_resolution must contain exactly three integers, got '{text}'"
        )
    if any(v <= 0 for v in parts):
        raise argparse.ArgumentTypeError(
            f"grid_resolution must be positive, got '{text}'"
        )
    return tuple(parts)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args():
    import platform
    p = argparse.ArgumentParser(description="Tet simulation smoke test")
    default_backend = "metal" if platform.system() == "Darwin" else "cuda"
    p.add_argument("--backend", default=default_backend,
                   choices=["cuda", "dx", "vk", "metal"])
    p.add_argument("--advance_frames", type=int, default=30)
    p.add_argument("--mesh", default="grid_10x10x20", choices=["single", "cube", "grid_10x10x20"],
                   help="Tet mesh to test: a single tetrahedron or a 5-tet cube")
    p.add_argument(
        "--grid_resolution",
        type=parse_grid_resolution,
        default=(10, 10, 20),
        help="Resolution for grid mesh as 'nx,ny,nz' (default: 10,10,20)",
    )
    p.add_argument("--no_floor", default=False, action="store_true",
                   help="Disable floor collision for debugging")
    p.add_argument("--headless", default=False, action="store_true")
    p.add_argument("--use_gpu", default=False, action="store_true", help="Force GPU mode for testing")
    return p.parse_args()


# --------------------------------------------------------------------------
# Build a unit cube as a tet mesh (8 vertices, 5 tets)
# --------------------------------------------------------------------------
def make_unit_tet_cube(center=(0.0, 0.5, 0.0), scale=0.4):
    """Return (vertices [N,3], tets [M,4]) for a cube split into 5 tets."""
    cx, cy, cz = center
    s = scale
    # 8 corners of a cube
    verts = np.array([
        [cx - s, cy - s, cz - s],  # 0
        [cx + s, cy - s, cz - s],  # 1
        [cx + s, cy + s, cz - s],  # 2
        [cx - s, cy + s, cz - s],  # 3
        [cx - s, cy - s, cz + s],  # 4
        [cx + s, cy - s, cz + s],  # 5
        [cx + s, cy + s, cz + s],  # 6
        [cx - s, cy + s, cz + s],  # 7
    ], dtype=np.float64)

    print("Cube vertices:\n", verts)

    # Standard 5-tet decomposition of a cube
    tets = np.array([
        # [0, 1, 2, 3],
        [0, 1, 3, 4],
        [1, 4, 5, 6],
        [1, 3, 4, 6],
        [3, 4, 6, 7],
        [1, 2, 3, 6],
    ], dtype=np.int32)

    return verts, tets

def make_single_tet(center=(0.0, 0.5, 0.0), scale=0.4):
    """Return (vertices [4,3], tets [1,4]) for a single tetrahedron."""
    cx, cy, cz = center
    s = scale

    # Regular tetrahedron around origin, then scaled and translated.
    # Edge length of the unscaled tetra is 2*sqrt(2), so scale controls size.
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    verts = verts * s + np.array([cx, cy, cz], dtype=np.float64)

    tets = np.array([
        [0, 1, 2, 3],
    ], dtype=np.int32)

    return verts, tets


def make_grid_tet(origin=(-0.2, 0.3, -0.2), size=(0.4, 0.4, 0.8), resolution=(10, 10, 20)):
    """Return (vertices [N,3], tets [M,4]) for a structured 6-tet-per-cell grid."""

    rx, ry, rz = resolution
    if rx <= 0 or ry <= 0 or rz <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")

    origin = np.asarray(origin, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    cell = size / np.asarray([rx, ry, rz], dtype=np.float64)

    def vid(ix, iy, iz):
        # Same indexing order as the C++ grid_vertex_index style loops.
        return ((ix * (ry + 1) + iy) * (rz + 1) + iz)

    verts = np.empty(((rx + 1) * (ry + 1) * (rz + 1), 3), dtype=np.float64)
    for ix in range(rx + 1):
        for iy in range(ry + 1):
            for iz in range(rz + 1):
                idx = vid(ix, iy, iz)
                verts[idx] = origin + cell * np.array([ix, iy, iz], dtype=np.float64)

    def orient_positive(a, b, c, d):
        vol6 = np.linalg.det(np.column_stack((verts[b] - verts[a], verts[c] - verts[a], verts[d] - verts[a])))
        if vol6 > 0.0:
            return [a, b, c, d]
        return [a, c, b, d]

    tets = []
    for ix in range(rx):
        for iy in range(ry):
            for iz in range(rz):
                v000 = vid(ix, iy, iz)
                v100 = vid(ix + 1, iy, iz)
                v010 = vid(ix, iy + 1, iz)
                v110 = vid(ix + 1, iy + 1, iz)
                v001 = vid(ix, iy, iz + 1)
                v101 = vid(ix + 1, iy, iz + 1)
                v011 = vid(ix, iy + 1, iz + 1)
                v111 = vid(ix + 1, iy + 1, iz + 1)

                local = [
                    [v000, v100, v110, v111],
                    [v000, v110, v010, v111],
                    [v000, v010, v011, v111],
                    [v000, v011, v001, v111],
                    [v000, v001, v101, v111],
                    [v000, v101, v100, v111],
                ]
                for tet in local:
                    tets.append(orient_positive(*tet))

    tets = np.asarray(tets, dtype=np.int32)
    return verts, tets


def tet_signed_volume(v0, v1, v2, v3):
    """Signed tetra volume using vertex order (v0,v1,v2,v3)."""
    return np.linalg.det(np.column_stack((v1 - v0, v2 - v0, v3 - v0))) / 6.0


def report_tet_volume_stats(verts, tets):
    signed = np.array([tet_signed_volume(*verts[t]) for t in tets], dtype=np.float64)
    abs_v = np.abs(signed)
    print(
        "Tet signed volume stats:",
        f"min={signed.min():.8f}, max={signed.max():.8f}, "
        f"neg_count={(signed < 0).sum()}, zero_count={(abs_v < 1e-12).sum()}, "
        f"sum_signed={signed.sum():.8f}, sum_abs={abs_v.sum():.8f}",
    )


def report_runtime_tet_stats(verts, tets, frame):
    signed = np.array([tet_signed_volume(*verts[t]) for t in tets], dtype=np.float64)
    has_nan = np.isnan(verts).any() or np.isnan(signed).any()
    print(
        f"  frame {frame + 1:3d}: min_signed={signed.min():.8e}, "
        f"max_signed={signed.max():.8e}, neg_count={(signed < 0).sum()}, has_nan={has_nan}"
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    args = parse_args()

    solver = lcs.NewtonSolver()
    solver.init_device(backend_name=args.backend)

    config = solver.get_config()
    config.use_floor = False
    config.floor = lcs.Float3(0.0, 0.0, 0.0)
    config.use_ccd_linesearch = False
    config.use_self_collision = False    # keep it simple for smoke test
    config.nonlinear_iter_count = 10  # Increased for stability
    config.use_gpu = args.use_gpu  # Use GPU if requested

    # ---- Register tet body -----------------------------------------------
    if args.mesh == "cube":
        verts, tets = make_unit_tet_cube(center=(0.0, 0.5, 0.0), scale=0.2)
    elif args.mesh == "grid_10x10x20":
        verts, tets = make_grid_tet(origin=(-0.2, 0.3, -0.2), size=(0.4, 0.4, 0.8), resolution=args.grid_resolution)
    else:
        verts, tets = make_single_tet(center=(0.0, 0.5, 0.0), scale=0.2)

    print(f"Tet mesh: {len(verts)} vertices, {len(tets)} tets")
    report_tet_volume_stats(verts, tets)

    # Spring benchmark body
    spring_verts = verts.copy()
    spring_verts[:, 0] -= 0.35
    tet_spring = solver.create_world_data_from_tet_array("tet_spring", spring_verts, tets)
    tet_spring.set_physics_material_tet(
        model="Spring",
        youngs_modulus=1e4,
        poisson_ratio=0.4,
    )
    tet_spring.add_fixed_point_by_method("Front")
    spring_fixed_vids = tet_spring.get_fixed_point_indices()
    reg_spring = solver.register_world_data(tet_spring)
    print(f"Registered tet_spring with id={reg_spring}, fixed vertices={spring_fixed_vids}")

    # ARAP body to compare with spring benchmark
    arap_verts = verts.copy()
    arap_verts[:, 0] += 0.35
    tet_arap = solver.create_world_data_from_tet_array("tet_arap", arap_verts, tets)
    tet_arap.set_physics_material_tet(
        model="ARAP",
        youngs_modulus=1e5,
        poisson_ratio=0.4,
    )
    tet_arap.add_fixed_point_by_method("Left")
    arap_fixed_vids = tet_arap.get_fixed_point_indices()
    reg_arap = solver.register_world_data(tet_arap)
    print(f"Registered tet_arap with id={reg_arap}, fixed vertices={arap_fixed_vids}")

    # ---- Initialize solver -----------------------------------------------
    solver.init_solver()
    print("Solver initialized.")

    # ---- Headless run -------------------------------------------------------
    output_dir = os.path.join(root, "Resources", "OutputMesh")
    os.makedirs(output_dir, exist_ok=True)

    if args.headless:
        solver.save_sim_result(obj_path=os.path.join(output_dir, "tet_init.obj"))
        avg_y_diff = []
        for frame in range(args.advance_frames):
            if config.use_gpu:
                solver.physics_step_gpu()
            else:
                solver.physics_step_cpu()
            spring_out, _ = solver.get_object_sim_result_by_registration_id(reg_spring)
            arap_out, _ = solver.get_object_sim_result_by_registration_id(reg_arap)

            if len(spring_out):
                report_runtime_tet_stats(np.asarray(spring_out), tets, frame)
            if len(arap_out):
                report_runtime_tet_stats(np.asarray(arap_out), tets, frame)

            if len(spring_out) and len(arap_out):
                spring_y = np.asarray(spring_out)[:, 1].mean()
                arap_y = np.asarray(arap_out)[:, 1].mean()
                diff = abs(spring_y - arap_y)
                avg_y_diff.append(diff)
                print(f"    compare spring_vs_arap: avg_y_diff={diff:.6f}")

            all_verts = []
            reg_ids = [reg_spring, reg_arap]
            for rid in reg_ids:
                verts_out, _ = solver.get_object_sim_result_by_registration_id(rid)
                if len(verts_out):
                    all_verts.append(verts_out)

            if all_verts:
                verts_stack = np.concatenate(all_verts, axis=0)
                min_y = verts_stack[:, 1].min()
                max_y = verts_stack[:, 1].max()
                avg_y = verts_stack[:, 1].mean()
            else:
                min_y = float('nan')
                max_y = float('nan')
                avg_y = float('nan')

            print(
                f"  frame {frame+1:3d}: bodies={len(reg_ids)}, min_y={min_y:.4f}, "
                f"max_y={max_y:.4f}, avg_y={avg_y:.4f}"
            )

        solver.save_sim_result(obj_path=os.path.join(output_dir, "tet_result.obj"))

        if avg_y_diff:
            final_diff = avg_y_diff[-1]
            mean_diff = float(np.mean(avg_y_diff))
            early_window = min(len(avg_y_diff), 10)
            early_mean = float(np.mean(avg_y_diff[:early_window]))
            print(
                f"Spring/ARAP trajectory diff: final={final_diff:.6f}, mean={mean_diff:.6f}, "
                f"early_mean={early_mean:.6f}"
            )
            if early_mean > 0.08 or final_diff > 5.0:
                raise RuntimeError(
                    "ARAP deviates too much from Spring benchmark: "
                    f"early_mean={early_mean:.6f}, final={final_diff:.6f}"
                )
    else:
        # Interactive GUI
        try:
            import utils.polyscope_gui
            gui = utils.polyscope_gui.SimulationGUI(solver, config, output_dir)
            gui.show()
        except ImportError:
            print("polyscope_gui not available, running headless instead.")
            for _ in range(args.advance_frames):
                if config.use_gpu:
                    solver.physics_step_gpu()
                else:
                    solver.physics_step_cpu()

    solver.cleanup_device()
    print("Done.")


if __name__ == "__main__":
    main()
