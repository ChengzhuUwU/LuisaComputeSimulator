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
    p.add_argument("--mesh", default="cube", choices=["single", "cube"],
                   help="Tet mesh to test: a single tetrahedron or a 5-tet cube")
    p.add_argument("--no_floor", action="store_true",
                   help="Disable floor collision for debugging")
    p.add_argument("--headless", action="store_true")
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
    config.use_gpu = False  # Force CPU mode

    # ---- Register tet body -----------------------------------------------
    if args.mesh == "cube":
        verts, tets = make_unit_tet_cube(center=(0.0, 0.5, 0.0), scale=0.2)
    else:
        verts, tets = make_single_tet(center=(0.0, 0.5, 0.0), scale=0.2)

    print(f"Tet mesh: {len(verts)} vertices, {len(tets)} tets")
    report_tet_volume_stats(verts, tets)

    tet_body = solver.create_world_data_from_tet_array("tet_cube", verts, tets)
    tet_body.set_physics_material_tet(
        model="Spring",
        youngs_modulus=1e5,
        poisson_ratio=0.4,
    )
    tet_body.add_fixed_point_by_method("Left") 
    fixed_vids = tet_body.get_fixed_point_indices()
    reg_id = solver.register_world_data(tet_body)
    print(f"Registered tet_cube with id={reg_id}, fixed vertices={fixed_vids}")

    # ---- Initialize solver -----------------------------------------------
    solver.init_solver()
    print("Solver initialized.")

    # ---- Headless run -------------------------------------------------------
    output_dir = os.path.join(root, "Resources", "OutputMesh")
    os.makedirs(output_dir, exist_ok=True)

    if args.headless:
        solver.save_sim_result(obj_path=os.path.join(output_dir, "tet_init.obj"))
        for frame in range(args.advance_frames):
            if config.use_gpu:
                solver.physics_step_gpu()
            else:
                solver.physics_step_cpu()
            verts_out, _ = solver.get_object_sim_result_by_registration_id(reg_id)
            if len(verts_out):
                report_runtime_tet_stats(np.asarray(verts_out), tets, frame)

            all_verts = []
            reg_ids = [reg_id]  
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
