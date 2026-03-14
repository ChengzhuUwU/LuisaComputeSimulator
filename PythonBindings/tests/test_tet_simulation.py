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
    p = argparse.ArgumentParser(description="Tet simulation smoke test")
    p.add_argument("--backend", default="metal",
                   choices=["cuda", "dx", "vk", "metal"])
    p.add_argument("--advance_frames", type=int, default=30)
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

    # Standard 5-tet decomposition of a cube
    tets = np.array([
        [0, 1, 3, 4],
        [1, 4, 5, 6],
        [1, 3, 4, 6],
        [3, 4, 6, 7],
        [1, 2, 3, 6],
    ], dtype=np.int32)

    return verts, tets


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
    config.use_self_collision = False    # keep it simple for smoke test
    config.nonlinear_iter_count = 1
    config.use_ccd_linesearch = False

    # ---- Register tet body -----------------------------------------------
    verts, tets = make_unit_tet_cube(center=(0.0, 0.5, 0.0), scale=0.2)
    print(f"Tet mesh: {len(verts)} vertices, {len(tets)} tets")

    tet_body = solver.create_world_data_from_tet_array("tet_cube", verts, tets)
    tet_body.set_physics_material_tet(
        model="ARAP",
        youngs_modulus=1e5,
        poisson_ratio=0.4,
    )
    tet_body.add_fixed_point_by_method("Left") 
    reg_id = solver.register_world_data(tet_body)
    print(f"Registered tet_cube with id={reg_id}")

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
            # if (frame + 1) % 10 == 0:
            verts_out, faces_out = solver.get_object_sim_result_by_registration_id(reg_id)
            min_y = verts_out[:, 1].min() if len(verts_out) else float('nan')
            max_y = verts_out[:, 1].max() if len(verts_out) else float('nan')
            avg_y = verts_out[:, 1].mean() if len(verts_out) else float('nan')
            print(f"  frame {frame+1:3d}: min_y={min_y:.4f}, max_y={max_y:.4f}, avg_y={avg_y:.4f}")

        solver.save_sim_result(obj_path=os.path.join(output_dir, "tet_result.obj"))
        print(f"Saved result to {output_dir}")
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
