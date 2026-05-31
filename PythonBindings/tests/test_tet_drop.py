from utils.test_script_path import PROJECT_ROOT
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

import numpy as np

# -- locate the built lcs_py module ----------------------------------------
import lcs_py as lcs


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args():
    import platform
    p = argparse.ArgumentParser(description="Tet simulation smoke test")
    default_backend = "metal" if platform.system() == "Darwin" else "cuda"
    p.add_argument("--backend", default=default_backend,
                   choices=["cuda", "dx", "metal", "vk", "fallback", "cpu", "remote"])
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

def make_unit_tet_cube2(center=(0.0, 0.5, 0.0), scale=0.4):
    """Return (vertices [4,3], tets [1,4]) for a unit regular tetrahedron."""
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


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    args = parse_args()

    solver = lcs.NewtonSolver()
    solver.init_device(backend_name=args.backend)

    config = solver.get_config()
    # config.use_floor = True
    # config.floor = lcs.Float3(0.0, 0.0, 0.0)
    # config.use_ccd_linesearch = True
    # config.use_self_collision = False    # keep it simple for smoke test
    config.set_nonlinear_iter_count(3)  # Increased for stability
    config.set_use_gpu(False)  # Force CPU mode

    # ---- Register random tet bodies (non-overlapping at init) ------------
    num_bodies = 20
    tet_scale = 0.2
    rng = np.random.default_rng(42)

    base_verts, tets = make_unit_tet_cube(center=(0.0, 0.0, 0.0), scale=tet_scale)
    tet_centroid = base_verts.mean(axis=0)
    tet_radius = np.linalg.norm(base_verts - tet_centroid, axis=1).max()
    min_center_dist = 2.2 * tet_radius

    centers = []
    max_trials = 20000
    for _ in range(max_trials):
        if len(centers) >= num_bodies:
            break
        candidate = np.array([
            rng.uniform(-1.0, 1.0),
            rng.uniform(1.5, 7.0),
            rng.uniform(-1.0, 1.0),
        ], dtype=np.float64)
        if all(np.linalg.norm(candidate - c) >= min_center_dist for c in centers):
            centers.append(candidate)

    if len(centers) < num_bodies:
        raise RuntimeError(
            f"Could only place {len(centers)} / {num_bodies} tet bodies without overlap."
        )

    reg_ids = []
    for i, c in enumerate(centers):
        verts, _ = make_unit_tet_cube(center=tuple(c.tolist()), scale=tet_scale)
        tet_body = solver.create_world_data_from_tet_array(f"tet_{i:02d}", verts, tets)
        tet_body.set_physics_material_tet(
            model="ARAP",
            youngs_modulus=1e5,
            poisson_ratio=0.4,
        )
        reg_id = solver.register_world_data(tet_body)
        reg_ids.append(reg_id)

    print(
        f"Registered {len(reg_ids)} tet bodies, each with {len(base_verts)} vertices and {len(tets)} tet."
    )


    bowl_mesh_path = os.path.join(PROJECT_ROOT, 'Resources', 'InputMesh', 'bowl', 'bowl.obj')
    bowl = solver.create_world_data_from_file_path('bowl', bowl_mesh_path)
    bowl.set_simulation_type(lcs.MaterialType.Cloth)
    bowl.set_physics_material_cloth(thickness=0.001)
    bowl.set_scale(10.0)
    bowl.set_translation(0.0, 1.1, 0.0)
    bowl.add_fixed_point_by_method("All") 
    bowl_id = solver.register_world_data(bowl)


    # ---- Initialize solver -----------------------------------------------
    solver.init_solver()
    print("Solver initialized.")

    # ---- Headless run -------------------------------------------------------
    output_dir = os.path.join(PROJECT_ROOT, "Resources", "OutputMesh")
    os.makedirs(output_dir, exist_ok=True)

    if args.headless:
        solver.save_sim_result(obj_path=os.path.join(output_dir, "tet_init.obj"))
        for frame in range(args.advance_frames):
            if config.use_gpu:
                solver.physics_step_gpu()
            else:
                solver.physics_step_cpu()
                
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
