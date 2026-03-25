import argparse
import os
import sys

import numpy as np
import trimesh

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(root, "build", "bin"))
import lcs_py as lcs


def parse_args():
    parser = argparse.ArgumentParser(description="Rigid joint animation test")
    parser.add_argument("--backend", type=str, default="metal", choices=["cuda", "dx", "vk", "metal"])
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--advance_frames", type=int, default=60)
    parser.add_argument(
        "--joint",
        type=str,
        default="fixed",
        choices=["fixed", "prismatic", "revolute"],
        help="Joint type to validate.",
    )
    return parser.parse_args()


args = parse_args()
solver = lcs.NewtonSolver()
solver.init_device(backend_name=args.backend)

cube_mesh_path = os.path.join(root, "Resources", "InputMesh", "cube.obj")
cube_mesh = trimesh.load(cube_mesh_path, process=False)


# Body A: kinematic driver (fixed all vertices) to provide a moving target.
def make_driver_body(offset: float = 0.0):
    body = solver.create_world_data_from_array("driver", cube_mesh.vertices, cube_mesh.faces)
    body.set_simulation_type(lcs.MaterialType.Rigid)
    body.set_scale(0.10)
    body.set_translation(0.2, 0.20, 0.0 + offset)
    # body.add_fixed_point_by_method("All")
    return solver.register_world_data(body)


# Body B: dynamic rigid body constrained by joint.
def make_follower_body(offset: float = 0.0):
    body = solver.create_world_data_from_array("follower", cube_mesh.vertices, cube_mesh.faces)
    body.set_simulation_type(lcs.MaterialType.Rigid)
    body.set_scale(0.10)
    body.set_translation(0.0, 0.2, 0.0+ offset)
    return solver.register_world_data(body)

# Body C
def make_obstacle_body(offset: float = 0.0):
    body = solver.create_world_data_from_array("Obstacle", cube_mesh.vertices, cube_mesh.faces)
    body.set_simulation_type(lcs.MaterialType.Rigid)
    body.set_scale(0.10)
    body.set_translation(-0.05, 0.01, 0.0+ offset)
    body.add_fixed_point_by_method("All")
    return solver.register_world_data(body)


def add_joint(driver_id: int, follower_id: int, joint: str):
    a0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    b0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    if joint == "fixed":
        solver.add_fixed_joint(
            driver_id,
            follower_id,
            a0,
            b0,
            stiffness_pos=2.0e4,
            stiffness_rot=4.0e3,
        )
    elif joint == "prismatic":
        solver.add_prismatic_joint(
            driver_id,
            follower_id,
            a0,
            b0,
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            stiffness_pos=2.0e4,
            stiffness_rot=4.0e3,
        )
    elif joint == "revolute":
        solver.add_revolute_joint(
            driver_id,
            follower_id,
            a0,
            b0,
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            stiffness_pos=2.0e4,
            stiffness_axis=2.0e3,
        )

fixed_joint_offset = 0.0
fixed_joint_a = make_driver_body(fixed_joint_offset)
fixed_joint_b = make_follower_body(fixed_joint_offset)
add_joint(fixed_joint_a, fixed_joint_b, "fixed")
fixed_joint_c = make_obstacle_body(fixed_joint_offset)

prismatic_joint_offset = 0.2
prismatic_joint_a = make_driver_body(prismatic_joint_offset)
prismatic_joint_b = make_follower_body(prismatic_joint_offset)
add_joint(prismatic_joint_a, prismatic_joint_b, "prismatic")
prismatic_joint_c = make_obstacle_body(prismatic_joint_offset)

revolute_joint_offset = 0.4
revolute_joint_a = make_driver_body(revolute_joint_offset)
revolute_joint_b = make_follower_body(revolute_joint_offset)
add_joint(revolute_joint_a, revolute_joint_b, "revolute")
revolute_joint_c = make_obstacle_body(revolute_joint_offset)


solver.init_solver()
config_ref = solver.get_config()
config_ref.use_floor = True
output_dir = os.path.join(root, "Resources", "OutputMesh")
os.makedirs(output_dir, exist_ok=True)


def update_driver(frame_idx: int):
    t = frame_idx * config_ref.implicit_dt
    translation = np.array([
        0.08 * np.sin(2.0 * t),
        0.20 + 0.03 * np.sin(1.5 * t),
        0.0,
    ], dtype=np.float32)
    rotation = np.array([
        0.0,
        0.0,
        35.0 * np.sin(1.5 * t),
    ], dtype=np.float32)
    # solver.update_per_body_animation(driver_id, translation, rotation)


if args.headless:
    solver.save_sim_result(os.path.join(output_dir, f"joint_{args.joint}_init.obj"))
    for frame in range(args.advance_frames):
        update_driver(frame)
        if config_ref.use_gpu:
            solver.physics_step_gpu()
        else:
            solver.physics_step_cpu()
    solver.save_sim_result(os.path.join(output_dir, f"joint_{args.joint}_result.obj"))
else:
    import utils.polyscope_gui

    class JointSimulationGUI(utils.polyscope_gui.SimulationGUI):
        def _physics_step(self):
            update_driver(config_ref.current_frame)
            super()._physics_step()

    gui = JointSimulationGUI(solver, config_ref, output_dir)
    gui.show()

solver.cleanup_device()
