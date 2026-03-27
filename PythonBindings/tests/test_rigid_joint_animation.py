"""
Numerical behavior checks for rigid Joint constraints.

Headless mode (`--headless`):
- Builds 3 isolated joint scenes and advances simulation.
- Computes quantitative metrics from simulated vertex trajectories.
- Raises AssertionError if any metric violates expected behavior.

GUI mode:
- Runs the same scenes and prints metrics each physics step.
"""

import os
import sys

import numpy as np
import trimesh

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(root, "build", "bin"))
import lcs_py as lcs

import utils.arg_parser
from utils.animation_transform import DefaultTransformAnimation
from utils.body_animator import BodyAnimator

args = utils.arg_parser.parse_args()

solver = lcs.NewtonSolver()
solver.init_device(backend_name=args.backend)

cube_mesh_path = os.path.join(root, "Resources", "InputMesh", "cube.obj")
cube_mesh = trimesh.load(cube_mesh_path, process=False)

SCALE = 0.10
animators = []


def wrap_angle_rad(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def estimate_rotation_matrix(ref_vertices, curr_vertices):
    c_ref = np.mean(ref_vertices, axis=0)
    c_cur = np.mean(curr_vertices, axis=0)
    x = ref_vertices - c_ref
    y = curr_vertices - c_cur
    u, _, vt = np.linalg.svd(x.T @ y)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    return r


def estimate_yaw_z(ref_vertices, curr_vertices):
    r = estimate_rotation_matrix(ref_vertices, curr_vertices)
    return float(np.arctan2(r[1, 0], r[0, 0]))


def get_vertices(registration_id):
    verts, _ = solver.get_object_sim_result_by_registration_id(registration_id)
    return np.asarray(verts, dtype=np.float64)


def get_center(registration_id):
    return np.mean(get_vertices(registration_id), axis=0)


def make_animated_driver(name, tx, ty, tz, transform: DefaultTransformAnimation):
    body = solver.create_world_data_from_array(name, cube_mesh.vertices, cube_mesh.faces)
    body.set_simulation_type(lcs.MaterialType.Rigid)
    body.set_scale(SCALE)
    body.set_translation(tx, ty, tz)

    animator = BodyAnimator(body)
    animator.add_rule_by_method(body, "All", transform)

    body_id = solver.register_world_data(body)
    animator.set_mesh_index(body_id)
    animators.append(animator)
    return body_id


def make_follower(name, tx, ty, tz):
    body = solver.create_world_data_from_array(name, cube_mesh.vertices, cube_mesh.faces)
    body.set_simulation_type(lcs.MaterialType.Rigid)
    body.set_scale(SCALE)
    body.set_translation(tx, ty, tz)
    return solver.register_world_data(body)


anchor = np.zeros(3, dtype=np.float32)

# Scene A: Fixed joint -> relative position and orientation should stay locked.
fixed_driver = make_animated_driver(
    "fixed_driver",
    0.00,
    0.25,
    0.00,
    DefaultTransformAnimation(
        use_rotate=True,
        rot_axis=[0.0, 0.0, 1.0],
        rot_ang_vel_deg=120.0,
    ),
)
fixed_follower = make_follower("fixed_follower", 0.20, 0.25, 0.00)
solver.add_fixed_joint(
    fixed_driver,
    fixed_follower,
    anchor,
    anchor,
    stiffness_pos=5.0e4,
    stiffness_rot=1.0e4,
)

# Scene B: Prismatic joint (body-local axis = [1,1,0]) -> sliding along that axis is free.
# The axis co-rotates with the driver body, so validation is done in the driver's local frame.
prismatic_driver = make_animated_driver(
    "prismatic_driver",
    1.00,
    0.25,
    0.00,
    DefaultTransformAnimation(
        # use_translate=True,
        # translate=[0.2, 0.2, 0.0],
        use_rotate=True,
        rot_axis=[0.0, 0.0, 1.0],
        rot_ang_vel_deg=120.0,
    ),
)
prismatic_follower = make_follower("prismatic_follower", 0.80, 0.25, 0.00)
solver.add_prismatic_joint(
    prismatic_driver,
    prismatic_follower,
    anchor,
    anchor,
    np.array([1.0, 0.0, 0.0], dtype=np.float32),
    stiffness_pos=5.0e4,
    stiffness_rot=1.0e4,
    slide_min=-100000.0,
    slide_max=100000.0,
)

# Scene C: Revolute joint (axis = Z) -> relative twist around Z should be free.
revolute_driver = make_animated_driver(
    "revolute_driver",
    2.00,
    0.25,
    0.00,
    DefaultTransformAnimation(
        # use_rotate=True,
        # rot_axis=[0.0, 0.0, 1.0],
        # rot_ang_vel_deg=120.0,
    ),
)
revolute_follower = make_follower("revolute_follower", 2.20, 0.25, 0.00)
solver.add_revolute_joint(
    revolute_driver,
    revolute_follower,
    anchor,
    anchor,
    np.array([0.0, 0.0, 1.0], dtype=np.float32),
    np.array([0.0, 0.0, 1.0], dtype=np.float32),
    np.array([0.0, 0.0, 1.0], dtype=np.float32),
    stiffness_pos=5.0e4,
    stiffness_axis=2.0e3,
)

solver.init_solver()
config_ref = solver.get_config()
config_ref.use_floor = False
config_ref.use_self_collision = False
config_ref.gravity = lcs.Float3(0.0, -9.0, 0.0)

output_dir = os.path.join(root, "Resources", "OutputMesh")
os.makedirs(output_dir, exist_ok=True)

tracked_ids = [
    fixed_driver,
    fixed_follower,
    prismatic_driver,
    prismatic_follower,
    revolute_driver,
    revolute_follower,
]
rest_vertices = {bid: get_vertices(bid) for bid in tracked_ids}
rest_centers = {bid: np.mean(rest_vertices[bid], axis=0) for bid in tracked_ids}

rest_rel_fixed = rest_centers[fixed_follower] - rest_centers[fixed_driver]
rest_rel_prismatic = rest_centers[prismatic_follower] - rest_centers[prismatic_driver]
rest_rel_revolute = rest_centers[revolute_follower] - rest_centers[revolute_driver]


def update_animation():
    for animator in animators:
        animator.update_animation(solver, config_ref.current_frame, config_ref.implicit_dt)


def physics_step():
    update_animation()
    if config_ref.use_gpu:
        solver.physics_step_gpu()
    else:
        solver.physics_step_cpu()


def compute_metrics():
    fixed_driver_vertices = get_vertices(fixed_driver)
    fixed_follower_vertices = get_vertices(fixed_follower)
    prismatic_driver_vertices = get_vertices(prismatic_driver)
    prismatic_follower_vertices = get_vertices(prismatic_follower)
    revolute_driver_vertices = get_vertices(revolute_driver)
    revolute_follower_vertices = get_vertices(revolute_follower)

    fixed_driver_center = np.mean(fixed_driver_vertices, axis=0)
    fixed_follower_center = np.mean(fixed_follower_vertices, axis=0)
    prismatic_driver_center = np.mean(prismatic_driver_vertices, axis=0)
    prismatic_follower_center = np.mean(prismatic_follower_vertices, axis=0)
    revolute_driver_center = np.mean(revolute_driver_vertices, axis=0)
    revolute_follower_center = np.mean(revolute_follower_vertices, axis=0)

    fixed_driver_rot = estimate_rotation_matrix(rest_vertices[fixed_driver], fixed_driver_vertices)
    expected_fixed_rel = fixed_driver_rot @ rest_rel_fixed
    fixed_rel_delta = (fixed_follower_center - fixed_driver_center) - expected_fixed_rel
    fixed_pos_error = float(np.linalg.norm(fixed_rel_delta))

    prismatic_rel_delta = (prismatic_follower_center - prismatic_driver_center) - rest_rel_prismatic
    prismatic_driver_motion = prismatic_driver_center - rest_centers[prismatic_driver]

    # Validate prismatic constraint in driver's rotated local frame.
    # The sliding axis is body-local [1,1,0]/sqrt(2); it co-rotates with the driver.
    # Constraint: (p_B - p_A) = A*(d0_local + t*axis_local), so in local frame
    # (R.T @ current_relative - d0_local) must lie along axis_local.
    prismatic_driver_rot = estimate_rotation_matrix(
        rest_vertices[prismatic_driver], prismatic_driver_vertices
    )
    prismatic_axis_local = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    prismatic_axis_local = prismatic_axis_local / float(np.linalg.norm(prismatic_axis_local))
    # Current relative position in driver's local frame, minus rest offset d0_local
    current_relative = prismatic_follower_center - prismatic_driver_center
    dev_local = prismatic_driver_rot.T @ current_relative - rest_rel_prismatic
    prismatic_free_frac = float(np.dot(dev_local, prismatic_axis_local))
    prismatic_locked_vec = dev_local - prismatic_free_frac * prismatic_axis_local
    prismatic_locked_error = float(np.linalg.norm(prismatic_locked_vec))

    fixed_driver_yaw = estimate_yaw_z(rest_vertices[fixed_driver], fixed_driver_vertices)
    fixed_follower_yaw = estimate_yaw_z(rest_vertices[fixed_follower], fixed_follower_vertices)
    fixed_yaw_error = abs(wrap_angle_rad(fixed_follower_yaw - fixed_driver_yaw))

    revolute_driver_rot = estimate_rotation_matrix(rest_vertices[revolute_driver], revolute_driver_vertices)
    revolute_driver_yaw = estimate_yaw_z(rest_vertices[revolute_driver], revolute_driver_vertices)
    revolute_follower_yaw = estimate_yaw_z(rest_vertices[revolute_follower], revolute_follower_vertices)
    revolute_relative_yaw = abs(wrap_angle_rad(revolute_follower_yaw - revolute_driver_yaw))
    expected_revolute_rel = revolute_driver_rot @ rest_rel_revolute
    revolute_rel_delta = (revolute_follower_center - revolute_driver_center) - expected_revolute_rel
    revolute_pos_error = float(np.linalg.norm(revolute_rel_delta))

    return {
        "fixed_pos_error": fixed_pos_error,
        "fixed_yaw_error": fixed_yaw_error,
        "prismatic_rel_delta": prismatic_rel_delta,
        "prismatic_driver_motion": prismatic_driver_motion,
        "prismatic_locked_error": prismatic_locked_error,
        "prismatic_free_frac": prismatic_free_frac,
        "revolute_driver_yaw_abs": abs(revolute_driver_yaw),
        "revolute_relative_yaw": revolute_relative_yaw,
        "revolute_pos_error": revolute_pos_error,
    }


def validate_metrics(metrics):
    assert metrics["fixed_pos_error"] < 2.0e-3, (
        f"Fixed joint failed position lock: error={metrics['fixed_pos_error']:.6e}"
    )
    assert metrics["fixed_yaw_error"] < 5.0e-2, (
        f"Fixed joint failed orientation lock: yaw error={metrics['fixed_yaw_error']:.6f} rad"
    )

    assert float(np.linalg.norm(metrics["prismatic_driver_motion"])) > 5.0e-2, (
        f"Prismatic driver did not move enough: total={float(np.linalg.norm(metrics['prismatic_driver_motion'])):.6f}"
    )
    assert float(np.linalg.norm(metrics["prismatic_rel_delta"])) > 5.0e-2, (
        f"Prismatic follower appears decoupled from driver: rel_delta_norm={float(np.linalg.norm(metrics['prismatic_rel_delta'])):.6f}"
    )
    assert metrics["prismatic_locked_error"] < 1.0e-2, (
        f"Prismatic locked-plane drift too large in driver local frame: error={metrics['prismatic_locked_error']:.6f}"
    )

    assert metrics["revolute_driver_yaw_abs"] > 5.0e-1, (
        f"Revolute driver did not rotate enough: yaw={metrics['revolute_driver_yaw_abs']:.6f} rad"
    )
    assert metrics["revolute_relative_yaw"] > 3.0e-1, (
        f"Revolute free-twist behavior failed: relative yaw={metrics['revolute_relative_yaw']:.6f} rad"
    )
    assert metrics["revolute_pos_error"] < 1.5e-1, (
        f"Revolute position coupling too weak: rel-position error={metrics['revolute_pos_error']:.6f}"
    )


def print_metrics(metrics):
    print("[joint-check] fixed_pos_error        =", f"{metrics['fixed_pos_error']:.6e}")
    print("[joint-check] fixed_yaw_error(rad)   =", f"{metrics['fixed_yaw_error']:.6e}")
    print("[joint-check] prismatic_rel_delta    =", metrics["prismatic_rel_delta"])
    print("[joint-check] prismatic_driver_move  =", metrics["prismatic_driver_motion"])
    print("[joint-check] prismatic_locked_err   =", f"{metrics['prismatic_locked_error']:.6e}")
    print("[joint-check] prismatic_free_frac    =", f"{metrics['prismatic_free_frac']:.6f}")
    print("[joint-check] revolute_driver_yaw    =", f"{metrics['revolute_driver_yaw_abs']:.6e}")
    print("[joint-check] revolute_relative_yaw  =", f"{metrics['revolute_relative_yaw']:.6e}")
    print("[joint-check] revolute_pos_error     =", f"{metrics['revolute_pos_error']:.6e}")


if args.headless:
    solver.save_sim_result(os.path.join(output_dir, "joint_constraint_test_init.obj"))
    for _ in range(args.advance_frames):
        physics_step()
    solver.save_sim_result(os.path.join(output_dir, "joint_constraint_test_result.obj"))

    metrics = compute_metrics()
    print_metrics(metrics)
    validate_metrics(metrics)
    print("[joint-check] PASS")
else:
    import utils.polyscope_gui

    class JointCheckGUI(utils.polyscope_gui.SimulationGUI):
        def _physics_step(self):
            physics_step()
            metrics = compute_metrics()
            print_metrics(metrics)

    gui = JointCheckGUI(solver, config_ref, output_dir)
    gui.show()

solver.cleanup_device()
