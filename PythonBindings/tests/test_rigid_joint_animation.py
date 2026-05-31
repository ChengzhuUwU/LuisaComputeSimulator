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

import numpy as np
import trimesh

import lcs_py as lcs


from run_tests import create_default_parser
from utils.animation_transform import DefaultTransformAnimation
from utils.body_animator import BodyAnimator
from utils.test_script_path import PROJECT_ROOT

args = create_default_parser().parse_args()

solver = lcs.NewtonSolver()
solver.init_device(backend_name=args.backend)

cube_mesh_path = os.path.join(PROJECT_ROOT, "Resources", "InputMesh", "cube.obj")
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

# Scene B: Prismatic joint (body-local axis = [1,0,0]) -> sliding along that axis is free.
# The axis co-rotates with the driver body, so validation is done in the driver's local frame.
prismatic_driver = make_animated_driver(
    "prismatic_driver",
    1.00,
    0.25,
    0.00,
    DefaultTransformAnimation(
        use_translate=False,
        use_rotate=True,
        rot_axis=[0.0, 0.0, 1.0],
        rot_ang_vel_deg=120.0,
    ),
)
prismatic_follower = make_follower("prismatic_follower", 1.20, 0.25, 0.00)
solver.add_prismatic_joint(
    prismatic_driver,
    prismatic_follower,
    anchor,
    anchor,
    np.array([1.0, 0.0, 0.0], dtype=np.float32),
    stiffness_pos=5.0e4,
    stiffness_rot=1.0e4,
    slide_min=0.1,
    slide_max=0.3,
)

# Scene C: Revolute joint (axis = Z) -> relative twist around Z should be free.
revolute_driver = make_animated_driver(
    "revolute_driver",
    2.00,
    0.25,
    0.00,
    DefaultTransformAnimation(
        use_rotate=True,
        rot_axis=[0.0, 0.0, 1.0],
        rot_ang_vel_deg=120.0,
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

# Scene D: Fixed joint (translation-only driver) -> follower must track position
# AND maintain orientation even though driver never rotates.
fixed2_driver = make_animated_driver(
    "fixed2_driver",
    0.00,
    0.60,
    0.00,
    DefaultTransformAnimation(
        use_translate=True,
        translate=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        use_rotate=False,
    ),
)
fixed2_follower = make_follower("fixed2_follower", 0.20, 0.60, 0.00)
solver.add_fixed_joint(
    fixed2_driver,
    fixed2_follower,
    anchor,
    anchor,
    stiffness_pos=5.0e4,
    stiffness_rot=1.0e4,
)

# Scene E: Prismatic joint (Y-axis slide, driver translates along X)
# -> follower tracks X motion but can slide freely along Y under gravity.
prismatic2_driver = make_animated_driver(
    "prismatic2_driver",
    1.00,
    0.60,
    0.00,
    DefaultTransformAnimation(
        use_translate=True,
        translate=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        use_rotate=False,
    ),
)
prismatic2_follower = make_follower("prismatic2_follower", 1.20, 0.60, 0.00)
solver.add_prismatic_joint(
    prismatic2_driver,
    prismatic2_follower,
    anchor,
    anchor,
    np.array([0.0, 1.0, 0.0], dtype=np.float32),  # slide axis = Y (gravity direction)
    stiffness_pos=5.0e4,
    stiffness_rot=1.0e4,
    slide_min=-0.5,
    slide_max=0.5,
)

# Scene F: Revolute joint (hinge = Z, driver rotates around X)
# -> follower position follows driver's X-rotation, but Z-twist remains free.
revolute2_driver = make_animated_driver(
    "revolute2_driver",
    2.00,
    0.60,
    0.00,
    DefaultTransformAnimation(
        use_rotate=True,
        rot_axis=[1.0, 0.0, 0.0],  # rotate around X
        rot_ang_vel_deg=90.0,
    ),
)
revolute2_follower = make_follower("revolute2_follower", 2.20, 0.60, 0.00)
solver.add_revolute_joint(
    revolute2_driver,
    revolute2_follower,
    anchor,
    anchor,
    np.array([0.0, 0.0, 1.0], dtype=np.float32),  # world hint
    np.array([0.0, 0.0, 1.0], dtype=np.float32),  # hinge axis = Z (body-A local)
    np.array([0.0, 0.0, 1.0], dtype=np.float32),  # hinge axis = Z (body-B local)
    stiffness_pos=5.0e4,
    stiffness_axis=2.0e3,
)

solver.init_solver()
config_ref = solver.get_config()
config_ref.set_use_floor(False)
config_ref.set_use_self_collision(False)
config_ref.set_gravity(lcs.Float3(0.0, -9.0, 0.0))
config_ref.set_use_gpu(False)

output_dir = os.path.join(PROJECT_ROOT, "Resources", "OutputMesh")
os.makedirs(output_dir, exist_ok=True)

tracked_ids = [
    fixed_driver,
    fixed_follower,
    prismatic_driver,
    prismatic_follower,
    revolute_driver,
    revolute_follower,
    fixed2_driver,
    fixed2_follower,
    prismatic2_driver,
    prismatic2_follower,
    revolute2_driver,
    revolute2_follower,
]
rest_vertices = {bid: get_vertices(bid) for bid in tracked_ids}
rest_centers = {bid: np.mean(rest_vertices[bid], axis=0) for bid in tracked_ids}

rest_rel_fixed = rest_centers[fixed_follower] - rest_centers[fixed_driver]
rest_rel_prismatic = rest_centers[prismatic_follower] - rest_centers[prismatic_driver]
rest_rel_revolute = rest_centers[revolute_follower] - rest_centers[revolute_driver]
rest_rel_fixed2 = rest_centers[fixed2_follower] - rest_centers[fixed2_driver]
rest_rel_prismatic2 = rest_centers[prismatic2_follower] - rest_centers[prismatic2_driver]
rest_rel_revolute2 = rest_centers[revolute2_follower] - rest_centers[revolute2_driver]


def update_animation():
    for animator in animators:
        animator.update_animation(solver, config_ref.get_current_frame(), config_ref.get_implicit_dt())


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
    # The sliding axis is body-local [1,0,0]; it co-rotates with the driver.
    # Constraint: (p_B - p_A) = A*(d0_perp_local + s*axis_local), so in local frame
    # (R.T @ current_relative - d0_perp_local) must lie along axis_local and s must
    # stay inside the configured slide range.
    prismatic_driver_rot = estimate_rotation_matrix(
        rest_vertices[prismatic_driver], prismatic_driver_vertices
    )
    prismatic_driver_yaw = estimate_yaw_z(rest_vertices[prismatic_driver], prismatic_driver_vertices)
    prismatic_axis_local = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    prismatic_axis_local = prismatic_axis_local / float(np.linalg.norm(prismatic_axis_local))
    rest_axis_distance = float(np.dot(rest_rel_prismatic, prismatic_axis_local))
    rest_rel_prismatic_perp = rest_rel_prismatic - rest_axis_distance * prismatic_axis_local
    current_relative = prismatic_follower_center - prismatic_driver_center
    current_relative_local = prismatic_driver_rot.T @ current_relative
    dev_local = current_relative_local - rest_rel_prismatic_perp
    prismatic_free_frac = float(np.dot(dev_local, prismatic_axis_local))
    prismatic_locked_vec = dev_local - prismatic_free_frac * prismatic_axis_local
    prismatic_locked_error = float(np.linalg.norm(prismatic_locked_vec))
    prismatic_axis_distance = float(np.dot(current_relative_local, prismatic_axis_local))
    prismatic_rest_axis_distance = rest_axis_distance
    prismatic_axis_distance_delta = prismatic_axis_distance - prismatic_rest_axis_distance

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

    # --- Scene D: Fixed2 (translation-only driver) ---
    fixed2_driver_vertices = get_vertices(fixed2_driver)
    fixed2_follower_vertices = get_vertices(fixed2_follower)
    fixed2_driver_center = np.mean(fixed2_driver_vertices, axis=0)
    fixed2_follower_center = np.mean(fixed2_follower_vertices, axis=0)
    # Driver doesn't rotate, so expected relative = rest relative (identity rotation).
    fixed2_rel_delta = (fixed2_follower_center - fixed2_driver_center) - rest_rel_fixed2
    fixed2_pos_error = float(np.linalg.norm(fixed2_rel_delta))
    # Follower should not rotate either.
    fixed2_follower_yaw = estimate_yaw_z(rest_vertices[fixed2_follower], fixed2_follower_vertices)
    fixed2_yaw_error = abs(wrap_angle_rad(fixed2_follower_yaw))
    # Verify driver actually translated significantly.
    fixed2_driver_motion = float(np.linalg.norm(fixed2_driver_center - rest_centers[fixed2_driver]))

    # --- Scene E: Prismatic2 (Y-axis slide, driver translates along X) ---
    prismatic2_driver_vertices = get_vertices(prismatic2_driver)
    prismatic2_follower_vertices = get_vertices(prismatic2_follower)
    prismatic2_driver_center = np.mean(prismatic2_driver_vertices, axis=0)
    prismatic2_follower_center = np.mean(prismatic2_follower_vertices, axis=0)
    # Driver doesn't rotate, so local frame = world frame.
    prismatic2_axis_local = np.array([0.0, 1.0, 0.0])
    prismatic2_current_rel = prismatic2_follower_center - prismatic2_driver_center
    prismatic2_rest_axis_dist = float(np.dot(rest_rel_prismatic2, prismatic2_axis_local))
    prismatic2_rest_perp = rest_rel_prismatic2 - prismatic2_rest_axis_dist * prismatic2_axis_local
    prismatic2_dev = prismatic2_current_rel - prismatic2_rest_perp
    prismatic2_slide = float(np.dot(prismatic2_dev, prismatic2_axis_local))
    prismatic2_locked_vec = prismatic2_dev - prismatic2_slide * prismatic2_axis_local
    prismatic2_locked_error = float(np.linalg.norm(prismatic2_locked_vec))
    prismatic2_axis_distance = float(np.dot(prismatic2_current_rel, prismatic2_axis_local))
    prismatic2_axis_delta = prismatic2_axis_distance - prismatic2_rest_axis_dist
    # Verify driver translated along X.
    prismatic2_driver_x_motion = float(prismatic2_driver_center[0] - rest_centers[prismatic2_driver][0])

    # --- Scene F: Revolute2 (hinge=Z, driver rotates around X) ---
    revolute2_driver_vertices = get_vertices(revolute2_driver)
    revolute2_follower_vertices = get_vertices(revolute2_follower)
    revolute2_driver_center = np.mean(revolute2_driver_vertices, axis=0)
    revolute2_follower_center = np.mean(revolute2_follower_vertices, axis=0)
    revolute2_driver_rot = estimate_rotation_matrix(rest_vertices[revolute2_driver], revolute2_driver_vertices)
    # Position should follow driver's rotation.
    expected_revolute2_rel = revolute2_driver_rot @ rest_rel_revolute2
    revolute2_rel_delta = (revolute2_follower_center - revolute2_driver_center) - expected_revolute2_rel
    revolute2_pos_error = float(np.linalg.norm(revolute2_rel_delta))
    # Z-twist should be free: measure yaw difference.
    revolute2_driver_yaw = estimate_yaw_z(rest_vertices[revolute2_driver], revolute2_driver_vertices)
    revolute2_follower_yaw = estimate_yaw_z(rest_vertices[revolute2_follower], revolute2_follower_vertices)
    revolute2_relative_yaw = abs(wrap_angle_rad(revolute2_follower_yaw - revolute2_driver_yaw))
    # Verify driver actually rotated around X (roll/pitch changed).
    # Standard ZYX Euler: roll (X-rotation) = arctan2(R[2,1], R[2,2]).
    revolute2_driver_pitch = abs(float(np.arctan2(revolute2_driver_rot[2, 1], revolute2_driver_rot[2, 2])))

    return {
        "fixed_pos_error": fixed_pos_error,
        "fixed_yaw_error": fixed_yaw_error,
        "prismatic_rel_delta": prismatic_rel_delta,
        "prismatic_driver_motion": prismatic_driver_motion,
        "prismatic_locked_error": prismatic_locked_error,
        "prismatic_free_frac": prismatic_free_frac,
        "prismatic_axis_distance": prismatic_axis_distance,
        "prismatic_rest_axis_distance": prismatic_rest_axis_distance,
        "prismatic_axis_distance_delta": prismatic_axis_distance_delta,
        "prismatic_driver_yaw_abs": abs(prismatic_driver_yaw),
        "revolute_driver_yaw_abs": abs(revolute_driver_yaw),
        "revolute_relative_yaw": revolute_relative_yaw,
        "revolute_pos_error": revolute_pos_error,
        # Scene D
        "fixed2_pos_error": fixed2_pos_error,
        "fixed2_yaw_error": fixed2_yaw_error,
        "fixed2_driver_motion": fixed2_driver_motion,
        # Scene E
        "prismatic2_locked_error": prismatic2_locked_error,
        "prismatic2_slide": prismatic2_slide,
        "prismatic2_axis_distance": prismatic2_axis_distance,
        "prismatic2_axis_delta": prismatic2_axis_delta,
        "prismatic2_driver_x_motion": prismatic2_driver_x_motion,
        # Scene F
        "revolute2_pos_error": revolute2_pos_error,
        "revolute2_relative_yaw": revolute2_relative_yaw,
        "revolute2_driver_pitch": revolute2_driver_pitch,
    }


def validate_metrics(metrics):
    assert metrics["fixed_pos_error"] < 2.0e-3, (
        f"Fixed joint failed position lock: error={metrics['fixed_pos_error']:.6e}"
    )
    assert metrics["fixed_yaw_error"] < 5.0e-2, (
        f"Fixed joint failed orientation lock: yaw error={metrics['fixed_yaw_error']:.6f} rad"
    )

    assert metrics["prismatic_driver_yaw_abs"] > 5.0e-2, (
        f"Prismatic driver did not rotate enough: yaw={metrics['prismatic_driver_yaw_abs']:.6f} rad"
    )
    assert metrics["prismatic_locked_error"] < 1.0e-2, (
        f"Prismatic locked-plane drift too large in driver local frame: error={metrics['prismatic_locked_error']:.6f}"
    )
    assert 0.1 <= metrics["prismatic_axis_distance"] <= 0.3, (
        f"Prismatic slide coordinate out of range [0.1, 0.3]: s={metrics['prismatic_axis_distance']:.6f}"
    )

    assert metrics["revolute_driver_yaw_abs"] > 5.0e-2, (
        f"Revolute driver did not rotate enough: yaw={metrics['revolute_driver_yaw_abs']:.6f} rad"
    )
    assert metrics["revolute_relative_yaw"] > 1.0e-1, (
        f"Revolute free-twist behavior failed: relative yaw={metrics['revolute_relative_yaw']:.6f} rad"
    )
    assert metrics["revolute_pos_error"] < 1.5e-1, (
        f"Revolute position coupling too weak: rel-position error={metrics['revolute_pos_error']:.6f}"
    )

    # --- Scene D: Fixed2 (translation driver) ---
    assert metrics["fixed2_driver_motion"] > 0.05, (
        f"Fixed2 driver did not translate enough: motion={metrics['fixed2_driver_motion']:.6f}"
    )
    assert metrics["fixed2_pos_error"] < 2.0e-3, (
        f"Fixed2 joint failed position lock under translation: error={metrics['fixed2_pos_error']:.6e}"
    )
    assert metrics["fixed2_yaw_error"] < 5.0e-2, (
        f"Fixed2 joint failed orientation lock (should stay zero): yaw={metrics['fixed2_yaw_error']:.6f} rad"
    )

    # --- Scene E: Prismatic2 (Y-axis slide under gravity, driver translates X) ---
    assert metrics["prismatic2_driver_x_motion"] > 0.05, (
        f"Prismatic2 driver did not translate enough along X: dx={metrics['prismatic2_driver_x_motion']:.6f}"
    )
    assert metrics["prismatic2_locked_error"] < 1.0e-2, (
        f"Prismatic2 locked-plane drift too large: error={metrics['prismatic2_locked_error']:.6f}"
    )
    # Gravity pulls follower down (negative Y), so slide delta should be negative.
    assert metrics["prismatic2_axis_delta"] < -0.01, (
        f"Prismatic2 follower did not slide under gravity: delta={metrics['prismatic2_axis_delta']:.6f}"
    )
    assert -0.5 <= metrics["prismatic2_axis_distance"] <= 0.5, (
        f"Prismatic2 slide out of limit range [-0.5, 0.5]: s={metrics['prismatic2_axis_distance']:.6f}"
    )

    # --- Scene F: Revolute2 (hinge=Z, driver rotates around X) ---
    assert metrics["revolute2_driver_pitch"] > 5.0e-2, (
        f"Revolute2 driver did not pitch enough: pitch={metrics['revolute2_driver_pitch']:.6f} rad"
    )
    assert metrics["revolute2_pos_error"] < 1.5e-1, (
        f"Revolute2 position coupling too weak: error={metrics['revolute2_pos_error']:.6f}"
    )
    assert metrics["revolute2_relative_yaw"] > 1.0e-2, (
        f"Revolute2 Z-twist should be free but follower tracked driver: rel_yaw={metrics['revolute2_relative_yaw']:.6f} rad"
    )


def print_metrics(metrics):
    print("[joint-check] fixed_pos_error        =", f"{metrics['fixed_pos_error']:.6e}")
    print("[joint-check] fixed_yaw_error(rad)   =", f"{metrics['fixed_yaw_error']:.6e}")
    print("[joint-check] prismatic_rel_delta    =", metrics["prismatic_rel_delta"])
    print("[joint-check] prismatic_driver_move  =", metrics["prismatic_driver_motion"])
    print("[joint-check] prismatic_locked_err   =", f"{metrics['prismatic_locked_error']:.6e}")
    print("[joint-check] prismatic_free_frac    =", f"{metrics['prismatic_free_frac']:.6f}")
    print("[joint-check] prismatic_axis_dist    =", f"{metrics['prismatic_axis_distance']:.6f}")
    print("[joint-check] prismatic_axis_rest    =", f"{metrics['prismatic_rest_axis_distance']:.6f}")
    print("[joint-check] prismatic_axis_delta   =", f"{metrics['prismatic_axis_distance_delta']:.6f}")
    print("[joint-check] prismatic_driver_yaw   =", f"{metrics['prismatic_driver_yaw_abs']:.6e}")
    print("[joint-check] revolute_driver_yaw    =", f"{metrics['revolute_driver_yaw_abs']:.6e}")
    print("[joint-check] revolute_relative_yaw  =", f"{metrics['revolute_relative_yaw']:.6e}")
    print("[joint-check] revolute_pos_error     =", f"{metrics['revolute_pos_error']:.6e}")
    print("[joint-check] --- Scene D: Fixed2 (translation) ---")
    print("[joint-check] fixed2_pos_error       =", f"{metrics['fixed2_pos_error']:.6e}")
    print("[joint-check] fixed2_yaw_error       =", f"{metrics['fixed2_yaw_error']:.6e}")
    print("[joint-check] fixed2_driver_motion   =", f"{metrics['fixed2_driver_motion']:.6f}")
    print("[joint-check] --- Scene E: Prismatic2 (Y-slide, X-translate) ---")
    print("[joint-check] prismatic2_locked_err  =", f"{metrics['prismatic2_locked_error']:.6e}")
    print("[joint-check] prismatic2_slide       =", f"{metrics['prismatic2_slide']:.6f}")
    print("[joint-check] prismatic2_axis_dist   =", f"{metrics['prismatic2_axis_distance']:.6f}")
    print("[joint-check] prismatic2_axis_delta  =", f"{metrics['prismatic2_axis_delta']:.6f}")
    print("[joint-check] prismatic2_driver_x    =", f"{metrics['prismatic2_driver_x_motion']:.6f}")
    print("[joint-check] --- Scene F: Revolute2 (hinge=Z, pitch=X) ---")
    print("[joint-check] revolute2_pos_error    =", f"{metrics['revolute2_pos_error']:.6e}")
    print("[joint-check] revolute2_rel_yaw      =", f"{metrics['revolute2_relative_yaw']:.6e}")
    print("[joint-check] revolute2_driver_pitch =", f"{metrics['revolute2_driver_pitch']:.6e}")


from utils.test_runner import TestRunner


class JointCheckTest(TestRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_pre_step(self, _frame_idx):
        update_animation()

    def on_post_step(self, _frame_idx):
        metrics = compute_metrics()
        print_metrics(metrics)

    def run(self, advance_frames):
        super().run(advance_frames)
        if self.headless:
            metrics = compute_metrics()
            print_metrics(metrics)
            validate_metrics(metrics)
            print("[joint-check] PASS")


runner = JointCheckTest(solver, config_ref, output_dir, headless=args.headless)
runner.run(advance_frames=args.advance_frames)

solver.cleanup_device()
