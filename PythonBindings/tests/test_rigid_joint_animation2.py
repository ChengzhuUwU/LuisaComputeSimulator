from utils.test_script_path import PROJECT_ROOT
"""
Joint Constraint Showcase: Articulated Crane System.

Demonstrates all three joint types working together in a single mechanical system:

    [Ground] --(Revolute Y)--> [Turntable/Base]
              --(Revolute X)--> [Boom Arm]
              --(Prismatic along boom)--> [Trolley]
              --(Fixed)--> [Hook]
              --(Revolute Z)--> [Payload/Pendulum]

System description:
  - The turntable is an animated driver that rotates around Y (vertical axis).
  - The boom arm is attached to the turntable via a revolute joint (hinge = local X),
    allowing it to tilt up/down under gravity.
  - A trolley slides along the boom arm via a prismatic joint (slide axis = boom local Z),
    free to move along the boom length.
  - A hook is rigidly fixed to the trolley (fixed joint) — demonstrates rigid coupling
    in a chain.
  - A payload hangs from the hook via a revolute joint (hinge = local X),
    creating a pendulum that swings freely as the crane rotates.

The result is a dynamic system where:
  - The turntable's rotation creates centrifugal effects on the boom.
  - The boom tilts under gravity (limited by the revolute joint stiffness).
  - The trolley slides along the boom due to centrifugal + gravity components.
  - The payload swings as a pendulum, responding to all upstream motions.

Usage:
  GUI mode:   python test_rigid_joint_animation2.py
  Headless:   python test_rigid_joint_animation2.py --headless --advance_frames 300
"""

import os

import numpy as np
import trimesh

import lcs_py as lcs

import utils.arg_parser
from utils.animation_transform import DefaultTransformAnimation
from utils.body_animator import BodyAnimator

args = utils.arg_parser.parse_args()

solver = lcs.NewtonSolver()
solver.init_device(backend_name=args.backend)

cube_mesh_path = os.path.join(PROJECT_ROOT, "Resources", "InputMesh", "cube.obj")
cube_mesh = trimesh.load(cube_mesh_path, process=False)

animators = []
anchor = np.zeros(3, dtype=np.float32)


def make_body(name, tx, ty, tz, scale=0.10, is_driver=False, transform=None):
    """Create a rigid body cube at the given position."""
    body = solver.create_world_data_from_array(name, cube_mesh.vertices, cube_mesh.faces)
    body.set_simulation_type(lcs.MaterialType.Rigid)
    body.set_scale(scale)
    body.set_translation(tx, ty, tz)

    if is_driver and transform is not None:
        animator = BodyAnimator(body)
        animator.add_rule_by_method(body, "All", transform)
        body_id = solver.register_world_data(body)
        animator.set_mesh_index(body_id)
        animators.append(animator)
    else:
        body_id = solver.register_world_data(body)

    return body_id


# ============================================================
# Build the crane system
# ============================================================

# Heights and offsets
BASE_Y = 0.50
BOOM_OFFSET_Z = 0.20
TROLLEY_OFFSET_Z = 0.20
HOOK_OFFSET_Y = -0.15
PAYLOAD_OFFSET_Y = -0.15

# --- 1. Turntable (animated driver, rotates around Y) ---
turntable_id = make_body(
    "turntable",
    0.0, BASE_Y, 0.0,
    scale=0.12,
    is_driver=True,
    transform=DefaultTransformAnimation(
        use_rotate=True,
        rot_axis=[0.0, 1.0, 0.0],
        rot_ang_vel_deg=60.0,  # Slow rotation for visual clarity
    ),
)

# --- 2. Boom Arm (connected to turntable via revolute joint, hinge = X) ---
# Placed slightly in front of turntable along Z.
boom_id = make_body(
    "boom_arm",
    0.0, BASE_Y, BOOM_OFFSET_Z,
    scale=0.08,
)

# Revolute joint: turntable -> boom, hinge around X-axis.
# This allows the boom to tilt up/down (pitch) but not yaw or roll independently.
solver.add_revolute_joint(
    turntable_id,
    boom_id,
    anchor,  # anchor at turntable center
    anchor,  # anchor at boom center
    np.array([1.0, 0.0, 0.0], dtype=np.float32),  # axis_world hint
    np.array([1.0, 0.0, 0.0], dtype=np.float32),  # axis_a_local (turntable)
    np.array([1.0, 0.0, 0.0], dtype=np.float32),  # axis_b_local (boom)
    stiffness_pos=8.0e4,
    stiffness_axis=5.0e3,
)

# --- 3. Trolley (slides along boom via prismatic joint, axis = Z in boom local) ---
# Placed further along Z from the boom.
trolley_id = make_body(
    "trolley",
    0.0, BASE_Y, BOOM_OFFSET_Z + TROLLEY_OFFSET_Z,
    scale=0.06,
)

# Prismatic joint: boom -> trolley, slide along local Z (boom's forward direction).
solver.add_prismatic_joint(
    boom_id,
    trolley_id,
    anchor,
    anchor,
    np.array([0.0, 0.0, 1.0], dtype=np.float32),  # slide axis = Z (boom local)
    stiffness_pos=5.0e4,
    stiffness_rot=1.0e4,
    slide_min=0.05,
    slide_max=0.50,
)

# --- 4. Hook (rigidly attached to trolley via fixed joint) ---
# Placed below the trolley.
hook_id = make_body(
    "hook",
    0.0, BASE_Y + HOOK_OFFSET_Y, BOOM_OFFSET_Z + TROLLEY_OFFSET_Z,
    scale=0.04,
)

# Fixed joint: trolley -> hook (rigid coupling, no relative motion).
solver.add_fixed_joint(
    trolley_id,
    hook_id,
    anchor,
    anchor,
    stiffness_pos=1.0e5,
    stiffness_rot=2.0e4,
)

# --- 5. Payload / Pendulum (hangs from hook via revolute joint, hinge = X) ---
# Placed below the hook.
payload_id = make_body(
    "payload",
    0.0, BASE_Y + HOOK_OFFSET_Y + PAYLOAD_OFFSET_Y, BOOM_OFFSET_Z + TROLLEY_OFFSET_Z,
    scale=0.07,
)

# Revolute joint: hook -> payload, hinge around X-axis.
# This creates a pendulum that can swing in the YZ plane (relative to hook).
solver.add_revolute_joint(
    hook_id,
    payload_id,
    anchor,
    anchor,
    np.array([1.0, 0.0, 0.0], dtype=np.float32),  # axis_world hint
    np.array([1.0, 0.0, 0.0], dtype=np.float32),  # axis_a_local (hook)
    np.array([1.0, 0.0, 0.0], dtype=np.float32),  # axis_b_local (payload)
    stiffness_pos=5.0e4,
    stiffness_axis=1.0e3,
)

# ============================================================
# Solver configuration
# ============================================================

solver.init_solver()
config_ref = solver.get_config()
config_ref.set_use_floor(False)
config_ref.set_use_self_collision(False)
config_ref.set_gravity(lcs.Float3(0.0, -9.8, 0.0))
config_ref.set_use_gpu(False)

output_dir = os.path.join(PROJECT_ROOT, "Resources", "OutputMesh")
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# Tracking setup
# ============================================================

body_ids = {
    "turntable": turntable_id,
    "boom": boom_id,
    "trolley": trolley_id,
    "hook": hook_id,
    "payload": payload_id,
}


def get_vertices(registration_id):
    verts, _ = solver.get_object_sim_result_by_registration_id(registration_id)
    return np.asarray(verts, dtype=np.float64)


def get_center(registration_id):
    return np.mean(get_vertices(registration_id), axis=0)


rest_centers = {name: get_center(bid) for name, bid in body_ids.items()}


def update_animation():
    for animator in animators:
        animator.update_animation(solver, config_ref.current_frame, config_ref.implicit_dt)


def physics_step():
    update_animation()
    if config_ref.use_gpu:
        solver.physics_step_gpu()
    else:
        solver.physics_step_cpu()


def compute_showcase_metrics():
    """Compute metrics that demonstrate the system is behaving as a coupled mechanism."""
    centers = {name: get_center(bid) for name, bid in body_ids.items()}

    # Turntable rotation (should be driven).
    turntable_motion = np.linalg.norm(centers["turntable"] - rest_centers["turntable"])

    # Boom tilt: Y-displacement from rest (gravity pulls it down).
    boom_y_drop = rest_centers["boom"][1] - centers["boom"][1]

    # Trolley slide: distance from boom center along the boom's current forward direction.
    trolley_to_boom = centers["trolley"] - centers["boom"]
    trolley_distance = float(np.linalg.norm(trolley_to_boom))

    # Hook should track trolley rigidly (fixed joint).
    # Compare distance (magnitude) since the pair rotates together.
    hook_to_trolley = centers["hook"] - centers["trolley"]
    hook_rest_offset = rest_centers["hook"] - rest_centers["trolley"]
    rest_distance = float(np.linalg.norm(hook_rest_offset))
    current_distance = float(np.linalg.norm(hook_to_trolley))
    hook_distance_drift = abs(current_distance - rest_distance)

    # Payload pendulum: angular deviation from vertical.
    payload_to_hook = centers["payload"] - centers["hook"]
    payload_vertical_component = payload_to_hook[1]  # should be negative (hanging)
    payload_horizontal = np.sqrt(payload_to_hook[0] ** 2 + payload_to_hook[2] ** 2)
    payload_swing_angle = float(np.arctan2(payload_horizontal, abs(payload_vertical_component)))

    # Overall chain connectivity: turntable -> boom -> trolley -> hook -> payload
    # All should have moved from rest (system is coupled).
    all_moved = all(
        np.linalg.norm(centers[name] - rest_centers[name]) > 1e-4
        for name in ["boom", "trolley", "hook", "payload"]
    )

    return {
        "turntable_motion": float(turntable_motion),
        "boom_y_drop": float(boom_y_drop),
        "trolley_distance_from_boom": trolley_distance,
        "hook_drift_from_trolley": hook_distance_drift,
        "payload_swing_angle_rad": payload_swing_angle,
        "all_bodies_moved": all_moved,
        "centers": centers,
    }


def print_showcase_metrics(metrics):
    print("=" * 60)
    print("  CRANE SYSTEM SHOWCASE METRICS")
    print("=" * 60)
    print(f"  Turntable motion from rest   : {metrics['turntable_motion']:.4f} m")
    print(f"  Boom Y-drop (gravity tilt)   : {metrics['boom_y_drop']:.4f} m")
    print(f"  Trolley distance from boom    : {metrics['trolley_distance_from_boom']:.4f} m")
    print(f"  Hook drift from trolley       : {metrics['hook_drift_from_trolley']:.6f} m  (fixed joint)")
    print(f"  Payload swing angle           : {np.degrees(metrics['payload_swing_angle_rad']):.2f} deg")
    print(f"  All downstream bodies moved   : {metrics['all_bodies_moved']}")
    print("=" * 60)

    # Print body positions for visualization reference.
    print("\n  Body positions (world):")
    for name, center in metrics["centers"].items():
        print(f"    {name:12s}: ({center[0]:+.4f}, {center[1]:+.4f}, {center[2]:+.4f})")
    print()


def validate_showcase(metrics):
    """Sanity checks: the system should behave as a coupled mechanism."""
    # Turntable must have moved (it's animated).
    assert metrics["turntable_motion"] > 0.01, (
        f"Turntable did not move: {metrics['turntable_motion']:.6f}"
    )
    # All downstream bodies should have moved from rest (coupled chain).
    assert metrics["all_bodies_moved"], "Not all bodies moved — chain coupling broken"
    # Hook should stay close to trolley (fixed joint integrity).
    assert metrics["hook_drift_from_trolley"] < 5.0e-2, (
        f"Hook drifted from trolley (fixed joint broken): {metrics['hook_drift_from_trolley']:.6f}"
    )
    # Payload should be hanging (swing angle < 90 deg means it hasn't flipped).
    assert metrics["payload_swing_angle_rad"] < np.pi / 2, (
        f"Payload flipped above horizontal: angle={np.degrees(metrics['payload_swing_angle_rad']):.1f} deg"
    )


# ============================================================
# Main execution
# ============================================================

if args.headless:
    solver.save_sim_result(os.path.join(output_dir, "crane_showcase_init.obj"))

    print(f"Running crane showcase for {args.advance_frames} frames...")
    for frame_idx in range(args.advance_frames):
        physics_step()
        # Print progress every 50 frames.
        if (frame_idx + 1) % 50 == 0:
            m = compute_showcase_metrics()
            print(f"\n--- Frame {frame_idx + 1} ---")
            print_showcase_metrics(m)

    solver.save_sim_result(os.path.join(output_dir, "crane_showcase_result.obj"))

    print("\n--- FINAL STATE ---")
    metrics = compute_showcase_metrics()
    print_showcase_metrics(metrics)
    validate_showcase(metrics)
    print("[crane-showcase] PASS")

else:
    import utils.polyscope_gui

    class CraneShowcaseGUI(utils.polyscope_gui.SimulationGUI):
        def _physics_step(self):
            physics_step()
            if config_ref.current_frame % 10 == 0:
                metrics = compute_showcase_metrics()
                print_showcase_metrics(metrics)

    gui = CraneShowcaseGUI(solver, config_ref, output_dir)
    gui.show()

solver.cleanup_device()
