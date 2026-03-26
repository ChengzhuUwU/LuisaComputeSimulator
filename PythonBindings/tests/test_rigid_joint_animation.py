"""
test_rigid_joint_animation.py

Three spatially separated scenes, each targeting a distinct joint type:

  Scene 1 (z=0.00) — Fixed joint
    Driver translates along X (the prismatic slide axis in scene 2).
    All 6 DOF locked → follower is dragged in X, X gap stays constant.
    Verify: `center diff x` remains ~constant in print_info output.

  Scene 2 (z=0.25) — Prismatic joint  (slide axis = X)
    Same driver X translation as scene 1.
    X is the FREE axis → follower is NOT dragged; X gap grows each frame.
    Y/Z and rotation are still constrained (follower does not drift in Y/Z).
    Verify: `center diff x` grows while `center diff y` stays ~constant.

  Scene 3 (z=0.50) — Revolute joint  (rotation axis = Z)
    Driver is pinned in place. Follower starts displaced from the driver anchor.
    Rotation around Z is free → follower swings like a pendulum under gravity.
    Translation away from the driver anchor is resisted.
    Verify: distance between centers stays ~constant while follower angle changes.
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


def make_animated_driver(name, tx, ty, tz, transform: DefaultTransformAnimation):
    """Kinematic body with all vertices fixed; driven each frame via BodyAnimator."""
    body = solver.create_world_data_from_array(name, cube_mesh.vertices, cube_mesh.faces)
    body.set_simulation_type(lcs.MaterialType.Rigid)
    body.set_scale(SCALE)
    body.set_translation(tx, ty, tz)
    anim = BodyAnimator(body)
    anim.add_rule_by_method(body, "All", transform)
    body_id = solver.register_world_data(body)
    anim.set_mesh_index(body_id)
    animators.append(anim)
    return body_id


def make_static_driver(name, tx, ty, tz):
    """Kinematic body pinned in place for the entire simulation (no animation)."""
    body = solver.create_world_data_from_array(name, cube_mesh.vertices, cube_mesh.faces)
    body.set_simulation_type(lcs.MaterialType.Rigid)
    body.set_scale(SCALE)
    body.set_translation(tx, ty, tz)
    body.add_fixed_point_by_method("All")
    return solver.register_world_data(body)


def make_follower(name, tx, ty, tz):
    """Free dynamic rigid body; subject to gravity and joint forces only."""
    body = solver.create_world_data_from_array(name, cube_mesh.vertices, cube_mesh.faces)
    body.set_simulation_type(lcs.MaterialType.Rigid)
    body.set_scale(SCALE)
    body.set_translation(tx, ty, tz)
    return solver.register_world_data(body)


anchor = np.zeros(3, dtype=np.float32)

# ==========================================================================
# Scene 1 — Fixed joint  (z = 0.0)
# Driver translates along X. All 6 DOF are locked → follower is dragged in X,
# maintaining the initial X gap. Verify: center diff x stays ~constant.
# ==========================================================================
z1 = 0.0
driver_fixed = make_animated_driver(
    "fixed_driver", 0.15, 0.20, z1,
    DefaultTransformAnimation(
        use_translate=True, translate=[0.15, 0.0, 0.0],  # move along X
        # use_rotate=True, rot_axis=[0.0, 0.0, 1.0], rot_ang_vel_deg=45.0
    ),
)
follower_fixed = make_follower("fixed_follower", -0.05, 0.20, z1)
solver.add_fixed_joint(
    driver_fixed, follower_fixed, anchor, anchor,
    stiffness_pos=2.0e4, stiffness_rot=4.0e3,
)

# ==========================================================================
# Scene 2 — Prismatic joint, slide axis = X  (z = 0.25)
# Same driver X translation as scene 1. X is the FREE axis → follower is NOT
# dragged; it stays put while the driver slides away. X gap grows each frame.
# Y/Z displacement is still constrained: follower does not drift in Y or Z.
# Verify: center diff x grows; center diff y stays ~constant.
# ==========================================================================
z2 = 0.25
driver_prismatic = make_animated_driver(
    "prismatic_driver", 0.25, 0.20, z2,
    DefaultTransformAnimation(
        use_translate=True, translate=[0.15, 0.0, 0.0]  # same X motion as scene 1
        # use_rotate=True, rot_axis=[0.0, 0.0, 1.0], rot_ang_vel_deg=45.0
        )
)
follower_prismatic = make_follower("prismatic_follower", -0.05, 0.20, z2)
solver.add_prismatic_joint(
    driver_prismatic, follower_prismatic, anchor, anchor,
    np.array([1.0, 0.0, 0.0], dtype=np.float32),  # slide axis = X
    stiffness_pos=2.0e4, stiffness_rot=4.0e3,
)

# ==========================================================================
# Scene 3 — Revolute joint, rotation axis = Z  (z = 0.50)
# Driver is pinned in place. Follower starts displaced from the driver.
# Rotation around Z is free → follower swings like a pendulum under gravity.
# Translation away from driver anchor is resisted.
# ==========================================================================
z3 = 0.50
driver_revolute = make_static_driver("revolute_driver", 0.05, 0.35, z3)
follower_revolute = make_follower("revolute_follower", 0.15, 0.20, z3)
solver.add_revolute_joint(
    driver_revolute, follower_revolute, anchor, anchor,
    np.array([0.0, 0.0, 1.0], dtype=np.float32),  # world axis
    np.array([0.0, 0.0, 1.0], dtype=np.float32),  # axis in body A local frame
    np.array([0.0, 0.0, 1.0], dtype=np.float32),  # axis in body B local frame
    stiffness_pos=2.0e4, stiffness_axis=2.0e3,
)

# --------------------------------------------------------------------------
solver.init_solver()
config_ref = solver.get_config()
config_ref.use_floor = True
output_dir = os.path.join(root, "Resources", "OutputMesh")
os.makedirs(output_dir, exist_ok=True)


def update_animation():
    for anim in animators:
        anim.update_animation(solver, config_ref.current_frame, config_ref.implicit_dt)

def print_transform(label, body_id):
    t = solver.get_rigid_body_translation(body_id)
    s = solver.get_rigid_body_scaling(body_id)
    q = solver.get_rigid_body_rotation_quaternion(body_id)
    rv = solver.get_rigid_body_rotation_axis_angle(body_id)  # rotation vector = axis * angle(rad)
    rv_np = np.array(rv, dtype=np.float32)
    angle_rad = np.linalg.norm(rv_np)
    angle_deg = np.degrees(angle_rad)
    if angle_rad > 1e-8:
        axis = rv_np / angle_rad
    else:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    q_norm = np.linalg.norm(np.array(q, dtype=np.float32))
    print(
        f"  {label}: pos=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f})  "
        f"scale=({s[0]:.3f},{s[1]:.3f},{s[2]:.3f})  "
        f"quat=({q[0]:.3f},{q[1]:.3f},{q[2]:.3f},{q[3]:.3f}|n={q_norm:.3f})  "
        f"axis=({axis[0]:.2f},{axis[1]:.2f},{axis[2]:.2f})  angle={angle_deg:.1f}°"
    )


def print_info():
    def centers(reg_id):
        pos, _ = solver.get_object_sim_result_by_registration_id(reg_id)
        return np.mean(pos, axis=0)  # (x, y, z)

    # Scene 1 — Fixed: X gap should stay ~constant as driver translates along X.
    ca = centers(driver_fixed);   cb = centers(follower_fixed)
    dx = cb[0] - ca[0];           dy = cb[1] - ca[1]
    print(f"[fixed]     driver x={ca[0]:.3f}  follower x={cb[0]:.3f}  diff_x={dx:.3f} (should stay ~-0.20)")
    print_transform("  fixed_driver  ", driver_fixed)
    print_transform("  fixed_follower", follower_fixed)

    # Scene 2 — Prismatic (slide=X): X gap should grow as driver translates along X.
    ca = centers(driver_prismatic); cb = centers(follower_prismatic)
    dx = cb[0] - ca[0];             dy = cb[1] - ca[1]
    print(f"[prismatic] driver x={ca[0]:.3f}  follower x={cb[0]:.3f}  diff_x={dx:.3f} (should grow more negative)  diff_y={dy:.3f} (should stay ~0)")
    print_transform("  prismatic_driver  ", driver_prismatic)
    print_transform("  prismatic_follower", follower_prismatic)

    # Scene 3 — Revolute: distance between centers should stay ~constant (pendulum).
    ca = centers(driver_revolute); cb = centers(follower_revolute)
    dist = np.linalg.norm(cb - ca)
    print(f"[revolute]  center dist={dist:.3f} (should stay ~constant)  follower pos=({cb[0]:.3f}, {cb[1]:.3f})")
    print_transform("  revolute_driver  ", driver_revolute)
    print_transform("  revolute_follower", follower_revolute)

if args.headless:
    solver.save_sim_result(os.path.join(output_dir, "joint_demo_init.obj"))
    for _ in range(args.advance_frames):
        update_animation()
        if config_ref.use_gpu:
            solver.physics_step_gpu()
        else:
            solver.physics_step_cpu()
        print_info()
    solver.save_sim_result(os.path.join(output_dir, "joint_demo_result.obj"))
else:
    import utils.polyscope_gui

    class JointDemoGUI(utils.polyscope_gui.SimulationGUI):
        def _physics_step(self):
            update_animation()
            super()._physics_step()
            print_info()

    gui = JointDemoGUI(solver, config_ref, output_dir)
    gui.show()

solver.cleanup_device()
