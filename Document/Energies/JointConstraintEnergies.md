# Joint Constraint Energies

This note documents the unified rigid joint constraint implementation in:

- `Solver/Energies/detail/fixed_joint_constaint.hpp`
- `Solver/Energies/detail/prismatic_joint_constaint.hpp`
- `Solver/Energies/detail/revolute_joint_constaint.hpp`
- `Solver/Energies/joint_constraint_energy.cpp`

The current implementation is a **body-local rest formulation**.

## 1. ABD State and Notation

For two rigid bodies A and B, the 8 ABD blocks are:

$$
\mathbf{q}=(q_0,q_1,q_2,q_3,q_4,q_5,q_6,q_7),
$$

where

- $q_0, q_4 \in \mathbb{R}^3$: body origins,
- $A=[q_1\ q_2\ q_3]$, $B=[q_5\ q_6\ q_7]$.

World position of a body-local point $r$:

$$
p_A(r)=q_0 + A r, \qquad p_B(r)=q_4 + B r.
$$

## 2. Stored Rest Data (Body-Local)

Per joint, initialization computes and stores:

- `anchor_a_local`, `anchor_b_local`
- `rest_position_delta` as
  $$
  d_0^A = A_0^{-1}\bigl(p_{B0}(r_b)-p_{A0}(r_a)\bigr)
  $$
- `rest_rot_col0/1/2_a_to_b` as columns of
  $$
  R_{ab}^0 = A_0^{-1}B_0
  $$

So the runtime target relation is body-local:

- positional target: $A d_0^A$
- orientation target (fixed/prismatic): $B = A R_{ab}^0$.

## 3. Fixed Joint

Residuals:

$$
r_{pos} = \bigl(p_B(r_b)-p_A(r_a)\bigr) - A d_0^A,
$$

$$
r_{rot,j} = q_{5+j} - A\,c_j,\quad j\in\{0,1,2\},
$$

where $c_j$ is column $j$ of $R_{ab}^0$.

Energy:

$$
E_{fixed} = \frac{k_{pos}}{2}\|r_{pos}\|^2 + \frac{k_{rot}}{2}\sum_{j=0}^2 \|r_{rot,j}\|^2.
$$

This locks full relative pose in body-local rest sense.

## 4. Prismatic Joint

Let $n=\text{normalize}(axis_{world})$, and

$$
P = I - nn^T.
$$

Residuals:

$$
r_{pos} = P\Bigl(\bigl(p_B(r_b)-p_A(r_a)\bigr)-A d_0^A\Bigr),
$$

$$
r_{rot,j} = q_{5+j} - A\,c_j,\quad j\in\{0,1,2\}.
$$

Energy:

$$
E_{prismatic} = \frac{k_{pos}}{2}\|r_{pos}\|^2 + \frac{k_{rot}}{2}\sum_{j=0}^2 \|r_{rot,j}\|^2.
$$

Interpretation:

- translation perpendicular to $n$ is locked,
- translation along $n$ is free,
- relative orientation is locked to rest.

So prismatic is **not** fixed: it keeps one translational DOF (along axis).

## 5. Revolute Joint

Residuals:

$$
r_{pos} = \bigl(p_B(r_b)-p_A(r_a)\bigr) - A d_0^A,
$$

$$
r_{axis}=A\,a_{local}-B\,b_{local},
$$

where $a_{local}=$ `axis_a_local`, $b_{local}=$ `axis_b_local`.

Energy:

$$
E_{revolute}=\frac{k_{pos}}{2}\|r_{pos}\|^2 + \frac{k_{axis}}{2}\|r_{axis}\|^2.
$$

Interpretation:

- anchor position is fully locked (3 DOF),
- the two hinge axes must coincide in world space,
- twist around the shared hinge axis is free.

Note: `axis_world` is not used in the revolute energy expression; revolute uses only `axis_a_local` and `axis_b_local`.

## 6. Gradient and Hessian Structure

All residuals are linear in ABD unknowns, so each term has exact quadratic form:

$$
E=\frac{s}{2}\|Cq+b\|^2,\quad \nabla E=sC^T(Cq+b),\quad \nabla^2E=sC^TC.
$$

Therefore joint gradient/Hessian are exact analytic results from linear coefficients (constant Hessian for each constraint term).

## 7. Parameter Mapping

- Fixed: `stiffness.x = k_pos`, `stiffness.y = k_rot`
- Prismatic: `stiffness.x = k_pos`, `stiffness.y = k_rot`
- Revolute: `stiffness.x = k_pos`, `stiffness.y = k_axis`

## 8. DOF Summary

- Fixed: lock 6, free 0
- Prismatic: lock 5, free 1 (translation along `axis_world`)
- Revolute: lock 5, free 1 (rotation around hinge axis)

## 9. Validation

Behavior validation script:

- `PythonBindings/tests/test_rigid_joint_animation.py`

Headless run example:

```bash
<LCS_PYTHON_EXECUTABLE> PythonBindings/tests/test_rigid_joint_animation.py --headless --advance_frames 30
```

The script checks fixed lock, prismatic free-axis translation, and revolute free twist numerically.

## 10. Migration Guide (World-Space -> Body-Local)

This section summarizes how the current body-local formulation differs from the previous world-space style expressions.

### 10.1 Core Variable Migration

Old idea (world-space target):

- Position target was directly a world delta or direct anchor coincidence without body-local rest transport.
- Orientation lock was often expressed as direct column equality (`q_{1+k} - q_{5+k}`), which implies identity rest relative rotation.

New idea (body-local target):

- Store rest relative position in body-A local frame:
  $$
  d_0^A = A_0^{-1}\bigl(p_{B0}(r_b)-p_{A0}(r_a)\bigr)
  $$
  and enforce runtime target with `A * d_0^A`.
- Store rest relative rotation:
  $$
  R_{ab}^0 = A_0^{-1}B_0
  $$
  and enforce `B - A * R_{ab}^0` (for fixed/prismatic orientation lock).

### 10.2 Per-Joint Formula Mapping

| Joint | Old-style residual (typical) | New residual (implemented) | Effect |
|---|---|---|---|
| Fixed | $r_{pos}=p_B(r_b)-p_A(r_a)$, $r_{rot,j}=q_{5+j}-q_{1+j}$ | $r_{pos}=(p_B(r_b)-p_A(r_a)) - A d_0^A$, $r_{rot,j}=q_{5+j}-A c_j$ | Locks full relative pose around non-identity rest relation |
| Prismatic | $r_{pos}=P(p_B-p_A)$, $r_{rot,j}=q_{5+j}-q_{1+j}$ | $r_{pos}=P((p_B(r_b)-p_A(r_a))-A d_0^A)$, $r_{rot,j}=q_{5+j}-A c_j$ | Preserves 1 translational DOF along axis while respecting rest relative pose |
| Revolute | Often world-axis projection/alignment form | $r_{pos}=(p_B(r_b)-p_A(r_a)) - A d_0^A$, $r_{axis}=A a_{local}-B b_{local}$ | Preserves 1 rotational DOF (twist) with body-local hinge-axis consistency |

Here $c_j$ is column $j$ of $R_{ab}^0$.

### 10.3 Why This Migration Matters

- If initial relative orientation is not identity, direct `q_{1+k} - q_{5+k}` will over-constrain toward identity and create bias.
- If initial anchor relation is not world-fixed, omitting `A d_0^A` causes rest-target inconsistency when body A rotates.
- Body-local rest storage makes constraint objectives invariant to the initial global placement and better matches articulated-joint semantics.

### 10.4 Migration Checklist (Code Review)

- Initialization (`init_sim_data.cpp`):
  - confirm `rest_position_delta = A0^{-1}(p_b0 - p_a0)`
  - confirm `rest_rot_col*` from `R_ab0 = A0^{-1}B0`
- Data struct / upload:
  - confirm `rest_rot_col0/1/2_a_to_b` exists in `JointConstraint`
  - confirm CPU->GPU upload includes all three columns
- Energy-only path (`joint_constraint_energy.cpp`):
  - fixed/prismatic use `target_delta = A * d0_local`
  - fixed/prismatic orientation use `B - A * R_ab0`
  - revolute uses `A * axis_a_local - B * axis_b_local`
- Eval path (`detail/*joint*_constaint.hpp` + host eval):
  - `compute_energy(...)` and `evaluate(...)` use identical residual definitions
- Behavior test:
  - run `PythonBindings/tests/test_rigid_joint_animation.py --headless --advance_frames 30`
