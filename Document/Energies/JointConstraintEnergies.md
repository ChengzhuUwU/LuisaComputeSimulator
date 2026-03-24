# Joint Constraint Energies

This document covers the three joint constraint energies implemented in
`Solver/Energies/detail/`:

- `fixed_joint_constaint.hpp` — **Fixed Joint**
- `prismatic_joint_constaint.hpp` — **Prismatic Joint**
- `revolute_joint_constaint.hpp` — **Revolute Joint**

All three use the **ABD (Affine Body Dynamics)** representation.  
Each rigid body is parameterized by 4 "column vectors" $(p, a_0, a_1, a_2) \in \mathbb{R}^{3\times4}$:

$$
\mathbf{q}^{(A)} = (q_0, q_1, q_2, q_3), \qquad
\mathbf{q}^{(B)} = (q_4, q_5, q_6, q_7)
$$

where $q_0, q_4 \in \mathbb{R}^3$ are the body origins and $q_1, q_2, q_3$ (resp. $q_5, q_6, q_7$) are the columns of the deformation/rotation matrix $A$ (resp. $B$).

The world-space position of a local point $r$ on body $A$ is:

$$
p_A(r) = q_0 + q_1 r_x + q_2 r_y + q_3 r_z
$$

---

## 1. Fixed Joint

**Files:** `fixed_joint_energy.h / .cpp`, `detail/fixed_joint_constaint.hpp`

### 1.1 Constraint Description

A fixed joint locks **both the relative position and relative orientation** of two bodies.  
Two penalty terms are summed:

| Term | Residual | Meaning |
|------|----------|---------|
| Position | $r_\text{pos} = p_B(r^B) - p_A(r^A)$ | Anchor points must coincide |
| Orientation (row $k$) | $r_k = q_{1+k} - q_{5+k}$ | Column $k$ of $A$ equals column $k$ of $B$ |

### 1.2 Energy

$$
E_\text{fixed} = \frac{k_\text{pos}}{2} \|r_\text{pos}\|^2 + \frac{k_\text{rot}}{2} \sum_{k=0}^{2} \|r_k\|^2
$$

where

$$
r_\text{pos} = (q_4 + q_5 r^B_x + q_6 r^B_y + q_7 r^B_z) - (q_0 + q_1 r^A_x + q_2 r^A_y + q_3 r^A_z)
$$

### 1.3 Gradient and Hessian

All constraints are **linear in $\mathbf{q}$**, so the energy is a quadratic form:

$$
E = \frac{s}{2} \|\mathbf{C}\mathbf{q} + b\|^2, \quad
\nabla_{\mathbf{q}} E = s\, \mathbf{C}^\top(\mathbf{C}\mathbf{q} + b), \quad
\nabla^2_{\mathbf{q}} E = s\, \mathbf{C}^\top \mathbf{C}
$$

Each linear constraint contributes coefficient matrices $C_i \in \mathbb{R}^{3\times3}$ for each DOF block $q_i$.  
The implementation uses the shared `add_linear_term` kernel:

```
r = bias + Σ_i  coeff[i] * q[i]
grad[i]       += stiffness * coeff[i] * r
hessian[i][j] += stiffness * coeff[i] * coeff[j]   (outer product of 3×3 matrices)
```

**Position term** — coefficient layout $(i = 0 \dots 7)$:

$$
C = \bigl(-I,\ -r^A_x I,\ -r^A_y I,\ -r^A_z I,\ I,\ r^B_x I,\ r^B_y I,\ r^B_z I\bigr)
$$

**Orientation term for row $k$** — only DOF blocks $1+k$ and $5+k$ are non-zero:

$$
C_{1+k} = I, \quad C_{5+k} = -I
$$

The full Hessian is an $8\times8$ block matrix (each block is $3\times3$), hence 64 blocks stored in the `EnergyEvalResult<8,64,...>`.

---

## 2. Prismatic Joint

**Files:** `prismatic_joint_energy.h / .cpp`, `detail/prismatic_joint_constaint.hpp`

### 2.1 Constraint Description

A prismatic joint allows **sliding along one axis** $\hat{n}$ (world-space) but locks all other relative motion.  
Define the **plane projector** orthogonal to $\hat{n}$:

$$
P = I - \hat{n}\hat{n}^\top, \qquad \hat{n} = \frac{n_\text{world}}{\|n_\text{world}\|}
$$

| Term | Residual | Meaning |
|------|----------|---------|
| Position (in-plane) | $r_\text{pos} = P(p_B(r^B) - p_A(r^A))$ | Relative displacement perpendicular to axis must be zero |
| Orientation (row $k$) | $r_k = q_{1+k} - q_{5+k}$ | Relative orientation locked (no rotation) |

### 2.2 Energy

$$
E_\text{prismatic} = \frac{k_\text{pos}}{2} \|P(p_B - p_A)\|^2 + \frac{k_\text{rot}}{2} \sum_{k=0}^{2} \|q_{1+k} - q_{5+k}\|^2
$$

### 2.3 Gradient and Hessian

The structure is identical to the Fixed Joint, with the single change that $I$ in the position coefficient block is replaced by $P$:

$$
C_\text{pos} = \bigl(-P,\ -r^A_x P,\ -r^A_y P,\ -r^A_z P,\ P,\ r^B_x P,\ r^B_y P,\ r^B_z P\bigr)
$$

The orientation term is unchanged.  
Because $P$ is a constant (given a fixed world axis), the Hessian is still **exact and constant** (no second-order correction needed).

> **Note:** The energy computation in the `.cpp` shader uses the equivalent direct form  
> $r_\text{pos} = d - \hat{n}(\hat{n} \cdot d)$ where $d = p_B - p_A$, which avoids an explicit matrix multiply on the GPU.

---

## 3. Revolute Joint

**Files:** `revolute_joint_energy.h / .cpp`, `detail/revolute_joint_constaint.hpp`

### 3.1 Constraint Description

A revolute joint allows **rotation around one hinge axis** while locking the anchor position and forcing the local hinge axes of both bodies to align with the world hinge axis $\hat{n}$.

Let

$$
a_A = q_1 (\alpha_x) + q_2 (\alpha_y) + q_3 (\alpha_z), \qquad
a_B = q_5 (\beta_x) + q_6 (\beta_y) + q_7 (\beta_z)
$$

be the hinge axis in world space, where $\alpha = $ `axis_a_local` and $\beta = $ `axis_b_local`.

| Term | Residual | Meaning |
|------|----------|---------|
| Position | $r_\text{pos} = p_B(r^B) - p_A(r^A)$ | Anchor points coincide (full 3-DOF lock) |
| Axis A alignment | $r_{aA} = P\, a_A$ | Body A's hinge axis ∥ world axis |
| Axis B alignment | $r_{aB} = P\, a_B$ | Body B's hinge axis ∥ world axis |

### 3.2 Energy

$$
E_\text{revolute} = \frac{k_\text{pos}}{2} \|p_B - p_A\|^2 + \frac{k_\text{axis}}{2} \|P\, a_A\|^2 + \frac{k_\text{axis}}{2} \|P\, a_B\|^2
$$

### 3.3 Gradient and Hessian

**Position term** — same coefficient layout as Fixed Joint:

$$
C_\text{pos} = \bigl(-I,\ -r^A_x I,\ -r^A_y I,\ -r^A_z I,\ I,\ r^B_x I,\ r^B_y I,\ r^B_z I\bigr)
$$

**Axis-A alignment term** — non-zero only in body-A rotation blocks:

$$
C_{1} = \alpha_x P, \quad C_{2} = \alpha_y P, \quad C_{3} = \alpha_z P, \quad \text{all others } = 0
$$

**Axis-B alignment term** — non-zero only in body-B rotation blocks:

$$
C_{5} = \beta_x P, \quad C_{6} = \beta_y P, \quad C_{7} = \beta_z P, \quad \text{all others } = 0
$$

Again the Hessian is exact-quadratic (no nonlinear terms), assembled via the same `add_linear_term` kernel.

> **Key difference from Prismatic:** The revolute joint enforces the *full* 3-DOF positional lock (uses $I$ not $P$ for the position term), but relaxes the *rotational* constraint — instead of locking all three orientation columns, it only requires that each body's designated hinge axis aligns with the shared world axis.

---

## 4. Parameter Summary

| Joint | Inputs beyond `indices_a/b`, `anchor_a/b`, `stiffness` | `stiffness.x` | `stiffness.y` |
|-------|--------------------------------------------------------|---------------|---------------|
| Fixed | — | $k_\text{pos}$ | $k_\text{rot}$ |
| Prismatic | `axis_world` | $k_\text{pos}$ | $k_\text{rot}$ |
| Revolute | `axis_world`, `axis_a_local`, `axis_b_local` | $k_\text{pos}$ | $k_\text{axis}$ |

---

## 5. Degrees of Freedom Comparison

| Joint | Locked DOF | Free DOF |
|-------|-----------|---------|
| Fixed | 6 (3 translation + 3 rotation) | 0 |
| Prismatic | 5 (2 translation ⊥ axis + 3 rotation) | 1 (translation ∥ axis) |
| Revolute | 4 (3 translation + 1 rotation ⊥ hinge) | 2 (rotation ∥ hinge per body, but shared → effectively 1 DOF rotation) |
