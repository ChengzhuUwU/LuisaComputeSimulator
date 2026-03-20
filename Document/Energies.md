# Energy Models

This document describes the constitutive energy models used in LuisaComputeSimulator for physics simulation.

## Overview

The simulator uses an **energy-based formulation** where the physics is described by minimizing a total energy functional:

$$E_{total} = E_{internal} + E_{external}$$

Where internal energies model material behavior and external energies handle constraints.

The current soft-body pipeline supports **Spring** and **ARAP tetrahedral FEM** energies, and these soft constraints are assembled in the same global system as cloth and rigid-body energies to enable coupled simulation.

---

## Constitutive Models for Soft Bodies

### 1. Stretch Energy

Models the stretching resistance of cloth/soft bodies.

#### Spring Energy (Linear)

$$E = \frac{k}{2} (|p_i - p_j| - L_0)^2$$

Where:
- $k$ is stiffness
- $L_0$ is rest length
- $p_i, p_j$ are vertex positions

**Characteristics:** Simple, fast, but limited to small deformations.

**Implementation:** `stretch_spring_energy` in [`Solver/Energies/detail/stretch_spring_energy.hpp`](../Solver/Energies/detail/stretch_spring_energy.hpp)

#### Stable NeoHookean Energy (Tetrahedral FEM)

$$E = V\left[\frac{\mu}{2}(\mathrm{tr}(F^T F) - 3) - \mu(\det(F)-1) + \frac{\lambda}{2}(\det(F)-1)^2\right]$$

Where:
- $F$ is the deformation gradient ($F = \frac{\partial x}{\partial X}$)
- $C = F^T F$ is the right Cauchy-Green tensor
- $\mu$ and $\lambda$ are Lamé parameters (derived from Young's modulus and Poisson's ratio)

**Characteristics:**
- Stable for large deformations
- Physically-based hyperelastic material
- Volume preservation behavior

**Implementation:** `stable_neo_hookean_energy` in [`Solver/Energies/detail/stable_neo_hookean_energy.hpp`](../Solver/Energies/detail/stable_neo_hookean_energy.hpp)


#### ARAP (As-Rigid-As-Possible) Energy

$$E_{ARAP} = \mu V \|F - R\|_F^2$$

Where $F$ is deformation gradient, $R$ is the polar rotation extracted from $F$, $\mu$ is shear stiffness, and $V$ is tet rest volume.

**Characteristics:**
- Preserves local rigidity
- Good for shape matching applications
- Rotation-invariant energy formulation

**Implementation:** `arap_tet_energy` in [`Solver/Energies/detail/arap_tet_energy.hpp`](../Solver/Energies/detail/arap_tet_energy.hpp)

---

### 2. Bending Energy

Models resistance to bending/folding:

$$E = \frac{k_b}{2} (\theta - \theta_0)^2$$

Where $\theta$ is the dihedral angle between adjacent faces.

**Implementation:** `bending_energy` in [`Solver/Energies/detail/bending_energy.hpp`](../Solver/Energies/detail/bending_energy.hpp)

---

### 3. Face Stretch Energy (BW98-Style Membrane FEM)

Membrane energy on triangle faces using stretch and shear invariants.

Let $F=[F_u,F_v] \in \mathbb{R}^{3\times 2}$ be the 2D-to-3D deformation gradient on the face. The implementation uses:

$$E = A\left(E_{stretch} + E_{shear}\right)$$

$$E_{stretch} = \frac{\mu}{2}\left[(\|F_u\|-1)^2 + (\|F_v\|-1)^2\right]$$

$$E_{shear} = \frac{\lambda}{2}(F_u\cdot F_v)^2$$

Where $A$ is the rest-face area, and $(\mu,\lambda)$ are material parameters.

**Implementation:** `stretch_face_energy` in [`Solver/Energies/detail/stretch_face_energy.hpp`](../Solver/Energies/detail/stretch_face_energy.hpp)

---

### 4. Inertia Energy

#### Soft Body Inertia

$$E_{inertia} = \frac{1}{2h^2} m k_d \|x-\tilde{x}\|^2$$

For soft bodies, the implementation is a per-vertex implicit inertia term with Dirichlet scaling $k_d$ ($k_d=10^6$ for fixed points and $k_d=1$ for active points).

**Implementation:** `soft_inertia_energy` in [`Solver/Energies/detail/soft_inertia_energy.hpp`](../Solver/Energies/detail/soft_inertia_energy.hpp)

#### Affine Body Dynamics (ABD) Inertia

For rigid bodies, uses reduced-space formulation:

$$E_{inertia}=\frac{1}{2h^2} \|q-\tilde{q}\|_M^2$$

Where $\Delta_i$ are the 3D delta columns in ABD state, $M\in\mathbb{R}^{4\times 4}$ is the body mass matrix, and $\alpha$ is the scaled stiffness factor used by the solver.

**Implementation:** `abd_inertia_energy` in [`Solver/Energies/detail/abd_inertia_energy.hpp`](../Solver/Energies/detail/abd_inertia_energy.hpp)

---

### 5. Orthogonality Energy

Ensures rigid body affine matrix $A$ stays close to rotation (penalizes non-orthogonality):

$$E_{ortho} = \kappa V \|A^T A - I\|_F^2$$

Where $\kappa$ is the rigid stiffness, $V$ is the body's volume.

**Implementation:** `abd_ortho_energy` in [`Solver/Energies/detail/abd_ortho_energy.hpp`](../Solver/Energies/detail/abd_ortho_energy.hpp)

---

### 6. Contact Energy

#### Quadratic Contact

$$E_{contact} = \frac{1}{2} k (d - \hat{d})^2$$

Where:
- $d$ is penetration distance
- $\hat{d}$ is the barrier activation distance

#### Log-Barrier Contact

$$E_{contact} = -k (d - \hat{d})^2 \ln\left(\frac{d}{\hat{d}}\right)$$

Provide more forces in extremly contact.

#### Ground Collision + Friction (Implementation Mapping)

Ground-contact repulsion (quadratic / barrier switch) and friction energy-gradient-hessian are implemented in:

- `ground_collision_energy` in [`Solver/Energies/detail/ground_collision_energy.hpp`](../Solver/Energies/detail/ground_collision_energy.hpp)

This implementation provides:

- `repulsive_energy`, `repulsive_first_derivative`, `repulsive_second_derivative`
- `friction_energy`, `friction_gradient_hessian`

---

## Affine Body Dynamics (ABD)

The simulator uses Affine Body Dynamics for efficient rigid body simulation.

### Concept

Rigid body motion is represented as an affine transformation:

$$x = A \overline{x} + p$$

Where:
- $\overline{x}$ is the rest state position in model space
- $A = RS$ is the rotation and scaling matrix
- $p$ is translation

### Jacobian

$$ J = 
\begin{bmatrix}
1 & 0 & 0 & \overline{x}_1 & \overline{x}_2 & \overline{x}_3 & & & & & & \\
0 & 1 & 0 & & & & \overline{x}_1 & \overline{x}_2 & \overline{x}_3 & & & \\
0 & 0 & 1 & & & & & & & \overline{x}_1 & \overline{x}_2 & \overline{x}_3 \\
\end{bmatrix} \in R^{3 \times 12} $$

### Reduced Space

Using ABD, the system solves in a reduced space:

- **Soft bodies:** Full space ($J = I_3$)
- **Rigid bodies:** Reduced space with 12 DOF per body

---

## Configuration

### Cloth Material Parameters

```cpp
ClothMaterial{
    .stretch_model = ConstitutiveStretchModelCloth::Spring,      // or StableNeoHookean, ARAP
    .bending_model = ConstitutiveBendingModelCloth::Bending,
    .thickness = 0.001f,           // Thickness for collision
    .youngs_modulus = 1e6f,        // Stretch stiffness
    .poisson_ratio = 0.0f,         // Poisson effect (for NeoHookean)
    .area_bending_stiffness = 1e-5f  // Bending stiffness
}
```

### Available Stretch Models

| Model | Use Case | Parameters |
|-------|----------|------------|
| `Spring` | Basic cloth, real-time applications | youngs_modulus |
| `StableNeoHookean` | Large deformations, physically-based | youngs_modulus, poisson_ratio |
| `ARAP` | Shape matching, local rigidity | youngs_modulus |

### Tetrahedral Material Parameters

```cpp
TetMaterial{
    .model = ConstitutiveModelTet::ARAP, // or ConstitutiveModelTet::StableNeoHookean
    .youngs_modulus = 1e6f,
    .poisson_ratio = 0.4f
}
```

### Coupled System Assembly

Soft-body tetrahedral energies, cloth energies, rigid-body inertia/orthogonality, and contact energies are assembled into one Newton system. This enables direct coupling among soft bodies, cloth, and rigid bodies in the same simulation step.

---

## Mathematical Details

For VF/EE contact, we have barycentric weight $w \in R^4$, area weighted stiffness $k = \kappa a$, direction $n$ of shortest distance $d$ (With positions $x = [x_1^T, x_2^T, x_3^T, x_4^T]^T$).

$$d = || t || = || \sum_i^4 w_i x_i || $$

- For VF : $w_1 = 1, (w_1 + w_2 + w_3) = -1$
- For EE : $(w_1 + w_2) = 1, (w_2 + w_3) = -1$

Considering $w$ is constant, so we can have:

$$
\frac{\partial t}{\partial x} = 
\begin{bmatrix}
w_1 I_3, w_2 I_3, w_3 I_3, w_4 I_3
\end{bmatrix} \in R^{3 \times 12}  \quad \text{and} \quad
\frac{\partial^2 t}{\partial x^2} = 0
$$

> So this is Gauss-Newton, which result in problem in some configurations, but this is enough for most cases.

We can use different type of contact energy, include: 

- A **quadratic** formulation of energy $E = \frac{1}{2} k (d-\hat{d})^2$
- A **log-barrier** formulation of energy $E = -(d - \hat{d})^2 \ln (\frac{d}{\hat{d}})$ 
   - Or use Codimentional-IPC enhanced energy, which modeling the thickness $\epsilon$

Then we have:

$$
\frac{\partial E}{\partial x} = \frac{\partial E}{\partial d}  \frac{\partial d}{\partial t} \frac{\partial t}{\partial x} = \frac{\partial E}{\partial d} \frac{t^T}{d} \frac{\partial t}{\partial x}
 = \frac{\partial E}{\partial d} n^T \frac{\partial t}{\partial x}
$$

$$
\frac{\partial^2 E}{\partial x^2} = \frac{\partial^2 E}{\partial d^2} (n^T \frac{\partial t}{\partial x}) (n^T \frac{\partial t}{\partial x})^T
$$

### Contact Implentation

We set $k_1 = \partial E / \partial d$:
- For quadratic formulation: $k_1 = k (d-\hat{d})$
- For log-barrier formulation: $k_1 = (\hat{d} - d)(2 \ln (\frac{d}{\hat{d}}) - \frac{\hat{d}}{d} + 1 )$

And set $k_2 = \partial^2 E / \partial d^2$
- For quadratic formulation: $k_2 = k$
- For log-barrier formulation: $k_2 = (\frac{\hat{d}}{d} + 2)\frac{\hat{d}}{d} - 2\ln (\frac{d}{\hat d}) -3$

For $i$'s vertex in VF/EE pair:

$$ \nabla E_i = k_1 w_i n $$

And:

$$ \nabla E_{ij}^2 = k_2 w_i w_j n n^T $$





## Reduced System of Affine-Body-Dynamics 

A Jacobian matrix $J$ map the relation ship between position $x$ (of vertex) and state $q$ (of body) :

$$ \frac{\partial E}{\partial q} = \frac{\partial E}{\partial x} \frac{\partial x}{\partial q} = \frac{\partial E}{\partial x} J$$

$$ \frac{\partial^2 E}{\partial q_i \partial q_j} 
= (\frac{\partial x}{\partial q_j})^T  \frac{\partial^2 E}{\partial x^2} \frac{\partial x}{\partial q_i} + \cancel{\frac{\partial E}{\partial x} \frac{\partial^2 x}{\partial q_i \partial q_j}} 
= J_j^T \frac{\partial^2 E}{\partial x_i \partial x_j} J_i$$

We simplify the symbolic as: $\textcolor{red}{g} = \nabla E_{x_i}$, and $\textcolor{red}{H} = \nabla^2 E_{x_{i, j}}$:

$$\nabla E_{q_i} = J^T \nabla E_{x_i} = J^T \textcolor{red}{g}$$

$$\nabla E_{q_i, q_j}^2 = J_i^T \nabla^2 E_{x_i, x_j} J_j = J_i^T \textcolor{red}{H} J_j$$

For **Soft Body** (cloth, soft-body, rods...), we use full-space simulation:

$$J_s = I_3$$

For **Rigid (Affine) Body**, we use reduced-space simulation (Where $\overline{x}$ is the position in **model space**):

$$ J_r = 
\begin{bmatrix}
1 & 0 & 0 & \overline{x}_1 & \overline{x}_2 & \overline{x}_3 & & & & & & \\
0 & 1 & 0 & & & & \overline{x}_1 & \overline{x}_2 & \overline{x}_3 & & & \\
0 & 0 & 1 & & & & & & & \overline{x}_1 & \overline{x}_2 & \overline{x}_3 \\
\end{bmatrix} \in R^{3 \times 12} $$

So we can simplify the calculation. 

### For gradient

$$\nabla E_{q_i} = J^T \textcolor{red}{g} = 
\begin{bmatrix}
{g}
\\ {g}_{0} \overline{x} 
\\ {g}_{1} \overline{x} 
\\ {g}_{2} \overline{x} 
\end{bmatrix} \in R^{12}$$ 

Where $g_{i}$ is the *i*'s element in $g$.

### For hessian

For hessian $\nabla E_{q_i, q_j}^2 = J_i^T \nabla^2 E J_j$, we have 4 cases:

> $i,j$ are vertices from VF/EE Pair

---

(1) **Soft Vert - Soft Vert**, $J_i = I_3, J_j = I_3$ :

$$
\nabla^2 E_{q_i, q_j} = J_i^T \textcolor{red}{H} J_j = I_3^T \textcolor{red}{H} I_3
= H \in R^{3 \times 3}
$$

This is actullly what we do in full-space simulation.

---

(2) **Soft Vert - Rigid Vert**, $J_i = I_3 , J_j = J_r$ :

$$
\nabla^2 E_{q_i, q_j} = J_i^T \textcolor{red}{H} J_j = 
\begin{bmatrix}
H
& H_{:,1} \overline{x}_j^T
& H_{:,2} \overline{x}_j^T
& H_{:,3} \overline{x}_j^T
\end{bmatrix} \in R^{3 \times 12}
$$

Where $H_{:,j}$ is the *j*'th column in $H$.

---

(3) **Rigid Vert - Soft Vert**, $J_i = J_r, J_j = I_3$ :

$$
\nabla^2 E_{q_i, q_j} = J_i^T \textcolor{red}{H} J_j = 
\begin{bmatrix}
H
\\ \overline{x}_i H_{1,:}
\\ \overline{x}_i H_{2,:}
\\ \overline{x}_i H_{3,:}
\end{bmatrix} \in R^{12 \times 3}
$$

Where $H_{i,:}$ is the *i*'th row in $H$.

---

(4) **Rigid Vert - Rigid Vert**, $J_i = J_r , J_j = J_r$: 

$$
\nabla^2 E_{q_i, q_j} = J_i^T \textcolor{red}{H} J_j = 
\begin{bmatrix}
H 
& H_{:,1} \textcolor{red}{\overline{x}_j}^T    
& H_{:,2} \textcolor{red}{\overline{x}_j}^T    
& H_{:,3} \textcolor{red}{\overline{x}_j}^T 
\\
\textcolor{green}{\overline{x}_i} H_{1,:}
& H_{1,1} \textcolor{green}{\overline{x}_i} \textcolor{red}{\overline{x}_j}^T    
& H_{1,2} \textcolor{green}{\overline{x}_i} \textcolor{red}{\overline{x}_j}^T    
& H_{1,3} \textcolor{green}{\overline{x}_i} \textcolor{red}{\overline{x}_j}^T 
\\
\textcolor{green}{\overline{x}_i} H_{2,:}  
& H_{2,1} \textcolor{green}{\overline{x}_i} \textcolor{red}{\overline{x}_j}^T    
& H_{2,2} \textcolor{green}{\overline{x}_i} \textcolor{red}{\overline{x}_j}^T    
& H_{2,3} \textcolor{green}{\overline{x}_i} \textcolor{red}{\overline{x}_j}^T 
\\
\textcolor{green}{\overline{x}_i} H_{3,:}
& H_{3,1} \textcolor{green}{\overline{x}_i} \textcolor{red}{\overline{x}_j}^T    
& H_{3,2} \textcolor{green}{\overline{x}_i} \textcolor{red}{\overline{x}_j}^T    
& H_{3,3} \textcolor{green}{\overline{x}_i} \textcolor{red}{\overline{x}_j}^T  
\end{bmatrix} \in R^{12 \times 12}
$$
