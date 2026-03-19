# Energy Models

This document describes the constitutive energy models used in LuisaComputeSimulator for physics simulation.

## Overview

The simulator uses an **energy-based formulation** where the physics is described by minimizing a total energy functional:

$$E_{total} = E_{internal} + E_{external}$$

Where internal energies model material behavior and external energies handle constraints.

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

**Implementation:** `SpringEnergy` in `Solver/Energies/spring_energy.cpp`

#### Stable NeoHookean Energy (Finite Strain)

$$E = \frac{\mu}{2}(tr(C) - 3) + \frac{\lambda}{2}(\det(F) - 1)^2 - \mu \ln(\det(F))$$

Where:
- $F$ is the deformation gradient ($F = \frac{\partial x}{\partial X}$)
- $C = F^T F$ is the right Cauchy-Green tensor
- $\mu$ and $\lambda$ are Lamé parameters (derived from Young's modulus and Poisson's ratio)

**Characteristics:** 
- Stable for large deformations
- Physically-based hyperelastic material
- Volume preservation behavior

**Implementation:** `NeoHookeanEnergy` in `Solver/Energies/neohookean_energy.cpp`

#### ARAP (As-Rigid-As-Possible) Energy

$$E_{ARAP} = \sum_{ij} w_{ij} \| (p_i - p_j) - R_i (X_i - X_j) \|^2$$

Where $R_i$ is the rotation matrix that best aligns the deformed positions with rest positions locally.

**Characteristics:**
- Preserves local rigidity
- Good for shape matching applications
- Rotation-invariant energy formulation

**Implementation:** `ARAPEnergy` in `Solver/Energies/arap_energy.cpp`

---

### 2. Bending Energy

Models resistance to bending/folding:

$$E = \frac{k_b}{2} (\theta - \theta_0)^2$$

Where $\theta$ is the dihedral angle between adjacent faces.

**Implementation:** `BendingEnergy` in `Solver/Energies/bending_energy_kernel.cpp`

---

### 3. Face Stretch Energy

Energy based on triangle face deformation:

$$E = \frac{k}{2} (A - A_0)^2$$

Where $A$ is current area and $A_0$ is rest area.

**Implementation:** `StretchFaceEnergy` in `Solver/Energies/stretch_face_energy.cpp`

---

### 4. Inertia Energy

#### Soft Body Inertia

$$E_{inertia} = \frac{1}{2} m v^T v$$

For soft bodies, uses full-space formulation with per-vertex masses.

**Implementation:** `SoftInertiaEnergy` in `Solver/Energies/soft_inertia_energy.cpp`

#### Affine Body Dynamics (ABD) Inertia

For rigid bodies, uses reduced-space formulation:

$$E_{inertia} = \frac{1}{2} \dot{q}^T M \dot{q}$$

Where $q$ represents the reduced coordinates (translation + rotation).

**Implementation:** `AbdInertiaEnergy` in `Solver/Energies/abd_inertia_energy.cpp`

---

### 5. Orthogonality Energy

Ensures rigid body rotation matrices remain orthogonal:

$$E_{ortho} = \frac{k_o}{2} \|R^T R - I\|^2$$

**Implementation:** `AbdOrthoEnergy` in `Solver/Energies/abd_ortho_energy.cpp`

---

### 6. Contact Energy

#### Quadratic Contact

$$E_{contact} = \frac{1}{2} k (d - \hat{d})^2$$

Where:
- $d$ is penetration distance
- $\hat{d}$ is the barrier activation distance

#### Log-Barrier Contact

$$E_{contact} = -k (d - \hat{d})^2 \ln\left(\frac{d}{\hat{d}}\right)$$

More robust for thin objects.

---

## Affine Body Dynamics (ABD)

The simulator uses Affine Body Dynamics for efficient rigid body simulation.

### Concept

Rigid body motion is represented as an affine transformation:

$$x = R \overline{x} + t$$

Where:
- $\overline{x}$ is the position in model space
- $R$ is the rotation matrix
- $t$ is translation

### Jacobian

$$J = \frac{\partial x}{\partial q} = 
\begin{bmatrix}
I_3 & -\overline{x}_\times
\end{bmatrix}$$

Where $\overline{x}_\times$ is the skew-symmetric cross-product matrix.

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
    .model = ConstitutiveModelTet::Corotated,
    .youngs_modulus = 1e6f,
    .poisson_ratio = 0.4f
}
```

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
