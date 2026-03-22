# Stable Neo-Hookean Energy

This note summarizes the formulas used by
[`stable_neo_hookean_energy.hpp`](/Users/huohuo/Desktop/UntanglingProject/Codes/ForMerge/LuisaComputeSimulator/Solver/Energies/detail/stable_neo_hookean_energy.hpp).

## 1. Deformation Gradient

For a tetrahedron with current positions `x0, x1, x2, x3`,

$$D_s = [x_1 - x_0, x_2 - x_0, x_3 - x_0], \quad F = D_s D_m^{-1}.$$

The rest volume is

$$V = \frac{|\det(D_m)|}{6}.$$

## 2. Energy Density

The stable Neo-Hookean density used in this project is

$$\psi(F) = \frac{\mu}{2}(\lVert F \rVert_F^2 - 3) - \mu (J - 1) + \frac{\lambda}{2}(J - 1)^2,$$

where

$$J = \det(F).$$

The tet energy is

$$E(x) = V \psi(F(x)).$$

## 2.1 Material Parameters

In most simulation setups, the Lamé parameters are computed from Young's modulus $E$ and Poisson ratio $\nu$:

$$\mu = \frac{E}{2(1+\nu)}, \quad \lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}.$$

This is the 3D conversion used by
[`FemUtils::convert_lame_params_3d(...)`](/Users/huohuo/Desktop/UntanglingProject/Codes/ForMerge/LuisaComputeSimulator/Solver/Energies/detail/fem_utils.h).

So the parameter pipeline is

$$ (E, \nu) \rightarrow (\mu, \lambda) \rightarrow \psi(F). $$

## 3. Determinant Derivatives

Define the cofactor matrix

$$\frac{\partial J}{\partial F} = \operatorname{cof}(F).$$

In column form,

$$\operatorname{cof}(F) = \begin{bmatrix} F_1 \times F_2 & F_2 \times F_0 & F_0 \times F_1 \end{bmatrix},$$

where $F_k$ is column $k$ of $F$.

Also define

$$c = \lambda (J - 1) - \mu.$$

## 4. First Piola Stress / Gradient wrt F

The derivative of the density wrt $F$ is

$$\frac{\partial \psi}{\partial F} = \mu F + c \operatorname{cof}(F).$$

This is the first Piola-Kirchhoff stress:

$$P = \mu F + c \operatorname{cof}(F).$$

## 5. Hessian wrt F

The exact Hessian wrt flattened $F$ is

$$\frac{\partial^2 \psi}{\partial F^2} = \mu I_9 + \lambda \mathrm{vec}(\operatorname{cof}(F)) \mathrm{vec}(\operatorname{cof}(F))^T + c \frac{\partial^2 J}{\partial F^2}.$$

Using the 3 block-columns of $F$, the determinant Hessian has the block form

$$\frac{\partial^2 J}{\partial F^2} = \begin{bmatrix} 0 & -[F_2]_\times & [F_1]_\times \\ [F_2]_\times & 0 & -[F_0]_\times \\ -[F_1]_\times & [F_0]_\times & 0 \end{bmatrix},$$

where the skew matrix is

$$[v]_\times = \begin{bmatrix} 0 & -v_z & v_y \\ v_z & 0 & -v_x \\ -v_y & v_x & 0 \end{bmatrix}.$$

Therefore the exact `F`-space Hessian used in code is

$$H_F = \mu I_9 + \lambda g_J g_J^T + c H_J,$$

with

$$g_J = \mathrm{vec}(\operatorname{cof}(F)).$$

## 6. Chain Rule to Vertex Space

Because $F$ depends linearly on the 12 tet DOFs,

$$\mathrm{vec}(F) = B x,$$

where

$$B = \frac{\partial \mathrm{vec}(F)}{\partial x}.$$

In code this is `FemUtils::get_dFdx(dm_inv)`.

Then

$$\nabla_x E = V B^T \mathrm{vec}(P),$$

and

$$\nabla_x^2 E = V B^T H_F B.$$

## 7. PSD Projection Used by Host Evaluation

The host-side `evaluate(...)` path symmetrizes the assembled $12 \times 12$ Hessian and projects it to PSD:

First, after assembling the vertex-space Hessian

$$H_x = V B^T H_F B,$$

we form its symmetric part:

$$H_{sym} = \frac{1}{2}(H_x + H_x^T).$$

Then we compute the eigendecomposition of this real symmetric matrix:

$$H_{sym} = Q \Lambda Q^T.$$

Here:

- $Q$ is the orthogonal matrix whose columns are the eigenvectors of $H_{sym}$
- $\Lambda$ is the diagonal matrix of eigenvalues
- since $H_{sym}$ is symmetric, these eigenvectors can be chosen orthonormal, so $Q^T Q = I$

So the `Q` in this section is not a constitutive quantity from the SNHK material model itself. It is the eigenvector matrix obtained by diagonalizing the symmetrized $12 \times 12$ Hessian in vertex space.

$$H_x^{PSD} = Q \max(\Lambda, 0) Q^T,$$

where

$$H_{sym} = Q \Lambda Q^T.$$

This matches the unit-test comparison strategy, which compares against a finite-difference Hessian after PSD projection.
