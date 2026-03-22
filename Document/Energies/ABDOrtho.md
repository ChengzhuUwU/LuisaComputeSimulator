# ABD Orthogonality Energy

This note summarizes the formulas used by
[`abd_ortho_energy.hpp`](/Users/huohuo/Desktop/UntanglingProject/Codes/ForMerge/LuisaComputeSimulator/Solver/Energies/detail/abd_ortho_energy.hpp).

## 1. Variable

Let

$$A = [a_0, a_1, a_2] \in \mathbb{R}^{3\times 3}$$

be the affine matrix of an ABD body, written by columns.

## 2. Energy

The code penalizes deviation from orthonormal columns:

$$E(A) = \kappa \|A^T A - I\|_F^2.$$

where $\kappa \geq 10^6$ is the rigid body material stiffness.

Expanded in column form,

$$E(A) = \kappa \sum_{i=0}^2 \sum_{j=0}^2 \left(a_i^T a_j - \delta_{ij}\right)^2.$$

This is exactly what `compute_energy(...)` accumulates.

## 3. Gradient

In matrix form,

$$\nabla_A E = 4\kappa A(A^T A - I).$$

Column-wise, the code uses

$$\frac{\partial E}{\partial a_i} = 4\kappa \left[-a_i + (a_i^T a_0)a_0 + (a_i^T a_1)a_1 + (a_i^T a_2)a_2\right].$$

This matches `g0`, `g1`, and `g2` in `evaluate(...)`.

## 4. Hessian Blocks

The block Hessian with respect to columns of `A` is

$$H_{ij} = 4\kappa \left[a_j a_i^T + (a_i^T a_j)I\right], \quad i \neq j.$$

and for diagonal blocks

$$H_{ii} = 4\kappa \left[a_i a_i^T + (\|a_i\|^2 - 1)I + \sum_{k=0}^2 a_k a_k^T\right].$$

These are the exact formulas assembled in the nested loop of `evaluate(...)`.
