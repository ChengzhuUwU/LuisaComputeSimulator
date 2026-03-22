# Hookean Spring Energy

This note summarizes the formulas used by
[`hookean_spring_energy.hpp`](/Users/huohuo/Desktop/UntanglingProject/Codes/ForMerge/LuisaComputeSimulator/Solver/Energies/detail/hookean_spring_energy.hpp).

## 1. Energy

For two vertices `x0` and `x1`, define

$$d = x_0 - x_1, \quad \ell = \|d\|, \quad c = \ell - \ell_0.$$

where `ell_0` is the rest length.

The spring energy is

$$E(x_0,x_1) = \frac{1}{2} kc^2,$$

with $k$ is the spring stiffness.

In the implementation, `l` is clamped from below by `eps = 1e-8` to avoid division by zero.

## 2. Gradient

Let

$$n = \frac{d}{\ell}.$$

Then

$$\frac{\partial E}{\partial x_0} = kc n, \quad \frac{\partial E}{\partial x_1} = -kc n.$$ 

This is `g0` and `g1` in `evaluate(...)`.

## 3. Hessian


The code uses the PSD spring Hessian

$$H_e = kn n^T + kw_t (I - n n^T),$$

where

$$w_t = \max\left(1 - \frac{\ell_0}{\ell}, 0\right).$$

The full `2 x 2` block Hessian is

$$\begin{bmatrix} H_e & -H_e \\ -H_e & H_e \end{bmatrix}.$$

So tangential stiffness is clamped to be non-negative by the `max(...)` in `tangent_weight`.
