# Bending Energy

This note summarizes the formulas used by
[`bending_energy.hpp`](/Users/huohuo/Desktop/UntanglingProject/Codes/ForMerge/LuisaComputeSimulator/Solver/Energies/detail/bending_energy.hpp).

## 1. Configuration

The energy is defined on two adjacent triangles sharing edge `(x0, x1)`,
with opposite vertices `x2` and `x3`.

The code evaluates the signed dihedral angle

$$\theta = \theta(x_0,x_1,x_2,x_3).$$

and compares it against a rest angle `theta_rest`.

## 2. Energy

The scalar energy is

$$E(x) = \frac{1}{2} k (\theta - \theta_{\text{rest}})^2.$$ 

where $k$ is the bending stiffness.

## 3. Signed Dihedral Angle

The code first forms the two face normals

$$n_1 = (x_1-x_0)\times(x_2-x_0), \quad n_2 = (x_2-x_3)\times(x_1-x_3).$$

Then

$$\theta = s \arccos\left(\frac{n_1\cdot n_2}{\|n_1\|\|n_2\|}\right).$$

with sign

$$s = \operatorname{sign}\left((n_2 \times n_1)\cdot(x_1-x_2)\right).$$

This is exactly `face_dihedral_angle(...)`.

## 4. Angle Gradient

The implementation uses the standard hinge-angle gradient.

Define

$$e_0 = x_1 - x_0, \quad e_1 = x_2 - x_0, \quad e_2 = x_3 - x_0, \quad e_3 = x_2 - x_1, \quad e_4 = x_3 - x_1.$$

and

$$n_1 = e_0 \times e_1, \quad n_2 = e_2 \times e_0.$$

Then the code computes

$$\frac{\partial \theta}{\partial x_2} = -\frac{\|e_0\|}{\|n_1\|^2} n_1.$$

$$\frac{\partial \theta}{\partial x_1} = -\frac{e_0\cdot e_3}{\|e_0\|\|n_1\|^2} n_1 - \frac{e_0\cdot e_4}{\|e_0\|\|n_2\|^2} n_2.$$

$$\frac{\partial \theta}{\partial x_0} = \frac{e_0\cdot e_1}{\|e_0\|\|n_1\|^2} n_1 + \frac{e_0\cdot e_2}{\|e_0\|\|n_2\|^2} n_2.$$

$$\frac{\partial \theta}{\partial x_3} = -\frac{\|e_0\|}{\|n_2\|^2} n_2.$$

These are the formulas in `face_dihedral_angle_grad(...)`, up to the index remapping done by `compute_d_theta_d_x(...)`.

## 5. Gradient and Hessian

Let

$$\Delta \theta = \theta - \theta_{\text{rest}}.$$

The gradient is

$$\frac{\partial E}{\partial x_i} = \kappa \Delta \theta \frac{\partial \theta}{\partial x_i}.$$ 

The implementation uses a Gauss-Newton style Hessian:

$$\frac{\partial^2 E}{\partial x_i \partial x_j} \approx \kappa \frac{\partial \theta}{\partial x_i}\left(\frac{\partial \theta}{\partial x_j}\right)^T.$$ 

So the second derivative of the angle itself is ignored, and only the outer-product term is kept.
