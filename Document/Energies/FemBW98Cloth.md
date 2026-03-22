# FEM BW98 Cloth Energy

This note summarizes the formulas used by
[`fem_BW98_cloth_energy.hpp`](/Users/huohuo/Desktop/UntanglingProject/Codes/ForMerge/LuisaComputeSimulator/Solver/Energies/detail/fem_BW98_cloth_energy.hpp).

## 1. Deformation Gradient

For a triangle with current positions `x0, x1, x2`,

$$D_s = [x_1-x_0, x_2-x_0] \in \mathbb{R}^{3\times 2}, \quad F = D_s D_m^{-1}.$$

The rest area is $A$.

Write the two columns of `F` as

$$F = [F_u, F_v]$$

## 2. Energy Density

The implementation splits the membrane energy into stretch and shear parts:

$$\psi(F) = \psi_{\text{stretch}}(F) + \psi_{\text{shear}}(F).$$

### 2.1 Stretch term

$$\psi_{\text{stretch}}(F) = \frac{\mu}{2}\left[(\|F_u\|-1)^2 + (\|F_v\|-1)^2\right].$$

### 2.2 Shear term

$$\psi_{\text{shear}}(F) = \frac{\lambda}{2}(F_u \cdot F_v)^2.$$

The triangle energy is

$$E(x) = A \psi(F(x)).$$

## 3. Gradient wrt F

For the stretch part,

$$\frac{\partial \psi_{\text{stretch}}}{\partial F_u} = \mu \left(1 - \frac{1}{\|F_u\|}\right) F_u, \quad \frac{\partial \psi_{\text{stretch}}}{\partial F_v} = \mu \left(1 - \frac{1}{\|F_v\|}\right) F_v.$$

This is the matrix returned by `stretch_gradient(...)`.

For the shear part, let

$$w = F_u \cdot F_v.$$

Then

$$\frac{\partial \psi_{\text{shear}}}{\partial F_u} = \lambda w F_v, \quad \frac{\partial \psi_{\text{shear}}}{\partial F_v} = \lambda w F_u.$$

This is `shear_gradient(...)`.

## 4. Hessian wrt F

### 4.1 Stretch Hessian

The code builds a PSD form for each column block:

$$H_{\text{stretch},u} = \mu \left[\max\left(0, 1-\frac{1}{\|F_u\|}\right) I + \min\left(1,\frac{1}{\|F_u\|}\right) \hat{F}_u \hat{F}_u^T\right].$$

$$H_{\text{stretch},v} = \mu \left[\max\left(0, 1-\frac{1}{\|F_v\|}\right) I + \min\left(1,\frac{1}{\|F_v\|}\right) \hat{F}_v \hat{F}_v^T\right].$$

where

$$\hat{F}_u = \frac{F_u}{\|F_u\|}, \quad \hat{F}_v = \frac{F_v}{\|F_v\|}.$$

The off-diagonal stretch blocks are zero.

### 4.2 Shear Hessian

The shear Hessian is assembled in a projected PSD form from the invariant

$$I_6 = F_u \cdot F_v.$$

The code first constructs a symmetric coupling matrix `H` and then computes

$$\lambda_0 = \frac{1}{2}\left(I_2 + \sqrt{I_2^2 + 12 I_6^2}\right), \quad I_2 = \|F_u\|^2 + \|F_v\|^2.$$

followed by the PSD expression implemented in `shear_hessian(...)`.

So the project uses a stabilized Hessian rather than the raw exact second derivative of
`0.5 * lambda * (F_u \cdot F_v)^2`.

## 5. Chain Rule to Vertex Space

Because `F` depends linearly on the 9 vertex DOFs,

$$\mathrm{vec}(F) = Bx,$$

where `B = FemUtils::get_dFdx(dm_inv)`.

If

$$P = \frac{\partial \psi}{\partial F}, \quad H_F = \frac{\partial^2 \psi}{\partial F^2}.$$

then the assembled triangle gradient and Hessian are

$$\nabla_x E = A B^T \mathrm{vec}(P), \quad \nabla_x^2 E = A B^T H_F B.$$

This is exactly how `evaluate(...)` computes the `3 x 3` block output.
