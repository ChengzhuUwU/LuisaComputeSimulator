# ARAP Tet Energy

This note summarizes the formulas used by
[`arap_tet_energy.hpp`](/Users/huohuo/Desktop/UntanglingProject/Codes/ForMerge/LuisaComputeSimulator/Solver/Energies/detail/arap_tet_energy.hpp).

## 1. Deformation Gradient

For a tetrahedron with current positions `x0, x1, x2, x3`,

$$D_s = [x_1 - x_0, x_2 - x_0, x_3 - x_0], \quad F = D_s D_m^{-1}.$$

The rest volume is

$$V = \frac{|\det(D_m)|}{6}.$$

## 2. Energy Density

The code uses the tet ARAP density

$$\psi(F) = \|F - R(F)\|_F^2.$$

where `R(F)` is the rotation from the polar decomposition of `F`.

The tet energy is

$$E(x) = \mu V \psi(F(x)).$$

Although the input struct also stores `lambda`, this implementation only uses `mu`.

## 3. Polar Rotation

The rotation is extracted by SVD:

$$F = U \Sigma V^T, \quad R = U V^T.$$

This is what `polar_rotation(F)` computes in the code.

## 4. First Derivative wrt F

With the ARAP density written in this project, the code uses

$$\frac{\partial \psi}{\partial F} = 2(F - R).$$

So the first Piola-like derivative is

$$P = 2(F - R).$$

The vertex-space gradient then becomes

$$\nabla_x E = \mu V B^T \mathrm{vec}(P).$$

where

$$\mathrm{vec}(F) = Bx, \quad B = \frac{\partial \mathrm{vec}(F)}{\partial x}.$$

In code, `B` is `FemUtils::get_dFdx(dm_inv)`.

## 5. Hessian wrt F

The implementation uses the standard ARAP spectral form built from the SVD of `F`.

Define three twist modes

$$Q_k = \frac{1}{\sqrt{2}} U T_k V^T, \quad k=0,1,2.$$

where `T_k` are the three skew basis matrices encoded by
`make_twist_mode_0/1/2()`.

Let

$$t_k = \mathrm{vec}(Q_k), \quad \Sigma = \operatorname{diag}(\sigma_0,\sigma_1,\sigma_2).$$


Then the `F`-space Hessian used in code is written compactly using a sum:

$$H_F = 2I_9 - \sum_{k=0}^2 \alpha_k\, t_k t_k^T,$$

where

$$\alpha_0 = \frac{4}{\max(\sigma_0+\sigma_1,\varepsilon)},\quad \alpha_1 = \frac{4}{\max(\sigma_1+\sigma_2,\varepsilon)},\quad \alpha_2 = \frac{4}{\max(\sigma_0+\sigma_2,\varepsilon)}.$$ 

Here `\varepsilon = 1e-8` is the numerical safeguard used in code.

This is exactly the matrix assembled in `evaluate_template(...)`.

## 6. Chain Rule to Vertex Space

The assembled tet Hessian is

$$\nabla_x^2 E = \mu V B^T H_F B.$$

So the host/device outputs are simply the blocks of

$$G = \mu V B^T \mathrm{vec}(2(F-R)), \quad H = \mu V B^T H_F B.$$ 

Using the sum form of $H_F$ we can equivalently write

$$H = \mu V \left(2 B^T B - \sum_{k=0}^2 \alpha_k (B^T t_k)(B^T t_k)^T\right),$$

which highlights the assembled Hessian as a sum of rank-one contributions.
