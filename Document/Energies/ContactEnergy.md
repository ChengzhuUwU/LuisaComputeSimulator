# Contact Energy

This note summarizes the formulas used by
[`ground_collision_energy.hpp`](/Users/huohuo/Desktop/UntanglingProject/Codes/ForMerge/LuisaComputeSimulator/Solver/Energies/detail/ground_collision_energy.hpp).

## 1. Repulsive Contact Energy

The detail file exposes two repulsion modes controlled by `collision_type`.

Let $d$ is the current distance, $\xi$ is the material thickness, $\hat{d}$ is the contact range, $\kappa$ is contact stiffness.

### 1.1 Quadratic mode

When `collision_type == 0`, the code uses

$$c = d - \hat{d} - \xi.$$ 

$$E_{\text{rep}}(d) = \frac{1}{2} \kappa c^2.$$ 

Its derivatives are

$$\frac{dE_{\text{rep}}}{dd} = \kappa(d - \xi - \hat{d}), \quad \frac{d^2E_{\text{rep}}}{dd^2} = \kappa.$$ 

### 1.2 IPC barrier mode

Otherwise the code calls the IPC barrier kernel with shifted distance $d - \xi$:

$$E_{\text{rep}}(d) = \kappa\, b(d-\xi,\hat{d}).$$

where

$$b(s,\hat{d}) = -(s-\hat{d})^2 \log\left(\frac{s}{\hat{d}}\right).$$

So

$$\frac{dE_{\text{rep}}}{dd} = \kappa\, b'(d-\xi,\hat{d}),$$

$$\frac{d^2E_{\text{rep}}}{dd^2} = \kappa\, b''(d-\xi,\hat{d}).$$

The scalar derivatives implemented in
[`cipc_kernel.hpp`](/Users/huohuo/Desktop/UntanglingProject/Codes/ForMerge/LuisaComputeSimulator/Solver/CollisionDetector/cipc_kernel.hpp)
are

$$b'(s,\hat{d}) = (\hat{d}-s)\left[2\log\left(\frac{s}{\hat{d}}\right) - \frac{\hat{d}}{s} + 1\right].$$

$$b''(s,\hat{d}) = \left(\frac{\hat{d}}{s}+2\right)\frac{\hat{d}}{s} - 2\log\left(\frac{s}{\hat{d}}\right) - 3.$$ 

## 2. Friction Energy

The contact detail file delegates friction to
[`friction_kernel.hpp`](/Users/huohuo/Desktop/UntanglingProject/Codes/ForMerge/LuisaComputeSimulator/Solver/CollisionDetector/friction_kernel.hpp).

Let

$$\lambda_\mu = \texttt{lambda\_mu}, \quad n = \texttt{normal}, \quad \Delta x = \texttt{rel\_dx}, \quad \varepsilon = \texttt{friction\_eps}.$$ 

First define the tangential projector

$$P = I - nn^T, \quad u = P \Delta x, \quad s = \|u\|.$$ 

The regularized friction potential is

$$E_{\text{fric}} = \lambda_\mu f_0(s,\varepsilon).$$

where

$$f_0(s,\varepsilon) = \begin{cases} \dfrac{s^2}{\varepsilon} - \dfrac{s^3}{3\varepsilon^2}, & s < \varepsilon, \\ s - \dfrac{\varepsilon}{3}, & s \ge \varepsilon. \end{cases}$$

## 3. Friction Gradient and Hessian

The helper returns the gradient and Hessian with respect to `rel_dx`.

Define

$$f_1(s,\varepsilon) = \begin{cases} \dfrac{2s}{\varepsilon} - \dfrac{s^2}{\varepsilon^2}, & s < \varepsilon, \\ 1, & s \ge \varepsilon. \end{cases}$$

$$f_2(s,\varepsilon) = \begin{cases} \dfrac{2}{\varepsilon} - \dfrac{2s}{\varepsilon^2}, & s < \varepsilon, \\ 0, & s \ge \varepsilon. \end{cases}$$

Then, away from the small-velocity safeguard,

$$\nabla_{\Delta x} E_{\text{fric}} = \lambda_\mu \frac{f_1(s,\varepsilon)}{s} u,$$

$$\nabla_{\Delta x}^2 E_{\text{fric}} = \lambda_\mu \left[\left(f_2(s,\varepsilon) - \frac{f_1(s,\varepsilon)}{s}\right)\bar{u}\bar{u}^T + \frac{f_1(s,\varepsilon)}{s} P\right],$$

where

$$\bar{u} = \frac{u}{\|u\|}.$$ 

For very small tangential slip, the implementation switches to

$$\nabla_{\Delta x} E_{\text{fric}} = 0, \quad \nabla_{\Delta x}^2 E_{\text{fric}} = \lambda_\mu \frac{f_1(s,\varepsilon)}{s} P.$$ 

using the threshold `small_s = 1e-6`.
