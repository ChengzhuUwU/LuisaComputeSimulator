## Adding a New Energy to the System

This guide walks through all the steps needed to introduce a new energy term — from the mathematical implementation to system-level integration and testing.

---

### Step 1 — Implement the Energy in `Solver/Energies/detail/` directory

Add your energy implementation in `Solver/Energies/detail/*_energy.hpp` (or `*_constraint.hpp`).

Because the system has both **device (GPU shader)** and **host (CPU)** execution paths, all functions must be written in **template form** so they work with both LuisaCompute types (`luisa::compute::Var<float>`, `luisa::compute::Var<float3>`, `luisa::compute::Var<float3x3>`) and plain C++ types (`float`, `float3`, `float3x3`).

Reference: 
- [`hookean_spring_energy.hpp`](/Solver/Energies/detail/hookean_spring_energy.hpp) — a spring energy between two vertices producing a $6 \times 1$ gradient (2 `Float3` vector) and $6 \times 6$ hessian (4 `Float3x3` matrix) .
- [`fixed_joint_constraint.hpp`](/Solver/Energies/detail/fixed_joint_constaint.hpp) - a fixed joint contraint between two rigid bodies, producing a $24 \times 1$ gradient (8 `Float3` vector) and $24 \times 24$ hessian (144 `Float3x3` matrix), since each rigid body (affine body) has 12 DOF. 

These use a shared `add_linear_term` utility to accumulate the gradient/Hessian contributions of each linear residual term into an output struct with `.gradients` and `.hessians` fields.

---

### Step 2 — Unit-Test Gradient & Hessian

Add a test case in [`UnitTest/test_gradient_hessian.cpp`](/UnitTest/test_gradient_hessian.cpp) to verify that your symbolic gradient and Hessian match the central-difference approximation.

> **Important:** You must implement the gradient/hessian symbolically — do **not** use central differences as the production implementation.

Enable tests at build time:

```bash
cmake -S . -B build -D LCS_ENABLE_TEST=ON
```

---

### Step 3 — Describe the Registed Data

