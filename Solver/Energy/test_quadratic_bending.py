import jax

import jax.numpy as jnp


Q0 = jnp.array([
    [12.0, 12.0, -12.0, -12.0],
    [12.0, 12.0, -12.0, -12.0],
    [-12.0, -12.0, 12.0, 12.0],
    [-12.0, -12.0, 12.0, 12.0] 
])
Q = jnp.zeros((12, 12))
for i in range(4):
    for j in range(4):
        Q = Q.at[i*3:(i+1)*3, j*3:(j+1)*3].set(Q0[i][j] * jnp.eye(3))

def quadratic_bending_energy(x, Q, k):
    return 0.5 * k * x @ Q @ x

k = 1e-2 * 0.5
energy_fn = lambda x: quadratic_bending_energy(x, Q, k)
grad_fn = jax.grad(energy_fn)
hess_fn = jax.jacfwd(jax.grad(energy_fn))

# 示例输入：四个顶点的坐标
# 例如：v0, v1, v2, v3 = [x, y, z]，拼成一个 (12,) 的向量
vertices = jnp.array([
    -0.5, -0.0244989, -0.5,  # v0
    0.5, -0.0244989, 0.5,  # v1
    0.5, -0.024502208, -0.5,  # v2
    -0.5, 0, 0.5   # v3
])
# vertices = jnp.array([
#     -0.5, 0, -0.5,  # v0
#      0.5, 0,  0.5,  # v1
#      0.5, 0, -0.5,  # v2
#     -0.5, 0,  0.5   # v3
# ])

# 计算能量、梯度和 Hessian
energy = energy_fn(vertices)
gradient = grad_fn(vertices)
hessian = hess_fn(vertices)

print("Energy:", energy)
print("Gradient:", gradient)
print("Hessian:\n", hessian)

# force = float3(0, 0.0029389416, 0), float3(0, 0.0029389416, 0), float3(0, -0.0029389416, 0), float3(0, -0.0029389416, 0)