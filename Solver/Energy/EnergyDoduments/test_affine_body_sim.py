# affine_body_single_with_ground_and_ortho.py
"""
Single Affine Body simulation (12 DOF) with:
 - Jacobians precomputed from model-space coordinates X
 - Direct assembly of A (12x12) and b (12) each solve
 - No spring/edge energies
 - Ground collision energy at y=0 (penalty)
 - Soft orthogonality energy on the 3x3 affine matrix A
 - q prediction includes affine gravity term computed from vertex masses
 - Matplotlib 3D visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
EPS = 1e-12
GRAVITY = np.array([0.0, -9.8, 0.0], dtype=float).reshape((3,1))
np.set_printoptions(linewidth=1000)

USE_ORTHO = True
USE_AUTO_DIFF = False

# ----------------------
# Utility / Jacobian
# ----------------------
# initial q: p = [0,0,0], A = Identity
def rotation_matrix_from_euler(rx, ry, rz):
    """
    Compute 3x3 rotation matrix from Euler angles (in radians).
    Rotations are applied in order: x, then y, then z.
    """
    # Rotation around x-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    # Rotation around y-axis
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    # Rotation around z-axis
    Rz = np.array([
        [np.cos(rz), np.sin(rz), 0],
        [-np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    # Combined rotation: R = Rz @ Ry @ Rx
    R = Rz @ Ry @ Rx
    return R

def build_J_for_vertex(Xi):
    # Ensure Xi is a 1D vector of length 3 to avoid deprecated
    # implicit array-to-scalar conversion (NumPy 1.25+).
    # Incoming Xi may be shaped (3,1); reshape to (3,).
    Xi = np.asarray(Xi).reshape(3,)
    I3 = np.eye(3)
    J = np.zeros((3,12), dtype=float)
    J[:, 0:3] = I3
    J[0, 3] = Xi[0]
    J[1, 4] = Xi[0]
    J[2, 5] = Xi[0]
    J[0, 6] = Xi[1]
    J[1, 7] = Xi[1]
    J[2, 8] = Xi[1]
    J[0, 9] = Xi[2]
    J[1, 10] = Xi[2]
    J[2, 11] = Xi[2]
    # J[0, 3:6] = Xi.T
    # J[1, 6:9] = Xi.T
    # J[2, 9:12] = Xi.T
    return J

def q_to_x_all(q, X):
    p = q[0:3]
    A1 = q[3:6]
    A2 = q[6:9]
    A3 = q[9:12]
    A = np.column_stack([A1, A2, A3])  # 3x3
    x = X.copy()
    for vid in range(X.shape[0]):
        x[vid] = A @ X[vid] + p
    # x = (A @ X.T).T + p  # (N,3)
    return x

# ----------------------
# q prediction including affine gravity
# ----------------------
def compute_body_mass(Js_all, vert_mass):
    numVerts = len(Js_all)
    M = np.zeros((12, 12), dtype=float)
    for i in range(numVerts):
        J = Js_all[i]
        M += vert_mass[i] * J.T @ J
    # print(f'body mass = \n{M}')
    return M

def compute_init_Gravity(x, Js_all, vert_mass, body_mass):
    N = x.shape[0]
    global_gravity = np.zeros((12,1), dtype=float)
    for vid in range(N):
        m = vert_mass[vid]
        J = Js_all[vid]
        global_gravity += J.T @ (m * GRAVITY)
    body_mass_inv = np.linalg.inv(body_mass)
    global_gravity = body_mass_inv @ global_gravity
    # print(f'global G = \n{global_gravity}')
    # global_gravity = np.zeros(12, dtype=float)
    # global_gravity[0:3] = GRAVITY
    return global_gravity

def compute_tetmesh_body_force(positions, tets, body_force_density):
    f = body_force_density
    body_force = np.zeros(12)

    def per_tet(tet):
        p0 = positions[tet[0]]
        p1 = positions[tet[1]]
        p2 = positions[tet[2]]
        p3 = positions[tet[3]]

        e1 = p1 - p0
        e2 = p2 - p0
        e3 = p3 - p0

        D = np.dot(e1, np.cross(e2, e3))
        V = D / 6.0

        # Qs computation
        def Q_p(i):
            return D * (p0[i] + p1[i] + p2[i] + p3[i]) / 24.0

        Qs = np.array([Q_p(0), Q_p(1), Q_p(2)])  # (3,)

        contrib = np.zeros(12)
        contrib[0:3] += (f * V)      # translation part
        contrib[3:6] += (f[0] * Qs)  # affine col1
        contrib[6:9] += (f[1] * Qs)  # affine col2
        contrib[9:12] += (f[2] * Qs) # affine col3

        return contrib

    for tet in tets:
        body_force += per_tet(tet)
    return body_force

def predict_q_with_affine_gravity(q, vq, vert_mass, dt, gravity):
    affine_g = gravity.copy()
    q_tilde = q + vq * dt + affine_g * dt * dt
    # print(f'Dim of G = {affine_g.shape}, vq = {vq.shape}, q = {q.shape}')
    return q_tilde

# ----------------------
# Ortho energy (numpy translation of given wp funcs)
# ----------------------
def energy_ortho_mat(A_mat, stiffness):
    """
    A_mat: 3x3 with columns A1,A2,A3
    returns scalar energy
    energy_ortho = sum_{i,j} (dot(A[:,i], A[:,j]) - delta_ij)^2 * stiffness
    """
    e = 0.0
    for i in range(3):
        for j in range(3):
            target = 1.0 if i == j else 0.0
            term = np.dot(A_mat[:,i], A_mat[:,j]) - target
            e += term * term
    return e * stiffness

def grad_ortho_col(i, A_mat, stiffness):
    grad = -A_mat[:,i].copy().reshape((3,1))
    for j in range(3):
        grad += np.dot(A_mat[:,i], A_mat[:,j]) * A_mat[:,j].reshape((3,1))
    grad *= (4.0 * stiffness)
    return grad

def hessian_ortho_ij(i, j, A_mat, stiffness):
    hess = np.zeros((3,3), dtype=float)
    if i == j:
        # qiqiT = outer(A[i],A[i])
        qiqiT = np.outer(A_mat[:,i], A_mat[:,i])
        qiTqi = np.dot(A_mat[:,i], A_mat[:,i]) - 1.0
        term2 = np.diag(np.full(3, qiTqi))
        # sum_k outer(A[k],A[k])
        for k in range(3):
            hess += np.outer(A_mat[:,k], A_mat[:,k])
        hess += qiqiT + term2
    else:
        # hess = outer(A[j], A[i]) + diag(vec3(dot(A[j],A[i])))
        hess = np.outer(A_mat[:,j], A_mat[:,i]) + np.diag(np.full(3, np.dot(A_mat[:,j], A_mat[:,i])))
    hess *= (4.0 * stiffness)
    return hess

def assemble_ortho_energy(A_mat12, b_vec12, q, stiffness_ortho):
    Acols = np.column_stack([q[3:6], q[6:9], q[9:12]])  # 3x3 cols

    print(f'Input ortho A = \n{Acols}')

    B = np.zeros_like(b_vec12)
    H = np.zeros_like(A_mat12)

    # gradient blocks for columns i
    for i in range(3):
        grad_i = grad_ortho_col(i, Acols, stiffness_ortho)  # (3,)
        # print(f'grad = {grad_i}, B = {B[3 + 3*i: 3 + 3*(i+1)]}')
        B[3 + 3*i: 3 + 3*(i+1)] += -grad_i

    # Hessian blocks
    for i in range(3):
        for j in range(3):
            hess_ij = hessian_ortho_ij(i, j, Acols, stiffness_ortho)  # (3,3)
            idx_i = slice(3 + 3*i, 3 + 3*(i+1))
            idx_j = slice(3 + 3*j, 3 + 3*(j+1))
            H[idx_i, idx_j] += hess_ij
            # print(f'hessian of {i} adj {j} = {hess_ij}')

    print(f'Output ortho B = {B}, H = {H}')
    return B, H

def analytic_grad_hess_from_q(A_mat12, b_vec12, q, stiffness):
    """
    Returns (grad12, hess12x12) for the ortho energy.
    grad has zeros in entries 0..2.
    """
    A = np.column_stack([q[3:6], q[6:9], q[9:12]])
    f12 = np.zeros_like(b_vec12)
    H12 = np.zeros_like(A_mat12)
    s4 = 4.0 * stiffness

    # gradient blocks (3-vector per column)
    for p in range(3):
        grad_p = -A[:,p].copy()
        for k in range(3):
            grad_p += np.dot(A[:,p], A[:,k]) * A[:,k]
        grad_p *= s4
        f12[3 + 3*p : 3 + 3*p + 3] = -grad_p.reshape((3,1))

    # Hessian blocks H_{pq} 3x3 each
    sum_outer = np.zeros((3,3), dtype=float)
    for k in range(3):
        sum_outer += np.outer(A[:,k], A[:,k])

    for p in range(3):
        for q in range(3):
            c_pq = np.dot(A[:,p], A[:,q]) - (1.0 if p == q else 0.0)
            block = np.outer(A[:,p], A[:,q]) + c_pq * np.eye(3)
            if p == q:
                block += sum_outer
            block *= s4
            i0 = 3 + 3*p; j0 = 3 + 3*q
            H12[i0:i0+3, j0:j0+3] = block

    # b_vec12 -= grad12
    # A_mat12 += H12
    return f12, H12


# ----------------------
# Assembly functions
# ----------------------
def assemble_inertia_perbody(A_mat12, b_vec12, q, q_tilde, Js_all, body_mass, dt, fixed_scale=1.0):
    h2_inv = 1.0 / (dt*dt)
    M = body_mass
    # Ensure q and q_tilde are column vectors (12x1)
    diff = (q - q_tilde)
    # print(f'shape of M = {M.shape}, shape of diff = {q.shape} and {q_tilde.shape}')
    f_i = -h2_inv * (M @ diff)
    H_i = h2_inv * M
    # print(f'inertia force = \n{f_i.reshape((12,))} H = \n{H_i} ')
    return f_i, H_i
    # b_vec12 += g_i
    # A_mat12 += H_i

def assemble_ground_penalty(A_mat12, b_vec12, x, Js_all, stiffnes_collision):
    N = x.shape[0]
    outB = np.zeros_like(b_vec12)
    outA = np.zeros_like(A_mat12)
    for vid in range(N):
        yi = x[vid][1][0]
        # print(f'xi = {x[vid]}')
        if yi < 0:
            C = - yi
            # print(f'c = {C}')
            f = stiffnes_collision * C * np.array([0.0, 1.0, 0.0]).reshape((3,1))
            H = np.zeros((3,3), dtype=float)
            H[1,1] = stiffnes_collision
            J = Js_all[vid]

            globalB = J.T @ f
            globalA = J.T @ (H @ J)
            # print(f'global b = {globalB}')
            # globalA = hessian_proj_SPD(globalA)
            outB += globalB
            outA += globalA
            # b_vec12 += globalB
            # A_mat12 += globalA

    return outB, outA
            # print(f'nemetrial penetration = {C}')
            # print(f'vid {vid} g = \n{J.T @ f} , H = \n{J.T @ (H @ J)} ')
            
            # for ii in range(4):
            #     print(f'for vert {vid} force = {tmpb[3*ii:3*ii+3]}')
            # idx = 0
            # for ii in range(4):
            #     for jj in range(ii, 4):
            #         print(f'for vert {vid} {idx}s hess = {A_mat12[3*ii:3*ii+3][3*jj:3*jj+3]}')
            #         idx += 1


# ----------------------
# Solver helper
# ----------------------
def solve_linear_system(A_mat12, b_vec12, reg=1e-8):
    # regularize a bit for stability
    A = A_mat12.copy()
    # A += np.eye(12) * reg
    b = b_vec12.copy()
    try:
        dq = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        dq, *_ = np.linalg.lstsq(A, b, rcond=None)
    return dq

# ----------------------
# Energy evaluation (for linesearch)
# ----------------------
def compute_total_energy(X, body_mass, dt, stiffness_ground, q, q_tilde, stiffness_ortho):
    x = q_to_x_all(q, X)
    h2_inv = 1.0 / (dt*dt)
    energy_inertia = 0.0
    N = X.shape[0]
    energy_inertia = 0.5 * h2_inv * (q - q_tilde).reshape(12, 1).T @ body_mass @ (q - q_tilde).reshape(12, 1)
    print(f'diff = {q - q_tilde}, mass = {body_mass}, E = {energy_inertia}')
    # for i in range(N):
    #     diff = x[i] - x_tilde[i]
    #     energy_inertia += 0.5 * vert_mass[i] * h2_inv * np.dot(diff, diff)
    energy_ground = 0.0
    for i in range(N):
        yi = x[i,1]
        if yi < 0.0:
            pen = yi
            energy_ground += 0.5 * stiffness_ground * (pen * pen)
            # print(f'Vert {i} penetration = {pen}, ground energy += {0.5 * stiffness_ground * (pen * pen)}')
    energy_ortho = 0.0
    if USE_ORTHO:
        Acols = np.column_stack([q[3:6], q[6:9], q[9:12]])
        e = stiffness_ortho * np.square(np.linalg.norm(Acols * Acols.T - np.identity(3,float), 'fro'))
        # for i in range(3):
        #     for j in range(3):
        #         target = 1.0 if i == j else 0.0
        #         term = np.dot(Acols[:,i], Acols[:,j]) - target
        #         e += term * term
        energy_ortho += e
        # energy_ortho = energy_ortho_mat(Acols, stiffness_ortho)
    print(f'Energy inertia = {energy_inertia}, gound = {energy_ground}, ortho = {energy_ortho}')
    return energy_inertia + energy_ground + energy_ortho, (energy_inertia, energy_ground, energy_ortho)

import jax
import jax.numpy as jnp

def jax_q_to_x(q, X):
    """Affine transform: x = t + A @ X"""
    t = q[0:3]
    A = jnp.column_stack([q[3:6], q[6:9], q[9:12]])  # (3,3)
    return (A @ X) + t  # (N,3)

def jax_func_body(q, body_mass, dt, q_tilde, stiffness_ortho):
    h2_inv = 1.0 / (dt * dt)

    # inertia energy
    dq = q - q_tilde
    # Ensure scalar output
    energy_inertia = 0.5 * h2_inv * jnp.squeeze(dq.T @ body_mass @ dq)

    # orthogonality penalty
    Acols = jnp.column_stack([q[3:6], q[6:9], q[9:12]])  # (3,3)
    M = Acols @ Acols.T - jnp.eye(3)
    frob_norm_sq = jnp.sum(M**2)
    energy_ortho = stiffness_ortho * frob_norm_sq
    return energy_inertia + energy_ortho
    return energy_ortho

def jax_func_inertia(q, body_mass, dt, q_tilde, stiffness_ortho):
    h2_inv = 1.0 / (dt * dt)
    dq = q - q_tilde
    energy_inertia = 0.5 * h2_inv * jnp.squeeze(dq.T @ body_mass @ dq)
    return energy_inertia

def jax_func_orthogonality(q, body_mass, dt, q_tilde, stiffness_ortho):
    Acols = jnp.column_stack([q[3:6], q[6:9], q[9:12]])  # (3,3)
    M = Acols @ Acols.T - jnp.eye(3)
    frob_norm_sq = jnp.sum(M**2)
    energy_ortho = stiffness_ortho * frob_norm_sq
    return energy_ortho

def jax_func_vert(q, X, stiffness_ground):
    x = jax_q_to_x(q, X)  # (N,3)
    yi = x[1]  # y coordinate
    penetration = jnp.minimum(yi, 0.0)
    # print(f'jax penetration = {penetration}')
    energy_ground = 0.5 * stiffness_ground * penetration * penetration
    return energy_ground
    
jax_grad_body = jax.grad(jax_func_body)  # gradient wrt q
jax_grad_vert = jax.grad(jax_func_vert)  # gradient wrt q
jax_hess_body = jax.hessian(jax_func_body)  # gradient wrt q
jax_hess_vert = jax.hessian(jax_func_vert)  # gradient wrt q

jax_grad_x_to_q = jax.jacobian(jax_q_to_x)

def hessian_proj_SPD(Mat):
    """
    Project a symmetric matrix to the nearest symmetric positive semi-definite matrix
    by setting negative eigenvalues to zero.
    """
    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(Mat)
    # Clamp negative eigenvalues to zero
    eigvals_clamped = np.maximum(eigvals, 0)
    # Reconstruct the matrix
    Mat_psd = eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T
    return Mat_psd

# ----------------------
# Main simulation: single body
# ----------------------
def run_simulation():
    # Simple tetrahedron-like 4 vertices (model-space)
    # X = np.array([
    #     np.array([0.0, 0.0, 0.0], float).reshape(3, 1),
    #     np.array([1.0, 0.0, 0.0], float).reshape(3, 1),
    #     np.array([0.0, 1.0, 0.0], float).reshape(3, 1),
    #     np.array([0.0, 0.0, 1.0], float).reshape(3, 1),
    # ], dtype=float)
    # num_verts = X.shape[0]
    # edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    # vert_mass = np.array([5, 6.220084, 6.220084, 6.220084], dtype=float) 

    import trimesh
    mesh = trimesh.load("/Users/huohuo/Desktop/OpenSource/Codim-IPC/Projects/FEMShell/input/cube.obj")
    # mesh = trimesh.load("/Users/huohuo/Desktop/Project/LuisaComputeSolver/Resources/InputMesh/sphere63.obj")
    X = mesh.vertices  # numpy array of shape (n, 3)
    num_verts = X.shape[0]
    X = X.reshape((-1, 3, 1))
    vec1 = X[:,0]
    print(f'X1 = {X[:,0].shape}')
    edges = mesh.edges_unique
    total_mass = 1.0
    vert_mass = np.ones(num_verts) * (total_mass / num_verts)

    # Example usage:
    init_rotate = np.zeros((3,3), float)
    init_q = np.zeros(12, dtype=float).reshape(12, 1)
    init_q[0:3, :] = np.array([0.0, -0.1, 0.0]).reshape(3, 1)  # lift slightly above ground
    # rx, ry, rz = np.pi/6, 0.0, np.pi/6
    rx, ry, rz = 0.0, 0.0, 0.0
    R = rotation_matrix_from_euler(rx, ry, rz)
    R = R
    init_q[3:6, 0] = R[:,0]; init_q[6:9, 0] = R[:,1]; init_q[9:12, 0] = R[:,2]
    q = init_q.copy()
    vq = np.zeros_like(q)

    print(f'init q = {init_q.T}')

    # Precompute Jacobians once (they only depend on model coords X)
    Js_all = [build_J_for_vertex(X[i]) for i in range(num_verts)]
    # print(f'Numerical J = \n{build_J_for_vertex(np.array([1,2,3],float))}')
    # print(f'Jax       J = \n{jax_grad_x_to_q(jnp.array(init_q), np.array([1,2,3],float).reshape(3,1))}')

    # print(jax_grad_x_to_q(jnp.array(init_q), np.array([1,2,3],float).reshape(3,1)).shape)

    # q_jax = jnp.array(init_q)
    # for vid in range(num_verts):
    #     J = jax_grad_x_to_q(q_jax, X[vid])
    #     Js_all[vid] = J
    #     print(f'Jax jacobian of vid {vid} = {J}')
    # for vid in range(num_verts):
    #     J = Js_all[vid]
    #     print(f'Numerical jacobian of vid {vid} = {J}')
    
    # Precompute affine gravity
    body_mass = compute_body_mass(Js_all, vert_mass)
    affine_gravity = compute_init_Gravity(X, Js_all, vert_mass, body_mass)
    # affine_gravity = compute_tetmesh_body_force(X, [[0,1,2,3]], GRAVITY)

    # print(f'Initial body mass = \n{body_mass}')
    
    # simulation params
    dt = 0.1
    nsteps = 400
    stiffness_ground = 1e5     # penalty stiffness for ground
    stiffness_ortho = 1e5      # orthogonality soft constraint stiffness (you can tune)
    newton_iters = 2

    # visualization state
    state = {"q": q, "vq": vq}
    paused = {"value": True}; single_step = {"value": False}; reset_flag = {"value": False}

    def step_once():
        q = state["q"]
        vq = state["vq"]
        q_step_bg = q.copy()

        q_tilde = predict_q_with_affine_gravity(q, vq, vert_mass, dt, affine_gravity)

        # Newton solve in reduced q (12 DOF)
        for it in range(newton_iters):
            # current x and J
            x = q_to_x_all(q, X)
            q_iter_bg = q.copy()

            # assemble A (12x12) and b (12)
            A_mat12 = np.zeros((12,12), dtype=float)
            b_vec12 = np.zeros((12,1), dtype=float)

            if USE_AUTO_DIFF:
                q_jax = jnp.array(q)
                q_tilde_jax = jnp.array(q_tilde)

                E = jax_func_body(q_jax, body_mass, dt, q_tilde_jax, stiffness_ortho)
                g = jax_grad_body(q_jax, body_mass, dt, q_tilde_jax, stiffness_ortho)
                H = jax_hess_body(q_jax, body_mass, dt, q_tilde_jax, stiffness_ortho)
                H = H.reshape(12, 12)
                b_vec12 -= g
                A_mat12 += H

                # bb, AA = assemble_inertia_perbody(A_mat12, b_vec12, q, q_tilde, Js_all, body_mass, dt, fixed_scale=1.0)
                # b_vec12 += bb
                # A_mat12 += AA

                # bb, AA = assemble_ortho_energy(A_mat12, b_vec12, q, stiffness_ortho)
                # b_vec12 += bb
                # A_mat12 += AA

                bb, AA = assemble_ground_penalty(A_mat12, b_vec12, x, Js_all, stiffness_ground)
                b_vec12 += bb
                A_mat12 += AA

                # print(f'numerical b = \n{bb.T}')
                # print(f'numerical A = \n{AA}')

                # for vid in range(num_verts):
                #     E = jax_func_vert(q_jax, X[vid], stiffness_ground)
                #     g = jax_grad_vert(q_jax, X[vid], stiffness_ground)
                #     H = jax_hess_vert(q_jax, X[vid], stiffness_ground)
                #     H = H.reshape(12, 12)
                #     b_vec12 -= g
                #     A_mat12 += H

                #     print(f'g shape = {g.shape} H size = {H.shape}')
                #     if x[vid][1] < 0:
                #         print(f'vid {vid} g = \n{g.T} , H = \n{H} ')

                # print(f'final E = \n{E}')
                # print(f'final B = \n{b_vec12}')
                # print(f'final A = \n{A_mat12}')
            else:
                bb, AA = assemble_inertia_perbody(A_mat12, b_vec12, q, q_tilde, Js_all, body_mass, dt, fixed_scale=1.0)
                b_vec12 += bb
                A_mat12 += AA
                if USE_ORTHO:
                    # assemble_ortho_energy(A_mat12, b_vec12, q, stiffness_ortho)
                    bb, AA = analytic_grad_hess_from_q(A_mat12, b_vec12, q, stiffness_ortho)
                    b_vec12 += bb
                    A_mat12 += AA
                bb, AA = assemble_ground_penalty(A_mat12, b_vec12, x, Js_all, stiffness_ground)
                b_vec12 += bb
                A_mat12 += AA
            
            # solve for dq
            dq = solve_linear_system(A_mat12, b_vec12, reg=1e-8)

            use_ls = True
            alpha = 1.0

            # convergence check (in reduced space)
            print(f'In newton iter {it}, maxP = {np.linalg.norm(dq, np.inf)}')
            if np.linalg.norm(dq, np.inf) < 1e-6:
                break

            if use_ls:
                def get_energy():
                    # print(f'AutoDiffE = {E} = (Inertia {E_inertia} + ortho {E_otrho} + Ground {E_gound}), Numerical = {E_numerical} = Inertia {energy_inertia} + Ortho {energy_ortho} + Ground {energy_ground}')

                    if USE_AUTO_DIFF:
                        E = 0
                        q_jax = jnp.array(q)
                        q_tilde_jax = jnp.array(q_tilde)
                        E += jax_func_body(q_jax, body_mass, dt, q_tilde_jax, stiffness_ortho)
                        for vid in range(num_verts):
                            E += jax_func_vert(q_jax, X[vid], stiffness_ground)
                        return E, []
                    else:
                        return compute_total_energy(X, body_mass, dt, stiffness_ground, q, q_tilde, stiffness_ortho)   
                
                # linesearch in q-space
                q = q_iter_bg
                x = q_to_x_all(q, X)
                print(f'Before linesearch, energy evaluation:')
                energy_init, _ = get_energy()    
                
                for ls in range(20):
                    q = q_iter_bg + alpha * dq
                    x = q_to_x_all(q, X)
                    energy_trial, _ = get_energy()
                    if energy_trial <= energy_init:
                        energy_init = energy_trial
                        break
                    alpha *= 0.5
                    # print(f'Line search {ls} : alpha = {alpha}, prev {energy_init} < energy = {energy_trial} ')

            # apply step
            q = q_iter_bg + alpha * dq

        # update velocity
        vq = (q - q_step_bg) / dt

        # write back state
        state["q"] = q.copy()
        state["vq"] = vq.copy()

        # compute new world positions
        x_final = q_to_x_all(q, X)
        return x_final, q.copy()
    

    # x, qcur = step_once()
    # return
    

    # ---------- Visualization ----------
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')  # 原来的z轴现在是y轴
    ax.set_zlabel('Y')  # 原来的y轴现在是z轴
    ax.set_box_aspect([1,1,1])


    def on_key(event):
        if event.key == " ":
            paused["value"] = not paused["value"]
        elif event.key == "w":
            single_step["value"] = True
        elif event.key == "e":
            reset_flag["value"] = True

    fig.canvas.mpl_connect("key_press_event", on_key)

    # initial draw
    init_x = q_to_x_all(init_q, X)
    scat = ax.scatter(init_x[:,0], init_x[:,2], init_x[:,1], c='r', s=10)
    # draw simple edges to visualize tetra
    lines = []
    for (i,j) in edges:
        line, = ax.plot([init_x[i,0], init_x[j,0]], [init_x[i,2], init_x[j,2]], [init_x[i,1], init_x[j,1]], 'b-')
        lines.append(line)

    # ground plane for visual cue (y=0)
    xx = np.linspace(-5, 5, 2)
    zz = np.linspace(-5, 5, 2)
    XX, ZZ = np.meshgrid(xx, zz)
    YY = np.zeros_like(XX)
    ax.plot_surface(XX, ZZ, YY, alpha=0.1, color='grey')

    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    frame_count = 0
    def update(frame):
        nonlocal frame_count

        if reset_flag["value"]:
            state["q"][:] = init_q
            state["vq"][:] = 0
            reset_flag["value"] = False
            x = q_to_x_all(init_q, X)
            x = np.squeeze(x, axis=-1)
            scat._offsets3d = (x[:,0], x[:,2], x[:,1])
            for idx, (i,j) in enumerate(edges):
                lines[idx].set_data([x[i,0], x[j,0]], [x[i,2], x[j,2]])
                lines[idx].set_3d_properties([x[i,1], x[j,1]])
        if not paused["value"] or single_step["value"]:
            x, qcur = step_once(); single_step["value"] = False
            # print(f'dim x = {x.shape}')
            # squeeze the last dimension to get shape (4, 3)
            x = np.squeeze(x, axis=-1)
            scat._offsets3d = (x[:,0], x[:,2], x[:,1])
            for idx, (i,j) in enumerate(edges):
                lines[idx].set_data([x[i,0], x[j,0]], [x[i,2], x[j,2]])
                lines[idx].set_3d_properties([x[i,1], x[j,1]])
            time_text.set_text(f"step: {frame_count}") # , y = {x[0,1]}/{x[1,1]}/{x[2,1]}/{x[3,1]}
            frame_count += 1
        return [scat] + lines + [time_text]

    ani = FuncAnimation(fig, update, frames=nsteps, interval=30, blit=False)
    plt.title("Affine Body (single) with Ground & Ortho Soft Constraint")
    plt.show()




if __name__ == "__main__":
    run_simulation()
