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
GRAVITY = np.array([0.0, -9.8, 0.0], dtype=float)

# ----------------------
# Utility / Jacobian
# ----------------------
def build_J_for_vertex(Xi):
    """
    Xi: (3,)
    returns J: shape (3,12)
    J = [ I3 | Xi[0]*I3 | Xi[1]*I3 | Xi[2]*I3 ]
    """
    I3 = np.eye(3)
    J = np.zeros((3,12), dtype=float)
    J[:, 0:3] = I3
    J[:, 3:6] = Xi[0] * I3
    J[:, 6:9] = Xi[1] * I3
    J[:, 9:12] = Xi[2] * I3
    return J

def q_to_x_and_J_all(q, X):
    """
    q: (12,)
    X: (N,3)
    return x (N,3), Js (list of 3x12)  (we allow passing Js_all precomputed)
    """
    p = q[0:3]
    A1 = q[3:6]
    A2 = q[6:9]
    A3 = q[9:12]
    A = np.column_stack([A1, A2, A3])  # 3x3
    x = (A @ X.T).T + p  # (N,3)
    return x

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
    """
    gradient wrt column i (A[:,i]) shape (3,)
    grad = -A[i] + sum_j dot(A[i],A[j]) * A[j]  then * 4 * stiffness
    Note: A[i] denotes column i.
    """
    grad = -A_mat[:,i].copy()
    for j in range(3):
        grad += np.dot(A_mat[:,i], A_mat[:,j]) * A_mat[:,j]
    grad *= (4.0 * stiffness)
    return grad

def hessian_ortho_ij(i, j, A_mat, stiffness):
    """
    Hessian block 3x3 for (col i, col j) as given in description.
    Multiply final by 4*stiffness.
    """
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

# ----------------------
# Assembly functions
# ----------------------
def assemble_inertia(A_mat12, b_vec12, x, x_tilde, Js_all, vert_mass, dt, fixed_scale=1.0):
    """
    For each vertex:
      g_i = -m * h^{-2} * (x - x_tilde)   (3,)
      H_i = m * h^{-2} * I3
      accumulate into b and A:
         b += J^T * g_i
         A += J^T * H_i * J
    fixed_scale: if you want to stiffen (dirichlet style) use a large value
    """
    N = x.shape[0]
    h2_inv = 1.0 / (dt*dt)
    for vid in range(N):
        m = vert_mass[vid]
        diff = x[vid] - x_tilde[vid]
        g_i = -m * h2_inv * diff
        H_i = np.eye(3) * (m * h2_inv)
        # if fixed_scale != 1.0:
        #     g_i *= fixed_scale
        #     H_i *= fixed_scale
        J = Js_all[vid]
        b_vec12 += J.T @ g_i
        A_mat12 += J.T @ (H_i @ J)

def compute_init_G(x, Js_all, vert_mass):
    N = x.shape[0]
    global_gravity = np.zeros(12, dtype=float)
    for vid in range(N):
        m = vert_mass[vid]
        J = Js_all[vid]
        global_gravity += J.T @ (m * GRAVITY)
    global_gravity /= np.sum(vert_mass)
    print(f'global G = {global_gravity}')
    return global_gravity

def assemble_ground_penalty(A_mat12, b_vec12, x, Js_all, k_ground):
    """
    Ground at y=0. For each vertex with x.y < 0:
      pen = x.y (negative)
      force f = -k * pen * [0,1,0]  (3,)
      H = k * [0,1,0] * [0,1,0]^T  i.e. only H[1,1] = k
    accumulate:
      b += J.T * f
      A += J.T * H * J
    """
    N = x.shape[0]
    for vid in range(N):
        yi = x[vid,1]
        if yi < 0.0:
            pen = yi  # negative
            f = -k_ground * pen * np.array([0.0, 1.0, 0.0])
            H = np.zeros((3,3), dtype=float)
            H[1,1] = k_ground
            J = Js_all[vid]
            b_vec12 += J.T @ f
            A_mat12 += J.T @ (H @ J)

def assemble_ortho_energy(A_mat12, b_vec12, q, stiffness_ortho):
    """
    A_mat12, b_vec12 modified in-place.
    q: (12,) ; columns A1,A2,A3 at indices [3:6],[6:9],[9:12]
    Ortho energy acts directly on these 9 variables (3 columns).
    We assemble gradient and Hessian blocks.
    """
    Acols = np.column_stack([q[3:6], q[6:9], q[9:12]])  # 3x3 cols
    # gradient blocks for columns i
    for i in range(3):
        grad_i = grad_ortho_col(i, Acols, stiffness_ortho)  # (3,)
        idx_i = slice(3 + 3*i, 3 + 3*(i+1))
        # note: b is the RHS vector; previous code used cgB_q += J^T * g (g = -dE/dx)
        # For ortho energy, grad_ortho_col returns dE/d(A_col). We need to add -grad to b (so b += -grad)
        b_vec12[idx_i] += -grad_i
    # Hessian blocks
    for i in range(3):
        for j in range(3):
            hess_ij = hessian_ortho_ij(i, j, Acols, stiffness_ortho)  # (3,3)
            idx_i = slice(3 + 3*i, 3 + 3*(i+1))
            idx_j = slice(3 + 3*j, 3 + 3*(j+1))
            A_mat12[idx_i, idx_j] += hess_ij

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
def compute_total_energy(x, x_tilde, vert_mass, dt, stiffness_ground, q, stiffness_ortho):
    """
    energy components: inertia + ground + orthogonal
    """
    h2_inv = 1.0 / (dt*dt)
    energy_inertia = 0.0
    N = x.shape[0]
    for i in range(N):
        diff = x[i] - x_tilde[i]
        energy_inertia += 0.5 * vert_mass[i] * h2_inv * np.dot(diff, diff)
    energy_ground = 0.0
    for i in range(N):
        yi = x[i,1]
        if yi < 0.0:
            pen = yi
            energy_ground += 0.5 * stiffness_ground * (pen * pen)
    Acols = np.column_stack([q[3:6], q[6:9], q[9:12]])
    energy_ortho = energy_ortho_mat(Acols, stiffness_ortho)
    return energy_inertia + energy_ground + energy_ortho, (energy_inertia, energy_ground, energy_ortho)

# ----------------------
# q prediction including affine gravity
# ----------------------
def predict_q_with_affine_gravity(q, vq, vert_mass, dt, gravity):
    """
    q_tilde = q + vq * dt + affine_g * dt^2 (only applied to translation p)
    affine_g = sum(m_i * g) / total_mass
    """
    total_mass = np.sum(vert_mass)
    # each vertex has same gravity g so sum(m_i*g)=total_mass*g -> affine_g = g
    affine_g = gravity.copy()
    q_tilde = q + vq * dt + affine_g * dt * dt
    return q_tilde

# ----------------------
# Main simulation: single body
# ----------------------
def run_simulation():
    # Simple tetrahedron-like 4 vertices (model-space)
    X = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    num_verts = X.shape[0]

    # initial q: p = [0,0,0], A = Identity
    q = np.zeros(12, dtype=float)
    q[0:3] = np.array([0.0, -0.2, 0.0])  # lift slightly above ground
    q[3:6] = np.array([1.0, 0.0, 0.0])
    q[6:9] = np.array([0.0, 1.0, 0.0])
    q[9:12] = np.array([0.0, 0.0, 1.0])
    init_q = q

    vq = np.zeros(12, dtype=float)

    # per-vertex masses (can be varied)
    vert_mass = np.array([5, 6.220084, 6.220084, 6.220084], dtype=float)

    # Precompute Jacobians once (they only depend on model coords X)
    Js_all = [build_J_for_vertex(X[i]) for i in range(num_verts)]

    # Precompute affine gravity
    affine_gravity = compute_init_G(X, Js_all, vert_mass)

    # simulation params
    dt = 0.02
    nsteps = 400
    stiffness_ground = 1e5     # penalty stiffness for ground
    stiffness_ortho = 1e3      # orthogonality soft constraint stiffness (you can tune)
    newton_iters = 12

    # visualization state
    state = {"q": q, "vq": vq}
    paused = {"value": False}; single_step = {"value": False}; reset_flag = {"value": False}

    def step_once():
        q = state["q"]
        vq = state["vq"]
        vq = state["vq"]

        # predict q_tilde with affine gravity
        q_tilde = predict_q_with_affine_gravity(q, vq, vert_mass, dt, affine_gravity)

        # compute x_tilde from q_tilde
        x_tilde = q_to_x_and_J_all(q_tilde, X)

        # Newton solve in reduced q (12 DOF)
        for it in range(newton_iters):
            # current x and J
            x = q_to_x_and_J_all(q, X)

            # assemble A (12x12) and b (12)
            A_mat12 = np.zeros((12,12), dtype=float)
            b_vec12 = np.zeros(12, dtype=float)

            # inertia
            assemble_inertia(A_mat12, b_vec12, x, x_tilde, Js_all, vert_mass, dt, fixed_scale=1.0)

            # ground collision penalty
            assemble_ground_penalty(A_mat12, b_vec12, x, Js_all, stiffness_ground)

            # ortho energy
            assemble_ortho_energy(A_mat12, b_vec12, q, stiffness_ortho)

            # solve for dq
            dq = solve_linear_system(A_mat12, b_vec12, reg=1e-8)

            # linesearch in q-space
            energy_init, _ = compute_total_energy(x, x_tilde, vert_mass, dt, stiffness_ground, q, stiffness_ortho)
            alpha = 1.0
            for ls in range(20):
                q_trial = q + alpha * dq
                x_trial = q_to_x_and_J_all(q_trial, X)
                energy_trial, _ = compute_total_energy(x_trial, x_tilde, vert_mass, dt, stiffness_ground, q_trial, stiffness_ortho)
                if energy_trial <= energy_init:
                    energy_init = energy_trial
                    break
                alpha *= 0.5
            # apply step
            q = q + alpha * dq

            # convergence check (in reduced space)
            if np.linalg.norm(alpha * dq, np.inf) < 1e-6:
                break

        # update velocity
        vq = (q - state["q"]) / dt

        # write back state
        state["q"] = q
        state["vq"] = vq

        # compute new world positions
        x_final = q_to_x_and_J_all(q, X)
        return x_final, q
    

    # return
    

    # ---------- Visualization ----------
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_zlim(-1.5, 1.5)
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
    x0 = q_to_x_and_J_all(q, X)
    scat = ax.scatter(x0[:,0], x0[:,1], x0[:,2], c='r', s=60)
    lines = []
    # draw simple edges to visualize tetra
    edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    for (i,j) in edges:
        line, = ax.plot([x0[i,0], x0[j,0]], [x0[i,1], x0[j,1]], [x0[i,2], x0[j,2]], 'b-')
        lines.append(line)

    # ground plane for visual cue (y=0)
    xx = np.linspace(-1.5, 1.5, 2)
    zz = np.linspace(-1.5, 1.5, 2)
    XX, ZZ = np.meshgrid(xx, zz)
    YY = np.zeros_like(XX)
    ax.plot_surface(XX, YY, ZZ, alpha=0.1, color='grey')

    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    frame_count = 0
    def update(frame):
        nonlocal frame_count

        if reset_flag["value"]:
            state["q"][:] = init_q
            state["vq"][:] = 0
            reset_flag["value"] = False
        if not paused["value"] or single_step["value"]:
            x, qcur = step_once(); single_step["value"] = False
            scat._offsets3d = (x[:,0], x[:,1], x[:,2])
            for idx, (i,j) in enumerate(edges):
                lines[idx].set_data([x[i,0], x[j,0]], [x[i,1], x[j,1]])
                lines[idx].set_3d_properties([x[i,2], x[j,2]])
            time_text.set_text(f"step: {frame_count}")
            frame_count += 1
        return [scat] + lines + [time_text]
    
        # x, qcur = step_once()
        # scat._offsets3d = (x[:,0], x[:,1], x[:,2])
        # for idx, (i,j) in enumerate(edges):
        #     lines[idx].set_data([x[i,0], x[j,0]], [x[i,1], x[j,1]])
        #     lines[idx].set_3d_properties([x[i,2], x[j,2]])
        # time_text.set_text(f"step: {frame_count}")
        # frame_count += 1
        # return [scat] + lines + [time_text]

    ani = FuncAnimation(fig, update, frames=nsteps, interval=30, blit=False)
    plt.title("Affine Body (single) with Ground & Ortho Soft Constraint")
    plt.show()

    # frame_count = {"v":0}
    # def update(frame):
    #     if reset_flag["value"]:
    #         state["q"][:] = init_q
    #         state["vq"][:] = 0
    #         reset_flag["value"] = False
    #     if not paused["value"] or single_step["value"]:
    #         x = step_once(); single_step["value"] = False
    #         scat._offsets3d = (x[:,0], x[:,1], x[:,2])
    #     time_text.set_text(f"step: {frame_count['v']}")
    #     frame_count["v"] += 1
    #     return [scat, time_text]
    # ani = FuncAnimation(fig, update, frames=nsteps, interval=30, blit=False)
    # save_mp4 = False
    # if save_mp4:
    #     writer = FFMpegWriter(fps=30)
    #     ani.save("output.mp4", writer=writer)
    #     print("Saved to output.mp4")

    # plt.title("Affine Body with Controls (space=pause, n=step, r=reset)")
    # plt.show()

if __name__ == "__main__":
    run_simulation()
