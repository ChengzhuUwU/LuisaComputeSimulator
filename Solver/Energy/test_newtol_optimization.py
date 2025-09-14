import jax

import jax.numpy as jnp
import numpy as np
import scipy

EPS = 1e-12
GRAVITY = np.array([0.0, -9.8, 0.0], dtype=float)

pring_detail = False

def predict_position(x, x_step_start, x_iter_start, x_tilde, v, cgX, is_fixed, substep_dt):
    N = x.shape[0]
    # gravity applied to all non-fixed vertices
    for vid in range(N):
        x_prev = x_step_start[vid].copy()
        v_prev = v[vid].copy()
        outer_acc = GRAVITY
        v_pred = v_prev + substep_dt * outer_acc

        if is_fixed[vid]:
            v_pred = v_prev.copy()

        x_iter_start[vid] = x_prev.copy()
        x_pred = x_prev + substep_dt * v_pred
        x_tilde[vid] = x_pred
        x[vid] = x_prev.copy()

def update_velocity(x, v, x_step_start, v_step_start, substep_dt, fix_scene=False, damping=0.0):
    N = x.shape[0]
    for vid in range(N):
        x_begin = x_step_start[vid].copy()
        x_end = x[vid].copy()

        dx = x_end - x_begin
        vel = dx / substep_dt

        v[vid] = vel.copy()
        v_step_start[vid] = vel.copy()
        x_step_start[vid] = x_end.copy()

def evaluate_inertia(x, x_tilde, cgB, cgA, is_fixed, vert_mass, substep_dt, stiffness_dirichlet=0.0):

    N = x.shape[0]
    h = substep_dt
    h2_inv = 1.0 / (h * h)
    for vid in range(N):
        x_k = x[vid]
        x_t = x_tilde[vid]
        mass = vert_mass[vid]
        gradient = -mass * h2_inv * (x_k - x_t)   # shape (3,)
        hessian = np.eye(3) * (mass * h2_inv)    # shape (3,3)

        if is_fixed[vid]:
            gradient = gradient * (stiffness_dirichlet)
            hessian = hessian * (stiffness_dirichlet)
        
        if pring_detail: print(f'    vid {vid} mass = {mass} inertia move = {np.linalg.norm(x_k - x_t)} gradient {gradient}, hessian diag {np.diag(hessian)}')

        cgB[vid] += gradient
        cgA[vid] += hessian

def evaluate_springs(x, cgB, cgA_diag, cgA_offdiag_stretch_spring, edges, rest_lengths, stiffness_stretch):
    M = edges.shape[0]
    for eid in range(M):
        i, j = int(edges[eid,0]), int(edges[eid,1])
        xi = x[i]
        xj = x[j]
        diff = xj - xi
        l = np.linalg.norm(diff)
        if l < EPS:
            l = EPS
        L0 = float(rest_lengths[eid])
        C = l - L0
        dir_vec = diff / l
        xxT = np.outer(diff, diff)    # shape (3,3)
        x_inv = 1.0 / l
        x_squared_inv = x_inv * x_inv
        k = float(stiffness_stretch)

        # force on i is k * dir * C, on j is -force_i
        f_i = k * dir_vec * C
        f_j = -f_i

        # He as in C++
        He = k * x_squared_inv * xxT + k * max(1.0 - L0 * x_inv, 0.0) * (np.eye(3) - x_squared_inv * xxT)

        if pring_detail: print(f'    eid {eid} spring l={l:.6e}, L0={L0:.6e}, C={C:.6e}, force_i={f_i}, He_diag={np.diag(He)}')

        # accumulate forces into cgB (note C++ used atomic add per component)
        cgB[i] += f_i
        cgB[j] += f_j

        # accumulate diagonal Hessian contribution into both endpoints
        cgA_diag[i] += He
        cgA_diag[j] += He
        cgA_offdiag_stretch_spring[eid] = -He.copy()

def assemble_dense_system(cgB, cgA_diag, edges, cgA_offdiag):
    N = cgB.shape[0]
    A = np.zeros((3*N, 3*N), dtype=float)
    b = cgB.reshape(3*N)  # keep sign consistent with how you use gradient; user may want -b for linear solve

    # fill diagonal blocks
    for vi in range(N):
        A[3*vi:3*vi+3, 3*vi:3*vi+3] = cgA_diag[vi]

    # fill off-diagonals from edges
    M = edges.shape[0]
    for eid in range(M):
        i, j = int(edges[eid,0]), int(edges[eid,1])
        off = cgA_offdiag[eid]  # this is -He (3x3)
        # place at (i,j) and (j,i). Because He symmetric, off.T == off
        A[3*i:3*i+3, 3*j:3*j+3] += off
        A[3*j:3*j+3, 3*i:3*i+3] += off

    return A, b

def compute_energy(x, x_tilde, is_fixed, vert_mass, substep_dt, edges, rest_lengths, stiffness_stretch):
    N = x.shape[0]
    M = edges.shape[0]
    h = substep_dt
    h2_inv = 1.0 / (h * h)

    energy_inertia = 0.0
    for vid in range(N):
        x_k = x[vid]
        x_t = x_tilde[vid]
        mass = vert_mass[vid]
        diff = x_k - x_t
        energy_inertia += 0.5 * mass * h2_inv * np.dot(diff, diff)
        if pring_detail: print(f'    vid {vid} inertia energy {0.5 * mass * h2_inv * np.dot(diff, diff):.6e} (|dx| = {np.linalg.norm(diff):.6e})')

    energy_spring = 0.0
    for eid in range(M):
        i, j = int(edges[eid,0]), int(edges[eid,1])
        diff = x[i] - x[j]
        l = np.linalg.norm(diff)
        L0 = rest_lengths[eid]
        C = l - L0
        k = stiffness_stretch
        energy_spring += 0.5 * k * C * C
        if pring_detail: print(f'    eid {eid} spring energy {0.5 * k * C * C:.6e} (l={l:.6e}, L0={L0:.6e})')

    if pring_detail: print(f'      Energy: inertia {energy_inertia:.6e}, spring {energy_spring:.6e}, total {energy_inertia + energy_spring:.6e}')
    total_energy = energy_inertia + energy_spring
    return total_energy, [energy_inertia, energy_spring]


if __name__ == "__main__":
    # simple chain of 3 vertices in a line
    
    init_x  = np.array([[-0.5, 0, -0.5,],
                        [ 0.5, 0, -0.5,],
                        [-0.5, 0,  0.5,],
                        [ 0.5, 0,  0.5,]], dtype=float)
    init_v = np.array([[-0.5, 0, -0.5,],
                       [ 0.5, 0, -0.5,],
                       [-0.5, 0,  0.5,],
                       [ 0.5, 0,  0.5,]], dtype=float)
    
    # x = init_x.copy()
    # v = init_x.copy()
    x  = np.array([[-0.501755, -0.0708859, -0.497971],
                   [0.498333, -0.0711678, -0.498333],
                   [-0.5, 0, 0.5],
                   [0.497971, -0.0708859, 0.501755]], dtype=float)
    v = np.array([[ -0.0282639, -0.839662, 0.0325353,],
                  [ -0.0268902, -0.844535, 0.0268902,],
                  [ 0, -1.87321e-11, 0,],
                  [ -0.0325353, -0.839662, 0.0282639,]], dtype=float)

    num_verts = x.shape[0]

    is_fixed = np.array([False, False, True, False], dtype=bool)
    vert_mass = np.array([10/3, 10/6, 10/6, 10/3], dtype=float)

    edges = np.array([
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 3],
        [2, 3],
    ], dtype=int)
    bending_edges = np.array([
        [0, 3, 1, 2],
    ], dtype=int)
    faces = np.array([
        [0, 3, 1],
        [0, 2, 3]
    ], dtype=int)
    rest_lengths = np.array([
        np.linalg.norm(init_x[1] - init_x[0]),
        np.linalg.norm(init_x[2] - init_x[0]),
        np.linalg.norm(init_x[3] - init_x[0]),
        np.linalg.norm(init_x[3] - init_x[1]),
        np.linalg.norm(init_x[3] - init_x[2]),
    ], dtype=float)
    num_edges = edges.shape[0]


    substep_dt = 0.05
    stiffness_dirichlet = 1e9   # if you want to strongly pin fixed vertices
    stiffness_stretch = 1e4

    def single_frame():
        global x, v

        # step start copies
        x_step_start = x.copy()
        v_step_start = v.copy()
        x_tilde = np.zeros_like(x)

        x_iter_start = x.copy()

        # 1) predict
        predict_position(x, x_step_start, x_iter_start, x_tilde, v, None, is_fixed, substep_dt)

        def single_newton():

            x_iter_start = x.copy()

            # cg storage
            cgB = np.zeros((num_verts,3), dtype=float)
            cgA_diag = np.zeros((num_verts,3,3), dtype=float)
            cgA_offdiag = np.zeros((num_edges,3,3), dtype=float)

            # 2) evaluate inertia (fills cgB, cgA_diag)
            evaluate_inertia(x, x_tilde, cgB, cgA_diag, is_fixed, vert_mass, substep_dt, stiffness_dirichlet)

            # 3) evaluate springs (adds contributions)
            evaluate_springs(x, cgB, cgA_diag, cgA_offdiag, edges, rest_lengths, stiffness_stretch)

            # 4) assemble dense system
            A, b = assemble_dense_system(cgB, cgA_diag, edges, cgA_offdiag)

            reg = 1e-8
            rhs = b.copy()  # step direction depends on your sign convention
            try:
                # dx_flat = scipy.sparse.linalg.spsolve(scipy.sparse.csc_matrix(A + np.eye(3*num_verts)*reg), rhs)
                dx_flat = np.linalg.solve(A, rhs)
                dx = dx_flat.reshape(num_verts,3)

                if pring_detail:
                    np.set_printoptions(linewidth=200)
                    print(f'    Assembled system A:')
                    for row in A:
                        print(" ".join(f"{x:10.2f}" for x in row))
                    print(f'    Assembled system b: {b}')
                    print(f'    Assembled system x: {dx}')

            except np.linalg.LinAlgError:
                print("\nMatrix singular or ill-conditioned; cannot solve directly.")

            return dx
            
        num_newton_iters = 10
        for iter in range(num_newton_iters):

            init_energy, _ = compute_energy(x, x_tilde, is_fixed, vert_mass, substep_dt, edges, rest_lengths, stiffness_stretch)            

            dx = single_newton()

            alpha = 1.0
            line_search_iter = 0
            while True:
                curr_energy, energy_list = compute_energy(x + alpha * dx, x_tilde, is_fixed, vert_mass, substep_dt, edges, rest_lengths, stiffness_stretch)
                print(f'  In linesearch {line_search_iter} : alpha {alpha:.3e} energy {curr_energy:.6e} = {energy_list} (init {init_energy:.6e})')
                if curr_energy > init_energy and alpha > 1e-6:
                    alpha *= 0.5
                    line_search_iter += 1
                else:
                    if alpha < 1e-6:
                        RuntimeError(f'  Linesearch failed to find descent direction, stopping Newton.')
                        exit(1)
                    # if line_search_iter == 0:
                    #     print(f'  No linesearch needed')
                    break
                
            max_move = np.linalg.norm(dx, np.inf)
            print(f'   In iter {iter}, Infinity norm = {max_move:.6e}, energy = {curr_energy:.6e}, step length {alpha:.3e} after {line_search_iter} linesearch')

            x = x_iter_start + dx
            x_iter_start = x.copy()

            if np.linalg.norm(dx, np.inf) < 1e-2 * substep_dt:
                print(f"   Newton converged (inf norm of step < {1e-2 * substep_dt}).")
                return
            
        update_velocity(x, v, x_step_start, v_step_start, substep_dt, fix_scene=False, damping=0.0)
            







    for frame in range(1):
        print(f"\n=== Frame {frame} ===")
        single_frame()
        # print("Vertex positions:\n", x)
        # print("Vertex velocities:\n", v)
        # 保存为 .obj 文件
        def save_obj(filename, vertices, faces):
            with open(filename, 'w') as f:
                f.write(f"g\n")
                for v in vertices:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                f.write(f"g\n")
                for face in faces:
                    # OBJ 索引从1开始
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
                print(f'Saved {filename}')
                
        # x.T 是 (4,3) 顶点，faces 是 (2,3)
        # save_obj(f'Resources/OutputMesh/frame_{frame}.obj', x, faces)
