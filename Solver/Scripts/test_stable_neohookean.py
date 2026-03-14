"""
Deep check: verify the Term C formula via direct cofactor derivative.
The correct formula for d(cofF)/dF in vertex space is critical.
"""
import numpy as np

def E_to_lame(E, nu):
    mu  = E / (2.0 * (1.0 + nu))
    lmd = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lmd

mu, lmd = E_to_lame(1e5, 0.4)

def make_Dm_inv():
    v = np.array([
        [-0.2, -0.2, -0.2],
        [ 0.2, -0.2, -0.2],
        [-0.2,  0.2, -0.2],
        [-0.2, -0.2,  0.2],
    ])
    x0, x1, x2, x3 = v
    Dm = np.column_stack([x1-x0, x2-x0, x3-x0])
    return np.linalg.inv(Dm), abs(np.linalg.det(Dm)) / 6.0

Dm_inv, volume = make_Dm_inv()

def compute_B(Dm_inv):
    B = np.zeros((4, 3))
    B[1] = Dm_inv[:, 0]
    B[2] = Dm_inv[:, 1]
    B[3] = Dm_inv[:, 2]
    B[0] = -(B[1] + B[2] + B[3])
    return B

B = compute_B(Dm_inv)

def cofactor(F):
    cofF = np.zeros((3,3))
    for r in range(3):
        for c in range(3):
            rows = [i for i in range(3) if i != r]
            cols = [j for j in range(3) if j != c]
            cofF[r, c] = ((-1)**(r+c)) * np.linalg.det(F[np.ix_(rows, cols)])
    return cofF

def compute_F(x):
    x0, x1, x2, x3 = x
    Ds = np.column_stack([x1-x0, x2-x0, x3-x0])
    return Ds @ Dm_inv

# ---- Verify Term C via finite differences on cofactor ----
# K_C[a][b]_{ij} = coeff * d/dx_{beta,j} [sum_{r,c} cofF[r,c] * d(F[r,c])/dx_{alpha,i}]
# = coeff * sum_{r,c} d(cofF[r,c])/dF[s,t] * dF[s,t]/dx_{alpha,i} * dF[r,c]/dx_{beta,j}
# 
# Alternatively by direct finite diff of the force g[alpha]_i = sum_{r,c} P[r,c]*dF[r,c]/dx_{alpha,i}
# where P = mu*F + coeff_cof * cofF
#
# Let's compute the Hessian contribution from cofF term via FD on the cofF-force:

def cofF_force(x, coeff, alpha, i):
    """Force contribution from coeff*cofF on vertex alpha, spatial dim i."""
    F = compute_F(x)
    cofF = cofactor(F)
    # g_cofF[alpha][i] = sum_{r,c} coeff*cofF[r,c] * dF[r,c]/dx_{alpha,i}
    # dF[r,c]/dx_{alpha,i} = B[alpha,c] * delta_{r,i}
    # So g_cofF[alpha][i] = coeff * sum_c cofF[i,c] * B[alpha,c]
    val = 0.0
    for c in range(3):
        val += cofF[i, c] * B[alpha, c]
    return coeff * val

def cofF_force_all(x, coeff):
    """All 12 cofF force components."""
    g = np.zeros((4,3))
    for a in range(4):
        for i in range(3):
            g[a,i] = cofF_force(x, coeff, a, i)
    return g

# We want d(cofF_force_all)/dx, which gives the geometric stiffness (Term C)
# But coeff itself depends on I3, so we need to be careful
# Here we treat coeff as a constant to isolate Term C

v_rest = np.array([
    [-0.2, -0.2, -0.2],
    [ 0.2, -0.2, -0.2],
    [-0.2,  0.2, -0.2],
    [-0.2, -0.2,  0.2],
])

F_rest = compute_F(v_rest)
I3_rest = np.linalg.det(F_rest)
coeff_actual = lmd*(I3_rest - 1) - mu  # -mu at rest

# FD Term C (geometric stiffness = d(coeff*cofF_force)/dx with coeff frozen)
def fd_term_C(x, coeff, eps=1e-4):
    """FD Hessian of the cofF force (with frozen coeff = geometric stiffness term)."""
    H_C = np.zeros((12,12))
    g0 = cofF_force_all(x, coeff)
    for b in range(4):
        for j in range(3):
            xp = x.copy(); xp[b,j] += eps
            xm = x.copy(); xm[b,j] -= eps
            gp = cofF_force_all(xp, coeff)
            gm = cofF_force_all(xm, coeff)
            dg = (gp - gm) / (2*eps)
            for a in range(4):
                H_C[a*3:(a+1)*3, b*3+j] = dg[a]
    return H_C

print("Computing FD Term C (geometric stiffness) at rest...")
H_C_fd = fd_term_C(v_rest, 1.0, eps=1e-4)  # coeff=1, multiply by coeff_actual later
print("Done.")

# Now compute our GPU Term C formula (with coeff=1):
def gpu_term_C(F, B, coeff, volume):
    """GPU formula for Term C."""
    H = np.zeros((12,12))
    for alpha in range(4):
        for beta in range(4):
            cx = B[alpha,1]*B[beta,2] - B[alpha,2]*B[beta,1]
            cy = B[alpha,2]*B[beta,0] - B[alpha,0]*B[beta,2]
            cz = B[alpha,0]*B[beta,1] - B[alpha,1]*B[beta,0]
            w = np.array([cx, cy, cz])
            skew_w = np.array([
                [    0, -w[2],  w[1]],
                [ w[2],     0, -w[0]],
                [-w[1],  w[0],     0],
            ])
            FSkew = F @ skew_w
            for i in range(3):
                for j in range(3):
                    H[alpha*3+i, beta*3+j] += coeff * volume * FSkew[i, j]
    return H

H_C_gpu = gpu_term_C(F_rest, B, 1.0, volume)

print("\n--- FD Term C (coeff=1) eigenvalues ---")
print(np.sort(np.linalg.eigvalsh(H_C_fd * volume)))

print("\n--- GPU Term C (coeff=1) eigenvalues ---")
print(np.sort(np.linalg.eigvalsh(H_C_gpu)))

print(f"\nFD vs GPU Term C max diff (coeff=1, rest): {np.max(np.abs(H_C_fd*volume - H_C_gpu)):.4e}")

# ---- Check if GPU Term C formula is the TRANSPOSE of FD ----
H_C_gpu_T = H_C_gpu.T
print(f"\nFD vs GPU^T Term C max diff: {np.max(np.abs(H_C_fd*volume - H_C_gpu_T)):.4e}")

# ---- Look at specific blocks ----
print("\nFD Term C block [0,1] (coeff=1, vol=1):")
print((H_C_fd * volume)[:3, 3:6].round(6))

print("\nGPU Term C block [0,1] (coeff=1):")
print(H_C_gpu[:3, 3:6].round(6))

print("\nDiff block [0,1]:")
print((H_C_fd*volume - H_C_gpu)[:3, 3:6].round(6))

# ---- Also check skew formula directly ----
# The formula in GPU code: K_C[a][b]_{ij} = [F * skew(w)]_{ij}
# where w = B[alpha] x B[beta]
# skew(w)[row=j][col=k] convention gives:
# [F * skew(w)]_{ij} = sum_k F[ik] * skew(w)[kj]  (matrix multiply F (3x3) @ skew (3x3))
# F[ik] = F_np[i,k] (row-major convention)
# F_luisa[k][i] -> when we write F[k][i] in Luisa code, k is col, i is row
# -> F_luisa[k][i] = F_np[i,k] (standard)
# So FSkew = F_np @ skew_w is the correct numpy equivalent

# Verify with FD for a specific block
print("\n--- Verify specific cross product term via FD ---")
alpha, beta = 0, 1
i_test, j_test = 0, 1  # entry (0,1) of block [0,1]
eps = 1e-5

# FD: d(cofF_force[alpha,i])/dx[beta,j]  with coeff=1, at rest
def cofF_force_single(x, alpha, i):
    F = compute_F(x)
    cofF = cofactor(F)
    return sum(cofF[i, c] * B[alpha, c] for c in range(3))

xp = v_rest.copy(); xp[beta, j_test] += eps
xm = v_rest.copy(); xm[beta, j_test] -= eps
fd_val = (cofF_force_single(xp, alpha, i_test) - cofF_force_single(xm, alpha, i_test)) / (2*eps)
print(f"FD: d(cofF_force[{alpha},{i_test}])/dx[{beta},{j_test}] = {fd_val * volume:.6f}")
print(f"GPU: [F*skew(B[{alpha}]xB[{beta}])][{i_test},{j_test}]*vol = {H_C_gpu[alpha*3+i_test, beta*3+j_test]:.6f}")

# ---- What IS the correct Term C formula? ----
print("\n=== DERIVING CORRECT TERM C ===")
# g_cofF[alpha,i] = sum_c cofF[i,c] * B[alpha,c]
# 
# d(g_cofF[alpha,i])/dx[beta,j] = sum_c d(cofF[i,c])/dx[beta,j] * B[alpha,c]
# 
# d(cofF[i,c])/dx[beta,j]:
#   cofF[i,c] = cofactor(F)[i,c] = sum_{r!=i, s!=c} epsilon[i,r,?]*epsilon[c,s,?]*F[r,s]*F[??,??]
#   More precisely: cofF[i,c] = (F^{-T} * det(F))[i,c]
#   Or: cofF[i,c] = 0.5 * sum_{r,s,p,q} eps[i,r,s]*eps[c,p,q]*F[r,p]*F[s,q]
#
# dF[r,p]/dx[beta,j] = delta[r,j] * B[beta,p]  (row j of F, col p)
# Wait: F[r,p] = (Ds * Dm_inv)[r,p] = sum_k Ds[r,k] * Dm_inv[k,p]
# And Ds[r,k] = (x[k+1] - x[0])[r]
# dDs[r,k]/dx[beta,r'] = delta[r,r'] * (delta[beta,k+1] - delta[beta,0])
# dF[r,p]/dx[beta,r'] = delta[r,r'] * sum_k (delta[beta,k+1]-delta[beta,0]) * Dm_inv[k,p]
#                     = delta[r,r'] * B[beta,p]
# So dF[r,p]/dx[beta,j] = delta[r,j] * B[beta,p]
#
# d(cofF[i,c])/dx[beta,j]:
# cofF[i,c] = eps[i,r,s] * eps[c,p,q] * F[r,p] * F[s,q]  (using 3D Levi-Civita)
# (where repeated indices summed, i,c are fixed)
# Actually the correct formula is:
# cofF[i,c] = 0.5 * eps_{i,r,s} * eps_{c,p,q} * F_{r,p} * F_{s,q}
# 
# d(cofF[i,c])/dF[r0,p0] = eps_{i,r0,s} * eps_{c,p0,q} * F[s,q]  (from symmetry, factor 2 cancels)
# 
# So:
# d(cofF[i,c])/dx[beta,j] = sum_{r0,p0} d(cofF[i,c])/dF[r0,p0] * dF[r0,p0]/dx[beta,j]
#   = sum_{r0,p0} eps_{i,r0,s}*eps_{c,p0,q}*F[s,q] * delta[r0,j] * B[beta,p0]
#   = sum_{p0,s,q} eps_{i,j,s} * eps_{c,p0,q} * F[s,q] * B[beta,p0]
#   = eps_{i,j,s} * F[s,q] * (sum_{p0} eps_{c,p0,q} * B[beta,p0])
#   Let N[beta,c,q] = sum_{p0} eps_{c,p0,q} * B[beta,p0]  = (B[beta] x e_q)[c] ?
#   No: sum_p eps[c,p,q] * B[beta,p] = (epsilon_c * B[beta])[q] in cross product sense
#   Actually: sum_p eps[c,p,q] * B[beta,p] = (-1) * (B[beta] x e_c)[q]? Let me be explicit.

# Let's just verify numerically:
# K_C[alpha,beta]_{ij} = sum_c d(cofF[i,c])/dx[beta,j] * B[alpha,c]
# = sum_c [sum_{p0,s,q} eps[i,j,s]*eps[c,p0,q]*F[s,q]*B[beta,p0]] * B[alpha,c]
# = sum_{s,q,p0,c} eps[i,j,s] * eps[c,p0,q] * F[s,q] * B[beta,p0] * B[alpha,c]

def levi_civita(a,b,c):
    s = [(a,b,c),(b,c,a),(c,a,b)]
    if (a,b,c) in s: 
        t = [a,b,c]
        if len(set(t)) < 3: return 0
        if (a < b and b > c) or (a > b and b < c): 
            pass
        # Just use the sign
    perm = [a,b,c]
    if len(set(perm)) < 3: return 0
    # Count inversions
    inv = 0
    for x in range(3):
        for y in range(x+1,3):
            if perm[x] > perm[y]: inv += 1
    return 1 if inv % 2 == 0 else -1

eps = np.zeros((3,3,3))
for a in range(3):
    for b in range(3):
        for c in range(3):
            eps[a,b,c] = levi_civita(a,b,c)

print("\n--- Deriving correct Term C formula ---")
# K_C[alpha,beta]_{ij} = sum_{s,q,p0,c} eps[i,j,s] * eps[c,p0,q] * F[s,q] * B[beta,p0] * B[alpha,c]

def correct_term_C(F, B, coeff, volume):
    """Compute Term C using explicit Levi-Civita formula."""
    H = np.zeros((12,12))
    for alpha in range(4):
        for beta in range(4):
            for i in range(3):
                for j in range(3):
                    val = 0.0
                    for s in range(3):
                        for q_idx in range(3):
                            for p0 in range(3):
                                for c in range(3):
                                    val += (eps[i,j,s] * eps[c,p0,q_idx] * 
                                           F[s,q_idx] * B[beta,p0] * B[alpha,c])
                    H[alpha*3+i, beta*3+j] = coeff * volume * val
    return H

H_C_correct_lc = correct_term_C(F_rest, B, 1.0, volume)
print("\nCorrect Term C (Levi-Civita) eigenvalues:")
print(np.sort(np.linalg.eigvalsh(H_C_correct_lc)))

print(f"\nFD vs Levi-Civita Term C max diff: {np.max(np.abs(H_C_fd*volume - H_C_correct_lc)):.4e}")
print(f"GPU vs Levi-Civita Term C max diff: {np.max(np.abs(H_C_gpu - H_C_correct_lc)):.4e}")

print("\nLevi-Civita Term C block [0,1]:")
print(H_C_correct_lc[:3, 3:6].round(6))
print("\nGPU Term C block [0,1]:")
print(H_C_gpu[:3, 3:6].round(6))
print("\nFD Term C block [0,1]:")
print((H_C_fd * volume)[:3, 3:6].round(6))

# ---- Simplify the Levi-Civita formula ----
# sum_{s,q,p0,c} eps[i,j,s] * eps[c,p0,q] * F[s,q] * B[beta,p0] * B[alpha,c]
# = sum_s eps[i,j,s] * sum_{q,p0,c} eps[c,p0,q] * F[s,q] * B[beta,p0] * B[alpha,c]
# = sum_s eps[i,j,s] * sum_q F[s,q] * (sum_{c,p0} eps[c,p0,q] * B[alpha,c] * B[beta,p0])
# Let A_q = sum_{c,p0} eps[c,p0,q] * B[alpha,c] * B[beta,p0]
#         = (B[alpha] x B[beta])[q]  (cross product)
# So K_C = sum_{s,q} eps[i,j,s] * F[s,q] * (B[alpha] x B[beta])[q]
#        = sum_s eps[i,j,s] * F[s,:] . (B[alpha] x B[beta])
#        = sum_s eps[i,j,s] * (F @ cross)[s]
#        where cross = B[alpha] x B[beta]

def simplified_term_C(F, B, coeff, volume):
    H = np.zeros((12,12))
    for alpha in range(4):
        for beta in range(4):
            cross = np.cross(B[alpha], B[beta])
            Fcross = F @ cross  # (3,): Fcross[s] = F[s,:] . cross
            for i in range(3):
                for j in range(3):
                    val = 0.0
                    for s in range(3):
                        val += eps[i,j,s] * Fcross[s]
                    H[alpha*3+i, beta*3+j] = coeff * volume * val
    return H

H_C_simplified = simplified_term_C(F_rest, B, 1.0, volume)
print("\nSimplified Term C (via Levi-Civita) eigenvalues:")
print(np.sort(np.linalg.eigvalsh(H_C_simplified)))
print(f"FD vs simplified: {np.max(np.abs(H_C_fd*volume - H_C_simplified)):.4e}")
print(f"GPU vs simplified: {np.max(np.abs(H_C_gpu - H_C_simplified)):.4e}")

# ---- Expand eps[i,j,s] * Fcross[s] ----
# K_C_{ij} = sum_s eps[i,j,s] * Fcross[s]
# This is the (i,j) entry of skew(Fcross)!
# skew(w)[i,j] = sum_s eps[i,j,s] * w[s]? Let me verify.
# Actually: (w x v)[i] = sum_{j,k} eps[i,j,k] * w[j] * v[k]
# And the cross product matrix: [w]_x [i,j] = -sum_k eps[i,k,j] * w[k]?
# No. The skew-symmetric matrix corresponding to w:
# [w]_x = [[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]
# [w]_x [i,j] = sum_k eps[k,i,j] * w[k]  (note: eps[k,i,j] not eps[i,j,k]!)
# OR: sum_s eps[i,j,s] * w[s] = ?
# eps[i,j,s]: for i=0,j=1: eps[0,1,s] = 1 if s=2, -1 if s=1,0,2... 
# eps[0,1,2]=1, eps[0,1,0]=eps[0,1,1]=0  so sum_s eps[0,1,s]*w[s] = w[2]
# And skew(w)[0,1] = -w[2]  -> so sum_s eps[i,j,s]*w[s] = -skew(w)[i,j]?
# Wait: skew(w)[0,1] = -w[2], sum_s eps[0,1,s]*w[s] = w[2] -> opposite sign
# So K_C_{ij} = sum_s eps[i,j,s]*Fcross[s] = -skew(Fcross)[i,j] ?

print("\n--- Checking eps[i,j,s] vs skew convention ---")
Fcross_test = F_rest @ np.cross(B[0], B[1])
for i in range(3):
    for j in range(3):
        lc_val = sum(eps[i,j,s]*Fcross_test[s] for s in range(3))
        skew_val = np.array([[0,-Fcross_test[2],Fcross_test[1]],
                              [Fcross_test[2],0,-Fcross_test[0]],
                              [-Fcross_test[1],Fcross_test[0],0]])[i,j]
        if abs(lc_val) > 1e-10 or abs(skew_val) > 1e-10:
            print(f"  [{i},{j}]: LC={lc_val:.4f}, skew(Fcross)={skew_val:.4f}, ratio={lc_val/(skew_val+1e-20):.2f}")

# So K_C_{ij} = skew(Fcross)_{ij} or its negative?
H_C_skewFcross = np.zeros((12,12))
for alpha in range(4):
    for beta in range(4):
        cross = np.cross(B[alpha], B[beta])
        Fcross = F_rest @ cross
        # K_C_{ij} = sum_s eps[i,j,s]*Fcross[s] = skew(Fcross)[i,j]? or -skew?
        skewFcross = np.array([[0,-Fcross[2],Fcross[1]],
                                [Fcross[2],0,-Fcross[0]],
                                [-Fcross[1],Fcross[0],0]])
        for i in range(3):
            for j in range(3):
                val = sum(eps[i,j,s]*Fcross[s] for s in range(3))
                H_C_skewFcross[alpha*3+i, beta*3+j] = 1.0 * volume * val

print(f"\nFD vs skew(F*cross(B[a],B[b])): {np.max(np.abs(H_C_fd*volume - H_C_skewFcross)):.4e}")
print(f"GPU vs skew(F*cross(B[a],B[b])): {np.max(np.abs(H_C_gpu - H_C_skewFcross)):.4e}")

# Check transposed
print(f"\nFD vs skew^T: {np.max(np.abs(H_C_fd*volume - H_C_skewFcross.T)):.4e}")

# ---- Summary so far ----
print("\n=== SUMMARY ===")
print(f"GPU Term C formula: F @ skew(B[a]xB[b])  ->  error vs FD: {np.max(np.abs(H_C_fd*volume - H_C_gpu)):.4e}")
print(f"Correct formula: skew(F*(B[a]xB[b]))     ->  error vs FD: {np.max(np.abs(H_C_fd*volume - H_C_skewFcross)):.4e}")
print(f"Levi-Civita derivation                    ->  error vs FD: {np.max(np.abs(H_C_fd*volume - H_C_correct_lc)):.4e}")

print("\nBlock [0,1] comparison:")
print("FD:    ", (H_C_fd*volume)[:3, 3:6].round(4))
print("GPU:   ", H_C_gpu[:3, 3:6].round(4))
print("skewFC:", H_C_skewFcross[:3, 3:6].round(4))
