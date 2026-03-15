// Stable Neo-Hookean (SNHk) tetrahedral elastic energy.
//
// Reference implementations:
//   [1] Ando (2023) ppf-contact-solver/snhk.hpp   (GPU analytic path)
//   [2] Smith et al. (2018) "Stable Neo-Hookean Flesh Simulation"
//   [3] Xu et al. (2015) "Nonlinear Material Design Using Principal Stretches"
//
// Energy function (polynomial variant from [1]):
//   I2 = ||F||_F^2     (sum of squared singular values)
//   I3 = det(F)        (= product of singular values)
//   Psi = mu/2*(I2-3) - mu*(I3-1) + lmd/2*(I3-1)^2
//
// CPU gradient/Hessian pipeline (SVD-based, eigenanalysis PD projection):
//   1. F = Ds * Dm_inv
//   2. F = U * diag(S) * V^T
//   3. dPsi/dS, d2Psi/dS2  (closed-form)
//   4. eigenanalysis:
//      - Diagonal modes (3): D_k = flatten(U[:,k] outer V[:,k]), coeff = d2_diag eigenvalues
//      - Twist modes (3 pairs): q+/q- of (e_ij +/- e_ji)/sqrt2, PD-clamped
//   5. convert to vertex space via B-selector outer products
//
// GPU gradient/Hessian pipeline (analytical, no SVD):
//   Gradient: 1st PK stress P = mu*F + (lmd*(I3-1) - mu) * cofF
//   Hessian:  three terms (A=mu*I, B=lmd*cofF*cofF^T, C=geo.stiffness clamped)
//
#pragma once

#include "Core/float_n.h"
#include "Core/float_nxn.h"
#include "Core/lc_to_eigen.h"
#include <luisa/luisa-compute.h>
#include <cmath>

namespace lcs
{
	namespace TetElasticEnergyUtils
	{

		// ============================================================
		// SNHk in singular-value space
		// Energy:  Psi = mu/2*(I2-3) - mu*(I3-1) + lmd/2*(I3-1)^2
		//   I2 = a^2+b^2+c^2,  I3 = a*b*c
		// ============================================================

		inline float snhk_energy(float a, float b, float c, float mu, float lmd)
		{
			float I2 = a * a + b * b + c * c;
			float I3 = a * b * c;
			return 0.5f * mu * (I2 - 3.0f)
				- mu * (I3 - 1.0f)
				+ 0.5f * lmd * (I3 - 1.0f) * (I3 - 1.0f);
		}

		// dPsi/da_i
		inline void snhk_deda(float a, float b, float c, float mu, float lmd, float deda[3])
		{
			float I3 = a * b * c;
			float coeff = lmd * (I3 - 1.0f) - mu; // = lmd*(I3-1) - mu
			// dPsi/da = mu*a + coeff * (bc)
			deda[0] = mu * a + coeff * (b * c);
			deda[1] = mu * b + coeff * (a * c);
			deda[2] = mu * c + coeff * (a * b);
		}

		// d2Psi / (da_i * da_j) - Correct analytical Hessian formula
		// Key insight: Diagonal terms include μ term, off-diagonal terms include coeff term
		inline void snhk_d2ed2a(float a, float b, float c, float mu, float lmd, float d2[3][3])
		{
			float I3 = a * b * c;
			float coeff = lmd * (I3 - 1.0f) - mu; // [λ(I₃-1) - μ]

			// Diagonal:  d2Psi/da^2 = μ + λ*(bc)^2
			d2[0][0] = mu + lmd * (b * c) * (b * c);
			d2[1][1] = mu + lmd * (a * c) * (a * c);
			d2[2][2] = mu + lmd * (a * b) * (a * b);

			// Off-diagonal:  d2Psi/(da*db) = λ*I3*c + coeff*c
			//   This can also be written as: λ*a*b*c*c + [λ(I₃-1)-μ]*c
			d2[0][1] = d2[1][0] = lmd * I3 * c + coeff * c;
			d2[0][2] = d2[2][0] = lmd * I3 * b + coeff * b;
			d2[1][2] = d2[2][1] = lmd * I3 * a + coeff * a;
		}

		// ============================================================
		// Eigenanalysis helpers
		// ============================================================

		// 2x2 symmetric eigendecomposition for PD projection.
		// Input: [p, q; q, r].  Output: eigenvalues L[0..1], eigenvectors as columns of out_cos/sin
		// We clamp negative eigenvalues in the caller.
		static inline void sym2x2_eig(float p, float q, float r,
			float& L0, float& L1,
			float& c, float& s)
		{
			// Jacobi rotation
			float diff = p - r;
			float disc = std::sqrt(diff * diff + 4.0f * q * q);
			L0 = 0.5f * (p + r + disc);
			L1 = 0.5f * (p + r - disc);

			if (std::abs(q) < 1e-12f)
			{
				c = 1.0f;
				s = 0.0f;
			}
			else
			{
				float t = diff / (2.0f * q);
				float sgn = (t >= 0.0f) ? 1.0f : -1.0f;
				float tan_val = sgn / (std::abs(t) + std::sqrt(1.0f + t * t));
				float cos_val = 1.0f / std::sqrt(1.0f + tan_val * tan_val);
				c = cos_val;
				s = tan_val * cos_val;
			}
		}

		// ============================================================
		// eigenanalysis: singular-value space -> F-space Hessian
		//
		// Flat index convention: flat(F)[r + c*3] = F[c][r]
		//   (column-major within F, concatenated column by column)
		//
		// Total 9x9 Hessian = sum of rank-1 PD-clamped modes:
		//
		// 1. "Stretch" modes from diagonal basis D_k = U[:,k] outer V[:,k]:
		//    The 3x3 sub-Hessian in {D0, D1, D2} space is exactly d2[i][j].
		//    Eigendecompose d2 (3x3 symmetric) -> get 3 eigenmodes with PD clamp.
		//
		// 2. "Twist" modes for each off-diagonal pair (i, j), i < j:
		//    q+ = (e_ij + e_ji) / sqrt(2),  lambda+ = (d2[i][j] + sigma_ij) / 2
		//    q- = (e_ij - e_ji) / sqrt(2),  lambda- = (d2[i][j] - sigma_ij) / 2
		//    where e_ij = flatten(U[:,i] outer V[:,j])
		//    sigma_ij = (deda[i] + deda[j]) / (S[i] + S[j])
		//    All lambdas clamped to >= 0.
		// ============================================================

		// Build U[:,k] outer V[:,k] flattened as a 9-vector.
		// flat[r + c*3] = U[k][r] * V[k][c]   (col-major convention)
		static inline void make_stretch_mode(int k,
			const luisa::float3x3&				 U,
			const luisa::float3x3&				 V,
			float								 out[9])
		{
			for (int c = 0; c < 3; c++)
				for (int r = 0; r < 3; r++)
					out[r + c * 3] = U[k][r] * V[k][c];
		}

		// Build U[:,ui] outer V[:,vj] flattened.
		static inline void make_twist_mode(int ui, int vj,
			const luisa::float3x3& U,
			const luisa::float3x3& V,
			float				   out[9])
		{
			for (int c = 0; c < 3; c++)
				for (int r = 0; r < 3; r++)
					out[r + c * 3] = U[ui][r] * V[vj][c];
		}

		// Add coeff * q * q^T to H (PD clamp: skip if coeff <= 0).
		static inline void add_rank1(lcs::float9x9& H, const float q[9], float coeff)
		{
			if (coeff <= 0.0f)
				return;
			for (int a = 0; a < 9; a++)
				for (int b = 0; b < 9; b++)
					H.scalar(a, b) += coeff * q[a] * q[b];
		}

		// Full 9x9 Hessian via eigenanalysis with PD projection.
		inline lcs::float9x9 eigenanalysis_hessian(
			const luisa::float3x3& U,
			const luisa::float3x3& V,
			const luisa::float3&   S,
			const float			   deda[3],
			const float			   d2[3][3],
			float				   eps = 1e-8f)
		{
			lcs::float9x9 H;
			H.set_zero();

			// ---- Part 1: Stretch modes (diagonal modes D0, D1, D2) ----
			// The 3x3 matrix d2 is the Hessian in the {D0,D1,D2} subspace.
			// We eigendecompose d2 to get PD-projected contribution.
			//
			// d2 is 3x3 symmetric. We do a full Jacobi-style diagonalization:
			// Use 3 Givens rotations (one per off-diagonal pair).
			// Result: d2 = R * diag(lam) * R^T
			// Each mode: q_k = sum_i R[i][k] * D_i,  coeff = max(lam_k, 0)
			{
				float D[3][9];
				for (int k = 0; k < 3; k++)
					make_stretch_mode(k, U, V, D[k]);

				// Copy d2 to a working matrix (symmetric, we only need upper triangle)
				float M[3][3];
				for (int i = 0; i < 3; i++)
					for (int j = 0; j < 3; j++)
						M[i][j] = d2[i][j];

				// Accumulated rotation (starts as identity)
				float R[3][3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };

				// Jacobi iterations (10 sweeps is overkill for 3x3, but safe)
				for (int iter = 0; iter < 20; iter++)
				{
					// Find max off-diagonal element
					int	  pi = 0, pj = 1;
					float maxval = std::abs(M[0][1]);
					if (std::abs(M[0][2]) > maxval)
					{
						maxval = std::abs(M[0][2]);
						pi = 0;
						pj = 2;
					}
					if (std::abs(M[1][2]) > maxval)
					{
						maxval = std::abs(M[1][2]);
						pi = 1;
						pj = 2;
					}
					if (maxval < 1e-12f)
						break;

					// 2x2 Jacobi rotation to zero out M[pi][pj]
					float Mpp = M[pi][pi], Mpq = M[pi][pj], Mqq = M[pj][pj];
					float L0, L1, c, s;
					sym2x2_eig(Mpp, Mpq, Mqq, L0, L1, c, s);

					// Apply rotation: M = G^T M G
					// G = identity with G[pi][pi]=c, G[pj][pj]=c, G[pi][pj]=s, G[pj][pi]=-s
					// Update M row/col p and q
					// Column update: for each row k != pi, pj
					for (int k = 0; k < 3; k++)
					{
						if (k == pi || k == pj)
							continue;
						float tmp_ip = c * M[k][pi] + s * M[k][pj];
						float tmp_iq = -s * M[k][pi] + c * M[k][pj];
						M[k][pi] = M[pi][k] = tmp_ip;
						M[k][pj] = M[pj][k] = tmp_iq;
					}
					// Diagonal blocks
					M[pi][pi] = c * c * Mpp + 2.0f * c * s * Mpq + s * s * Mqq;
					M[pj][pj] = s * s * Mpp - 2.0f * c * s * Mpq + c * c * Mqq;
					M[pi][pj] = M[pj][pi] = 0.0f;

					// Update rotation matrix R = R * G
					for (int k = 0; k < 3; k++)
					{
						float tmp_i = c * R[k][pi] + s * R[k][pj];
						float tmp_j = -s * R[k][pi] + c * R[k][pj];
						R[k][pi] = tmp_i;
						R[k][pj] = tmp_j;
					}
				}

				// Now M[i][i] are eigenvalues, R[:,k] are eigenvectors
				for (int k = 0; k < 3; k++)
				{
					float lam = M[k][k];
					// Build eigenvector in 9-space: q_k = sum_i R[i][k] * D[i]
					float q[9] = { 0 };
					for (int i = 0; i < 3; i++)
						for (int e = 0; e < 9; e++)
							q[e] += R[i][k] * D[i][e];
					add_rank1(H, q, lam); // PD clamp inside add_rank1
				}
			}

			// ---- Part 2: Twist modes for each off-diagonal pair (i < j) ----
			for (int i = 0; i < 3; i++)
			{
				for (int j = i + 1; j < 3; j++)
				{
					float eij[9], eji[9];
					make_twist_mode(i, j, U, V, eij);
					make_twist_mode(j, i, U, V, eji);

					float denom = S[i] + S[j];
					float sigma_ij = (std::abs(denom) > eps)
						? (deda[i] + deda[j]) / denom
						: 0.0f;

					float lambda_plus = 0.5f * (d2[i][j] + sigma_ij);
					float lambda_minus = 0.5f * (d2[i][j] - sigma_ij);

					float sqrt2_inv = 0.7071067811865475f;
					float qp[9], qm[9];
					for (int e = 0; e < 9; e++)
					{
						qp[e] = (eij[e] + eji[e]) * sqrt2_inv;
						qm[e] = (eij[e] - eji[e]) * sqrt2_inv;
					}
					add_rank1(H, qp, lambda_plus);
					add_rank1(H, qm, lambda_minus);
				}
			}

			return H;
		}

		// dPsi/dF = U * diag(deda) * V^T
		// Original implementation - keep as-is since modifications made it worse
		inline luisa::float3x3 eigenanalysis_force(
			const luisa::float3x3& U,
			const luisa::float3x3& V,
			const float			   deda[3])
		{
			// dedF[c][r] = sum_k  U[k][r] * deda[k] * V[k][c]
			luisa::float3x3 result;
			for (int c = 0; c < 3; c++)
				for (int r = 0; r < 3; r++)
				{
					float s = 0.0f;
					for (int k = 0; k < 3; k++)
						s += U[k][r] * deda[k] * V[k][c];
					result[c][r] = s;
				}
			return result;
		}

		// ============================================================
		// Gauss-Newton Hessian approximation: H_GN = g * g^T / normg^2
		// This is always PSD and may improve convergence in non-convex regions.
		// ============================================================
		inline lcs::float9x9 gauss_newton_hessian(
			const luisa::float3x3& dPdF)
		{
			lcs::float9x9 H;
			H.set_zero();

			// Flatten dPdF to 9-vector (column-major: [c0r0, c0r1, c0r2, c1r0, ...])
			float g[9];
			for (int c = 0; c < 3; c++)
				for (int r = 0; r < 3; r++)
					g[r + c * 3] = dPdF[c][r];

			// Compute norm^2
			float norm_sq = 0.0f;
			for (int i = 0; i < 9; i++)
				norm_sq += g[i] * g[i];

			if (norm_sq < 1e-16f)
			{
				// Zero gradient: return zero Hessian
				return H;
			}

			// H = g * g^T / norm_sq
			for (int i = 0; i < 9; i++)
				for (int j = 0; j < 9; j++)
					H.scalar(i, j) = g[i] * g[j] / norm_sq;

			return H;
		}

		// ============================================================
		// B-selector based conversion: F-space -> vertex space
		// B[alpha][k]: coefficient that vertex alpha contributes to F column k
		// alpha=0..3 (vertices), k=0..2 (F column index)
		// ============================================================

		// Compute B selectors from Dm_inv (col-major: Dm_inv[col][row])
		// F = Ds * Dm_inv  =>  dF[c][r] / dx_{alpha,r} = B[alpha][c]
		inline void compute_B(const luisa::float3x3& Dm_inv, float B[4][3])
		{
			// Vertex 1,2,3 correspond to Dm_inv columns 0,1,2
			for (int k = 0; k < 3; k++)
			{
				B[1][k] = Dm_inv[0][k]; // Dm_inv col 0, row k
				B[2][k] = Dm_inv[1][k]; // Dm_inv col 1, row k
				B[3][k] = Dm_inv[2][k]; // Dm_inv col 2, row k
				B[0][k] = -(B[1][k] + B[2][k] + B[3][k]);
			}
		}

		// Convert gradient from F-space to vertex space.
		// gradient[alpha]_i = sum_c dedF[c][i] * B[alpha][c]
		inline void convert_force(
			const luisa::float3x3& dedF, // col-major: dedF[col][row]
			const float			   B[4][3],
			luisa::float3		   gradient[4])
		{
			for (int alpha = 0; alpha < 4; alpha++)
			{
				float g[3] = { 0.0f, 0.0f, 0.0f };
				for (int c = 0; c < 3; c++)
					for (int i = 0; i < 3; i++)
						g[i] += dedF[c][i] * B[alpha][c];
				gradient[alpha] = luisa::make_float3(g[0], g[1], g[2]);
			}
		}

		// Convert 9x9 F-space Hessian to vertex-space 3x3 blocks.
		//
		// K[alpha][beta]_{i,j}
		//   = sum_{ca, cb}  H9[i + ca*3][j + cb*3] * B[alpha][ca] * B[beta][cb]
		//
		// H9 uses flat index: flat(F)[r + c*3] -> H9.scalar(r + c*3, r' + c'*3)
		inline void convert_hessian(
			const lcs::float9x9& H9,
			const float			 B[4][3],
			luisa::float3x3		 hessian[16]) // hessian[alpha*4+beta]
		{
			for (int alpha = 0; alpha < 4; alpha++)
			{
				for (int beta = 0; beta < 4; beta++)
				{
					luisa::float3x3 K;
					for (int j = 0; j < 3; j++)		// col of K (spatial dim of beta)
						for (int i = 0; i < 3; i++) // row of K (spatial dim of alpha)
						{
							float val = 0.0f;
							for (int ca = 0; ca < 3; ca++)	   // F column for alpha
								for (int cb = 0; cb < 3; cb++) // F column for beta
									val += H9.scalar(i + ca * 3, j + cb * 3)
										* B[alpha][ca] * B[beta][cb];
							K[j][i] = val; // col-major: K[col][row]
						}
					hessian[alpha * 4 + beta] = K;
				}
			}
		}

		// ============================================================
		// Public CPU interface
		// ============================================================

		inline float compute_energy(
			const luisa::float3& x0, const luisa::float3& x1,
			const luisa::float3& x2, const luisa::float3& x3,
			const luisa::float3x3& Dm_inv,
			float mu, float lambda, float volume)
		{
			luisa::float3x3 Ds = luisa::make_float3x3(x1 - x0, x2 - x0, x3 - x0);
			luisa::float3x3 F = Ds * Dm_inv;

			luisa::float3x3															   U, V;
			luisa::float3															   S;
			Eigen::JacobiSVD<EigenFloat3x3, Eigen::ComputeFullU | Eigen::ComputeFullV> svd;
			svd.compute(float3x3_to_eigen3x3(F));
			U = eigen3x3_to_float3x3(svd.matrixU());
			V = eigen3x3_to_float3x3(svd.matrixV());
			auto singular_values = svd.singularValues();
			S = luisa::make_float3(singular_values(0), singular_values(1), singular_values(2));

			return volume * snhk_energy(S[0], S[1], S[2], mu, lambda);
		}

		inline void compute_gradient_hessian(
			const luisa::float3& x0, const luisa::float3& x1,
			const luisa::float3& x2, const luisa::float3& x3,
			const luisa::float3x3& Dm_inv,
			float mu, float lambda, float volume,
			luisa::float3	gradient[4],
			luisa::float3x3 hessian[16],
			bool			use_gauss_newton = false)
		{
			// 1. Deformation gradient
			luisa::float3x3 Ds = luisa::make_float3x3(x1 - x0, x2 - x0, x3 - x0);
			luisa::float3x3 F = Ds * Dm_inv;

			// 2. SVD:  F = U * diag(S) * V^T
			// Note: float3x3 is column-major; float3x3_to_eigen3x3 converts it to
			// row-major for Eigen's SVD. The result U, V from Eigen correspond to
			// the transposed F. This is corrected in eigenanalysis_force indexing.
			luisa::float3x3															   U, V;
			luisa::float3															   S;
			Eigen::JacobiSVD<EigenFloat3x3, Eigen::ComputeFullU | Eigen::ComputeFullV> svd;
			svd.compute(float3x3_to_eigen3x3(F));
			U = eigen3x3_to_float3x3(svd.matrixU());
			V = eigen3x3_to_float3x3(svd.matrixV());
			auto singular_values = svd.singularValues();
			S = luisa::make_float3(singular_values(0), singular_values(1), singular_values(2));

			// 3. SNHk derivatives in singular-value space
			float deda[3];
			float d2[3][3];
			snhk_deda(S[0], S[1], S[2], mu, lambda, deda);
			snhk_d2ed2a(S[0], S[1], S[2], mu, lambda, d2);

			// 4. Map gradient to F-space
			// Standard formula: dPsi/dF = U * diag(deda) * V^T
			// In col-major: dedF[c][r] = sum_k U[k][r] * deda[k] * V[k][c]
			// (verify this is correct for transposed F)
			luisa::float3x3 dedF = eigenanalysis_force(U, V, deda);

			// 5. Map Hessian to F-space (9x9)
			lcs::float9x9 d2edF2;
			if (use_gauss_newton)
			{
				// Gauss-Newton approximation: H_GN = g * g^T / norm(g)^2
				// Always PSD, may improve convergence away from convex regions
				d2edF2 = gauss_newton_hessian(dedF);
			}
			else
			{
				// Full Hessian with eigenanalysis and PD projection
				d2edF2 = eigenanalysis_hessian(U, V, S, deda, d2);
			}

			// 6. B selectors from Dm_inv
			float B[4][3];
			compute_B(Dm_inv, B);

			// 7. Convert to vertex space and scale by volume
			luisa::float3	g_raw[4];
			luisa::float3x3 h_raw[16];
			convert_force(dedF, B, g_raw);
			convert_hessian(d2edF2, B, h_raw);

			for (int a = 0; a < 4; a++)
				gradient[a] = volume * g_raw[a];
			for (int ab = 0; ab < 16; ab++)
				for (int c = 0; c < 3; c++)
					for (int r = 0; r < 3; r++)
						hessian[ab][c][r] = volume * h_raw[ab][c][r];
		}

		// ============================================================
		// GPU (LuisaCompute DSL) path
		//
		// Gradient (1st PK stress):
		//   P = mu*F + (lmd*(I3-1) - mu) * cofF
		//   g[alpha]_i = volume * sum_c P[c][i] * B[alpha][c]
		//
		// Hessian = Term A + Term B + Term C (PD-safe)
		//
		//   Term A: mu * I_{9x9}
		//     -> K_A[a][b]_{ij} = mu * dot(B[a], B[b]) * delta_{ij}
		//
		//   Term B: lmd * flat(cofF) * flat(cofF)^T
		//     -> Let  c_a[i] = sum_k cofF[k][i] * B[a][k]
		//     -> K_B[a][b]_{ij} = lmd * c_a[i] * c_b[j]
		//
		//   Note: Term A + Term B alone is always PSD (Term A is strictly PD,
		//   Term B is rank-1 PSD; their sum is PSD).
		//
		//   Term C: coeff_C * d(cofF)/dF  [PD-clamped coefficient]
		//     Derived via Levi-Civita contraction:
		//       d(cofF[m,c])/dF[n,l] = sum_{s,q} eps[m,n,s] * eps[c,l,q] * F[s,q]
		//     After contracting with B selectors:
		//       K_C[a,b]_{mn} = coeff_C * sum_s eps[m,n,s] * (F @ (B[a]xB[b]))[s]
		//                     = coeff_C * (-skew(F @ (B[a]xB[b])))[m,n]
		//
		//     where:
		//       cross = B[a] x B[b]   (standard 3D cross product)
		//       Fc    = F * cross      (3x3 matrix times 3-vector = 3-vector)
		//       -skew(Fc)[i,j]:
		//         row 0: [0,  Fc.z, -Fc.y]
		//         row 1: [-Fc.z,  0,  Fc.x]
		//         row 2: [Fc.y, -Fc.x,  0]
		//
		//   PSD Safety for Term C:
		//     The geometric stiffness tensor G = d(cofF)/dF (as 9x9) is SYMMETRIC
		//     but INDEFINITE: its min eigenvalue satisfies
		//       min_eig(G) >= -(sigma_1(F) + sigma_2(F))
		//     where sigma_1 >= sigma_2 >= sigma_3 are the singular values of F.
		//
		//     For H9 = mu*I + coeff_C * G to be PSD:
		//       coeff_C * |min_eig_G| <= mu
		//       coeff_C <= mu / (sigma_1 + sigma_2)
		//
		//     Since sigma_1 + sigma_2 <= sqrt(2) * ||F||_F  (by Cauchy-Schwarz),
		//     a sufficient PSD condition is:
		//       coeff_C_safe = min(coeff_C_raw, mu / (sqrt(2) * ||F||_F))
		//
		//     This clamp is applied below.  The unclamped value is
		//       coeff_C_raw = max(lmd*(I3-1) - mu, 0)
		// ============================================================

		inline luisa::compute::Float compute_energy_gpu(
			const luisa::compute::Float3&	x0,
			const luisa::compute::Float3&	x1,
			const luisa::compute::Float3&	x2,
			const luisa::compute::Float3&	x3,
			const luisa::compute::Float3x3& Dm_inv,
			const luisa::compute::Float		mu,
			const luisa::compute::Float		lambda,
			const luisa::compute::Float		volume)
		{
			using namespace luisa::compute;

			Float3x3 Ds = make_float3x3(x1 - x0, x2 - x0, x3 - x0);
			Float3x3 F = Ds * Dm_inv;

			// I2 = ||F||_F^2
			Float I2 = 0.0f;
			for (int c = 0; c < 3; c++)
				for (int r = 0; r < 3; r++)
					I2 = I2 + F[c][r] * F[c][r];

			Float I3 = determinant(F);

			Float psi = 0.5f * mu * (I2 - 3.0f)
				- mu * (I3 - 1.0f)
				+ 0.5f * lambda * (I3 - 1.0f) * (I3 - 1.0f);

			return volume * psi;
		}

		inline void compute_gradient_hessian_gpu(
			const luisa::compute::Float3&	x0,
			const luisa::compute::Float3&	x1,
			const luisa::compute::Float3&	x2,
			const luisa::compute::Float3&	x3,
			const luisa::compute::Float3x3& Dm_inv,
			const luisa::compute::Float		mu,
			const luisa::compute::Float		lambda,
			const luisa::compute::Float		volume,
			luisa::compute::Float3			gradient[4],
			luisa::compute::Float3x3		hessian[16])
		{
			using namespace luisa::compute;

			Float3x3 Ds = make_float3x3(x1 - x0, x2 - x0, x3 - x0);
			Float3x3 F = Ds * Dm_inv;

			Float I3 = determinant(F);

			// ---- Cofactor matrix (exact, no division by det) ----
			// cofF[col c][row r] = cofactor of F at (row=r, col=c)
			// For col-major F3x3: F[col][row], so F[c][r] is entry (row=r, col=c).
			// cofactor(r,c) = (-1)^(r+c) * minor(r,c)
			Float3x3 cofF;
			// Col 0 of cofF (cofactors of column 0 of F, i.e. partial det w.r.t. F[:,0])
			cofF[0][0] = F[1][1] * F[2][2] - F[2][1] * F[1][2];	   // cof(0,0) = M11 [rows 1,2 cols 1,2]
			cofF[0][1] = -(F[1][0] * F[2][2] - F[2][0] * F[1][2]); // cof(1,0)
			cofF[0][2] = F[1][0] * F[2][1] - F[2][0] * F[1][1];	   // cof(2,0)
			// Col 1
			cofF[1][0] = -(F[0][1] * F[2][2] - F[2][1] * F[0][2]); // cof(0,1)
			cofF[1][1] = F[0][0] * F[2][2] - F[2][0] * F[0][2];	   // cof(1,1)
			cofF[1][2] = -(F[0][0] * F[2][1] - F[2][0] * F[0][1]); // cof(2,1)
			// Col 2
			cofF[2][0] = F[0][1] * F[1][2] - F[1][1] * F[0][2];	   // cof(0,2)
			cofF[2][1] = -(F[0][0] * F[1][2] - F[1][0] * F[0][2]); // cof(1,2)
			cofF[2][2] = F[0][0] * F[1][1] - F[1][0] * F[0][1];	   // cof(2,2)

			// ---- 1st PK stress: P = mu*F + coeff_cof * cofF ----
			Float	 coeff_cof = lambda * (I3 - 1.0f) - mu;
			Float3x3 P;
			for (int c = 0; c < 3; c++)
				for (int r = 0; r < 3; r++)
					P[c][r] = mu * F[c][r] + coeff_cof * cofF[c][r];

			// ---- B selectors ----
			Float B[4][3];
			for (int k = 0; k < 3; k++)
			{
				B[1][k] = Dm_inv[0][k];
				B[2][k] = Dm_inv[1][k];
				B[3][k] = Dm_inv[2][k];
				B[0][k] = -(B[1][k] + B[2][k] + B[3][k]);
			}

			// ---- Gradient: g[alpha]_i = volume * sum_c P[c][i] * B[alpha][c] ----
			for (int alpha = 0; alpha < 4; alpha++)
			{
				Float g[3] = { 0.0f, 0.0f, 0.0f };
				for (int c = 0; c < 3; c++)
					for (int i = 0; i < 3; i++)
						g[i] = g[i] + P[c][i] * B[alpha][c];
				gradient[alpha] = volume * make_float3(g[0], g[1], g[2]);
			}

			// ---- Precompute cof_a[alpha][i] = sum_c cofF[c][i] * B[alpha][c] ----
			Float cof_a[4][3];
			for (int alpha = 0; alpha < 4; alpha++)
				for (int i = 0; i < 3; i++)
				{
					cof_a[alpha][i] = 0.0f;
					for (int c = 0; c < 3; c++)
						cof_a[alpha][i] = cof_a[alpha][i] + cofF[c][i] * B[alpha][c];
				}

			// ---- PD-safe coefficient for Term C ----
			// coeff_C_raw = max(lmd*(I3-1) - mu, 0)
			// The geometric stiffness G has min_eig(G) >= -(sigma_1+sigma_2).
			// For H9 = mu*I + coeff_C*G to be PSD, we need:
			//   coeff_C <= mu / (sigma_1 + sigma_2)
			// Since sigma_1+sigma_2 <= sqrt(2)*||F||_F, the sufficient condition is:
			//   coeff_C_safe = min(coeff_C_raw, mu / (sqrt(2) * ||F||_F))
			Float coeff_C_raw = max(coeff_cof, Float(0.0f));
			// Frobenius norm of F: ||F||_F = sqrt(sum_{c,r} F[c][r]^2)
			Float F_frob2 = 0.0f;
			for (int c = 0; c < 3; c++)
				for (int r = 0; r < 3; r++)
					F_frob2 = F_frob2 + F[c][r] * F[c][r];
			Float F_frob = luisa::compute::sqrt(F_frob2 + Float(1e-14f));
			// Safe upper bound: mu / (sqrt(2) * ||F||_F)
			Float coeff_C_limit = mu / (Float(1.4142135f) * F_frob);
			Float coeff_C = min(coeff_C_raw, coeff_C_limit);

			// ---- Hessian blocks ----
			for (int alpha = 0; alpha < 4; alpha++)
			{
				for (int beta = 0; beta < 4; beta++)
				{
					// Term A dot product
					Float BdotB = 0.0f;
					for (int k = 0; k < 3; k++)
						BdotB = BdotB + B[alpha][k] * B[beta][k];

					// Term C: cross product  c = B[alpha] x B[beta]
					Float cx = B[alpha][1] * B[beta][2] - B[alpha][2] * B[beta][1];
					Float cy = B[alpha][2] * B[beta][0] - B[alpha][0] * B[beta][2];
					Float cz = B[alpha][0] * B[beta][1] - B[alpha][1] * B[beta][0];

					// Fc = F * c  (3x3 col-major F times 3-vector c)
					// F is col-major: F[col][row], so F[col][row] = F_numpy[row,col]
					// Fc[r] = sum_k F[k][r] * c[k]
					Float Fc0 = F[0][0] * cx + F[1][0] * cy + F[2][0] * cz; // Fc[0]
					Float Fc1 = F[0][1] * cx + F[1][1] * cy + F[2][1] * cz; // Fc[1]
					Float Fc2 = F[0][2] * cx + F[1][2] * cy + F[2][2] * cz; // Fc[2]

					// K_C[alpha][beta]_{ij} = coeff_C * (-skew(Fc))[i,j]
					//
					// Standard skew(v)[i,j]:
					//   row 0: [ 0,  -v.z,  v.y]
					//   row 1: [ v.z,   0, -v.x]
					//   row 2: [-v.y,  v.x,  0 ]
					//
					// -skew(Fc)[i,j]:
					//   row 0: [ 0,   Fc.z, -Fc.y]  ->  [i=0,j=0]:0, [0,1]:Fc2, [0,2]:-Fc1
					//   row 1: [-Fc.z,  0,   Fc.x]  ->  [1,0]:-Fc2,  [1,1]:0,   [1,2]: Fc0
					//   row 2: [ Fc.y,-Fc.x,  0  ]  ->  [2,0]: Fc1,  [2,1]:-Fc0, [2,2]:0

					Float3x3 K;
					for (int j = 0; j < 3; j++)		// col = beta spatial dim
						for (int i = 0; i < 3; i++) // row = alpha spatial dim
						{
							// Term A
							Float val = mu * BdotB * (i == j ? Float(1.0f) : Float(0.0f));

							// Term B
							val = val + lambda * cof_a[alpha][i] * cof_a[beta][j];

							// Term C: -skew(Fc)[i,j]
							Float cross_term = 0.0f;
							if (i == 0)
							{
								if (j == 1)
									cross_term = Fc2;
								else if (j == 2)
									cross_term = -Fc1;
							}
							else if (i == 1)
							{
								if (j == 0)
									cross_term = -Fc2;
								else if (j == 2)
									cross_term = Fc0;
							}
							else
							{ // i == 2
								if (j == 0)
									cross_term = Fc1;
								else if (j == 1)
									cross_term = -Fc0;
							}

							val = val + coeff_C * cross_term;

							K[j][i] = volume * val; // col-major: K[col][row]
						}
					hessian[alpha * 4 + beta] = K;
				}
			}
		}

	} // namespace TetElasticEnergyUtils
} // namespace lcs
