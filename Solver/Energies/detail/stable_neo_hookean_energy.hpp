#pragma once

#include "Core/float_n.h"
#include "Core/scalar.h"
#include "Core/svd_3x3.h"
#include "Energies/detail/energy_detail_common.hpp"
#include "luisa/core/basic_types.h"

namespace lcs::detail::stable_neo_hookean_energy
{
	namespace
	{
		template <typename Float, typename Float3, typename Float3x3>
		inline Float compute_energy_impl(
			const Float3&	x0,
			const Float3&	x1,
			const Float3&	x2,
			const Float3&	x3,
			const Float3x3& Dm_inv,
			const Float		mu,
			const Float		lambda,
			const Float		volume)
		{
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
		template <typename Float, typename Float3, typename Float3x3>
		inline void compute_gradient_hessian_impl(
			const Float3&	x0,
			const Float3&	x1,
			const Float3&	x2,
			const Float3&	x3,
			const Float3x3& Dm_inv,
			const Float		mu,
			const Float		lambda,
			const Float		volume,
			Float3			gradient[4],
			Float3x3		hessian[16])
		{
			// using namespace luisa::compute;

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
				gradient[alpha] = volume * makeFloat3(g[0], g[1], g[2]);
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
			Float coeff_C_raw = max_scalar(coeff_cof, Float(0.0f));
			// Frobenius norm of F: ||F||_F = sqrt(sum_{c,r} F[c][r]^2)
			Float F_frob2 = 0.0f;
			for (int c = 0; c < 3; c++)
				for (int r = 0; r < 3; r++)
					F_frob2 = F_frob2 + F[c][r] * F[c][r];
			Float F_frob = sqrt_scalar(F_frob2 + Float(1e-14f));
			// Safe upper bound: mu / (sqrt(2) * ||F||_F)
			Float coeff_C_limit = mu / (Float(1.4142135f) * F_frob);
			Float coeff_C = min_scalar(coeff_C_raw, coeff_C_limit);

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
							Float val = mu * BdotB * lcs::select(i == j, Float(1.0f), Float(0.0f));

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
	}; // namespace
	template <typename Vec3T, typename Mat3T, typename ScalarT>
	struct Input
	{
		Vec3T	x0;
		Vec3T	x1;
		Vec3T	x2;
		Vec3T	x3;
		Mat3T	dm_inv;
		ScalarT mu;
		ScalarT lambda;
		ScalarT volume;
	};

	template <typename Vec3T, typename Mat3T, typename ScalarT>
	[[nodiscard]] inline auto compute_energy(const Input<Vec3T, Mat3T, ScalarT>& in)
	{
		return compute_energy_impl(
			in.x0, in.x1, in.x2, in.x3, in.dm_inv, in.mu, in.lambda, in.volume);
	}

	template <typename Vec3T, typename Mat3T, typename ScalarT>
	[[nodiscard]] inline auto evaluate(const Input<Vec3T, Mat3T, ScalarT>& in)
	{
		std::decay_t<Vec3T> gradients[4];
		std::decay_t<Mat3T> hessians[16];
		compute_gradient_hessian_impl(
			in.x0, in.x1, in.x2, in.x3, in.dm_inv, in.mu, in.lambda, in.volume, gradients, hessians);

		using GradientOutT = std::decay_t<Vec3T>;
		using HessianOutT = std::decay_t<Mat3T>;
		EnergyEvalResult<4, 16, GradientOutT, HessianOutT> out{};
		for (int i = 0; i < 4; i++)
		{
			out.gradients[i] = gradients[i];
		}
		for (int i = 0; i < 16; i++)
		{
			out.hessians[i] = hessians[i];
		}
		return out;
	}

	template <typename Vec3T, typename Mat3T, typename ScalarT>
	[[nodiscard]] inline auto evaluate_host(const Input<Vec3T, Mat3T, ScalarT>& in)
	{
		std::decay_t<Vec3T> gradients[4];
		std::decay_t<Mat3T> hessians[16];
		compute_gradient_hessian_impl(
			in.x0, in.x1, in.x2, in.x3, in.dm_inv, in.mu, in.lambda, in.volume, gradients, hessians);

		using GradientOutT = std::decay_t<Vec3T>;
		using HessianOutT = std::decay_t<Mat3T>;
		EnergyEvalResult<4, 16, GradientOutT, HessianOutT> out{};
		for (int i = 0; i < 4; i++)
		{
			out.gradients[i] = gradients[i];
		}
		for (int i = 0; i < 16; i++)
		{
			out.hessians[i] = hessians[i];
		}
		return out;
	}

} // namespace lcs::detail::stable_neo_hookean_energy
