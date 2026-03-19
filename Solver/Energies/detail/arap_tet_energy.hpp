#pragma once

#include "Core/svd_3x3.h"
#include "Core/float_n.h"
#include "Energies/detail/energy_detail_common.hpp"
#include "SimulationCore/base_mesh.h"
#include <type_traits>

namespace lcs::detail::arap_tet_energy
{
	constexpr float sqrt2 = 1.4142135623730951f;

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

	[[nodiscard]] inline float3x3 make_twist_mode_0()
	{
		// Row-major in reference:
		// [ 0 -1  0 ]
		// [ 1  0  0 ]
		// [ 0  0  0 ]
		// Here float3x3 is column-major M[col][row].
		return luisa::make_float3x3(luisa::make_float3(0.0f, 1.0f, 0.0f),
			luisa::make_float3(-1.0f, 0.0f, 0.0f),
			luisa::make_float3(0.0f, 0.0f, 0.0f));
	}

	[[nodiscard]] inline float3x3 make_twist_mode_1()
	{
		// Row-major in reference:
		// [ 0  0  0 ]
		// [ 0  0  1 ]
		// [ 0 -1  0 ]
		return luisa::make_float3x3(luisa::make_float3(0.0f, 0.0f, 0.0f),
			luisa::make_float3(0.0f, 0.0f, -1.0f),
			luisa::make_float3(0.0f, 1.0f, 0.0f));
	}

	[[nodiscard]] inline float3x3 make_twist_mode_2()
	{
		// Row-major in reference:
		// [ 0  0  1 ]
		// [ 0  0  0 ]
		// [-1  0  0 ]
		return luisa::make_float3x3(luisa::make_float3(0.0f, 0.0f, -1.0f),
			luisa::make_float3(0.0f, 0.0f, 0.0f),
			luisa::make_float3(1.0f, 0.0f, 0.0f));
	}

	[[nodiscard]] inline float9 vec_col_major(const float3x3& m)
	{
		float9 v;
		for (int c = 0; c < 3; c++)
		{
			for (int r = 0; r < 3; r++)
			{
				v.scalar(r + 3 * c) = m[c][r];
			}
		}
		return v;
	}

	[[nodiscard]] inline int levi_civita(int a, int b, int c)
	{
		if (a == b || b == c || a == c)
		{
			return 0;
		}
		if ((a == 0 && b == 1 && c == 2)
			|| (a == 1 && b == 2 && c == 0)
			|| (a == 2 && b == 0 && c == 1))
		{
			return 1;
		}
		return -1;
	}

	[[nodiscard]] inline float9x9 det_hessian_F_space(const float3x3& F)
	{
		float9x9 H;
		H.set_zero();
		for (int c = 0; c < 3; ++c)
		{
			for (int r = 0; r < 3; ++r)
			{
				int p = r + 3 * c; // F_{r,c}
				for (int d = 0; d < 3; ++d)
				{
					for (int s = 0; s < 3; ++s)
					{
						int	  q = s + 3 * d; // F_{s,d}
						float val = 0.0f;
						for (int m = 0; m < 3; ++m)
						{
							for (int n = 0; n < 3; ++n)
							{
								val += static_cast<float>(levi_civita(r, s, m) * levi_civita(c, d, n)) * F[n][m];
							}
						}
						H.scalar(p, q) = val;
					}
				}
			}
		}
		return H;
	}

	[[nodiscard]] inline float3x3 polar_rotation(const float3x3& F)
	{
		float3x3 U, V;
		float3	 S;
		lcs::svd(F, U, S, V);
		return U * transpose(V);
	}

	[[nodiscard]] inline float3x3 deformation_gradient(const float3& x0,
		const float3&												 x1,
		const float3&												 x2,
		const float3&												 x3,
		const float3x3&												 dm_inv)
	{
		float3x3 Ds = luisa::make_float3x3(x1 - x0, x2 - x0, x3 - x0);
		return Ds * dm_inv;
	}

	[[nodiscard]] inline float compute_energy(const Input<float3, float3x3, float>& in)
	{
		float3x3 F = deformation_gradient(in.x0, in.x1, in.x2, in.x3, in.dm_inv);
		float3x3 R = polar_rotation(F);
		float	 J = determinant(F);
		// Match reference core: E_arap = kappa * v * ||F - R||^2.
		float psi_arap = 0.0f;
		for (int c = 0; c < 3; c++)
		{
			for (int r = 0; r < 3; r++)
			{
				float d = F[c][r] - R[c][r];
				psi_arap += d * d;
			}
		}
		float psi_vol = 0.5f * in.lambda * (J - 1.0f) * (J - 1.0f);
		return in.volume * (0.5f * in.mu * psi_arap + psi_vol);
	}

	[[nodiscard]] inline float9x9 arap_hessian_F_space(const float3x3& F)
	{
		float3x3 U, V;
		float3	 Sigma;
		lcs::svd(F, U, Sigma, V);

		float3x3 T0 = (1.0f / sqrt2) * U * make_twist_mode_0() * transpose(V);
		float3x3 T1 = (1.0f / sqrt2) * U * make_twist_mode_1() * transpose(V);
		float3x3 T2 = (1.0f / sqrt2) * U * make_twist_mode_2() * transpose(V);

		float9 t0 = vec_col_major(T0);
		float9 t1 = vec_col_major(T1);
		float9 t2 = vec_col_major(T2);

		float s0 = Sigma[0];
		float s1 = Sigma[1];
		float s2 = Sigma[2];

		float9x9 H;
		H.set_zero();
		for (int i = 0; i < 9; i++)
		{
			H.scalar(i, i) = 2.0f;
		}

		auto subtract_mode = [&](float9x9& mat, const float9& t, float coeff)
		{
			for (int i = 0; i < 9; i++)
			{
				for (int j = 0; j < 9; j++)
				{
					mat.scalar(i, j) -= coeff * t.scalar(i) * t.scalar(j);
				}
			}
		};

		const float eps = 1e-8f;
		auto		safe_coeff = [&](float denom)
		{
			// Reference form: 4 / (s_i + s_j), with small denominator guard.
			return 4.0f / std::max(denom, eps);
		};
		subtract_mode(H, t0, safe_coeff(s0 + s1));
		subtract_mode(H, t1, safe_coeff(s1 + s2));
		subtract_mode(H, t2, safe_coeff(s0 + s2));

		return H;
	}

	inline void compute_B(const float3x3& dm_inv, float B[4][3])
	{
		for (int k = 0; k < 3; k++)
		{
			B[1][k] = dm_inv[0][k];
			B[2][k] = dm_inv[1][k];
			B[3][k] = dm_inv[2][k];
			B[0][k] = -(B[1][k] + B[2][k] + B[3][k]);
		}
	}

	inline void convert_force(const float3x3& dEdF, const float B[4][3], float3 gradient[4])
	{
		for (int a = 0; a < 4; a++)
		{
			float3 g = luisa::make_float3(0.0f);
			for (int c = 0; c < 3; c++)
			{
				for (int r = 0; r < 3; r++)
				{
					g[r] += dEdF[c][r] * B[a][c];
				}
			}
			gradient[a] = g;
		}
	}

	inline void convert_hessian(const float9x9& H9, const float B[4][3], float3x3 hessian[16])
	{
		for (int a = 0; a < 4; a++)
		{
			for (int b = 0; b < 4; b++)
			{
				float3x3 K = luisa::make_float3x3(0.0f);
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						float val = 0.0f;
						for (int ca = 0; ca < 3; ca++)
						{
							for (int cb = 0; cb < 3; cb++)
							{
								val += H9.scalar(i + 3 * ca, j + 3 * cb) * B[a][ca] * B[b][cb];
							}
						}
						K[j][i] = val;
					}
				}
				hessian[a * 4 + b] = K;
			}
		}
	}

	[[nodiscard]] inline auto evaluate_host(const Input<float3, float3x3, float>& in)
	{
		float3x3 F = deformation_gradient(in.x0, in.x1, in.x2, in.x3, in.dm_inv);
		float3x3 R = polar_rotation(F);
		float	 J = determinant(F);

		float3x3 cofF;
		cofF[0][0] = F[1][1] * F[2][2] - F[2][1] * F[1][2];
		cofF[0][1] = -(F[1][0] * F[2][2] - F[2][0] * F[1][2]);
		cofF[0][2] = F[1][0] * F[2][1] - F[2][0] * F[1][1];
		cofF[1][0] = -(F[0][1] * F[2][2] - F[2][1] * F[0][2]);
		cofF[1][1] = F[0][0] * F[2][2] - F[2][0] * F[0][2];
		cofF[1][2] = -(F[0][0] * F[2][1] - F[2][0] * F[0][1]);
		cofF[2][0] = F[0][1] * F[1][2] - F[1][1] * F[0][2];
		cofF[2][1] = -(F[0][0] * F[1][2] - F[1][0] * F[0][2]);
		cofF[2][2] = F[0][0] * F[1][1] - F[1][0] * F[0][1];

		// kappa = 0.5 * mu to stay consistent with existing solver convention.
		float3x3 dEdF = in.mu * in.volume * (F - R)
			+ in.lambda * in.volume * (J - 1.0f) * cofF;

		// d2E/dF2 (ARAP part)
		float9x9 H9 = arap_hessian_F_space(F);
		for (int i = 0; i < 9; i++)
		{
			for (int j = 0; j < 9; j++)
			{
				H9.scalar(i, j) *= 0.5f * in.mu * in.volume;
			}
		}

		// d2E/dF2 (volumetric part): lambda * v * (vec(cofF)vec(cofF)^T + (J-1) * d2J/dF2)
		float9	 vec_cof = vec_col_major(cofF);
		float9x9 HJ = det_hessian_F_space(F);
		for (int i = 0; i < 9; ++i)
		{
			for (int j = 0; j < 9; ++j)
			{
				H9.scalar(i, j) += in.lambda * in.volume
					* (vec_cof.scalar(i) * vec_cof.scalar(j) + (J - 1.0f) * HJ.scalar(i, j));
			}
		}

		float B[4][3];
		compute_B(in.dm_inv, B);

		EnergyEvalResult<4, 16, float3, float3x3> out{};
		convert_force(dEdF, B, out.gradients.data());
		convert_hessian(H9, B, out.hessians.data());
		return out;
	}

	// Device path uses a compile-safe polar approximation, while host path above
	// follows the exact SVD-based ARAP formula from the Eigen reference.
	[[nodiscard]] inline luisa::compute::Var<float3x3> polar_rotation_approx(
		const luisa::compute::Var<float3x3>& F)
	{
		using namespace luisa::compute;
		Var<float3x3> R = F;
		for (int i = 0; i < 5; i++)
		{
			R = 0.5f * (R + inverse(transpose(R)));
		}
		return R;
	}

	[[nodiscard]] inline luisa::compute::Var<float> compute_energy(
		const Input<luisa::compute::Var<float3>, luisa::compute::Var<float3x3>, luisa::compute::Var<float>>& in)
	{
		using namespace luisa::compute;
		Var<float3x3> Ds = make_float3x3(in.x1 - in.x0, in.x2 - in.x0, in.x3 - in.x0);
		Var<float3x3> F = Ds * in.dm_inv;
		auto		  R = polar_rotation_approx(F);
		Var<float>	  J = determinant(F);

		Var<float> psi = 0.0f;
		for (int c = 0; c < 3; c++)
		{
			for (int r = 0; r < 3; r++)
			{
				auto d = F[c][r] - R[c][r];
				psi = psi + d * d;
			}
		}
		psi = 0.5f * in.mu * psi + 0.5f * in.lambda * (J - 1.0f) * (J - 1.0f);
		return in.volume * psi;
	}

	[[nodiscard]] inline auto evaluate(
		const Input<luisa::compute::Var<float3>, luisa::compute::Var<float3x3>, luisa::compute::Var<float>>& in)
	{
		using namespace luisa::compute;
		Var<float3x3> Ds = make_float3x3(in.x1 - in.x0, in.x2 - in.x0, in.x3 - in.x0);
		Var<float3x3> F = Ds * in.dm_inv;
		auto		  R = polar_rotation_approx(F);
		Var<float>	  J = determinant(F);

		Var<float3x3> cofF;
		cofF[0][0] = F[1][1] * F[2][2] - F[2][1] * F[1][2];
		cofF[0][1] = -(F[1][0] * F[2][2] - F[2][0] * F[1][2]);
		cofF[0][2] = F[1][0] * F[2][1] - F[2][0] * F[1][1];
		cofF[1][0] = -(F[0][1] * F[2][2] - F[2][1] * F[0][2]);
		cofF[1][1] = F[0][0] * F[2][2] - F[2][0] * F[0][2];
		cofF[1][2] = -(F[0][0] * F[2][1] - F[2][0] * F[0][1]);
		cofF[2][0] = F[0][1] * F[1][2] - F[1][1] * F[0][2];
		cofF[2][1] = -(F[0][0] * F[1][2] - F[1][0] * F[0][2]);
		cofF[2][2] = F[0][0] * F[1][1] - F[1][0] * F[0][1];

		auto P = in.mu * (F - R) + in.lambda * (J - 1.0f) * cofF;

		using GradientOutT = std::decay_t<decltype(in.x0)>;
		using HessianOutT = std::decay_t<decltype(in.dm_inv)>;
		EnergyEvalResult<4, 16, GradientOutT, HessianOutT> out{};

		Var<float> B[4][3];
		for (int k = 0; k < 3; k++)
		{
			B[1][k] = in.dm_inv[0][k];
			B[2][k] = in.dm_inv[1][k];
			B[3][k] = in.dm_inv[2][k];
			B[0][k] = -(B[1][k] + B[2][k] + B[3][k]);
		}

		for (int a = 0; a < 4; a++)
		{
			Var<float3> g = make_float3(0.0f);
			for (int c = 0; c < 3; c++)
			{
				for (int i = 0; i < 3; i++)
				{
					g[i] = g[i] + P[c][i] * B[a][c];
				}
			}
			out.gradients[a] = in.volume * g;
		}

		for (int a = 0; a < 4; a++)
		{
			for (int b = 0; b < 4; b++)
			{
				Var<float> bdot = 0.0f;
				for (int k = 0; k < 3; k++)
				{
					bdot = bdot + B[a][k] * B[b][k];
				}

				Var<float3> cof_a = make_float3(0.0f);
				Var<float3> cof_b = make_float3(0.0f);
				for (int i = 0; i < 3; i++)
				{
					for (int c = 0; c < 3; c++)
					{
						cof_a[i] = cof_a[i] + cofF[c][i] * B[a][c];
						cof_b[i] = cof_b[i] + cofF[c][i] * B[b][c];
					}
				}

				out.hessians[a * 4 + b] = in.volume
					* (in.mu * bdot * make_float3x3(1.0f)
						+ in.lambda * outer_product(cof_a, cof_b));
			}
		}

		return out;
	}

} // namespace lcs::detail::arap_tet_energy
