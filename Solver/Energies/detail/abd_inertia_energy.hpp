#pragma once

#include "Energies/detail/energy_detail_common.hpp"

namespace lcs::detail::abd_inertia_energy
{
	template <typename ScalarT, typename Vec3T, typename Mat4T>
	[[nodiscard]] inline ScalarT compute_energy(
		const Vec3T (&delta)[4],
		const Mat4T&  mass_matrix,
		const ScalarT scaled_stiffness)
	{
		ScalarT energy = 0.0f;
		for (int ii = 0; ii < 4; ii++)
		{
			for (int jj = 0; jj < 4; jj++)
			{
				energy += 0.5f * scaled_stiffness * mass_matrix[ii][jj] * dot(delta[ii], delta[jj]);
			}
		}
		return energy;
	}

	template <typename ScalarT, typename Vec3T, typename Mat3T, typename Mat4T>
	[[nodiscard]] inline auto evaluate(
		const Vec3T (&delta)[4],
		const Mat4T&  mass_matrix,
		const ScalarT scaled_stiffness,
		const Mat3T&  identity)
	{
		const Vec3T g0 = scaled_stiffness
			* (mass_matrix[0][0] * delta[0] + mass_matrix[0][1] * delta[1] + mass_matrix[0][2] * delta[2]
				+ mass_matrix[0][3] * delta[3]);
		const Vec3T g1 = scaled_stiffness
			* (mass_matrix[1][0] * delta[0] + mass_matrix[1][1] * delta[1] + mass_matrix[1][2] * delta[2]
				+ mass_matrix[1][3] * delta[3]);
		const Vec3T g2 = scaled_stiffness
			* (mass_matrix[2][0] * delta[0] + mass_matrix[2][1] * delta[1] + mass_matrix[2][2] * delta[2]
				+ mass_matrix[2][3] * delta[3]);
		const Vec3T g3 = scaled_stiffness
			* (mass_matrix[3][0] * delta[0] + mass_matrix[3][1] * delta[1] + mass_matrix[3][2] * delta[2]
				+ mass_matrix[3][3] * delta[3]);
		const Mat3T h00 = scaled_stiffness * mass_matrix[0][0] * identity;

		using GradientOutT = std::decay_t<decltype(g0)>;
		using HessianOutT = std::decay_t<decltype(h00)>;
		EnergyEvalResult<4, 16, GradientOutT, HessianOutT> out{};

		out.gradients[0] = g0;
		out.gradients[1] = g1;
		out.gradients[2] = g2;
		out.gradients[3] = g3;

		for (int ii = 0; ii < 4; ii++)
		{
			for (int jj = 0; jj < 4; jj++)
			{
				out.hessians[ii * 4 + jj] = scaled_stiffness * mass_matrix[ii][jj] * identity;
			}
		}
		return out;
	}

} // namespace lcs::detail::abd_inertia_energy
