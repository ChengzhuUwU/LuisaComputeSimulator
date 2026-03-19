#pragma once

#include "Energies/detail/energy_detail_common.hpp"

namespace lcs::detail::bending_energy
{
	template <typename ScalarT>
	[[nodiscard]] inline ScalarT compute_energy(
		const ScalarT delta_angle,
		const ScalarT stiffness)
	{
		return 0.5f * stiffness * delta_angle * delta_angle;
	}

	template <typename ScalarT, typename Vec3T, typename Mat3T>
	[[nodiscard]] inline auto evaluate(
		const Vec3T (&dtheta_dx)[4],
		const ScalarT delta_angle,
		const ScalarT stiffness)
	{
		const auto g0 = stiffness * delta_angle * dtheta_dx[0];
		const auto h00 = stiffness * outer_product(dtheta_dx[0], dtheta_dx[0]);

		using GradientOutT = std::decay_t<decltype(g0)>;
		using HessianOutT = std::decay_t<decltype(h00)>;
		EnergyEvalResult<4, 16, GradientOutT, HessianOutT> out{};

		for (int ii = 0; ii < 4; ii++)
		{
			out.gradients[ii] = stiffness * delta_angle * dtheta_dx[ii];
		}
		for (int ii = 0; ii < 4; ii++)
		{
			for (int jj = 0; jj < 4; jj++)
			{
				out.hessians[ii * 4 + jj] = stiffness * outer_product(dtheta_dx[ii], dtheta_dx[jj]);
			}
		}
		return out;
	}

} // namespace lcs::detail::bending_energy
