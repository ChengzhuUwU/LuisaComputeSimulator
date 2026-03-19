#pragma once

#include "Energies/detail/energy_detail_common.hpp"
#include <sys/types.h>
#include <type_traits>

namespace lcs::detail::abd_ortho_energy
{
	template <typename ScalarT, typename Mat3T>
	[[nodiscard]] inline ScalarT compute_energy(
		const Mat3T&  A,
		const ScalarT stiffness)
	{
		ScalarT energy = 0.0f;
		for (int ii = 0; ii < 3; ii++)
		{
			for (int jj = 0; jj < 3; jj++)
			{
				const ScalarT term = dot(A[ii], A[jj]) - (ii == jj ? 1.0f : 0.0f);
				energy += term * term;
			}
		}
		return stiffness * energy;
	}

	template <typename ScalarT, typename Vec3T, typename Mat3T>
	[[nodiscard]] inline auto evaluate(
		const Mat3T&  A,
		const ScalarT stiffness,
		const Mat3T&  identity)
	{
		const auto factor = 4.0f * stiffness;
		const auto g0 = factor * ((-1.0f) * A[0] + dot(A[0], A[0]) * A[0] + dot(A[0], A[1]) * A[1] + dot(A[0], A[2]) * A[2]);
		const auto g1 = factor * ((-1.0f) * A[1] + dot(A[1], A[0]) * A[0] + dot(A[1], A[1]) * A[1] + dot(A[1], A[2]) * A[2]);
		const auto g2 = factor * ((-1.0f) * A[2] + dot(A[2], A[0]) * A[0] + dot(A[2], A[1]) * A[1] + dot(A[2], A[2]) * A[2]);
		const auto h00 = factor * (outer_product(A[0], A[0]) + (dot(A[0], A[0]) - 1.0f) * identity + outer_product(A[0], A[0]) + outer_product(A[1], A[1]) + outer_product(A[2], A[2]));

		using GradientOutT = std::decay_t<decltype(g0)>;
		using HessianOutT = std::decay_t<decltype(h00)>;
		EnergyEvalResult<3, 9, GradientOutT, HessianOutT> out{};

		out.gradients[0] = g0;
		out.gradients[1] = g1;
		out.gradients[2] = g2;

		for (int ii = 0; ii < 3; ii++)
		{
			for (int jj = 0; jj < 3; jj++)
			{
				auto h = outer_product(A[jj], A[ii]) + dot(A[ii], A[jj]) * identity;
				if (ii == jj)
				{
					h = outer_product(A[ii], A[ii]) + (dot(A[ii], A[ii]) - 1.0f) * identity;
					for (int kk = 0; kk < 3; kk++)
					{
						h = h + outer_product(A[kk], A[kk]);
					}
				}
				out.hessians[ii * 3 + jj] = factor * h;
			}
		}
		return out;
	}

} // namespace lcs::detail::abd_ortho_energy
