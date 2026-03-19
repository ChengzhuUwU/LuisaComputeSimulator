#pragma once

#include "Energies/detail/energy_detail_common.hpp"
#include "SimulationCore/base_mesh.h"
#include <type_traits>

namespace lcs::detail::stretch_spring_energy
{
	template <typename ScalarT, typename Vec3T>
	struct Input
	{
		Vec3T	direction;
		ScalarT stretch_constraint;
		ScalarT stiffness;
		ScalarT tangent_weight;
	};

	template <typename ScalarT>
	[[nodiscard]] inline ScalarT compute_energy(ScalarT stiffness, ScalarT stretch_constraint)
	{
		return 0.5f * stiffness * stretch_constraint * stretch_constraint;
	}

	template <typename ScalarT, typename Vec3T, typename Mat3T>
	[[nodiscard]] inline auto evaluate(
		const Input<ScalarT, Vec3T>& in,
		const Mat3T&				 identity)
	{
		const auto g0 = in.stiffness * in.direction * in.stretch_constraint;
		const auto g1 = -g0;

		const auto nn_t = outer_product(in.direction, in.direction);
		const auto he = in.stiffness * nn_t
			+ in.stiffness * in.tangent_weight * (identity - nn_t);

		using GradientOutT = std::decay_t<decltype(g0)>;
		using HessianOutT = std::decay_t<decltype(he)>;
		EdgeEvalResult<GradientOutT, HessianOutT> out{};

		out.gradients[0] = g0;
		out.gradients[1] = g1;

		out.hessians[0] = he;
		out.hessians[1] = -1.0f * he;
		out.hessians[2] = -1.0f * he;
		out.hessians[3] = he;

		return out;
	}

} // namespace lcs::detail::stretch_spring_energy
