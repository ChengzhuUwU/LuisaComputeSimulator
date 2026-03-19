#pragma once

#include "Energies/detail/energy_detail_common.hpp"
#include "SimulationCore/base_mesh.h"
#include <type_traits>
#include <algorithm>

namespace lcs::detail::stretch_spring_energy
{
	[[nodiscard]] inline float scalar_max(float a, float b) noexcept
	{
		return std::max(a, b);
	}

	[[nodiscard]] inline luisa::compute::Float scalar_max(
		const luisa::compute::Float& a,
		const luisa::compute::Float& b) noexcept
	{
		return luisa::compute::max(a, b);
	}

	template <typename ScalarT, typename Vec3T>
	struct Input
	{
		Vec3T	x0;
		Vec3T	x1;
		ScalarT rest_length;
		ScalarT stiffness;
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
		const ScalarT eps = 1e-8f;
		const Vec3T	  diff = in.x0 - in.x1;
		const ScalarT l = scalar_max(length(diff), eps);
		const ScalarT stretch_constraint = l - in.rest_length;
		const Vec3T	  direction = diff / l;
		const ScalarT tangent_weight = scalar_max(1.0f - in.rest_length / l, 0.0f);

		const Vec3T g0 = in.stiffness * direction * stretch_constraint;
		const Vec3T g1 = -g0;

		const Mat3T nn_t = outer_product(direction, direction);
		const Mat3T he = in.stiffness * nn_t
			+ in.stiffness * tangent_weight * (identity - nn_t);

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
