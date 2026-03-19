#pragma once

#include "Energies/detail/energy_detail_common.hpp"
#include "SimulationCore/base_mesh.h"
#include <type_traits>

namespace lcs::detail::soft_inertia_energy
{
	template <typename ScalarT, typename Vec3T>
	struct Input
	{
		Vec3T	x_new;
		Vec3T	x_tilde;
		ScalarT mass;
		ScalarT inv_h2;
		ScalarT stiffness_dirichlet;
	};

	template <typename ScalarT, typename Vec3T>
	[[nodiscard]] inline ScalarT compute_energy(const Input<ScalarT, Vec3T>& in)
	{
		const Vec3T diff = in.x_new - in.x_tilde;
		return 0.5f * in.stiffness_dirichlet * in.mass * in.inv_h2 * length_squared_vec(diff);
	}

	template <typename ScalarT, typename Vec3T, typename Mat3T>
	[[nodiscard]] inline auto evaluate(
		const Input<ScalarT, Vec3T>& in,
		const Mat3T&				 identity)
	{
		const ScalarT scaled = in.stiffness_dirichlet * in.mass * in.inv_h2;
		const Vec3T	  gradient = scaled * (in.x_new - in.x_tilde);
		const Mat3T	  hessian = scaled * identity;

		using GradientOutT = std::decay_t<decltype(gradient)>;
		using HessianOutT = std::decay_t<decltype(hessian)>;
		SingleVertexEvalResult<GradientOutT, HessianOutT> out{};

		out.gradients[0] = gradient;
		out.hessians[0] = hessian;
		return out;
	}

} // namespace lcs::detail::soft_inertia_energy
