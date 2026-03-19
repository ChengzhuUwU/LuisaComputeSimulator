#pragma once

#include "CollisionDetector/cipc_kernel.hpp"
#include "CollisionDetector/friction_kernel.hpp"

namespace lcs::detail::ground_collision_energy
{
	template <typename ScalarT>
	[[nodiscard]] inline ScalarT repulsive_energy(
		ScalarT curr_dist,
		ScalarT thickness,
		ScalarT d_hat,
		ScalarT stiff,
		uint	collision_type)
	{
		if (collision_type == 0u)
		{
			const ScalarT c = curr_dist - d_hat - thickness;
			return 0.5f * stiff * c * c;
		}
		return stiff * ipc::barrier(curr_dist - thickness, d_hat);
	}

	template <typename ScalarT>
	[[nodiscard]] inline ScalarT repulsive_first_derivative(
		ScalarT dist,
		ScalarT thickness,
		ScalarT d_hat,
		ScalarT stiff,
		uint	collision_type)
	{
		if (collision_type == 0u)
		{
			return stiff * (dist - thickness - d_hat);
		}
		return stiff * ipc::barrier_first_derivative(dist - thickness, d_hat);
	}

	template <typename ScalarT>
	[[nodiscard]] inline ScalarT repulsive_second_derivative(
		ScalarT dist,
		ScalarT thickness,
		ScalarT d_hat,
		ScalarT stiff,
		uint	collision_type)
	{
		if (collision_type == 0u)
		{
			return stiff;
		}
		return stiff * ipc::barrier_second_derivative(dist - thickness, d_hat);
	}

	template <typename ScalarT, typename Vec3T>
	[[nodiscard]] inline ScalarT friction_energy(
		ScalarT		 lambda_mu,
		const Vec3T& normal,
		const Vec3T& rel_dx,
		ScalarT		 friction_eps)
	{
		return Friction::ipc_barrier::compute_friction_energy(lambda_mu, normal, rel_dx, friction_eps);
	}

	template <typename ScalarT, typename Vec3T, typename Mat3T>
	[[nodiscard]] inline auto friction_gradient_hessian(
		ScalarT		 lambda_mu,
		const Vec3T& normal,
		const Vec3T& rel_dx,
		ScalarT		 friction_eps)
	{
		return Friction::ipc_barrier::compute_friction_gradient_hessian(lambda_mu, normal, rel_dx, friction_eps);
	}

} // namespace lcs::detail::ground_collision_energy
