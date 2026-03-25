#pragma once

#include "Core/float_nxn.h"
#include "Energies/detail/energy_detail_common.hpp"
#include "luisa/core/mathematics.h"
#include <type_traits>

namespace lcs::detail::prismatic_joint_constaint
{
	template <typename ScalarT, typename Vec3T, typename Mat3T>
	using PrismaticJointEvalResult = EnergyEvalResult<8, 64, Vec3T, Mat3T>;

	template <typename Vec3T>
	[[nodiscard]] inline Vec3T safe_normalize_axis(const Vec3T& axis)
	{
		// auto n2 = dot(axis, axis);
		// if (n2 < 1.0e-12f)
		// {
		// 	return Vec3T(1.0f, 0.0f, 0.0f);
		// }
		// return axis / luisa::sqrt(n2);
		return normalize_vec(axis);
	}

	template <typename ScalarT, typename Vec3T, typename Mat3T>
	[[nodiscard]] inline auto evaluate(
		const Vec3T (&q)[8],
		const Vec3T&  anchor_a_local,
		const Vec3T&  anchor_b_local,
		const Vec3T&  rest_position_delta,
		const Vec3T&  axis_world,
		const ScalarT stiffness_pos,
		const ScalarT stiffness_rot,
		const Mat3T&  identity)
	{
		PrismaticJointEvalResult<ScalarT, Vec3T, Mat3T> out{};
		for (auto& g : out.gradients)
		{
			g = zero3;
		}
		for (auto& H : out.hessians)
		{
			H = zero3x3;
		}

		auto add_linear_term = [&](const Mat3T(&coeff)[8], const Vec3T& bias, const ScalarT stiffness)
		{
			Vec3T r = bias;
			for (int i = 0; i < 8; ++i)
			{
				r += coeff[i] * q[i];
			}

			for (int i = 0; i < 8; ++i)
			{
				out.gradients[i] += stiffness * (coeff[i] * r);
				for (int j = 0; j < 8; ++j)
				{
					out.hessians[i * 8 + j] = out.hessians[i * 8 + j] + stiffness * (coeff[i] * coeff[j]);
				}
			}
		};

		const Mat3T I = identity;
		const Mat3T Z = 0.0f * identity;
		const Vec3T n = safe_normalize_axis(axis_world);
		const Mat3T P = I - outer_product(n, n);

		// Translational lock on plane orthogonal to the slide axis.
		{
			Mat3T coeff[8] = { (-1.0f) * P,
				-anchor_a_local.x * P,
				-anchor_a_local.y * P,
				-anchor_a_local.z * P,
				P,
				anchor_b_local.x * P,
				anchor_b_local.y * P,
				anchor_b_local.z * P };
			add_linear_term(coeff, (-1.0f) * (P * rest_position_delta), stiffness_pos);
		}

		// Keep relative orientation fixed to model a standard prismatic joint.
		for (int row = 0; row < 3; ++row)
		{
			Mat3T coeff[8] = { Z, Z, Z, Z, Z, Z, Z, Z };
			coeff[1 + row] = I;
			coeff[5 + row] = (-1.0f) * I;
			add_linear_term(coeff, zero3, stiffness_rot);
		}

		return out;
	}

	template <typename ScalarT, typename Vec3T, typename Mat3T>
	[[nodiscard]] inline ScalarT compute_energy(
		const Vec3T (&q)[8],
		const Vec3T&  anchor_a_local,
		const Vec3T&  anchor_b_local,
		const Vec3T&  rest_position_delta,
		const Vec3T&  axis_world,
		const ScalarT stiffness_pos,
		const ScalarT stiffness_rot,
		const Mat3T&  identity)
	{
		const Vec3T n = safe_normalize_axis(axis_world);
		const Mat3T P = identity - outer_product(n, n);

		Vec3T p_a = q[0] + q[1] * anchor_a_local.x + q[2] * anchor_a_local.y + q[3] * anchor_a_local.z;
		Vec3T p_b = q[4] + q[5] * anchor_b_local.x + q[6] * anchor_b_local.y + q[7] * anchor_b_local.z;
		Vec3T r_pos = P * ((p_b - p_a) - rest_position_delta);

		ScalarT energy = 0.5f * stiffness_pos * dot(r_pos, r_pos);
		for (int row = 0; row < 3; ++row)
		{
			Vec3T r_rot = q[1 + row] - q[5 + row];
			energy += 0.5f * stiffness_rot * dot(r_rot, r_rot);
		}
		return energy;
	}

} // namespace lcs::detail::prismatic_joint_constaint
