#pragma once

#include "Core/float_nxn.h"
#include "Energies/detail/energy_detail_common.hpp"
#include "luisa/core/mathematics.h"
#include <type_traits>

namespace lcs::detail::revolute_joint_constaint
{
	template <typename ScalarT, typename Vec3T, typename Mat3T>
	using RevoluteJointEvalResult = EnergyEvalResult<8, 64, Vec3T, Mat3T>;

	template <typename Vec3T>
	[[nodiscard]] inline Vec3T safe_normalize_axis(const Vec3T& axis)
	{
		// auto n2 = dot(axis, axis);
		// if (n2 < 1.0e-12f)
		// {
		// 	return Vec3T(1.0f, 0.0f, 0.0f);
		// }
		// return axis / sqrt_scalar(n2);
		return normalize_vec(axis);
	}

	template <typename ScalarT, typename Vec3T, typename Mat3T>
	[[nodiscard]] inline auto evaluate(
		const Vec3T (&q)[8],
		const Vec3T&  anchor_a_local,
		const Vec3T&  anchor_b_local,
		const Vec3T&  axis_world,
		const Vec3T&  axis_a_local,
		const Vec3T&  axis_b_local,
		const ScalarT stiffness_pos,
		const ScalarT stiffness_axis,
		const Mat3T&  identity)
	{
		RevoluteJointEvalResult<ScalarT, Vec3T, Mat3T> out{};
		for (auto& g : out.gradients)
		{
			g = zero3;
		}
		for (auto& H : out.hessians)
		{
			H = zero3x3;
		}

		auto add_linear_term = [&](const Mat3T(&coeff)[8], const ScalarT stiffness)
		{
			Vec3T r = zero3;
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

		// Anchor coincidence.
		{
			Mat3T coeff[8] = { (-1.0f) * I,
				-anchor_a_local.x * I,
				-anchor_a_local.y * I,
				-anchor_a_local.z * I,
				I,
				anchor_b_local.x * I,
				anchor_b_local.y * I,
				anchor_b_local.z * I };
			add_linear_term(coeff, stiffness_pos);
		}

		// Body A hinge axis must align with world hinge axis.
		{
			Mat3T coeff[8] = { Z,
				axis_a_local.x * P,
				axis_a_local.y * P,
				axis_a_local.z * P,
				Z,
				Z,
				Z,
				Z };
			add_linear_term(coeff, stiffness_axis);
		}

		// Body B hinge axis must align with world hinge axis.
		{
			Mat3T coeff[8] = { Z,
				Z,
				Z,
				Z,
				Z,
				axis_b_local.x * P,
				axis_b_local.y * P,
				axis_b_local.z * P };
			add_linear_term(coeff, stiffness_axis);
		}

		return out;
	}

	template <typename ScalarT, typename Vec3T, typename Mat3T>
	[[nodiscard]] inline ScalarT compute_energy(
		const Vec3T (&q)[8],
		const Vec3T&  anchor_a_local,
		const Vec3T&  anchor_b_local,
		const Vec3T&  axis_world,
		const Vec3T&  axis_a_local,
		const Vec3T&  axis_b_local,
		const ScalarT stiffness_pos,
		const ScalarT stiffness_axis,
		const Mat3T&  identity)
	{
		const Vec3T n = safe_normalize_axis(axis_world);
		const Mat3T P = identity - outer_product(n, n);

		Vec3T p_a = q[0] + q[1] * anchor_a_local.x + q[2] * anchor_a_local.y + q[3] * anchor_a_local.z;
		Vec3T p_b = q[4] + q[5] * anchor_b_local.x + q[6] * anchor_b_local.y + q[7] * anchor_b_local.z;
		Vec3T r_pos = p_b - p_a;

		Vec3T axis_a_world = q[1] * axis_a_local.x + q[2] * axis_a_local.y + q[3] * axis_a_local.z;
		Vec3T axis_b_world = q[5] * axis_b_local.x + q[6] * axis_b_local.y + q[7] * axis_b_local.z;
		Vec3T r_axis_a = P * axis_a_world;
		Vec3T r_axis_b = P * axis_b_world;

		ScalarT energy = 0.5f * stiffness_pos * dot(r_pos, r_pos);
		energy += 0.5f * stiffness_axis * dot(r_axis_a, r_axis_a);
		energy += 0.5f * stiffness_axis * dot(r_axis_b, r_axis_b);
		return energy;
	}

} // namespace lcs::detail::revolute_joint_constaint
