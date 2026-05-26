#pragma once

#include "Core/float_nxn.h"
#include "Energies/detail/energy_detail_common.hpp"
#include "luisa/core/mathematics.h"
#include <type_traits>

namespace lcs::detail::revolute_joint_constaint
{
	template <typename ScalarT, typename Vec3T, typename Mat3T>
	using RevoluteJointEvalResult = EnergyEvalResult<8, 64, Vec3T, Mat3T>;

	template <typename ScalarT, typename Vec3T, typename Mat3T>
	[[nodiscard]] inline auto evaluate(
		const Vec3T (&q)[8],
		const Vec3T&  anchor_a_local,
		const Vec3T&  anchor_b_local,
		const Vec3T&  rest_position_delta_local_a,
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

		// Anchor delta preservation: (p_B - p_A) - A*d0_local = 0.
		// This preserves the initial relative anchor offset expressed in body A's local frame.
		// If both anchors were initialized at the same world position, d0_local = 0 and
		// this reduces to strict anchor coincidence.
		{
			Mat3T coeff[8] = { (-1.0f) * I,
				-(anchor_a_local.x + rest_position_delta_local_a.x) * I,
				-(anchor_a_local.y + rest_position_delta_local_a.y) * I,
				-(anchor_a_local.z + rest_position_delta_local_a.z) * I,
				I,
				anchor_b_local.x * I,
				anchor_b_local.y * I,
				anchor_b_local.z * I };
			add_linear_term(coeff, zero3, stiffness_pos);
		}

		// Hinge axis consistency in body-local form:
		// A * axis_a_local - B * axis_b_local = 0.
		{
			Mat3T coeff[8] = { Z,
				axis_a_local.x * I,
				axis_a_local.y * I,
				axis_a_local.z * I,
				Z,
				-axis_b_local.x * I,
				-axis_b_local.y * I,
				-axis_b_local.z * I };
			add_linear_term(coeff, zero3, stiffness_axis);
		}

		return out;
	}

	template <typename ScalarT, typename Vec3T, typename Mat3T>
	[[nodiscard]] inline ScalarT compute_energy(
		const Vec3T (&q)[8],
		const Vec3T&  anchor_a_local,
		const Vec3T&  anchor_b_local,
		const Vec3T&  rest_position_delta_local_a,
		const Vec3T&  axis_a_local,
		const Vec3T&  axis_b_local,
		const ScalarT stiffness_pos,
		const ScalarT stiffness_axis,
		const Mat3T&  identity)
	{
		Vec3T p_a = q[0] + q[1] * anchor_a_local.x + q[2] * anchor_a_local.y + q[3] * anchor_a_local.z;
		Vec3T p_b = q[4] + q[5] * anchor_b_local.x + q[6] * anchor_b_local.y + q[7] * anchor_b_local.z;
		Vec3T target_delta = q[1] * rest_position_delta_local_a.x + q[2] * rest_position_delta_local_a.y + q[3] * rest_position_delta_local_a.z;
		Vec3T r_pos = (p_b - p_a) - target_delta;

		Vec3T axis_a_world = q[1] * axis_a_local.x + q[2] * axis_a_local.y + q[3] * axis_a_local.z;
		Vec3T axis_b_world = q[5] * axis_b_local.x + q[6] * axis_b_local.y + q[7] * axis_b_local.z;
		Vec3T r_axis = axis_a_world - axis_b_world;

		ScalarT energy = 0.5f * stiffness_pos * dot(r_pos, r_pos);
		energy += 0.5f * stiffness_axis * dot(r_axis, r_axis);
		return energy;
	}

} // namespace lcs::detail::revolute_joint_constaint
