#pragma once

#include "Core/float_n.h"
#include "Core/float_nxn.h"
#include "Energies/detail/energy_detail_common.hpp"
#include <type_traits>

namespace lcs::detail::fixed_joint_constaint
{
	template <typename ScalarT, typename Vec3T, typename Mat3T>
	using FixedJointEvalResult = EnergyEvalResult<8, 64, Vec3T, Mat3T>;

	template <typename ScalarT, typename Vec3T, typename Mat3T>
	[[nodiscard]] inline auto evaluate(
		const Vec3T (&q)[8],
		const Vec3T&  anchor_a_local,
		const Vec3T&  anchor_b_local,
		const Vec3T&  rest_position_delta_local_a,
		const Vec3T&  rest_rot_col0_a_to_b,
		const Vec3T&  rest_rot_col1_a_to_b,
		const Vec3T&  rest_rot_col2_a_to_b,
		const ScalarT stiffness_pos,
		const ScalarT stiffness_rot,
		const Mat3T&  identity)
	{
		FixedJointEvalResult<ScalarT, Vec3T, Mat3T> out{};
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
		const Mat3T Z = zero3x3;

		// Anchor delta preservation: (p_B - p_A) - A*d0_local = 0.
		// This preserves the initial relative anchor offset expressed in body A's local frame.
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

		// Orientation lock in body-local rest frame: B - A * R_ab0 = 0.
		const Vec3T rest_cols[3] = { rest_rot_col0_a_to_b, rest_rot_col1_a_to_b, rest_rot_col2_a_to_b };
		for (int col = 0; col < 3; ++col)
		{
			Mat3T coeff[8] = { Z, Z, Z, Z, Z, Z, Z, Z };
			coeff[1] = (-rest_cols[col].x) * I;
			coeff[2] = (-rest_cols[col].y) * I;
			coeff[3] = (-rest_cols[col].z) * I;
			coeff[5 + col] = I;
			add_linear_term(coeff, zero3, stiffness_rot);
		}

		return out;
	}

	template <typename ScalarT, typename Vec3T, typename Mat3T>
	[[nodiscard]] inline ScalarT compute_energy(
		const Vec3T (&q)[8],
		const Vec3T&  anchor_a_local,
		const Vec3T&  anchor_b_local,
		const Vec3T&  rest_position_delta_local_a,
		const Vec3T&  rest_rot_col0_a_to_b,
		const Vec3T&  rest_rot_col1_a_to_b,
		const Vec3T&  rest_rot_col2_a_to_b,
		const ScalarT stiffness_pos,
		const ScalarT stiffness_rot,
		const Mat3T&  identity)
	{
		ScalarT energy = 0.0f;

		Vec3T p_a = q[0] + q[1] * anchor_a_local.x + q[2] * anchor_a_local.y + q[3] * anchor_a_local.z;
		Vec3T p_b = q[4] + q[5] * anchor_b_local.x + q[6] * anchor_b_local.y + q[7] * anchor_b_local.z;
		Vec3T target_delta = q[1] * rest_position_delta_local_a.x + q[2] * rest_position_delta_local_a.y + q[3] * rest_position_delta_local_a.z;
		Vec3T r_pos = (p_b - p_a) - target_delta;
		energy += 0.5f * stiffness_pos * dot(r_pos, r_pos);

		const Vec3T rest_cols[3] = { rest_rot_col0_a_to_b, rest_rot_col1_a_to_b, rest_rot_col2_a_to_b };
		for (int col = 0; col < 3; ++col)
		{
			Vec3T target_col = q[1] * rest_cols[col].x + q[2] * rest_cols[col].y + q[3] * rest_cols[col].z;
			Vec3T r_rot = q[5 + col] - target_col;
			energy += 0.5f * stiffness_rot * dot(r_rot, r_rot);
		}

		return energy;
	}

} // namespace lcs::detail::fixed_joint_constaint
