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
		const Vec3T&  rest_position_delta_local_a,
		const Vec3T&  rest_rot_col0_a_to_b,
		const Vec3T&  rest_rot_col1_a_to_b,
		const Vec3T&  rest_rot_col2_a_to_b,
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
		// Position target is body-local rest relation: A * d0_local.
		{
			Mat3T coeff[8] = { (-1.0f) * P,
				-(anchor_a_local.x + rest_position_delta_local_a.x) * P,
				-(anchor_a_local.y + rest_position_delta_local_a.y) * P,
				-(anchor_a_local.z + rest_position_delta_local_a.z) * P,
				P,
				anchor_b_local.x * P,
				anchor_b_local.y * P,
				anchor_b_local.z * P };
			add_linear_term(coeff, zero3, stiffness_pos);
		}

		// Keep relative orientation fixed in body-local rest frame: B - A * R_ab0 = 0.
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
		const Vec3T&  axis_world,
		const ScalarT stiffness_pos,
		const ScalarT stiffness_rot,
		const Mat3T&  identity)
	{
		const Vec3T n = safe_normalize_axis(axis_world);
		const Mat3T P = identity - outer_product(n, n);

		Vec3T p_a = q[0] + q[1] * anchor_a_local.x + q[2] * anchor_a_local.y + q[3] * anchor_a_local.z;
		Vec3T p_b = q[4] + q[5] * anchor_b_local.x + q[6] * anchor_b_local.y + q[7] * anchor_b_local.z;
		Vec3T target_delta = q[1] * rest_position_delta_local_a.x + q[2] * rest_position_delta_local_a.y + q[3] * rest_position_delta_local_a.z;
		Vec3T r_pos = P * ((p_b - p_a) - target_delta);

		ScalarT energy = 0.5f * stiffness_pos * dot(r_pos, r_pos);
		const Vec3T rest_cols[3] = { rest_rot_col0_a_to_b, rest_rot_col1_a_to_b, rest_rot_col2_a_to_b };
		for (int col = 0; col < 3; ++col)
		{
			Vec3T target_col = q[1] * rest_cols[col].x + q[2] * rest_cols[col].y + q[3] * rest_cols[col].z;
			Vec3T r_rot = q[5 + col] - target_col;
			energy += 0.5f * stiffness_rot * dot(r_rot, r_rot);
		}
		return energy;
	}

} // namespace lcs::detail::prismatic_joint_constaint
