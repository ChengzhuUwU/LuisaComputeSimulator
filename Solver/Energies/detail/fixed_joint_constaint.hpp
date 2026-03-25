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
		const Vec3T&  rest_position_delta,
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

		// Anchor coincidence: (pB + B * rb) - (pA + A * ra) = 0
		{
			Mat3T coeff[8] = { (-1.0f) * I,
				-anchor_a_local.x * I,
				-anchor_a_local.y * I,
				-anchor_a_local.z * I,
				I,
				anchor_b_local.x * I,
				anchor_b_local.y * I,
				anchor_b_local.z * I };
			add_linear_term(coeff, (-1.0f) * rest_position_delta, stiffness_pos);
		}

		// Orientation lock: A_i - B_i = 0 (i = 0..2)
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
		const ScalarT stiffness_pos,
		const ScalarT stiffness_rot,
		const Mat3T&  identity)
	{
		ScalarT energy = 0.0f;

		Vec3T p_a = q[0] + q[1] * anchor_a_local.x + q[2] * anchor_a_local.y + q[3] * anchor_a_local.z;
		Vec3T p_b = q[4] + q[5] * anchor_b_local.x + q[6] * anchor_b_local.y + q[7] * anchor_b_local.z;
		Vec3T r_pos = (p_b - p_a) - rest_position_delta;
		energy += 0.5f * stiffness_pos * dot(r_pos, r_pos);

		for (int row = 0; row < 3; ++row)
		{
			Vec3T r_rot = q[1 + row] - q[5 + row];
			energy += 0.5f * stiffness_rot * dot(r_rot, r_rot);
		}

		return energy;
	}

} // namespace lcs::detail::fixed_joint_constaint
