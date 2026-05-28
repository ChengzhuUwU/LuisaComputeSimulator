#pragma once

#include "Core/float_nxn.h"
#include "Core/scalar.h"
#include "Energies/detail/energy_detail_common.hpp"
#include "luisa/core/mathematics.h"
#include <type_traits>

namespace lcs::detail::prismatic_joint_constaint
{
	template <typename ScalarT, typename Vec3T, typename Mat3T>
	using PrismaticJointEvalResult = EnergyEvalResult<8, 64, Vec3T, Mat3T>;

	template <typename ScalarT, typename Vec3T, typename Mat3T>
	[[nodiscard]] inline auto evaluate(
		const Vec3T (&q)[8],
		const Vec3T&  anchor_a_local,
		const Vec3T&  anchor_b_local,
		const Vec3T&  rest_position_delta_local_a,
		const Vec3T&  rest_rot_col0_a_to_b,
		const Vec3T&  rest_rot_col1_a_to_b,
		const Vec3T&  rest_rot_col2_a_to_b,
		const Vec3T&  axis_a_local,
		const ScalarT stiffness_pos,
		const ScalarT stiffness_rot,
		const ScalarT slide_min,
		const ScalarT slide_max,
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

		// Translational constraint: cross-product formulation with body-local axis.
		// Anchor delta preservation: d = (p_B - p_A) - A*d0_local  (linear in q)
		// a = A * axis_a_local             (linear in q)
		// r = d x a                        (zero when d is parallel to a)
		// E_pos = k/2 * ||r||^2
		//
		// Since all coefficient matrices D_i = d_i * I and Ac_i = a_i * I,
		// the Jacobian simplifies: J_i = skew(w_i) where w_i = a_i * d - d_i * a.
		// Gradient:  g_i = k * cross(r, w_i)
		// Hessian:   H_ij = k * [dot(w_i,w_j)*I - outer(w_j,w_i) - skew(cross(w_i,w_j))]
		{
			Vec3T p_a = q[0] + q[1] * anchor_a_local.x + q[2] * anchor_a_local.y + q[3] * anchor_a_local.z;
			Vec3T p_b = q[4] + q[5] * anchor_b_local.x + q[6] * anchor_b_local.y + q[7] * anchor_b_local.z;
			Vec3T target_delta = q[1] * rest_position_delta_local_a.x
				+ q[2] * rest_position_delta_local_a.y
				+ q[3] * rest_position_delta_local_a.z;
			Vec3T d = (p_b - p_a) - target_delta;
			Vec3T a = q[1] * axis_a_local.x + q[2] * axis_a_local.y + q[3] * axis_a_local.z;

			Vec3T r = cross_vec(d, a);

			// w_i = a_i * d - d_i * a  for each DOF block
			Vec3T w[8];
			w[0] = a;
			w[1] = axis_a_local.x * d + (anchor_a_local.x + rest_position_delta_local_a.x) * a;
			w[2] = axis_a_local.y * d + (anchor_a_local.y + rest_position_delta_local_a.y) * a;
			w[3] = axis_a_local.z * d + (anchor_a_local.z + rest_position_delta_local_a.z) * a;
			w[4] = (-1.0f) * a;
			w[5] = (-anchor_b_local.x) * a;
			w[6] = (-anchor_b_local.y) * a;
			w[7] = (-anchor_b_local.z) * a;

			for (int i = 0; i < 8; ++i)
			{
				out.gradients[i] += stiffness_pos * cross_vec(r, w[i]);
				for (int j = 0; j < 8; ++j)
				{
					Vec3T cw = cross_vec(w[i], w[j]);
					out.hessians[i * 8 + j] = out.hessians[i * 8 + j]
						+ stiffness_pos * (dot(w[i], w[j]) * I - outer_product(w[j], w[i]) - skew(cw));
				}
			}

			// Slide limit: s = dot(d, a). Penalise when s < slide_min or s > slide_max.
			// v_i = ∂s/∂q_i = alpha_i * d + delta_i * a  (exact gradient of the scalar s).
			// Full Hessian: outer(v_i, v_j) + residual*(alpha_i*delta_j + alpha_j*delta_i)*I.
			{
				const ScalarT s = dot(d, a);
				ScalarT		  residual = 0.0f;
				// if (s < slide_min)
				// 	residual = s - slide_min;
				// else if (s > slide_max)
				// 	residual = s - slide_max;
				residual = max_scalar(s - slide_max, ScalarT(0.0f)) + min_scalar(s - slide_min, ScalarT(0.0f));

				// if (residual != 0.0f)
				{
					// alpha_i = coefficient of q_i in a; delta_i = coefficient of q_i in d.
					const ScalarT alpha[8] = { 0.0f,
						axis_a_local.x, axis_a_local.y, axis_a_local.z,
						0.0f, 0.0f, 0.0f, 0.0f };
					const ScalarT delta[8] = { -1.0f,
						-(anchor_a_local.x + rest_position_delta_local_a.x),
						-(anchor_a_local.y + rest_position_delta_local_a.y),
						-(anchor_a_local.z + rest_position_delta_local_a.z),
						1.0f, anchor_b_local.x, anchor_b_local.y, anchor_b_local.z };

					Vec3T v[8];
					for (int i = 0; i < 8; ++i)
						v[i] = alpha[i] * d + delta[i] * a;

					for (int i = 0; i < 8; ++i)
					{
						out.gradients[i] += stiffness_pos * residual * v[i];
						for (int j = 0; j < 8; ++j)
						{
							out.hessians[i * 8 + j] = out.hessians[i * 8 + j]
								+ stiffness_pos * (outer_product(v[i], v[j]) + residual * (alpha[i] * delta[j] + alpha[j] * delta[i]) * I);
						}
					}
				}
			}
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
		const Vec3T&  axis_a_local,
		const ScalarT stiffness_pos,
		const ScalarT stiffness_rot,
		const ScalarT slide_min,
		const ScalarT slide_max,
		const Mat3T&  identity)
	{
		Vec3T p_a = q[0] + q[1] * anchor_a_local.x + q[2] * anchor_a_local.y + q[3] * anchor_a_local.z;
		Vec3T p_b = q[4] + q[5] * anchor_b_local.x + q[6] * anchor_b_local.y + q[7] * anchor_b_local.z;
		Vec3T target_delta = q[1] * rest_position_delta_local_a.x + q[2] * rest_position_delta_local_a.y + q[3] * rest_position_delta_local_a.z;
		Vec3T d = (p_b - p_a) - target_delta;
		Vec3T a = q[1] * axis_a_local.x + q[2] * axis_a_local.y + q[3] * axis_a_local.z;
		Vec3T r_pos = cross_vec(d, a);

		ScalarT		energy = 0.5f * stiffness_pos * dot(r_pos, r_pos);
		const Vec3T rest_cols[3] = { rest_rot_col0_a_to_b, rest_rot_col1_a_to_b, rest_rot_col2_a_to_b };
		for (int col = 0; col < 3; ++col)
		{
			Vec3T target_col = q[1] * rest_cols[col].x + q[2] * rest_cols[col].y + q[3] * rest_cols[col].z;
			Vec3T r_rot = q[5 + col] - target_col;
			energy += 0.5f * stiffness_rot * dot(r_rot, r_rot);
		}
		// Slide limit energy.
		{
			const ScalarT s = dot(d, a);
			if (s < slide_min)
				energy += 0.5f * stiffness_pos * (s - slide_min) * (s - slide_min);
			else if (s > slide_max)
				energy += 0.5f * stiffness_pos * (s - slide_max) * (s - slide_max);
		}
		return energy;
	}

} // namespace lcs::detail::prismatic_joint_constaint
