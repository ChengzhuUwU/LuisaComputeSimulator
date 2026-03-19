#pragma once

#include "Core/float_n.h"
#include "Core/float_nxn.h"
#include "Core/lc_to_eigen.h"
#include "Core/svd_2x2.h"
#include "Core/svd_3x3.h"
#include <type_traits>
#include <luisa/luisa-compute.h>

namespace lcs
{

	namespace FemUtils
	{

		inline std::array<double, 2> convert_lame_params_3d(const float young_mod, const float poiss_rat)
		{
			double mu = young_mod / (2.0 * (1.0 + poiss_rat));
			double lambda = young_mod * poiss_rat / ((1.0 + poiss_rat) * (1.0 - 2.0 * poiss_rat));
			return { mu, lambda };
		}

		// 2D cloth uses a surface model; plane-stress lambda is typically more stable/physical than 3D lambda.
		inline std::array<double, 2> convert_lame_params_2d(const float young_mod, const float poiss_rat)
		{
			double mu = young_mod / (2.0 * (1.0 + poiss_rat));
			double lambda = young_mod * poiss_rat / (1.0 - poiss_rat * poiss_rat);
			return { mu, lambda };
		}

		// ========================================================================
		// IMPROVED IMPLEMENTATION (coordinate-system-independent)
		// ========================================================================

		// inline float2x2 get_Dm_inv(const float3& x_0, const float3& x_1, const float3& x_2)
		// {
		// 	float3		   r_1 = x_1 - x_0;
		// 	float3		   r_2 = x_2 - x_0;
		// 	float3		   cross = cross_vec(r_1, r_2);
		// 	float3		   axis_1 = normalize_vec(r_1);
		// 	float3		   axis_2 = normalize_vec(cross_vec(cross, axis_1));
		// 	float2		   uv0 = float2(dot_vec(axis_1, x_0), dot_vec(axis_2, x_0));
		// 	float2		   uv1 = float2(dot_vec(axis_1, x_1), dot_vec(axis_2, x_1));
		// 	float2		   uv2 = float2(dot_vec(axis_1, x_2), dot_vec(axis_2, x_2));
		// 	float2		   duv0 = uv1 - uv0;
		// 	float2		   duv1 = uv2 - uv0;
		// 	const float2x2 duv = float2x2(duv0, duv1);
		// 	const float2x2 inv_duv = luisa::inverse(duv);
		// 	return inv_duv;
		// }

		inline float2x2 get_Dm_inv(const float3& x_0, const float3& x_1, const float3& x_2)
		{
			// STRATEGY: Use the longest edge as the primary axis instead of arbitrarily
			// choosing first & second edges. This makes Dm_inv independent of vertex labeling.

			// Step 1: Compute all three edge vectors
			float3 e01 = x_1 - x_0;
			float3 e02 = x_2 - x_0;
			float3 e12 = x_2 - x_1;

			// Step 2: Find the longest edge (order-independent and physically meaningful)
			float len01 = luisa::length(e01);
			float len02 = luisa::length(e02);
			float len12 = luisa::length(e12);

			float3 longest_edge;
			if (len01 >= len02 && len01 >= len12)
			{
				longest_edge = e01;
			}
			else if (len02 >= len01 && len02 >= len12)
			{
				longest_edge = e02;
			}
			else
			{
				longest_edge = e12; // e12 is longest
			}

			// Step 3: Compute surface normal
			float3 normal = luisa::cross(e01, e02);
			float3 normal_normalized = luisa::normalize(normal);

			// Step 4: Build orthonormal coordinate frame
			// axis1: along the longest edge
			float3 axis_1 = luisa::normalize(longest_edge);
			// axis2: perpendicular to both normal and axis1
			float3 axis_2 = luisa::normalize(luisa::cross(normal_normalized, axis_1));

			// Step 5: Project all vertices onto the plane
			// Use triangle centroid as origin for symmetry
			float3 centroid = (x_0 + x_1 + x_2) / 3.0f;

			float2 uv0 = float2(
				luisa::dot(axis_1, x_0 - centroid),
				luisa::dot(axis_2, x_0 - centroid));
			float2 uv1 = float2(
				luisa::dot(axis_1, x_1 - centroid),
				luisa::dot(axis_2, x_1 - centroid));
			float2 uv2 = float2(
				luisa::dot(axis_1, x_2 - centroid),
				luisa::dot(axis_2, x_2 - centroid));

			// Step 6: Build Dm matrix from edge vectors in the new basis
			float2		   duv0 = uv1 - uv0; // Edge from vertex 0 to 1 in material space
			float2		   duv1 = uv2 - uv0; // Edge from vertex 0 to 2 in material space
			const float2x2 duv = float2x2(duv0, duv1);

			// Step 7: Invert to get Dm_inv
			const float2x2 inv_duv = luisa::inverse(duv);
			return inv_duv;
		}

		// ========================================================================
		// ALTERNATIVE HELPER: For debugging / verification
		// ========================================================================
		// This function can be used to verify that the improved version works correctly

		inline void verify_dm_inv_invariance(const float3& x_0, const float3& x_1, const float3& x_2)
		{
			// Compute Dm_inv for original ordering
			float2x2 dm_inv_012 = get_Dm_inv(x_0, x_1, x_2);

			// Compute Dm_inv for different ordering
			float2x2 dm_inv_021 = get_Dm_inv(x_0, x_2, x_1); // Swapped x_1 and x_2

			// Ideally, these should be identical (or close up to numerical precision)
			// Print for verification:
			LUISA_INFO("DMInv original (0,1,2): [{}, {}]", dm_inv_012[0], dm_inv_012[1]);
			LUISA_INFO("DMInv swapped  (0,2,1): [{}, {}]", dm_inv_021[0], dm_inv_021[1]);
		}

		inline float2x3 make_diff_mat3x2()
		{
			float2x3 result;
			result.set_zero();
			// x2 - x1
			result[0][0] = float(-1.0f);
			result[0][1] = float(1.0f);
			// x3 - x1
			result[1][0] = float(-1.0f);
			result[1][2] = float(1.0f);
			return result;
		}
		inline float3x3 make_diff_mat3x3()
		{
			float3x3 result = zero3x3;
			// x2 - x1
			result[0][0] = -1.0f;
			result[0][1] = 1.0f;
			// x3 - x1
			result[1][0] = -1.0f;
			result[1][2] = 1.0f;
			// x4 - x1
			result[2][0] = -1.0f;
			result[2][3] = 1.0f;
			return result;
		}
		inline Var<float2x3> make_diff_mat3x2_Var()
		{
			Var<float2x3> result;
			result->set_zero();
			// x2 - x1
			result.cols[0][0] = -1.0f;
			result.cols[0][1] = 1.0f;
			// x3 - x1
			result.cols[1][0] = -1.0f;
			result.cols[1][2] = 1.0f;
			return result;
		}

		inline float6 flatten(const float2x3& F)
		{
			float6 R;
			R.vec[0] = F.cols[0];
			R.vec[1] = F.cols[1];
			return R;
		}
		inline float9 flatten(const float3x3& F)
		{
			float9 R;
			R.vec[0] = F.cols[0];
			R.vec[1] = F.cols[1];
			R.vec[2] = F.cols[2];
			return R;
		}
		inline Var<float6> flatten(const Var<float2x3>& F)
		{
			Var<float6> R;
			R.vec[0] = F.cols[0];
			R.vec[1] = F.cols[1];
			return R;
		}
		inline Var<float9> flatten(const Var<float3x3>& F)
		{
			Var<float9> R;
			R.vec[0] = F[0];
			R.vec[1] = F[1];
			R.vec[2] = F[2];
			return R;
		}

		namespace
		{
			template <typename DmInvType, typename DiffMatType>
			static inline DiffMatType get_dFdx_2D(const DmInvType& InverseDm)
			{
				const auto d0 = InverseDm[0][0];
				const auto d1 = InverseDm[0][1];
				const auto d2 = InverseDm[1][0];
				const auto d3 = InverseDm[1][1];

				DiffMatType result_cm;
				if constexpr (std::is_same_v<DmInvType, LargeMatrix<9, 6>>)
				{
					result_cm.set_zero();
				}
				else if constexpr (std::is_same_v<DmInvType, Var<LargeMatrix<9, 6>>>)
				{
					result_cm->set_zero();
				}

				const auto val_1 = -d0 - d1;
				const auto val_2 = -d2 - d3;
				set_matrix_scalar(result_cm, 0, 0, val_1);
				set_matrix_scalar(result_cm, 0, 3, val_2);
				set_matrix_scalar(result_cm, 1, 1, val_1);
				set_matrix_scalar(result_cm, 1, 4, val_2);
				set_matrix_scalar(result_cm, 2, 2, val_1);
				set_matrix_scalar(result_cm, 2, 5, val_2);
				set_matrix_scalar(result_cm, 3, 0, d0);
				set_matrix_scalar(result_cm, 3, 3, d2);
				set_matrix_scalar(result_cm, 4, 1, d0);
				set_matrix_scalar(result_cm, 4, 4, d2);
				set_matrix_scalar(result_cm, 5, 2, d0);
				set_matrix_scalar(result_cm, 5, 5, d2);
				set_matrix_scalar(result_cm, 6, 0, d1);
				set_matrix_scalar(result_cm, 6, 3, d3);
				set_matrix_scalar(result_cm, 7, 1, d1);
				set_matrix_scalar(result_cm, 7, 4, d3);
				set_matrix_scalar(result_cm, 8, 2, d1);
				set_matrix_scalar(result_cm, 8, 5, d3);

				return result_cm;
			}

			template <typename DmInvType, typename DiffMatType>
			static inline DiffMatType get_dFdx_3D(const DmInvType& InverseDm)
			{
				const auto m = InverseDm[0][0];
				const auto n = InverseDm[0][1];
				const auto o = InverseDm[0][2];
				const auto p = InverseDm[1][0];
				const auto q = InverseDm[1][1];
				const auto r = InverseDm[1][2];
				const auto s = InverseDm[2][0];
				const auto t = InverseDm[2][1];
				const auto u = InverseDm[2][2];

				DiffMatType result_cm;
				if constexpr (std::is_same_v<DmInvType, LargeMatrix<12, 9>>)
				{
					result_cm.set_zero();
				}
				else if constexpr (std::is_same_v<DmInvType, Var<LargeMatrix<12, 9>>>)
				{
					result_cm->set_zero();
				}

				const auto val_1 = -m - n - o;
				const auto val_2 = -p - q - r;
				const auto val_3 = -s - t - u;
				set_matrix_scalar(result_cm, 0, 0, val_1);
				set_matrix_scalar(result_cm, 0, 3, val_2);
				set_matrix_scalar(result_cm, 0, 6, val_3);
				set_matrix_scalar(result_cm, 1, 1, val_1);
				set_matrix_scalar(result_cm, 1, 4, val_2);
				set_matrix_scalar(result_cm, 1, 7, val_3);
				set_matrix_scalar(result_cm, 2, 2, val_1);
				set_matrix_scalar(result_cm, 2, 5, val_2);
				set_matrix_scalar(result_cm, 2, 8, val_3);
				set_matrix_scalar(result_cm, 3, 0, m);
				set_matrix_scalar(result_cm, 3, 3, p);
				set_matrix_scalar(result_cm, 3, 6, s);
				set_matrix_scalar(result_cm, 4, 1, m);
				set_matrix_scalar(result_cm, 4, 4, p);
				set_matrix_scalar(result_cm, 4, 7, s);
				set_matrix_scalar(result_cm, 5, 2, m);
				set_matrix_scalar(result_cm, 5, 5, p);
				set_matrix_scalar(result_cm, 5, 8, s);
				set_matrix_scalar(result_cm, 6, 0, n);
				set_matrix_scalar(result_cm, 6, 3, q);
				set_matrix_scalar(result_cm, 6, 6, t);
				set_matrix_scalar(result_cm, 7, 1, n);
				set_matrix_scalar(result_cm, 7, 4, q);
				set_matrix_scalar(result_cm, 7, 7, t);
				set_matrix_scalar(result_cm, 8, 2, n);
				set_matrix_scalar(result_cm, 8, 5, q);
				set_matrix_scalar(result_cm, 8, 8, t);
				set_matrix_scalar(result_cm, 9, 0, o);
				set_matrix_scalar(result_cm, 9, 3, r);
				set_matrix_scalar(result_cm, 9, 6, u);
				set_matrix_scalar(result_cm, 10, 1, o);
				set_matrix_scalar(result_cm, 10, 4, r);
				set_matrix_scalar(result_cm, 10, 7, u);
				set_matrix_scalar(result_cm, 11, 2, o);
				set_matrix_scalar(result_cm, 11, 5, r);
				set_matrix_scalar(result_cm, 11, 8, u);

				return result_cm;
			}

		} // namespace

		inline LargeMatrix<9, 6> get_dFdx(const luisa::float2x2& InverseDm)
		{
			return get_dFdx_2D<luisa::float2x2, LargeMatrix<9, 6>>(InverseDm);
		}
		inline Var<LargeMatrix<9, 6>> get_dFdx(const Var<luisa::float2x2>& InverseDm)
		{
			return get_dFdx_2D<Var<luisa::float2x2>, Var<LargeMatrix<9, 6>>>(InverseDm);
		}
		inline LargeMatrix<12, 9> get_dFdx(const luisa::float3x3& InverseDm)
		{
			return get_dFdx_3D<luisa::float3x3, LargeMatrix<12, 9>>(InverseDm);
		}
		inline Var<LargeMatrix<12, 9>> get_dFdx(const Var<luisa::float3x3>& InverseDm)
		{
			return get_dFdx_3D<Var<luisa::float3x3>, Var<LargeMatrix<12, 9>>>(InverseDm);
		}

		// dFdx.T * dedF (9x6 mult 6x1 => 9x1)
		inline luisa::float3x3 convert_force(const float2x3& dedF, const luisa::float2x2& inv_rest2x2)
		{
			lcs::LargeVector<9> gradients = transpose(get_dFdx(inv_rest2x2)) * FemUtils::flatten(dedF);
			luisa::float3x3		result;
			result.cols[0] = gradients.block(0);
			result.cols[1] = gradients.block(1);
			result.cols[2] = gradients.block(2);
			return result;
		}
		inline float9x9 convert_hessian(const float6x6& d2ed2f, const luisa::float2x2& inv_rest2x2)
		{
			lcs::LargeMatrix<9, 6> dFdx = get_dFdx(inv_rest2x2);
			return transpose(dFdx) * d2ed2f * dFdx;
		}
		inline Var<luisa::float3x3> convert_force(const Var<float2x3>& dedF, const Var<luisa::float2x2>& inv_rest2x2)
		{
			Var<lcs::LargeVector<9>> gradients = transpose(get_dFdx(inv_rest2x2)) * FemUtils::flatten(dedF);
			Var<luisa::float3x3>	 result;
			result[0] = gradients->block(0);
			result[1] = gradients->block(1);
			result[2] = gradients->block(2);
			return result;
		}
		inline Var<float9x9> convert_hessian(const Var<float6x6>& d2ed2f, const Var<luisa::float2x2>& inv_rest2x2)
		{
			Var<lcs::LargeMatrix<9, 6>> dFdx = get_dFdx(inv_rest2x2);
			return transpose(dFdx) * d2ed2f * dFdx;
		}

	} // namespace FemUtils

} // namespace lcs