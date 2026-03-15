#pragma once

#include "Core/float_n.h"
#include "Core/float_nxn.h"
#include "Core/lc_to_eigen.h"
#include "Core/svd_2x2.h"
#include "Core/svd_3x3.h"
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
		inline Var<float6> flatten(const Var<float2x3>& F)
		{
			Var<float6> R;
			R.vec[0] = F.cols[0];
			R.vec[1] = F.cols[1];
			return R;
		}

		inline LargeMatrix<9, 6> get_dFdx(const luisa::float2x2& InverseDm)
		{
			const float d0 = InverseDm[0][0];
			const float d1 = InverseDm[0][1];
			const float d2 = InverseDm[1][0];
			const float d3 = InverseDm[1][1];
			const float s0 = d0 + d1;
			const float s1 = d2 + d3;

			lcs::LargeMatrix<9, 6> result;
			for (int i = 0; i < 3; i++)
			{
				result.scalar(i, i) = -s0;
				result.scalar(i, i + 3) = -s1;
			}
			for (int i = 0; i < 3; i++)
			{
				result.scalar(i + 3, i) = d0;
				result.scalar(i + 3, i + 3) = d2;
			}
			for (int i = 0; i < 3; i++)
			{
				result.scalar(i + 6, i) = d1;
				result.scalar(i + 6, i + 3) = d3;
			}
			return result;
		}
		inline LargeMatrix<6, 9> get_dFdx_T(const luisa::float2x2& InverseDm)
		{
			lcs::LargeMatrix<6, 9> dfdx_T = lcs::LargeMatrix<6, 9>::zero();

			const float d0 = InverseDm[0][0];
			const float d1 = InverseDm[0][1];
			const float d2 = InverseDm[1][0];
			const float d3 = InverseDm[1][1];
			const float s0 = d0 + d1;
			const float s1 = d2 + d3;

			dfdx_T.scalar<0, 0>() = -s0;
			dfdx_T.scalar<3, 0>() = -s1;
			dfdx_T.scalar<1, 1>() = -s0;
			dfdx_T.scalar<4, 1>() = -s1;
			dfdx_T.scalar<2, 2>() = -s0;
			dfdx_T.scalar<5, 2>() = -s1;
			dfdx_T.scalar<0, 3>() = d0;
			dfdx_T.scalar<3, 3>() = d2;
			dfdx_T.scalar<1, 4>() = d0;
			dfdx_T.scalar<4, 4>() = d2;
			dfdx_T.scalar<2, 5>() = d0;
			dfdx_T.scalar<5, 5>() = d2;
			dfdx_T.scalar<0, 6>() = d1;
			dfdx_T.scalar<3, 6>() = d3;
			dfdx_T.scalar<1, 7>() = d1;
			dfdx_T.scalar<4, 7>() = d3;
			dfdx_T.scalar<2, 8>() = d1;
			dfdx_T.scalar<5, 8>() = d3;
			return dfdx_T;
		}
		inline Var<LargeMatrix<6, 9>> get_dFdx_T(const Var<luisa::float2x2>& InverseDm)
		{
			Var<LargeMatrix<6, 9>> dfdx_T;
			dfdx_T->set_zero();

			const auto d0 = InverseDm[0][0];
			const auto d1 = InverseDm[0][1];
			const auto d2 = InverseDm[1][0];
			const auto d3 = InverseDm[1][1];
			const auto s0 = d0 + d1;
			const auto s1 = d2 + d3;

			dfdx_T->scalar<0, 0>() = -s0;
			dfdx_T->scalar<3, 0>() = -s1;
			dfdx_T->scalar<1, 1>() = -s0;
			dfdx_T->scalar<4, 1>() = -s1;
			dfdx_T->scalar<2, 2>() = -s0;
			dfdx_T->scalar<5, 2>() = -s1;
			dfdx_T->scalar<0, 3>() = d0;
			dfdx_T->scalar<3, 3>() = d2;
			dfdx_T->scalar<1, 4>() = d0;
			dfdx_T->scalar<4, 4>() = d2;
			dfdx_T->scalar<2, 5>() = d0;
			dfdx_T->scalar<5, 5>() = d2;
			dfdx_T->scalar<0, 6>() = d1;
			dfdx_T->scalar<3, 6>() = d3;
			dfdx_T->scalar<1, 7>() = d1;
			dfdx_T->scalar<4, 7>() = d3;
			dfdx_T->scalar<2, 8>() = d1;
			dfdx_T->scalar<5, 8>() = d3;
			return dfdx_T;
		}

		// dedF * dFdx (6x1 mult 6x9 => 1x9)
		inline luisa::float3x3 convert_force(const float2x3& dedF, const luisa::float2x2& inv_rest2x2)
		{
			const float3x2	g_T = (make_diff_mat3x2() * inv_rest2x2).transpose();
			const float3x2	dedF_T = dedF.transpose();
			luisa::float3x3 result;
			for (unsigned i = 0; i < 3; ++i)
			{
				for (unsigned dim = 0; dim < 3; ++dim)
				{
					result[i][dim] = luisa::dot(g_T[i], dedF_T[dim]);
				}
			}
			return result;
		}
		inline float9x9 convert_hessian(const float6x6& d2ed2f, const luisa::float2x2& inv_rest2x2)
		{
			lcs::LargeMatrix<6, 9> dfdx_T = get_dFdx_T(inv_rest2x2);

			float9x9 result;
			result.set_zero();
			for (unsigned i = 0; i < 6; ++i)
			{
				for (unsigned j = 0; j < 6; ++j)
				{
					result = result
						+ d2ed2f.scalar(j, i) * float9x9::outer_product(dfdx_T.column(i), dfdx_T.column(j));
				}
			}
			return result; // dfdx.transpose() * d2ed2f * dfdx;
		}
		inline Var<luisa::float3x3> convert_force(const Var<float2x3>& dedF, const Var<luisa::float2x2>& inv_rest2x2)
		{
			const Var<float3x2>	 g_T = transpose(make_diff_mat3x2_Var() * inv_rest2x2);
			const Var<float3x2>	 dedF_T = transpose(dedF);
			Var<luisa::float3x3> result;
			for (unsigned i = 0; i < 3; ++i)
			{
				for (unsigned dim = 0; dim < 3; ++dim)
				{
					result[i][dim] = luisa::compute::dot(g_T.cols[i], dedF_T.cols[dim]);
				}
			}
			return result;
		}
		inline Var<float9x9> convert_hessian(const Var<float6x6>& d2ed2f, const Var<luisa::float2x2>& inv_rest2x2)
		{
			Var<LargeMatrix<6, 9>> dfdx_T = get_dFdx_T(inv_rest2x2);
			Var<float9x9>		   result;
			result->set_zero();
			for (unsigned i = 0; i < 6; ++i)
			{
				for (unsigned j = 0; j < 6; ++j)
				{
					result = result + d2ed2f->scalar(j, i) * outer_product(dfdx_T->column(i), dfdx_T->column(j));
				}
			}
			return result; // dfdx.transpose() * d2ed2f * dfdx;
		}

	} // namespace FemUtils

} // namespace lcs