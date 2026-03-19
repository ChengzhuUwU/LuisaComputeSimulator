#pragma once

#include "Core/float_nxn.h"
#include "Core/svd_3x3.h"
#include "Core/float_n.h"
#include "Core/xbasic_types.h"
#include "Energies/detail/energy_detail_common.hpp"
#include "Energies/detail/fem_utils.h"
#include "SimulationCore/base_mesh.h"
#include <type_traits>

namespace lcs::detail::arap_tet_energy
{
	constexpr float sqrt2 = 1.4142135623730951f;
	constexpr float rsqrt2 = 1.0f / sqrt2;

	template <typename Vec3T, typename Mat3T, typename ScalarT>
	struct Input
	{
		Vec3T	x0;
		Vec3T	x1;
		Vec3T	x2;
		Vec3T	x3;
		Mat3T	dm_inv;
		ScalarT mu;
		ScalarT lambda;
		ScalarT volume;
	};

	[[nodiscard]] inline float3x3 make_twist_mode_0()
	{
		// Row-major in reference:
		// [ 0 -1  0 ]
		// [ 1  0  0 ]
		// [ 0  0  0 ]
		// Here float3x3 is column-major M[col][row].
		return luisa::make_float3x3(luisa::make_float3(0.0f, 1.0f, 0.0f),
			luisa::make_float3(-1.0f, 0.0f, 0.0f),
			luisa::make_float3(0.0f, 0.0f, 0.0f));
	}

	[[nodiscard]] inline float3x3 make_twist_mode_1()
	{
		// Row-major in reference:
		// [ 0  0  0 ]
		// [ 0  0  1 ]
		// [ 0 -1  0 ]
		return luisa::make_float3x3(luisa::make_float3(0.0f, 0.0f, 0.0f),
			luisa::make_float3(0.0f, 0.0f, -1.0f),
			luisa::make_float3(0.0f, 1.0f, 0.0f));
	}

	[[nodiscard]] inline float3x3 make_twist_mode_2()
	{
		// Row-major in reference:
		// [ 0  0  1 ]
		// [ 0  0  0 ]
		// [-1  0  0 ]
		return luisa::make_float3x3(luisa::make_float3(0.0f, 0.0f, -1.0f),
			luisa::make_float3(0.0f, 0.0f, 0.0f),
			luisa::make_float3(1.0f, 0.0f, 0.0f));
	}

	[[nodiscard]] inline float3x3 polar_rotation(const float3x3& F)
	{
		float3x3 U, V;
		float3	 S;
		lcs::svd(F, U, S, V);
		return U * transpose(V);
	}

	[[nodiscard]] inline Var<float3x3> polar_rotation(const Var<float3x3>& F)
	{
		Var<float3x3> U, V;
		Var<float3>	  S;
		lcs::svd(F, U, S, V);
		return U * transpose(V);
	}

	[[nodiscard]] inline float3x3 get_F(const float3& x0,
		const float3&								  x1,
		const float3&								  x2,
		const float3&								  x3,
		const float3x3&								  dm_inv)
	{
		float3x3 Ds = luisa::make_float3x3(x1 - x0, x2 - x0, x3 - x0);
		return Ds * dm_inv;
	}

	[[nodiscard]] inline Var<float3x3> get_F(const Var<float3>& x0,
		const Var<float3>&										x1,
		const Var<float3>&										x2,
		const Var<float3>&										x3,
		const Var<float3x3>&									dm_inv)
	{
		Var<float3x3> Ds = luisa::compute::make_float3x3(x1 - x0, x2 - x0, x3 - x0);
		return Ds * dm_inv;
	}

	[[nodiscard]] inline float compute_energy(const Input<float3, float3x3, float>& in)
	{
		float3x3 F = get_F(in.x0, in.x1, in.x2, in.x3, in.dm_inv);
		float3x3 R = polar_rotation(F);
		// Match reference core: E_arap = kappa * v * ||F - R||^2.
		//  == I2 - 2 * I1 + 3, where I1 = tr(F^T R) and I2 = tr(R^T F F^T R).
		float psi_arap = sqr_frobenius(F - R);
		return in.volume * in.mu * (psi_arap);
	}
	[[nodiscard]] inline auto compute_energy(const Input<Var<float3>, Var<float3x3>, Var<float>>& in)
	{
		using namespace luisa::compute;
		Var<float3x3> F = get_F(in.x0, in.x1, in.x2, in.x3, in.dm_inv);
		Var<float3x3> R = polar_rotation(F);
		Var<float>	  psi_arap = sqr_frobenius(F - R);
		return in.volume * in.mu * (psi_arap);
	}

	template <typename ScalarT, typename Vec3T, typename Mat3T, typename Vec9T, typename Mat9T>
	[[nodiscard]] inline auto evaluate_template(const Input<Vec3T, Mat3T, ScalarT>& in)
	{

		Mat3T F = get_F(in.x0, in.x1, in.x2, in.x3, in.dm_inv);
		Mat3T U, V;
		Vec3T S;
		lcs::svd(F, U, S, V);

		Mat3T R = U * transpose(V);

		Mat3T dEdF = 2.0f * (F - R);

		// d2E/dF2 (ARAP part)
		Mat3T	Q0 = rsqrt2 * U * make_twist_mode_0() * transpose(V);
		Mat3T	Q1 = rsqrt2 * U * make_twist_mode_1() * transpose(V);
		Mat3T	Q2 = rsqrt2 * U * make_twist_mode_2() * transpose(V);
		Vec9T	t0 = FemUtils::flatten(Q0);
		Vec9T	t1 = FemUtils::flatten(Q1);
		Vec9T	t2 = FemUtils::flatten(Q2);
		ScalarT s0 = S[0];
		ScalarT s1 = S[1];
		ScalarT s2 = S[2];

		Mat9T H9;
		if constexpr (std::is_same_v<Mat3T, float3x3>)
		{
			H9.set_zero();
			H9.set_diag(2.0f * identity3x3); // dEdF = 2 * I - 2 * H1
		}
		else
		{
			H9->set_zero();
			H9->set_diag(2.0f * identity3x3); // dEdF = 2 * I - 2 * H1
		}
		auto subtract_mode = [&](Mat9T& mat, const Vec9T& t, const ScalarT coeff)
		{
			mat = mat - coeff * outer_product_largevec(t, t);
		};

		const ScalarT eps = 1e-8f;
		subtract_mode(H9, t0, 4.0f / max_scalar(s0 + s1, eps)); // 2/(s0+s1) * 2 from the chain rule
		subtract_mode(H9, t1, 4.0f / max_scalar(s1 + s2, eps));
		subtract_mode(H9, t2, 4.0f / max_scalar(s0 + s2, eps));

		EnergyEvalResult<4, 16, Vec3T, Mat3T> out{};

		ScalarT stiffness = in.mu * in.volume;

		auto dFdx = FemUtils::get_dFdx(in.dm_inv);
		auto G = stiffness * transpose(dFdx) * FemUtils::flatten(dEdF); // Vec12
		auto H = stiffness * transpose(dFdx) * H9 * dFdx;				// Mat12x12
		for (int i = 0; i < 4; i++)
		{
			if constexpr (std::is_same_v<Vec3T, float3>)
				out.gradients[i] = G.block(i);
			else
				out.gradients[i] = G->block(i);

			for (int j = 0; j < 4; j++)
			{
				if constexpr (std::is_same_v<Mat3T, float3x3>)
					out.hessians[i * 4 + j] = H.block(j, i);
				else
					out.hessians[i * 4 + j] = H->block(j, i);
			}
		}
		return out;
	}

	[[nodiscard]] inline auto evaluate_host(const Input<float3, float3x3, float>& in)
	{
		return evaluate_template<float, float3, float3x3, float9, float9x9>(in);
	}

	[[nodiscard]] inline auto evaluate(
		const Input<luisa::compute::Var<float3>, luisa::compute::Var<float3x3>, luisa::compute::Var<float>>& in)
	{
		return evaluate_template<
			luisa::compute::Var<float>,
			luisa::compute::Var<float3>,
			luisa::compute::Var<float3x3>,
			luisa::compute::Var<float9>,
			luisa::compute::Var<float9x9>>(in);
	}

} // namespace lcs::detail::arap_tet_energy
