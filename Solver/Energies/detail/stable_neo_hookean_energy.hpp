#pragma once

#include "Core/float_n.h"
#include "Core/lc_to_eigen.h"
#include "Core/scalar.h"
#include "Core/svd_3x3.h"
#include "Energies/detail/energy_detail_common.hpp"
#include "Energies/detail/fem_utils.h"
#include "luisa/core/basic_types.h"
#include <type_traits>

namespace lcs::detail::stable_neo_hookean_energy
{
	template <typename Vec3T, typename Mat3T, typename ScalarT>
	struct Input;

	namespace
	{
		template <typename Float3, typename Float3x3>
		[[nodiscard]] inline auto get_F(
			const Float3& x0,
			const Float3& x1,
			const Float3& x2,
			const Float3& x3,
			const Float3x3& Dm_inv)
		{
			auto Ds = make_float3x3(x1 - x0, x2 - x0, x3 - x0);
			return Ds * Dm_inv;
		}

		template <typename Float, typename Float3x3>
		[[nodiscard]] inline auto partial_J_partial_F(const Float3x3& F)
		{
			Float3x3 pJpF;
			pJpF[0] = cross_vec(F[1], F[2]);
			pJpF[1] = cross_vec(F[2], F[0]);
			pJpF[2] = cross_vec(F[0], F[1]);
			return pJpF;
		}

		template <typename Float, typename Float3, typename Float3x3>
		[[nodiscard]] inline auto first_pk_stress(const Float3x3& F, const Float mu, const Float lambda)
		{
			const Float	  J = determinant(F);
			const Float3x3 pJpF = partial_J_partial_F<Float, Float3x3>(F);
			const Float	  coeff = lambda * (J - Float(1.0f)) - mu;
			return mu * F + coeff * pJpF;
		}

		template <typename Float, typename Float3, typename Float3x3>
		[[nodiscard]] inline auto cross_product_matrix(const Float3& v)
		{
			Float3x3 result;
			result[0] = makeFloat3(Float(0.0f), -v[2], v[1]);
			result[1] = makeFloat3(v[2], Float(0.0f), -v[0]);
			result[2] = makeFloat3(-v[1], v[0], Float(0.0f));
			return result;
		}

		template <typename Float, typename Float3, typename Float3x3, typename Float9x9>
		[[nodiscard]] inline auto exact_hessian_wrt_F(const Float3x3& F, const Float mu, const Float lambda)
		{
			const Float3x3 pJpF = partial_J_partial_F<Float, Float3x3>(F);
			const Float	  J = determinant(F);
			const Float	  coeff = lambda * (J - Float(1.0f)) - mu;

			Float9x9 H9;
			if constexpr (std::is_same_v<Float9x9, float9x9>)
			{
				H9.set_zero();
				H9.set_diag(mu * identity3x3);
			}
			else
			{
				H9->set_zero();
				H9->set_diag(mu * identity3x3);
			}

			H9 = H9 + lambda * outer_product_largevec(FemUtils::flatten(pJpF), FemUtils::flatten(pJpF));

			const auto f0hat = coeff * cross_product_matrix<Float, Float3, Float3x3>(F[0]);
			const auto f1hat = coeff * cross_product_matrix<Float, Float3, Float3x3>(F[1]);
			const auto f2hat = coeff * cross_product_matrix<Float, Float3, Float3x3>(F[2]);

			if constexpr (std::is_same_v<Float9x9, float9x9>)
			{
				H9.block(0, 1) = H9.block(0, 1) - f2hat;
				H9.block(1, 0) = H9.block(1, 0) + f2hat;
				H9.block(0, 2) = H9.block(0, 2) + f1hat;
				H9.block(2, 0) = H9.block(2, 0) - f1hat;
				H9.block(1, 2) = H9.block(1, 2) - f0hat;
				H9.block(2, 1) = H9.block(2, 1) + f0hat;
			}
			else
			{
				H9->block(0, 1) = H9->block(0, 1) - f2hat;
				H9->block(1, 0) = H9->block(1, 0) + f2hat;
				H9->block(0, 2) = H9->block(0, 2) + f1hat;
				H9->block(2, 0) = H9->block(2, 0) - f1hat;
				H9->block(1, 2) = H9->block(1, 2) - f0hat;
				H9->block(2, 1) = H9->block(2, 1) + f0hat;
			}
			return H9;
		}

		template <typename ScalarT, typename Vec3T, typename Mat3T, typename Vec9T, typename Mat9T>
		[[nodiscard]] inline auto evaluate_exact_template(const Input<Vec3T, Mat3T, ScalarT>& in)
		{
			const Mat3T F = get_F(in.x0, in.x1, in.x2, in.x3, in.dm_inv);
			const Mat3T dEdF = first_pk_stress<ScalarT, Vec3T, Mat3T>(F, in.mu, in.lambda);
			const Mat9T d2EdF2 = exact_hessian_wrt_F<ScalarT, Vec3T, Mat3T, Mat9T>(F, in.mu, in.lambda);
			const auto  dFdx = FemUtils::get_dFdx(in.dm_inv);
			const auto  G = in.volume * transpose(dFdx) * FemUtils::flatten(dEdF);
			const auto  H = in.volume * transpose(dFdx) * d2EdF2 * dFdx;

			EnergyEvalResult<4, 16, Vec3T, Mat3T> out{};
			for (int i = 0; i < 4; ++i)
			{
				if constexpr (std::is_same_v<Vec3T, float3>)
					out.gradients[i] = G.block(i);
				else
					out.gradients[i] = G->block(i);

				for (int j = 0; j < 4; ++j)
				{
					if constexpr (std::is_same_v<Mat3T, float3x3>)
						out.hessians[i * 4 + j] = H.block(j, i);
					else
						out.hessians[i * 4 + j] = H->block(j, i);
				}
			}
			return out;
		}

		template <typename Float, typename Float3, typename Float3x3>
		inline Float compute_energy_impl(
			const Float3&	x0,
			const Float3&	x1,
			const Float3&	x2,
			const Float3&	x3,
			const Float3x3& Dm_inv,
			const Float		mu,
			const Float		lambda,
			const Float		volume)
		{
			Float3x3 F = get_F(x0, x1, x2, x3, Dm_inv);

			// I2 = ||F||_F^2
			Float I2 = 0.0f;
			for (int c = 0; c < 3; c++)
				for (int r = 0; r < 3; r++)
					I2 = I2 + F[c][r] * F[c][r];

			Float I3 = determinant(F);

			Float psi = 0.5f * mu * (I2 - 3.0f)
				- mu * (I3 - 1.0f)
				+ 0.5f * lambda * (I3 - 1.0f) * (I3 - 1.0f);

			return volume * psi;
		}
		template <typename Float, typename Float3, typename Float3x3>
		inline void compute_gradient_hessian_impl(
			const Float3&	x0,
			const Float3&	x1,
			const Float3&	x2,
			const Float3&	x3,
			const Float3x3& Dm_inv,
			const Float		mu,
			const Float		lambda,
			const Float		volume,
			Float3			gradient[4],
			Float3x3		hessian[16])
		{
			auto eval = evaluate_exact_template<Float, Float3, Float3x3, float9, float9x9>(
				Input<Float3, Float3x3, Float>{x0, x1, x2, x3, Dm_inv, mu, lambda, volume});
			for (int i = 0; i < 4; ++i)
			{
				gradient[i] = eval.gradients[i];
			}
			for (int i = 0; i < 16; ++i)
			{
				hessian[i] = eval.hessians[i];
			}
		}
	}; // namespace
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

	template <typename Vec3T, typename Mat3T, typename ScalarT>
	[[nodiscard]] inline auto compute_energy(const Input<Vec3T, Mat3T, ScalarT>& in)
	{
		return compute_energy_impl(
			in.x0, in.x1, in.x2, in.x3, in.dm_inv, in.mu, in.lambda, in.volume);
	}

	template <typename Vec3T, typename Mat3T, typename ScalarT>
	[[nodiscard]] inline auto evaluate(const Input<Vec3T, Mat3T, ScalarT>& in)
	{
		return evaluate_exact_template<
			ScalarT,
			Vec3T,
			Mat3T,
			std::decay_t<decltype(FemUtils::flatten(in.dm_inv))>,
			std::conditional_t<std::is_same_v<Mat3T, float3x3>, float9x9, luisa::compute::Var<float9x9>>>(in);
	}

	[[nodiscard]] inline auto evaluate(const Input<float3, float3x3, float>& in)
	{
		auto out = evaluate_exact_template<float, float3, float3x3, float9, float9x9>(in);

		float12x12 H12;
		H12.set_zero();
		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < 4; ++j)
			{
				H12.block(i, j) = out.hessians[i * 4 + j];
			}
		}

		EigenFloat12x12 H_eigen = float12x12_to_eigen12x12(H12);
		H_eigen = 0.5f * (H_eigen + H_eigen.transpose());
		H_eigen = spd_projection<12>(H_eigen);

		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < 4; ++j)
			{
				out.hessians[i * 4 + j] = eigen3x3_to_float3x3(H_eigen.block<3, 3>(i * 3, j * 3));
			}
		}
		return out;
	}

	template <typename Vec3T, typename Mat3T, typename ScalarT>
	[[nodiscard]] inline auto evaluate_host(const Input<Vec3T, Mat3T, ScalarT>& in)
	{
		return evaluate(in);
	}

} // namespace lcs::detail::stable_neo_hookean_energy
