#pragma once

#include "Energies/detail/energy_detail_common.hpp"
#include "Energies/detail/fem_utils.h"
#include "SimulationCore/base_mesh.h"
#include <type_traits>

namespace lcs::detail::stretch_face_energy
{
	template <typename T>
	[[nodiscard]] inline T sqr(T x)
	{
		return x * x;
	}

	template <typename Vec3T, typename Mat2T, typename ScalarT>
	struct Input
	{
		Vec3T	x0;
		Vec3T	x1;
		Vec3T	x2;
		Mat2T	dm_inv;
		ScalarT mu;
		ScalarT lambda;
		ScalarT area;
	};

	[[nodiscard]] inline float stretch_energy(const float2x3& F, float mu)
	{
		const auto i5u = luisa::dot(F[0], F[0]);
		const auto i5v = luisa::dot(F[1], F[1]);
		return 0.5f * mu * (sqr(luisa::sqrt(i5u) - 1.0f) + sqr(luisa::sqrt(i5v) - 1.0f));
	}

	[[nodiscard]] inline float shear_energy(const float2x3& F, float lmd)
	{
		const auto i6 = luisa::dot(F[0], F[1]);
		return 0.5f * lmd * sqr(i6);
	}

	[[nodiscard]] inline luisa::compute::Var<float> stretch_energy(
		const luisa::compute::Var<float2x3>& F,
		luisa::compute::Var<float>			 mu)
	{
		const auto i5u = luisa::compute::dot(F.cols[0], F.cols[0]);
		const auto i5v = luisa::compute::dot(F.cols[1], F.cols[1]);
		return 0.5f * mu * (sqr(luisa::compute::sqrt(i5u) - 1.0f) + sqr(luisa::compute::sqrt(i5v) - 1.0f));
	}

	[[nodiscard]] inline luisa::compute::Var<float> shear_energy(
		const luisa::compute::Var<float2x3>& F,
		luisa::compute::Var<float>			 lmd)
	{
		const auto i6 = luisa::compute::dot(F.cols[0], F.cols[1]);
		return 0.5f * lmd * sqr(i6);
	}

	[[nodiscard]] inline float2x3 stretch_gradient(const float2x3& F, const float mu)
	{
		const float3& Fu = F.cols[0];
		const float3& Fv = F.cols[1];

		const float I5u = luisa::dot(Fu, Fu);
		const float I5v = luisa::dot(Fv, Fv);

		float sqrtI5u = luisa::sqrt(I5u);
		float sqrtI5v = luisa::sqrt(I5v);
		float invSqrtI5u = 1.0f / sqrtI5u;
		float invSqrtI5v = 1.0f / sqrtI5v;

		float2x3 result;
		result.cols[0] = (sqrtI5u - 1.0f) * invSqrtI5u * Fu;
		result.cols[1] = (sqrtI5v - 1.0f) * invSqrtI5v * Fv;
		return mu * result;
	}

	[[nodiscard]] inline float2x3 shear_gradient(const float2x3& F, const float lmd)
	{
		float	 w = luisa::dot(F.cols[0], F.cols[1]);
		float2x3 result;
		result.cols[0] = w * F.cols[1];
		result.cols[1] = w * F.cols[0];
		return lmd * result;
	}

	[[nodiscard]] inline float6x6 stretch_hessian(const float2x3& F, float mu)
	{
		float6x6 H = float6x6::zero();

		const float3& Fu = F.cols[0];
		const float3& Fv = F.cols[1];

		const float I5u = luisa::dot(Fu, Fu);
		const float I5v = luisa::dot(Fv, Fv);

		float sqrtI5u = luisa::sqrt(I5u);
		float sqrtI5v = luisa::sqrt(I5v);
		float invSqrtI5u = 1.0f / sqrtI5u;
		float invSqrtI5v = 1.0f / sqrtI5v;

		H.scalar(0, 0) = H.scalar(1, 1) = H.scalar(2, 2) = luisa::max(0.0f, 1.0f - invSqrtI5u);
		H.scalar(3, 3) = H.scalar(4, 4) = H.scalar(5, 5) = luisa::max(0.0f, 1.0f - invSqrtI5v);

		float3 fu = F.cols[0] * invSqrtI5u;
		float3 fv = F.cols[1] * invSqrtI5v;

		float uCoeff = (invSqrtI5u < 1.0f) ? invSqrtI5u : 1.0f;
		float vCoeff = (invSqrtI5v < 1.0f) ? invSqrtI5v : 1.0f;
		H.block(0, 0) = H.block(0, 0) + uCoeff * outer_product(fu, fu);
		H.block(1, 1) = H.block(1, 1) + vCoeff * outer_product(fv, fv);
		return mu * H;
	}

	[[nodiscard]] inline float6x6 shear_hessian(const float2x3& F, float mu)
	{
		float6x6 H = float6x6::zero();

		const float3& Fu = F.cols[0];
		const float3& Fv = F.cols[1];

		const float I6 = luisa::dot(Fu, Fv);
		const float signI6 = luisa::sign(I6);

		H.scalar<3, 0>() = H.scalar<4, 1>() = H.scalar<5, 2>() = H.scalar<0, 3>() =
			H.scalar<1, 4>() = H.scalar<2, 5>() = 1.0f;

		const float6 g = FemUtils::flatten(F * luisa::make_float2x2(0, 1, 1, 0));

		const float I2 = luisa::dot(Fu, Fu) + luisa::dot(Fv, Fv);
		const float lambda0 = 0.5f * (I2 + luisa::sqrt(I2 * I2 + 12.0f * I6 * I6));

		const float6 q0 = (I6 * H * g + lambda0 * g).normalize();
		float6x6	 T = float6x6::identity();
		T = 0.5f * (T + signI6 * H);
		const float6 Tq = T * q0;
		const float	 normTq = Tq.squared_norm();

		H = luisa::abs(I6) * (T - float6x6::outer_product(Tq, Tq) / normTq)
			+ lambda0 * float6x6::outer_product(q0, q0);

		return mu * H;
	}

	[[nodiscard]] inline luisa::compute::Var<float2x3> stretch_gradient(
		const luisa::compute::Var<float2x3>& F,
		const luisa::compute::Var<float>	 mu)
	{
		const auto& Fu = F.cols[0];
		const auto& Fv = F.cols[1];

		const auto I5u = luisa::compute::dot(Fu, Fu);
		const auto I5v = luisa::compute::dot(Fv, Fv);

		const auto sqrtI5u = luisa::compute::sqrt(I5u);
		const auto sqrtI5v = luisa::compute::sqrt(I5v);
		const auto invSqrtI5u = 1.0f / sqrtI5u;
		const auto invSqrtI5v = 1.0f / sqrtI5v;

		luisa::compute::Var<float2x3> result;
		result.cols[0] = (sqrtI5u - 1.0f) * invSqrtI5u * Fu;
		result.cols[1] = (sqrtI5v - 1.0f) * invSqrtI5v * Fv;
		return mu * result;
	}

	[[nodiscard]] inline luisa::compute::Var<float2x3> shear_gradient(
		const luisa::compute::Var<float2x3>& F,
		const luisa::compute::Var<float>	 lmd)
	{
		luisa::compute::Var<float>	  w = luisa::compute::dot(F.cols[0], F.cols[1]);
		luisa::compute::Var<float2x3> result;
		result.cols[0] = w * F.cols[1];
		result.cols[1] = w * F.cols[0];
		return lmd * result;
	}

	[[nodiscard]] inline luisa::compute::Var<float6x6> stretch_hessian(
		const luisa::compute::Var<float2x3>& F,
		luisa::compute::Var<float>			 mu)
	{
		using Float6x6 = luisa::compute::Var<float6x6>;
		using Float3 = luisa::compute::Var<float3>;
		using Float = luisa::compute::Var<float>;

		luisa::compute::Var<float6x6> H;
		H->set_zero();

		const Float3& Fu = F.cols[0];
		const Float3& Fv = F.cols[1];

		const Float I5u = luisa::compute::dot(Fu, Fu);
		const Float I5v = luisa::compute::dot(Fv, Fv);

		const Float sqrtI5u = luisa::compute::sqrt(I5u);
		const Float sqrtI5v = luisa::compute::sqrt(I5v);
		const Float invSqrtI5u = 1.0f / sqrtI5u;
		const Float invSqrtI5v = 1.0f / sqrtI5v;

		H->scalar(0, 0) = H->scalar(1, 1) = H->scalar(2, 2) = luisa::compute::max(0.0f, 1.0f - invSqrtI5u);
		H->scalar(3, 3) = H->scalar(4, 4) = H->scalar(5, 5) = luisa::compute::max(0.0f, 1.0f - invSqrtI5v);

		Float3 fu = F.cols[0] * invSqrtI5u;
		Float3 fv = F.cols[1] * invSqrtI5v;

		luisa::compute::Var<float> uCoeff = luisa::compute::min(invSqrtI5u, 1.0f);
		luisa::compute::Var<float> vCoeff = luisa::compute::min(invSqrtI5v, 1.0f);
		H->block(0, 0) = H->block(0, 0) + uCoeff * outer_product(fu, fu);
		H->block(1, 1) = H->block(1, 1) + vCoeff * outer_product(fv, fv);
		return mu * H;
	}

	[[nodiscard]] inline luisa::compute::Var<float6x6> shear_hessian(
		const luisa::compute::Var<float2x3>& F,
		luisa::compute::Var<float>			 mu)
	{
		using Float = luisa::compute::Var<float>;
		using Float3 = luisa::compute::Var<float3>;
		using Float6 = luisa::compute::Var<float6>;
		using Float6x6 = luisa::compute::Var<float6x6>;

		Float6x6 H;
		H->set_zero();

		const Float3& Fu = F.cols[0];
		const Float3& Fv = F.cols[1];

		const Float I6 = luisa::compute::dot(Fu, Fv);
		const Float signI6 = luisa::compute::sign(I6);

		H->scalar<3, 0>() = H->scalar<4, 1>() = H->scalar<5, 2>() = H->scalar<0, 3>() =
			H->scalar<1, 4>() = H->scalar<2, 5>() = 1.0f;

		luisa::compute::Var<float2x2> tmp = luisa::compute::make_float2x2(luisa::compute::make_float2(0.0f, 1.0f),
			luisa::compute::make_float2(1.0f, 0.0f));
		const Float6				  g = FemUtils::flatten(F * tmp);

		const Float I2 = luisa::compute::dot(Fu, Fu) + luisa::compute::dot(Fv, Fv);
		const Float lambda0 = 0.5f * (I2 + luisa::compute::sqrt(I2 * I2 + 12.0f * I6 * I6));

		const Float6 q0 = (I6 * H * g + lambda0 * g)->normalize();

		Float6x6 T;
		T->set_identity();
		T = 0.5f * (T + signI6 * H);
		const Float6 Tq = T * q0;
		const auto	 normTq = Tq->squared_norm();

		H = luisa::compute::abs(I6) * (T - outer_product(Tq, Tq) / normTq) + lambda0 * outer_product(q0, q0);

		return mu * H;
	}

	template <typename Vec3T, typename Mat2T, typename ScalarT>
	[[nodiscard]] inline auto compute_energy(const Input<Vec3T, Mat2T, ScalarT>& in)
	{
		const auto F = makeFloat2x3(in.x1 - in.x0, in.x2 - in.x0) * in.dm_inv;
		return in.area * (stretch_energy(F, in.mu) + shear_energy(F, in.lambda));
	}

	[[nodiscard]] inline auto evaluate(const Input<float3, float2x2, float>& in)
	{
		float2x3 F = makeFloat2x3(in.x1 - in.x0, in.x2 - in.x0) * in.dm_inv;

		float2x3 de0dF = stretch_gradient(F, in.mu);
		float6x6 d2e0dF2 = stretch_hessian(F, in.mu);

		float2x3 de1dF = shear_gradient(F, in.lambda);
		float6x6 d2e1dF2 = shear_hessian(F, in.lambda);

		float2x3 dedF = de0dF + de1dF;
		float6x6 d2edF2 = d2e0dF2 + d2e1dF2;

		LargeMatrix<9, 6> dFdx = FemUtils::get_dFdx(in.dm_inv);
		LargeVector<9>	  gradients = in.area * transpose(dFdx) * FemUtils::flatten(dedF);
		LargeMatrix<9, 9> hessians = in.area * transpose(dFdx) * d2edF2 * dFdx;

		// float3x3 gradients = in.area * FemUtils::convert_force(dedF, in.dm_inv);
		// float9x9 hessians = in.area * FemUtils::convert_hessian(d2edF2, in.dm_inv);

		EnergyEvalResult<3, 9, float3, float3x3> out{};

		// out.gradients[0] = gradients[0];
		// out.gradients[1] = gradients[1];
		// out.gradients[2] = gradients[2];
		out.gradients[0] = gradients.block(0);
		out.gradients[1] = gradients.block(1);
		out.gradients[2] = gradients.block(2);

		out.hessians[0] = hessians.block(0, 0);
		out.hessians[1] = hessians.block(1, 0);
		out.hessians[2] = hessians.block(2, 0);
		out.hessians[3] = hessians.block(0, 1);
		out.hessians[4] = hessians.block(1, 1);
		out.hessians[5] = hessians.block(2, 1);
		out.hessians[6] = hessians.block(0, 2);
		out.hessians[7] = hessians.block(1, 2);
		out.hessians[8] = hessians.block(2, 2);
		return out;
	}

	[[nodiscard]] inline auto evaluate(const Input<luisa::compute::Var<float3>, luisa::compute::Var<float2x2>, luisa::compute::Var<float>>& in)
	{
		luisa::compute::Var<float2x3> F = makeFloat2x3(in.x1 - in.x0, in.x2 - in.x0) * in.dm_inv;

		auto de0dF = stretch_gradient(F, in.mu);
		auto d2e0dF2 = stretch_hessian(F, in.mu);

		auto de1dF = shear_gradient(F, in.lambda);
		auto d2e1dF2 = shear_hessian(F, in.lambda);

		auto dedF = de0dF + de1dF;
		auto d2edF2 = d2e0dF2 + d2e1dF2;

		// luisa::compute::Var<float3x3> gradients = in.area * FemUtils::convert_force(dedF, in.dm_inv);
		// luisa::compute::Var<float9x9> hessians = in.area * FemUtils::convert_hessian(d2edF2, in.dm_inv);

		Var<LargeMatrix<9, 6>> dFdx = FemUtils::get_dFdx(in.dm_inv);
		Var<LargeVector<9>>	   gradients = in.area * transpose(dFdx) * FemUtils::flatten(dedF);
		Var<LargeMatrix<9, 9>> hessians = in.area * transpose(dFdx) * d2edF2 * dFdx;

		using GradientOutT = std::decay_t<decltype(gradients->block(0))>;
		using HessianOutT = std::decay_t<decltype(hessians->block(0, 0))>;
		EnergyEvalResult<3, 9, GradientOutT, HessianOutT> out{};

		// out.gradients[0] = gradients[0];
		// out.gradients[1] = gradients[1];
		// out.gradients[2] = gradients[2];
		out.gradients[0] = gradients->block(0);
		out.gradients[1] = gradients->block(1);
		out.gradients[2] = gradients->block(2);

		out.hessians[0] = hessians->block(0, 0);
		out.hessians[1] = hessians->block(1, 0);
		out.hessians[2] = hessians->block(2, 0);
		out.hessians[3] = hessians->block(0, 1);
		out.hessians[4] = hessians->block(1, 1);
		out.hessians[5] = hessians->block(2, 1);
		out.hessians[6] = hessians->block(0, 2);
		out.hessians[7] = hessians->block(1, 2);
		out.hessians[8] = hessians->block(2, 2);
		return out;
	}

} // namespace lcs::detail::stretch_face_energy
