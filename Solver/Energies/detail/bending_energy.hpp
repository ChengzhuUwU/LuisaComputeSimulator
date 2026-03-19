#pragma once

#include "Energies/detail/energy_detail_common.hpp"
#include "SimulationCore/base_mesh.h"
#include <luisa/dsl/builtin.h>
#include <array>
#include <cassert>

namespace lcs::detail::bending_energy
{
	using Float3 = luisa::compute::Float3;
	using Float = luisa::compute::Float;

	namespace
	{
		using HostVector12 = std::array<float3, 4>;
		static inline HostVector12 face_dihedral_angle_grad(const float3& v2, const float3& v0, const float3& v1, const float3& v3)
		{
			const float3 e0 = v1 - v0;
			const float3 e1 = v2 - v0;
			const float3 e2 = v3 - v0;
			const float3 e3 = v2 - v1;
			const float3 e4 = v3 - v1;
			const float3 n1 = luisa::cross(e0, e1);
			const float3 n2 = luisa::cross(e2, e0);
			const float	 n1_sqnm = luisa::dot(n1, n1);
			const float	 n2_sqnm = luisa::dot(n2, n2);
			const float	 e0_norm = luisa::length(e0);
			assert(n1_sqnm > 0.0f);
			assert(n2_sqnm > 0.0f);
			assert(e0_norm > 0.0f);

			HostVector12 grad;
			grad[0] = -e0_norm / n1_sqnm * n1;
			grad[1] = -luisa::dot(e0, e3) / (e0_norm * n1_sqnm) * n1 - luisa::dot(e0, e4) / (e0_norm * n2_sqnm) * n2;
			grad[2] = luisa::dot(e0, e1) / (e0_norm * n1_sqnm) * n1 + luisa::dot(e0, e2) / (e0_norm * n2_sqnm) * n2;
			grad[3] = -e0_norm / n2_sqnm * n2;
			return grad;
		}

		using DeviceVector12 = luisa::compute::ArrayFloat3<4>;
		static inline DeviceVector12 face_dihedral_angle_grad(const Float3& v2,
			const Float3&													v0,
			const Float3&													v1,
			const Float3&													v3)
		{

			const Float3 e0 = v1 - v0;
			const Float3 e1 = v2 - v0;
			const Float3 e2 = v3 - v0;
			const Float3 e3 = v2 - v1;
			const Float3 e4 = v3 - v1;
			const Float3 n1 = luisa::compute::cross(e0, e1);
			const Float3 n2 = luisa::compute::cross(e2, e0);
			const Float	 n1_sqnm = luisa::compute::dot(n1, n1);
			const Float	 n2_sqnm = luisa::compute::dot(n2, n2);
			const Float	 e0_norm = luisa::compute::length(e0);
			luisa::compute::device_assert(n1_sqnm > 0.0f);
			luisa::compute::device_assert(n2_sqnm > 0.0f);
			luisa::compute::device_assert(e0_norm > 0.0f);

			DeviceVector12 grad;
			grad[0] = -e0_norm / n1_sqnm * n1;
			grad[1] = -luisa::compute::dot(e0, e3) / (e0_norm * n1_sqnm) * n1
				- luisa::compute::dot(e0, e4) / (e0_norm * n2_sqnm) * n2;
			grad[2] = luisa::compute::dot(e0, e1) / (e0_norm * n1_sqnm) * n1
				+ luisa::compute::dot(e0, e2) / (e0_norm * n2_sqnm) * n2;
			grad[3] = -e0_norm / n2_sqnm * n2;
			return grad;
		}

		static inline float face_dihedral_angle(const float3& v0, const float3& v1, const float3& v2, const float3& v3)
		{
			const float3 n1 = luisa::cross(v1 - v0, v2 - v0);
			const float3 n2 = luisa::cross(v2 - v3, v1 - v3);
			float		 dot = luisa::dot(n1, n2) / luisa::sqrt(luisa::dot(n1, n1) * luisa::dot(n2, n2));
			float		 angle = luisa::acos(luisa::max(-1.0f, luisa::min(1.0f, dot)));
			float		 sign = luisa::sign(luisa::dot(luisa::cross(n2, n1), v1 - v2));
			angle = sign * angle;
			return angle;
		}

		static inline Float face_dihedral_angle(const Float3& v0, const Float3& v1, const Float3& v2, const Float3& v3)
		{
			const Float3 n1 = luisa::compute::cross(v1 - v0, v2 - v0);
			const Float3 n2 = luisa::compute::cross(v2 - v3, v1 - v3);
			Float		 dot = luisa::compute::dot(n1, n2)
				/ luisa::compute::sqrt(luisa::compute::dot(n1, n1) * luisa::compute::dot(n2, n2));
			Float angle = luisa::compute::acos(luisa::compute::max(-1.0f, luisa::compute::min(1.0f, dot)));
			Float sign = luisa::compute::sign(luisa::compute::dot(luisa::compute::cross(n2, n1), v1 - v2));
			angle = sign * angle;
			return angle;
		}
	} // namespace

	inline float compute_d_theta_d_x(const float3& x2, const float3& x1, const float3& x0, const float3& x3, float3 gradient[4])
	{
		const auto angle = face_dihedral_angle(x0, x1, x2, x3);
		const auto angle_grad = face_dihedral_angle_grad(x0, x1, x2, x3);
		gradient[2] = angle_grad[0];
		gradient[1] = angle_grad[1];
		gradient[0] = angle_grad[2];
		gradient[3] = angle_grad[3];
		return angle;
	}

	inline Float compute_d_theta_d_x(const Float3& x2, const Float3& x1, const Float3& x0, const Float3& x3, Float3 gradient[4])
	{
		const auto angle = face_dihedral_angle(x0, x1, x2, x3);
		const auto angle_grad = face_dihedral_angle_grad(x0, x1, x2, x3);
		gradient[2] = angle_grad[0];
		gradient[1] = angle_grad[1];
		gradient[0] = angle_grad[2];
		gradient[3] = angle_grad[3];
		return angle;
	}

	inline float compute_theta(const float3& x2, const float3& x1, const float3& x0, const float3& x3)
	{
		return face_dihedral_angle(x0, x1, x2, x3);
	}

	inline Float compute_theta(const Float3& x2, const Float3& x1, const Float3& x0, const Float3& x3)
	{
		return face_dihedral_angle(x0, x1, x2, x3);
	}

	template <typename ScalarT, typename Vec3T>
	[[nodiscard]] inline ScalarT compute_energy(
		const Vec3T&  x0,
		const Vec3T&  x1,
		const Vec3T&  x2,
		const Vec3T&  x3,
		const ScalarT rest_angle,
		const ScalarT stiffness)
	{
		Vec3T	   dtheta_dx[4] = { Vec3T{}, Vec3T{}, Vec3T{}, Vec3T{} };
		const auto angle = compute_d_theta_d_x(x0, x1, x2, x3, dtheta_dx);
		const auto delta_angle = angle - rest_angle;
		return 0.5f * stiffness * delta_angle * delta_angle;
	}

	template <typename ScalarT, typename Vec3T, typename Mat3T>
	[[nodiscard]] inline auto evaluate(
		const Vec3T&  x0,
		const Vec3T&  x1,
		const Vec3T&  x2,
		const Vec3T&  x3,
		const ScalarT rest_angle,
		const ScalarT stiffness)
	{
		Vec3T	   dtheta_dx[4] = { Vec3T{}, Vec3T{}, Vec3T{}, Vec3T{} };
		const auto angle = compute_d_theta_d_x(x0, x1, x2, x3, dtheta_dx);
		const auto delta_angle = angle - rest_angle;
		const auto g0 = stiffness * delta_angle * dtheta_dx[0];
		const auto h00 = stiffness * outer_product(dtheta_dx[0], dtheta_dx[0]);

		using GradientOutT = std::decay_t<decltype(g0)>;
		using HessianOutT = std::decay_t<decltype(h00)>;
		EnergyEvalResult<4, 16, GradientOutT, HessianOutT> out{};

		for (int ii = 0; ii < 4; ii++)
		{
			out.gradients[ii] = stiffness * delta_angle * dtheta_dx[ii];
		}
		for (int ii = 0; ii < 4; ii++)
		{
			for (int jj = 0; jj < 4; jj++)
			{
				out.hessians[ii * 4 + jj] = stiffness * outer_product(dtheta_dx[ii], dtheta_dx[jj]);
			}
		}
		return out;
	}

} // namespace lcs::detail::bending_energy
