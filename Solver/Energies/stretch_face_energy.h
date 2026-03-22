#pragma once

#include "Energies/energy.h"
#include "Energies/energy_offsets.h"
#include "Energies/detail/fem_utils.h"
#include "Energies/detail/fem_BW98_cloth_energy.hpp"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/simulation_data.h"
#include <luisa/dsl/builtin.h>

namespace lcs
{
	namespace StretchEnergy
	{
		namespace detail
		{
			using lcs::detail::fem_BW98_cloth_energy::shear_energy;
			using lcs::detail::fem_BW98_cloth_energy::shear_gradient;
			using lcs::detail::fem_BW98_cloth_energy::shear_hessian;
			using lcs::detail::fem_BW98_cloth_energy::stretch_energy;
			using lcs::detail::fem_BW98_cloth_energy::stretch_gradient;
			using lcs::detail::fem_BW98_cloth_energy::stretch_hessian;
		} // namespace detail

		inline void compute_gradient_hessian(const float3& x0,
			const float3&								   x1,
			const float3&								   x2,
			const float2x2&								   Dm,
			const float									   mu,
			const float									   lambda,
			const float									   area,
			float3x3&									   dedx,
			float9x9&									   d2edx2)
		{
			const lcs::detail::fem_BW98_cloth_energy::Input<float3, float2x2, float> input{
				.x0 = x0,
				.x1 = x1,
				.x2 = x2,
				.dm_inv = Dm,
				.mu = mu,
				.lambda = lambda,
				.area = area
			};
			auto eval = lcs::detail::fem_BW98_cloth_energy::evaluate(input);
			dedx = make_float3x3(eval.gradients[0], eval.gradients[1], eval.gradients[2]);
			d2edx2.set_zero();
			d2edx2.block(0, 0) = eval.hessians[0];
			d2edx2.block(1, 0) = eval.hessians[1];
			d2edx2.block(2, 0) = eval.hessians[2];
			d2edx2.block(0, 1) = eval.hessians[3];
			d2edx2.block(1, 1) = eval.hessians[4];
			d2edx2.block(2, 1) = eval.hessians[5];
			d2edx2.block(0, 2) = eval.hessians[6];
			d2edx2.block(1, 2) = eval.hessians[7];
			d2edx2.block(2, 2) = eval.hessians[8];
		}

		inline void compute_gradient_hessian(const Var<float3>& x0,
			const Var<float3>&									x1,
			const Var<float3>&									x2,
			const Var<float2x2>&								Dm,
			const Var<float>									mu,
			const Var<float>									lambda,
			const Var<float>									area,
			Var<float3x3>&										dedx,
			Var<float9x9>&										d2edx2)
		{
			const lcs::detail::fem_BW98_cloth_energy::Input<Var<float3>, Var<float2x2>, Var<float>> input{
				.x0 = x0,
				.x1 = x1,
				.x2 = x2,
				.dm_inv = Dm,
				.mu = mu,
				.lambda = lambda,
				.area = area
			};
			auto eval = lcs::detail::fem_BW98_cloth_energy::evaluate(input);
			dedx = make_float3x3(eval.gradients[0], eval.gradients[1], eval.gradients[2]);
			d2edx2->set_zero();
			d2edx2->block(0, 0) = eval.hessians[0];
			d2edx2->block(1, 0) = eval.hessians[1];
			d2edx2->block(2, 0) = eval.hessians[2];
			d2edx2->block(0, 1) = eval.hessians[3];
			d2edx2->block(1, 1) = eval.hessians[4];
			d2edx2->block(2, 1) = eval.hessians[5];
			d2edx2->block(0, 2) = eval.hessians[6];
			d2edx2->block(1, 2) = eval.hessians[7];
			d2edx2->block(2, 2) = eval.hessians[8];
		}

		inline float compute_energy(const float3& x0,
			const float3&						  x1,
			const float3&						  x2,
			const float2x2&						  Dm,
			const float							  mu,
			const float							  lambda,
			const float							  area)
		{
			const lcs::detail::fem_BW98_cloth_energy::Input<float3, float2x2, float> input{
				.x0 = x0,
				.x1 = x1,
				.x2 = x2,
				.dm_inv = Dm,
				.mu = mu,
				.lambda = lambda,
				.area = area
			};
			return lcs::detail::fem_BW98_cloth_energy::compute_energy(input);
		}

		inline Var<float> compute_energy(const Var<float3>& x0,
			const Var<float3>&								x1,
			const Var<float3>&								x2,
			const Var<float2x2>&							Dm,
			const Var<float>								mu,
			const Var<float>								lambda,
			const Var<float>								area)
		{
			const lcs::detail::fem_BW98_cloth_energy::Input<Var<float3>, Var<float2x2>, Var<float>> input{
				.x0 = x0,
				.x1 = x1,
				.x2 = x2,
				.dm_inv = Dm,
				.mu = mu,
				.lambda = lambda,
				.area = area
			};
			return lcs::detail::fem_BW98_cloth_energy::compute_energy(input);
		}

	} // namespace StretchEnergy

	class StretchFaceEnergy : public Energy
	{
	public:
		StretchFaceEnergy(luisa::compute::BufferView<float> sa_system_energy) noexcept;
		void   compile(AsyncCompiler& compiler) override;
		void   device_compute_energy(luisa::compute::Stream& stream) override;
		void   device_compute_energy(luisa::compute::Stream&			stream,
			  const Constitutions::StretchFace<luisa::compute::Buffer>& constraint,
			  const luisa::compute::Buffer<float3>&						sa_x,
			  size_t													dispatch_count);
		void   device_evaluate(luisa::compute::Stream&					stream,
			  const Constitutions::StretchFace<luisa::compute::Buffer>& constraint,
			  const luisa::compute::Buffer<float3>&						sa_x,
			  size_t													dispatch_count);
		double host_evaluate(const std::vector<float>& host_energy) override;
		// Host-side evaluation for stretch faces
		void host_evaluate(lcs::SimulationData<std::vector>& host_sim_data, lcs::MeshData<std::vector>& host_mesh_data);

	private:
		luisa::compute::BufferView<float>																				  _sa_system_energy;
		luisa::compute::Shader<1, Constitutions::StretchFace<luisa::compute::Buffer>, luisa::compute::BufferView<float3>> _shader;
		luisa::compute::Shader<1, Constitutions::StretchFace<luisa::compute::Buffer>, luisa::compute::BufferView<float3>> _eval_shader;
	};

} // namespace lcs
