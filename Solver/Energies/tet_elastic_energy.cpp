#include "tet_elastic_energy.h"
#include "SimulationCore/base_mesh.h"
#include "Utils/cpu_parallel.h"
#include "Utils/reduce_helper.h"

using namespace luisa::compute;

namespace lcs
{
	TetElasticEnergy::TetElasticEnergy(BufferView<float> sa_system_energy) noexcept
		: _sa_system_energy(sa_system_energy)
	{
	}

	void TetElasticEnergy::compile(AsyncCompiler& compiler)
	{
		luisa::compute::ShaderOption default_option = { .enable_debug_info = false };

		// ---- Energy shader ----
		compiler.compile<1>(
			_shader,
			[sa_system_energy = _sa_system_energy](
				Var<Constitutions::StressTet<luisa::compute::Buffer>> constraint,
				Var<BufferView<float3>>								  sa_x)
			{
				auto& sa_tets = constraint.constraint_indices;
				auto& sa_Dm_inv = constraint.sa_stress_tets_Dm_inv;
				auto& sa_rest_volume = constraint.sa_stress_tets_rest_volume;
				auto& sa_mu_lambda = constraint.sa_stress_tets_mu_lambda;

				const UInt	tid = dispatch_id().x;
				const UInt4 tet = sa_tets->read(tid);

				Float3 x0 = sa_x->read(tet[0]);
				Float3 x1 = sa_x->read(tet[1]);
				Float3 x2 = sa_x->read(tet[2]);
				Float3 x3 = sa_x->read(tet[3]);

				Float3x3 Dm_inv = sa_Dm_inv->read(tid);
				Float	 volume = sa_rest_volume->read(tid);
				Float2	 mu_lambda = sa_mu_lambda->read(tid);
				Float	 mu = mu_lambda[0];
				Float	 lam = mu_lambda[1];

				Float energy = TetElasticEnergyUtils::compute_energy_gpu(x0, x1, x2, x3, Dm_inv, mu, lam, volume);

				energy = ParallelIntrinsic::block_intrinsic_reduce(
					tid, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
				$if(tid % 256 == 0)
				{
					sa_system_energy->atomic(offset_tet_elastic).fetch_add(energy);
				};
			},
			default_option);

		// ---- Gradient/Hessian (evaluate) shader ----
		compiler.compile<1>(
			_eval_shader,
			[](Var<Constitutions::StressTet<luisa::compute::Buffer>> constraint,
				Var<BufferView<float3>>								 sa_x)
			{
				auto& sa_tets = constraint.constraint_indices;
				auto& sa_Dm_inv = constraint.sa_stress_tets_Dm_inv;
				auto& sa_rest_volume = constraint.sa_stress_tets_rest_volume;
				auto& sa_mu_lambda = constraint.sa_stress_tets_mu_lambda;
				auto& out_gradient = constraint.constraint_gradients; // Buffer<float3>, size = num_tets*4
				auto& out_hessian = constraint.constraint_hessians;	  // Buffer<float3x3>, size = num_tets*16

				const UInt	tid = dispatch_id().x;
				const UInt4 tet = sa_tets->read(tid);

				Float3 x0 = sa_x->read(tet[0]);
				Float3 x1 = sa_x->read(tet[1]);
				Float3 x2 = sa_x->read(tet[2]);
				Float3 x3 = sa_x->read(tet[3]);

				Float3x3 Dm_inv = sa_Dm_inv->read(tid);
				Float	 volume = sa_rest_volume->read(tid);
				Float2	 mu_lambda = sa_mu_lambda->read(tid);
				Float	 mu = mu_lambda[0];
				Float	 lam = mu_lambda[1];

				Float3	 gradient[4];
				Float3x3 hessian[16];
				TetElasticEnergyUtils::compute_gradient_hessian_gpu(
					x0, x1, x2, x3, Dm_inv, mu, lam, volume, gradient, hessian);

				// Write gradients: 4 entries per tet
				for (uint v = 0; v < 4; v++)
					out_gradient->write(tid * 4 + v, gradient[v]);

				// Write hessians: 16 blocks per tet
				for (uint i = 0; i < 4; i++)
					for (uint j = 0; j < 4; j++)
						out_hessian->write(tid * 16 + i * 4 + j, hessian[i * 4 + j]);
			},
			default_option);
	}

	void TetElasticEnergy::device_compute_energy(luisa::compute::Stream& stream)
	{
	}

	void TetElasticEnergy::device_compute_energy(luisa::compute::Stream& stream,
		const Constitutions::StressTet<luisa::compute::Buffer>&			 constraint,
		const luisa::compute::Buffer<float3>&							 sa_x,
		size_t															 dispatch_count)
	{
		stream << _shader(constraint, sa_x).dispatch(dispatch_count);
	}

	void TetElasticEnergy::device_evaluate(luisa::compute::Stream& stream,
		const Constitutions::StressTet<luisa::compute::Buffer>&	   constraint,
		const luisa::compute::Buffer<float3>&					   sa_x,
		size_t													   dispatch_count)
	{
		stream << _eval_shader(constraint, sa_x.view()).dispatch(dispatch_count);
	}

	double TetElasticEnergy::host_evaluate(const std::vector<float>& host_energy)
	{
		return host_energy[offset_tet_elastic];
	}

	void TetElasticEnergy::host_evaluate(lcs::SimulationData<std::vector>& host_sim_data,
		lcs::MeshData<std::vector>&										   host_mesh_data)
	{
		auto& stress_tets = host_sim_data.get_stress_tet_data();
		if (!stress_tets.is_valid())
			return;

		const uint num_tets = stress_tets.get_num_indices();
		CpuParallel::parallel_for(
			0,
			num_tets,
			[sa_x = std::span(host_sim_data.sa_x),
				sa_tets = std::span(stress_tets.constraint_indices),
				sa_Dm_inv = std::span(stress_tets.sa_stress_tets_Dm_inv),
				sa_volume = std::span(stress_tets.sa_stress_tets_rest_volume),
				sa_mu_lambda = std::span(stress_tets.sa_stress_tets_mu_lambda),
				out_gradient = std::span(stress_tets.constraint_gradients),
				out_hessian = std::span(stress_tets.constraint_hessians)](const uint tid)
			{
				const uint4			   tet = sa_tets[tid];
				const float3&		   x0 = sa_x[tet[0]];
				const float3&		   x1 = sa_x[tet[1]];
				const float3&		   x2 = sa_x[tet[2]];
				const float3&		   x3 = sa_x[tet[3]];
				const luisa::float3x3& Dm_inv = sa_Dm_inv[tid];
				const float			   volume = sa_volume[tid];
				const auto [mu, lam] = sa_mu_lambda[tid];

				luisa::float3	gradient[4];
				luisa::float3x3 hessian[16];
				TetElasticEnergyUtils::compute_gradient_hessian(
					x0, x1, x2, x3, Dm_inv, mu, lam, volume, gradient, hessian);

				for (uint v = 0; v < 4; v++)
					out_gradient[tid * 4 + v] = gradient[v];
				for (uint i = 0; i < 16; i++)
					out_hessian[tid * 16 + i] = hessian[i];
			});
	}

} // namespace lcs
