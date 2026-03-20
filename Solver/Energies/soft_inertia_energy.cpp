#include "soft_inertia_energy.h"
#include "Energies/detail/soft_inertia_energy.hpp"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/scene_params.h"
#include "Utils/cpu_parallel.h"
#include "Utils/reduce_helper.h"

using namespace luisa::compute;

namespace lcs
{
	SoftInertiaEnergy::SoftInertiaEnergy(BufferView<float3> sa_q_tilde_view, BufferView<float> sa_system_energy_view) noexcept
		: _sa_q_tilde_view(sa_q_tilde_view)
		, _sa_system_energy_view(sa_system_energy_view)
	{
	}

	void SoftInertiaEnergy::compile(AsyncCompiler& compiler)
	{
		luisa::compute::ShaderOption default_option = compiler.default_option();
		compiler.compile<1>(
			_shader,
			[sa_q_tilde = _sa_q_tilde_view, sa_system_energy = _sa_system_energy_view](
				Var<Constitutions::SoftInertia<luisa::compute::Buffer>> constraint, Var<BufferView<float3>> sa_q, Float substep_dt)
			{
				auto& soft_inertia_indices = constraint.constraint_indices;
				auto& sa_vert_mass = constraint.sa_soft_vert_mass;
				auto& sa_stiffness_dirichlet = constraint.sa_stiffness_dirichlet;

				const Uint index = dispatch_id().x;
				const Uint vid = soft_inertia_indices.read(index);

				Float energy = 0.0f;
				{
					const Float3 x_new = sa_q->read(vid);
					const Float3 x_tilde = sa_q_tilde->read(vid);
					const Float	 mass = sa_vert_mass->read(vid);
					const Float	 stiffness_dirichlet = sa_stiffness_dirichlet->read(vid);
					const Float	 inv_h2 = 1.0f / (substep_dt * substep_dt);

					const detail::soft_inertia_energy::Input<Float, Float3> input{
						.x_new = x_new,
						.x_tilde = x_tilde,
						.mass = mass,
						.inv_h2 = inv_h2,
						.stiffness_dirichlet = stiffness_dirichlet
					};
					energy = detail::soft_inertia_energy::compute_energy(input);
				};

				energy = ParallelIntrinsic::block_intrinsic_reduce(energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
				$if(vid % 256 == 0)
				{
					sa_system_energy->atomic(offset_inertia).fetch_add(energy);
				};
			},
			default_option);

		// gradient/hessian evaluate shader (soft inertia)
		compiler.compile<1>(
			_eval_shader,
			[sa_q_tilde = _sa_q_tilde_view](Var<Constitutions::SoftInertia<luisa::compute::Buffer>> contraint,
				Var<BufferView<float3>>																sa_q,
				Float																				substep_dt)
			{
				auto& sa_vert_mass = contraint.sa_soft_vert_mass;
				auto& sa_stiffness_dirichlet = contraint.sa_stiffness_dirichlet;
				auto& output_gradient = contraint.constraint_gradients;
				auto& output_hessian = contraint.constraint_hessians;

				const UInt	vid = dispatch_id().x;
				const Float h = substep_dt;
				const Float h_2_inv = 1.0f / (h * h);

				const Float3 x_k = sa_q->read(vid);
				const Float3 x_tilde = sa_q_tilde->read(vid);
				const Float	 mass = sa_vert_mass->read(vid);
				const Float	 stiffness_dirichlet = sa_stiffness_dirichlet->read(vid);

				const detail::soft_inertia_energy::Input<Float, Float3> input{
					.x_new = x_k,
					.x_tilde = x_tilde,
					.mass = mass,
					.inv_h2 = h_2_inv,
					.stiffness_dirichlet = stiffness_dirichlet
				};
				Float3x3 identity = identity3x3;
				auto	 eval = detail::soft_inertia_energy::evaluate(input, identity);

				output_gradient->write(vid, eval.gradients[0]);
				output_hessian->write(vid, eval.hessians[0]);
			},
			default_option);
	}

	void SoftInertiaEnergy::device_compute_energy(luisa::compute::Stream& stream)
	{
		// This class does not know which constitution to dispatch; caller should use the stored _shader directly.
		// Left intentionally empty — caller will dispatch using shader member exposed via friend or directly if needed.
	}

	void SoftInertiaEnergy::device_compute_energy(luisa::compute::Stream& stream,
		const Constitutions::SoftInertia<luisa::compute::Buffer>&		  constraint,
		const luisa::compute::Buffer<float3>&							  sa_q,
		float															  substep_dt,
		size_t															  dispatch_count)
	{
		stream << _shader(constraint, sa_q.view(), substep_dt).dispatch(dispatch_count);
	}

	void SoftInertiaEnergy::device_evaluate(luisa::compute::Stream& stream,
		const Constitutions::SoftInertia<luisa::compute::Buffer>&	constraint,
		const luisa::compute::Buffer<float3>&						sa_q,
		float														substep_dt,
		size_t														dispatch_count)
	{
		stream << _eval_shader(constraint, sa_q.view(), substep_dt).dispatch(dispatch_count);
	}

	double SoftInertiaEnergy::host_evaluate(const std::vector<float>& host_energy)
	{
		return host_energy[offset_inertia];
	}

	void SoftInertiaEnergy::host_evaluate(lcs::SimulationData<std::vector>& host_sim_data,
		lcs::MeshData<std::vector>&											host_mesh_data)
	{
		auto& inertia_data = host_sim_data.get_soft_inertia_data();
		if (inertia_data.is_valid())
		{
			CpuParallel::parallel_for(0,
				host_sim_data.num_verts_soft,
				[sa_x = std::span(host_sim_data.sa_x),
					sa_x_tilde = std::span(host_sim_data.sa_q_tilde),
					sa_q_property = std::span(host_sim_data.sa_q_property),
					sa_vert_mass = std::span(inertia_data.sa_soft_vert_mass),
					sa_stiffness_dirichlet = std::span(inertia_data.sa_stiffness_dirichlet),
					output_gradient = std::span(inertia_data.constraint_gradients),
					output_hessian = std::span(inertia_data.constraint_hessians),
					substep_dt = get_scene_params().get_substep_dt()](const uint vid)
				{
					const float h = substep_dt;
					const float h_2_inv = 1.f / (h * h);

					const float3 x_k = sa_x[vid];
					const float3 x_tilde = sa_x_tilde[vid];

					// LUISA_INFO("Soft Inertia Energy: vid = {}, dof is fixed = {} , diff = {}", vid, sa_q_property[vid].is_fixed(), x_k - x_tilde);

					const float mass = sa_vert_mass[vid];
					const float stiffness_dirichlet = sa_stiffness_dirichlet[vid];

					const detail::soft_inertia_energy::Input<float, float3> input{
						.x_new = x_k,
						.x_tilde = x_tilde,
						.mass = mass,
						.inv_h2 = h_2_inv,
						.stiffness_dirichlet = stiffness_dirichlet
					};
					auto eval = detail::soft_inertia_energy::evaluate(input, luisa::float3x3::eye(1.0f));
					{
						output_gradient[vid] = eval.gradients[0];
						output_hessian[vid] = eval.hessians[0];
					}
				});
		}
	}

} // namespace lcs
