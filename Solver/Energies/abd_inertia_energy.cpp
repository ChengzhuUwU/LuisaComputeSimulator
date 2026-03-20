#include "abd_inertia_energy.h"
#include "Energies/detail/abd_inertia_energy.hpp"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/scene_params.h"
#include "Utils/cpu_parallel.h"
#include "Utils/reduce_helper.h"

using namespace luisa::compute;

namespace lcs
{
	AbdInertiaEnergy::AbdInertiaEnergy(BufferView<float3> sa_q_tilde, BufferView<float> sa_system_energy) noexcept
		: _sa_q_tilde(sa_q_tilde)
		, _sa_system_energy(sa_system_energy)
	{
	}

	void AbdInertiaEnergy::compile(AsyncCompiler& compiler)
	{
		luisa::compute::ShaderOption default_option = compiler.default_option();
		compiler.compile<1>(
			_shader,
			[sa_q_tilde = _sa_q_tilde, sa_system_energy = _sa_system_energy](
				Var<Constitutions::AbdInertia<luisa::compute::Buffer>> constraint, Var<BufferView<float3>> sa_q, Float substep_dt)
			{
				auto& sa_affine_bodies = constraint.constraint_indices;
				auto& sa_vert_mass = constraint.sa_affine_bodies_mass_matrix;
				auto& sa_stiffness_dirichlet = constraint.sa_stiffness_dirichlet;

				const Uint	body_idx = dispatch_id().x;
				const Uint4 affine_body = sa_affine_bodies->read(body_idx);

				Float energy = 0.0f;
				{
					auto   mass_matrix = sa_vert_mass->read(body_idx);
					Float3 delta[4] = {
						sa_q.read(affine_body[0]) - sa_q_tilde->read(affine_body[0]),
						sa_q.read(affine_body[1]) - sa_q_tilde->read(affine_body[1]),
						sa_q.read(affine_body[2]) - sa_q_tilde->read(affine_body[2]),
						sa_q.read(affine_body[3]) - sa_q_tilde->read(affine_body[3]),
					};
					const Float scaled_stiffness = sa_stiffness_dirichlet->read(body_idx) / (substep_dt * substep_dt);
					energy = detail::abd_inertia_energy::compute_energy(delta, mass_matrix, scaled_stiffness);
				};

				energy = ParallelIntrinsic::block_intrinsic_reduce(
					energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
				$if(body_idx % 256 == 0)
				{
					sa_system_energy->atomic(offset_abd_inertia).fetch_add(energy);
				};
			},
			default_option);

		// gradient/hessian evaluate shader for ABD inertia
		compiler.compile<1>(
			_eval_shader,
			[sa_q_tilde = _sa_q_tilde](Var<Constitutions::AbdInertia<luisa::compute::Buffer>> constraint,
				Var<BufferView<float3>>														  sa_q,
				Float																		  substep_dt)
			{
				auto& sa_affine_bodies = constraint.constraint_indices;
				auto& sa_vert_mass = constraint.sa_affine_bodies_mass_matrix;
				auto& sa_stiffness_dirichlet = constraint.sa_stiffness_dirichlet;

				const UInt	body_idx = dispatch_id().x;
				const UInt4 affine_body = sa_affine_bodies->read(body_idx);

				const Float h = substep_dt;
				const Float h_2_inv = 1.0f / (h * h);

				Float3 delta[4] = { sa_q->read(affine_body[0]) - sa_q_tilde->read(affine_body[0]),
					sa_q->read(affine_body[1]) - sa_q_tilde->read(affine_body[1]),
					sa_q->read(affine_body[2]) - sa_q_tilde->read(affine_body[2]),
					sa_q->read(affine_body[3]) - sa_q_tilde->read(affine_body[3]) };

				Float4x4	mass_matrix = sa_vert_mass->read(body_idx);
				const Float scaled_stiffness = sa_stiffness_dirichlet->read(body_idx) * h_2_inv;
				Float3x3	identity = identity3x3;
				auto		eval = detail::abd_inertia_energy::evaluate(delta, mass_matrix, scaled_stiffness, identity);

				auto& abd_gradients = constraint.constraint_gradients;
				auto& abd_hessians = constraint.constraint_hessians;

				abd_gradients->write(4 * body_idx + 0, eval.gradients[0]);
				abd_gradients->write(4 * body_idx + 1, eval.gradients[1]);
				abd_gradients->write(4 * body_idx + 2, eval.gradients[2]);
				abd_gradients->write(4 * body_idx + 3, eval.gradients[3]);

				for (uint ii = 0; ii < 4; ii++)
				{
					for (uint jj = 0; jj < 4; jj++)
					{
						abd_hessians->write(body_idx * 16 + ii * 4 + jj, eval.hessians[ii * 4 + jj]);
					}
				}
			},
			default_option);
	}

	void AbdInertiaEnergy::device_compute_energy(luisa::compute::Stream& stream)
	{
		// Caller should use the typed overload below to dispatch with the appropriate buffers and counts.
	}

	void AbdInertiaEnergy::device_compute_energy(luisa::compute::Stream& stream,
		const Constitutions::AbdInertia<luisa::compute::Buffer>&		 constraint,
		const luisa::compute::Buffer<float3>&							 sa_q,
		float															 substep_dt,
		size_t															 dispatch_count)
	{
		stream << _shader(constraint, sa_q.view(), substep_dt).dispatch(dispatch_count);
	}

	void AbdInertiaEnergy::device_evaluate(luisa::compute::Stream& stream,
		const Constitutions::AbdInertia<luisa::compute::Buffer>&   constraint,
		const luisa::compute::Buffer<float3>&					   sa_q,
		float													   substep_dt,
		size_t													   dispatch_count)
	{
		stream << _eval_shader(constraint, sa_q.view(), substep_dt).dispatch(dispatch_count);
		// std::vector<float3>	  host_gradients(constraint.constraint_indices.size() * 4);
		// std::vector<float3x3> host_hessians(constraint.constraint_indices.size() * 16);
		// stream << constraint.constraint_gradients.copy_to(host_gradients.data());
		// stream << constraint.constraint_hessians.copy_to(host_hessians.data());
		// stream << synchronize();
		// for (uint i = 0; i < host_hessians.size(); i++)
		// {
		// 	LUISA_INFO("Hessian {} (Body {}, ii = {}, ii = {}) {}", i, i / 16, (i % 16) / 4, i % 4, host_hessians[i]);
		// }
	}

	double AbdInertiaEnergy::host_evaluate(const std::vector<float>& host_energy)
	{
		return host_energy[offset_abd_inertia];
	}

	void AbdInertiaEnergy::host_evaluate(lcs::SimulationData<std::vector>& host_sim_data,
		lcs::MeshData<std::vector>& /*host_mesh_data*/)
	{
		auto& abd_data = host_sim_data.get_abd_inertia_data();

		if (abd_data.is_valid())
		{
			CpuParallel::parallel_for(
				0,
				abd_data.get_num_indices(),
				[abd_gradients = std::span(abd_data.constraint_gradients),
					abd_hessians = std::span(abd_data.constraint_hessians),
					abd_indices = std::span(abd_data.constraint_indices),
					abd_mass_matrix = std::span(abd_data.sa_affine_bodies_mass_matrix),
					abd_stiffness_dirichlet = std::span(abd_data.sa_stiffness_dirichlet),
					abd_q = std::span(host_sim_data.sa_q),
					abd_q_tilde = std::span(host_sim_data.sa_q_tilde)](const uint body_idx)
				{
					const float substep_dt = get_scene_params().get_substep_dt();
					const float h = substep_dt;
					const float h_2_inv = 1.f / (h * h);

					const uint4 indices = abd_indices[body_idx];

					float3		delta_q[4] = { abd_q[indices[0]] - abd_q_tilde[indices[0]],
							 abd_q[indices[1]] - abd_q_tilde[indices[1]],
							 abd_q[indices[2]] - abd_q_tilde[indices[2]],
							 abd_q[indices[3]] - abd_q_tilde[indices[3]] };
					float4x4	mass_matrix = abd_mass_matrix[body_idx];
					const float scaled_stiffness = abd_stiffness_dirichlet[body_idx] * h_2_inv;
					auto		eval = detail::abd_inertia_energy::evaluate(delta_q, mass_matrix, scaled_stiffness, float3x3::eye(1.0f));

					abd_gradients[4 * body_idx + 0] = eval.gradients[0];
					abd_gradients[4 * body_idx + 1] = eval.gradients[1];
					abd_gradients[4 * body_idx + 2] = eval.gradients[2];
					abd_gradients[4 * body_idx + 3] = eval.gradients[3];

					for (uint ii = 0; ii < 4; ii++)
					{
						for (uint jj = 0; jj < 4; jj++)
						{
							abd_hessians[body_idx * 16 + ii * 4 + jj] = eval.hessians[ii * 4 + jj];
						}
					}
				},
				32);
		}
		// for (uint i = 0; i < abd_data.constraint_hessians.size(); i++)
		// {
		// 	LUISA_INFO("Hessian {} (Body {}, ii = {}, ii = {}) {}", i, i / 16, (i % 16) / 4, i % 4, abd_data.constraint_hessians[i]);
		// }
	}

} // namespace lcs
