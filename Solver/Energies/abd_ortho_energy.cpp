#include "abd_ortho_energy.h"
#include "Energies/detail/abd_ortho_energy.hpp"
#include "SimulationCore/scene_params.h"
#include "Utils/cpu_parallel.h"
#include "Utils/reduce_helper.h"

using namespace luisa::compute;

namespace lcs
{
	AbdOrthoEnergy::AbdOrthoEnergy(BufferView<float> sa_system_energy, BufferView<float3> sa_q) noexcept
		: _sa_system_energy(sa_system_energy)
		, _sa_q(sa_q)
	{
	}

	void AbdOrthoEnergy::compile(AsyncCompiler& compiler)
	{
		luisa::compute::ShaderOption default_option = compiler.default_option();
		compiler.compile<1>(
			_shader,
			[sa_system_energy = _sa_system_energy,
				sa_q = _sa_q](Var<Constitutions::AbdOrthogonality<luisa::compute::Buffer>> constraint)
			{
				using namespace luisa::compute;
				auto& abd_ortho_indices = constraint.constraint_indices;
				auto& abd_kappa = constraint.abd_kappa;
				auto& abd_volume = constraint.abd_volume;

				const Uint body_idx = dispatch_id().x;

				const Uint3 indices = abd_ortho_indices->read(body_idx);

				Float energy = 0.0f;
				{
					Float3x3 A;
					A[0] = sa_q->read(indices[0]);
					A[1] = sa_q->read(indices[1]);
					A[2] = sa_q->read(indices[2]);
					const Float stiffness = abd_kappa->read(body_idx) * abd_volume->read(body_idx);
					energy = detail::abd_ortho_energy::compute_energy(A, stiffness);
				};

				energy = ParallelIntrinsic::block_intrinsic_reduce(
					energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
				$if(body_idx % 256 == 0)
				{
					sa_system_energy->atomic(offset_abd_ortho).fetch_add(energy);
				};
			},
			default_option);

		// evaluate orthogonality gradient/hessian shader
		compiler.compile<1>(
			_eval_shader,
			[sa_q = _sa_q](Var<Constitutions::AbdOrthogonality<luisa::compute::Buffer>> constraint)
			{
				auto& abd_indices = constraint.constraint_indices;
				auto& abd_gradients = constraint.constraint_gradients;
				auto& abd_hessians = constraint.constraint_hessians;
				auto& abd_volume = constraint.abd_volume;
				auto& abd_kappa = constraint.abd_kappa;

				const UInt body_idx = dispatch_id().x;

				const Uint3 indices = abd_indices->read(body_idx);

				Float3x3 A = make_float3x3(sa_q->read(indices[0]), sa_q->read(indices[1]), sa_q->read(indices[2]));

				const Float kappa = abd_kappa->read(body_idx);
				const Float V = abd_volume->read(body_idx);
				const Float stiffness = kappa * V;
				auto		eval = detail::abd_ortho_energy::evaluate<Float, Float3, Float3x3>(A, stiffness, make_float3x3(1.0f));

				abd_gradients->write(3 * body_idx + 0, eval.gradients[0]);
				abd_gradients->write(3 * body_idx + 1, eval.gradients[1]);
				abd_gradients->write(3 * body_idx + 2, eval.gradients[2]);

				for (uint ii = 0; ii < 3; ii++)
				{
					for (uint jj = 0; jj < 3; jj++)
					{
						abd_hessians->write(9 * body_idx + ii * 3 + jj, eval.hessians[ii * 3 + jj]);
					}
				}
			},
			default_option);
	}

	void AbdOrthoEnergy::device_compute_energy(luisa::compute::Stream& stream)
	{
		// left empty — use typed overload below
	}

	void AbdOrthoEnergy::device_compute_energy(luisa::compute::Stream& stream,
		const Constitutions::AbdOrthogonality<luisa::compute::Buffer>& constraint,
		const luisa::compute::Buffer<float3>&						   sa_q,
		size_t														   dispatch_count)
	{
		stream << _shader(constraint).dispatch(dispatch_count);
	}

	void AbdOrthoEnergy::device_evaluate(luisa::compute::Stream&	   stream,
		const Constitutions::AbdOrthogonality<luisa::compute::Buffer>& constraint,
		const luisa::compute::Buffer<float3>&						   sa_q,
		size_t														   dispatch_count)
	{
		stream << _eval_shader(constraint).dispatch(dispatch_count);
	}

	double AbdOrthoEnergy::host_evaluate(const std::vector<float>& host_energy)
	{
		return host_energy[offset_abd_ortho];
	}

	void AbdOrthoEnergy::host_evaluate(lcs::SimulationData<std::vector>& host_sim_data,
		lcs::MeshData<std::vector>&										 host_mesh_data)
	{
		auto& abd_data = host_sim_data.get_abd_orthogonality_data();

		if (abd_data.is_valid())
		{
			CpuParallel::parallel_for(
				0,
				abd_data.get_num_indices(),
				[abd_q = std::span(host_sim_data.sa_q),
					abd_gradients = std::span(abd_data.constraint_gradients),
					abd_hessians = std::span(abd_data.constraint_hessians),
					sa_affine_bodies_kappa = std::span(abd_data.abd_kappa),
					sa_affine_bodies_volume = std::span(abd_data.abd_volume),
					abd_ortho_indices = std::span(abd_data.constraint_indices)](const uint body_idx)
				{
					const float substep_dt = get_scene_params().get_substep_dt();
					const float h = substep_dt;
					const float h_2_inv = 1.f / (h * h);

					const uint3 indices = abd_ortho_indices[body_idx];
					float3x3	A = luisa::make_float3x3(abd_q[indices[0]], abd_q[indices[1]], abd_q[indices[2]]);

					const float kappa = sa_affine_bodies_kappa[body_idx];
					const float V = sa_affine_bodies_volume[body_idx];

					const float stiff = kappa * V;
					auto		eval = detail::abd_ortho_energy::evaluate<float, float3, float3x3>(A, stiff, float3x3::eye(1.0f));

					auto* body_grad_ptr = &abd_gradients[3 * body_idx];
					body_grad_ptr[0] = eval.gradients[0];
					body_grad_ptr[1] = eval.gradients[1];
					body_grad_ptr[2] = eval.gradients[2];

					for (uint ii = 0; ii < 3; ii++)
					{
						for (uint jj = 0; jj < 3; jj++)
						{
							abd_hessians[9 * body_idx + ii * 3 + jj] = eval.hessians[ii * 3 + jj];
						}
					}
				},
				32);
		}
	}

} // namespace lcs
