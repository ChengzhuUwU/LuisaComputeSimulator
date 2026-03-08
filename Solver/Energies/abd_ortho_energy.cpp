#include "abd_ortho_energy.h"
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
		luisa::compute::ShaderOption default_option = { .enable_debug_info = false };
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
					for (uint ii = 0; ii < 3; ii++)
					{
						for (uint jj = 0; jj < 3; jj++)
						{
							Float term = dot(A[ii], A[jj]) - (ii == jj ? 1.0f : 0.0f);
							energy += term * term;
						}
					}
					Float stiffness_ortho = abd_kappa->read(body_idx);
					Float volume = abd_volume->read(body_idx);
					energy *= stiffness_ortho * volume;
				};

				energy = ParallelIntrinsic::block_intrinsic_reduce(
					body_idx, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
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

				Float3	 ortho_gradient[3] = { Zero3 };
				Float3x3 ortho_hessian[6] = { Zero3x3 };

				const Uint3 indices = abd_indices->read(body_idx);

				Float3x3 A = make_float3x3(sa_q->read(indices[0]), sa_q->read(indices[1]), sa_q->read(indices[2]));

				const Float kappa = abd_kappa->read(body_idx);
				const Float V = abd_volume->read(body_idx);

				Float stiff = kappa * V;
				for (uint ii = 0; ii < 3; ii++)
				{
					Float3 grad = (-1.0f) * A[ii];
					for (uint jj = 0; jj < 3; jj++)
					{
						grad += dot_vec(A[ii], A[jj]) * A[jj];
					}
					ortho_gradient[ii] += 4.0f * stiff * grad;
				}
				uint idx = 0;
				for (uint ii = 0; ii < 3; ii++)
				{
					for (uint jj = ii; jj < 3; jj++)
					{
						Float3x3 hessian = Zero3x3;
						if (ii == jj)
						{
							Float3x3 qiqiT = outer_product(A[ii], A[ii]);
							Float3x3 qiTqi = (dot_vec(A[ii], A[ii]) - 1.0f) * Identity3x3;
							hessian = qiqiT + qiTqi;
							for (uint kk = 0; kk < 3; kk++)
							{
								hessian = hessian + outer_product(A[kk], A[kk]);
							}
						}
						else
						{
							hessian = outer_product(A[jj], A[ii]) + dot_vec(A[ii], A[jj]) * Identity3x3;
						}
						ortho_hessian[idx] = ortho_hessian[idx] + 4.0f * stiff * hessian;
						idx += 1;
					}
				}

				auto write_to_grad = [&](const uint i)
				{ abd_gradients->write(3 * body_idx + i, ortho_gradient[i]); };
				auto read_hess_block = [&](const uint ii, const uint jj) -> Float3x3
				{
					if (ii == 0 && jj == 0)
						return ortho_hessian[0];
					if (ii == 0 && jj == 1)
						return ortho_hessian[1];
					if (ii == 0 && jj == 2)
						return ortho_hessian[2];
					if (ii == 1 && jj == 0)
						return transpose(ortho_hessian[1]);
					if (ii == 1 && jj == 1)
						return ortho_hessian[3];
					if (ii == 1 && jj == 2)
						return ortho_hessian[4];
					if (ii == 2 && jj == 0)
						return transpose(ortho_hessian[2]);
					if (ii == 2 && jj == 1)
						return transpose(ortho_hessian[4]);
					return ortho_hessian[5];
				};

				write_to_grad(0);
				write_to_grad(1);
				write_to_grad(2);

				for (uint ii = 0; ii < 3; ii++)
				{
					for (uint jj = 0; jj < 3; jj++)
					{
						abd_hessians->write(9 * body_idx + ii * 3 + jj, read_hess_block(ii, jj));
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
					float3	 ortho_gradient[3] = { Zero3 };
					float3x3 ortho_hessian[6] = { Zero3x3 };

					const float substep_dt = get_scene_params().get_substep_dt();
					const float h = substep_dt;
					const float h_2_inv = 1.f / (h * h);

					const uint3 indices = abd_ortho_indices[body_idx];
					float3x3	A = luisa::make_float3x3(abd_q[indices[0]], abd_q[indices[1]], abd_q[indices[2]]);

					const float kappa = sa_affine_bodies_kappa[body_idx];
					const float V = sa_affine_bodies_volume[body_idx];

					float stiff = kappa * V;
					for (uint ii = 0; ii < 3; ii++)
					{
						float3 grad = (-1.0f) * A[ii];
						for (uint jj = 0; jj < 3; jj++)
						{
							grad += dot_vec(A[ii], A[jj]) * A[jj];
						}
						ortho_gradient[ii] += 4.0f * stiff * grad;
					}
					uint idx = 0;
					for (uint ii = 0; ii < 3; ii++)
					{
						for (uint jj = ii; jj < 3; jj++)
						{
							float3x3 hessian = Zero3x3;
							if (ii == jj)
							{
								float3x3 qiqiT = outer_product(A[ii], A[ii]);
								float3x3 qiTqi = (dot_vec(A[ii], A[ii]) - 1.0f) * Identity3x3;
								hessian = qiqiT + qiTqi;
								for (uint kk = 0; kk < 3; kk++)
								{
									hessian = hessian + outer_product(A[kk], A[kk]);
								}
							}
							else
							{
								hessian = outer_product(A[jj], A[ii]) + dot_vec(A[ii], A[jj]) * Identity3x3;
							}
							ortho_hessian[idx] = ortho_hessian[idx] + 4.0f * stiff * hessian;
							idx += 1;
						}
					}

					auto* body_grad_ptr = &abd_gradients[3 * body_idx];
					body_grad_ptr[0] = ortho_gradient[0];
					body_grad_ptr[1] = ortho_gradient[1];
					body_grad_ptr[2] = ortho_gradient[2];

					auto read_hess_block = [&](const uint ii, const uint jj) -> float3x3
					{
						if (ii == 0 && jj == 0)
							return ortho_hessian[0];
						if (ii == 0 && jj == 1)
							return ortho_hessian[1];
						if (ii == 0 && jj == 2)
							return ortho_hessian[2];
						if (ii == 1 && jj == 0)
							return transpose(ortho_hessian[1]);
						if (ii == 1 && jj == 1)
							return ortho_hessian[3];
						if (ii == 1 && jj == 2)
							return ortho_hessian[4];
						if (ii == 2 && jj == 0)
							return transpose(ortho_hessian[2]);
						if (ii == 2 && jj == 1)
							return transpose(ortho_hessian[4]);
						return ortho_hessian[5];
					};

					for (uint ii = 0; ii < 3; ii++)
					{
						for (uint jj = 0; jj < 3; jj++)
						{
							abd_hessians[9 * body_idx + ii * 3 + jj] = read_hess_block(ii, jj);
						}
					}
				},
				32);
		}
	}

} // namespace lcs
