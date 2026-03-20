#include "stretch_face_energy.h"
#include "Energies/detail/stretch_face_energy.hpp"
#include "SimulationCore/base_mesh.h"
#include "Utils/cpu_parallel.h"
#include "Utils/reduce_helper.h"

using namespace luisa::compute;

namespace lcs
{
	StretchFaceEnergy::StretchFaceEnergy(BufferView<float> sa_system_energy) noexcept
		: _sa_system_energy(sa_system_energy)
	{
	}

	void StretchFaceEnergy::compile(AsyncCompiler& compiler)
	{
		luisa::compute::ShaderOption default_option = compiler.default_option();
		compiler.compile<1>(
			_shader,
			[sa_system_energy = _sa_system_energy](Var<Constitutions::StretchFace<luisa::compute::Buffer>> constraint,
				Var<BufferView<float3>>																	   sa_x)
			{
				auto& sa_faces = constraint.constraint_indices;
				auto& sa_stretch_faces_rest_area = constraint.sa_stretch_faces_rest_area;
				auto& sa_stretch_faces_Dm_inv = constraint.sa_stretch_faces_Dm_inv;
				auto& sa_stretch_faces_mu_lambda = constraint.sa_stretch_faces_mu_lambda;

				const Uint fid = dispatch_id().x;
				Float	   energy = 0.0f;
				{
					const Uint3 face = sa_faces->read(fid);
					Float3		vert_pos[3] = { sa_x->read(face[0]), sa_x->read(face[1]), sa_x->read(face[2]) };

					Float2x2 Dm_inv = sa_stretch_faces_Dm_inv->read(fid);
					Float	 area = sa_stretch_faces_rest_area->read(fid);

					Float2 mu_lambda = sa_stretch_faces_mu_lambda->read(fid);
					Float  mu_cloth = mu_lambda[0];
					Float  lambda_cloth = mu_lambda[1];
					// lambda_cloth = 0.0f;

					const detail::stretch_face_energy::Input<Float3, Float2x2, Float> input{
						.x0 = vert_pos[0],
						.x1 = vert_pos[1],
						.x2 = vert_pos[2],
						.dm_inv = Dm_inv,
						.mu = mu_cloth,
						.lambda = lambda_cloth,
						.area = area
					};
					energy = detail::stretch_face_energy::compute_energy(input);
				};

				energy = ParallelIntrinsic::block_intrinsic_reduce(energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
				$if(fid % 256 == 0)
				{
					sa_system_energy->atomic(offset_stretch_face).fetch_add(energy);
				};
			},
			default_option);

		// gradient/hessian evaluate shader
		compiler.compile<1>(
			_eval_shader,
			[](Var<Constitutions::StretchFace<luisa::compute::Buffer>> constraint, Var<BufferView<float3>> sa_x)
			{
				auto& sa_faces = constraint.constraint_indices;
				auto& sa_stretch_faces_Dm_inv = constraint.sa_stretch_faces_Dm_inv;
				auto& sa_stretch_faces_rest_area = constraint.sa_stretch_faces_rest_area;
				auto& sa_stretch_faces_mu_lambda = constraint.sa_stretch_faces_mu_lambda;
				auto& output_gradient_ptr = constraint.constraint_gradients;
				auto& output_hessian_ptr = constraint.constraint_hessians;

				const UInt	fid = dispatch_id().x;
				const UInt3 face = sa_faces->read(fid);

				Float3 vert_pos[3] = { sa_x->read(face[0]), sa_x->read(face[1]), sa_x->read(face[2]) };

				Float2x2 Dm_inv = sa_stretch_faces_Dm_inv->read(fid);
				Float	 area = sa_stretch_faces_rest_area->read(fid);

				Float2 mu_lambda = sa_stretch_faces_mu_lambda->read(fid);
				Float  mu_cloth = mu_lambda[0];
				Float  lambda_cloth = mu_lambda[1];
				// lambda_cloth = 0.0f;

				const detail::stretch_face_energy::Input<Float3, Float2x2, Float> input{
					.x0 = vert_pos[0],
					.x1 = vert_pos[1],
					.x2 = vert_pos[2],
					.dm_inv = Dm_inv,
					.mu = mu_cloth,
					.lambda = lambda_cloth,
					.area = area
				};
				auto eval = detail::stretch_face_energy::evaluate(input);

				// Output
				{
					output_gradient_ptr->write(fid * 3 + 0, eval.gradients[0]);
					output_gradient_ptr->write(fid * 3 + 1, eval.gradients[1]);
					output_gradient_ptr->write(fid * 3 + 2, eval.gradients[2]);
				}
				{
					for (uint i = 0; i < 9; i++)
					{
						output_hessian_ptr->write(fid * 9 + i, eval.hessians[i]);
					}
				}
			},
			default_option);
	}

	void StretchFaceEnergy::device_compute_energy(luisa::compute::Stream& stream)
	{
	}

	void StretchFaceEnergy::device_compute_energy(luisa::compute::Stream& stream,
		const Constitutions::StretchFace<luisa::compute::Buffer>&		  constraint,
		const luisa::compute::Buffer<float3>&							  sa_x,
		size_t															  dispatch_count)
	{
		stream << _shader(constraint, sa_x).dispatch(dispatch_count);
	}

	void StretchFaceEnergy::device_evaluate(luisa::compute::Stream& stream,
		const Constitutions::StretchFace<luisa::compute::Buffer>&	constraint,
		const luisa::compute::Buffer<float3>&						sa_x,
		size_t														dispatch_count)
	{
		stream << _eval_shader(constraint, sa_x.view()).dispatch(dispatch_count);
	}

	double StretchFaceEnergy::host_evaluate(const std::vector<float>& host_energy)
	{
		return host_energy[offset_stretch_face];
	}

	void StretchFaceEnergy::host_evaluate(lcs::SimulationData<std::vector>& host_sim_data,
		lcs::MeshData<std::vector>&											host_mesh_data)
	{
		auto& stretch_faces = host_sim_data.get_stretch_face_data();

		CpuParallel::parallel_for(
			0,
			stretch_faces.get_num_indices(),
			[sa_x = std::span(host_sim_data.sa_x),
				sa_faces = std::span(stretch_faces.constraint_indices),
				sa_stretch_faces_rest_area = std::span(stretch_faces.sa_stretch_faces_rest_area),
				sa_stretch_faces_Dm_inv = std::span(stretch_faces.sa_stretch_faces_Dm_inv),
				sa_stretch_faces_mu_lambda = std::span(stretch_faces.sa_stretch_faces_mu_lambda),
				output_gradient_ptr = std::span(stretch_faces.constraint_gradients),
				output_hessian_ptr = std::span(stretch_faces.constraint_hessians)](const uint fid)
			{
				uint3 face = sa_faces[fid];

				float3	 vert_pos[3] = { sa_x[face[0]], sa_x[face[1]], sa_x[face[2]] };
				float2x2 Dm_inv = sa_stretch_faces_Dm_inv[fid];
				float	 area = sa_stretch_faces_rest_area[fid];

				auto [mu_cloth, lambda_cloth] = sa_stretch_faces_mu_lambda[fid];
				// lambda_cloth = 0.0f;

				// LUISA_INFO("BW98 Info: Fid = {}, Face = {}, lambda = {}, mu = {}", fid, face, lambda_cloth, mu_cloth);
				const detail::stretch_face_energy::Input<float3, float2x2, float> input{
					.x0 = vert_pos[0],
					.x1 = vert_pos[1],
					.x2 = vert_pos[2],
					.dm_inv = Dm_inv,
					.mu = mu_cloth,
					.lambda = lambda_cloth,
					.area = area
				};
				auto eval = detail::stretch_face_energy::evaluate(input);

				output_gradient_ptr[fid * 3 + 0] = eval.gradients[0];
				output_gradient_ptr[fid * 3 + 1] = eval.gradients[1];
				output_gradient_ptr[fid * 3 + 2] = eval.gradients[2];

				for (uint i = 0; i < 9; i++)
				{
					output_hessian_ptr[fid * 9 + i] = eval.hessians[i];
				}
			});
	}

} // namespace lcs
