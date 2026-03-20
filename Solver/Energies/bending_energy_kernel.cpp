#include "bending_energy_kernel.h"
#include "Energies/detail/bending_energy.hpp"
#include "SimulationCore/scene_params.h"
#include "Utils/cpu_parallel.h"
#include "Utils/reduce_helper.h"

using namespace luisa::compute;

namespace lcs
{
	BendingEnergy::BendingEnergy(BufferView<float> sa_system_energy) noexcept
		: _sa_system_energy(sa_system_energy)
	{
	}

	void BendingEnergy::compile(AsyncCompiler& compiler)
	{
		luisa::compute::ShaderOption default_option = compiler.default_option();
		compiler.compile<1>(
			_shader,
			[sa_system_energy = _sa_system_energy](Var<Constitutions::BendingEdge<luisa::compute::Buffer>> constraint,
				Var<BufferView<float3>>																	   sa_x,
				Float																					   scaling)
			{
				auto& sa_edges = constraint.constraint_indices;
				auto& sa_bending_edges_rest_angle = constraint.sa_bending_edges_rest_angle;
				auto& sa_bending_edges_rest_area = constraint.sa_bending_edges_rest_area;
				auto& sa_bending_edges_stiffness = constraint.sa_bending_edges_stiffness;

				const Uint eid = dispatch_id().x;
				Float	   energy = 0.0f;
				{
					const Uint4 edge = sa_edges->read(eid);

					Float3 vert_pos[4] = { sa_x.read(edge[0]), sa_x.read(edge[1]), sa_x.read(edge[2]), sa_x.read(edge[3]) };
					Float  rest_angle = sa_bending_edges_rest_angle->read(eid);
					Float  area = sa_bending_edges_rest_area->read(eid);
					Float  stiff = sa_bending_edges_stiffness->read(eid) * scaling * area;
					energy = detail::bending_energy::compute_energy(
						vert_pos[0], vert_pos[1], vert_pos[2], vert_pos[3], rest_angle, stiff);
				};

				energy = ParallelIntrinsic::block_intrinsic_reduce(energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
				$if(eid % 256 == 0)
				{
					sa_system_energy->atomic(offset_bending).fetch_add(energy);
				};
			},
			default_option);

		// evaluate gradient/hessian shader
		compiler.compile<1>(
			_eval_shader,
			[](Var<Constitutions::BendingEdge<luisa::compute::Buffer>> constraint, Var<BufferView<float3>> sa_x, Float scaling)
			{
				auto& sa_edges = constraint.constraint_indices;
				auto& sa_bending_edges_rest_angle = constraint.sa_bending_edges_rest_angle;
				auto& sa_bending_edges_rest_area = constraint.sa_bending_edges_rest_area;
				auto& sa_bending_edges_stiffness = constraint.sa_bending_edges_stiffness;
				auto& output_gradient_ptr = constraint.constraint_gradients;
				auto& output_hessian_ptr = constraint.constraint_hessians;

				const UInt	eid = dispatch_id().x;
				const UInt4 edge = sa_edges->read(eid);

				Float3 vert_pos[4] = {
					sa_x->read(edge[0]),
					sa_x->read(edge[1]),
					sa_x->read(edge[2]),
					sa_x->read(edge[3]),
				};

				const Float rest_angle = sa_bending_edges_rest_angle->read(eid);
				const Float area = sa_bending_edges_rest_area->read(eid);
				const Float stiff = sa_bending_edges_stiffness->read(eid) * scaling * area;

				auto eval = detail::bending_energy::evaluate<Float, Float3, Float3x3>(
					vert_pos[0], vert_pos[1], vert_pos[2], vert_pos[3], rest_angle, stiff);
				{
					output_gradient_ptr->write(eid * 4 + 0, eval.gradients[0]);
					output_gradient_ptr->write(eid * 4 + 1, eval.gradients[1]);
					output_gradient_ptr->write(eid * 4 + 2, eval.gradients[2]);
					output_gradient_ptr->write(eid * 4 + 3, eval.gradients[3]);

					for (uint ii = 0; ii < 4; ii++)
					{
						for (uint jj = 0; jj < 4; jj++)
						{
							output_hessian_ptr->write(eid * 16 + ii * 4 + jj, eval.hessians[ii * 4 + jj]);
						}
					}
				}
			},
			default_option);
	}

	void BendingEnergy::device_compute_energy(luisa::compute::Stream& stream)
	{
	}

	void BendingEnergy::device_compute_energy(luisa::compute::Stream& stream,
		const Constitutions::BendingEdge<luisa::compute::Buffer>&	  constraint,
		const luisa::compute::Buffer<float3>&						  sa_x,
		float														  scaling,
		size_t														  dispatch_count)
	{
		stream << _shader(constraint, sa_x, scaling).dispatch(dispatch_count);
	}

	void BendingEnergy::device_evaluate(luisa::compute::Stream&	  stream,
		const Constitutions::BendingEdge<luisa::compute::Buffer>& constraint,
		const luisa::compute::Buffer<float3>&					  sa_x,
		float													  scaling,
		size_t													  dispatch_count)
	{
		stream << _eval_shader(constraint, sa_x.view(), scaling).dispatch(dispatch_count);
	}

	double BendingEnergy::host_evaluate(const std::vector<float>& host_energy)
	{
		return host_energy[offset_bending];
	}

	void BendingEnergy::host_evaluate(lcs::SimulationData<std::vector>& host_sim_data, lcs::MeshData<std::vector>& host_mesh_data)
	{
		auto& bending_edges = host_sim_data.get_bending_edge_data();

		CpuParallel::parallel_for(
			0,
			bending_edges.get_num_indices(),
			[sa_x = std::span(host_sim_data.sa_x),
				sa_bending_edges = std::span(bending_edges.constraint_indices),
				sa_bending_edges_Q = std::span(bending_edges.sa_bending_edges_Q),
				sa_bending_edges_rest_angle = std::span(bending_edges.sa_bending_edges_rest_angle),
				sa_bending_edges_rest_area = std::span(bending_edges.sa_bending_edges_rest_area),
				sa_bending_edges_stiffness = std::span(bending_edges.sa_bending_edges_stiffness),
				output_gradient_ptr = std::span(bending_edges.constraint_gradients),
				output_hessian_ptr = std::span(bending_edges.constraint_hessians),
				scaling = get_scene_params().get_bending_stiffness_scaling()](const uint eid)
			{
				uint4  edge = sa_bending_edges[eid];
				float3 vert_pos[4] = { sa_x[edge[0]], sa_x[edge[1]], sa_x[edge[2]], sa_x[edge[3]] };

				const float rest_angle = sa_bending_edges_rest_angle[eid];
				const float area = sa_bending_edges_rest_area[eid];
				const float stiff = sa_bending_edges_stiffness[eid] * scaling * area;

				auto eval = detail::bending_energy::evaluate<float, float3, float3x3>(
					vert_pos[0], vert_pos[1], vert_pos[2], vert_pos[3], rest_angle, stiff);

				for (uint ii = 0; ii < 4; ii++)
				{
					output_gradient_ptr[eid * 4 + ii] = eval.gradients[ii];
					for (uint jj = 0; jj < 4; jj++)
					{
						output_hessian_ptr[eid * 16 + ii * 4 + jj] = eval.hessians[ii * 4 + jj];
					}
				}
			});
	}

} // namespace lcs