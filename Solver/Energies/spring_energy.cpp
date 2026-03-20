#include "spring_energy.h"
#include "Energies/detail/stretch_spring_energy.hpp"
#include "SimulationCore/base_mesh.h"
#include "Utils/cpu_parallel.h"

using namespace luisa::compute;

namespace lcs
{
	SpringEnergy::SpringEnergy(BufferView<float> sa_system_energy) noexcept
		: _sa_system_energy(sa_system_energy)
	{
	}

	void SpringEnergy::compile(AsyncCompiler& compiler)
	{
		luisa::compute::ShaderOption default_option = compiler.default_option();
		compiler.compile<1>(
			_shader,
			[sa_system_energy = _sa_system_energy](Var<Constitutions::StretchSpring<luisa::compute::Buffer>> constraint,
				Var<BufferView<float3>>																		 sa_x)
			{
				auto& sa_edges = constraint.constraint_indices;
				auto& sa_edge_rest_state_length = constraint.sa_stretch_spring_rest_state_length;
				auto& sa_stretch_spring_stiffness = constraint.sa_stretch_spring_stiffness;

				const Uint eid = dispatch_id().x;
				Float	   energy = 0.0f;
				{
					const Uint2	 edge = sa_edges->read(eid);
					const Float	 rest_edge_length = sa_edge_rest_state_length->read(eid);
					const Float3 diff = sa_x->read(edge[1]) - sa_x->read(edge[0]);
					const Float	 l = sqrt_scalar(length_squared_vec(diff));
					const Float	 C = l - rest_edge_length;
					const Float	 stiffness = sa_stretch_spring_stiffness->read(eid);
					energy = detail::stretch_spring_energy::compute_energy(stiffness, C);
				};

				energy = ParallelIntrinsic::block_intrinsic_reduce(energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
				$if(eid % 256 == 0)
				{
					sa_system_energy->atomic(offset_stretch_spring).fetch_add(energy);
				};
			},
			default_option);

		// Gradient/Hessian evaluate shader (moved from centralized newton compile)
		compiler.compile<1>(
			_eval_shader,
			[sa_x = luisa::compute::BufferView<float3>()](Var<Constitutions::StretchSpring<luisa::compute::Buffer>> constraint,
				Var<BufferView<float3>>																				sa_x_in)
			{
				auto& sa_edges = constraint.constraint_indices;
				auto& sa_rest_length = constraint.sa_stretch_spring_rest_state_length;
				auto& sa_stretch_spring_stiffness = constraint.sa_stretch_spring_stiffness;
				auto& output_gradient_ptr = constraint.constraint_gradients;
				auto& output_hessian_ptr = constraint.constraint_hessians;

				const UInt eid = dispatch_id().x;
				UInt2	   edge = sa_edges->read(eid);

				Float3 vert_pos[2] = { sa_x_in->read(edge.x), sa_x_in->read(edge.y) };

				const Float L = sa_rest_length->read(eid);
				const Float stiffness_spring = sa_stretch_spring_stiffness->read(eid);

				const detail::stretch_spring_energy::Input<Float, Float3> input{
					.x0 = vert_pos[0],
					.x1 = vert_pos[1],
					.rest_length = L,
					.stiffness = stiffness_spring,
				};
				Float3x3 identify = float3x3::eye(1.0f);
				auto	 eval = detail::stretch_spring_energy::evaluate(input, identify);

				// Output
				{
					output_gradient_ptr->write(eid * 2 + 0, eval.gradients[0]);
					output_gradient_ptr->write(eid * 2 + 1, eval.gradients[1]);

					output_hessian_ptr->write(eid * 4 + 0, eval.hessians[0]); // (0, 0)
					output_hessian_ptr->write(eid * 4 + 1, eval.hessians[1]); // (0, 1)
					output_hessian_ptr->write(eid * 4 + 2, eval.hessians[2]); // (1, 0)
					output_hessian_ptr->write(eid * 4 + 3, eval.hessians[3]); // (1, 1)
				}
			},
			default_option);
	}

	void SpringEnergy::device_compute_energy(luisa::compute::Stream& stream)
	{
	}

	void SpringEnergy::device_compute_energy(luisa::compute::Stream& stream,
		const Constitutions::StretchSpring<luisa::compute::Buffer>&	 constraint,
		const luisa::compute::Buffer<float3>&						 sa_x,
		size_t														 dispatch_count)
	{
		stream << _shader(constraint, sa_x).dispatch(dispatch_count);
	}

	void SpringEnergy::device_evaluate(luisa::compute::Stream&		stream,
		const Constitutions::StretchSpring<luisa::compute::Buffer>& constraint,
		const luisa::compute::Buffer<float3>&						sa_x,
		size_t														dispatch_count)
	{
		// dispatch gradient/hessian evaluate shader
		stream << _eval_shader(constraint, sa_x.view()).dispatch(dispatch_count);
	}

	double SpringEnergy::host_evaluate(const std::vector<float>& host_energy)
	{
		return host_energy[offset_stretch_spring];
	}

	void SpringEnergy::host_evaluate(lcs::SimulationData<std::vector>& host_sim_data, lcs::MeshData<std::vector>& host_mesh_data)
	{
		auto& stretch_springs = host_sim_data.get_stretch_spring_data();

		CpuParallel::parallel_for(0,
			stretch_springs.get_num_indices(),
			[sa_x = std::span(host_sim_data.sa_x),
				sa_edges = std::span(stretch_springs.constraint_indices),
				sa_rest_length = std::span(stretch_springs.sa_stretch_spring_rest_state_length),
				sa_stretch_spring_stiffness = std::span(stretch_springs.sa_stretch_spring_stiffness),
				output_gradient_ptr = std::span(stretch_springs.constraint_gradients),
				output_hessian_ptr = std::span(stretch_springs.constraint_hessians)](const uint eid)
			{
				uint2 edge = sa_edges[eid];

				float3 vert_pos[2] = { sa_x[edge[0]], sa_x[edge[1]] };

				const float L = sa_rest_length[eid];
				const float stiffness_stretch_spring = sa_stretch_spring_stiffness[eid];

				const detail::stretch_spring_energy::Input<float, float3> input{
					.x0 = vert_pos[0],
					.x1 = vert_pos[1],
					.rest_length = L,
					.stiffness = stiffness_stretch_spring,
				};
				auto eval = detail::stretch_spring_energy::evaluate(input, luisa::make_float3x3(1.0f));

				output_gradient_ptr[eid * 2 + 0] = eval.gradients[0];
				output_gradient_ptr[eid * 2 + 1] = eval.gradients[1];

				output_hessian_ptr[eid * 4 + 0] = eval.hessians[0];
				output_hessian_ptr[eid * 4 + 1] = eval.hessians[1];
				output_hessian_ptr[eid * 4 + 2] = eval.hessians[2];
				output_hessian_ptr[eid * 4 + 3] = eval.hessians[3];
			});
	}

} // namespace lcs
