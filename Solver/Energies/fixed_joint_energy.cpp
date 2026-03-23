#include "fixed_joint_energy.h"
#include "Energies/detail/fixed_joint_constaint.hpp"
#include "Utils/reduce_helper.h"

using namespace luisa::compute;

namespace lcs
{
	FixedJointEnergy::FixedJointEnergy(BufferView<float> sa_system_energy, BufferView<luisa::float3> sa_q) noexcept
		: _sa_system_energy(sa_system_energy)
		, _sa_q(sa_q)
	{
	}

	void FixedJointEnergy::compile(AsyncCompiler& compiler)
	{
		auto default_option = compiler.default_option();
		compiler.compile<1>(
			_shader,
			[sa_system_energy = _sa_system_energy, sa_q = _sa_q](Var<BufferView<uint4>> indices_a,
				Var<BufferView<uint4>>													indices_b,
				Var<BufferView<float3>>													anchor_a_local,
				Var<BufferView<float3>>													anchor_b_local,
				Var<BufferView<float2>>													stiffness)
			{
				const UInt	joint_idx = dispatch_id().x;
				const UInt4 idx_a = indices_a->read(joint_idx);
				const UInt4 idx_b = indices_b->read(joint_idx);

				Float3 q[8] = {
					sa_q->read(idx_a.x), sa_q->read(idx_a.y), sa_q->read(idx_a.z), sa_q->read(idx_a.w),
					sa_q->read(idx_b.x), sa_q->read(idx_b.y), sa_q->read(idx_b.z), sa_q->read(idx_b.w)
				};

				const Float3 anchor_a = anchor_a_local->read(joint_idx);
				const Float3 anchor_b = anchor_b_local->read(joint_idx);
				const Float2 stiff = stiffness->read(joint_idx);

				Float energy = detail::fixed_joint_constaint::compute_energy<Float, Float3, Float3x3>(
					q,
					anchor_a,
					anchor_b,
					stiff.x,
					stiff.y,
					make_float3x3(1.0f));
				energy = ParallelIntrinsic::block_intrinsic_reduce(energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
				$if(joint_idx % 256 == 0)
				{
					sa_system_energy->atomic(offset_fixed_joint).fetch_add(energy);
				};
			},
			default_option);
	}

	void FixedJointEnergy::device_compute_energy(luisa::compute::Stream& stream)
	{
		// left empty - use typed overload below
	}

	void FixedJointEnergy::device_compute_energy(luisa::compute::Stream& stream,
		const luisa::compute::Buffer<luisa::uint4>&						 indices_a,
		const luisa::compute::Buffer<luisa::uint4>&						 indices_b,
		const luisa::compute::Buffer<luisa::float3>&					 anchor_a_local,
		const luisa::compute::Buffer<luisa::float3>&					 anchor_b_local,
		const luisa::compute::Buffer<luisa::float2>&					 stiffness,
		size_t															 dispatch_count)
	{
		stream << _shader(indices_a, indices_b, anchor_a_local, anchor_b_local, stiffness).dispatch(dispatch_count);
	}

	double FixedJointEnergy::host_evaluate(const std::vector<float>& host_energy)
	{
		return host_energy[offset_fixed_joint];
	}

} // namespace lcs
