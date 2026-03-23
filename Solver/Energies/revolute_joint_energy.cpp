#include "revolute_joint_energy.h"
#include "Utils/reduce_helper.h"

using namespace luisa::compute;

namespace lcs
{
	RevoluteJointEnergy::RevoluteJointEnergy(BufferView<float> sa_system_energy, BufferView<luisa::float3> sa_q) noexcept
		: _sa_system_energy(sa_system_energy)
		, _sa_q(sa_q)
	{
	}

	void RevoluteJointEnergy::compile(AsyncCompiler& compiler)
	{
		auto default_option = compiler.default_option();
		compiler.compile<1>(
			_shader,
			[sa_system_energy = _sa_system_energy, sa_q = _sa_q](Var<BufferView<uint4>> indices_a,
				Var<BufferView<uint4>>													indices_b,
				Var<BufferView<float3>>													anchor_a_local,
				Var<BufferView<float3>>													anchor_b_local,
				Var<BufferView<float3>>													axis_world,
				Var<BufferView<float3>>													axis_a_local,
				Var<BufferView<float3>>													axis_b_local,
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
				const Float3 axis = axis_world->read(joint_idx);
				const Float3 axis_a = axis_a_local->read(joint_idx);
				const Float3 axis_b = axis_b_local->read(joint_idx);
				const Float2 stiff = stiffness->read(joint_idx);

				Float3 p_a = q[0] + q[1] * anchor_a.x + q[2] * anchor_a.y + q[3] * anchor_a.z;
				Float3 p_b = q[4] + q[5] * anchor_b.x + q[6] * anchor_b.y + q[7] * anchor_b.z;
				Float3 r_pos = p_b - p_a;

				Float3 axis_a_world = q[1] * axis_a.x + q[2] * axis_a.y + q[3] * axis_a.z;
				Float3 axis_b_world = q[5] * axis_b.x + q[6] * axis_b.y + q[7] * axis_b.z;
				Float3 r_axis_a = axis_a_world - axis * dot(axis, axis_a_world);
				Float3 r_axis_b = axis_b_world - axis * dot(axis, axis_b_world);

				Float energy = 0.5f * stiff.x * dot(r_pos, r_pos);
				energy += 0.5f * stiff.y * dot(r_axis_a, r_axis_a);
				energy += 0.5f * stiff.y * dot(r_axis_b, r_axis_b);
				energy = ParallelIntrinsic::block_intrinsic_reduce(energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
				$if(joint_idx % 256 == 0)
				{
					sa_system_energy->atomic(offset_revolute_joint).fetch_add(energy);
				};
			},
			default_option);
	}

	void RevoluteJointEnergy::device_compute_energy(luisa::compute::Stream& stream)
	{
		// left empty - use typed overload below
	}

	void RevoluteJointEnergy::device_compute_energy(luisa::compute::Stream& stream,
		const luisa::compute::Buffer<luisa::uint4>&							indices_a,
		const luisa::compute::Buffer<luisa::uint4>&							indices_b,
		const luisa::compute::Buffer<luisa::float3>&						anchor_a_local,
		const luisa::compute::Buffer<luisa::float3>&						anchor_b_local,
		const luisa::compute::Buffer<luisa::float3>&						axis_world,
		const luisa::compute::Buffer<luisa::float3>&						axis_a_local,
		const luisa::compute::Buffer<luisa::float3>&						axis_b_local,
		const luisa::compute::Buffer<luisa::float2>&						stiffness,
		size_t																dispatch_count)
	{
		stream << _shader(indices_a,
			indices_b,
			anchor_a_local,
			anchor_b_local,
			axis_world,
			axis_a_local,
			axis_b_local,
			stiffness)
					  .dispatch(dispatch_count);
	}

	double RevoluteJointEnergy::host_evaluate(const std::vector<float>& host_energy)
	{
		return host_energy[offset_revolute_joint];
	}

} // namespace lcs
