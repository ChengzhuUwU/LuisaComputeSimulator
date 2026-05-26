#include "joint_constraint_energy.h"
#include "Energies/detail/fixed_joint_constaint.hpp"
#include "Energies/detail/prismatic_joint_constaint.hpp"
#include "Energies/detail/revolute_joint_constaint.hpp"
#include "Utils/cpu_parallel.h"
#include "Utils/reduce_helper.h"

using namespace luisa::compute;

namespace lcs
{

	JointConstraintEnergy::JointConstraintEnergy(BufferView<float> sa_system_energy,
		BufferView<luisa::float3>								   sa_q) noexcept
		: _sa_system_energy(sa_system_energy)
		, _sa_q(sa_q)
	{
	}

	void JointConstraintEnergy::compile(AsyncCompiler& compiler)
	{
		auto default_option = compiler.default_option();

		// ── Energy-only shader ──────────────────────────────────────────────────
		compiler.compile<1>(
			_shader,
			[sa_system_energy = _sa_system_energy, sa_q = _sa_q](
				Var<Constitutions::JointConstraint<luisa::compute::Buffer>> joint)
			{
				const UInt joint_idx = dispatch_id().x;

				const auto indices = joint.constraint_indices->read(joint_idx);
				Float3	   q[8] = {
					sa_q->read(indices[0]),
					sa_q->read(indices[1]),
					sa_q->read(indices[2]),
					sa_q->read(indices[3]),
					sa_q->read(indices[4]),
					sa_q->read(indices[5]),
					sa_q->read(indices[6]),
					sa_q->read(indices[7])
				};

				const Float3 anchor_a = joint.anchor_a_local->read(joint_idx);
				const Float3 anchor_b = joint.anchor_b_local->read(joint_idx);
				const Float3 rest_pos_delta = joint.rest_position_delta->read(joint_idx);
				const Float3 rest_rot_c0 = joint.rest_rot_col0_a_to_b->read(joint_idx);
				const Float3 rest_rot_c1 = joint.rest_rot_col1_a_to_b->read(joint_idx);
				const Float3 rest_rot_c2 = joint.rest_rot_col2_a_to_b->read(joint_idx);
				const Float3 axis_a = joint.axis_a_local->read(joint_idx);
				const Float3 axis_b = joint.axis_b_local->read(joint_idx);
				const Float2 stiff = joint.stiffness->read(joint_idx);
				const UInt	 jtype = joint.joint_type->read(joint_idx);

				Float3 p_a = q[0] + q[1] * anchor_a.x + q[2] * anchor_a.y + q[3] * anchor_a.z;
				Float3 p_b = q[4] + q[5] * anchor_b.x + q[6] * anchor_b.y + q[7] * anchor_b.z;

				Float energy = 0.0f;

				$if(jtype == static_cast<uint>(JointConstraintType::Fixed))
				{
					Float3 target_delta = q[1] * rest_pos_delta.x + q[2] * rest_pos_delta.y + q[3] * rest_pos_delta.z;
					Float3 r_pos = (p_b - p_a) - target_delta;
					energy = 0.5f * stiff.x * dot(r_pos, r_pos);
					Float3 rest_rot_cols[3] = { rest_rot_c0, rest_rot_c1, rest_rot_c2 };
					for (int col = 0; col < 3; ++col)
					{
						Float3 target_col = q[1] * rest_rot_cols[col].x + q[2] * rest_rot_cols[col].y + q[3] * rest_rot_cols[col].z;
						Float3 r_rot = q[5 + col] - target_col;
						energy += 0.5f * stiff.y * dot(r_rot, r_rot);
					}
				}
				$elif(jtype == static_cast<uint>(JointConstraintType::Prismatic))
				{
					Float3 target_delta = q[1] * rest_pos_delta.x + q[2] * rest_pos_delta.y + q[3] * rest_pos_delta.z;
					Float3 d = (p_b - p_a) - target_delta;
					Float3 ax = q[1] * axis_a.x + q[2] * axis_a.y + q[3] * axis_a.z;
					Float3 r_pos = cross(d, ax);
					energy = 0.5f * stiff.x * dot(r_pos, r_pos);
					Float3 rest_rot_cols[3] = { rest_rot_c0, rest_rot_c1, rest_rot_c2 };
					for (int col = 0; col < 3; ++col)
					{
						Float3 target_col = q[1] * rest_rot_cols[col].x + q[2] * rest_rot_cols[col].y + q[3] * rest_rot_cols[col].z;
						Float3 r_rot = q[5 + col] - target_col;
						energy += 0.5f * stiff.y * dot(r_rot, r_rot);
					}
					// Slide limit energy.
					const Float2 slims = joint.slide_limits->read(joint_idx);
					const Float	 s = dot(d, ax);
					$if(s < slims.x)
					{
						energy += 0.5f * stiff.x * (s - slims.x) * (s - slims.x);
					};
					$if(s > slims.y)
					{
						energy += 0.5f * stiff.x * (s - slims.y) * (s - slims.y);
					};
				}
				$else
				{
					// Revolute
					Float3 target_delta = q[1] * rest_pos_delta.x + q[2] * rest_pos_delta.y + q[3] * rest_pos_delta.z;
					Float3 r_pos = (p_b - p_a) - target_delta;
					energy = 0.5f * stiff.x * dot(r_pos, r_pos);

					Float3 axis_a_world = q[1] * axis_a.x + q[2] * axis_a.y + q[3] * axis_a.z;
					Float3 axis_b_world = q[5] * axis_b.x + q[6] * axis_b.y + q[7] * axis_b.z;
					Float3 r_axis = axis_a_world - axis_b_world;
					energy += 0.5f * stiff.y * dot(r_axis, r_axis);
				};

				energy = ParallelIntrinsic::block_intrinsic_reduce(
					energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
				$if(joint_idx % 256 == 0)
				{
					sa_system_energy->atomic(offset_joint_constraint).fetch_add(energy);
				};
			},
			default_option);

		// ── Gradient / Hessian evaluation shader ────────────────────────────────
		compiler.compile<1>(
			_eval_shader,
			[sa_q = _sa_q](Var<Constitutions::JointConstraint<luisa::compute::Buffer>> joint)
			{
				const UInt joint_idx = dispatch_id().x;

				const auto indices = joint.constraint_indices->read(joint_idx);
				Float3	   q[8] = {
					sa_q->read(indices[0]),
					sa_q->read(indices[1]),
					sa_q->read(indices[2]),
					sa_q->read(indices[3]),
					sa_q->read(indices[4]),
					sa_q->read(indices[5]),
					sa_q->read(indices[6]),
					sa_q->read(indices[7])
				};

				const Float3   anchor_a = joint.anchor_a_local->read(joint_idx);
				const Float3   anchor_b = joint.anchor_b_local->read(joint_idx);
				const Float3   rest_pos_delta = joint.rest_position_delta->read(joint_idx);
				const Float3   rest_rot_c0 = joint.rest_rot_col0_a_to_b->read(joint_idx);
				const Float3   rest_rot_c1 = joint.rest_rot_col1_a_to_b->read(joint_idx);
				const Float3   rest_rot_c2 = joint.rest_rot_col2_a_to_b->read(joint_idx);
				const Float3   axis_a = joint.axis_a_local->read(joint_idx);
				const Float3   axis_b = joint.axis_b_local->read(joint_idx);
				const Float2   stiff = joint.stiffness->read(joint_idx);
				const UInt	   jtype = joint.joint_type->read(joint_idx);
				const Float3x3 I = make_float3x3(1.0f);

				$if(jtype == static_cast<uint>(JointConstraintType::Fixed))
				{
					auto eval = detail::fixed_joint_constaint::evaluate<Float, Float3, Float3x3>(
						q, anchor_a, anchor_b, rest_pos_delta, rest_rot_c0, rest_rot_c1, rest_rot_c2, stiff.x, stiff.y, I);
					for (uint i = 0; i < 8; ++i)
						joint.constraint_gradients->write(joint_idx * 8u + i, eval.gradients[i]);
					for (uint i = 0; i < 8; ++i)
						for (uint j = 0; j < 8; ++j)
							joint.constraint_hessians->write(joint_idx * 64u + i * 8u + j, eval.hessians[i * 8 + j]);
				}
				$elif(jtype == static_cast<uint>(JointConstraintType::Prismatic))
				{
					const Float2 slims = joint.slide_limits->read(joint_idx);

					auto eval = detail::prismatic_joint_constaint::evaluate<Float, Float3, Float3x3>(
						q, anchor_a, anchor_b, rest_pos_delta, rest_rot_c0, rest_rot_c1, rest_rot_c2, axis_a, stiff.x, stiff.y, slims.x, slims.y, I);
					for (uint i = 0; i < 8; ++i)
						joint.constraint_gradients->write(joint_idx * 8u + i, eval.gradients[i]);
					for (uint i = 0; i < 8; ++i)
						for (uint j = 0; j < 8; ++j)
							joint.constraint_hessians->write(joint_idx * 64u + i * 8u + j, eval.hessians[i * 8 + j]);
				}
				$else
				{
					// Revolute
					auto eval = detail::revolute_joint_constaint::evaluate<Float, Float3, Float3x3>(
						q, anchor_a, anchor_b, rest_pos_delta, axis_a, axis_b, stiff.x, stiff.y, I);
					for (uint i = 0; i < 8; ++i)
						joint.constraint_gradients->write(joint_idx * 8u + i, eval.gradients[i]);
					for (uint i = 0; i < 8; ++i)
						for (uint j = 0; j < 8; ++j)
							joint.constraint_hessians->write(joint_idx * 64u + i * 8u + j, eval.hessians[i * 8 + j]);
				};
			},
			default_option);
	}

	void JointConstraintEnergy::host_evaluate(lcs::SimulationData<std::vector>& host_sim_data, lcs::MeshData<std::vector>& host_mesh_data)
	{

		auto& joint_data = host_sim_data.get_joint_constraint_data();
		if (joint_data.is_valid())
		{
			CpuParallel::parallel_for(0, joint_data.get_num_indices(),
				[&](const uint joint_idx)
				{
					const auto indices = joint_data.constraint_indices[joint_idx];
					float3	   q[8] = {
						host_sim_data.sa_q[indices[0]],
						host_sim_data.sa_q[indices[1]],
						host_sim_data.sa_q[indices[2]],
						host_sim_data.sa_q[indices[3]],
						host_sim_data.sa_q[indices[4]],
						host_sim_data.sa_q[indices[5]],
						host_sim_data.sa_q[indices[6]],
						host_sim_data.sa_q[indices[7]]
					};

					const float3   anchor_a = joint_data.anchor_a_local[joint_idx];
					const float3   anchor_b = joint_data.anchor_b_local[joint_idx];
					const float3   rest_pos_delta = joint_data.rest_position_delta[joint_idx];
					const float3   rest_rot_c0 = joint_data.rest_rot_col0_a_to_b[joint_idx];
					const float3   rest_rot_c1 = joint_data.rest_rot_col1_a_to_b[joint_idx];
					const float3   rest_rot_c2 = joint_data.rest_rot_col2_a_to_b[joint_idx];
					const float3   axis_a = joint_data.axis_a_local[joint_idx];
					const float3   axis_b = joint_data.axis_b_local[joint_idx];
					const float2   stiff = joint_data.stiffness[joint_idx];
					const uint	   jtype = joint_data.joint_type[joint_idx];
					const float3x3 I = make_float3x3(1.0f);

					if (jtype == static_cast<uint>(JointConstraintType::Fixed))
					{
						auto eval = detail::fixed_joint_constaint::evaluate<float, float3, float3x3>(
							q, anchor_a, anchor_b, rest_pos_delta, rest_rot_c0, rest_rot_c1, rest_rot_c2, stiff.x, stiff.y, I);
						for (uint i = 0; i < 8; ++i)
							joint_data.constraint_gradients[joint_idx * 8u + i] = eval.gradients[i];
						for (uint i = 0; i < 8; ++i)
							for (uint j = 0; j < 8; ++j)
								joint_data.constraint_hessians[joint_idx * 64u + i * 8u + j] = eval.hessians[i * 8 + j];
					}
					else if (jtype == static_cast<uint>(JointConstraintType::Prismatic))
					{
						const float2 slims = joint_data.slide_limits[joint_idx];

						auto eval = detail::prismatic_joint_constaint::evaluate<float, float3, float3x3>(
							q, anchor_a, anchor_b, rest_pos_delta, rest_rot_c0, rest_rot_c1, rest_rot_c2, axis_a, stiff.x, stiff.y, slims.x, slims.y, I);
						for (uint i = 0; i < 8; ++i)
							joint_data.constraint_gradients[joint_idx * 8u + i] = eval.gradients[i];
						for (uint i = 0; i < 8; ++i)
							for (uint j = 0; j < 8; ++j)
								joint_data.constraint_hessians[joint_idx * 64u + i * 8u + j] = eval.hessians[i * 8 + j];
					}
					else
					{
						// Revolute
						auto eval = detail::revolute_joint_constaint::evaluate<float, float3, float3x3>(
							q, anchor_a, anchor_b, rest_pos_delta, axis_a, axis_b, stiff.x, stiff.y, I);
						for (uint i = 0; i < 8; ++i)
							joint_data.constraint_gradients[joint_idx * 8u + i] = eval.gradients[i];
						for (uint i = 0; i < 8; ++i)
							for (uint j = 0; j < 8; ++j)
								joint_data.constraint_hessians[joint_idx * 64u + i * 8u + j] = eval.hessians[i * 8 + j];
					}
				});
		}
	}

	void JointConstraintEnergy::device_compute_energy(luisa::compute::Stream& /*stream*/)
	{
		// left empty — use typed overload below
	}

	void JointConstraintEnergy::device_compute_energy(luisa::compute::Stream& stream,
		const Constitutions::JointConstraint<luisa::compute::Buffer>&		  joint_data,
		size_t																  dispatch_count)
	{
		stream << _shader(joint_data).dispatch(dispatch_count);
	}

	void JointConstraintEnergy::device_evaluate(luisa::compute::Stream& stream,
		const Constitutions::JointConstraint<luisa::compute::Buffer>&	joint_data,
		size_t															dispatch_count)
	{
		stream << _eval_shader(joint_data).dispatch(dispatch_count);
	}

	double JointConstraintEnergy::host_evaluate(const std::vector<float>& host_energy)
	{
		return host_energy[offset_joint_constraint];
	}

} // namespace lcs
