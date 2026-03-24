#pragma once

#include "Energies/energy.h"
#include "Energies/energy_offsets.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/simulation_data.h"

namespace lcs
{
	// Unified energy class for Fixed, Prismatic, and Revolute joint constraints.
	// Replaces the three separate FixedJointEnergy / PrismaticJointEnergy /
	// RevoluteJointEnergy classes.  The joint type is stored per-element in
	// JointConstraint::joint_type and branched on at shader runtime.
	class JointConstraintEnergy : public Energy
	{
	public:
		JointConstraintEnergy(luisa::compute::BufferView<float> sa_system_energy,
			luisa::compute::BufferView<luisa::float3>			sa_q) noexcept;

		void compile(AsyncCompiler& compiler) override;

		// Base-class no-op (typed overload required).
		void device_compute_energy(luisa::compute::Stream& stream) override;

		// Compute per-joint scalar energy and accumulate to sa_system_energy[offset_joint_constraint].
		void device_compute_energy(luisa::compute::Stream&				  stream,
			const Constitutions::JointConstraint<luisa::compute::Buffer>& joint_data,
			size_t														  dispatch_count);

		// Evaluate per-joint gradient (8 float3) and Hessian (64 float3x3) and write
		// them into joint_data.constraint_gradients / constraint_hessians.
		void device_evaluate(luisa::compute::Stream&					  stream,
			const Constitutions::JointConstraint<luisa::compute::Buffer>& joint_data,
			size_t														  dispatch_count);

		double host_evaluate(const std::vector<float>& host_energy) override;
		void   host_evaluate(lcs::SimulationData<std::vector>& host_sim_data, lcs::MeshData<std::vector>& host_mesh_data);

	private:
		luisa::compute::BufferView<float>		  _sa_system_energy;
		luisa::compute::BufferView<luisa::float3> _sa_q;

		// Energy-only shader (no grad/hess).
		luisa::compute::Shader<1, Constitutions::JointConstraint<luisa::compute::Buffer>> _shader;

		// Gradient/Hessian evaluation shader.
		luisa::compute::Shader<1, Constitutions::JointConstraint<luisa::compute::Buffer>> _eval_shader;
	};

} // namespace lcs
