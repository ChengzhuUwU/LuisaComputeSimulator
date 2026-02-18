#pragma once

#include "Energies/energy.h"
#include "Energies/energy_offsets.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/simulation_data.h"
#include <luisa/dsl/builtin.h>

namespace lcs
{
	class AbdOrthoEnergy : public Energy
	{
	public:
		AbdOrthoEnergy(luisa::compute::BufferView<float> sa_system_energy, luisa::compute::BufferView<float3> sa_q) noexcept;
		void   compile(AsyncCompiler& compiler) override;
		void   device_compute_energy(luisa::compute::Stream& stream) override;
		void   device_compute_energy(luisa::compute::Stream&				 stream,
			  const Constitutions::AbdOrthogonality<luisa::compute::Buffer>& constraint,
			  const luisa::compute::Buffer<float3>&							 sa_q,
			  size_t														 dispatch_count);
		void   device_evaluate(luisa::compute::Stream&						 stream,
			  const Constitutions::AbdOrthogonality<luisa::compute::Buffer>& constraint,
			  const luisa::compute::Buffer<float3>&							 sa_q,
			  size_t														 dispatch_count);
		double host_evaluate(const std::vector<float>& host_energy) override;
		// Host-side evaluation for ABD orthogonality
		void host_evaluate(lcs::SimulationData<std::vector>& host_sim_data, lcs::MeshData<std::vector>& host_mesh_data);

	private:
		luisa::compute::BufferView<float>												   _sa_system_energy;
		luisa::compute::BufferView<float3>												   _sa_q;
		luisa::compute::Shader<1, Constitutions::AbdOrthogonality<luisa::compute::Buffer>> _shader;
		luisa::compute::Shader<1, Constitutions::AbdOrthogonality<luisa::compute::Buffer>> _eval_shader;
	};

} // namespace lcs
