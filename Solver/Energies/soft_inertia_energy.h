#pragma once

#include "SimulationCore/base_mesh.h"
#include "Energies/energy.h"
#include "SimulationCore/simulation_data.h"
#include "Energies/energy_offsets.h"
#include <luisa/dsl/builtin.h>

namespace lcs
{
	class SoftInertiaEnergy : public Energy
	{
	public:
		SoftInertiaEnergy(luisa::compute::BufferView<float3> sa_q_tilde_view,
			luisa::compute::BufferView<float>				 sa_system_energy_view) noexcept;
		void   compile(AsyncCompiler& compiler) override;
		void   device_compute_energy(luisa::compute::Stream& stream) override;
		void   device_compute_energy(luisa::compute::Stream&			stream,
			  const Constitutions::SoftInertia<luisa::compute::Buffer>& constraint,
			  const luisa::compute::Buffer<float3>&						sa_q,
			  float														substep_dt,
			  size_t													dispatch_count);
		void   device_evaluate(luisa::compute::Stream&					stream,
			  const Constitutions::SoftInertia<luisa::compute::Buffer>& constraint,
			  const luisa::compute::Buffer<float3>&						sa_q,
			  float														substep_dt,
			  size_t													dispatch_count);
		double host_evaluate(const std::vector<float>& host_energy) override;
		// Host-side evaluation: compute per-constraint gradients and hessians on CPU
		void host_evaluate(lcs::SimulationData<std::vector>& host_sim_data, lcs::MeshData<std::vector>& host_mesh_data);

	private:
		luisa::compute::BufferView<float3>																						 _sa_q_tilde_view;
		luisa::compute::BufferView<float>																						 _sa_system_energy_view;
		luisa::compute::Shader<1, Constitutions::SoftInertia<luisa::compute::Buffer>, luisa::compute::BufferView<float3>, float> _shader;
		luisa::compute::Shader<1, Constitutions::SoftInertia<luisa::compute::Buffer>, luisa::compute::BufferView<float3>, float> _eval_shader;
	};

} // namespace lcs
