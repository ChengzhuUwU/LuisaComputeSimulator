#pragma once

#include "Core/svd_3x3.h"
#include "Energies/energy.h"
#include "Energies/energy_offsets.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/simulation_data.h"
#include <luisa/dsl/builtin.h>

namespace lcs
{

} // namespace lcs

namespace lcs
{
	class TetElasticEnergy : public Energy
	{
	public:
		TetElasticEnergy(luisa::compute::BufferView<float> sa_system_energy) noexcept;
		void   compile(AsyncCompiler& compiler) override;
		void   device_compute_energy(luisa::compute::Stream& stream) override;
		void   device_compute_energy(luisa::compute::Stream&		  stream,
			  const Constitutions::StressTet<luisa::compute::Buffer>& constraint,
			  const luisa::compute::Buffer<float3>&					  sa_x,
			  size_t												  dispatch_count);
		void   device_evaluate(luisa::compute::Stream&				  stream,
			  const Constitutions::StressTet<luisa::compute::Buffer>& constraint,
			  const luisa::compute::Buffer<float3>&					  sa_x,
			  size_t												  dispatch_count);
		double host_evaluate(const std::vector<float>& host_energy) override;
		void   host_evaluate(lcs::SimulationData<std::vector>& host_sim_data,
			  lcs::MeshData<std::vector>&					   host_mesh_data);

	private:
		luisa::compute::BufferView<float> _sa_system_energy;

		// Energy-only shader (per tet)
		luisa::compute::Shader<1, Constitutions::StressTet<luisa::compute::Buffer>, luisa::compute::BufferView<float3>>
			_shader;

		// Gradient/Hessian evaluate shader (per tet)
		luisa::compute::Shader<1, Constitutions::StressTet<luisa::compute::Buffer>, luisa::compute::BufferView<float3>>
			_eval_shader;
	};

} // namespace lcs
