#pragma once

#include "Energies/energy.h"
#include "Energies/detail/bending_energy.hpp"
#include "Energies/energy_offsets.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/simulation_data.h"
#include <luisa/dsl/builtin.h>

namespace lcs
{
	class BendingEnergy : public Energy
	{
	public:
		BendingEnergy(luisa::compute::BufferView<float> sa_system_energy) noexcept;
		void   compile(AsyncCompiler& compiler) override;
		void   device_compute_energy(luisa::compute::Stream& stream) override;
		void   device_compute_energy(luisa::compute::Stream&			stream,
			  const Constitutions::BendingEdge<luisa::compute::Buffer>& constraint,
			  const luisa::compute::Buffer<float3>&						sa_x,
			  float														scaling,
			  size_t													dispatch_count);
		void   device_evaluate(luisa::compute::Stream&					stream,
			  const Constitutions::BendingEdge<luisa::compute::Buffer>& constraint,
			  const luisa::compute::Buffer<float3>&						sa_x,
			  float														scaling,
			  size_t													dispatch_count);
		double host_evaluate(const std::vector<float>& host_energy) override;
		// Host-side bending evaluation
		void host_evaluate(lcs::SimulationData<std::vector>& host_sim_data, lcs::MeshData<std::vector>& host_mesh_data);

	private:
		luisa::compute::BufferView<float>																						 _sa_system_energy;
		luisa::compute::Shader<1, Constitutions::BendingEdge<luisa::compute::Buffer>, luisa::compute::BufferView<float3>, float> _shader;
		luisa::compute::Shader<1, Constitutions::BendingEdge<luisa::compute::Buffer>, luisa::compute::BufferView<float3>, float> _eval_shader;
	};

} // namespace lcs

namespace lcs
{
	namespace BendingEnergyUtils
	{
		using Float3 = luisa::compute::Float3;
		using Float = luisa::compute::Float;

		float compute_d_theta_d_x(const float3& x2, const float3& x1, const float3& x0, const float3& x3, float3 gradient[4]);
		Float compute_d_theta_d_x(const Float3& x2, const Float3& x1, const Float3& x0, const Float3& x3, Float3 gradient[4]);
		float compute_theta(const float3& x2, const float3& x1, const float3& x0, const float3& x3);
		Float compute_theta(const Float3& x2, const Float3& x1, const Float3& x0, const Float3& x3);

	}; // namespace BendingEnergyUtils
} // namespace lcs
