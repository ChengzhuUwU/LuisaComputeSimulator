#pragma once

#include "Energies/energy.h"
#include "Energies/energy_offsets.h"
#include <luisa/dsl/builtin.h>

namespace lcs
{
	class FixedJointEnergy : public Energy
	{
	public:
		FixedJointEnergy(luisa::compute::BufferView<float> sa_system_energy,
			luisa::compute::BufferView<luisa::float3>	   sa_q) noexcept;
		void   compile(AsyncCompiler& compiler) override;
		void   device_compute_energy(luisa::compute::Stream& stream) override;
		void   device_compute_energy(luisa::compute::Stream& stream,
			  const luisa::compute::Buffer<luisa::uint4>&	 indices_a,
			  const luisa::compute::Buffer<luisa::uint4>&	 indices_b,
			  const luisa::compute::Buffer<luisa::float3>&	 anchor_a_local,
			  const luisa::compute::Buffer<luisa::float3>&	 anchor_b_local,
			  const luisa::compute::Buffer<luisa::float2>&	 stiffness,
			  size_t										 dispatch_count);
		double host_evaluate(const std::vector<float>& host_energy) override;

	private:
		luisa::compute::BufferView<float>		  _sa_system_energy;
		luisa::compute::BufferView<luisa::float3> _sa_q;
		luisa::compute::Shader<1,
			luisa::compute::BufferView<luisa::uint4>,
			luisa::compute::BufferView<luisa::uint4>,
			luisa::compute::BufferView<luisa::float3>,
			luisa::compute::BufferView<luisa::float3>,
			luisa::compute::BufferView<luisa::float2>>
			_shader;
	};

} // namespace lcs
