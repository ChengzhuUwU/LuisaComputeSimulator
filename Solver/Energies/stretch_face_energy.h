#pragma once

#include "Energies/energy.h"
#include "Energies/energy_offsets.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/simulation_data.h"
#include "Energies/fem_utils.h"
#include <luisa/dsl/builtin.h>

namespace lcs
{
	namespace StretchEnergy
	{
		namespace detail
		{
			float	   stretch_energy(const float2x3& F, float mu);
			float	   shear_energy(const float2x3& F, float lmd);
			Var<float> stretch_energy(const Var<float2x3>& F, Var<float> mu);
			Var<float> shear_energy(const Var<float2x3>& F, Var<float> lmd);

			float2x3 stretch_gradient(const float2x3& F, const float mu);
			float2x3 shear_gradient(const float2x3& F, const float lmd);

			float6x6 stretch_hessian(const float2x3& F, float mu);
			float6x6 shear_hessian(const float2x3& F, float mu);

			Var<float2x3> stretch_gradient(const Var<float2x3>& F, const Var<float> mu);
			Var<float2x3> shear_gradient(const Var<float2x3>& F, const Var<float> lmd);
			Var<float6x6> stretch_hessian(const Var<float2x3>& F, Var<float> mu);
			Var<float6x6> shear_hessian(const Var<float2x3>& F, Var<float> mu);

		} // namespace detail

		void compute_gradient_hessian(const float3& x0,
			const float3&							x1,
			const float3&							x2,
			const float2x2&							Dm,
			const float								mu,
			const float								lambda,
			const float								area,
			float3x3&								dedx,
			float9x9&								d2edx2);
		void compute_gradient_hessian(const Var<float3>& x0,
			const Var<float3>&							 x1,
			const Var<float3>&							 x2,
			const Var<float2x2>&						 Dm,
			const Var<float>							 mu,
			const Var<float>							 lambda,
			const Var<float>							 area,
			Var<float3x3>&								 dedx,
			Var<float9x9>&								 d2edx2);

		float	   compute_energy(const float3& x0,
				 const float3&					x1,
				 const float3&					x2,
				 const float2x2&				Dm,
				 const float					mu,
				 const float					lambda,
				 const float					area);
		Var<float> compute_energy(const Var<float3>& x0,
			const Var<float3>&						 x1,
			const Var<float3>&						 x2,
			const Var<float2x2>&					 Dm,
			const Var<float>						 mu,
			const Var<float>						 lambda,
			const Var<float>						 area);

	}; // namespace StretchEnergy

} // namespace lcs

namespace lcs
{
	class StretchFaceEnergy : public Energy
	{
	public:
		StretchFaceEnergy(luisa::compute::BufferView<float> sa_system_energy) noexcept;
		void   compile(AsyncCompiler& compiler) override;
		void   device_compute_energy(luisa::compute::Stream& stream) override;
		void   device_compute_energy(luisa::compute::Stream&			stream,
			  const Constitutions::StretchFace<luisa::compute::Buffer>& constraint,
			  const luisa::compute::Buffer<float3>&						sa_x,
			  size_t													dispatch_count);
		void   device_evaluate(luisa::compute::Stream&					stream,
			  const Constitutions::StretchFace<luisa::compute::Buffer>& constraint,
			  const luisa::compute::Buffer<float3>&						sa_x,
			  size_t													dispatch_count);
		double host_evaluate(const std::vector<float>& host_energy) override;
		// Host-side evaluation for stretch faces
		void host_evaluate(lcs::SimulationData<std::vector>& host_sim_data, lcs::MeshData<std::vector>& host_mesh_data);

	private:
		luisa::compute::BufferView<float>																				  _sa_system_energy;
		luisa::compute::Shader<1, Constitutions::StretchFace<luisa::compute::Buffer>, luisa::compute::BufferView<float3>> _shader;
		luisa::compute::Shader<1, Constitutions::StretchFace<luisa::compute::Buffer>, luisa::compute::BufferView<float3>> _eval_shader;
	};

} // namespace lcs
