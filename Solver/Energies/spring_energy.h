#pragma once

#include <luisa/dsl/builtin.h>
#include "SimulationCore/base_mesh.h"
#include "Utils/reduce_helper.h"
#include "Energies/energy.h"
#include "Energies/energy_offsets.h"
#include "SimulationCore/simulation_data.h"

namespace lcs
{
class SpringEnergy : public Energy
{
  public:
    SpringEnergy(luisa::compute::BufferView<float> sa_system_energy) noexcept;
    void   compile(AsyncCompiler& compiler) override;
    void   device_compute_energy(luisa::compute::Stream& stream) override;
    void   device_compute_energy(luisa::compute::Stream&                                     stream,
                                 const Constitutions::StretchSpring<luisa::compute::Buffer>& constraint,
                                 const luisa::compute::Buffer<float3>&                       sa_x,
                                 size_t dispatch_count);
    void   device_evaluate(luisa::compute::Stream&                                     stream,
                           const Constitutions::StretchSpring<luisa::compute::Buffer>& constraint,
                           const luisa::compute::Buffer<float3>&                       sa_x,
                           size_t                                                      dispatch_count);
    double host_evaluate(const std::vector<float>& host_energy) override;
    // Host-side eval for stretch springs: fill per-constraint gradients/hessians
    void host_evaluate(lcs::SimulationData<std::vector>& host_sim_data, lcs::MeshData<std::vector>& host_mesh_data);

  private:
    luisa::compute::BufferView<float> _sa_system_energy;
    luisa::compute::Shader<1, Constitutions::StretchSpring<luisa::compute::Buffer>, luisa::compute::BufferView<float3>> _shader;
    luisa::compute::Shader<1, Constitutions::StretchSpring<luisa::compute::Buffer>, luisa::compute::BufferView<float3>> _eval_shader;
};

}  // namespace lcs
