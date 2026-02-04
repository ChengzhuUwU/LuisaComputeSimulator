#pragma once

#include "Energies/energy.h"
#include "Energies/energy_offsets.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/simulation_data.h"
#include <luisa/dsl/builtin.h>

namespace lcs
{
class AbdInertiaEnergy : public Energy
{
  public:
    AbdInertiaEnergy(luisa::compute::BufferView<float3> sa_q_tilde,
                     luisa::compute::BufferView<float>  sa_system_energy) noexcept;
    void   compile(AsyncCompiler& compiler) override;
    void   device_compute_energy(luisa::compute::Stream& stream) override;
    void   device_compute_energy(luisa::compute::Stream&                                  stream,
                                 const Constitutions::AbdInertia<luisa::compute::Buffer>& constraint,
                                 const luisa::compute::Buffer<float3>&                    sa_q,
                                 float                                                    substep_dt,
                                 size_t                                                   dispatch_count);
    void   device_evaluate(luisa::compute::Stream&                                  stream,
                           const Constitutions::AbdInertia<luisa::compute::Buffer>& constraint,
                           const luisa::compute::Buffer<float3>&                    sa_q,
                           float                                                    substep_dt,
                           size_t                                                   dispatch_count);
    double host_evaluate(const std::vector<float>& host_energy) override;
    // Host-side evaluation for ABD inertia
    void host_evaluate(lcs::SimulationData<std::vector>& host_sim_data, lcs::MeshData<std::vector>& host_mesh_data);

  private:
    luisa::compute::BufferView<float3> _sa_q_tilde;
    luisa::compute::BufferView<float>  _sa_system_energy;
    luisa::compute::Shader<1, Constitutions::AbdInertia<luisa::compute::Buffer>, luisa::compute::BufferView<float3>, float> _shader;
    luisa::compute::Shader<1, Constitutions::AbdInertia<luisa::compute::Buffer>, luisa::compute::BufferView<float3>, float> _eval_shader;
};

}  // namespace lcs
