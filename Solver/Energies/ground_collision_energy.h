#pragma once

#include <luisa/dsl/builtin.h>
#include "Energies/energy_offsets.h"
#include "Utils/reduce_helper.h"
#pragma once

#include "Energies/energy.h"
#include "Energies/energy_offsets.h"
#include "SimulationCore/simulation_data.h"
#include <luisa/dsl/builtin.h>

namespace lcs
{
class GroundCollisionEnergy : public Energy
{
  public:
    GroundCollisionEnergy(luisa::compute::BufferView<float>  sa_rest_vert_area,
                          luisa::compute::BufferView<uint>   sa_is_fixed,
                          luisa::compute::BufferView<float>  sa_contact_active_verts_offset,
                          luisa::compute::BufferView<float>  sa_contact_active_verts_d_hat,
                          luisa::compute::BufferView<float>  sa_contact_active_verts_friction_coeff,
                          luisa::compute::BufferView<float3> sa_x_step_start,
                          luisa::compute::BufferView<float>  sa_system_energy) noexcept;

    void compile(AsyncCompiler& compiler) override;

    void device_compute_energy(luisa::compute::Stream& stream) override;
    void device_compute_energy(luisa::compute::Stream&               stream,
                               const luisa::compute::Buffer<float3>& sa_x,
                               float                                 floor_y,
                               bool                                  use_ground_collision,
                               float                                 stiffness,
                               uint                                  collision_type,
                               size_t                                dispatch_count);

    double host_evaluate(const std::vector<float>& host_energy) override;

  private:
    luisa::compute::BufferView<float>  _sa_rest_vert_area;
    luisa::compute::BufferView<uint>   _sa_is_fixed;
    luisa::compute::BufferView<float>  _sa_contact_active_verts_offset;
    luisa::compute::BufferView<float>  _sa_contact_active_verts_d_hat;
    luisa::compute::BufferView<float>  _sa_contact_active_verts_friction_coeff;
    luisa::compute::BufferView<float3> _sa_x_step_start;
    luisa::compute::BufferView<float>  _sa_system_energy;

    luisa::compute::Shader<1, luisa::compute::BufferView<float3>, float, bool, float, uint> _shader;
};

}  // namespace lcs
