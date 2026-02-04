#include "inertia_energy.h"
#include "Utils/reduce_helper.h"

using namespace luisa::compute;

namespace lcs
{
InertiaEnergy::InertiaEnergy(BufferView<float3> sa_q_tilde_view, BufferView<float> sa_system_energy_view) noexcept
    : _sa_q_tilde_view(sa_q_tilde_view)
    , _sa_system_energy_view(sa_system_energy_view)
{
}

void InertiaEnergy::compile(AsyncCompiler& compiler)
{
    luisa::compute::ShaderOption default_option = {.enable_debug_info = false};
    compiler.compile<1>(
        _shader,
        [sa_q_tilde = _sa_q_tilde_view, sa_system_energy = _sa_system_energy_view](
            Var<Constitutions::SoftInertia<luisa::compute::Buffer>> constraint, Var<BufferView<float3>> sa_q, Float substep_dt)
        {
            auto& soft_inertia_indices   = constraint.constraint_indices;
            auto& sa_vert_mass           = constraint.sa_soft_vert_mass;
            auto& sa_stiffness_dirichlet = constraint.sa_stiffness_dirichlet;

            const Uint index = dispatch_id().x;
            const Uint vid   = soft_inertia_indices.read(index);

            Float energy = 0.0f;
            {
                Float3      x_new          = sa_q->read(vid);
                Float3      x_tilde        = sa_q_tilde->read(vid);
                Float       mass           = sa_vert_mass->read(vid);
                const Float squared_inv_dt = 1.0f / (substep_dt * substep_dt);
                energy = squared_inv_dt * length_squared_vec(x_new - x_tilde) * mass / (2.0f);
                {
                    Float stiffness_dirichlet = sa_stiffness_dirichlet->read(vid);
                    energy                    = stiffness_dirichlet * energy;
                };
            };

            energy = ParallelIntrinsic::block_intrinsic_reduce(vid, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
            $if(vid % 256 == 0)
            {
                sa_system_energy->atomic(offset_inertia).fetch_add(energy);
            };
        },
        default_option);
}

void InertiaEnergy::device_compute_energy(luisa::compute::Stream& stream)
{
    // This class does not know which constitution to dispatch; caller should use the stored _shader directly.
    // Left intentionally empty — caller will dispatch using shader member exposed via friend or directly if needed.
}

void InertiaEnergy::device_compute_energy(luisa::compute::Stream& stream,
                                          const Constitutions::SoftInertia<luisa::compute::Buffer>& constraint,
                                          const luisa::compute::Buffer<float3>& sa_q,
                                          float                                 substep_dt,
                                          size_t                                dispatch_count)
{
    stream << _shader(constraint, sa_q.view(), substep_dt).dispatch(dispatch_count);
}

double InertiaEnergy::host_evaluate(const std::vector<float>& host_energy)
{
    return host_energy[offset_inertia];
}

}  // namespace lcs
