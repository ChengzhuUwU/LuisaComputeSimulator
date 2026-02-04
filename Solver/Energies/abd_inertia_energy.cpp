#include "abd_inertia_energy.h"
#include "Utils/reduce_helper.h"

using namespace luisa::compute;

namespace lcs
{
AbdInertiaEnergy::AbdInertiaEnergy(BufferView<float3> sa_q_tilde, BufferView<float> sa_system_energy) noexcept
    : _sa_q_tilde(sa_q_tilde)
    , _sa_system_energy(sa_system_energy)
{
}

void AbdInertiaEnergy::compile(AsyncCompiler& compiler)
{
    luisa::compute::ShaderOption default_option = {.enable_debug_info = false};
    compiler.compile<1>(
        _shader,
        [sa_q_tilde = _sa_q_tilde, sa_system_energy = _sa_system_energy](
            Var<Constitutions::AbdInertia<luisa::compute::Buffer>> constraint, Var<BufferView<float3>> sa_q, Float substep_dt)
        {
            auto& sa_affine_bodies       = constraint.constraint_indices;
            auto& sa_vert_mass           = constraint.sa_affine_bodies_mass_matrix;
            auto& sa_stiffness_dirichlet = constraint.sa_stiffness_dirichlet;

            const Uint  body_idx    = dispatch_id().x;
            const Uint4 affine_body = sa_affine_bodies->read(body_idx);

            Float energy = 0.0f;
            {
                const Float h                   = substep_dt;
                const Float squared_inv_dt      = 1.0f / (h * h);
                Float       stiffness_dirichlet = sa_stiffness_dirichlet->read(body_idx);

                auto   mass_matrix = sa_vert_mass->read(body_idx);
                Float3 delta[4]    = {
                    sa_q.read(affine_body[0]) - sa_q_tilde->read(affine_body[0]),
                    sa_q.read(affine_body[1]) - sa_q_tilde->read(affine_body[1]),
                    sa_q.read(affine_body[2]) - sa_q_tilde->read(affine_body[2]),
                    sa_q.read(affine_body[3]) - sa_q_tilde->read(affine_body[3]),
                };

                for (uint ii = 0; ii < 4; ii++)
                {
                    for (uint jj = 0; jj < 4; jj++)
                    {
                        Float mass = mass_matrix[ii][jj];
                        energy += squared_inv_dt * dot(delta[ii], delta[jj]) * mass / (2.0f);
                    }
                }

                energy *= stiffness_dirichlet;
            };

            energy = ParallelIntrinsic::block_intrinsic_reduce(
                body_idx, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
            $if(body_idx % 256 == 0)
            {
                sa_system_energy->atomic(offset_abd_inertia).fetch_add(energy);
            };
        },
        default_option);
}

void AbdInertiaEnergy::device_compute_energy(luisa::compute::Stream& stream)
{
    // Caller should use the typed overload below to dispatch with the appropriate buffers and counts.
}

void AbdInertiaEnergy::device_compute_energy(luisa::compute::Stream& stream,
                                             const Constitutions::AbdInertia<luisa::compute::Buffer>& constraint,
                                             const luisa::compute::Buffer<float3>& sa_q,
                                             float                                 substep_dt,
                                             size_t                                dispatch_count)
{
    stream << _shader(constraint, sa_q.view(), substep_dt).dispatch(dispatch_count);
}

double AbdInertiaEnergy::host_evaluate(const std::vector<float>& host_energy)
{
    return host_energy[offset_abd_inertia];
}

}  // namespace lcs
#include "abd_inertia_energy.h"

// Implementation file left intentionally minimal. Kernel functor is header-only.
