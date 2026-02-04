#include "abd_ortho_energy.h"
#include "Utils/reduce_helper.h"

using namespace luisa::compute;

namespace lcs
{
AbdOrthoEnergy::AbdOrthoEnergy(BufferView<float> sa_system_energy, BufferView<float3> sa_q) noexcept
    : _sa_system_energy(sa_system_energy)
    , _sa_q(sa_q)
{
}

void AbdOrthoEnergy::compile(AsyncCompiler& compiler)
{
    luisa::compute::ShaderOption default_option = {.enable_debug_info = false};
    compiler.compile<1>(
        _shader,
        [sa_system_energy = _sa_system_energy, sa_q = _sa_q](
            Var<Constitutions::AbdOrthogonality<luisa::compute::Buffer>> constraint, Var<BufferView<float3>> sa_q_view)
        {
            using namespace luisa::compute;
            auto& abd_ortho_indices = constraint.constraint_indices;
            auto& abd_kappa         = constraint.abd_kappa;
            auto& abd_volume        = constraint.abd_volume;

            const Uint body_idx = dispatch_id().x;

            const Uint3 indices = abd_ortho_indices->read(body_idx);

            Float energy = 0.0f;
            {
                Float3x3 A;
                A[0] = sa_q_view->read(indices[0]);
                A[1] = sa_q_view->read(indices[1]);
                A[2] = sa_q_view->read(indices[2]);
                for (uint ii = 0; ii < 3; ii++)
                {
                    for (uint jj = 0; jj < 3; jj++)
                    {
                        Float term = dot(A[ii], A[jj]) - (ii == jj ? 1.0f : 0.0f);
                        energy += term * term;
                    }
                }
                Float stiffness_ortho = abd_kappa->read(body_idx);
                Float volume          = abd_volume->read(body_idx);
                energy *= stiffness_ortho * volume;
            };

            energy = ParallelIntrinsic::block_intrinsic_reduce(
                body_idx, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
            $if(body_idx % 256 == 0)
            {
                sa_system_energy->atomic(offset_abd_ortho).fetch_add(energy);
            };
        },
        default_option);
}

void AbdOrthoEnergy::device_compute_energy(luisa::compute::Stream& stream)
{
    // left empty — use typed overload below
}

void AbdOrthoEnergy::device_compute_energy(luisa::compute::Stream& stream,
                                           const Constitutions::AbdOrthogonality<luisa::compute::Buffer>& constraint,
                                           const luisa::compute::Buffer<float3>& sa_q,
                                           size_t                                dispatch_count)
{
    stream << _shader(constraint, sa_q.view()).dispatch(dispatch_count);
}

double AbdOrthoEnergy::host_evaluate(const std::vector<float>& host_energy)
{
    return host_energy[offset_abd_ortho];
}

}  // namespace lcs
#include "abd_ortho_energy.h"

// Implementation file left intentionally minimal. Kernel functor is header-only.
