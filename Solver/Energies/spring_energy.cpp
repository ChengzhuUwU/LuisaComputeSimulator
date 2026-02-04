#include "spring_energy.h"

using namespace luisa::compute;

namespace lcs
{
SpringEnergy::SpringEnergy(BufferView<float> sa_system_energy) noexcept
    : _sa_system_energy(sa_system_energy)
{
}

void SpringEnergy::compile(AsyncCompiler& compiler)
{
    luisa::compute::ShaderOption default_option = {.enable_debug_info = false};
    compiler.compile<1>(
        _shader,
        [sa_system_energy = _sa_system_energy](Var<Constitutions::StretchSpring<luisa::compute::Buffer>> constraint,
                                               Var<BufferView<float3>> sa_x,
                                               Float                   stiffness_spring)
        {
            auto& sa_edges                    = constraint.constraint_indices;
            auto& sa_edge_rest_state_length   = constraint.sa_stretch_spring_rest_state_length;
            auto& sa_stretch_spring_stiffness = constraint.sa_stretch_spring_stiffness;

            const Uint eid    = dispatch_id().x;
            Float      energy = 0.0f;
            {
                const Uint2 edge             = sa_edges->read(eid);
                const Float rest_edge_length = sa_edge_rest_state_length->read(eid);
                Float3      diff             = sa_x->read(edge[1]) - sa_x->read(edge[0]);
                Float       orig_lengthsqr   = length_squared_vec(diff);
                Float       l                = sqrt_scalar(orig_lengthsqr);
                Float       l0               = rest_edge_length;
                Float       C                = l - l0;
                energy                       = 0.5f * sa_stretch_spring_stiffness->read(eid) * C * C;
            };

            energy = ParallelIntrinsic::block_intrinsic_reduce(eid, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
            $if(eid % 256 == 0)
            {
                sa_system_energy->atomic(offset_stretch_spring).fetch_add(energy);
            };
        },
        default_option);
}

void SpringEnergy::device_compute_energy(luisa::compute::Stream& stream)
{
}

void SpringEnergy::device_compute_energy(luisa::compute::Stream& stream,
                                         const Constitutions::StretchSpring<luisa::compute::Buffer>& constraint,
                                         const luisa::compute::Buffer<float3>& sa_x,
                                         float                                 stiffness_spring,
                                         size_t                                dispatch_count)
{
    stream << _shader(constraint, sa_x, stiffness_spring).dispatch(dispatch_count);
}

double SpringEnergy::host_evaluate(const std::vector<float>& host_energy)
{
    return host_energy[offset_stretch_spring];
}

}  // namespace lcs
