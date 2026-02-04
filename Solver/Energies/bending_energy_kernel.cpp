#include "bending_energy_kernel.h"
#include "Utils/reduce_helper.h"

using namespace luisa::compute;

namespace lcs
{
BendingEnergy::BendingEnergy(BufferView<float> sa_system_energy) noexcept
    : _sa_system_energy(sa_system_energy)
{
}

void BendingEnergy::compile(AsyncCompiler& compiler)
{
    luisa::compute::ShaderOption default_option = {.enable_debug_info = false};
    compiler.compile<1>(
        _shader,
        [sa_system_energy = _sa_system_energy](Var<Constitutions::BendingEdge<luisa::compute::Buffer>> constraint,
                                               Var<BufferView<float3>> sa_x,
                                               Float                   scaling)
        {
            auto& sa_edges                    = constraint.constraint_indices;
            auto& sa_bending_edges_rest_angle = constraint.sa_bending_edges_rest_angle;
            auto& sa_bending_edges_rest_area  = constraint.sa_bending_edges_rest_area;
            auto& sa_bending_edges_stiffness  = constraint.sa_bending_edges_stiffness;

            const Uint eid    = dispatch_id().x;
            Float      energy = 0.0f;
            {
                const Uint4 edge = sa_edges->read(eid);

                Float3 vert_pos[4] = {sa_x.read(edge[0]), sa_x.read(edge[1]), sa_x.read(edge[2]), sa_x.read(edge[3])};
                Float rest_angle = sa_bending_edges_rest_angle->read(eid);
                Float angle =
                    BendingEnergyUtils::compute_theta(vert_pos[0], vert_pos[1], vert_pos[2], vert_pos[3]);
                Float delta_angle = angle - rest_angle;
                Float area        = sa_bending_edges_rest_area->read(eid);
                energy = 0.5f * sa_bending_edges_stiffness->read(eid) * scaling * area * delta_angle * delta_angle;
            };

            energy = ParallelIntrinsic::block_intrinsic_reduce(eid, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
            $if(eid % 256 == 0)
            {
                sa_system_energy->atomic(offset_bending).fetch_add(energy);
            };
        },
        default_option);
}

void BendingEnergy::device_compute_energy(luisa::compute::Stream& stream)
{
}

void BendingEnergy::device_compute_energy(luisa::compute::Stream& stream,
                                          const Constitutions::BendingEdge<luisa::compute::Buffer>& constraint,
                                          const luisa::compute::Buffer<float3>& sa_x,
                                          float                                 scaling,
                                          size_t                                dispatch_count)
{
    stream << _shader(constraint, sa_x, scaling).dispatch(dispatch_count);
}

double BendingEnergy::host_evaluate(const std::vector<float>& host_energy)
{
    return host_energy[offset_bending];
}

}  // namespace lcs
