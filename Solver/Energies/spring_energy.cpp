#include "spring_energy.h"
#include "SimulationCore/base_mesh.h"
#include "Utils/cpu_parallel.h"

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

void SpringEnergy::host_evaluate(lcs::SimulationData<std::vector>& host_sim_data, lcs::MeshData<std::vector>& host_mesh_data)
{
    auto& stretch_springs = host_sim_data.get_stretch_spring_data();

    CpuParallel::parallel_for(0,
                              stretch_springs.get_num_indices(),
                              [sa_x     = std::span(host_sim_data.sa_x),
                               sa_edges = std::span(stretch_springs.constraint_indices),
                               sa_rest_length = std::span(stretch_springs.sa_stretch_spring_rest_state_length),
                               sa_stretch_spring_stiffness = std::span(stretch_springs.sa_stretch_spring_stiffness),
                               output_gradient_ptr = std::span(stretch_springs.constraint_gradients),
                               output_hessian_ptr = std::span(stretch_springs.constraint_hessians)](const uint eid)
                              {
                                  uint2 edge = sa_edges[eid];

                                  float3   vert_pos[2]  = {sa_x[edge[0]], sa_x[edge[1]]};
                                  float3   gradients[2] = {Zero3, Zero3};
                                  float3x3 He           = luisa::make_float3x3(0.0f);

                                  const float L = sa_rest_length[eid];
                                  const float stiffness_stretch_spring = sa_stretch_spring_stiffness[eid];

                                  float3 diff = vert_pos[0] - vert_pos[1];
                                  float  l    = max_scalar(length_vec(diff), Epsilon);
                                  float  l0   = L;
                                  float  C    = l - l0;

                                  float3   dir   = diff / l;
                                  float3x3 nnT   = outer_product(dir, dir);
                                  float    x_inv = 1.f / l;

                                  gradients[0] = stiffness_stretch_spring * dir * C;
                                  gradients[1] = -gradients[0];
                                  He           = stiffness_stretch_spring * nnT
                                       + stiffness_stretch_spring * max_scalar(1.0f - L * x_inv, 0.0f)
                                             * (luisa::make_float3x3(1.0f) - nnT);

                                  output_gradient_ptr[eid * 2 + 0] = gradients[0];
                                  output_gradient_ptr[eid * 2 + 1] = gradients[1];

                                  output_hessian_ptr[eid * 4 + 0] = He;
                                  output_hessian_ptr[eid * 4 + 1] = He;
                                  output_hessian_ptr[eid * 4 + 2] = -1.0f * He;
                                  output_hessian_ptr[eid * 4 + 3] = -1.0f * He;
                              });
}

}  // namespace lcs
