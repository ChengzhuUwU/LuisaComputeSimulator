#include "stretch_face_energy.h"
#include "Utils/reduce_helper.h"

using namespace luisa::compute;

namespace lcs
{
StretchFaceEnergy::StretchFaceEnergy(BufferView<float> sa_system_energy) noexcept
    : _sa_system_energy(sa_system_energy)
{
}

void StretchFaceEnergy::compile(AsyncCompiler& compiler)
{
    luisa::compute::ShaderOption default_option = {.enable_debug_info = false};
    compiler.compile<1>(
        _shader,
        [sa_system_energy = _sa_system_energy](Var<Constitutions::StretchFace<luisa::compute::Buffer>> constraint,
                                               Var<BufferView<float3>> sa_x)
        {
            auto& sa_faces                   = constraint.constraint_indices;
            auto& sa_stretch_faces_rest_area = constraint.sa_stretch_faces_rest_area;
            auto& sa_stretch_faces_Dm_inv    = constraint.sa_stretch_faces_Dm_inv;
            auto& sa_stretch_faces_mu_lambda = constraint.sa_stretch_faces_mu_lambda;

            const Uint fid    = dispatch_id().x;
            Float      energy = 0.0f;
            {
                const Uint3 face   = sa_faces->read(fid);
                Float3 vert_pos[3] = {sa_x->read(face[0]), sa_x->read(face[1]), sa_x->read(face[2])};

                Float2x2 Dm_inv = sa_stretch_faces_Dm_inv->read(fid);
                Float    area   = sa_stretch_faces_rest_area->read(fid);

                Float2 mu_lambda    = sa_stretch_faces_mu_lambda->read(fid);
                Float  mu_cloth     = mu_lambda[0];
                Float  lambda_cloth = mu_lambda[1];

                energy = StretchEnergy::compute_energy(
                    vert_pos[0], vert_pos[1], vert_pos[2], Dm_inv, mu_cloth, lambda_cloth, area);
            };

            energy = ParallelIntrinsic::block_intrinsic_reduce(fid, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
            $if(fid % 256 == 0)
            {
                sa_system_energy->atomic(offset_stretch_face).fetch_add(energy);
            };
        },
        default_option);
}

void StretchFaceEnergy::device_compute_energy(luisa::compute::Stream& stream)
{
}

void StretchFaceEnergy::device_compute_energy(luisa::compute::Stream& stream,
                                              const Constitutions::StretchFace<luisa::compute::Buffer>& constraint,
                                              const luisa::compute::Buffer<float3>& sa_x,
                                              size_t                                dispatch_count)
{
    stream << _shader(constraint, sa_x).dispatch(dispatch_count);
}

double StretchFaceEnergy::host_evaluate(const std::vector<float>& host_energy)
{
    return host_energy[offset_stretch_face];
}

}  // namespace lcs
