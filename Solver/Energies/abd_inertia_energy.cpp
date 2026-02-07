#include "abd_inertia_energy.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/scene_params.h"
#include "Utils/cpu_parallel.h"
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

    // gradient/hessian evaluate shader for ABD inertia
    compiler.compile<1>(
        _eval_shader,
        [sa_q_tilde = _sa_q_tilde](Var<Constitutions::AbdInertia<luisa::compute::Buffer>> constraint,
                                   Var<BufferView<float3>>                                sa_q,
                                   Float                                                  substep_dt)
        {
            auto& sa_affine_bodies       = constraint.constraint_indices;
            auto& sa_vert_mass           = constraint.sa_affine_bodies_mass_matrix;
            auto& sa_stiffness_dirichlet = constraint.sa_stiffness_dirichlet;

            const UInt  body_idx    = dispatch_id().x;
            const UInt4 affine_body = sa_affine_bodies->read(body_idx);

            const Float h       = substep_dt;
            const Float h_2_inv = 1.0f / (h * h);

            Float3 delta[4] = {sa_q->read(affine_body[0]) - sa_q_tilde->read(affine_body[0]),
                               sa_q->read(affine_body[1]) - sa_q_tilde->read(affine_body[1]),
                               sa_q->read(affine_body[2]) - sa_q_tilde->read(affine_body[2]),
                               sa_q->read(affine_body[3]) - sa_q_tilde->read(affine_body[3])};

            Float4x4 mass_matrix = sa_vert_mass->read(body_idx);
            Float3   gradient[4] = {Zero3, Zero3, Zero3, Zero3};

            for (uint ii = 0; ii < 4; ii++)
            {
                for (uint jj = 0; jj < 4; jj++)
                {
                    gradient[ii] += mass_matrix[ii][jj] * delta[jj];
                }
            }

            auto& abd_gradients = constraint.constraint_gradients;
            auto& abd_hessians  = constraint.constraint_hessians;

            abd_gradients->write(4 * body_idx + 0, h_2_inv * gradient[0]);
            abd_gradients->write(4 * body_idx + 1, h_2_inv * gradient[1]);
            abd_gradients->write(4 * body_idx + 2, h_2_inv * gradient[2]);
            abd_gradients->write(4 * body_idx + 3, h_2_inv * gradient[3]);

            abd_hessians->write(16 * body_idx + 0, h_2_inv * mass_matrix[0][0] * make_float3x3(1.0f));
            abd_hessians->write(16 * body_idx + 1, h_2_inv * mass_matrix[1][1] * make_float3x3(1.0f));
            abd_hessians->write(16 * body_idx + 2, h_2_inv * mass_matrix[2][2] * make_float3x3(1.0f));
            abd_hessians->write(16 * body_idx + 3, h_2_inv * mass_matrix[3][3] * make_float3x3(1.0f));

            uint idx = 4;
            for (uint ii = 0; ii < 4; ii++)
            {
                for (uint jj = 0; jj < 4; jj++)
                {
                    if (ii != jj)
                    {
                        abd_hessians->write(body_idx * 16 + idx,
                                            h_2_inv * mass_matrix[ii][jj] * make_float3x3(1.0f));
                        idx += 1;
                    }
                }
            }
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

void AbdInertiaEnergy::device_evaluate(luisa::compute::Stream& stream,
                                       const Constitutions::AbdInertia<luisa::compute::Buffer>& constraint,
                                       const luisa::compute::Buffer<float3>& sa_q,
                                       float                                 substep_dt,
                                       size_t                                dispatch_count)
{
    stream << _eval_shader(constraint, sa_q.view(), substep_dt).dispatch(dispatch_count);
}

double AbdInertiaEnergy::host_evaluate(const std::vector<float>& host_energy)
{
    return host_energy[offset_abd_inertia];
}

void AbdInertiaEnergy::host_evaluate(lcs::SimulationData<std::vector>& host_sim_data,
                                     lcs::MeshData<std::vector>& /*host_mesh_data*/)
{
    auto& abd_data = host_sim_data.get_abd_inertia_data();

    if (abd_data.is_valid())
    {
        CpuParallel::parallel_for(
            0,
            abd_data.get_num_indices(),
            [abd_gradients           = std::span(abd_data.constraint_gradients),
             abd_hessians            = std::span(abd_data.constraint_hessians),
             abd_indices             = std::span(abd_data.constraint_indices),
             abd_mass_matrix         = std::span(abd_data.sa_affine_bodies_mass_matrix),
             abd_stiffness_dirichlet = std::span(abd_data.sa_stiffness_dirichlet),
             abd_q                   = std::span(host_sim_data.sa_q),
             abd_q_tilde             = std::span(host_sim_data.sa_q_tilde)](const uint body_idx)
            {
                const float substep_dt = get_scene_params().get_substep_dt();
                const float h          = substep_dt;
                const float h_2_inv    = 1.f / (h * h);

                const uint4 indices = abd_indices[body_idx];

                float3   delta_q[4]  = {abd_q[indices[0]] - abd_q_tilde[indices[0]],
                                        abd_q[indices[1]] - abd_q_tilde[indices[1]],
                                        abd_q[indices[2]] - abd_q_tilde[indices[2]],
                                        abd_q[indices[3]] - abd_q_tilde[indices[3]]};
                float4x4 mass_matrix = abd_mass_matrix[body_idx];
                float3   gradient[4] = {Zero3, Zero3, Zero3, Zero3};

                // apply dirichlet stiffness if present
                {
                    mass_matrix = abd_stiffness_dirichlet[body_idx] * mass_matrix;
                }

                for (uint ii = 0; ii < 4; ii++)
                {
                    for (uint jj = 0; jj < 4; jj++)
                    {
                        gradient[ii] += mass_matrix[ii][jj] * delta_q[jj];
                    }
                }

                abd_gradients[4 * body_idx + 0] = h_2_inv * gradient[0];
                abd_gradients[4 * body_idx + 1] = h_2_inv * gradient[1];
                abd_gradients[4 * body_idx + 2] = h_2_inv * gradient[2];
                abd_gradients[4 * body_idx + 3] = h_2_inv * gradient[3];

                abd_hessians[16 * body_idx + 0] = float3x3::eye(h_2_inv * mass_matrix[0][0]);
                abd_hessians[16 * body_idx + 1] = float3x3::eye(h_2_inv * mass_matrix[1][1]);
                abd_hessians[16 * body_idx + 2] = float3x3::eye(h_2_inv * mass_matrix[2][2]);
                abd_hessians[16 * body_idx + 3] = float3x3::eye(h_2_inv * mass_matrix[3][3]);

                uint idx = 4;
                for (uint ii = 0; ii < 4; ii++)
                {
                    for (uint jj = 0; jj < 4; jj++)
                    {
                        if (ii != jj)
                        {
                            abd_hessians[body_idx * 16 + idx] = float3x3::eye(h_2_inv * mass_matrix[ii][jj]);
                            idx += 1;
                        }
                    }
                }
            },
            32);
    }
}

}  // namespace lcs
