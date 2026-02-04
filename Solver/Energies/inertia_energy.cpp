#include "inertia_energy.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/scene_params.h"
#include "Utils/cpu_parallel.h"
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

void InertiaEnergy::host_evaluate(lcs::SimulationData<std::vector>& host_sim_data, lcs::MeshData<std::vector>& host_mesh_data)
{
    auto& inertia_data = host_sim_data.get_soft_inertia_data();
    if (inertia_data.is_valid())
    {
        CpuParallel::parallel_for(0,
                                  host_sim_data.num_verts_soft,
                                  [sa_x          = std::span(host_sim_data.sa_x),
                                   sa_x_tilde    = std::span(host_sim_data.sa_q_tilde),
                                   sa_q_is_fixed = std::span(host_sim_data.sa_q_is_fixed),
                                   sa_vert_mass  = std::span(inertia_data.sa_soft_vert_mass),
                                   sa_stiffness_dirichlet = std::span(inertia_data.sa_stiffness_dirichlet),
                                   output_gradient = std::span(inertia_data.constraint_gradients),
                                   output_hessian  = std::span(inertia_data.constraint_hessians),
                                   substep_dt      = get_scene_params().get_substep_dt()](const uint vid)
                                  {
                                      const float h       = substep_dt;
                                      const float h_2_inv = 1.f / (h * h);

                                      float3 x_k     = sa_x[vid];
                                      float3 x_tilde = sa_x_tilde[vid];

                                      float    mass     = sa_vert_mass[vid];
                                      float3   gradient = mass * h_2_inv * (x_k - x_tilde);
                                      float3x3 hessian  = mass * h_2_inv * luisa::float3x3::eye(1.0f);

                                      {
                                          const float stiffness_dirichlet = sa_stiffness_dirichlet[vid];
                                          gradient = stiffness_dirichlet * gradient;
                                          hessian  = stiffness_dirichlet * hessian;
                                      }
                                      {
                                          output_gradient[vid] = gradient;
                                          output_hessian[vid]  = hessian;
                                      }
                                  });
    }

    auto& abd_data = host_sim_data.get_abd_inertia_data();

    if (abd_data.is_valid())
    {
        const uint prefix = host_sim_data.num_verts_soft;
        CpuParallel::parallel_for(
            0,
            abd_data.get_num_indices(),
            [abd_gradients           = std::span(abd_data.constraint_gradients),
             abd_hessians            = std::span(abd_data.constraint_hessians),
             abd_indices             = std::span(abd_data.constraint_indices),
             abd_mass_matrix         = std::span(abd_data.sa_affine_bodies_mass_matrix),
             abd_stiffness_dirichlet = std::span(abd_data.sa_stiffness_dirichlet),
             abd_q                   = std::span(host_sim_data.sa_q),
             sa_q_is_fixed           = std::span(host_sim_data.sa_q_is_fixed),
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
