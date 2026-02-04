#include "ground_collision_energy.h"
#include "CollisionDetector/cipc_kernel.hpp"
#include "CollisionDetector/friction_kernel.hpp"

using namespace luisa::compute;

namespace lcs
{
GroundCollisionEnergy::GroundCollisionEnergy(BufferView<float>  sa_rest_vert_area,
                                             BufferView<uint>   sa_is_fixed,
                                             BufferView<float>  sa_contact_active_verts_offset,
                                             BufferView<float>  sa_contact_active_verts_d_hat,
                                             BufferView<float>  sa_contact_active_verts_friction_coeff,
                                             BufferView<float3> sa_x_step_start,
                                             BufferView<float>  sa_system_energy) noexcept
    : _sa_rest_vert_area(sa_rest_vert_area)
    , _sa_is_fixed(sa_is_fixed)
    , _sa_contact_active_verts_offset(sa_contact_active_verts_offset)
    , _sa_contact_active_verts_d_hat(sa_contact_active_verts_d_hat)
    , _sa_contact_active_verts_friction_coeff(sa_contact_active_verts_friction_coeff)
    , _sa_x_step_start(sa_x_step_start)
    , _sa_system_energy(sa_system_energy)
{
}

void GroundCollisionEnergy::compile(AsyncCompiler& compiler)
{
    luisa::compute::ShaderOption default_option = {.enable_debug_info = false};
    compiler.compile<1>(
        _shader,
        [sa_rest_vert_area                      = _sa_rest_vert_area,
         sa_is_fixed                            = _sa_is_fixed,
         sa_contact_active_verts_offset         = _sa_contact_active_verts_offset,
         sa_contact_active_verts_d_hat          = _sa_contact_active_verts_d_hat,
         sa_contact_active_verts_friction_coeff = _sa_contact_active_verts_friction_coeff,
         sa_x_step_start                        = _sa_x_step_start,
         sa_system_energy                       = _sa_system_energy](
            Var<BufferView<float3>> sa_x, Float floor_y, Bool use_ground_collision, Float stiffness, Uint collision_type)
        {
            const Uint vid = dispatch_id().x;

            Float energy_repulsive = 0.0f;
            Float energy_friction  = 0.0f;
            Bool  is_fixed         = sa_is_fixed->read(vid) != 0;

            $if(use_ground_collision & !is_fixed)
            {
                Float d_hat     = sa_contact_active_verts_d_hat->read(vid);
                Float thickness = sa_contact_active_verts_offset->read(vid);
                Float area      = sa_rest_vert_area->read(vid);
                Float stiff     = stiffness * area;

                Float3 normal = make_float3(0.0f, 1.0f, 0.0f);

                Float3 x_k = sa_x->read(vid);
                Float3 x_0 = sa_x_step_start->read(vid);

                Float curr_dist = x_k.y - floor_y;
                $if(curr_dist - thickness < d_hat)
                {
                    $if(collision_type == 0)
                    {
                        Float C          = curr_dist - d_hat - thickness;
                        energy_repulsive = 0.5f * stiff * C * C;
                    }
                    $else
                    {
                        energy_repulsive = stiff * ipc::barrier(curr_dist - thickness, d_hat);
                    };
                };

                Float init_dist = x_0.y - floor_y;
                $if(init_dist - thickness < d_hat)
                {
                    Float3 rel_dx = x_k - x_0;

                    Float k1 = 0.0f;
                    $if(collision_type == 0)
                    {
                        Float C = init_dist - thickness - d_hat;
                        k1      = stiff * C;
                    }
                    $else
                    {
                        k1 = stiff * ipc::barrier_first_derivative(init_dist - thickness, d_hat);
                    };

                    Float friction_mu  = sa_contact_active_verts_friction_coeff->read(vid);
                    Float friction_eps = Friction::ando_barrier::friction_eps;

                    auto lambda = -k1 * friction_mu;
                    energy_friction =
                        Friction::ipc_barrier::compute_friction_energy(lambda, normal, rel_dx, friction_eps);
                };
            };

            Float2 energy =
                ParallelIntrinsic::block_intrinsic_reduce(vid,
                                                          make_float2(energy_repulsive, energy_friction),
                                                          ParallelIntrinsic::warp_reduce_op_sum<float2>);
            $if(vid % 256 == 0)
            {
                sa_system_energy->atomic(offset_ground_collision).fetch_add(energy.x);
                sa_system_energy->atomic(offset_ground_friction).fetch_add(energy.y);
            };
        },
        default_option);
}

void GroundCollisionEnergy::device_compute_energy(luisa::compute::Stream& stream)
{
    // left empty — use overload below with explicit args
}

void GroundCollisionEnergy::device_compute_energy(luisa::compute::Stream&               stream,
                                                  const luisa::compute::Buffer<float3>& sa_x,
                                                  float                                 floor_y,
                                                  bool   use_ground_collision,
                                                  float  stiffness,
                                                  uint   collision_type,
                                                  size_t dispatch_count)
{
    stream << _shader(sa_x, floor_y, use_ground_collision, stiffness, collision_type).dispatch(dispatch_count);
}

double GroundCollisionEnergy::host_evaluate(const std::vector<float>& host_energy)
{
    return host_energy[offset_ground_collision];
}

}  // namespace lcs
