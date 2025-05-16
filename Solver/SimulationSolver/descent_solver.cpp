#include "SimulationSolver/descent_solver.h"
#include "Core/float_nxn.h"
#include "Core/scalar.h"
#include "Core/xbasic_types.h"
#include "SimulationCore/scene_params.h"
#include "Utils/cpu_parallel.h"
#include <luisa/dsl/sugar.h>

namespace lcsv
{

void DescentSolverCPU::reset_constrains(luisa::compute::Stream& stream)
{
    // auto fn_reset_template = [&](Buffer<float>& buffer)
    // {
    //     parallel_set(
    //         buffer.data(), 
    //         buffer.size(), 
    //         0.0f);
    // };

    // fn_reset_template(xpbd_data->sa_lambda_stretch_mass_spring);
    // fn_reset_template(xpbd_data->sa_lambda_bending);
}
void DescentSolverCPU::reset_collision_constrains(luisa::compute::Stream& stream)
{

}


/*
auto shader_predict_position = device.compile<1>([&]
{
    const UInt vid = dispatch_id().x;
    const Float3 gravity(0, -9.8f, 0);
    Float3 x_prev = xpbd_data->sa_x_start->read(vid);
    Float3 v_prev = xpbd_data->sa_v->read(vid);
    Float3 outer_acceleration = gravity; 
    const float substep_dt = lcsv::get_scene_params().get_substep_dt();
    Float3 v_pred = v_prev + substep_dt * outer_acceleration; 
    $if(mesh_data->sa_is_fixed->read(vid) != 0) { outer_acceleration = Zero3; v_pred = Zero3; };
    const Float3 x_pred = x_prev + substep_dt * v_pred;
    xpbd_data->sa_x->write(vid, x_pred);
}, {
    .enable_debug_info = use_debug_info
});

luisa::compute::Shader<1, luisa::compute::Buffer<luisa::Vector<float, 3>>,
                    luisa::compute::Buffer<luisa::Vector<float, 3>>,
                    luisa::compute::Buffer<luisa::Vector<float, 3>>, bool,
                    luisa::compute::Buffer<luisa::Vector<float, 3>>,
                    luisa::compute::Buffer<float>,
                    luisa::compute::Buffer<unsigned int>, float, bool> fn_predict_position;

fn_predict_position = device.compile<1>([&](
    PTR(float3) sa_iter_position, PTR(float3) sa_vert_velocity, PTR(float3) sa_iter_start_position, 
    const Bool predict_for_collision, PTR(float3) sa_next_position,
    PTR(float) sa_vert_mass, 
    PTR(uint) sa_is_fixed,
    const Float substep_dt,
    const Bool fix_scene)
{
    const Uint vid = dispatch_id().x;
    const Float3 gravity = make_float3(0, -9.8f, 0);
    const Float3 x_prev = sa_iter_start_position.read(vid);
    const Float3 v_prev = sa_vert_velocity.read(vid);
    Float3 outer_acceleration = gravity; 
    Float3 v_pred = v_prev + substep_dt * outer_acceleration; 
    $if (sa_is_fixed.read(vid) != 0) { outer_acceleration = Zero3; v_pred = Zero3; };
    const Float3 x_pred = x_prev + substep_dt * v_pred;
    sa_iter_position.write(vid, x_pred);
});
*/

void DescentSolverCPU::compile(luisa::compute::Device& device)
{
    const bool use_debug_info = false;
    using namespace luisa::compute;

    luisa::compute::ShaderOption default_option = {.enable_debug_info = false};

    luisa::compute::Buffer<float4x3> aaaa;
    auto makeHf = [](const Float3& force, const Float3x3& hessian) 
    {
        return Float4x3{force, hessian[0], hessian[1], hessian[2]};
    };

    auto writeHf = [](const Float3& force, const Float3x3& hessian, BufferView<float> sa_Hf, const Uint vid) 
    {
        const Uint offset = vid * 12;
        sa_Hf->write(offset + 0, force[0]);
        sa_Hf->write(offset + 1, force[1]);
        sa_Hf->write(offset + 2, force[2]);
        sa_Hf->write(offset + 3, hessian[0][0]);
        sa_Hf->write(offset + 4, hessian[0][1]);
        sa_Hf->write(offset + 5, hessian[0][2]);
        sa_Hf->write(offset + 6, hessian[1][0]);
        sa_Hf->write(offset + 7, hessian[1][1]);
        sa_Hf->write(offset + 8, hessian[1][2]);
        sa_Hf->write(offset + 9, hessian[2][0]);
        sa_Hf->write(offset + 10, hessian[2][1]);
        sa_Hf->write(offset + 11, hessian[2][2]);
    };
    auto extractHf = [](Float3& force, Float3x3& hessian, BufferView<float> sa_Hf, const Uint vid)
    {
        const Uint offset = vid * 12;
        force[0] = sa_Hf->read(offset + 0);
        force[1] = sa_Hf->read(offset + 1);
        force[2] = sa_Hf->read(offset + 2);
        hessian[0][0] = sa_Hf->read(offset + 3);
        hessian[0][1] = sa_Hf->read(offset + 4);
        hessian[0][2] = sa_Hf->read(offset + 5);
        hessian[1][0] = sa_Hf->read(offset + 6);
        hessian[1][1] = sa_Hf->read(offset + 7);
        hessian[1][2] = sa_Hf->read(offset + 8);
        hessian[2][0] = sa_Hf->read(offset + 9);
        hessian[2][1] = sa_Hf->read(offset + 10);
        hessian[2][2] = sa_Hf->read(offset + 11);
    };

    fn_predict_position = device.compile<1>(
        [
            sa_x = xpbd_data->sa_x.view(),
            sa_v = xpbd_data->sa_v.view(),
            sa_x_start = xpbd_data->sa_x_start.view(),
            sa_is_fixed = mesh_data->sa_is_fixed.view(),
            substep_dt = lcsv::get_scene_params().get_substep_dt()
        ]
    {
        const UInt vid = dispatch_id().x;
        const Float3 gravity(0, -9.8f, 0);
        Float3 x_prev = sa_x_start->read(vid);
        Float3 v_prev = sa_v->read(vid);
        Float3 outer_acceleration = gravity;
        Float3 v_pred = v_prev + substep_dt * outer_acceleration;
        $if(sa_is_fixed->read(vid) != 0) { outer_acceleration = Zero3; v_pred = Zero3; };
        const Float3 x_pred = x_prev + substep_dt * v_pred;
        sa_x->write(vid, x_pred);
    }, default_option);

    fn_update_velocity = device.compile<1>(
        [
            sa_x = xpbd_data->sa_x.view(),
            sa_v = xpbd_data->sa_v.view(),
            sa_iter_start_position = xpbd_data->sa_x_start.view(),
            sa_iter_position = xpbd_data->sa_x.view(),
            sa_velocity_start = xpbd_data->sa_v_start.view(),
            sa_vert_velocity = xpbd_data->sa_v.view(),
            sa_x_start = xpbd_data->sa_x_start.view(),
            substep_dt = lcsv::get_scene_params().get_substep_dt(),
            fix_scene = false,
            damping = get_scene_params().damping_cloth
        ]
        {
            const UInt vid = dispatch_id().x;
            Float3 x_k_init = sa_iter_start_position->read(vid);
            Float3 x_k = sa_iter_position->read(vid);

            Float3 dx = x_k - x_k_init;
            Float3 vel = dx / substep_dt;

            if (fix_scene) 
            {
                dx = Zero3;
                vel = Zero3;
                sa_iter_position->write(vid, sa_x_start->read(vid));
                return;
            }

            vel *= exp(-damping * substep_dt);

            sa_vert_velocity->write(vid, vel);
            sa_velocity_start->write(vid, vel);
            sa_iter_start_position->write(vid, x_k);
        }
    );

    fn_evaluate_inertia = device.compile<1>(
        [
            sa_Hf = xpbd_data->sa_Hf.view(),
            sa_iter_position = xpbd_data->sa_x.view(),
            sa_x_start = xpbd_data->sa_x_start.view(),
            sa_v = xpbd_data->sa_v.view(),
            sa_is_fixed = mesh_data->sa_is_fixed.view(),
            sa_vert_mass = mesh_data->sa_vert_mass.view(),
            substep_dt = lcsv::get_scene_params().get_substep_dt()
        , writeHf]
        {
            const UInt vid = dispatch_id().x;
            const Float3 gravity(0, -9.8f, 0);
            const float h = substep_dt;
            const float h_2_inv = 1.f / (h * h);

            Float3 x_k = sa_iter_position->read(vid);
            Float3 x_0 = sa_x_start->read(vid);
            Float3 v_0 = sa_v->read(vid);

            auto is_fixed = sa_is_fixed->read(vid);
            Float mass = sa_vert_mass->read(vid);
            Float3x3 mat = Identity3x3 * mass * h_2_inv;

            Float3 outer_force = mass * gravity;

            $if (is_fixed != 0)
            {
                mat += Identity3x3 * float(1E9);
            };

            Float3 gradient = -mass * h_2_inv * (x_k - x_0 - v_0 * h) + outer_force;
            writeHf(gradient, mat, sa_Hf, vid);
        }
    );

    auto outer_product = [](const Float3& left, const Float3& right)
    {
        return make_float3x3(
            left * right[0],
            left * right[1],
            left * right[2]
        );
    };

    fn_evaluate_stretch_spring = device.compile<1>(
        [
            sa_Hf = xpbd_data->sa_Hf.view(),
            sa_iter_position = xpbd_data->sa_x.view(),
            sa_vert_adj_edges = mesh_data->sa_vert_adj_edges_csr.view(),
            sa_edges = mesh_data->sa_edges.view(),
            sa_rest_length = mesh_data->sa_edges_rest_state_length.view(),
            stiffness_stretch = get_scene_params().stiffness_spring
        , writeHf, outer_product]
        {
            const Uint vid = dispatch_id().x;
            const Uint curr_prefix = sa_vert_adj_edges->read(vid);
            const Uint next_prefix = sa_vert_adj_edges->read(vid + 1);
            const Uint num_adj = next_prefix - curr_prefix;

            Float3 gradient = make_float3(0.0f);
            Float3x3 hessian = make_float3x3(0.0f);
            $for(j, 0u, num_adj)
            {
                const Uint adj_eid = sa_vert_adj_edges->read(curr_prefix + j);
                Uint2 edge = sa_edges->read(adj_eid);

                Float3 vert_pos[2] = {
                    sa_iter_position->read(edge[0]),
                    sa_iter_position->read(edge[1])
                };
                // Float3 force[2] = {make_float3(0.0f), make_float3(0.0f)};
                Float3 force[2] = {make_float3(0.0f), make_float3(0.0f)};

                const Float L = sa_rest_length->read(adj_eid);
                Float3 diff = vert_pos[1] - vert_pos[0];
                Float l = luisa::compute::max(luisa::compute::length(diff), Epsilon);
                Float C = l - L;

                Float3 dir = diff / l;
                Float3 force_0 = stiffness_stretch * dir * C;
                Float3 force_1 = -force_0;

                Float3x3 xxT = outer_product(diff, diff);
                Float x_inv = 1.f / l;
                Float x_squared_inv = x_inv * x_inv;
                Float3x3 He = stiffness_stretch * x_squared_inv * xxT + 
                                stiffness_stretch * luisa::compute::max(1.0f - L * x_inv, 0.0f) * (Identity3x3 - x_squared_inv * xxT);
                
                gradient += luisa::compute::select(force_0, force_1, vid == edge[0]);
                hessian += He;
            };
            writeHf(gradient, hessian, sa_Hf, vid);
        }
    );
    
    /*
    fn_evaluate_bending = device.compile<1>(
        [
            sa_Hf = xpbd_data->sa_Hf.view(),
            sa_iter_position = xpbd_data->sa_x.view(),
            sa_vert_adj_bending_edges = mesh_data->sa_vert_adj_bending_edges.view(),
            sa_bending_edges = mesh_data->sa_bending_edges.view(),
            sa_bending_edge_Q = mesh_data->sa_bending_edges_Q.view(),
            stiffness_bending = get_scene_params().get_stiffness_quadratic_bending()
        ]
        {
            const UInt vid = dispatch_id().x;
            const uint curr_prefix = sa_vert_adj_bending_edges->read(vid);
            const uint next_prefix = sa_vert_adj_bending_edges->read(vid + 1);
            const uint num_adj_edges = next_prefix - curr_prefix;

            Float4x3 hf = Zero4x3;
            for (uint j = 0; j < num_adj_edges; j++)
            {
                const uint adj_eid = sa_vert_adj_bending_edges->read(curr_prefix + j);
                Int4 edge = sa_bending_edges->read(adj_eid);

                Float3 vert_pos[4] = {
                    sa_iter_position->read(edge[0]),
                    sa_iter_position->read(edge[1]),
                    sa_iter_position->read(edge[2]),
                    sa_iter_position->read(edge[3])
                };

                Float4x4 Q = sa_bending_edge_Q->read(adj_eid);
                const uint offset = vid == edge[0] ? 0 : vid == edge[1] ? 1 : vid == edge[2] ? 2 : vid == edge[3] ? 3 : -1u;

                Float3 force = Zero3;
                Float3x3 hessian = Zero3x3;
                for (uint jj = 0; jj < 4; jj++) 
                {
                    force -= get(Q, offset, jj) * vert_pos[jj];
                }
                force *= stiffness_bending;
                hessian = stiffness_bending * get(Q, offset, offset) * Identity3x3;
                hf += makeHf(force, hessian);
            }
            sa_Hf->write(vid, sa_Hf->read(vid) + hf);
        }
    );
    */

    fn_step = device.compile<1>(
        [
            sa_Hf = xpbd_data->sa_Hf.view(),
            sa_iter_position = xpbd_data->sa_x.view()
        , extractHf]
        {
            const UInt vid = dispatch_id().x;
            Float3 f;
            Float3x3 H;
            extractHf(f, H, sa_Hf, vid);
            Float det = luisa::compute::determinant(H);
            $if (luisa::compute::abs(det) > Epsilon) 
            {
                // det
                Float3x3 H_inv = luisa::compute::inverse(H);
                Float3 dx = H_inv * f;
                dx *= 0.3f;
                sa_iter_position->write(vid, sa_iter_position->read(vid) + dx);
            };
        }
    );
}
void DescentSolverCPU::collision_detection(luisa::compute::Stream& stream)
{
    // TODO
}
void DescentSolverCPU::predict_position(luisa::compute::Stream& stream)
{
    // CpuParallel::parallel_for(0, mesh_data->num_verts, [&](const uint vid)
    // {
    //     Constrains::Core::predict_position(vid, 
    //         xpbd_data->sa_x.data(), 
    //         xpbd_data->sa_v.data(), 
    //         xpbd_data->sa_x_start.data(),
    //         false, 
    //         nullptr, 
    //         mesh_data->sa_vert_mass.data(), 
    //         mesh_data->sa_is_fixed.data(), 
    //         lcsv::get_scene_params().get_substep_dt(), 
    //         false);
    // });
}
void DescentSolverCPU::update_velocity(luisa::compute::Stream& stream)
{
    // CpuParallel::parallel_for(0, mesh_data->num_verts, [&](const uint vid)
    // {
    //     Constrains::Core::update_velocity(vid, 
    //         xpbd_data->sa_v.data(), 
    //         xpbd_data->sa_x.data(), 
    //         xpbd_data->sa_x_start.data(), 
    //         mesh_data->sa_x_frame_start.data(), 
    //         xpbd_data->sa_v_start.data(), 
    //         lcsv::get_scene_params().get_substep_dt(), 
    //         lcsv::get_scene_params().damping_cloth, 
    //         false);
    // });
}
// void CpuSolver::compute_energy(const Buffer<float3>& curr_position)
void compute_energy()
{
    // if (!lcsv::get_scene_params().print_xpbd_convergence) return;
    // // luisa::log_info("buffer size = {}", curr_position.size());

    // double energy = 0.0;
    // double energy_inertia = 0.f, energy_stretch = 0.f, energy_bending = 0.f;

    // // Inertia
    // {
    //     energy_inertia = CpuParallel::parallel_for_and_reduce_sum<double>(0, mesh_data->num_verts, [&](const uint vid)
    //     {
    //         return Constrains::Energy::compute_energy_inertia(vid, 
    //             curr_position.data(), 
    //             &lcsv::get_scene_params(), 
    //             mesh_data->sa_is_fixed.data(), 
    //             mesh_data->sa_vert_mass.data(), 
    //             xpbd_data->sa_x_start.data(), 
    //             xpbd_data->sa_v_start.data());
    //     });
    // }
    
    // // Stretch 
    // {
    //     const float stiffness = lcsv::get_scene_params().stiffness_stretch_spring;
    //     energy_stretch = CpuParallel::parallel_for_and_reduce_sum<double>(0, mesh_data->num_edges, [&](const uint eid)
    //     {
    //         return Constrains::Energy::compute_energy_stretch_mass_spring(
    //             eid, curr_position.data(), 
    //             xpbd_data->sa_merged_edges.data(), 
    //             xpbd_data->sa_merged_edges_rest_length.data(), 
    //             stiffness);
    //     });
    // }

    // // Bending
    // if (lcsv::get_scene_params().use_bending)
    // {   
    //     const auto bending_type = 
    //         (lcsv::get_scene_params().use_vbd_solver // Our VBD solver only add quadratic bending implementation
    //         || lcsv::get_scene_params().use_quadratic_bending_model) ?  
    //         Constrains::BendingTypeQuadratic : Constrains::BendingTypeDAB;
    //     const bool use_xpbd_solver = lcsv::get_scene_params().use_xpbd_solver;

    //     const float stiffness_bending_quadratic = lcsv::get_scene_params().get_stiffness_quadratic_bending();
    //     const float stiffness_bending_DAB = lcsv::get_scene_params().get_stiffness_DAB_bending();

    //     energy_bending = CpuParallel::parallel_for_and_reduce_sum<double>(0, mesh_data->num_bending_edges, [&](const uint eid)
    //     {
    //         float energy = 0.f;
    //         Constrains::Energy::compute_energy_bending(bending_type, eid, curr_position.data(), 
    //             xpbd_data->sa_merged_bending_edges.data(), 
    //             nullptr,
    //             nullptr, 
    //             xpbd_data->sa_merged_bending_edges_Q.data(),
    //             xpbd_data->sa_merged_bending_edges_angle.data(), 
    //             stiffness_bending_DAB, 
    //             stiffness_bending_quadratic, 
    //             use_xpbd_solver
    //         );
    //         return energy;
    //     });
    // }
    
    // // Obstacle Collisoin
    // float energy_obs_collision = 0.0f;

    // // Self Collision
    // float energy_self_collision = 0.0f;

    // double total_energy = energy_inertia + energy_stretch + energy_bending + energy_obs_collision + energy_self_collision;

    // mesh_data->sa_system_energy[energy_idx++] = total_energy;
}


// VBD constraints (energy)
// Buffer<float4x3>& CpuSolver::get_Hf()
// {
//     return xpbd_data->sa_Hf;
// }
void DescentSolverCPU::vbd_evaluate_inertia(luisa::compute::Stream& stream, Buffer<float3>& sa_iter_position, const uint cluster_idx)
{
    // auto& clusters = xpbd_data->clusterd_per_vertex_bending;
    // const uint next_prefix = clusters[cluster_idx + 1];
    // const uint curr_prefix = clusters[cluster_idx];
    // const uint num_verts_cluster = next_prefix - curr_prefix;

    // CpuParallel::parallel_for(0, num_verts_cluster, [&](const uint i)
    // {
    //     const uint vid = clusters[curr_prefix + i];
    //     float4x3 Hf = Constrains::VBD::compute_inertia(
    //         vid, sa_iter_position.data(), 
    //         xpbd_data->sa_x_start.data(), xpbd_data->sa_v.data(), 
    //         mesh_data->sa_is_fixed.data(), mesh_data->sa_vert_mass.data(), &lcsv::get_scene_params(),
    //         lcsv::get_scene_params().get_substep_dt());
    //     get_Hf()[vid] = Hf;
    // });
}
void DescentSolverCPU::vbd_evaluate_stretch_spring(luisa::compute::Stream& stream, Buffer<float3>& sa_iter_position, const uint cluster_idx)
{
    // auto& clusters = xpbd_data->clusterd_per_vertex_bending;
    // const uint next_prefix = clusters[cluster_idx + 1];
    // const uint curr_prefix = clusters[cluster_idx];
    // const uint num_verts_cluster = next_prefix - curr_prefix;
    
    // CpuParallel::parallel_for(0, num_verts_cluster, [&](const uint i)
    // {
    //     const uint vid = clusters[curr_prefix + i];
    //     float4x3 Hf = Constrains::VBD::compute_stretch_mass_spring(
    //             vid, sa_iter_position.data(), 
    //             mesh_data->sa_vert_adj_edges.data(),
    //             mesh_data->sa_edges.data(), mesh_data->sa_edges_rest_state_length.data(), 
    //             lcsv::get_scene_params().stiffness_stretch_spring);
    //     get_Hf()[vid] += Hf;
    // }, 32);
}
void DescentSolverCPU::vbd_evaluate_bending(luisa::compute::Stream& stream, Buffer<float3>& sa_iter_position, const uint cluster_idx)
{
    // auto& clusters = xpbd_data->clusterd_per_vertex_bending;
    // const uint next_prefix = clusters[cluster_idx + 1];
    // const uint curr_prefix = clusters[cluster_idx];
    // const uint num_verts_cluster = next_prefix - curr_prefix;

    // CpuParallel::parallel_for(0, num_verts_cluster, [&](const uint i)
    // {
    //     const uint vid = clusters[curr_prefix + i];
    //     float4x3 Hf = Constrains::VBD::compute_bending_quadratic(
    //             vid, sa_iter_position.data(),
    //             mesh_data->sa_vert_adj_bending_edges.data(), mesh_data->sa_bending_edges.data(), 
    //             mesh_data->sa_bending_edges_Q.data(), 
    //             lcsv::get_scene_params().get_stiffness_quadratic_bending());
    //     get_Hf()[vid] += Hf;
    // }, 32);
}
void DescentSolverCPU::vbd_step(luisa::compute::Stream& stream, Buffer<float3>& sa_iter_position, const uint cluster_idx)
{
    // auto& clusters = xpbd_data->clusterd_per_vertex_bending;
    // const uint next_prefix = clusters[cluster_idx + 1];
    // const uint curr_prefix = clusters[cluster_idx];
    // const uint num_verts_cluster = next_prefix - curr_prefix;

    // CpuParallel::parallel_for(0, num_verts_cluster, [&](const uint i)
    // {
    //     const uint vid = clusters[curr_prefix + i];
    //     float4x3 Hf = get_Hf()[vid];
    //     Float3x3 H = make_float3x3(get(Hf, 0), get(Hf, 1), get(Hf, 2));
    //     float3 f = get(Hf, 3);
    //     float det = determinant_mat(H);
    //     if (abs_scalar(det) > Epsilon)
    //     {
    //         Float3x3 H_inv = inverse_mat(H, det);
    //         float3 dx = H_inv * f;
    //         sa_iter_position[vid] += dx;
    //     }
    // }, 32);
}


void DescentSolverCPU::physics_step_xpbd(luisa::compute::Device& device, luisa::compute::Stream& stream)
{
    lcsv::SolverInterface::physics_step_prev_operation(device, stream);
}
void DescentSolverCPU::physics_step_vbd(luisa::compute::Device& device, luisa::compute::Stream& stream)
{
    lcsv::SolverInterface::physics_step_prev_operation(device, stream); 
    // Get frame start position and velocity
    CpuParallel::parallel_for(0, host_xpbd_data->sa_x.size(), [&](const uint vid)
    {
        host_xpbd_data->sa_x[vid] = host_mesh_data->sa_x_frame_start[vid];
        host_xpbd_data->sa_v[vid] = host_mesh_data->sa_v_frame_start[vid];
    });
    
    stream << xpbd_data->sa_x.copy_from(host_xpbd_data->sa_x.data())
           << xpbd_data->sa_v.copy_from(host_xpbd_data->sa_v.data())
           << xpbd_data->sa_x_start.copy_from(host_xpbd_data->sa_x.data())
           << xpbd_data->sa_v_start.copy_from(host_xpbd_data->sa_v.data())
           << mp_buffer_filler->fill(device, mesh_data->sa_system_energy, 0.0f)
           << luisa::compute::synchronize();
    
    const uint num_substep = lcsv::get_scene_params().print_xpbd_convergence ? 1 : lcsv::get_scene_params().num_substep;
    const uint constraint_iter_count = lcsv::get_scene_params().constraint_iter_count;

    // energy_idx = 0;

    for (uint substep = 0; substep < num_substep; substep++)
    {
        stream << fn_predict_position().dispatch(mesh_data->num_verts);

        for (uint iter = 0; iter < constraint_iter_count; iter++)
        {
            stream 
                << fn_evaluate_inertia().dispatch(mesh_data->num_verts)
                << fn_evaluate_stretch_spring().dispatch(mesh_data->num_verts)
                << fn_step().dispatch(mesh_data->num_verts);
        }

        stream << fn_update_velocity().dispatch(mesh_data->num_verts);
    }
    
    stream << luisa::compute::synchronize();

    // for (uint substep = 0; substep < num_substep; substep++) // 1 or 50 ?
    // {   { lcsv::get_scene_params().current_substep = substep; }
        
    //     {   
    //         predict_position(stream); 

    //         collision_detection(stream);

    //         // Constraint iteration part
    //         {
    //             for (uint iter = 0; iter < constraint_iter_count; iter++) // 200 or 1 ?
    //             {   
    //                 { lcsv::get_scene_params().current_it = iter; }
    //                 if (lcsv::get_scene_params().use_vbd_solver) { solve_constraints_VBD(stream); }
    //                 else { luisa::log_error("empty solver"); }
    //             }
    //         }

    //         update_velocity(stream); 
    //     }
    // }
    // luisa::log_info("Frame {:3} : cost = {:6.3f}", lcsv::get_scene_params().current_frame, frame_cost);
    
    // CpuParallel::parallel_for(0, host_xpbd_data->sa_x.size(), [&](const uint vid)
    // {
    //     if (!host_mesh_data->sa_is_fixed[vid])
    //     {
    //         host_xpbd_data->sa_x[vid] -= luisa::make_float3(0, 0.1, 0);
    //     }
    // });

    // Copy to host (if use GPU)
    // if constexpr (false)
    {
        stream << xpbd_data->sa_x.copy_to(host_xpbd_data->sa_x.data())
           << xpbd_data->sa_v.copy_to(host_xpbd_data->sa_v.data())
           << luisa::compute::synchronize();
    }
    
    // Return frame end position and velocity
    CpuParallel::parallel_for(0, host_xpbd_data->sa_x.size(), [&](const uint vid)
    {
        host_mesh_data->sa_x_frame_end[vid] = host_xpbd_data->sa_x[vid];
        host_mesh_data->sa_v_frame_end[vid] = host_xpbd_data->sa_v[vid];
    });
    lcsv::SolverInterface::physics_step_post_operation(device, stream); 
}
void DescentSolverCPU::solve_constraints_VBD(luisa::compute::Stream& stream)
{
    // auto& iter_position = xpbd_data->sa_x;

    // if (lcsv::get_scene_params().print_xpbd_convergence && lcsv::get_scene_params().current_it == 0) 
    // { 
    //     compute_energy(iter_position); 
    // }

    // for (uint cluster = 0; cluster < xpbd_data->num_clusters_per_vertex_bending; cluster++)
    // {
    //     const uint next_prefix = xpbd_data->clusterd_per_vertex_bending[cluster + 1];
    //     const uint curr_prefix = xpbd_data->clusterd_per_vertex_bending[cluster];
    //     const uint num_verts_cluster = next_prefix - curr_prefix;

    //     vbd_evaluate_inertia(iter_position, cluster);

    //     vbd_evaluate_stretch_spring(iter_position, cluster);
        
    //     vbd_evaluate_bending(iter_position, cluster);
        
    //     vbd_step(iter_position, cluster);
    // }

    // if (lcsv::get_scene_params().print_xpbd_convergence) 
    // { 
    //     compute_energy(iter_position); 
    // }
}


}