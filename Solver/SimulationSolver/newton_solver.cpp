#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include "CollisionDetector/libuipc/codim_ipc_simplex_normal_contact_function.h"
#include "Core/float_n.h"
#include "SimulationSolver/descent_solver.h"
#include "SimulationSolver/newton_solver.h"
#include "Core/float_nxn.h"
#include "Core/lc_to_eigen.h"
#include "Core/scalar.h"
#include "Core/xbasic_types.h"
#include "SimulationCore/scene_params.h"
#include "Utils/cpu_parallel.h"
#include "Utils/reduce_helper.h"
#include "luisa/backends/ext/pinned_memory_ext.hpp"
#include "luisa/core/logging.h"
#include "luisa/core/mathematics.h"
#include "luisa/dsl/builtin.h"
#include "luisa/dsl/stmt.h"
#include "luisa/runtime/buffer.h"
#include "luisa/runtime/stream.h"
#include <luisa/dsl/sugar.h>
#include <vector>

namespace lcs 
{

// template<typename T>
// void buffer_add(luisa::compute::BufferView<T> buffer, const Var<uint> dest, const Var<T>& value)
// {
//     buffer->write(dest, buffer->read(dest) + value);
// }
template<typename T>
void buffer_add(Var<luisa::compute::BufferView<T>>& buffer, const Var<uint> dest, const Var<T>& value)
{
    buffer->write(dest, buffer->read(dest) + value);
}
void atomic_buffer_add(Var<luisa::compute::BufferView<float3>>& buffer, const Var<uint> dest, const Var<float3>& value)
{
    buffer->atomic(dest)[0].fetch_add(value[0]);
    buffer->atomic(dest)[1].fetch_add(value[1]);
    buffer->atomic(dest)[2].fetch_add(value[2]);
}
void atomic_buffer_add(Var<luisa::compute::BufferView<float3x3>>& buffer, const Var<uint> dest, const Var<float3x3>& value)
{
    buffer->atomic(dest)[0][0].fetch_add(value[0][0]);
    buffer->atomic(dest)[0][1].fetch_add(value[0][1]);
    buffer->atomic(dest)[0][2].fetch_add(value[0][2]);
    buffer->atomic(dest)[1][0].fetch_add(value[1][0]);
    buffer->atomic(dest)[1][1].fetch_add(value[1][1]);
    buffer->atomic(dest)[1][2].fetch_add(value[1][2]);
    buffer->atomic(dest)[2][0].fetch_add(value[2][0]);
    buffer->atomic(dest)[2][1].fetch_add(value[2][1]);
    buffer->atomic(dest)[2][2].fetch_add(value[2][2]);
}

void NewtonSolver::compile(luisa::compute::Device& device)
{
    const bool use_debug_info = false;
    using namespace luisa::compute;

    luisa::compute::ShaderOption default_option = {.enable_debug_info = false};

    fn_reset_vector = device.compile<1>([](Var<BufferView<float3>> buffer)
    {
        const UInt vid = dispatch_id().x;
        // buffer->write(vid, target);
        buffer->write(vid, make_float3(0.0f));
    });
    fn_reset_float3x3 = device.compile<1>([](Var<BufferView<float3x3>> buffer)
    {
        const UInt vid = dispatch_id().x;
        buffer->write(vid, make_float3x3(0.0f));
    });

    fn_predict_position = device.compile<1>(
        [
            sa_x = sim_data->sa_x.view(),
            sa_x_step_start = sim_data->sa_x_step_start.view(), 
            sa_x_iter_start = sim_data->sa_x_iter_start.view(),
            sa_x_tilde = sim_data->sa_x_tilde.view(),
            sa_v = sim_data->sa_v.view(),
            sa_cgX = sim_data->sa_cgX.view(),
            sa_is_fixed = mesh_data->sa_is_fixed.view()
        ](const Float substep_dt)
    {
        const UInt vid = dispatch_id().x;
        const Float3 gravity = make_float3(0.0f, -9.8f, 0.0f);
        Float3 x_prev = sa_x_step_start->read(vid);
        Float3 v_prev = sa_v->read(vid);
        Float3 outer_acceleration = gravity;
        Float3 v_pred = v_prev + substep_dt * outer_acceleration;

        const Bool is_fixed = sa_is_fixed->read(vid);

        $if (is_fixed)
        { 
            v_pred = v_prev;
            // v_pred = make_float3(0.0f); 
        };

        sa_x_iter_start->write(vid, x_prev);
        Float3 x_pred = x_prev + substep_dt * v_pred;
        sa_x_tilde->write(vid, x_pred);
        sa_x->write(vid, x_prev);
        // sa_cgX->write(vid, make_float3(0.0f));
    }, default_option);

    fn_update_velocity = device.compile<1>(
        [
            sa_x = sim_data->sa_x.view(),
            sa_v = sim_data->sa_v.view(),
            sa_x_step_start = sim_data->sa_x_step_start.view(),
            sa_v_step_start = sim_data->sa_v_step_start.view()
        ](const Float substep_dt, const Bool fix_scene, const Float damping)
    {
        const UInt vid = dispatch_id().x;
        Float3 x_step_begin = sa_x_step_start->read(vid);
        Float3 x_step_end = sa_x->read(vid);

        Float3 dx = x_step_end - x_step_begin;
        Float3 vel = dx / substep_dt;

        $if (fix_scene) 
        {
            dx = make_float3(0.0f);
            vel = make_float3(0.0f);
            sa_x->write(vid, x_step_begin);
            return;
        };

        vel *= exp(-damping * substep_dt);
        
        sa_v->write(vid, vel);
        sa_v_step_start->write(vid, vel);
        sa_x_step_start->write(vid, x_step_end);
    }, default_option);

    fn_evaluate_inertia = device.compile<1>(
        [
            sa_x = sim_data->sa_x.view(),
            sa_x_tilde = sim_data->sa_x_tilde.view(),
            sa_cgB = sim_data->sa_cgB.view(),
            sa_cgA_diag = sim_data->sa_cgA_diag.view(),
            sa_is_fixed = mesh_data->sa_is_fixed.view(),
            sa_vert_mass = mesh_data->sa_vert_mass.view()
        ](const Float substep_dt, const Float stiffness_dirichlet)
    {
        const UInt vid = dispatch_id().x;
        const Float h = substep_dt;
        const Float h_2_inv = 1.0f / (h * h);

        Float3 x_k = sa_x->read(vid);
        Float3 x_tilde = sa_x_tilde->read(vid);
        Float mass = sa_vert_mass->read(vid);

        Float3 gradient = -mass * h_2_inv * (x_k - x_tilde);
        Float3x3 hessian = make_float3x3(1.0f) * mass * h_2_inv;

        $if (sa_is_fixed->read(vid) != 0) 
        {
            gradient = gradient + stiffness_dirichlet * gradient;
            hessian = hessian + stiffness_dirichlet * hessian;
            // hessian = make_float3x3(1.0f) * 1e9f;
            // gradient = make_float3(0.0f);
        };

        sa_cgB->write(vid, gradient); 
        sa_cgA_diag->write(vid, hessian);
    }, default_option);

    fn_evaluate_dirichlet = device.compile<1>(
        [
            sa_x = sim_data->sa_x.view(),
            sa_x_tilde = sim_data->sa_x_tilde.view(),
            sa_cgB = sim_data->sa_cgB.view(),
            sa_cgA_diag = sim_data->sa_cgA_diag.view(),
            sa_is_fixed = mesh_data->sa_is_fixed.view(),
            sa_vert_mass = mesh_data->sa_vert_mass.view()
        ](const Float substep_dt, const Float stiffness_dirichlet)
    {
        const UInt vid = dispatch_id().x;
        return;

        Bool is_fixed = sa_is_fixed->read(vid);
        $if (is_fixed) 
        {
            const Float h = substep_dt;
            const Float h_2_inv = 1.0f / (h * h);

            Float3 x_k = sa_x->read(vid);
            Float3 x_tilde = sa_x_tilde->read(vid);
            // Float3 gradient = stiffness_dirichlet * (x_k - x_tilde);
            // Float3x3 hessian = stiffness_dirichlet * make_float3x3(1.0f);
            Float mass = sa_vert_mass->read(vid);
            Float3 gradient  = stiffness_dirichlet * h_2_inv * mass * (x_k - x_tilde);
            Float3x3 hessian = stiffness_dirichlet * h_2_inv * mass * make_float3x3(1.0f);
            sa_cgB->write(vid, sa_cgB->read(vid) - gradient); 
            sa_cgA_diag->write(vid, sa_cgA_diag->read(vid) + hessian);
        };
    }, default_option);

    fn_evaluate_ground_collision = device.compile<1>(
        [
            sa_x = sim_data->sa_x.view(),
            sa_rest_vert_area = mesh_data->sa_rest_vert_area.view(),
            sa_cgB = sim_data->sa_cgB.view(),
            sa_cgA_diag = sim_data->sa_cgA_diag.view(),
            sa_is_fixed = mesh_data->sa_is_fixed.view(),
            sa_vert_mass = mesh_data->sa_vert_mass.view()
        ](
            Float floor_y,
            Bool use_ground_collision,
            Float stiffness,
            Float d_hat,
            Float thickness
        )
    {
        const UInt vid = dispatch_id().x;
        $if (use_ground_collision)
        {
            $if (!sa_is_fixed->read(vid))
            {
                Float3 x_k = sa_x->read(vid);
                Float diff = x_k.y - floor_y;
        
                Float3 force = sa_cgB->read(vid);
                Float3x3 hessian = sa_cgA_diag->read(vid);
                $if (diff < d_hat + thickness)
                {
                    Float C = d_hat + thickness - diff;
                    float3 normal = luisa::make_float3(0, 1, 0);
                    Float area = sa_rest_vert_area->read(vid);
                    Float stiff = stiffness * area;
                    force += stiff * C * normal;
                    hessian += stiff * outer_product(normal, normal);
                };
                sa_cgB->write(vid, force);
                sa_cgA_diag->write(vid, hessian);
            };
        };
    }, default_option);
    
    fn_evaluate_spring = device.compile<1>(
        [
            sa_x = sim_data->sa_x.view(),
            sa_cgB = sim_data->sa_cgB.view(),
            sa_cgA_diag = sim_data->sa_cgA_diag.view(),
            sa_cgA_offdiag_stretch_spring = sim_data->sa_cgA_offdiag_stretch_spring.view(),
            sa_edges = sim_data->sa_stretch_springs.view(),
            sa_rest_length = sim_data->sa_stretch_spring_rest_state_length.view()
            // sa_edges = sim_data->sa_merged_stretch_springs.view(),
            // sa_rest_length = sim_data->sa_merged_stretch_spring_rest_length.view()
        ](const Float stiffness_stretch)
    {
        // const Uint curr_prefix = culster->read(cluster_idx);
        // const UInt eid = curr_prefix + dispatch_id().x;

        const UInt eid = dispatch_id().x;
        UInt2 edge = sa_edges->read(eid);

        Float3 vert_pos[2] = { sa_x->read(edge.x), sa_x->read(edge.y) };
        Float3 force[2] = {make_float3(0.0f), make_float3(0.0f)};
        Float3x3 He = make_float3x3(0.0f);

        const Float L = sa_rest_length->read(eid);
        const Float stiffness_spring = stiffness_stretch;

        Float3 diff = vert_pos[1] - vert_pos[0];
        Float l = max(length(diff), Epsilon);
        Float l0 = L;
        Float C = l - l0;

        Float3 dir = diff / l;
        Float3x3 xxT = outer_product(diff, diff);
        Float x_inv = 1.f / l;
        Float x_squared_inv = x_inv * x_inv;

        force[0] = stiffness_spring * dir * C;
        force[1] = -force[0];
        He = stiffness_spring * x_squared_inv * xxT + stiffness_spring * max(1.0f - L * x_inv, 0.0f) * (make_float3x3(1.0f) - x_squared_inv * xxT);

        for (uint j = 0; j < 2; j++)
        {
            for (uint ii = 0; ii < 3; ii++)
            {
                sa_cgB->atomic(edge[j])[ii].fetch_add(force[j][ii]);
            }
        }

        for (uint j = 0; j < 2; j++)
        {
            for (uint ii = 0; ii < 3; ii++)
            {
                for (uint jj = 0; jj < 3; jj++)
                {
                    sa_cgA_diag->atomic(edge[j])[ii][jj].fetch_add(He[ii][jj]);
                    // sa_cgB->atomic(edge[ii])[jj].fetch_add(force[ii][jj]);
                }
            }
        }
        // atomic_buffer_add(sa_cgB, edge[0], force[0]);
        // atomic_buffer_add(sa_cgB, edge[1], force[1]);
        // atomic_buffer_add(sa_cgA_diag, edge[0], He);
        // atomic_buffer_add(sa_cgA_diag, edge[1], He);
        sa_cgA_offdiag_stretch_spring->write(eid, -1.0f * He); // eid * 2 + 0, -1.0f * He);
    }, default_option);

    fn_evaluate_bending = device.compile<1>(
        [
            sa_x = sim_data->sa_x.view(),
            sa_cgB = sim_data->sa_cgB.view(),
            sa_cgA_diag = sim_data->sa_cgA_diag.view(),
            sa_cgA_offdiag_bending = sim_data->sa_cgA_offdiag_bending.view(),
            sa_edges = sim_data->sa_bending_edges.view(),
            sa_bending_edges_Q = sim_data->sa_bending_edges_Q.view()
        ](const Float stiffness_bending)
    {
        // const Uint curr_prefix = culster->read(cluster_idx);
        // const UInt eid = curr_prefix + dispatch_id().x;

        const UInt eid = dispatch_id().x;
        UInt4 edge = sa_edges->read(eid);
        Float4x4 m_Q = sa_bending_edges_Q->read(eid);

        Float3 vert_pos[4] = { 
            sa_x->read(edge[0]), 
            sa_x->read(edge[1]), 
            sa_x->read(edge[2]), 
            sa_x->read(edge[3]), 
        };
        Float3 force[4] = {
            make_float3(0.0f), 
            make_float3(0.0f),
            make_float3(0.0f),
            make_float3(0.0f),
        };

        
        for (uint ii = 0; ii < 4; ii++) 
        {
            for (uint jj = 0; jj < 4; jj++) 
            {
                force[ii] -= m_Q[ii][jj] * vert_pos[jj]; // -Qx
            }
            force[ii] = stiffness_bending * force[ii];
        }
        for (uint j = 0; j < 4; j++)
        {
            for (uint ii = 0; ii < 3; ii++)
            {
                sa_cgB->atomic(edge[j])[ii].fetch_add(force[j][ii]);
            }
        }

        // Float3x3 hessian[10];
        {
            //  0   1   2   3
            // t1   4   5   6
            // t2  t5   7   8
            // t3  t6  t8   9 
            // hessian[0] = stiffness_bending * m_Q[0][0] * luisa::compute::make_float3x3(1.0f);
            // hessian[1] = stiffness_bending * m_Q[0][1] * luisa::compute::make_float3x3(1.0f);
            // hessian[2] = stiffness_bending * m_Q[0][2] * luisa::compute::make_float3x3(1.0f);
            // hessian[3] = stiffness_bending * m_Q[0][3] * luisa::compute::make_float3x3(1.0f);
            // hessian[4] = stiffness_bending * m_Q[1][1] * luisa::compute::make_float3x3(1.0f);
            // hessian[5] = stiffness_bending * m_Q[1][2] * luisa::compute::make_float3x3(1.0f);
            // hessian[6] = stiffness_bending * m_Q[1][3] * luisa::compute::make_float3x3(1.0f);
            // hessian[7] = stiffness_bending * m_Q[2][2] * luisa::compute::make_float3x3(1.0f);
            // hessian[8] = stiffness_bending * m_Q[2][3] * luisa::compute::make_float3x3(1.0f);
            // hessian[9] = stiffness_bending * m_Q[3][3] * luisa::compute::make_float3x3(1.0f);
        }
        for (uint j = 0; j < 4; j++)
        {
            const Float3x3& diag = stiffness_bending * m_Q[j][j] * luisa::compute::make_float3x3(1.0f);
            for (uint ii = 0; ii < 3; ii++)
            {
                for (uint jj = 0; jj < 3; jj++)
                {
                    sa_cgA_diag->atomic(edge[j])[ii][jj].fetch_add(diag[ii][jj]);
                }
            }
        }
  
        // sa_cgA_offdiag_bending->write(6 * eid + 0, stiffness_bending * m_Q[0][1] * luisa::compute::make_float3x3(1.0f));
        // sa_cgA_offdiag_bending->write(6 * eid + 1, stiffness_bending * m_Q[0][2] * luisa::compute::make_float3x3(1.0f));
        // sa_cgA_offdiag_bending->write(6 * eid + 2, stiffness_bending * m_Q[0][3] * luisa::compute::make_float3x3(1.0f));
        // sa_cgA_offdiag_bending->write(6 * eid + 3, stiffness_bending * m_Q[1][2] * luisa::compute::make_float3x3(1.0f));
        // sa_cgA_offdiag_bending->write(6 * eid + 4, stiffness_bending * m_Q[1][3] * luisa::compute::make_float3x3(1.0f));
        // sa_cgA_offdiag_bending->write(6 * eid + 5, stiffness_bending * m_Q[2][3] * luisa::compute::make_float3x3(1.0f));

        // uint off_diag_offset[4] = {0, 4, 7, 9};
        // for (uint j = 0; j < 4; j++)
        // {
        //     const Float3x3& off_diag = hessian[off_diag_offset[j]];
        //     for (uint ii = 0; ii < 3; ii++)
        //     {
        //         for (uint jj = 0; jj < 3; jj++)
        //         {
        //             sa_cgA_diag->atomic(edge[j])[ii][jj].fetch_add(off_diag[ii][jj]);
        //         }
        //     }
        // }
  
        // sa_cgA_offdiag_bending->write(6 * eid + 0, hessian[1]);
        // sa_cgA_offdiag_bending->write(6 * eid + 1, hessian[2]);
        // sa_cgA_offdiag_bending->write(6 * eid + 2, hessian[3]);
        // sa_cgA_offdiag_bending->write(6 * eid + 3, hessian[5]);
        // sa_cgA_offdiag_bending->write(6 * eid + 4, hessian[6]);
        // sa_cgA_offdiag_bending->write(6 * eid + 5, hessian[8]);
        
    }, default_option);

    // SpMV
    // PCG SPMV diagonal kernel
    fn_pcg_spmv_diag = device.compile<1>(
        [
            sa_cgA_diag = sim_data->sa_cgA_diag.view()
        ](
            Var<luisa::compute::BufferView<float3>> sa_input_vec, 
            Var<luisa::compute::BufferView<float3>> sa_output_vec
        )
    {
        const UInt vid = dispatch_id().x;
        Float3x3 A_diag = sa_cgA_diag->read(vid);
        Float3 input = sa_input_vec->read(vid);
        Float3 diag_output = A_diag * input;
        sa_output_vec->write(vid, diag_output);
    }, default_option);


    fn_pcg_spmv_offdiag_stretch_spring = device.compile<1>(
        [
            sa_stretch_springs = sim_data->sa_stretch_springs.view(),
            sa_cgA_offdiag_stretch_spring = sim_data->sa_cgA_offdiag_stretch_spring.view()
        ](
            Var<luisa::compute::BufferView<float3>> sa_input_vec, 
            Var<luisa::compute::BufferView<float3>> sa_output_vec
        )
    {
        const UInt eid = dispatch_id().x;
        UInt2 edge = sa_stretch_springs->read(eid);
        Float3x3 offdiag_hessian1 = sa_cgA_offdiag_stretch_spring->read(eid);
        Float3x3 offdiag_hessian2 = luisa::compute::transpose(offdiag_hessian1);

        Float3 input1 = sa_input_vec->read(edge[1]);
        Float3 input2 = sa_input_vec->read(edge[0]);
        
        Float3 output1 = offdiag_hessian1 * input1;
        Float3 output2 = offdiag_hessian2 * input2;

        atomic_buffer_add(sa_output_vec, edge[0], output1);
        atomic_buffer_add(sa_output_vec, edge[1], output2);
        // buffer_add(sa_output_vec, edge[0], output1);
        // buffer_add(sa_output_vec, edge[1], output2);
    }, default_option);

    fn_pcg_spmv_offdiag_bending = device.compile<1>(
        [
            sa_bending_edges = sim_data->sa_bending_edges.view(),
            sa_bending_edges_Q = sim_data->sa_bending_edges_Q.view(),
            sa_cgA_offdiag_bending = sim_data->sa_cgA_offdiag_bending.view()
        ](
            Var<luisa::compute::BufferView<float3>> sa_input_vec, 
            Var<luisa::compute::BufferView<float3>> sa_output_vec,
            Float stiffness_bending
        )
    {
        const UInt eid = dispatch_id().x;
        UInt4 edge = sa_bending_edges->read(eid);
        Float4x4 m_Q = sa_bending_edges_Q->read(eid);

        Float3 input_vec[4] = {
            sa_input_vec.read(edge[0]),
            sa_input_vec.read(edge[1]),
            sa_input_vec.read(edge[2]),
            sa_input_vec.read(edge[3]),
        }; 
        Float3 output_vec[4] = {
            make_float3(0.0f),
            make_float3(0.0f),
            make_float3(0.0f),
            make_float3(0.0f),
        };
        for (uint j = 0; j < 4; j++)
        {
            for (uint jj = 0; jj < 4; jj++)
            {
                if (j != jj)
                {
                    Float3x3 hessian = stiffness_bending * m_Q[j][jj] * luisa::compute::make_float3x3(1.0f);
                    output_vec[j] += hessian * input_vec[jj];
                }
            }
        }
        // Uint offset = 0;
        // for (uint j = 0; j < 4; j++)
        // {
        //     for (uint jj = j + 1; jj < 4; jj++)
        //     {
        //         //  0   1   2   3
        //         // t1   4   5   6
        //         // t2  t5   7   8
        //         // t3  t6  t8   9 
        //         Float3x3 hessian = sa_cgA_offdiag_bending->read(6 * eid + offset); offset += 1;
        //         output_vec[j] += hessian * input_vec[jj];
        //         output_vec[jj] += transpose(hessian) * input_vec[j];
        //     }
        // }
        atomic_buffer_add(sa_output_vec, edge[0], output_vec[0]);
        atomic_buffer_add(sa_output_vec, edge[1], output_vec[1]);
        atomic_buffer_add(sa_output_vec, edge[2], output_vec[2]);
        atomic_buffer_add(sa_output_vec, edge[3], output_vec[3]); 
    }, default_option);

    // Line search
    auto fn_reduce_and_add_energy = device.compile<1>(
        [
            sa_block_result = sim_data->sa_block_result.view(),
            sa_convergence = sim_data->sa_convergence.view()
        ]()
        {
            const Uint index = dispatch_id().x;
            Float energy = 0.0f;
            {
                energy = sa_block_result->read(index);
            };
            energy = ParallelIntrinsic::block_intrinsic_reduce(index, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);

            $if (index == 0)
            {
                sa_convergence->atomic(7).fetch_add(energy);
                // buffer_add(sa_convergence, 7, energy);
            };
        }
    );


    fn_apply_dx = device.compile<1>(
        [
            sa_x = sim_data->sa_x.view(),
            sa_x_iter_start = sim_data->sa_x_iter_start.view(),
            sa_cgX = sim_data->sa_cgX.view()
        ](const Float alpha) 
    {
        const UInt vid = dispatch_id().x;
        sa_x->write(vid, sa_x_iter_start->read(vid) + alpha * sa_cgX->read(vid));
    }, default_option);

    fn_apply_dx_non_constant = device.compile<1>(
        [
            sa_x = sim_data->sa_x.view(),
            sa_x_iter_start = sim_data->sa_x_iter_start.view(),
            sa_cgX = sim_data->sa_cgX.view()
        ](Var<BufferView<float>> alpha_buffer) 
    {
        const Float alpha = alpha_buffer.read(0);
        const UInt vid = dispatch_id().x;
        sa_x->write(vid, sa_x_iter_start->read(vid) + alpha * sa_cgX->read(vid));
    }, default_option);
    
}

// Host functions
// Outputs:
//          sa_x_iter_start
//          sa_x_tilde
//          sa_x
//          sa_cgX
void NewtonSolver::host_predict_position()
{
    CpuParallel::parallel_for(0, host_mesh_data->num_verts, 
        [
            sa_x = host_sim_data->sa_x.data(),
            sa_v = host_sim_data->sa_v.data(),
            sa_cgX = host_sim_data->sa_cgX.data(),
            sa_x_step_start = host_sim_data->sa_x_step_start.data(),
            sa_x_iter_start = host_sim_data->sa_x_iter_start.data(),
            sa_x_tilde = host_sim_data->sa_x_tilde.data(),
            sa_is_fixed = host_mesh_data->sa_is_fixed.data(),
            substep_dt = get_scene_params().get_substep_dt()
        ](const uint vid)
    {   
        const float3 gravity(0, -9.8f, 0);
        float3 x_prev = sa_x_step_start[vid];
        float3 v_prev = sa_v[vid];
        float3 outer_acceleration = gravity;
        // If we consider gravity energy here, then we will not consider it in potential energy 
        float3 v_pred = v_prev + substep_dt * outer_acceleration;
        if (sa_is_fixed[vid]) 
        { 
            // v_pred = Zero3; 
            v_pred = v_prev;
        };

        const float3 x_pred = x_prev + substep_dt * v_pred; 
        sa_x_iter_start[vid] = x_prev;
        sa_x_tilde[vid] = x_pred;

        // sa_x[vid] = x_pred;
        // sa_cgX[vid] = v_prev * substep_dt;
        sa_x[vid] = x_prev;
        // sa_cgX[vid] = luisa::make_float3(0.0f);
    });
}
void NewtonSolver::host_update_velocity()
{
    CpuParallel::parallel_for(0, host_mesh_data->num_verts, 
        [
            sa_x = host_sim_data->sa_x.data(),
            sa_v = host_sim_data->sa_v.data(),
            sa_x_step_start = host_sim_data->sa_x_step_start.data(),
            sa_v_step_start = host_sim_data->sa_v_step_start.data(),
            sa_is_fixed = host_mesh_data->sa_is_fixed.data(),
            substep_dt = get_scene_params().get_substep_dt(), 
            fix_scene = get_scene_params().fix_scene, 
            damping = get_scene_params().damping_cloth
        ](const uint vid)
    {   
        float3 x_step_begin = sa_x_step_start[vid];
        float3 x_step_end = sa_x[vid];

        float3 dx = x_step_end - x_step_begin;
        float3 vel = dx / substep_dt;

        if (fix_scene) 
        {
            dx = Zero3;
            vel = Zero3;
            sa_x[vid] = x_step_begin;
            return;
        };

        vel *= exp(-damping * substep_dt);

        sa_v[vid] = vel;
        sa_v_step_start[vid] = vel;
        sa_x_step_start[vid] = x_step_end;
    });
}
void NewtonSolver::host_reset_off_diag()
{
    // if constexpr (use_eigen)
    // {
    //     eigen_springA.setZero();
    // }
    // else 
    {
        CpuParallel::parallel_set(host_sim_data->sa_cgA_offdiag_stretch_spring, luisa::make_float3x3(0.0f));
        CpuParallel::parallel_set(host_sim_data->sa_cgA_offdiag_bending, luisa::make_float3x3(0.0f));
    }
}
void NewtonSolver::host_reset_cgB_cgX_diagA()
{
    // if constexpr (use_eigen)
    // {
    //     eigen_cgA.setZero();
    //     eigen_cgB.setZero();
    //     eigen_cgX.setZero();
    // }
    // else 
    {
        CpuParallel::parallel_set(host_sim_data->sa_cgA_diag, luisa::make_float3x3(0.0f));
        CpuParallel::parallel_set(host_sim_data->sa_cgB, luisa::make_float3(0.0f));
        CpuParallel::parallel_set(host_sim_data->sa_cgX, luisa::make_float3(0.0f));
    }
}
void NewtonSolver::host_evaluate_inertia()
{
    const float stiffness_dirichlet = get_scene_params().stiffness_dirichlet;

    CpuParallel::parallel_for(0, host_mesh_data->num_verts,
        [
            sa_cgB = host_sim_data->sa_cgB.data(),
            sa_cgA_diag = host_sim_data->sa_cgA_diag.data(),
            sa_x = host_sim_data->sa_x.data(),
            sa_x_tilde = host_sim_data->sa_x_tilde.data(),
            sa_is_fixed = host_mesh_data->sa_is_fixed.data(),
            sa_vert_mass = host_mesh_data->sa_vert_mass.data(),
            substep_dt = get_scene_params().get_substep_dt()
        , stiffness_dirichlet](const uint vid)
    {
        const float3 gravity(0, -9.8f, 0);
        const float h = substep_dt;
        const float h_2_inv = 1.f / (h * h);

        float3 x_k = sa_x[vid];
        float3 x_tilde = sa_x_tilde[vid];
        // float3 v_0 = sa_v[vid];

        float mass = sa_vert_mass[vid];
        float3 gradient = -mass * h_2_inv * (x_k - x_tilde);
        float3x3 hessian = luisa::make_float3x3(1.0f) * mass * h_2_inv;

        if (sa_is_fixed[vid])
        {
            gradient  = stiffness_dirichlet * gradient;
            hessian = stiffness_dirichlet * hessian;
        }
        {  
            // sa_cgX[vid] = dx_0;
            sa_cgB[vid] = gradient;
            sa_cgA_diag[vid] = hessian;
        }
    });
}
void NewtonSolver::host_evaluate_dirichlet()
{
    return;

    const float stiffness_dirichlet = get_scene_params().stiffness_dirichlet;
    const float substep_dt = get_scene_params().get_substep_dt();
    CpuParallel::parallel_for(0, host_mesh_data->num_verts,
        [
            sa_cgB = host_sim_data->sa_cgB.data(),
            sa_cgA_diag = host_sim_data->sa_cgA_diag.data(),
            sa_x = host_sim_data->sa_x.data(),
            sa_x_tilde = host_sim_data->sa_x_tilde.data(),
            sa_is_fixed = host_mesh_data->sa_is_fixed.data(),
            sa_vert_mass = host_mesh_data->sa_vert_mass.data()
        , stiffness_dirichlet, substep_dt](const uint vid)
    {
        bool is_fixed = sa_is_fixed[vid];

        if (is_fixed)
        {
            const float h = substep_dt;
            const float h_2_inv = 1.f / (h * h);

            float3 x_k = sa_x[vid];
            float3 x_tilde = sa_x_tilde[vid];
            // float3 gradient = -stiffness_dirichlet * (x_k - x_tilde);
            // float3x3 hessian = stiffness_dirichlet * luisa::make_float3x3(1.0f);
            float mass = sa_vert_mass[vid];
            float3 gradient  = stiffness_dirichlet * h_2_inv * mass * (x_k - x_tilde);
            float3x3 hessian = stiffness_dirichlet * h_2_inv * mass * luisa::make_float3x3(1.0f);
            // sa_cgB[vid] = -gradient;
            // sa_cgA_diag[vid] = hessian;
            sa_cgB[vid] = sa_cgB[vid] - gradient;
            sa_cgA_diag[vid] = sa_cgA_diag[vid] + hessian;
        };
    });
}
void NewtonSolver::host_evaluate_ground_collision()
{
    if (!get_scene_params().use_floor) return;

    auto* sa_is_fixed = host_mesh_data->sa_is_fixed.data();
    auto* sa_rest_vert_area = host_mesh_data->sa_rest_vert_area.data();

    const uint num_verts = host_mesh_data->num_verts;
    const float floor_y = get_scene_params().floor.y; 

    CpuParallel::parallel_for(0, num_verts, 
        [
            sa_cgB = host_sim_data->sa_cgB.data(),
            sa_cgA_diag = host_sim_data->sa_cgA_diag.data(),
            sa_x = host_sim_data->sa_x.data(),
            sa_is_fixed = host_mesh_data->sa_is_fixed.data(),
            sa_rest_vert_area = host_mesh_data->sa_rest_vert_area.data(),
            sa_vert_mass = host_mesh_data->sa_vert_mass.data(),
            substep_dt = get_scene_params().get_substep_dt(),
            d_hat = get_scene_params().d_hat,
            floor_y = get_scene_params().floor.y,
            thickness = get_scene_params().thickness,
            stiffness_ground = 1e7f
        ](const uint vid)
    {
        if (sa_is_fixed[vid]) return;
        if (get_scene_params().use_floor)
        {
            float3 x_k = sa_x[vid];
            float diff = x_k.y - get_scene_params().floor.y;
    
            float3 force = luisa::make_float3(0.0f);
            float3x3 hessian = makeFloat3x3(0.0f);
            if (diff < d_hat + thickness)
            {
                float C = d_hat + thickness - diff;
                float3 normal = luisa::make_float3(0, 1, 0);
                float area = sa_rest_vert_area[vid];
                float stiff = stiffness_ground * area;
                force = stiff * C * normal;
                hessian = stiff * outer_product(normal, normal);
            }
            {  
                // sa_cgX[vid] = dx_0;
                sa_cgB[vid] = sa_cgB[vid] + force;
                sa_cgA_diag[vid] = sa_cgA_diag[vid] + hessian;
            }
        }
    });
    // if constexpr (use_eigen) 
    // {
    //     const uint prefix_triplets_A = 9 * vid;
    //     const uint prefix_triplets_b = 3 * vid;
    //     // Assemble diagonal 3x3 block for vertex vid
    //     for (int ii = 0; ii < 3; ++ii)
    //     {
    //         for (int jj = 0; jj < 3; ++jj)
    //         {
    //             triplets_groundA[prefix_triplets_A + ii * 3 + jj] = Eigen::Triplet<float>(3 * vid + ii, 3 * vid + jj, hessian[jj][ii]); // mat[i][j] is ok???
    //         }
    //     }
    //     eigen_cgB.segment<3>(prefix_triplets_b) = float3_to_eigen3(force);
    // }
    // else 
    // if constexpr (use_eigen) { eigen_groundA.setFromTriplets(triplets_groundA.begin(), triplets_groundA.end()); eigen_cgA += eigen_groundA; }
}
void NewtonSolver::host_evaluete_spring()
{
    const uint num_verts = host_mesh_data->num_verts;
    const uint num_edges = host_mesh_data->num_edges;
    
    // auto& culster = host_xpbd_data->sa_clusterd_springs;
    // auto& sa_edges = host_mesh_data->sa_edges;
    // auto& sa_rest_length = host_mesh_data->sa_stretch_spring_rest_state_length;
    
    auto& culster = host_sim_data->sa_prefix_merged_springs;

    for (uint cluster_idx = 0; cluster_idx < host_sim_data->num_clusters_springs; cluster_idx++) 
    {
        const uint curr_prefix = culster[cluster_idx];
        const uint next_prefix = culster[cluster_idx + 1];
        const uint num_elements_clustered = next_prefix - curr_prefix;

        // CpuParallel::single_thread_for(0, mesh_data->num_edges, [&](const uint eid)
        CpuParallel::parallel_for(0, num_elements_clustered, 
            [
                sa_x = host_sim_data->sa_x.data(),
                sa_edges = host_sim_data->sa_merged_stretch_springs.data(),
                sa_rest_length = host_sim_data->sa_merged_stretch_spring_rest_length.data(),
                sa_cgB = host_sim_data->sa_cgB.data(),
                sa_cgA_diag = host_sim_data->sa_cgA_diag.data(),
                sa_cgA_offdiag_stretch_spring = host_sim_data->sa_cgA_offdiag_stretch_spring.data(),
                curr_prefix, stiffness_stretch = get_scene_params().stiffness_spring
            ](const uint index)
        {
            // const uint eid = culster[curr_prefix + index];
            const uint eid = curr_prefix + index;
            
            uint2 edge = sa_edges[eid];

            float3 vert_pos[2] = { sa_x[edge[0]],sa_x[edge[1]] };
            float3 force[2] = {Zero3, Zero3};
            float3x3 He = luisa::make_float3x3(0.0f);

            const float L = sa_rest_length[eid];
            const float stiffness_stretch_spring = stiffness_stretch;

            float3 diff = vert_pos[1] - vert_pos[0];
            float l = max_scalar(length_vec(diff), Epsilon);
            float l0 = L;
            float C = l - l0;

            float3 dir = diff / l;
            float3x3 xxT = outer_product(diff, diff);
            float x_inv = 1.f / l;
            float x_squared_inv = x_inv * x_inv;

            force[0] = stiffness_stretch_spring * dir * C;
            force[1] = -force[0];
            He = stiffness_stretch_spring * x_squared_inv * xxT + stiffness_stretch_spring * max_scalar(1.0f - L * x_inv, 0.0f) * (luisa::make_float3x3(1.0f) - x_squared_inv * xxT);
            
            // Stable but Responsive Cloth
            // if (C > 0.0f)
            // {
            //     force[0] = stiffness_stretch_spring * C * dir;
            //     force[1] = -force[0];
            //     He = stiffness_stretch_spring * x_squared_inv * xxT + stiffness_stretch_spring * (1.0f - L * x_inv) * (luisa::make_float3x3(1.0f) - x_squared_inv * xxT);
            // }
            
            // SPD Projection
            // Eigen::Matrix<float, 6, 6> orig_hessian;
            // orig_hessian.block<3, 3>(0, 0) = float3x3_to_eigen3x3(He);
            // orig_hessian.block<3, 3>(3, 3) = float3x3_to_eigen3x3(He);
            // orig_hessian.block<3, 3>(0, 3) = float3x3_to_eigen3x3(-1.0f * He);
            // orig_hessian.block<3, 3>(3, 0) = float3x3_to_eigen3x3(-1.0f * He);
            // Eigen::Matrix<float, 6, 6> projected_hessian = spd_projection(orig_hessian);

            // if constexpr (use_eigen) 
            // {
            //     const uint prefix_triplets_A = 9 * eid * 4;
            //     const uint prefix_triplets_b = 3 * eid * 2;
            //     // Assemble 3x3 blocks for edge (off-diagonal and diagonal)
            //     for (int ii = 0; ii < 3; ++ii)
            //     {
            //         for (int jj = 0; jj < 3; ++jj) 
            //         {
            //             // Diagonal blocks
            //             triplets_springA[prefix_triplets_A + 4 * (ii * 3 + jj) + 0] = Eigen::Triplet<float>(3 * edge[0] + ii, 3 * edge[0] + jj, He[ii][jj]);
            //             triplets_springA[prefix_triplets_A + 4 * (ii * 3 + jj) + 1] = Eigen::Triplet<float>(3 * edge[1] + ii, 3 * edge[1] + jj, He[ii][jj]);
            //             // Off-diagonal blocks
            //             triplets_springA[prefix_triplets_A + 4 * (ii * 3 + jj) + 2] = Eigen::Triplet<float>(3 * edge[0] + ii, 3 * edge[1] + jj, -He[ii][jj]);
            //             triplets_springA[prefix_triplets_A + 4 * (ii * 3 + jj) + 3] = Eigen::Triplet<float>(3 * edge[1] + ii, 3 * edge[0] + jj, -He[ii][jj]);
            //         }
            //     }
            //     // Assemble force to gradient
            //     eigen_cgB.segment<3>(3 * edge[0]) += float3_to_eigen3(force[0]);
            //     eigen_cgB.segment<3>(3 * edge[1]) += float3_to_eigen3(force[1]);
            // }
            // else
            {
                sa_cgB[edge[0]] = sa_cgB[edge[0]] + force[0];
                sa_cgB[edge[1]] = sa_cgB[edge[1]] + force[1];
                sa_cgA_diag[edge[0]] = sa_cgA_diag[edge[0]] + He;
                sa_cgA_diag[edge[1]] = sa_cgA_diag[edge[1]] + He;
                // sa_cgA_diag[edge[0]] = sa_cgA_diag[edge[0]] + eigen3x3_to_float3x3(projected_hessian.block<3, 3>(0, 0));
                // sa_cgA_diag[edge[1]] = sa_cgA_diag[edge[1]] + eigen3x3_to_float3x3(projected_hessian.block<3, 3>(3, 3));
                
                // if constexpr (use_upper_triangle)
                // {
                //     for (uint ii = 0; ii < 2; ii++)
                //     {
                //         for (uint jj = ii + 1; jj < 2; jj++)
                //         {
                //             const uint hessian_index = host_sim_data->sa_hessian_slot_per_edge[eid];
                //             sa_cgA_offdiag_stretch_spring[hessian_index] = sa_cgA_offdiag_stretch_spring[hessian_index] - He;
                //         }
                //     }
                // }
                // else 
                {
                    sa_cgA_offdiag_stretch_spring[eid] = -1.0f * He;
                    // sa_cgA_offdiag_stretch_spring[eid * 2 + 0] = eigen3x3_to_float3x3(projected_hessian.block<3, 3>(0, 3));
                    // sa_cgA_offdiag_stretch_spring[eid * 2 + 1] = eigen3x3_to_float3x3(projected_hessian.block<3, 3>(3, 0));
                }
                // const uint num_offdiag_upper = 1;
                // uint edge_offset = 0;
                // for (uint ii = 0; ii < 2; ii++)
                // {
                //     for (uint jj = ii + 1; jj < 2; jj++)
                //     {
                //         const uint hessian_index = host_xpbd_data->sa_clusterd_hessian_slot_per_edge[num_offdiag_upper * eid + 0];
                //         edge_offset += 1;
                //         float3x3 offdiag_hessian = -1.0f * He;
                //         const bool need_transpose = edge[ii] > edge[jj];
                //         sa_cgA_offdiag_stretch_spring[hessian_index] = sa_cgA_offdiag_stretch_spring[hessian_index] + need_transpose ? luisa::transpose(offdiag_hessian) : offdiag_hessian;
                //     }
                // }
                
            }
        }, 32);
    }

    // if constexpr (use_eigen) 
    // {
    //     // Add spring contributions to cgA and cg_b_vec
    //     eigen_springA.setFromTriplets(triplets_springA.begin(), triplets_springA.end());
    //     eigen_cgA += eigen_springA;
    // }
}
void NewtonSolver::host_evaluete_bending()
{
    const uint num_verts = host_mesh_data->num_verts;
    const uint num_edges = host_mesh_data->num_edges;
    
    auto& culster = host_sim_data->sa_prefix_merged_bending_edges;
    for (uint cluster_idx = 0; cluster_idx < host_sim_data->num_clusters_bending_edges; cluster_idx++) 
    {
        const uint curr_prefix = culster[cluster_idx];
        const uint next_prefix = culster[cluster_idx + 1];
        const uint num_elements_clustered = next_prefix - curr_prefix;

        CpuParallel::parallel_for(0, num_elements_clustered, 
            [
                sa_x = host_sim_data->sa_x.data(),
                sa_bending_edges = host_sim_data->sa_merged_bending_edges.data(),
                sa_bending_edges_Q = host_sim_data->sa_merged_bending_edges_Q.data(),
                sa_cgB = host_sim_data->sa_cgB.data(),
                sa_cgA_diag = host_sim_data->sa_cgA_diag.data(),
                sa_cgA_offdiag_bending = host_sim_data->sa_cgA_offdiag_bending.data(),
                curr_prefix, stiffness_bending = get_scene_params().get_stiffness_quadratic_bending()
            ](const uint index)
        {
            // const uint eid = culster[curr_prefix + index];
            const uint eid = curr_prefix + index;
            
            uint4 edge = sa_bending_edges[eid];
            float4x4 m_Q = sa_bending_edges_Q[eid];

            float3 vert_pos[4] = { sa_x[edge[0]], sa_x[edge[1]], sa_x[edge[2]], sa_x[edge[3]] };
            
            float3 force[4] = {Zero3, Zero3, Zero3, Zero3 };
            for (uint ii = 0; ii < 4; ii++) 
            {
                for (uint jj = 0; jj < 4; jj++) 
                {
                    force[ii] -= m_Q[ii][jj] * vert_pos[jj]; // -Qx
                }
                force[ii] = stiffness_bending * force[ii];
            }

            // float3x3 hessian[10];
            {
                //  0   1   2   3
                // t1   4   5   6
                // t2  t5   7   8
                // t3  t6  t8   9 
                // hessian[0] = stiffness_bending * m_Q[0][0] * luisa::make_float3x3(1.0f);
                // hessian[1] = stiffness_bending * m_Q[0][1] * luisa::make_float3x3(1.0f);
                // hessian[2] = stiffness_bending * m_Q[0][2] * luisa::make_float3x3(1.0f);
                // hessian[3] = stiffness_bending * m_Q[0][3] * luisa::make_float3x3(1.0f);
                // hessian[4] = stiffness_bending * m_Q[1][1] * luisa::make_float3x3(1.0f);
                // hessian[5] = stiffness_bending * m_Q[1][2] * luisa::make_float3x3(1.0f);
                // hessian[6] = stiffness_bending * m_Q[1][3] * luisa::make_float3x3(1.0f);
                // hessian[7] = stiffness_bending * m_Q[2][2] * luisa::make_float3x3(1.0f);
                // hessian[8] = stiffness_bending * m_Q[2][3] * luisa::make_float3x3(1.0f);
                // hessian[9] = stiffness_bending * m_Q[3][3] * luisa::make_float3x3(1.0f);
            }
            
            {
                sa_cgB[edge[0]] = sa_cgB[edge[0]] + force[0];
                sa_cgB[edge[1]] = sa_cgB[edge[1]] + force[1];
                sa_cgB[edge[2]] = sa_cgB[edge[2]] + force[2];
                sa_cgB[edge[3]] = sa_cgB[edge[3]] + force[3];

                sa_cgA_diag[edge[0]] = sa_cgA_diag[edge[0]] + stiffness_bending * m_Q[0][0] * luisa::make_float3x3(1.0f);
                sa_cgA_diag[edge[1]] = sa_cgA_diag[edge[1]] + stiffness_bending * m_Q[1][1] * luisa::make_float3x3(1.0f);
                sa_cgA_diag[edge[2]] = sa_cgA_diag[edge[2]] + stiffness_bending * m_Q[2][2] * luisa::make_float3x3(1.0f);
                sa_cgA_diag[edge[3]] = sa_cgA_diag[edge[3]] + stiffness_bending * m_Q[3][3] * luisa::make_float3x3(1.0f);

                // sa_cgA_offdiag_bending[6 * eid + 0] = stiffness_bending * m_Q[0][1] * luisa::make_float3x3(1.0f);
                // sa_cgA_offdiag_bending[6 * eid + 1] = stiffness_bending * m_Q[0][2] * luisa::make_float3x3(1.0f);
                // sa_cgA_offdiag_bending[6 * eid + 2] = stiffness_bending * m_Q[0][3] * luisa::make_float3x3(1.0f);
                // sa_cgA_offdiag_bending[6 * eid + 3] = stiffness_bending * m_Q[1][2] * luisa::make_float3x3(1.0f);
                // sa_cgA_offdiag_bending[6 * eid + 4] = stiffness_bending * m_Q[1][3] * luisa::make_float3x3(1.0f);
                // sa_cgA_offdiag_bending[6 * eid + 5] = stiffness_bending * m_Q[2][3] * luisa::make_float3x3(1.0f);

                // sa_cgA_diag[edge[0]] = sa_cgA_diag[edge[0]] + hessian[0];
                // sa_cgA_diag[edge[1]] = sa_cgA_diag[edge[1]] + hessian[4];
                // sa_cgA_diag[edge[2]] = sa_cgA_diag[edge[2]] + hessian[7];
                // sa_cgA_diag[edge[3]] = sa_cgA_diag[edge[3]] + hessian[9];

                // sa_cgA_offdiag_bending[6 * eid + 0] = hessian[1];
                // sa_cgA_offdiag_bending[6 * eid + 1] = hessian[2];
                // sa_cgA_offdiag_bending[6 * eid + 2] = hessian[3];
                // sa_cgA_offdiag_bending[6 * eid + 3] = hessian[5];
                // sa_cgA_offdiag_bending[6 * eid + 4] = hessian[6];
                // sa_cgA_offdiag_bending[6 * eid + 5] = hessian[8];
            }
        }, 32);
    }

}

// Device functions
void NewtonSolver::device_broadphase_ccd(luisa::compute::Stream& stream)
{
    const float thickness = get_scene_params().thickness;
    const float ccd_query_range = thickness + 0; // + d_hat ???
    
    mp_narrowphase_detector->reset_broadphase_count(stream);

    mp_lbvh_face->update_face_tree_leave_aabb(stream, thickness, sim_data->sa_x_iter_start, sim_data->sa_x, mesh_data->sa_faces);
    mp_lbvh_face->refit(stream);
    mp_lbvh_face->broad_phase_query_from_verts(stream, 
        sim_data->sa_x_iter_start, 
        sim_data->sa_x, 
        collision_data->broad_phase_collision_count.view(collision_data->get_vf_count_offset(), 1), 
        collision_data->broad_phase_list_vf, ccd_query_range);

    mp_lbvh_edge->update_edge_tree_leave_aabb(stream, thickness, sim_data->sa_x_iter_start, sim_data->sa_x, mesh_data->sa_edges);
    mp_lbvh_edge->refit(stream);
    mp_lbvh_edge->broad_phase_query_from_edges(stream, 
        sim_data->sa_x_iter_start, 
        sim_data->sa_x, 
        mesh_data->sa_edges, 
        collision_data->broad_phase_collision_count.view(collision_data->get_ee_count_offset(), 1), 
        collision_data->broad_phase_list_ee, ccd_query_range);
}
void NewtonSolver::device_broadphase_dcd(luisa::compute::Stream& stream)
{
    const float thickness = get_scene_params().thickness;
    const float d_hat = get_scene_params().d_hat;
    const float dcd_query_range = d_hat + thickness;

    mp_lbvh_face->update_face_tree_leave_aabb(stream, thickness, sim_data->sa_x, sim_data->sa_x, mesh_data->sa_faces);
    mp_lbvh_face->refit(stream);
    mp_lbvh_face->broad_phase_query_from_verts(stream, 
        sim_data->sa_x, 
        sim_data->sa_x, 
        collision_data->broad_phase_collision_count.view(collision_data->get_vf_count_offset(), 1), 
        collision_data->broad_phase_list_vf, dcd_query_range);

    mp_lbvh_edge->update_edge_tree_leave_aabb(stream, thickness, sim_data->sa_x, sim_data->sa_x, mesh_data->sa_edges);
    mp_lbvh_edge->refit(stream);
    mp_lbvh_edge->broad_phase_query_from_edges(stream, 
        sim_data->sa_x, 
        sim_data->sa_x, 
        mesh_data->sa_edges, 
        collision_data->broad_phase_collision_count.view(collision_data->get_ee_count_offset(), 1), 
        collision_data->broad_phase_list_ee, dcd_query_range);
}
void NewtonSolver::device_narrowphase_ccd(luisa::compute::Stream& stream)
{
    const float thickness = get_scene_params().thickness;
    const float d_hat = get_scene_params().d_hat;

    // mp_narrowphase_detector->host_narrow_phase_ccd_query_from_vf_pair(stream, 
    //     host_sim_data->sa_x_iter_start, 
    //     host_sim_data->sa_x_iter_start, 
    //     host_sim_data->sa_x, 
    //     host_sim_data->sa_x, 
    //     host_mesh_data->sa_faces, 
    //     1e-3);

    // mp_narrowphase_detector->host_narrow_phase_ccd_query_from_ee_pair(stream, 
    //     host_sim_data->sa_x_iter_start, 
    //     host_sim_data->sa_x_iter_start, 
    //     host_sim_data->sa_x, 
    //     host_sim_data->sa_x, 
    //     host_mesh_data->sa_edges, 
    //     host_mesh_data->sa_edges, 
    //     1e-3);
    
    // mp_narrowphase_detector->reset_narrowphase_count(stream);
    mp_narrowphase_detector->reset_toi(stream);

    mp_narrowphase_detector->vf_ccd_query(stream, 
        sim_data->sa_x_iter_start, 
        sim_data->sa_x_iter_start, 
        sim_data->sa_x, 
        sim_data->sa_x, 
        mesh_data->sa_faces, 
        d_hat, thickness);

    mp_narrowphase_detector->ee_ccd_query(stream, 
        sim_data->sa_x_iter_start, 
        sim_data->sa_x_iter_start, 
        sim_data->sa_x, 
        sim_data->sa_x, 
        mesh_data->sa_edges, 
        mesh_data->sa_edges, 
        d_hat, thickness);
}
void NewtonSolver::device_narrowphase_dcd(luisa::compute::Stream& stream)
{
    const float thickness = get_scene_params().thickness;
    const float d_hat = get_scene_params().d_hat;
    const float kappa = get_scene_params().stiffness_collision;

    mp_narrowphase_detector->vf_dcd_query_repulsion(stream, 
        sim_data->sa_x, 
        sim_data->sa_x, 
        mesh_data->sa_rest_x,
        mesh_data->sa_rest_x,
        mesh_data->sa_rest_vert_area,
        mesh_data->sa_rest_face_area,
        mesh_data->sa_faces, 
        d_hat, thickness, kappa);
        
    mp_narrowphase_detector->ee_dcd_query_repulsion(stream, 
        sim_data->sa_x, 
        sim_data->sa_x,
        mesh_data->sa_rest_x,
        mesh_data->sa_rest_x, 
        mesh_data->sa_rest_edge_area, 
        mesh_data->sa_rest_edge_area, 
        mesh_data->sa_edges, 
        mesh_data->sa_edges, 
        d_hat, thickness, kappa);
}
void NewtonSolver::device_update_contact_list(luisa::compute::Stream& stream)
{
    mp_narrowphase_detector->reset_broadphase_count(stream);
    mp_narrowphase_detector->reset_narrowphase_count(stream);

    device_broadphase_dcd(stream);

    mp_narrowphase_detector->download_broadphase_collision_count(stream);
    
    device_narrowphase_dcd(stream);

    mp_narrowphase_detector->download_narrowphase_collision_count(stream);
}
void NewtonSolver::device_ccd_line_search(luisa::compute::Stream& stream)
{
    device_broadphase_ccd(stream);

    mp_narrowphase_detector->download_broadphase_collision_count(stream);
    
    device_narrowphase_ccd(stream);
}
float NewtonSolver::device_compute_contact_energy(luisa::compute::Stream& stream, const luisa::compute::Buffer<float3>& curr_x)
{
    // stream << sim_data->sa_x.copy_from(sa_x.data());
    const float thickness = get_scene_params().thickness;
    const float d_hat = get_scene_params().d_hat;
    const float kappa = get_scene_params().stiffness_collision;

    mp_narrowphase_detector->reset_energy(stream);
    mp_narrowphase_detector->compute_penalty_energy_from_vf(stream, 
        curr_x, 
        curr_x, 
        mesh_data->sa_rest_x,
        mesh_data->sa_rest_x,
        mesh_data->sa_rest_vert_area,
        mesh_data->sa_rest_face_area,
        mesh_data->sa_faces, 
        d_hat, thickness, kappa);

    mp_narrowphase_detector->compute_penalty_energy_from_ee(stream, 
        curr_x, 
        curr_x, 
        mesh_data->sa_rest_x,
        mesh_data->sa_rest_x,
        mesh_data->sa_rest_edge_area,
        mesh_data->sa_rest_edge_area,
        mesh_data->sa_edges, 
        mesh_data->sa_edges, 
        d_hat, thickness, kappa);

    return mp_narrowphase_detector->download_energy(stream);
    // return 0.0f;
}
void NewtonSolver::device_SpMV(luisa::compute::Stream& stream, const luisa::compute::Buffer<float3>& input_ptr, luisa::compute::Buffer<float3>& output_ptr)
{
    stream 
        << fn_pcg_spmv_diag(input_ptr, output_ptr).dispatch(input_ptr.size());

    stream << fn_pcg_spmv_offdiag_stretch_spring(input_ptr, output_ptr).dispatch(host_sim_data->sa_stretch_springs.size());
    stream << fn_pcg_spmv_offdiag_bending(input_ptr, output_ptr, get_scene_params().get_stiffness_quadratic_bending()).dispatch(host_sim_data->sa_bending_edges.size());
    
    mp_narrowphase_detector->device_spmv(stream, input_ptr, output_ptr);
}

void NewtonSolver::host_SpMV(luisa::compute::Stream& stream, const std::vector<float3>& input_ptr, std::vector<float3>& output_ptr)
{
    constexpr bool use_eigen = ConjugateGradientSolver::use_eigen;
    constexpr bool use_upper_triangle = ConjugateGradientSolver::use_upper_triangle;

    // Diag
    CpuParallel::parallel_for(0, input_ptr.size(), [&](const uint vid)
    {
        float3x3 A_diag = host_sim_data->sa_cgA_diag[vid];
        float3 input_vec = input_ptr[vid];
        float3 diag_output = A_diag * input_vec;
        output_ptr[vid] = diag_output;
    });
    {
        // Assemble free
        // auto& sa_edges = host_mesh_data->sa_edges;
        // auto& cluster = host_xpbd_data->sa_clusterd_springs;
        {
            auto& sa_edges = host_sim_data->sa_merged_stretch_springs;
            auto& cluster = host_sim_data->sa_prefix_merged_springs;
            auto& off_diag_hessian_ptr = host_sim_data->sa_cgA_offdiag_stretch_spring;
            for (uint cluster_idx = 0; cluster_idx < host_sim_data->num_clusters_springs; cluster_idx++) 
            {
                const uint curr_prefix = cluster[cluster_idx];
                const uint next_prefix = cluster[cluster_idx + 1];
                const uint num_elements_clustered = next_prefix - curr_prefix;

                CpuParallel::parallel_for(0, num_elements_clustered, [&](const uint index)
                {
                    // const uint eid = cluster[curr_prefix + index];
                    const uint eid = curr_prefix + index;
                    const uint2 edge = sa_edges[eid];

                    float3x3 offdiag_hessian1 = off_diag_hessian_ptr[eid];
                    float3x3 offdiag_hessian2 = luisa::transpose(offdiag_hessian1);
                    float3 output_vec0 = offdiag_hessian1 * input_ptr[edge[1]];
                    float3 output_vec1 = offdiag_hessian2 * input_ptr[edge[0]];
                    output_ptr[edge[0]] += output_vec0;
                    output_ptr[edge[1]] += output_vec1;
                });
            }
        }
        {
            auto& sa_bending_edges = host_sim_data->sa_merged_bending_edges;
            auto& sa_bending_edges_Q = host_sim_data->sa_merged_bending_edges_Q;
            auto& cluster = host_sim_data->sa_prefix_merged_bending_edges;
            auto& off_diag_hessian_ptr = host_sim_data->sa_cgA_offdiag_bending;
            const float stiffness_bending = get_scene_params().get_stiffness_quadratic_bending();
            for (uint cluster_idx = 0; cluster_idx < host_sim_data->num_clusters_bending_edges; cluster_idx++) 
            {
                const uint curr_prefix = cluster[cluster_idx];
                const uint next_prefix = cluster[cluster_idx + 1];
                const uint num_elements_clustered = next_prefix - curr_prefix;

                CpuParallel::parallel_for(0, num_elements_clustered, [&](const uint index)
                {
                    const uint eid = curr_prefix + index;
                    const uint4 edge = sa_bending_edges[eid];
                    const float4x4 m_Q = sa_bending_edges_Q[eid];

                    float3 input_vec[4] = {
                        input_ptr[edge[0]],
                        input_ptr[edge[1]],
                        input_ptr[edge[2]],
                        input_ptr[edge[3]],
                    }; 
                    float3 output_vec[4] = {
                        luisa::make_float3(0.0f),
                        luisa::make_float3(0.0f),
                        luisa::make_float3(0.0f),
                        luisa::make_float3(0.0f),
                    };
                    for (uint j = 0; j < 4; j++)
                    {
                        for (uint jj = 0; jj < 4; jj++)
                        {
                            if (j != jj)
                            {
                                float3x3 hessian = stiffness_bending * m_Q[j][jj] * luisa::compute::make_float3x3(1.0f);
                                output_vec[j] += hessian * input_vec[jj];
                            }
                        }
                    }
                    // uint offset = 0;
                    // for (uint j = 0; j < 4; j++)
                    // {
                    //     for (uint jj = j + 1; jj < 4; jj++)
                    //     {
                    //         float3x3 hessian = off_diag_hessian_ptr[6 * eid + offset]; offset += 1;
                    //         output_vec[j] += hessian * input_vec[jj];
                    //         output_vec[jj] += transpose(hessian) * input_vec[j];
                    //     }
                    // }
                    output_ptr[edge[0]] += output_vec[0];
                    output_ptr[edge[1]] += output_vec[1];
                    output_ptr[edge[2]] += output_vec[2];
                    output_ptr[edge[3]] += output_vec[3];
                });
            }
        }
    }

    // Off-diag: Collision hessian
    mp_narrowphase_detector->host_spmv_repulsion(stream, input_ptr, output_ptr);

}
void NewtonSolver::host_line_search(luisa::compute::Stream& stream)
{
    
}
static inline float fast_infinity_norm(const std::vector<float3>& ptr) // Min value in array
{
    return CpuParallel::parallel_for_and_reduce(0, ptr.size(), [&](const uint vid)
    {
        return luisa::length(ptr[vid]);
    }, [](const float left, const float right) { return max_scalar(left, right); }, -1e9f); 
};

void NewtonSolver::physics_step_CPU(luisa::compute::Device& device, luisa::compute::Stream& stream)
{
    std::vector<float3>& sa_x_tilde = host_sim_data->sa_x_tilde;
    std::vector<float3>& sa_x = host_sim_data->sa_x;
    std::vector<float3>& sa_v = host_sim_data->sa_v;
    std::vector<float3>& sa_x_step_start = host_sim_data->sa_x_step_start;
    std::vector<float3>& sa_x_iter_start = host_sim_data->sa_x_iter_start;
    std::vector<float3>& sa_v_step_start = host_sim_data->sa_v_step_start;

    // Input
    {
        lcs::SolverInterface::physics_step_prev_operation(); 
        CpuParallel::parallel_for(0, sa_x.size(), [&](const uint vid)
        {
            sa_x_step_start[vid] = host_mesh_data->sa_x_frame_outer[vid];
            sa_v_step_start[vid] = host_mesh_data->sa_v_frame_outer[vid];
            sa_x[vid] = host_mesh_data->sa_x_frame_outer[vid];
            sa_v[vid] = host_mesh_data->sa_v_frame_outer[vid];
        });
    }
    
    std::vector<float3>& sa_cgX = host_sim_data->sa_cgX;
    std::vector<float3>& sa_cgB = host_sim_data->sa_cgB;
    std::vector<float3x3>& sa_cgA_diag = host_sim_data->sa_cgA_diag;

    constexpr bool use_eigen = ConjugateGradientSolver::use_eigen;
    constexpr bool use_upper_triangle = ConjugateGradientSolver::use_upper_triangle;

    static Eigen::SparseMatrix<float> eigen_cgA;
    static Eigen::SparseMatrix<float> eigen_groundA;
    static Eigen::SparseMatrix<float> eigen_springA;
    static std::vector<Eigen::Triplet<float>> triplets_springA;
    static std::vector<Eigen::Triplet<float>> triplets_inertiaA;
    static std::vector<Eigen::Triplet<float>> triplets_groundA;
    static Eigen::VectorXf eigen_cgB; 
    static Eigen::VectorXf eigen_cgX;
    if constexpr (use_eigen) 
    {
        Eigen::setNbThreads(12);
        if (eigen_cgB.size() == 0)
        {
            eigen_cgB.resize(mesh_data->num_verts * 3);
            eigen_cgX.resize(mesh_data->num_verts * 3);
            eigen_cgA.resize(mesh_data->num_verts * 3, mesh_data->num_verts * 3);
            eigen_cgA.reserve(mesh_data->num_verts * 9 + mesh_data->num_edges * 9 * 2);
            eigen_springA.resize(mesh_data->num_verts * 3, mesh_data->num_verts * 3);
            eigen_springA.reserve(mesh_data->num_edges * 9 * 4);
            eigen_groundA.resize(mesh_data->num_verts * 3, mesh_data->num_verts * 3);
            eigen_groundA.reserve(mesh_data->num_verts * 9);
            triplets_inertiaA.resize(mesh_data->num_verts * 9);
            triplets_springA.resize(mesh_data->num_edges * 9 * 4);
            triplets_groundA.resize(mesh_data->num_verts * 9);
        }
    }

    auto host_apply_dx = [&](const float alpha)
    {
        if (alpha < 0.0f || alpha > 1.0f) { luisa::log_error("Alpha is not safe : {}", alpha); }
        // Update sa_x
        CpuParallel::parallel_for(0, mesh_data->num_verts, [&](const uint vid)
        {
            sa_x[vid] = sa_x_iter_start[vid] + alpha * sa_cgX[vid];
        });
    };
    auto pcg_spmv = [&](const std::vector<float3>& input_ptr, std::vector<float3>& output_ptr) -> void
    {   
        host_SpMV(stream, input_ptr, output_ptr);

    };


    const float thickness = get_scene_params().thickness;
    const float d_hat = get_scene_params().d_hat;
    const float kappa = 1e5;

    auto update_contact_set = [&]()
    {
        stream 
            // << sim_data->sa_x_iter_start.copy_from(host_sim_data->sa_x_iter_start.data())
            << sim_data->sa_x.copy_from(sa_x.data())
            ;
            
        device_update_contact_list(stream);
        mp_narrowphase_detector->download_narrowphase_list(stream);
    };
    auto evaluate_contact = [&]()
    {
        stream 
            << sim_data->sa_cgB.copy_from(sa_cgB.data())
            << sim_data->sa_cgA_diag.copy_from(sa_cgA_diag.data());

        mp_narrowphase_detector->compute_repulsion_gradiant_hessian_and_assemble(stream, sim_data->sa_x, sim_data->sa_x, d_hat, thickness, sim_data->sa_cgB, sim_data->sa_cgA_diag);

        stream 
            << sim_data->sa_cgB.copy_to(sa_cgB.data())
            << sim_data->sa_cgA_diag.copy_to(sa_cgA_diag.data())
            << luisa::compute::synchronize();
    };
    auto ccd_get_toi = [&]() -> float
    {
        stream 
            << sim_data->sa_x_iter_start.copy_from(host_sim_data->sa_x_iter_start.data())
            << sim_data->sa_x.copy_from(host_sim_data->sa_x.data())
            // << luisa::compute::synchronize()
            ;
        
        device_ccd_line_search(stream);
        
        float toi = mp_narrowphase_detector->get_global_toi(stream);
        return toi; // 0.9f * toi
        // return 1.0f;
    };

    // Solve
    auto simple_solve = [&]() 
    {
        // Actually is Jacobi Preconditioned Gradient-Descent (Without weighting)
        CpuParallel::parallel_for(0, mesh_data->num_verts, [&](const uint vid)
        {
            float3x3 hessian = sa_cgA_diag[vid];
            float3 f = sa_cgB[vid];

            float det = luisa::determinant(hessian);
            if (luisa::abs(det) > Epsilon) 
            {
                float3x3 H_inv = luisa::inverse(hessian);
                float3 dx = H_inv * f;
                float dt = lcs::get_scene_params().get_substep_dt();
                float3 vel =  dx / dt;
                sa_x[vid] += dx;
            };
        }); 
    };
    
    
    auto compute_energy_interface = [&](const std::vector<float3> &curr_x)
    {
        stream << sim_data->sa_x_tilde.copy_from(sa_x_tilde.data());
        stream << sim_data->sa_x.copy_from(curr_x.data());
        // auto material_energy = host_compute_elastic_energy(curr_x);
        auto material_energy = device_compute_elastic_energy(stream, sim_data->sa_x);
        auto barrier_energy = device_compute_contact_energy(stream, sim_data->sa_x);
        // luisa::log_info(".       Energy = {} + {}", material_energy, barrier_energy);
        auto total_energy = material_energy + barrier_energy;
        if (is_nan_scalar(material_energy) || is_inf_scalar(material_energy)) { luisa::log_error("Material energy is not valid : {}", material_energy); }
        if (is_nan_scalar(barrier_energy) || is_inf_scalar(barrier_energy)) { luisa::log_error("Barrier energy is not valid : {}", material_energy); }
        return total_energy;
    };
    auto linear_solver_interface = [&]()
    {
        if constexpr (use_eigen) 
        {
            pcg_solver->eigen_solve(eigen_cgA, eigen_cgX, eigen_cgB, compute_energy_interface);
            // eigen_decompose_solve();
        } 
        else 
        {
            // simple_solve();
            pcg_solver->host_solve(stream, pcg_spmv, compute_energy_interface);
        }
    };

    const float substep_dt = lcs::get_scene_params().get_substep_dt();
    const bool use_energy_linesearch = get_scene_params().use_energy_linesearch;
    const bool use_ccd_linesearch = get_scene_params().use_ccd_linesearch;
    
    // Init LBVH
    {
        stream << sim_data->sa_x_step_start.copy_from(host_sim_data->sa_x_step_start.data());
        mp_lbvh_face->reduce_face_tree_aabb(stream, sim_data->sa_x_step_start, mesh_data->sa_faces);
        mp_lbvh_edge->reduce_edge_tree_aabb(stream, sim_data->sa_x_step_start, mesh_data->sa_edges);
        mp_lbvh_face->construct_tree(stream);
        mp_lbvh_edge->construct_tree(stream);
        stream << luisa::compute::synchronize();
    }
    // for (uint substep = 0; substep < get_scene_params().num_substep; substep++)
    {
        host_predict_position();
        
        // double barrier_nergy = compute_barrier_energy_from_broadphase_list();
        double prev_state_energy = Float_max;

        luisa::log_info("In frame {}:", get_scene_params().current_frame); 

        for (uint iter = 0; iter < get_scene_params().nonlinear_iter_count; iter++)
        {   get_scene_params().current_nonlinear_iter = iter;
                        
            host_reset_cgB_cgX_diagA();

            host_reset_off_diag();
            {
                host_evaluate_inertia();
                
                host_evaluate_ground_collision();
                
                host_evaluete_spring();

                host_evaluete_bending();
                
                update_contact_set();
                
                evaluate_contact();

                // host_evaluate_dirichlet();
                
                // for (uint vid = 0; vid < host_mesh_data->num_verts; vid++) luisa::log_info("Post Vert {}'s force = {}", vid, host_sim_data->sa_cgB[vid]);
                // if (iter == 0) // Always refresh for collision count is variant 
                if (use_energy_linesearch) { prev_state_energy = compute_energy_interface(sa_x); }
            }

            linear_solver_interface(); // Solve Ax=b

            // EigenFloat12 eigen_x; 
            // for (uint vid = 0; vid < 4; vid++)
            // {
            //     eigen_x.block<3, 1>( vid * 3, 0) = float3_to_eigen3(sa_cgX[vid]);
            // }
            // std::cout << "PCG   result = " << eigen_x.transpose() << std::endl;

            float alpha = 1.0f; float ccd_toi = 1.0f;
            host_apply_dx(alpha);
            
            if (use_ccd_linesearch)
            {
                ccd_toi = ccd_get_toi();
                alpha = ccd_toi;
                host_apply_dx(alpha);   
            }

            if (use_energy_linesearch)
            { 
                // Energy after CCD or just solving Axb
                auto curr_energy = compute_energy_interface(sa_x); 
                if (is_nan_scalar(curr_energy) || is_inf_scalar(curr_energy)) { luisa::log_error("Energy is not valid : {}", curr_energy); }
                
                uint line_search_count = 0;
                while (line_search_count < 20) // Compare energy
                {
                    if (curr_energy < prev_state_energy + Epsilon) 
                    { 
                        if (alpha != 1.0f)
                        {
                            luisa::log_info("     Line search {} break : alpha = {:6.5f}, curr energy = {:12.10f} , prev energy {:12.10f} , {}", 
                                line_search_count, alpha, curr_energy, prev_state_energy, 
                                ccd_toi != 1.0f ? "CCD toi = " + std::to_string(ccd_toi) : "");
                        }
                        break; 
                    }
                    if (line_search_count == 0)
                    {
                        luisa::log_info("     Line search {} : alpha = {:6.5f}, energy = {:12.10f} , prev state energy {:12.10f} {}", 
                            line_search_count, alpha, curr_energy, prev_state_energy, 
                            ccd_toi != 1.0f ? ", CCD toi = " + std::to_string(ccd_toi) : "");
                    }
                    alpha /= 2; host_apply_dx(alpha); 

                    curr_energy = compute_energy_interface(sa_x);
                    luisa::log_info("     Line search {} : alpha = {:6.5f}, energy = {:12.10f}", 
                        line_search_count, alpha, curr_energy);
                    
                    if (alpha < 1e-4) 
                    {
                        luisa::log_error("  Line search failed, energy = {}, prev state energy = {}", 
                            curr_energy, prev_state_energy);
                    }
                    line_search_count++;
                }
                prev_state_energy = curr_energy; // E_prev = E
            }

            // Non-linear iteration break condition
            {
                float max_move = 1e-2;
                float curr_max_step = fast_infinity_norm(sa_cgX); 
                if (curr_max_step < max_move * substep_dt) 
                {
                    luisa::log_info("  In non-linear iter {:2}: Iteration break for small searching direction {} < {}", iter, curr_max_step, max_move * substep_dt);
                    break;
                }
            }

            CpuParallel::parallel_copy(sa_x, sa_x_iter_start); // x_prev = x
        }
        host_update_velocity();
    }

    // Output
    {
        CpuParallel::parallel_for(0, sa_x.size(), [&](const uint vid)
        {
            host_mesh_data->sa_x_frame_outer[vid] = sa_x[vid];
            host_mesh_data->sa_v_frame_outer[vid] = sa_v[vid];
        });
        lcs::SolverInterface::physics_step_post_operation(); 
    }
}
void NewtonSolver::physics_step_GPU(luisa::compute::Device& device, luisa::compute::Stream& stream)
{
    lcs::SolverInterface::physics_step_prev_operation(); 
    // Get frame start position and velocity
    CpuParallel::parallel_for(0, host_sim_data->sa_x.size(), [&](const uint vid)
    {
        host_sim_data->sa_x_step_start[vid] = host_mesh_data->sa_x_frame_outer[vid];
        host_sim_data->sa_v_step_start[vid] = host_mesh_data->sa_v_frame_outer[vid];
    });
    
    // Upload to GPU
    stream << sim_data->sa_x_step_start.copy_from(host_sim_data->sa_x_step_start.data())
           << sim_data->sa_v_step_start.copy_from(host_sim_data->sa_v_step_start.data())
           << sim_data->sa_x.copy_from(host_sim_data->sa_x_step_start.data())
           << sim_data->sa_v.copy_from(host_sim_data->sa_v_step_start.data())
        
           << luisa::compute::synchronize();
    
    // const uint num_substep = lcs::get_scene_params().print_xpbd_convergence ? 1 : lcs::get_scene_params().num_substep;
    const uint num_substep = lcs::get_scene_params().num_substep;
    const uint nonlinear_iter_count = lcs::get_scene_params().nonlinear_iter_count;
    const float substep_dt = lcs::get_scene_params().get_substep_dt();

    auto& host_x_step_start = host_sim_data->sa_x_step_start;
    auto& host_x_iter_start = host_sim_data->sa_x_iter_start;
    auto& host_cgX = host_sim_data->sa_cgX;
    auto& host_x = host_sim_data->sa_x;
    auto& host_x_tilde = host_sim_data->sa_x_tilde;

    auto host_apply_dx = [&](const float alpha)
    {
        // Update sa_x
        CpuParallel::parallel_for(0, mesh_data->num_verts, [&](const uint vid)
        {
            host_x[vid] = host_x_iter_start[vid] + alpha * host_cgX[vid];
        });
    };
    auto device_apply_dx = [&](const float alpha)
    {
        stream << fn_apply_dx(alpha).dispatch(mesh_data->num_verts);
    };

    const float thickness = get_scene_params().thickness;
    const float d_hat = get_scene_params().d_hat;
    auto update_contact_set = [&]()
    {
        device_update_contact_list(stream);
        mp_narrowphase_detector->download_narrowphase_list(stream);
    };
    auto evaluate_contact = [&]()
    {
        mp_narrowphase_detector->compute_repulsion_gradiant_hessian_and_assemble(stream, sim_data->sa_x, sim_data->sa_x, d_hat, thickness, sim_data->sa_cgB, sim_data->sa_cgA_diag);
    };
    auto ccd_get_toi = [&]() -> float
    {
        device_ccd_line_search(stream);
        float toi = mp_narrowphase_detector->get_global_toi(stream);
        return toi; // 0.9f * toi
    };

    const bool use_ipc = true;
    const uint num_verts = host_mesh_data->num_verts;
    const uint num_edges = host_mesh_data->num_edges;
    const uint num_faces = host_mesh_data->num_faces;
    const uint num_blocks_verts = get_dispatch_block(num_verts, 256);

    auto pcg_spmv = [&](const luisa::compute::Buffer<float3>& input_ptr, luisa::compute::Buffer<float3>& output_ptr) -> void
    {   
        device_SpMV(stream, input_ptr, output_ptr);
    };
    auto compute_energy_interface = [&](const luisa::compute::Buffer<float3>& curr_x)
    {
        auto material_energy = device_compute_elastic_energy(stream, curr_x);
        auto barrier_energy = device_compute_contact_energy(stream, curr_x);;
        // luisa::log_info(".       Energy = {} + {}", material_energy, barrier_energy);
        auto total_energy = material_energy + barrier_energy;
        if (is_nan_scalar(material_energy) || is_inf_scalar(material_energy)) { luisa::log_error("Material energy is not valid : {}", material_energy); }
        if (is_nan_scalar(barrier_energy) || is_inf_scalar(barrier_energy)) { luisa::log_error("Barrier energy is not valid : {}", material_energy); }

        return total_energy;
    };
    // Init LBVH
    {
        stream << sim_data->sa_x_step_start.copy_from(host_sim_data->sa_x_step_start.data());
        mp_lbvh_face->reduce_face_tree_aabb(stream, sim_data->sa_x_step_start, mesh_data->sa_faces);
        mp_lbvh_edge->reduce_edge_tree_aabb(stream, sim_data->sa_x_step_start, mesh_data->sa_edges);
        mp_lbvh_face->construct_tree(stream);
        mp_lbvh_edge->construct_tree(stream);
        stream << luisa::compute::synchronize();
    }
    // for (uint substep = 0; substep < get_scene_params().num_substep; substep++)
    {
        // host_predict_position();
        stream 
            << fn_predict_position(substep_dt).dispatch(num_verts)
            // << sim_data->sa_x_step_start.copy_to(host_x_step_start.data())
            << sim_data->sa_x_tilde.copy_to(host_x_tilde.data()) // For calculate inertia energy
            << luisa::compute::synchronize();
        
        double prev_state_energy = Float_max;

        luisa::log_info("In frame {}:", get_scene_params().current_frame); 

        for (uint iter = 0; iter < get_scene_params().nonlinear_iter_count; iter++)
        {   get_scene_params().current_nonlinear_iter = iter;
            
            stream 
                << sim_data->sa_x_iter_start.copy_to(host_x_iter_start.data())
                << fn_reset_vector(sim_data->sa_cgX).dispatch(num_verts)
                << fn_reset_vector(sim_data->sa_cgB).dispatch(num_verts)
                << fn_reset_float3x3(sim_data->sa_cgA_diag).dispatch(num_verts)
                << fn_reset_float3x3(sim_data->sa_cgA_offdiag_stretch_spring).dispatch(sim_data->sa_cgA_offdiag_stretch_spring.size())
                << fn_reset_float3x3(sim_data->sa_cgA_offdiag_bending).dispatch(sim_data->sa_cgA_offdiag_bending.size())
                ;
            
            {
                stream << fn_evaluate_inertia(substep_dt, get_scene_params().stiffness_dirichlet).dispatch(num_verts);
                
                stream << fn_evaluate_ground_collision(get_scene_params().floor.y, get_scene_params().use_floor, 1e7f, get_scene_params().d_hat, get_scene_params().thickness).dispatch(num_verts);
    
                stream << fn_evaluate_spring(get_scene_params().stiffness_spring).dispatch(host_sim_data->sa_stretch_springs.size());

                stream << fn_evaluate_bending(get_scene_params().get_stiffness_quadratic_bending()).dispatch(host_sim_data->sa_bending_edges.size());
                     
                update_contact_set();

                evaluate_contact();

                // stream << fn_evaluate_dirichlet(substep_dt, get_scene_params().stiffness_dirichlet).dispatch(num_verts);

                if (get_scene_params().use_energy_linesearch) prev_state_energy = compute_energy_interface(sim_data->sa_x);
            }

            pcg_solver->device_solve(stream, pcg_spmv, compute_energy_interface);

            float alpha = 1.0f; float ccd_toi = 1.0f;
            host_apply_dx(alpha);
            device_apply_dx(alpha);
            
            if (get_scene_params().use_ccd_linesearch)
            {
                ccd_toi = ccd_get_toi();
                alpha = ccd_toi;
                host_apply_dx(alpha);
                device_apply_dx(alpha);
            }

            if (get_scene_params().use_energy_linesearch)
            { 
                // Energy after CCD or just solving Axb
                auto curr_energy = compute_energy_interface(sim_data->sa_x); 
                if (is_nan_scalar(curr_energy) || is_inf_scalar(curr_energy)) { luisa::log_error("Energy is not valid : {}", curr_energy); }
                
                uint line_search_count = 0;
                while (line_search_count < 20) // Compare energy
                {
                    if (curr_energy < prev_state_energy + Epsilon) 
                    { 
                        if (alpha != 1.0f)
                        {
                            luisa::log_info("     Line search {} break : alpha = {:6.5f}, curr energy = {:12.10f} , prev energy {:12.10f} , {}", 
                                line_search_count, alpha, curr_energy, prev_state_energy, 
                                ccd_toi != 1.0f ? "CCD toi = " + std::to_string(ccd_toi) : "");
                        }
                        break; 
                    }
                    if (line_search_count == 0)
                    {
                        luisa::log_info("     Line search {} : alpha = {:6.5f}, energy = {:12.10f} , prev state energy {:12.10f} {}", 
                            line_search_count, alpha, curr_energy, prev_state_energy, 
                            ccd_toi != 1.0f ? ", CCD toi = " + std::to_string(ccd_toi) : "");
                    }
                    alpha /= 2; host_apply_dx(alpha); device_apply_dx(alpha);

                    curr_energy = compute_energy_interface(sim_data->sa_x);
                    luisa::log_info("     Line search {} : alpha = {:6.5f}, energy = {:12.10f}", 
                        line_search_count, alpha, curr_energy);
                    
                    if (alpha < 1e-4) 
                    {
                        luisa::log_error("  Line search failed, energy = {}, prev state energy = {}", 
                            curr_energy, prev_state_energy);
                    }
                    line_search_count++;
                }
                prev_state_energy = curr_energy; // E_prev = E
            }

            // Non-linear iteration break condition
            {
                float max_move = 1e-2;
                float curr_max_step = fast_infinity_norm(host_sim_data->sa_cgX); 
                if (curr_max_step < max_move * substep_dt) 
                {
                    luisa::log_info("  In non-linear iter {:2}: Iteration break for small searching direction {} < {}", iter, curr_max_step, max_move * substep_dt);
                    break;
                }
            }
            
            stream
                << sim_data->sa_x.copy_from(host_x.data())
                << sim_data->sa_x_iter_start.copy_from(host_x_iter_start.data());

            stream << sim_data->sa_x.copy_to(sim_data->sa_x_iter_start) << luisa::compute::synchronize();
        }
        stream << fn_update_velocity(substep_dt, get_scene_params().fix_scene, get_scene_params().damping_cloth).dispatch(num_verts);
    }

    stream << luisa::compute::synchronize();
    
    // Copy to host
    {
        stream  << sim_data->sa_x.copy_to(host_sim_data->sa_x.data())
                << sim_data->sa_v.copy_to(host_sim_data->sa_v.data())
                << luisa::compute::synchronize();
    }
    
    // Return frame end position and velocity
    CpuParallel::parallel_for(0, host_sim_data->sa_x.size(), [&](const uint vid)
    {
        host_mesh_data->sa_x_frame_outer[vid] = host_sim_data->sa_x[vid];
        host_mesh_data->sa_v_frame_outer[vid] = host_sim_data->sa_v[vid];
    });
    lcs::SolverInterface::physics_step_post_operation(); 
}

} // namespace lcs