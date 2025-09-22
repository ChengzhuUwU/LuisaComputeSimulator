#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include "CollisionDetector/libuipc/codim_ipc_simplex_normal_contact_function.h"
#include "Core/affine_position.h"
#include "Core/float_n.h"
#include "Initializer/init_mesh_data.h"
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

// AMGCL
#if defined(USE_AMGCL_FOR_SIM) && USE_AMGCL_FOR_SIM
#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/relaxation/spai0.hpp>
#endif

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

static inline float fast_infinity_norm(const std::vector<float3>& ptr) // Min value in array
{
    return CpuParallel::parallel_for_and_reduce(0, ptr.size(), [&](const uint vid)
    {
        return luisa::length(ptr[vid]);
    }, [](const float left, const float right) { return max_scalar(left, right); }, -1e9f); 
};

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
        ](const Float substep_dt, const Float3 gravity)
    {
        const UInt vid = dispatch_id().x;
        // const Float3 gravity = make_float3(0.0f, -9.8f, 0.0f);
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
    
    if (host_sim_data->sa_stretch_springs.size() > 0) fn_evaluate_spring = device.compile<1>(
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

    if (host_sim_data->sa_bending_edges.size() > 0) fn_evaluate_bending = device.compile<1>(
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


    if (host_sim_data->sa_stretch_springs.size() > 0) fn_pcg_spmv_offdiag_stretch_spring = device.compile<1>(
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
        Float3 input_vec[2] = {
            sa_input_vec->read(edge[0]),
            sa_input_vec->read(edge[1])
        };
        Float3 output_vec[2] = {
            make_float3(0.0f),
            make_float3(0.0f)
        };

        output_vec[0] = offdiag_hessian1 * input_vec[1];
        output_vec[1] = offdiag_hessian2 * input_vec[0];

        atomic_buffer_add(sa_output_vec, edge[0], output_vec[0]);
        atomic_buffer_add(sa_output_vec, edge[1], output_vec[1]);
        // buffer_add(sa_output_vec, edge[0], output1);
        // buffer_add(sa_output_vec, edge[1], output2);
    }, default_option);

    if (host_sim_data->sa_bending_edges.size() > 0) fn_pcg_spmv_offdiag_bending = device.compile<1>(
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
    CpuParallel::parallel_for(0, host_sim_data->num_verts_soft, 
        [
            sa_x = host_sim_data->sa_x.data(),
            sa_v = host_sim_data->sa_v.data(),
            sa_cgX = host_sim_data->sa_cgX.data(),
            sa_x_step_start = host_sim_data->sa_x_step_start.data(),
            sa_x_iter_start = host_sim_data->sa_x_iter_start.data(),
            sa_x_tilde = host_sim_data->sa_x_tilde.data(),
            sa_is_fixed = host_mesh_data->sa_is_fixed.data(),
            substep_dt = get_scene_params().get_substep_dt(),
            gravity = get_scene_params().gravity
        ](const uint vid)
    {   
        // const float3 gravity(0, -9.8f, 0);
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

    // Vectorization
    CpuParallel::parallel_for(0, host_sim_data->sa_affine_bodies.size() * 4, [&](const uint block_idx)
    {
        float3 q_prev = host_sim_data->sa_affine_bodies_q_step_start[block_idx];
        float3 q_v = host_sim_data->sa_affine_bodies_q_v[block_idx];
        // float3 g = host_sim_data->sa_affine_bodies_gravity[block_idx];
        float3 g = get_scene_params().gravity;

        float substep_dt = get_scene_params().get_substep_dt();
        float3 q_pred = q_prev + q_v * substep_dt;
        if (block_idx % 4 == 0) q_pred += g * (substep_dt * substep_dt);
        
        // Output
        host_sim_data->sa_affine_bodies_q_tilde[block_idx] = q_pred;
        host_sim_data->sa_affine_bodies_q_iter_start[block_idx] = q_prev;
        host_sim_data->sa_affine_bodies_q[block_idx] = q_prev;
        // luisa::log_info("Body {}'s block_{} : q = {}, v = {} , dt = {} => q_tilde = {}", block_idx / 4, block_idx % 4, q_prev, q_v, substep_dt, q_pred);
    });
}
void NewtonSolver::host_update_velocity()
{
    CpuParallel::parallel_for(0, host_sim_data->num_verts_soft, 
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

    // Vectorization
    CpuParallel::parallel_for(0, host_sim_data->sa_affine_bodies.size() * 4, [&](const uint block_idx)
    {
        const float substep_dt = get_scene_params().get_substep_dt();
        const float damping = get_scene_params().damping_tet;

        float3 q_step_begin = host_sim_data->sa_affine_bodies_q_step_start[block_idx];
        float3 q_step_end = host_sim_data->sa_affine_bodies_q[block_idx];

        float3 vq = (q_step_end - q_step_begin) / substep_dt * exp(-damping * substep_dt);
        host_sim_data->sa_affine_bodies_q_v[block_idx] = vq;
        host_sim_data->sa_affine_bodies_q_step_start[block_idx] = q_step_end;
        // luisa::log_info("Body {} 's block {} : vel = {} = from {} to {}", block_idx / 4, block_idx, vq, q_step_begin, q_step_end);
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
        CpuParallel::parallel_set(host_sim_data->sa_cgA_offdiag_affine_body, luisa::make_float3x3(0.0f));
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
constexpr bool print_detail = false;
void NewtonSolver::host_evaluate_inertia()
{
    const float stiffness_dirichlet = get_scene_params().stiffness_dirichlet;

    CpuParallel::parallel_for(0, host_sim_data->num_verts_soft,
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
            if constexpr (print_detail) luisa::log_info("vid {}, mass: {}, move = {}, gradient: {}, hessian: {}", vid, mass, length_vec(x_k - x_tilde), gradient, hessian);
            // sa_cgX[vid] = dx_0;
            sa_cgB[vid] = gradient;
            sa_cgA_diag[vid] = hessian;
        }
    });

    float3* affine_body_cgB = &host_sim_data->sa_cgB[host_sim_data->num_verts_soft];
    float3x3* affine_body_cgA_diag = &host_sim_data->sa_cgA_diag[host_sim_data->num_verts_soft];
    CpuParallel::parallel_for(0, host_sim_data->sa_affine_bodies.size(), [&](const uint body_idx)
    {
        const float substep_dt = get_scene_params().get_substep_dt();
        const float h = substep_dt;
        const float h_2_inv = 1.f / (h * h);

        float3 q_k = host_sim_data->sa_affine_bodies_q[body_idx];
        float3 q_tilde = host_sim_data->sa_affine_bodies_q_tilde[body_idx];
        float3 delta[4] = {
            host_sim_data->sa_affine_bodies_q[4 * body_idx + 0] - host_sim_data->sa_affine_bodies_q_tilde[4 * body_idx + 0],
            host_sim_data->sa_affine_bodies_q[4 * body_idx + 1] - host_sim_data->sa_affine_bodies_q_tilde[4 * body_idx + 1],
            host_sim_data->sa_affine_bodies_q[4 * body_idx + 2] - host_sim_data->sa_affine_bodies_q_tilde[4 * body_idx + 2],
            host_sim_data->sa_affine_bodies_q[4 * body_idx + 3] - host_sim_data->sa_affine_bodies_q_tilde[4 * body_idx + 3],
        };
        float3x3 offdiags = host_sim_data->sa_affine_bodies_mass_matrix_compressed_offdiag[body_idx];
        float3x3 offdiag_first_col[3] = { 
            luisa::make_float3x3(offdiags[0], Zero3, Zero3), 
            luisa::make_float3x3(Zero3, offdiags[1], Zero3), 
            luisa::make_float3x3(Zero3, Zero3, offdiags[2]) };
        
        float3 gradient[4] = { Zero3, Zero3, Zero3, Zero3 };
        for (uint ii = 0; ii < 4; ii++)
        {
            gradient[ii] += host_sim_data->sa_affine_bodies_mass_matrix_diag[4 * body_idx + ii] * delta[ii];
        }
        for (uint ii = 0; ii < 3; ii++)
        {
            gradient[1 + ii] += offdiag_first_col[ii] * delta[0]; // First column offdiag
            gradient[0] += luisa::transpose(offdiag_first_col[ii]) * delta[1 + ii]; // First row offdiag
        }

        affine_body_cgB[4 * body_idx + 0] = -h_2_inv * gradient[0];
        affine_body_cgB[4 * body_idx + 1] = -h_2_inv * gradient[1];
        affine_body_cgB[4 * body_idx + 2] = -h_2_inv * gradient[2];
        affine_body_cgB[4 * body_idx + 3] = -h_2_inv * gradient[3];

        affine_body_cgA_diag[4 * body_idx + 0] = h_2_inv * host_sim_data->sa_affine_bodies_mass_matrix_diag[4 * body_idx + 0];
        affine_body_cgA_diag[4 * body_idx + 1] = h_2_inv * host_sim_data->sa_affine_bodies_mass_matrix_diag[4 * body_idx + 1];
        affine_body_cgA_diag[4 * body_idx + 2] = h_2_inv * host_sim_data->sa_affine_bodies_mass_matrix_diag[4 * body_idx + 2];
        affine_body_cgA_diag[4 * body_idx + 3] = h_2_inv * host_sim_data->sa_affine_bodies_mass_matrix_diag[4 * body_idx + 3];

        host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 0] = h_2_inv * luisa::transpose(offdiag_first_col[0]);
        host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 1] = h_2_inv * luisa::transpose(offdiag_first_col[1]);
        host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 2] = h_2_inv * luisa::transpose(offdiag_first_col[2]);
    });
}
void NewtonSolver::host_evaluate_orthogonality()
{
    float3* affine_body_cgB = &host_sim_data->sa_cgB[host_sim_data->num_verts_soft];
    float3x3* affine_body_cgA_diag = &host_sim_data->sa_cgA_diag[host_sim_data->num_verts_soft];

    CpuParallel::parallel_for_each_core(0, host_sim_data->sa_affine_bodies.size(), [&](const uint body_idx)
    {
        float3 body_force[4] = { Zero3 };
        float3x3 body_hessian[6] = { Zero3x3 };

        const float substep_dt = get_scene_params().get_substep_dt();
        const float h = substep_dt;
        const float h_2_inv = 1.f / (h * h);

        float3 A[3] = {
            host_sim_data->sa_affine_bodies_q[4 * body_idx + 1],
            host_sim_data->sa_affine_bodies_q[4 * body_idx + 2],
            host_sim_data->sa_affine_bodies_q[4 * body_idx + 3]
        };
        const float kappa = 1e5f;
        const float V = host_sim_data->sa_affine_bodies_volume[body_idx];

        float stiff = kappa ; //* V;
        for (uint ii = 0; ii < 3; ii++)
        {
            float3 grad = (-1.0f) * A[ii];
            for (uint jj = 0; jj < 3; jj++)
            {
                grad += dot_vec(A[ii], A[jj]) * A[jj];
            }
            // cgB.block<3, 1>(3 + 3 * ii, 0) -= 4 * stiff * float3_to_eigen3(grad);
            body_force[1 + ii] -= 4 * stiff * grad; // Force
        }
        uint idx = 0;
        for (uint ii = 0; ii < 3; ii++)
        {
            for (uint jj = ii; jj < 3; jj++)
            {
                float3x3 hessian = Zero3x3;
                if (ii == jj)
                {
                    float3x3 qiqiT = outer_product(A[ii], A[ii]);
                    float qiTqi = dot_vec(A[ii], A[ii]) - 1.0f;
                    float3x3 term2 = qiTqi * Identity3x3;
                    for (uint kk = 0; kk < 3; kk++)
                    {
                        hessian = hessian + outer_product(A[kk], A[kk]);
                    }
                    hessian = hessian + qiqiT + term2;
                }
                else
                {
                    hessian = outer_product(A[jj], A[ii]) + dot_vec(A[ii], A[jj]) * Identity3x3;
                }
                // luisa::log_info("hess of {} adj {} = {}", ii, jj, hessian);
                // cgA.block<3, 3>(3 + 3 * ii, 3 + 3 * jj) += 4.0f * stiff * float3x3_to_eigen3x3(hessian);
                body_hessian[idx] = body_hessian[idx] + 4.0f * stiff * hessian; idx += 1;
            }
        }

        //  0   1   2   3
        // t1   4   5   6
        // t2  t5   7   8
        // t3  t6  t8   9 
        affine_body_cgB[4 * body_idx + 0] += body_force[0];
        affine_body_cgB[4 * body_idx + 1] += body_force[1];
        affine_body_cgB[4 * body_idx + 2] += body_force[2];
        affine_body_cgB[4 * body_idx + 3] += body_force[3];
        affine_body_cgA_diag[4 * body_idx + 1] = affine_body_cgA_diag[4 * body_idx + 1] + body_hessian[0];
        affine_body_cgA_diag[4 * body_idx + 2] = affine_body_cgA_diag[4 * body_idx + 2] + body_hessian[3];
        affine_body_cgA_diag[4 * body_idx + 3] = affine_body_cgA_diag[4 * body_idx + 3] + body_hessian[5];
        host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 3] = host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 3] + body_hessian[1];
        host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 4] = host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 4] + body_hessian[2];
        host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 5] = host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 5] + body_hessian[4];
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

    const uint num_verts = host_sim_data->num_verts_soft;
    const float floor_y = get_scene_params().floor.y; 
    float d_hat = get_scene_params().d_hat;
    float thickness = get_scene_params().thickness;
    float stiffness_ground = 1e7f;

    CpuParallel::parallel_for(0, host_sim_data->num_verts_soft, 
        [
            sa_cgB = host_sim_data->sa_cgB.data(),
            sa_cgA_diag = host_sim_data->sa_cgA_diag.data(),
            sa_x = host_sim_data->sa_x.data(),
            sa_is_fixed = host_mesh_data->sa_is_fixed.data(),
            sa_rest_vert_area = host_mesh_data->sa_rest_vert_area.data(),
            sa_vert_mass = host_mesh_data->sa_vert_mass.data(),
            substep_dt = get_scene_params().get_substep_dt(),
            d_hat = d_hat,
            floor_y = floor_y,
            thickness = thickness,
            stiffness_ground = stiffness_ground
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

    float3* affine_body_cgB = &host_sim_data->sa_cgB[host_sim_data->num_verts_soft];
    float3x3* affine_body_cgA_diag = &host_sim_data->sa_cgA_diag[host_sim_data->num_verts_soft];

    CpuParallel::parallel_for_each_core(0, host_sim_data->sa_affine_bodies.size(), 
        [&](const uint body_idx)
    {
        const uint mesh_idx = host_sim_data->sa_affine_bodies[body_idx];
        const uint curr_prefix = host_mesh_data->prefix_num_verts[mesh_idx];
        const uint next_prefix = host_mesh_data->prefix_num_verts[mesh_idx + 1];
        
        float3 body_force[4] = { Zero3 };
        float3x3 body_hessian[10] = { Zero3x3 };
        // EigenFloat12 eigen_B = EigenFloat12::Zero();
        // EigenFloat12x12 eigen_A = EigenFloat12x12::Zero();
        for (uint vid = curr_prefix; vid < next_prefix; vid++)
        {
            float3 x_k = host_sim_data->sa_x[vid];
            float diff = x_k.y - floor_y;

            if (diff < d_hat + thickness)
            {
                float C = d_hat + thickness - diff;
                float3 normal = luisa::make_float3(0, 1, 0);
                float area = sa_rest_vert_area[vid];
                float stiff = 1e9f * area;
                float k1 = stiff * C;
                float3 model_x = host_mesh_data->sa_scaled_model_x[vid];
                float3 force = stiff * C * normal;
                float3x3 hessian = stiff * outer_product(normal, normal);
                {
                    float3x3 curr_hessian[10]; float3 curr_force[4];
                    AffineBodyDynamics::affine_Jacobian_to_gradient(model_x, force, curr_force);
                    AffineBodyDynamics::affine_Jacobian_to_hessian(model_x, model_x, hessian, curr_hessian);
                    for (uint jj = 0; jj < 4; jj++) { body_force[jj] += curr_force[jj]; }
                    for (uint jj = 0; jj < 10; jj++) { body_hessian[jj] = body_hessian[jj] + curr_hessian[jj]; }
                    // for (uint jj = 0; jj < 4; jj++) { luisa::log_info("For vert {} : force = {}", vid, curr_force[jj]); }
                    // for (uint jj = 0; jj < 10; jj++) { luisa::log_info("For vert {} : hess = {}", vid, curr_hessian[jj]); }
                }
            }
        }
        //  0   1   2   3
        // t1   4   5   6
        // t2  t5   7   8
        // t3  t6  t8   9 
        
        affine_body_cgB[4 * body_idx + 0] += body_force[0];
        affine_body_cgB[4 * body_idx + 1] += body_force[1];
        affine_body_cgB[4 * body_idx + 2] += body_force[2];
        affine_body_cgB[4 * body_idx + 3] += body_force[3];
        affine_body_cgA_diag[4 * body_idx + 0] = affine_body_cgA_diag[4 * body_idx + 0] + body_hessian[0];
        affine_body_cgA_diag[4 * body_idx + 1] = affine_body_cgA_diag[4 * body_idx + 1] + body_hessian[4];
        affine_body_cgA_diag[4 * body_idx + 2] = affine_body_cgA_diag[4 * body_idx + 2] + body_hessian[7];
        affine_body_cgA_diag[4 * body_idx + 3] = affine_body_cgA_diag[4 * body_idx + 3] + body_hessian[9];
        host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 0] = host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 0] + body_hessian[1];
        host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 1] = host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 1] + body_hessian[2];
        host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 2] = host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 2] + body_hessian[3];
        host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 3] = host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 3] + body_hessian[5];
        host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 4] = host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 4] + body_hessian[6];
        host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 5] = host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 5] + body_hessian[8];
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
void NewtonSolver::host_test_affine_body(luisa::compute::Stream& stream)
{
    EigenFloat12x12 cgA = EigenFloat12x12::Zero();
    EigenFloat12 cgB = EigenFloat12::Zero();

    // Inertia
    if constexpr (false)
    {
        CpuParallel::single_thread_for(0, host_sim_data->sa_affine_bodies.size(), [&](const uint body_idx)
        {
            const float substep_dt = get_scene_params().get_substep_dt();
            const float h = substep_dt;
            const float h_2_inv = 1.f / (h * h);
    
            auto M = host_sim_data->sa_affine_bodies_mass_matrix_full[body_idx];
            EigenFloat12 delta = EigenFloat12::Zero();
            
            delta.block<3, 1>(0, 0) = float3_to_eigen3(host_sim_data->sa_affine_bodies_q[4 * body_idx + 0] - host_sim_data->sa_affine_bodies_q_tilde[4 * body_idx + 0]);
            delta.block<3, 1>(3, 0) = float3_to_eigen3(host_sim_data->sa_affine_bodies_q[4 * body_idx + 1] - host_sim_data->sa_affine_bodies_q_tilde[4 * body_idx + 1]);
            delta.block<3, 1>(6, 0) = float3_to_eigen3(host_sim_data->sa_affine_bodies_q[4 * body_idx + 2] - host_sim_data->sa_affine_bodies_q_tilde[4 * body_idx + 2]);
            delta.block<3, 1>(9, 0) = float3_to_eigen3(host_sim_data->sa_affine_bodies_q[4 * body_idx + 3] - host_sim_data->sa_affine_bodies_q_tilde[4 * body_idx + 3]);
            
            EigenFloat12 gradient = h_2_inv * M * delta;
            EigenFloat12x12 hessian = h_2_inv * M;
    
            cgB -= gradient;
            cgA += hessian;
        });
    }

    // Ground collision
    if constexpr (false)
    {
        const float d_hat = get_scene_params().d_hat ;
        const float thickness = get_scene_params().thickness;
        CpuParallel::single_thread_for(0, host_sim_data->sa_affine_bodies.size(), [&](const uint body_idx)
        {
            if (get_scene_params().use_floor)
            {
                const uint mesh_idx = host_sim_data->sa_affine_bodies[body_idx];
                const uint curr_prefix = host_mesh_data->prefix_num_verts[mesh_idx];
                const uint next_prefix = host_mesh_data->prefix_num_verts[mesh_idx + 1];
                
                float3 body_force[4] = { Zero3 };
                float3x3 body_hessian[4][10] = { Zero3x3 };
                // EigenFloat12 eigen_B = EigenFloat12::Zero();
                // EigenFloat12x12 eigen_A = EigenFloat12x12::Zero();
                for (uint vid = curr_prefix; vid < next_prefix; vid++)
                {
                    float3 x_k = host_sim_data->sa_x[vid];
                    float diff = x_k.y - get_scene_params().floor.y;
        
                    if (diff < d_hat + thickness)
                    {
                        float C = d_hat + thickness - diff;
                        float3 normal = luisa::make_float3(0, 1, 0);
                        float area = host_mesh_data->sa_rest_vert_area[vid];
                        float stiff = 1e9f * area;
                        float k1 = stiff * C;
                        float3 model_x = host_mesh_data->sa_scaled_model_x[vid];
                        float3 force = stiff * C * normal;
                        float3x3 hessian = stiff * outer_product(normal, normal);
                        auto J = AffineBodyDynamics::get_jacobian_dxdq(model_x);
                        cgB += J.transpose() * float3_to_eigen3(force);
                        cgA += J.transpose() * float3x3_to_eigen3x3(hessian) * J;
                        //  0   1   2   3
                        // t1   4   5   6
                        // t2  t5   7   8
                        // t3  t6  t8   9 
                    }
                }
                float3* affine_body_cgB = &host_sim_data->sa_cgB[host_sim_data->num_verts_soft];
                float3x3* affine_body_cgA_diag = &host_sim_data->sa_cgA_diag[host_sim_data->num_verts_soft];
                
                const uint body_idx = 0;
                affine_body_cgB[4 * body_idx + 0] += eigen3_to_float3(cgB.block<3, 1>(0, 0));
                affine_body_cgB[4 * body_idx + 1] += eigen3_to_float3(cgB.block<3, 1>(3, 0));
                affine_body_cgB[4 * body_idx + 2] += eigen3_to_float3(cgB.block<3, 1>(6, 0));
                affine_body_cgB[4 * body_idx + 3] += eigen3_to_float3(cgB.block<3, 1>(9, 0));
                affine_body_cgA_diag[4 * body_idx + 0] = affine_body_cgA_diag[4 * body_idx + 0] + eigen3x3_to_float3x3(cgA.block<3, 3>(0, 0));
                affine_body_cgA_diag[4 * body_idx + 1] = affine_body_cgA_diag[4 * body_idx + 1] + eigen3x3_to_float3x3(cgA.block<3, 3>(3, 3));
                affine_body_cgA_diag[4 * body_idx + 2] = affine_body_cgA_diag[4 * body_idx + 2] + eigen3x3_to_float3x3(cgA.block<3, 3>(6, 6));
                affine_body_cgA_diag[4 * body_idx + 3] = affine_body_cgA_diag[4 * body_idx + 3] + eigen3x3_to_float3x3(cgA.block<3, 3>(9, 9));
                host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 0] = host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 0] + eigen3x3_to_float3x3(cgA.block<3, 3>(0, 3));
                host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 1] = host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 1] + eigen3x3_to_float3x3(cgA.block<3, 3>(0, 6));
                host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 2] = host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 2] + eigen3x3_to_float3x3(cgA.block<3, 3>(0, 9));
                host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 3] = host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 3] + eigen3x3_to_float3x3(cgA.block<3, 3>(3, 6));
                host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 4] = host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 4] + eigen3x3_to_float3x3(cgA.block<3, 3>(3, 9));
                host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 5] = host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 5] + eigen3x3_to_float3x3(cgA.block<3, 3>(6, 9));
            }
        });
    }

    const float d_hat = get_scene_params().d_hat ;
    const float thickness = get_scene_params().thickness;
    CpuParallel::single_thread_for(0, host_sim_data->sa_affine_bodies.size(), 
        [&](const uint body_idx)
    {
        const uint mesh_idx = host_sim_data->sa_affine_bodies[body_idx];
        const uint curr_prefix = host_mesh_data->prefix_num_verts[mesh_idx];
        const uint next_prefix = host_mesh_data->prefix_num_verts[mesh_idx + 1];
        
        float3 body_force[4] = { Zero3 };
        float3x3 body_hessian[4][10] = { Zero3x3 };
        // EigenFloat12 eigen_B = EigenFloat12::Zero();
        // EigenFloat12x12 eigen_A = EigenFloat12x12::Zero();

        // Orthogonality potential 
        if constexpr (false)
        {
            float3 A[3] = {
                host_sim_data->sa_affine_bodies_q[4 * body_idx + 1],
                host_sim_data->sa_affine_bodies_q[4 * body_idx + 2],
                host_sim_data->sa_affine_bodies_q[4 * body_idx + 3]
            };
            const float kappa = 1e5f;
            const float V = host_sim_data->sa_affine_bodies_volume[body_idx];

            float stiff = kappa ; //* V;
            for (uint ii = 0; ii < 3; ii++)
            {
                float3 grad = (-1.0f) * A[ii];
                for (uint jj = 0; jj < 3; jj++)
                {
                     grad += dot_vec(A[ii], A[jj]) * A[jj];
                }
                cgB.block<3, 1>(3 + 3 * ii, 0) -= 4 * stiff * float3_to_eigen3(grad);
                // body_force[1 + ii] -= 4 * stiff * g; // Force
                // luisa::log_info("Force of col {} = {}", 1 + ii, g);
            }
            for (uint ii = 0; ii < 3; ii++)
            {
                for (uint jj = 0; jj < 3; jj++)
                {
                    float3x3 hessian = Zero3x3;
                    if (ii == jj)
                    {
                        float3x3 qiqiT = outer_product(A[ii], A[ii]);
                        float qiTqi = dot_vec(A[ii], A[ii]) - 1.0f;
                        float3x3 term2 = qiTqi * Identity3x3;
                        for (uint kk = 0; kk < 3; kk++)
                        {
                            hessian = hessian + outer_product(A[kk], A[kk]);
                        }
                        hessian = hessian + qiqiT + term2;
                    }
                    else 
                    {
                        hessian = outer_product(A[jj], A[ii]) + dot_vec(A[ii], A[jj]) * Identity3x3;
                    }
                    // luisa::log_info("hess of {} adj {} = {}", ii, jj, hessian);
                    cgA.block<3, 3>(3 + 3 * ii, 3 + 3 * jj) += 4.0f * stiff * float3x3_to_eigen3x3(hessian);
                    // body_hessian[ii][jj] = body_hessian[ii][jj] + 4.0f * stiff * hessian; idx += 1;
                }
            }
        }
    });

    float3* affine_body_cgB = &host_sim_data->sa_cgB[host_sim_data->num_verts_soft];
    float3x3* affine_body_cgA_diag = &host_sim_data->sa_cgA_diag[host_sim_data->num_verts_soft];
    
    const uint body_idx = 0;
    affine_body_cgB[4 * body_idx + 0] += eigen3_to_float3(cgB.block<3, 1>(0, 0));
    affine_body_cgB[4 * body_idx + 1] += eigen3_to_float3(cgB.block<3, 1>(3, 0));
    affine_body_cgB[4 * body_idx + 2] += eigen3_to_float3(cgB.block<3, 1>(6, 0));
    affine_body_cgB[4 * body_idx + 3] += eigen3_to_float3(cgB.block<3, 1>(9, 0));
    affine_body_cgA_diag[4 * body_idx + 0] = affine_body_cgA_diag[4 * body_idx + 0] + eigen3x3_to_float3x3(cgA.block<3, 3>(0, 0));
    affine_body_cgA_diag[4 * body_idx + 1] = affine_body_cgA_diag[4 * body_idx + 1] + eigen3x3_to_float3x3(cgA.block<3, 3>(3, 3));
    affine_body_cgA_diag[4 * body_idx + 2] = affine_body_cgA_diag[4 * body_idx + 2] + eigen3x3_to_float3x3(cgA.block<3, 3>(6, 6));
    affine_body_cgA_diag[4 * body_idx + 3] = affine_body_cgA_diag[4 * body_idx + 3] + eigen3x3_to_float3x3(cgA.block<3, 3>(9, 9));
    host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 0] = host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 0] + eigen3x3_to_float3x3(cgA.block<3, 3>(0, 3));
    host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 1] = host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 1] + eigen3x3_to_float3x3(cgA.block<3, 3>(0, 6));
    host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 2] = host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 2] + eigen3x3_to_float3x3(cgA.block<3, 3>(0, 9));
    host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 3] = host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 3] + eigen3x3_to_float3x3(cgA.block<3, 3>(3, 6));
    host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 4] = host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 4] + eigen3x3_to_float3x3(cgA.block<3, 3>(3, 9));
    host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 5] = host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 5] + eigen3x3_to_float3x3(cgA.block<3, 3>(6, 9));


    // auto dq = cgA.inverse() * cgB;
    // CpuParallel::single_thread_for(0, host_sim_data->sa_affine_bodies.size(), [&](const uint body_idx)
    // {
    //     host_sim_data->sa_cgX[4 * body_idx + 0] = eigen3_to_float3(dq.block<3, 1>(0, 0));
    //     host_sim_data->sa_cgX[4 * body_idx + 1] = eigen3_to_float3(dq.block<3, 1>(3, 0));
    //     host_sim_data->sa_cgX[4 * body_idx + 2] = eigen3_to_float3(dq.block<3, 1>(6, 0));
    //     host_sim_data->sa_cgX[4 * body_idx + 3] = eigen3_to_float3(dq.block<3, 1>(9, 0));
    // });

}
void NewtonSolver::host_evaluete_spring()
{  
    // auto& culster = host_xpbd_data->sa_clusterd_springs;
    // auto& sa_edges = host_mesh_data->sa_edges;
    // auto& sa_rest_length = host_mesh_data->sa_stretch_spring_rest_state_length;
    
    auto& cluster = host_sim_data->sa_prefix_merged_springs;

    for (uint cluster_idx = 0; cluster_idx < host_sim_data->num_clusters_springs; cluster_idx++) 
    {
        const uint curr_prefix = cluster[cluster_idx];
        const uint next_prefix = cluster[cluster_idx + 1];
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
                cluster = host_sim_data->sa_clusterd_springs.data() + host_sim_data->num_clusters_springs + 1,
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
            // float3 dir = normalize_vec(diff);
            float3x3 nnT = outer_product(dir, dir);
            float x_inv = 1.f / l;
            float x_squared_inv = x_inv * x_inv;

            force[0] = stiffness_stretch_spring * dir * C;
            force[1] = -force[0];
            He = stiffness_stretch_spring * nnT + stiffness_stretch_spring * max_scalar(1.0f - L * x_inv, 0.0f) * (luisa::make_float3x3(1.0f) - nnT);
            
            if constexpr (print_detail) luisa::log_info("eid {} (orig = {}), edge ({}, {}), L {}, l {}, C {}, force0 {}, He {}", eid, cluster[curr_prefix + index], edge[0], edge[1], L, l, C, force[0], He);

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
}
void NewtonSolver::host_evaluete_bending()
{
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
        sim_data->sa_vert_affine_bodies_id,
        sim_data->sa_vert_affine_bodies_id,
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
        sim_data->sa_vert_affine_bodies_id,
        sim_data->sa_vert_affine_bodies_id,
        d_hat, thickness, kappa);
}
void NewtonSolver::device_update_contact_list(luisa::compute::Stream& stream)
{
    mp_narrowphase_detector->reset_broadphase_count(stream);
    mp_narrowphase_detector->reset_narrowphase_count(stream);

    device_broadphase_dcd(stream);

    mp_narrowphase_detector->download_broadphase_collision_count(stream);
    
    if (get_scene_params().use_self_collision) device_narrowphase_dcd(stream);

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
        // Stretch 6x6 offidag part
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
                    
                    // if (host_mesh_data->sa_is_fixed[edge[0]] || host_mesh_data->sa_is_fixed[edge[1]]) return;

                    float3x3 offdiag_hessian1 = off_diag_hessian_ptr[eid];
                    float3x3 offdiag_hessian2 = luisa::transpose(offdiag_hessian1);
                    float3 output_vec0 = offdiag_hessian1 * input_ptr[edge[1]];
                    float3 output_vec1 = offdiag_hessian2 * input_ptr[edge[0]];
                    output_ptr[edge[0]] += output_vec0;
                    output_ptr[edge[1]] += output_vec1;
                });
            }
        }
        // Bending 12x12 offdiag part
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

                    // if (
                    //     host_mesh_data->sa_is_fixed[edge[0]] || 
                    //     host_mesh_data->sa_is_fixed[edge[1]] ||
                    //     host_mesh_data->sa_is_fixed[edge[2]] ||
                    //     host_mesh_data->sa_is_fixed[edge[3]] 
                    // ) return;

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
        // Affine body 12x12 block
        {
            const uint affine_body_dof_prefix = host_sim_data->num_verts_soft;

            auto& off_diag_hessian_ptr = host_sim_data->sa_cgA_offdiag_affine_body;
            CpuParallel::parallel_for(0, host_sim_data->sa_affine_bodies.size(), [&](const uint body_idx)
            {
                float3 input_vec[4] = {
                    input_ptr[affine_body_dof_prefix + 4 * body_idx + 0],
                    input_ptr[affine_body_dof_prefix + 4 * body_idx + 1],
                    input_ptr[affine_body_dof_prefix + 4 * body_idx + 2],
                    input_ptr[affine_body_dof_prefix + 4 * body_idx + 3],
                }; 
                float3 output_vec[4] = {
                    luisa::make_float3(0.0f),
                    luisa::make_float3(0.0f),
                    luisa::make_float3(0.0f),
                    luisa::make_float3(0.0f),
                };

                uint offset = 0;
                for (uint j = 0; j < 4; j++)
                {
                    for (uint jj = j + 1; jj < 4; jj++)
                    {
                        float3x3 hessian = off_diag_hessian_ptr[6 * body_idx + offset]; offset += 1;
                        output_vec[j] += (hessian) * input_vec[jj];
                        output_vec[jj] += transpose(hessian) * input_vec[j];
                    }
                }
                output_ptr[affine_body_dof_prefix + 4 * body_idx + 0] += output_vec[0];
                output_ptr[affine_body_dof_prefix + 4 * body_idx + 1] += output_vec[1];
                output_ptr[affine_body_dof_prefix + 4 * body_idx + 2] += output_vec[2];
                output_ptr[affine_body_dof_prefix + 4 * body_idx + 3] += output_vec[3];
            });
        }
    }

    // Off-diag: Collision hessian
    mp_narrowphase_detector->host_spmv_repulsion(stream, input_ptr, output_ptr);

}
void NewtonSolver::host_solve_eigen(luisa::compute::Stream& stream, std::function<double(const std::vector<float3>&)> func_compute_energy)
{
    
    EigenFloat12 eigen_b; eigen_b.setZero();
    EigenFloat12x12 eigen_A; eigen_A.setZero();
    for (uint vid = 0; vid < 4; vid++)
    {
        eigen_b.block<3, 1>( vid * 3, 0) = float3_to_eigen3(host_sim_data->sa_cgB[vid]);
        eigen_A.block<3, 3>( vid * 3, vid * 3) = float3x3_to_eigen3x3(host_sim_data->sa_cgA_diag[vid]);
    }
    for (uint eid = 0; eid < 5; eid++)
    {
        float3x3 offdiag1 = host_sim_data->sa_cgA_offdiag_stretch_spring[eid];
        float3x3 offdiag2 = luisa::transpose(offdiag1);
        uint2 edge = host_sim_data->sa_merged_stretch_springs[eid];
        eigen_A.block<3, 3>( edge[0] * 3, edge[1] * 3) += float3x3_to_eigen3x3(offdiag1);
        eigen_A.block<3, 3>( edge[1] * 3, edge[0] * 3) += float3x3_to_eigen3x3(offdiag2);
    }

    auto eigen_dx = eigen_A.inverse() * eigen_b;
    // auto eigen_dx = eigen_pcg(eigen_A, eigen_b);

    // std::cout << "Eigen A = \n" << eigen_A << std::endl;
    // std::cout << "Eigen b = \n" << eigen_b.transpose() << std::endl;
    // std::cout << "Eigen result = " << eigen_dx.transpose() << std::endl;
    for (uint vid = 0; vid < 4; vid++)
    {
        host_sim_data->sa_cgX[vid] = eigen3_to_float3(eigen_dx.block<3, 1>(vid * 3, 0));
    }
    constexpr bool print_energy = false; double curr_energy = 0.0;
    if constexpr (print_energy)
    {
        host_apply_dx(1.0f);
        curr_energy = func_compute_energy(host_sim_data->sa_x);
    }
    const float infinity_norm = fast_infinity_norm(host_sim_data->sa_cgX);
    if (luisa::isnan(infinity_norm) || luisa::isinf(infinity_norm))
    {
        luisa::log_error("cgX exist NAN/INF value : {}", infinity_norm);
    }
    luisa::log_info("  In non-linear iter {:2}, EigenSolve error = {:7.6f}, max_element(p) = {:6.5f}{}", 
        get_scene_params().current_nonlinear_iter,
        (eigen_b - eigen_A * eigen_dx).norm(), infinity_norm, print_energy ? luisa::format(", energy = {:6.5f}", curr_energy) : ""
    );

}
void NewtonSolver::host_solve_amgcl(luisa::compute::Stream& stream, std::function<double(const std::vector<float3>&)> func_compute_energy)
{
    const uint num_verts = host_sim_data->sa_cgX.size();
    const uint num_dof = 3 * num_verts;

    auto assmble_amgcl_system = [&](
        std::vector<uint>& ptr,
        std::vector<uint>& col,
        std::vector<EigenFloat3x3>& val,
        std::vector<EigenFloat3>& rhs
    )
    {
        const uint offset_vf = host_collision_data->get_vf_count_offset();
        const uint offset_ee = host_collision_data->get_ee_count_offset();
        const auto& host_count = host_collision_data->narrow_phase_collision_count;
        const uint num_vf = host_count[offset_vf]; const uint num_ee = host_count[offset_ee];

        rhs.resize(num_verts);
        ptr.resize(num_verts + 1); ptr[0] = 0;

        // Init with material constraints adjacency
        std::vector<std::vector<uint>> adjacency(host_sim_data->vert_adj_material_force_verts);
        {
            // Add collision adjacency
            for (uint pair_idx = 0; pair_idx < num_vf + num_ee; pair_idx++)
            {
                uint4 indices;
                if (pair_idx < host_count[offset_vf])
                {
                    indices = host_collision_data->narrow_phase_list_vf[pair_idx].indices;
                }
                else
                {
                    indices = host_collision_data->narrow_phase_list_ee[pair_idx - num_vf].indices;
                }
                for (uint j = 0; j < 3; j++)
                {
                    for (uint jj = 0; jj < 3; jj++)
                    {
                        if (indices[j] != indices[jj])
                        {
                            adjacency[indices[j]].push_back(indices[jj]);
                            adjacency[indices[jj]].push_back(indices[j]);
                        }
                    }
                }
            }
    
            // Remove duplicate
            CpuParallel::parallel_for(0, num_verts, [&](const uint vid)
            {
                auto& adj_list = adjacency[vid];
                std::sort(adj_list.begin(), adj_list.end());
                adj_list.erase(unique(adj_list.begin(), adj_list.end()), adj_list.end());
                adj_list.insert(adj_list.begin(), vid); // Add diag entry
                // std::cout << "Vert " << vid << " has " << adj_list.size() << " adjacency: "; for (auto v : adj_list) std::cout << v << ", "; std::cout << std::endl;
            });
    
            CpuParallel::parallel_for_and_scan<uint>(0, num_verts, [&](const uint vid)
            {
                return adjacency[vid].size();
            }, [&](const uint vid, const uint block_prefix, const uint parallel_result)
            {
                ptr[vid + 1] = block_prefix;
            }, 0);
    
            uint hessian_block_count = ptr.back(); // luisa::log_info("Total hessian non-zero block count: {}", hessian_block_count);
    
            col.resize(hessian_block_count);
            val.resize(hessian_block_count);
        }
        
        // CpuParallel::parallel_set(val, total_off_diag_count, EigenFloat3x3::Zero());
        CpuParallel::parallel_for(0, num_verts, [&](const uint vid)
        {
            const uint prefix = ptr[vid];
            const auto& adj_list = adjacency[vid];
            for (uint j = 0; j < adj_list.size(); j++)
            {
                col[prefix + j] = adj_list[j];
                val[prefix + j] = EigenFloat3x3::Zero();
            }
        });

        {
            // Diag part
            CpuParallel::parallel_for(0, num_verts, [&](const uint vid)
            {
                const uint prefix = ptr[vid];
                const auto& adj_list = adjacency[vid];
                const uint offset = std::distance(adj_list.begin(), std::find(adj_list.begin(), adj_list.end(), vid));
                if (offset != 0)
                {
                    luisa::log_error("Vert {} diag not found in adjacency list", vid);
                }
                const auto diag_hessian = host_sim_data->sa_cgA_diag[vid];
                val[prefix + offset] = float3x3_to_eigen3x3(diag_hessian);
                rhs[vid] = float3_to_eigen3(host_sim_data->sa_cgB[vid]);
                // luisa::log_info("Vert {} diag offset = {}", vid, offset);
            });
    
            // Off-diag part
            CpuParallel::single_thread_for(0, host_sim_data->sa_merged_stretch_springs.size(), 
                [
                    sa_edges = host_sim_data->sa_merged_stretch_springs.data(),
                    off_diag_hessian_ptr = host_sim_data->sa_cgA_offdiag_stretch_spring.data(),
                    ptr = ptr.data(),
                    adjacency = adjacency.data(),
                    val = val.data()
                ](const uint eid)
            {
                const uint2 edge = sa_edges[eid];
                float3x3 negHe = off_diag_hessian_ptr[eid];
    
                for (uint j = 0; j < 2; j++)
                {
                    const uint left = edge[j];
                    const uint prefix = ptr[left];
                    const auto& adj_list = adjacency[left];
                    for (uint jj = 0; jj < 2; jj++)
                    {
                        if (j != jj)
                        {
                            const uint right = edge[jj];
                            const uint offset = std::distance(adj_list.begin(), std::find(adj_list.begin(), adj_list.end(), right));
                            // luisa::log_info("Edge {}: ({}, {}) -> ({}, {}), offset = {}", eid, edge[0], edge[1], left, right, offset);
                            val[prefix + offset] += float3x3_to_eigen3x3(negHe);
                        }
                    }
                }
            });
    
            CpuParallel::single_thread_for(0, host_sim_data->sa_merged_bending_edges.size(), 
                [
                    sa_edges = host_sim_data->sa_merged_bending_edges.data(),
                    sa_bending_edges_Q = host_sim_data->sa_merged_bending_edges_Q.data(),
                    off_diag_hessian_ptr = host_sim_data->sa_cgA_offdiag_bending.data(),
                    stiffness_bending = get_scene_params().get_stiffness_quadratic_bending(),
                    ptr = ptr.data(),
                    adjacency = adjacency.data(),
                    val = val.data()
                ](const uint eid)
            {
                const uint4 edge = sa_edges[eid];
                const float4x4 m_Q = sa_bending_edges_Q[eid];
    
                for (uint j = 0; j < 4; j++)
                {
                    const uint left = edge[j];
                    const uint prefix = ptr[left];
                    const auto& adj_list = adjacency[left];
                    for (uint jj = 0; jj < 4; jj++)
                    {
                        if (j != jj)
                        {
                            const uint right = edge[jj];
                            const uint offset = std::distance(adj_list.begin(), std::find(adj_list.begin(), adj_list.end(), right));
                            val[prefix + offset] += stiffness_bending * m_Q[j][jj] * EigenFloat3x3::Identity();
                            // luisa::log_info("Bending Edge {}: ({}, {}, {}, {}) -> ({}, {}), offset = {}", eid, edge[0], edge[1], edge[2], edge[3], left, right, offset);
                        }
                    }
                }
            });
    
            CpuParallel::single_thread_for(0, num_vf + num_ee, 
                [
                    narrow_phase_list_vf = host_collision_data->narrow_phase_list_vf.data(),
                    narrow_phase_list_ee = host_collision_data->narrow_phase_list_ee.data(),
                    ptr = ptr.data(),
                    adjacency = adjacency.data(),
                    val = val.data()
                , num_vf, num_ee](const uint pair_idx)
            {
                uint4 indices;
                float4 weight;
                float3 normal;
                float k2;
                if (pair_idx < num_vf)
                {
                    auto pair = narrow_phase_list_vf[pair_idx];
                    indices = pair.indices;
                    weight = CollisionPair::get_vf_weight(pair);
                    normal = CollisionPair::get_direction(pair);
                    k2 = CollisionPair::get_vf_stiff(pair)[1];
                }
                else
                {
                    auto pair = narrow_phase_list_ee[pair_idx - num_vf];
                    indices = pair.indices;
                    weight = CollisionPair::get_ee_weight(pair);
                    normal = CollisionPair::get_direction(pair);
                    k2 = CollisionPair::get_ee_stiff(pair)[1];
                }
    
                EigenFloat3 normal_eigen = float3_to_eigen3(normal);
                const auto xxT = k2 * normal_eigen * normal_eigen.transpose();
                for (uint j = 0; j < 4; j++)
                {
                    const uint left = indices[j];
                    const uint prefix = ptr[left];
                    const auto& adj_list = adjacency[left];
                    for (uint jj = 0; jj < 4; jj++)
                    {
                        if (j != jj)
                        {
                            const uint right = indices[jj];
                            const uint offset = std::distance(adj_list.begin(), std::find(adj_list.begin(), adj_list.end(), right));
                            auto hessian = weight[j] * weight[jj] * xxT;
                            val[prefix + offset] += hessian;
                            // luisa::log_info("Collision Pair {}: ({}, {}, {}, {}) -> ({}, {}), offset = {}", pair_idx, indices[0], indices[1], indices[2], indices[3], left, right, offset);
                        }
                    }
                }
            });
        }


    };

    auto convert_block_into_amgcl_system = [&](
        std::vector<uint>& ptr1,
        std::vector<uint>& col1,
        std::vector<EigenFloat3x3>& val1,
        std::vector<EigenFloat3>& rhs1,
        std::vector<uint>& ptr2,
        std::vector<uint>& col2,
        std::vector<float>& val2,
        std::vector<float>& rhs2
    )
    {
        const uint hessian_block_count = val1.size();

        ptr2.resize(num_dof + 1); ptr2.back() = hessian_block_count * 9;
        col2.resize(hessian_block_count * 9);
        val2.resize(hessian_block_count * 9, 0.0f);
        rhs2.resize(num_dof, 0.0f);

        CpuParallel::parallel_for(0, num_verts, [&](const uint vid)
        {
            const auto& b = rhs1[vid];
            const uint prefix = ptr1[vid];
            const uint adj_count = ptr1[vid + 1] - ptr1[vid];
            rhs2[3 * vid + 0] = b[0];
            rhs2[3 * vid + 1] = b[1];
            rhs2[3 * vid + 2] = b[2];
            ptr2[3 * vid + 0] = 9 * prefix + 0 * adj_count * 3;
            ptr2[3 * vid + 1] = 9 * prefix + 1 * adj_count * 3;
            ptr2[3 * vid + 2] = 9 * prefix + 2 * adj_count * 3;
            // luisa::log_info("Vert {} prefix = {}, adj_count = {}, row Prefix = {} / {} / {}", vid, prefix, adj_count, ptr2[3 * vid + 0], ptr2[3 * vid + 1], ptr2[3 * vid + 2]);

            for (uint j = 0; j < adj_count; j++)
            {
                for (uint ii = 0; ii < 3; ii++)
                {
                    const uint adj_vid = col1[prefix + j];
                    const auto& hessian = val1[prefix + j];
                    for (uint jj = 0; jj < 3; jj++)
                    {
                        const uint index = (9 * prefix + ii * adj_count * 3 + j * 3 + jj);
                        col2[index] = 3 * adj_vid + jj;
                        val2[index] = hessian(ii, jj);
                        // luisa::log_info("Hessian ({}, {}) -> ({}, {}), index = {}, val = {}", vid, ii, adj_vid, jj, index, val2[index]);
                    }
                }
            }
        });
    };

    auto solve_amgcl_system = [&](
        const std::vector<uint>& ptr,
        const std::vector<uint>& col,
        const std::vector<float>& val,
        const std::vector<float>& rhs,
        std::vector<float>& lhs
    )
    {
    #if defined (USE_AMGCL_FOR_SIM)  && USE_AMGCL_FOR_SIM
        lhs.resize(num_dof, 0.0f);

        using Backend = amgcl::backend::builtin<float>;

        using Solver = amgcl::make_solver<
            amgcl::amg<Backend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>, // Use AMG as preconditioner:
            amgcl::solver::cg<Backend> // And BiCGStab as iterative solver:
            >;

        boost::property_tree::ptree prm;
        prm.put("solver.tol", 1e-3);
        Solver solver(std::tie(num_dof, ptr, col, val), prm );
        return solver(rhs, lhs);
        // std::cout << "CG finished in " << iters << " iterations, resid = " << error << std::endl;
    #else
        luisa::log_error("AMGCL is not enabled in this build");
        return std::make_tuple(0u, 0.0f);
    #endif
    };
    std::vector<EigenFloat3> rhs;
    std::vector<uint> ptr;
    std::vector<uint> col;
    std::vector<EigenFloat3x3> val;
    assmble_amgcl_system(ptr, col, val, rhs);

    // for (auto prefix : ptr) { luisa::log_info("Prefix = {}", prefix); }
    // for (auto adj_vid : col) { luisa::log_info("adj_vid = {}", adj_vid); }
    // for (auto hess : val) { luisa::log_info("hess = {}", hess); }

    std::vector<uint> ptr2;
    std::vector<uint> col2;
    std::vector<float> val2;
    std::vector<float> rhs2;
    convert_block_into_amgcl_system(ptr, col, val, rhs, ptr2, col2, val2, rhs2);
    
    std::vector<float> lhs2; uint iter; float error;
    std::tie(iter, error) = solve_amgcl_system(ptr2, col2, val2, rhs2, lhs2);

    CpuParallel::parallel_for(0, num_verts, [&](const uint vid)
    {
        host_sim_data->sa_cgX[vid] = luisa::make_float3(
            lhs2[3 * vid + 0],
            lhs2[3 * vid + 1],
            lhs2[3 * vid + 2]
        );
    });

    constexpr bool print_energy = false; double curr_energy = 0.0;
    if constexpr (print_energy)
    {
        host_apply_dx(1.0f);
        curr_energy = func_compute_energy(host_sim_data->sa_x);
    }
    const float infinity_norm = fast_infinity_norm(host_sim_data->sa_cgX);
    if (luisa::isnan(infinity_norm) || luisa::isinf(infinity_norm))
    {
        luisa::log_error("cgX exist NAN/INF value : {}", infinity_norm);
    }
    luisa::log_info("  In non-linear iter {:2}, PCG : iter-count = {:3}, rTr error = {:7.6f}, max_element(p) = {:6.5f}{}", 
        get_scene_params().current_nonlinear_iter,
        iter, error, infinity_norm, print_energy ? luisa::format(", energy = {:6.5f}", curr_energy) : ""
    );
          
}
void NewtonSolver::host_line_search(luisa::compute::Stream& stream)
{
    
}


void NewtonSolver::host_apply_dx(const float alpha)
{
    if (alpha < 0.0f || alpha > 1.0f) { luisa::log_error("Alpha is not safe : {}", alpha); }

    // Update affine-body q
    float3* affine_body_cgX = &host_sim_data->sa_cgX[host_sim_data->num_verts_soft];
    CpuParallel::parallel_for(0, host_sim_data->sa_affine_bodies.size() * 4, [&](const uint block_idx)
    {
        host_sim_data->sa_affine_bodies_q[block_idx] = host_sim_data->sa_affine_bodies_q_iter_start[block_idx] + alpha * affine_body_cgX[block_idx];
        // luisa::log_info("Rigid body {}: q_{} = {}", block_idx / 4, block_idx % 4, host_sim_data->sa_affine_bodies_q[block_idx]);
    });

    // Update sa_x
    CpuParallel::parallel_for(0, host_mesh_data->num_verts, [&](const uint vid)
    {
        const bool is_rigid_body = host_mesh_data->sa_vert_mesh_type[vid] == Initializer::ShellTypeRigid;
        if (is_rigid_body)
        {
            const uint body_idx = host_sim_data->sa_vert_affine_bodies_id[vid];
            float3 p; float3x3 A;
            AffineBodyDynamics::extract_Ap_from_q(&host_sim_data->sa_affine_bodies_q[4 * body_idx], A, p);
            const float3 rest_x = host_mesh_data->sa_scaled_model_x[vid];
            const float3 affine_x = A * rest_x + p; // Affine position
            host_sim_data->sa_x[vid] = affine_x;
            // luisa::log_info("Rigid Body {}'s Vert {} apply transform, from {} to {}", body_idx, vid, rest_x, affine_x);
        }
        else
        {
            host_sim_data->sa_x[vid] = host_sim_data->sa_x_iter_start[vid] + alpha * host_sim_data->sa_cgX[vid];
        }
    });
}

void NewtonSolver::physics_step_CPU(luisa::compute::Device& device, luisa::compute::Stream& stream)
{
    // Input
    {
        lcs::SolverInterface::physics_step_prev_operation(); 
        CpuParallel::parallel_for(0, host_sim_data->sa_x.size(), [&](const uint vid)
        {
            host_sim_data->sa_x_step_start[vid] = host_mesh_data->sa_x_frame_outer[vid];
            host_sim_data->sa_v_step_start[vid] = host_mesh_data->sa_v_frame_outer[vid];
            host_sim_data->sa_x[vid] = host_mesh_data->sa_x_frame_outer[vid];
            host_sim_data->sa_v[vid] = host_mesh_data->sa_v_frame_outer[vid];
        });
        CpuParallel::parallel_for(0, host_sim_data->sa_affine_bodies_q.size(), [&](const uint vid)
        {
            host_sim_data->sa_affine_bodies_q_step_start[vid] = host_sim_data->sa_affine_bodies_q_outer[vid];
            host_sim_data->sa_affine_bodies_q[vid] = host_sim_data->sa_affine_bodies_q_outer[vid];
            host_sim_data->sa_affine_bodies_q_v[vid] = host_sim_data->sa_affine_bodies_q_v_outer[vid];
        });
    }
    
    constexpr bool use_eigen = ConjugateGradientSolver::use_eigen;
    constexpr bool use_upper_triangle = ConjugateGradientSolver::use_upper_triangle;

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
            << sim_data->sa_x.copy_from(host_sim_data->sa_x.data())
            ;
            
        device_update_contact_list(stream);
        mp_narrowphase_detector->download_narrowphase_list(stream);
    };
    auto evaluate_contact = [&]()
    {
        stream 
            << sim_data->sa_cgB.copy_from(host_sim_data->sa_cgB.data())
            << sim_data->sa_cgA_diag.copy_from(host_sim_data->sa_cgA_diag.data());

        mp_narrowphase_detector->compute_repulsion_gradiant_hessian_and_assemble(stream, sim_data->sa_x, sim_data->sa_x, d_hat, thickness, sim_data->sa_cgB, sim_data->sa_cgA_diag);

        stream 
            << sim_data->sa_cgB.copy_to(host_sim_data->sa_cgB.data())
            << sim_data->sa_cgA_diag.copy_to(host_sim_data->sa_cgA_diag.data())
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
    
    auto compute_energy_interface = [&](const std::vector<float3> &curr_x)
    {
        stream << sim_data->sa_x_tilde.copy_from(host_sim_data->sa_x_tilde.data());
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
        if constexpr (false) 
        {
            host_solve_eigen(stream, compute_energy_interface);
            // host_solve_amgcl(stream, compute_energy_interface);
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
    if (false)
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
        luisa::log_info("=== In frame {} ===", get_scene_params().current_frame); 

        host_predict_position();
        
        // double barrier_nergy = compute_barrier_energy_from_broadphase_list();
        double prev_state_energy = Float_max;

        for (uint iter = 0; iter < get_scene_params().nonlinear_iter_count; iter++)
        {   get_scene_params().current_nonlinear_iter = iter;
                        
            host_reset_cgB_cgX_diagA();

            host_reset_off_diag();

            // if constexpr (false)
            {
                host_evaluate_inertia();
                
                host_evaluate_ground_collision();
                
                host_evaluate_orthogonality();
                
                host_evaluete_spring();

                host_evaluete_bending();
                
                update_contact_set();
                
                evaluate_contact();

                // host_evaluate_dirichlet();
                
                // for (uint vid = 0; vid < host_mesh_data->num_verts; vid++) luisa::log_info("Post Vert {}'s force = {}", vid, host_sim_data->sa_cgB[vid]);
                // if (iter == 0) // Always refresh for collision count is variant 
                if (use_energy_linesearch) { prev_state_energy = compute_energy_interface(host_sim_data->sa_x); }
            }

            // host_test_affine_body(stream);

            linear_solver_interface(); // Solve Ax=b

            float alpha = 1.0f; float ccd_toi = 1.0f;
            host_apply_dx(alpha);
            
            if (use_ccd_linesearch)
            {
                ccd_toi = ccd_get_toi();
                alpha = ccd_toi;
                host_apply_dx(alpha);   
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
            } // That means: If the step is too small, then we dont need energy line-search (energy may not be descent in small step)

            if (use_energy_linesearch)
            { 
                // Energy after CCD or just solving Axb
                auto curr_energy = compute_energy_interface(host_sim_data->sa_x); 
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

                    curr_energy = compute_energy_interface(host_sim_data->sa_x);
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

            // CpuParallel::parallel_copy(host_sim_data->sa_x.data(), host_sim_data->sa_x_iter_start.data(), host_sim_data->num_verts_soft); // x_prev = x
            CpuParallel::parallel_copy(host_sim_data->sa_x, host_sim_data->sa_x_iter_start); // x_prev = x
            CpuParallel::parallel_copy(host_sim_data->sa_affine_bodies_q, host_sim_data->sa_affine_bodies_q_iter_start); // q_prev = q
        }
        host_update_velocity();
    }

    // Output
    {
        CpuParallel::parallel_for(0, host_sim_data->sa_x.size(), [&](const uint vid)
        {
            host_mesh_data->sa_x_frame_outer[vid] = host_sim_data->sa_x[vid];
            host_mesh_data->sa_v_frame_outer[vid] = host_sim_data->sa_v[vid];
        });
        CpuParallel::parallel_copy(host_sim_data->sa_affine_bodies_q, host_sim_data->sa_affine_bodies_q_outer);
        CpuParallel::parallel_copy(host_sim_data->sa_affine_bodies_q_v, host_sim_data->sa_affine_bodies_q_v_outer);
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

    auto device_apply_dx = [&](const float alpha)
    {
        stream << fn_apply_dx(alpha).dispatch(sim_data->sa_cgX.size());
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

    auto pcg_spmv = [&](const luisa::compute::Buffer<float3>& input_ptr, luisa::compute::Buffer<float3>& output_ptr) -> void
    {   
        device_SpMV(stream, input_ptr, output_ptr);
    };
    auto compute_energy_interface = [&](const luisa::compute::Buffer<float3>& curr_x)
    {
        // stream << sim_data->sa_x_tilde.copy_to(host_sim_data->sa_x_tilde.data());
        // auto material_energy = host_compute_elastic_energy(host_sim_data->sa_x);

        // stream << sim_data->sa_x.copy_to(host_sim_data->sa_x.data());
        // stream << luisa::compute::synchronize();

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
            << fn_predict_position(substep_dt, get_scene_params().gravity).dispatch(num_verts)
            // << sim_data->sa_x_step_start.copy_to(host_x_step_start.data())
            << sim_data->sa_x_tilde.copy_to(host_sim_data->sa_x_tilde.data()) // For calculate inertia energy
            << luisa::compute::synchronize();
        
        double prev_state_energy = Float_max;

        luisa::log_info("=== In frame {} ===", get_scene_params().current_frame); 

        for (uint iter = 0; iter < get_scene_params().nonlinear_iter_count; iter++)
        {   get_scene_params().current_nonlinear_iter = iter;
            
            stream 
                << sim_data->sa_x_iter_start.copy_to(host_sim_data->sa_x_iter_start.data())
                << fn_reset_vector(sim_data->sa_cgX).dispatch(sim_data->sa_cgX.size())
                << fn_reset_vector(sim_data->sa_cgB).dispatch(sim_data->sa_cgB.size())
                << fn_reset_float3x3(sim_data->sa_cgA_diag).dispatch(sim_data->sa_cgA_diag.size())
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

            // Non-linear iteration break condition
            {
                float max_move = 1e-2;
                float curr_max_step = fast_infinity_norm(host_sim_data->sa_cgX); 
                if (curr_max_step < max_move * substep_dt) 
                {
                    luisa::log_info("  In non-linear iter {:2}: Iteration break for small searching direction {} < {}", iter, curr_max_step, max_move * substep_dt);
                    break;
                }
            } // That means: If the step is too small, then we dont need energy line-search (energy may not be descent in small step)

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
            
            stream
                << sim_data->sa_x.copy_from(host_sim_data->sa_x.data())
                << sim_data->sa_x_iter_start.copy_from(host_sim_data->sa_x_iter_start.data());

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