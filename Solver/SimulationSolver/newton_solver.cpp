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
#include "luisa/core/clock.h"
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
template <typename T>
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

static inline float fast_infinity_norm(const std::vector<float3>& ptr)  // Min value in array
{
    return CpuParallel::parallel_for_and_reduce(
        0,
        ptr.size(),
        [&](const uint vid) { return luisa::length(ptr[vid]); },
        [](const float left, const float right) { return max_scalar(left, right); },
        -1e9f);
};

void NewtonSolver::compile(AsyncCompiler& compiler)
{
    const bool use_debug_info = false;
    using namespace luisa::compute;

    luisa::compute::ShaderOption default_option = {.enable_debug_info = false};

    compiler.compile<1>(fn_reset_vector,
                        [](Var<BufferView<float3>> buffer)
                        {
                            const UInt vid = dispatch_id().x;
                            // buffer->write(vid, target);
                            buffer->write(vid, make_float3(0.0f));
                        });
    compiler.compile<1>(fn_reset_float3x3,
                        [](Var<BufferView<float3x3>> buffer)
                        {
                            const UInt vid = dispatch_id().x;
                            buffer->write(vid, make_float3x3(0.0f));
                        });

    compiler.compile<1>(
        fn_predict_position,
        [sa_x            = sim_data->sa_x.view(),
         sa_x_step_start = sim_data->sa_x_step_start.view(),
         sa_x_iter_start = sim_data->sa_x_iter_start.view(),
         sa_x_tilde      = sim_data->sa_x_tilde.view(),
         sa_v            = sim_data->sa_v.view(),
         sa_cgX          = sim_data->sa_cgX.view(),
         sa_is_fixed     = mesh_data->sa_is_fixed.view()](const Float substep_dt, const Float3 gravity)
        {
            const UInt vid = dispatch_id().x;
            // const Float3 gravity = make_float3(0.0f, -9.8f, 0.0f);
            Float3 x_prev             = sa_x_step_start->read(vid);
            Float3 v_prev             = sa_v->read(vid);
            Float3 outer_acceleration = gravity;
            Float3 v_pred             = v_prev + substep_dt * outer_acceleration;

            const Bool is_fixed = sa_is_fixed->read(vid);

            $if(is_fixed)
            {
                v_pred = v_prev;
                // v_pred = make_float3(0.0f);
            };

            sa_x_iter_start->write(vid, x_prev);
            Float3 x_pred = x_prev + substep_dt * v_pred;
            sa_x_tilde->write(vid, x_pred);
            sa_x->write(vid, x_prev);
            // sa_cgX->write(vid, make_float3(0.0f));
        },
        default_option);

    compiler.compile<1>(
        fn_update_velocity,
        [sa_x            = sim_data->sa_x.view(),
         sa_v            = sim_data->sa_v.view(),
         sa_x_step_start = sim_data->sa_x_step_start.view(),
         sa_v_step_start = sim_data->sa_v_step_start.view()](const Float substep_dt, const Bool fix_scene, const Float damping)
        {
            const UInt vid          = dispatch_id().x;
            Float3     x_step_begin = sa_x_step_start->read(vid);
            Float3     x_step_end   = sa_x->read(vid);

            Float3 dx  = x_step_end - x_step_begin;
            Float3 vel = dx / substep_dt;

            $if(fix_scene)
            {
                dx  = make_float3(0.0f);
                vel = make_float3(0.0f);
                sa_x->write(vid, x_step_begin);
                return;
            };

            vel *= exp(-damping * substep_dt);

            sa_v->write(vid, vel);
            sa_v_step_start->write(vid, vel);
            sa_x_step_start->write(vid, x_step_end);
        },
        default_option);

    compiler.compile<1>(
        fn_evaluate_inertia,
        [sa_x        = sim_data->sa_x.view(),
         sa_x_tilde  = sim_data->sa_x_tilde.view(),
         sa_cgB      = sim_data->sa_cgB.view(),
         sa_cgA_diag = sim_data->sa_cgA_diag.view(),
         sa_is_fixed = mesh_data->sa_is_fixed.view(),
         sa_vert_mass = mesh_data->sa_vert_mass.view()](const Float substep_dt, const Float stiffness_dirichlet)
        {
            const UInt  vid     = dispatch_id().x;
            const Float h       = substep_dt;
            const Float h_2_inv = 1.0f / (h * h);

            Float3 x_k     = sa_x->read(vid);
            Float3 x_tilde = sa_x_tilde->read(vid);
            Float  mass    = sa_vert_mass->read(vid);

            Float3   gradient = -mass * h_2_inv * (x_k - x_tilde);
            Float3x3 hessian  = make_float3x3(1.0f) * mass * h_2_inv;

            $if(sa_is_fixed->read(vid) != 0)
            {
                gradient = gradient + stiffness_dirichlet * gradient;
                hessian  = hessian + stiffness_dirichlet * hessian;
                // hessian = make_float3x3(1.0f) * 1e9f;
                // gradient = make_float3(0.0f);
            };

            sa_cgB->write(vid, gradient);
            sa_cgA_diag->write(vid, hessian);
        },
        default_option);

    compiler.compile<1>(
        fn_evaluate_dirichlet,
        [sa_x        = sim_data->sa_x.view(),
         sa_x_tilde  = sim_data->sa_x_tilde.view(),
         sa_cgB      = sim_data->sa_cgB.view(),
         sa_cgA_diag = sim_data->sa_cgA_diag.view(),
         sa_is_fixed = mesh_data->sa_is_fixed.view(),
         sa_vert_mass = mesh_data->sa_vert_mass.view()](const Float substep_dt, const Float stiffness_dirichlet)
        {
            const UInt vid = dispatch_id().x;
            return;

            Bool is_fixed = sa_is_fixed->read(vid);
            $if(is_fixed)
            {
                const Float h       = substep_dt;
                const Float h_2_inv = 1.0f / (h * h);

                Float3 x_k     = sa_x->read(vid);
                Float3 x_tilde = sa_x_tilde->read(vid);
                // Float3 gradient = stiffness_dirichlet * (x_k - x_tilde);
                // Float3x3 hessian = stiffness_dirichlet * make_float3x3(1.0f);
                Float    mass     = sa_vert_mass->read(vid);
                Float3   gradient = stiffness_dirichlet * h_2_inv * mass * (x_k - x_tilde);
                Float3x3 hessian  = stiffness_dirichlet * h_2_inv * mass * make_float3x3(1.0f);
                sa_cgB->write(vid, sa_cgB->read(vid) - gradient);
                sa_cgA_diag->write(vid, sa_cgA_diag->read(vid) + hessian);
            };
        },
        default_option);

    compiler.compile<1>(
        fn_evaluate_ground_collision,
        [sa_x              = sim_data->sa_x.view(),
         sa_rest_vert_area = mesh_data->sa_rest_vert_area.view(),
         sa_cgB            = sim_data->sa_cgB.view(),
         sa_cgA_diag       = sim_data->sa_cgA_diag.view(),
         sa_is_fixed       = mesh_data->sa_is_fixed.view(),
         sa_vert_mass      = mesh_data->sa_vert_mass.view()](
            Float floor_y, Bool use_ground_collision, Float stiffness, Float d_hat, Float thickness)
        {
            const UInt vid = dispatch_id().x;
            $if(use_ground_collision)
            {
                $if(!sa_is_fixed->read(vid))
                {
                    Float3 x_k  = sa_x->read(vid);
                    Float  diff = x_k.y - floor_y;

                    Float3   force   = sa_cgB->read(vid);
                    Float3x3 hessian = sa_cgA_diag->read(vid);
                    $if(diff < d_hat + thickness)
                    {
                        Float  C      = d_hat + thickness - diff;
                        float3 normal = luisa::make_float3(0, 1, 0);
                        Float  area   = sa_rest_vert_area->read(vid);
                        Float  stiff  = stiffness * area;
                        force += stiff * C * normal;
                        hessian += stiff * outer_product(normal, normal);
                    };
                    sa_cgB->write(vid, force);
                    sa_cgA_diag->write(vid, hessian);
                };
            };
        },
        default_option);

    if (host_sim_data->sa_stretch_springs.size() > 0)
        compiler.compile<1>(
            fn_evaluate_spring,
            [sa_x = sim_data->sa_x.view(),
             // sa_cgB = sim_data->sa_cgB.view(),
             // sa_cgA_diag = sim_data->sa_cgA_diag.view(),
             output_gradient_ptr = sim_data->sa_stretch_springs_gradients.view(),
             output_hessian_ptr  = sim_data->sa_stretch_springs_hessians.view(),
             sa_edges            = sim_data->sa_stretch_springs.view(),
             sa_rest_length = sim_data->sa_stretch_spring_rest_state_length.view()](const Float stiffness_stretch)
            {
                const UInt eid  = dispatch_id().x;
                UInt2      edge = sa_edges->read(eid);

                Float3   vert_pos[2]  = {sa_x->read(edge.x), sa_x->read(edge.y)};
                Float3   gradients[2] = {make_float3(0.0f), make_float3(0.0f)};
                Float3x3 He           = make_float3x3(0.0f);

                const Float L                = sa_rest_length->read(eid);
                const Float stiffness_spring = stiffness_stretch;

                Float3 diff = vert_pos[0] - vert_pos[1];
                Float  l    = max(length(diff), Epsilon);
                Float  l0   = L;
                Float  C    = l - l0;

                Float3   dir           = diff / l;
                Float3x3 xxT           = outer_product(diff, diff);
                Float    x_inv         = 1.f / l;
                Float    x_squared_inv = x_inv * x_inv;

                gradients[0] = stiffness_spring * dir * C;
                gradients[1] = -gradients[0];
                He           = stiffness_spring * x_squared_inv * xxT
                     + stiffness_spring * max(1.0f - L * x_inv, 0.0f) * (make_float3x3(1.0f) - x_squared_inv * xxT);

                // Output
                {
                    output_gradient_ptr->write(eid * 2 + 0, gradients[0]);
                    output_gradient_ptr->write(eid * 2 + 1, gradients[1]);

                    output_hessian_ptr->write(eid * 4 + 0, He);
                    output_hessian_ptr->write(eid * 4 + 1, He);
                    output_hessian_ptr->write(eid * 4 + 2, -1.0f * He);
                    output_hessian_ptr->write(eid * 4 + 3, -1.0f * He);
                }
            },
            default_option);

    if (host_sim_data->sa_bending_edges.size() > 0)
        compiler.compile<1>(
            fn_evaluate_bending,
            [sa_x = sim_data->sa_x.view(),
             // sa_cgB = sim_data->sa_cgB.view(),
             // sa_cgA_diag = sim_data->sa_cgA_diag.view(),
             // sa_cgA_offdiag_bending = sim_data->sa_cgA_offdiag_bending.view(),
             output_gradient_ptr = sim_data->sa_bending_edges_gradients.view(),
             output_hessian_ptr  = sim_data->sa_bending_edges_hessians.view(),
             sa_edges            = sim_data->sa_bending_edges.view(),
             sa_bending_edges_Q  = sim_data->sa_bending_edges_Q.view()](const Float stiffness_bending)
            {
                // const Uint curr_prefix = culster->read(cluster_idx);
                // const UInt eid = curr_prefix + dispatch_id().x;

                const UInt eid  = dispatch_id().x;
                UInt4      edge = sa_edges->read(eid);
                Float4x4   m_Q  = sa_bending_edges_Q->read(eid);

                Float3 vert_pos[4] = {
                    sa_x->read(edge[0]),
                    sa_x->read(edge[1]),
                    sa_x->read(edge[2]),
                    sa_x->read(edge[3]),
                };
                Float3 gradients[4] = {
                    make_float3(0.0f),
                    make_float3(0.0f),
                    make_float3(0.0f),
                    make_float3(0.0f),
                };


                for (uint ii = 0; ii < 4; ii++)
                {
                    for (uint jj = 0; jj < 4; jj++)
                    {
                        gradients[ii] += m_Q[ii][jj] * vert_pos[jj];  // -Qx
                    }
                    gradients[ii] = stiffness_bending * gradients[ii];
                }

                // Output
                {
                    output_gradient_ptr->write(eid * 4 + 0, gradients[0]);
                    output_gradient_ptr->write(eid * 4 + 1, gradients[1]);
                    output_gradient_ptr->write(eid * 4 + 2, gradients[2]);
                    output_gradient_ptr->write(eid * 4 + 3, gradients[3]);

                    auto hess = stiffness_bending * luisa::make_float3x3(1.0f);
                    ;
                    output_hessian_ptr->write(eid * 16 + 0, m_Q[0][0] * hess);
                    output_hessian_ptr->write(eid * 16 + 1, m_Q[1][1] * hess);
                    output_hessian_ptr->write(eid * 16 + 2, m_Q[2][2] * hess);
                    output_hessian_ptr->write(eid * 16 + 3, m_Q[3][3] * hess);

                    uint idx = 4;
                    for (uint ii = 0; ii < 4; ii++)
                    {
                        for (uint jj = 0; jj < 4; jj++)
                        {
                            if (ii != jj)
                            {
                                output_hessian_ptr->write(eid * 16 + idx, m_Q[jj][ii] * hess);
                                idx += 1;
                            }
                        }
                    }
                }
            },
            default_option);

    // Assembly
    compiler.compile(
        fn_material_energy_assembly,
        [sa_cgB                 = sim_data->sa_cgB.view(),
         sa_cgA_diag            = sim_data->sa_cgA_diag.view(),
         sa_cgA_offdiag_triplet = sim_data->sa_cgA_fixtopo_offdiag_triplet.view(),

         sa_vert_adj_material_force_verts_csr = sim_data->sa_vert_adj_material_force_verts_csr.view(),
         sa_vert_adj_stretch_springs_csr      = sim_data->sa_vert_adj_stretch_springs_csr.view(),
         sa_vert_adj_bending_edges_csr        = sim_data->sa_vert_adj_bending_edges_csr.view(),

         sa_stretch_springs                    = sim_data->sa_stretch_springs.view(),
         sa_stretch_springs_gradients          = sim_data->sa_stretch_springs_gradients.view(),
         sa_stretch_springs_hessians           = sim_data->sa_stretch_springs_hessians.view(),
         sa_stretch_springs_offsets_in_adjlist = sim_data->sa_stretch_springs_offsets_in_adjlist.view(),

         sa_bending_edges                    = sim_data->sa_bending_edges.view(),
         sa_bending_edges_gradients          = sim_data->sa_bending_edges_gradients.view(),
         sa_bending_edges_hessians           = sim_data->sa_bending_edges_hessians.view(),
         sa_bending_edges_offsets_in_adjlist = sim_data->sa_bending_edges_offsets_in_adjlist.view()]()
        {
            const Uint vid         = dispatch_x();
            const Uint curr_prefix = sa_vert_adj_material_force_verts_csr->read(vid);
            const Uint next_prefix = sa_vert_adj_material_force_verts_csr->read(vid + 1);
            $if(next_prefix - curr_prefix != 0)
            {
                Float3   total_gradiant = Zero3;
                Float3x3 total_diag_A   = Zero3x3;

                const Uint curr_prefix_spring = sa_vert_adj_stretch_springs_csr->read(vid);
                const Uint next_prefix_spring = sa_vert_adj_stretch_springs_csr->read(vid + 1);
                $for(j, curr_prefix_spring, next_prefix_spring)
                {
                    const Uint  adj_eid = sa_vert_adj_stretch_springs_csr->read(j);
                    const Uint2 edge    = sa_stretch_springs->read(adj_eid);
                    const Uint  offset  = lcs::select(vid == edge[0], Uint(0), Uint(1));

                    const Float3   grad      = sa_stretch_springs_gradients->read(adj_eid * 2 + offset);
                    const Float3x3 diag_hess = sa_stretch_springs_hessians->read(adj_eid * 4 + offset);
                    total_gradiant           = total_gradiant + grad;
                    total_diag_A             = total_diag_A + diag_hess;

                    Float3x3 offdiag_hess = sa_stretch_springs_hessians->read(adj_eid * 4 + 2 + offset);
                    Uint offdiag_offset = sa_stretch_springs_offsets_in_adjlist->read(adj_eid * 2 + offset);
                    auto triplet = sa_cgA_offdiag_triplet->read(curr_prefix + offdiag_offset);
                    add_triplet_matrix(triplet, offdiag_hess);
                    sa_cgA_offdiag_triplet->write(curr_prefix + offdiag_offset, triplet);
                };

                const Uint curr_prefix_bending = sa_vert_adj_bending_edges_csr->read(vid);
                const Uint next_prefix_bending = sa_vert_adj_bending_edges_csr->read(vid + 1);
                $for(j, curr_prefix_bending, next_prefix_bending)
                {
                    const Uint  adj_eid = sa_vert_adj_bending_edges_csr->read(j);
                    const UInt4 edge    = sa_bending_edges->read(adj_eid);
                    const Uint  offset  = lcs::select(
                        vid == edge[0],
                        Uint(0),
                        lcs::select(vid == edge[1], Uint(1), lcs::select(vid == edge[2], Uint(2), Uint(3))));
                    const Float3   grad      = sa_bending_edges_gradients->read(adj_eid * 4 + offset);
                    const Float3x3 diag_hess = sa_bending_edges_hessians->read(adj_eid * 16 + offset);
                    total_gradiant           = total_gradiant + grad;
                    total_diag_A             = total_diag_A + diag_hess;
                    for (uint ii = 0; ii < 3; ii++)
                    {
                        Float3x3 offdiag_hess =
                            sa_bending_edges_hessians->read(adj_eid * 16 + 4 + offset * 3 + ii);
                        Uint offdiag_offset =
                            sa_bending_edges_offsets_in_adjlist->read(adj_eid * 12 + offset * 3 + ii);
                        auto triplet = sa_cgA_offdiag_triplet->read(curr_prefix + offdiag_offset);
                        add_triplet_matrix(triplet, offdiag_hess);
                        sa_cgA_offdiag_triplet->write(curr_prefix + offdiag_offset, triplet);
                    }
                };

                sa_cgB->write(vid, sa_cgB->read(vid) - total_gradiant);
                sa_cgA_diag->write(vid, sa_cgA_diag->read(vid) + total_diag_A);
            };
        });

    // SpMV
    // PCG SPMV diagonal kernel
    compiler.compile<1>(
        fn_pcg_spmv_diag,
        [sa_cgA_diag = sim_data->sa_cgA_diag.view()](Var<luisa::compute::Buffer<float3>> sa_input_vec,
                                                     Var<luisa::compute::Buffer<float3>> sa_output_vec)
        {
            const UInt vid         = dispatch_id().x;
            Float3x3   A_diag      = sa_cgA_diag->read(vid);
            Float3     input       = sa_input_vec->read(vid);
            Float3     diag_output = A_diag * input;
            sa_output_vec->write(vid, diag_output);
        },
        default_option);

    compiler.compile(
        fn_reset_cgA_offdiag_triplet,
        [sa_cgA_offdiag_triplet_info = sim_data->sa_cgA_fixtopo_offdiag_triplet_info.view(),
         sa_cgA_offdiag_triplet      = sim_data->sa_cgA_fixtopo_offdiag_triplet.view()]()
        {
            const Uint triplet_idx  = dispatch_x();
            const auto triplet_info = sa_cgA_offdiag_triplet_info->read(triplet_idx);
            sa_cgA_offdiag_triplet->write(
                triplet_idx,
                make_matrix_triplet(triplet_info[0], triplet_info[1], triplet_info[2], make_float3x3(0.0f)));
            ;
        },
        default_option);

    compiler.compile(
        fn_pcg_spmv_offdiag_perVert,
        [sa_vert_adj_material_force_verts_csr = sim_data->sa_vert_adj_material_force_verts_csr.view(),
         sa_cgA_fixtopo_offdiag_triplet       = sim_data->sa_cgA_fixtopo_offdiag_triplet.view()](
            Var<luisa::compute::Buffer<float3>> sa_input_vec, Var<luisa::compute::Buffer<float3>> sa_output_vec)
        {
            const Uint vid = dispatch_x();
            // TODO: Using parallel reduce
            const Uint curr_prefix = sa_vert_adj_material_force_verts_csr->read(vid);
            const Uint next_prefix = sa_vert_adj_material_force_verts_csr->read(vid + 1);
            Float3     output_vec  = sa_output_vec.read(vid);
            $for(j, curr_prefix, next_prefix)
            {
                const Uint adj_vid = sa_vert_adj_material_force_verts_csr->read(j);
                const auto triplet = sa_cgA_fixtopo_offdiag_triplet->read(j);
                // const Uint adj_vid = triplet->get_col_idx();
                output_vec += triplet->get_matrix() * sa_input_vec.read(adj_vid);
            };
            sa_output_vec.write(vid, output_vec);
        },
        default_option);

    compiler.compile(
        fn_pcg_spmv_offdiag_warp_rbk,
        [sa_cgA_offdiag_triplet = sim_data->sa_cgA_fixtopo_offdiag_triplet.view()](
            Var<luisa::compute::Buffer<float3>> sa_input_vec, Var<luisa::compute::Buffer<float3>> sa_output_vec)
        {
            const Uint     triplet_idx     = dispatch_x();
            const Uint     lane_idx        = triplet_idx % 32;
            auto           triplet         = sa_cgA_offdiag_triplet->read(triplet_idx);
            const Uint     vid             = triplet->get_row_idx();
            const Uint     adj_vid         = triplet->get_col_idx();
            const Uint     matrix_property = triplet->get_matrix_property();
            const Float3x3 mat             = read_triplet_matrix(triplet);
            const Float3   input           = sa_input_vec.read(adj_vid);
            const Float3   contrib         = mat * input;
            const Float3   contrib_prefix  = luisa::compute::warp_prefix_sum(contrib);

            // sa_output_vec.atomic(vid).x.fetch_add(contrib.x);
            // sa_output_vec.atomic(vid).y.fetch_add(contrib.y);
            // sa_output_vec.atomic(vid).z.fetch_add(contrib.z);

            $if(MatrixTriplet::is_last_col_in_row(matrix_property))
            {
                const Uint target_laneIdx = MatrixTriplet::read_first_col_info(matrix_property);
                const Float3 start_contrib_prefix = luisa::compute::warp_read_lane(contrib_prefix, target_laneIdx);
                const Float3 sum_contrib = contrib_prefix - start_contrib_prefix + contrib;
                $if(MatrixTriplet::write_use_atomic(matrix_property))
                {
                    sa_output_vec.atomic(vid).x.fetch_add(sum_contrib.x);
                    sa_output_vec.atomic(vid).y.fetch_add(sum_contrib.y);
                    sa_output_vec.atomic(vid).z.fetch_add(sum_contrib.z);
                }
                $else
                {
                    sa_output_vec.write(vid, sa_output_vec.read(vid) + sum_contrib);
                };
            };
        },
        default_option);

    compiler.compile<1>(
        fn_pcg_spmv_offdiag_block_rbk,
        [](Var<luisa::compute::Buffer<MatrixTriplet3x3>> sa_cgA_offdiag_triplet,
           Var<luisa::compute::Buffer<float3>>           sa_input_vec,
           Var<luisa::compute::Buffer<float3>>           sa_output_vec)
        {
            const Uint triplet_idx = dispatch_x();
            const Uint threadIdx   = triplet_idx % 256;
            const Uint warpIdx     = threadIdx / 32;
            const Uint laneIdx     = threadIdx % 32;

            auto       triplet         = sa_cgA_offdiag_triplet->read(triplet_idx);
            Uint       vid             = triplet->get_row_idx();
            const Uint adj_vid         = triplet->get_col_idx();
            Uint       matrix_property = triplet->get_matrix_property();

            Float3 contrib = Zero3;
            $if(MatrixTriplet::is_valid(matrix_property))
            {
                const Float3x3 mat   = read_triplet_matrix(triplet);
                const Float3   input = sa_input_vec.read(adj_vid);
                contrib              = mat * input;
                // sa_output_vec.atomic(vid).x.fetch_add(contrib.x);
                // sa_output_vec.atomic(vid).y.fetch_add(contrib.y);
                // sa_output_vec.atomic(vid).z.fetch_add(contrib.z);
            };


            luisa::compute::set_block_size(256u);
            luisa::compute::Shared<float3> cache_warp_sum(ParallelIntrinsic::warp_num);
            luisa::compute::Shared<float3> cache_target_prefix(ParallelIntrinsic::warp_num);


            const Float3 warp_prefix = luisa::compute::warp_prefix_sum(contrib);
            $if(laneIdx == 31)
            {
                cache_warp_sum[warpIdx] = warp_prefix + contrib;
            };
            luisa::compute::sync_block();
            $if(warpIdx == 0)
            {
                cache_warp_sum[threadIdx] = luisa::compute::warp_prefix_sum(cache_warp_sum[threadIdx]);  // Get warp's prefix in block
            };
            luisa::compute::sync_block();
            const Float3 curr_prefix = cache_warp_sum[warpIdx] + warp_prefix;

            luisa::compute::Shared<float3> cache_prefix(256);
            cache_prefix[threadIdx] = curr_prefix;
            luisa::compute::sync_block();

            // luisa::compute::Shared<float3> cache_prefix(256);
            // ParallelIntrinsic::sort_detail::block_intrinsic_scan_exclusive(triplet_idx, contrib, cache_warp_sum, cache_prefix);
            // const Float3 block_prefix = cache_prefix[threadIdx];
            // luisa::compute::sync_block();

            // $if(MatrixTriplet::is_last_col_in_row(matrix_property))
            // {
            //     const Uint target_threadIdx = MatrixTriplet::read_thread_id_of_first_colIdx_in_warp(matrix_property);
            //     const Float3 target_block_prefix = cache_prefix[target_threadIdx];
            //     const Float3 sum_contrib         = block_prefix - target_block_prefix + contrib;
            //     Uint         target_warpIdx      = 0;
            //     $if(MatrixTriplet::is_first_and_last_col_in_same_warp(matrix_property))
            //     {
            //         target_warpIdx = warpIdx;
            //     }
            //     $else
            //     {
            //         target_warpIdx = MatrixTriplet::read_lane_id_of_first_colIdx_in_warp(matrix_property);
            //     };
            //     device_assert(target_warpIdx == warpIdx, "Error rbk!");
            //     $if(MatrixTriplet::write_use_atomic(matrix_property))
            //     {
            //         sa_output_vec.atomic(vid).x.fetch_add(sum_contrib.x);
            //         sa_output_vec.atomic(vid).y.fetch_add(sum_contrib.y);
            //         sa_output_vec.atomic(vid).z.fetch_add(sum_contrib.z);
            //     }
            //     $else
            //     {
            //         sa_output_vec.write(vid, sa_output_vec.read(vid) + sum_contrib);
            //     };
            // };
            // return;

            $if(MatrixTriplet::is_first_col_in_row(matrix_property)
                & !MatrixTriplet::is_first_and_last_col_in_same_warp(matrix_property))
            {
                cache_target_prefix[warpIdx] = curr_prefix;
            };
            luisa::compute::sync_block();


            Float3 target_block_prefix = Zero3;
            Uint   target_laneIdx      = laneIdx;
            $if(MatrixTriplet::is_last_col_in_row(matrix_property)
                & MatrixTriplet::is_first_and_last_col_in_same_warp(matrix_property))
            {
                target_laneIdx = MatrixTriplet::read_first_col_info(matrix_property);
            };
            target_block_prefix = luisa::compute::warp_read_lane(curr_prefix, target_laneIdx);

            $if(MatrixTriplet::is_last_col_in_row(matrix_property))
            {
                const Uint target_threadIdx = MatrixTriplet::read_first_col_threadIdx(matrix_property);
                const Uint target_index     = MatrixTriplet::read_first_col_info(matrix_property);

                $if(MatrixTriplet::is_first_and_last_col_in_same_warp(matrix_property))
                {
                    // ! We can not read it in this condition statement, as target lane is not active, which will cause invalid access
                    // const Uint target_laneIdx = target_index;
                    // target_block_prefix = luisa::compute::warp_read_lane(cache_prefix[threadIdx], target_laneIdx);
                }
                $else
                {
                    const Uint target_warpIdx = target_index;
                    target_block_prefix       = cache_target_prefix[target_warpIdx];
                };
                const Float3 sum_contrib = curr_prefix - target_block_prefix + contrib;
                device_assert(!is_nan_vec(sum_contrib), "Error NaN detected in SpMV block rbk!");
                $if(MatrixTriplet::write_use_atomic(matrix_property))
                {
                    sa_output_vec.atomic(vid).x.fetch_add(sum_contrib.x);
                    sa_output_vec.atomic(vid).y.fetch_add(sum_contrib.y);
                    sa_output_vec.atomic(vid).z.fetch_add(sum_contrib.z);
                }
                $else
                {
                    sa_output_vec.write(vid, sa_output_vec.read(vid) + sum_contrib);
                };
            };
        });

    // Line search
    // auto fn_reduce_and_add_energy = compiler.compile<1>(
    //     [sa_block_result = sim_data->sa_block_result.view(),
    //      sa_convergence = sim_data->sa_convergence.view()]() {
    //         const Uint index = dispatch_id().x;
    //         Float energy = 0.0f;
    //         {
    //             energy = sa_block_result->read(index);
    //         };
    //         energy = ParallelIntrinsic::block_intrinsic_reduce(index, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);

    //         $if (index == 0) {
    //             sa_convergence->atomic(7).fetch_add(energy);
    //             // buffer_add(sa_convergence, 7, energy);
    //         };
    //     });

    compiler.compile<1>(
        fn_apply_dx,
        [sa_x            = sim_data->sa_x.view(),
         sa_x_iter_start = sim_data->sa_x_iter_start.view(),
         sa_cgX          = sim_data->sa_cgX.view()](const Float alpha)
        {
            const UInt vid = dispatch_id().x;
            sa_x->write(vid, sa_x_iter_start->read(vid) + alpha * sa_cgX->read(vid));
        },
        default_option);

    compiler.compile<1>(
        fn_apply_dx_non_constant,
        [sa_x            = sim_data->sa_x.view(),
         sa_x_iter_start = sim_data->sa_x_iter_start.view(),
         sa_cgX          = sim_data->sa_cgX.view()](Var<BufferView<float>> alpha_buffer)
        {
            const Float alpha = alpha_buffer.read(0);
            const UInt  vid   = dispatch_id().x;
            sa_x->write(vid, sa_x_iter_start->read(vid) + alpha * sa_cgX->read(vid));
        },
        default_option);
}


// Host functions
// Outputs:
//          sa_x_iter_start
//          sa_x_tilde
//          sa_x
//          sa_cgX
void NewtonSolver::host_predict_position()
{
    CpuParallel::parallel_for(0,
                              host_sim_data->num_verts_soft,
                              [sa_x            = host_sim_data->sa_x.data(),
                               sa_v            = host_sim_data->sa_v.data(),
                               sa_cgX          = host_sim_data->sa_cgX.data(),
                               sa_x_step_start = host_sim_data->sa_x_step_start.data(),
                               sa_x_iter_start = host_sim_data->sa_x_iter_start.data(),
                               sa_x_tilde      = host_sim_data->sa_x_tilde.data(),
                               sa_is_fixed     = host_mesh_data->sa_is_fixed.data(),
                               substep_dt      = get_scene_params().get_substep_dt(),
                               gravity         = get_scene_params().gravity](const uint vid)
                              {
                                  // const float3 gravity(0, -9.8f, 0);
                                  float3 x_prev             = sa_x_step_start[vid];
                                  float3 v_prev             = sa_v[vid];
                                  float3 outer_acceleration = gravity;
                                  // If we consider gravity energy here, then we will not consider it in potential energy
                                  float3 v_pred = v_prev + substep_dt * outer_acceleration;
                                  if (sa_is_fixed[vid])
                                  {
                                      // v_pred = Zero3;
                                      v_pred = v_prev;
                                  };

                                  const float3 x_pred  = x_prev + substep_dt * v_pred;
                                  sa_x_iter_start[vid] = x_prev;
                                  sa_x_tilde[vid]      = x_pred;

                                  // sa_x[vid] = x_pred;
                                  // sa_cgX[vid] = v_prev * substep_dt;
                                  sa_x[vid] = x_prev;
                                  // sa_cgX[vid] = luisa::make_float3(0.0f);
                              });

    // Vectorization
    CpuParallel::parallel_for(0,
                              host_sim_data->sa_affine_bodies.size() * 4,
                              [&](const uint block_idx)
                              {
                                  float3 q_prev = host_sim_data->sa_affine_bodies_q_step_start[block_idx];
                                  float3 q_v = host_sim_data->sa_affine_bodies_q_v[block_idx];
                                  // float3 g = host_sim_data->sa_affine_bodies_gravity[block_idx];
                                  float3 g = get_scene_params().gravity;

                                  float  substep_dt = get_scene_params().get_substep_dt();
                                  float3 q_pred     = q_prev + q_v * substep_dt;
                                  if (block_idx % 4 == 0)
                                      q_pred += g * (substep_dt * substep_dt);

                                  // Output
                                  host_sim_data->sa_affine_bodies_q_tilde[block_idx]      = q_pred;
                                  host_sim_data->sa_affine_bodies_q_iter_start[block_idx] = q_prev;
                                  host_sim_data->sa_affine_bodies_q[block_idx]            = q_prev;
                                  // LUISA_INFO("Body {}'s block_{} : q = {}, v = {} , dt = {} => q_tilde = {}", block_idx / 4, block_idx % 4, q_prev, q_v, substep_dt, q_pred);
                              });
}
void NewtonSolver::host_update_velocity()
{
    CpuParallel::parallel_for(0,
                              host_sim_data->num_verts_soft,
                              [sa_x            = host_sim_data->sa_x.data(),
                               sa_v            = host_sim_data->sa_v.data(),
                               sa_x_step_start = host_sim_data->sa_x_step_start.data(),
                               sa_v_step_start = host_sim_data->sa_v_step_start.data(),
                               sa_is_fixed     = host_mesh_data->sa_is_fixed.data(),
                               substep_dt      = get_scene_params().get_substep_dt(),
                               fix_scene       = get_scene_params().fix_scene,
                               damping         = get_scene_params().damping_cloth](const uint vid)
                              {
                                  float3 x_step_begin = sa_x_step_start[vid];
                                  float3 x_step_end   = sa_x[vid];

                                  float3 dx  = x_step_end - x_step_begin;
                                  float3 vel = dx / substep_dt;

                                  if (fix_scene)
                                  {
                                      dx        = Zero3;
                                      vel       = Zero3;
                                      sa_x[vid] = x_step_begin;
                                      return;
                                  };

                                  vel *= luisa::exp(-damping * substep_dt);

                                  sa_v[vid]            = vel;
                                  sa_v_step_start[vid] = vel;
                                  sa_x_step_start[vid] = x_step_end;
                              });

    // Vectorization
    CpuParallel::parallel_for(0,
                              host_sim_data->sa_affine_bodies.size() * 4,
                              [&](const uint block_idx)
                              {
                                  const float substep_dt = get_scene_params().get_substep_dt();
                                  const float damping    = get_scene_params().damping_tet;

                                  float3 q_step_begin = host_sim_data->sa_affine_bodies_q_step_start[block_idx];
                                  float3 q_step_end = host_sim_data->sa_affine_bodies_q[block_idx];

                                  float3 vq = (q_step_end - q_step_begin) / substep_dt
                                              * luisa::exp(-damping * substep_dt);
                                  host_sim_data->sa_affine_bodies_q_v[block_idx]          = vq;
                                  host_sim_data->sa_affine_bodies_q_step_start[block_idx] = q_step_end;
                                  // LUISA_INFO("Body {} 's block {} : vel = {} = from {} to {}", block_idx / 4, block_idx, vq, q_step_begin, q_step_end);
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
        CpuParallel::parallel_set(host_sim_data->sa_cgA_offdiag_affine_body, luisa::make_float3x3(0.0f));
        CpuParallel::parallel_for(
            0,
            host_sim_data->sa_cgA_fixtopo_offdiag_triplet.size(),
            [&](const uint idx)
            {
                auto triplet_info = host_sim_data->sa_cgA_fixtopo_offdiag_triplet_info[idx];
                host_sim_data->sa_cgA_fixtopo_offdiag_triplet[idx] = make_matrix_triplet(
                    triplet_info[0], triplet_info[1], triplet_info[2], luisa::make_float3x3(0.0f));
            });
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
void           NewtonSolver::host_evaluate_inertia()
{
    const float stiffness_dirichlet = get_scene_params().stiffness_dirichlet;

    CpuParallel::parallel_for(0,
                              host_sim_data->num_verts_soft,
                              [sa_cgB       = host_sim_data->sa_cgB.data(),
                               sa_cgA_diag  = host_sim_data->sa_cgA_diag.data(),
                               sa_x         = host_sim_data->sa_x.data(),
                               sa_x_tilde   = host_sim_data->sa_x_tilde.data(),
                               sa_is_fixed  = host_mesh_data->sa_is_fixed.data(),
                               sa_vert_mass = host_mesh_data->sa_vert_mass.data(),
                               substep_dt   = get_scene_params().get_substep_dt(),
                               stiffness_dirichlet](const uint vid)
                              {
                                  const float h       = substep_dt;
                                  const float h_2_inv = 1.f / (h * h);

                                  float3 x_k     = sa_x[vid];
                                  float3 x_tilde = sa_x_tilde[vid];
                                  // float3 v_0 = sa_v[vid];

                                  float    mass     = sa_vert_mass[vid];
                                  float3   gradient = -mass * h_2_inv * (x_k - x_tilde);
                                  float3x3 hessian  = luisa::make_float3x3(1.0f) * mass * h_2_inv;

                                  if (sa_is_fixed[vid])
                                  {
                                      gradient = stiffness_dirichlet * gradient;
                                      hessian  = stiffness_dirichlet * hessian;
                                  }
                                  {
                                      if constexpr (print_detail)
                                          LUISA_INFO("vid {}, mass: {}, move = {}, gradient: {}, hessian: {}",
                                                     vid,
                                                     mass,
                                                     length_vec(x_k - x_tilde),
                                                     gradient,
                                                     hessian);
                                      // sa_cgX[vid] = dx_0;
                                      sa_cgB[vid]      = gradient;
                                      sa_cgA_diag[vid] = hessian;
                                  }
                              });

    float3*   affine_body_cgB      = &host_sim_data->sa_cgB[host_sim_data->num_verts_soft];
    float3x3* affine_body_cgA_diag = &host_sim_data->sa_cgA_diag[host_sim_data->num_verts_soft];
    CpuParallel::parallel_for(
        0,
        host_sim_data->sa_affine_bodies.size(),
        [&](const uint body_idx)
        {
            const float substep_dt = get_scene_params().get_substep_dt();
            const float h          = substep_dt;
            const float h_2_inv    = 1.f / (h * h);

            float3 q_k      = host_sim_data->sa_affine_bodies_q[body_idx];
            float3 q_tilde  = host_sim_data->sa_affine_bodies_q_tilde[body_idx];
            float3 delta[4] = {
                host_sim_data->sa_affine_bodies_q[4 * body_idx + 0]
                    - host_sim_data->sa_affine_bodies_q_tilde[4 * body_idx + 0],
                host_sim_data->sa_affine_bodies_q[4 * body_idx + 1]
                    - host_sim_data->sa_affine_bodies_q_tilde[4 * body_idx + 1],
                host_sim_data->sa_affine_bodies_q[4 * body_idx + 2]
                    - host_sim_data->sa_affine_bodies_q_tilde[4 * body_idx + 2],
                host_sim_data->sa_affine_bodies_q[4 * body_idx + 3]
                    - host_sim_data->sa_affine_bodies_q_tilde[4 * body_idx + 3],
            };
            float3x3 offdiags = host_sim_data->sa_affine_bodies_mass_matrix_compressed_offdiag[body_idx];
            float3x3 offdiag_first_col[3] = {luisa::make_float3x3(offdiags[0], Zero3, Zero3),
                                             luisa::make_float3x3(Zero3, offdiags[1], Zero3),
                                             luisa::make_float3x3(Zero3, Zero3, offdiags[2])};

            float3 gradient[4] = {Zero3, Zero3, Zero3, Zero3};
            for (uint ii = 0; ii < 4; ii++)
            {
                gradient[ii] += host_sim_data->sa_affine_bodies_mass_matrix_diag[4 * body_idx + ii] * delta[ii];
            }
            for (uint ii = 0; ii < 3; ii++)
            {
                gradient[1 + ii] += offdiag_first_col[ii] * delta[0];  // First column offdiag
                gradient[0] += luisa::transpose(offdiag_first_col[ii]) * delta[1 + ii];  // First row offdiag
            }

            affine_body_cgB[4 * body_idx + 0] = -h_2_inv * gradient[0];
            affine_body_cgB[4 * body_idx + 1] = -h_2_inv * gradient[1];
            affine_body_cgB[4 * body_idx + 2] = -h_2_inv * gradient[2];
            affine_body_cgB[4 * body_idx + 3] = -h_2_inv * gradient[3];

            affine_body_cgA_diag[4 * body_idx + 0] =
                h_2_inv * host_sim_data->sa_affine_bodies_mass_matrix_diag[4 * body_idx + 0];
            affine_body_cgA_diag[4 * body_idx + 1] =
                h_2_inv * host_sim_data->sa_affine_bodies_mass_matrix_diag[4 * body_idx + 1];
            affine_body_cgA_diag[4 * body_idx + 2] =
                h_2_inv * host_sim_data->sa_affine_bodies_mass_matrix_diag[4 * body_idx + 2];
            affine_body_cgA_diag[4 * body_idx + 3] =
                h_2_inv * host_sim_data->sa_affine_bodies_mass_matrix_diag[4 * body_idx + 3];

            host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 0] =
                h_2_inv * luisa::transpose(offdiag_first_col[0]);
            host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 1] =
                h_2_inv * luisa::transpose(offdiag_first_col[1]);
            host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 2] =
                h_2_inv * luisa::transpose(offdiag_first_col[2]);
        });
}
void NewtonSolver::host_evaluate_orthogonality()
{
    float3*   affine_body_cgB      = &host_sim_data->sa_cgB[host_sim_data->num_verts_soft];
    float3x3* affine_body_cgA_diag = &host_sim_data->sa_cgA_diag[host_sim_data->num_verts_soft];

    CpuParallel::parallel_for_each_core(
        0,
        host_sim_data->sa_affine_bodies.size(),
        [&](const uint body_idx)
        {
            float3   body_force[4]   = {Zero3};
            float3x3 body_hessian[6] = {Zero3x3};

            const float substep_dt = get_scene_params().get_substep_dt();
            const float h          = substep_dt;
            const float h_2_inv    = 1.f / (h * h);

            float3      A[3]  = {host_sim_data->sa_affine_bodies_q[4 * body_idx + 1],
                                 host_sim_data->sa_affine_bodies_q[4 * body_idx + 2],
                                 host_sim_data->sa_affine_bodies_q[4 * body_idx + 3]};
            const float kappa = 1e5f;
            const float V     = host_sim_data->sa_affine_bodies_volume[body_idx];

            float stiff = kappa;  //* V;
            for (uint ii = 0; ii < 3; ii++)
            {
                float3 grad = (-1.0f) * A[ii];
                for (uint jj = 0; jj < 3; jj++)
                {
                    grad += dot_vec(A[ii], A[jj]) * A[jj];
                }
                // cgB.block<3, 1>(3 + 3 * ii, 0) -= 4 * stiff * float3_to_eigen3(grad);
                body_force[1 + ii] -= 4 * stiff * grad;  // Force
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
                        float    qiTqi = dot_vec(A[ii], A[ii]) - 1.0f;
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
                    // LUISA_INFO("hess of {} adj {} = {}", ii, jj, hessian);
                    // cgA.block<3, 3>(3 + 3 * ii, 3 + 3 * jj) += 4.0f * stiff * float3x3_to_eigen3x3(hessian);
                    body_hessian[idx] = body_hessian[idx] + 4.0f * stiff * hessian;
                    idx += 1;
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
            host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 3] =
                host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 3] + body_hessian[1];
            host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 4] =
                host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 4] + body_hessian[2];
            host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 5] =
                host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 5] + body_hessian[4];
        });
}
void NewtonSolver::host_evaluate_dirichlet()
{
    return;

    const float stiffness_dirichlet = get_scene_params().stiffness_dirichlet;
    const float substep_dt          = get_scene_params().get_substep_dt();
    CpuParallel::parallel_for(0,
                              host_mesh_data->num_verts,
                              [sa_cgB       = host_sim_data->sa_cgB.data(),
                               sa_cgA_diag  = host_sim_data->sa_cgA_diag.data(),
                               sa_x         = host_sim_data->sa_x.data(),
                               sa_x_tilde   = host_sim_data->sa_x_tilde.data(),
                               sa_is_fixed  = host_mesh_data->sa_is_fixed.data(),
                               sa_vert_mass = host_mesh_data->sa_vert_mass.data(),
                               stiffness_dirichlet,
                               substep_dt](const uint vid)
                              {
                                  bool is_fixed = sa_is_fixed[vid];

                                  if (is_fixed)
                                  {
                                      const float h       = substep_dt;
                                      const float h_2_inv = 1.f / (h * h);

                                      float3 x_k     = sa_x[vid];
                                      float3 x_tilde = sa_x_tilde[vid];
                                      // float3 gradient = -stiffness_dirichlet * (x_k - x_tilde);
                                      // float3x3 hessian = stiffness_dirichlet * luisa::make_float3x3(1.0f);
                                      float mass = sa_vert_mass[vid];
                                      float3 gradient = stiffness_dirichlet * h_2_inv * mass * (x_k - x_tilde);
                                      float3x3 hessian =
                                          stiffness_dirichlet * h_2_inv * mass * luisa::make_float3x3(1.0f);
                                      // sa_cgB[vid] = -gradient;
                                      // sa_cgA_diag[vid] = hessian;
                                      sa_cgB[vid]      = sa_cgB[vid] - gradient;
                                      sa_cgA_diag[vid] = sa_cgA_diag[vid] + hessian;
                                  };
                              });
}
void NewtonSolver::host_evaluate_ground_collision()
{
    if (!get_scene_params().use_floor)
        return;

    auto* sa_is_fixed       = host_mesh_data->sa_is_fixed.data();
    auto* sa_rest_vert_area = host_mesh_data->sa_rest_vert_area.data();

    const uint  num_verts        = host_sim_data->num_verts_soft;
    const float floor_y          = get_scene_params().floor.y;
    float       d_hat            = get_scene_params().d_hat;
    float       thickness        = get_scene_params().thickness;
    float       stiffness_ground = 1e7f;

    CpuParallel::parallel_for(0,
                              host_sim_data->num_verts_soft,
                              [sa_cgB            = host_sim_data->sa_cgB.data(),
                               sa_cgA_diag       = host_sim_data->sa_cgA_diag.data(),
                               sa_x              = host_sim_data->sa_x.data(),
                               sa_is_fixed       = host_mesh_data->sa_is_fixed.data(),
                               sa_rest_vert_area = host_mesh_data->sa_rest_vert_area.data(),
                               sa_vert_mass      = host_mesh_data->sa_vert_mass.data(),
                               substep_dt        = get_scene_params().get_substep_dt(),
                               d_hat             = d_hat,
                               floor_y           = floor_y,
                               thickness         = thickness,
                               stiffness_ground  = stiffness_ground](const uint vid)
                              {
                                  if (sa_is_fixed[vid])
                                      return;
                                  if (get_scene_params().use_floor)
                                  {
                                      float3 x_k  = sa_x[vid];
                                      float  diff = x_k.y - get_scene_params().floor.y;

                                      float3   force   = luisa::make_float3(0.0f);
                                      float3x3 hessian = makeFloat3x3(0.0f);
                                      if (diff < d_hat + thickness)
                                      {
                                          float  C      = d_hat + thickness - diff;
                                          float3 normal = luisa::make_float3(0, 1, 0);
                                          float  area   = sa_rest_vert_area[vid];
                                          float  stiff  = stiffness_ground * area;
                                          force         = stiff * C * normal;
                                          hessian       = stiff * outer_product(normal, normal);
                                      }
                                      {
                                          // sa_cgX[vid] = dx_0;
                                          sa_cgB[vid]      = sa_cgB[vid] + force;
                                          sa_cgA_diag[vid] = sa_cgA_diag[vid] + hessian;
                                      }
                                  }
                              });

    float3*   affine_body_cgB      = &host_sim_data->sa_cgB[host_sim_data->num_verts_soft];
    float3x3* affine_body_cgA_diag = &host_sim_data->sa_cgA_diag[host_sim_data->num_verts_soft];

    CpuParallel::parallel_for_each_core(
        0,
        host_sim_data->sa_affine_bodies.size(),
        [&](const uint body_idx)
        {
            const uint mesh_idx    = host_sim_data->sa_affine_bodies[body_idx];
            const uint curr_prefix = host_mesh_data->prefix_num_verts[mesh_idx];
            const uint next_prefix = host_mesh_data->prefix_num_verts[mesh_idx + 1];

            float3   body_force[4]    = {Zero3};
            float3x3 body_hessian[10] = {Zero3x3};
            // EigenFloat12 eigen_B = EigenFloat12::Zero();
            // EigenFloat12x12 eigen_A = EigenFloat12x12::Zero();
            for (uint vid = curr_prefix; vid < next_prefix; vid++)
            {
                float3 x_k  = host_sim_data->sa_x[vid];
                float  diff = x_k.y - floor_y;

                if (diff < d_hat + thickness)
                {
                    float    C       = d_hat + thickness - diff;
                    float3   normal  = luisa::make_float3(0, 1, 0);
                    float    area    = sa_rest_vert_area[vid];
                    float    stiff   = 1e9f * area;
                    float    k1      = stiff * C;
                    float3   model_x = host_mesh_data->sa_scaled_model_x[vid];
                    float3   force   = stiff * C * normal;
                    float3x3 hessian = stiff * outer_product(normal, normal);
                    {
                        float3x3 curr_hessian[10];
                        float3   curr_force[4];
                        AffineBodyDynamics::affine_Jacobian_to_gradient(model_x, force, curr_force);
                        AffineBodyDynamics::affine_Jacobian_to_hessian(model_x, model_x, hessian, curr_hessian);
                        for (uint jj = 0; jj < 4; jj++)
                        {
                            body_force[jj] += curr_force[jj];
                        }
                        for (uint jj = 0; jj < 10; jj++)
                        {
                            body_hessian[jj] = body_hessian[jj] + curr_hessian[jj];
                        }
                        // for (uint jj = 0; jj < 4; jj++) { LUISA_INFO("For vert {} : force = {}", vid, curr_force[jj]); }
                        // for (uint jj = 0; jj < 10; jj++) { LUISA_INFO("For vert {} : hess = {}", vid, curr_hessian[jj]); }
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
            host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 0] =
                host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 0] + body_hessian[1];
            host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 1] =
                host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 1] + body_hessian[2];
            host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 2] =
                host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 2] + body_hessian[3];
            host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 3] =
                host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 3] + body_hessian[5];
            host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 4] =
                host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 4] + body_hessian[6];
            host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 5] =
                host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 5] + body_hessian[8];
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
    EigenFloat12    cgB = EigenFloat12::Zero();

    // Inertia
    if constexpr (false)
    {
        CpuParallel::single_thread_for(
            0,
            host_sim_data->sa_affine_bodies.size(),
            [&](const uint body_idx)
            {
                const float substep_dt = get_scene_params().get_substep_dt();
                const float h          = substep_dt;
                const float h_2_inv    = 1.f / (h * h);

                auto         M     = host_sim_data->sa_affine_bodies_mass_matrix_full[body_idx];
                EigenFloat12 delta = EigenFloat12::Zero();

                delta.block<3, 1>(0, 0) =
                    float3_to_eigen3(host_sim_data->sa_affine_bodies_q[4 * body_idx + 0]
                                     - host_sim_data->sa_affine_bodies_q_tilde[4 * body_idx + 0]);
                delta.block<3, 1>(3, 0) =
                    float3_to_eigen3(host_sim_data->sa_affine_bodies_q[4 * body_idx + 1]
                                     - host_sim_data->sa_affine_bodies_q_tilde[4 * body_idx + 1]);
                delta.block<3, 1>(6, 0) =
                    float3_to_eigen3(host_sim_data->sa_affine_bodies_q[4 * body_idx + 2]
                                     - host_sim_data->sa_affine_bodies_q_tilde[4 * body_idx + 2]);
                delta.block<3, 1>(9, 0) =
                    float3_to_eigen3(host_sim_data->sa_affine_bodies_q[4 * body_idx + 3]
                                     - host_sim_data->sa_affine_bodies_q_tilde[4 * body_idx + 3]);

                EigenFloat12    gradient = h_2_inv * M * delta;
                EigenFloat12x12 hessian  = h_2_inv * M;

                cgB -= gradient;
                cgA += hessian;
            });
    }

    // Ground collision
    if constexpr (false)
    {
        const float d_hat     = get_scene_params().d_hat;
        const float thickness = get_scene_params().thickness;
        CpuParallel::single_thread_for(
            0,
            host_sim_data->sa_affine_bodies.size(),
            [&](const uint body_idx)
            {
                if (get_scene_params().use_floor)
                {
                    const uint mesh_idx    = host_sim_data->sa_affine_bodies[body_idx];
                    const uint curr_prefix = host_mesh_data->prefix_num_verts[mesh_idx];
                    const uint next_prefix = host_mesh_data->prefix_num_verts[mesh_idx + 1];

                    float3   body_force[4]       = {Zero3};
                    float3x3 body_hessian[4][10] = {Zero3x3};
                    // EigenFloat12 eigen_B = EigenFloat12::Zero();
                    // EigenFloat12x12 eigen_A = EigenFloat12x12::Zero();
                    for (uint vid = curr_prefix; vid < next_prefix; vid++)
                    {
                        float3 x_k  = host_sim_data->sa_x[vid];
                        float  diff = x_k.y - get_scene_params().floor.y;

                        if (diff < d_hat + thickness)
                        {
                            float    C       = d_hat + thickness - diff;
                            float3   normal  = luisa::make_float3(0, 1, 0);
                            float    area    = host_mesh_data->sa_rest_vert_area[vid];
                            float    stiff   = 1e9f * area;
                            float    k1      = stiff * C;
                            float3   model_x = host_mesh_data->sa_scaled_model_x[vid];
                            float3   force   = stiff * C * normal;
                            float3x3 hessian = stiff * outer_product(normal, normal);
                            auto     J       = AffineBodyDynamics::get_jacobian_dxdq(model_x);
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
                    affine_body_cgA_diag[4 * body_idx + 0] =
                        affine_body_cgA_diag[4 * body_idx + 0] + eigen3x3_to_float3x3(cgA.block<3, 3>(0, 0));
                    affine_body_cgA_diag[4 * body_idx + 1] =
                        affine_body_cgA_diag[4 * body_idx + 1] + eigen3x3_to_float3x3(cgA.block<3, 3>(3, 3));
                    affine_body_cgA_diag[4 * body_idx + 2] =
                        affine_body_cgA_diag[4 * body_idx + 2] + eigen3x3_to_float3x3(cgA.block<3, 3>(6, 6));
                    affine_body_cgA_diag[4 * body_idx + 3] =
                        affine_body_cgA_diag[4 * body_idx + 3] + eigen3x3_to_float3x3(cgA.block<3, 3>(9, 9));
                    host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 0] =
                        host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 0]
                        + eigen3x3_to_float3x3(cgA.block<3, 3>(0, 3));
                    host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 1] =
                        host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 1]
                        + eigen3x3_to_float3x3(cgA.block<3, 3>(0, 6));
                    host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 2] =
                        host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 2]
                        + eigen3x3_to_float3x3(cgA.block<3, 3>(0, 9));
                    host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 3] =
                        host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 3]
                        + eigen3x3_to_float3x3(cgA.block<3, 3>(3, 6));
                    host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 4] =
                        host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 4]
                        + eigen3x3_to_float3x3(cgA.block<3, 3>(3, 9));
                    host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 5] =
                        host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 5]
                        + eigen3x3_to_float3x3(cgA.block<3, 3>(6, 9));
                }
            });
    }

    const float d_hat     = get_scene_params().d_hat;
    const float thickness = get_scene_params().thickness;
    CpuParallel::single_thread_for(
        0,
        host_sim_data->sa_affine_bodies.size(),
        [&](const uint body_idx)
        {
            const uint mesh_idx    = host_sim_data->sa_affine_bodies[body_idx];
            const uint curr_prefix = host_mesh_data->prefix_num_verts[mesh_idx];
            const uint next_prefix = host_mesh_data->prefix_num_verts[mesh_idx + 1];

            float3   body_force[4]       = {Zero3};
            float3x3 body_hessian[4][10] = {Zero3x3};
            // EigenFloat12 eigen_B = EigenFloat12::Zero();
            // EigenFloat12x12 eigen_A = EigenFloat12x12::Zero();

            // Orthogonality potential
            if constexpr (false)
            {
                float3      A[3]  = {host_sim_data->sa_affine_bodies_q[4 * body_idx + 1],
                                     host_sim_data->sa_affine_bodies_q[4 * body_idx + 2],
                                     host_sim_data->sa_affine_bodies_q[4 * body_idx + 3]};
                const float kappa = 1e5f;
                const float V     = host_sim_data->sa_affine_bodies_volume[body_idx];

                float stiff = kappa;  //* V;
                for (uint ii = 0; ii < 3; ii++)
                {
                    float3 grad = (-1.0f) * A[ii];
                    for (uint jj = 0; jj < 3; jj++)
                    {
                        grad += dot_vec(A[ii], A[jj]) * A[jj];
                    }
                    cgB.block<3, 1>(3 + 3 * ii, 0) -= 4 * stiff * float3_to_eigen3(grad);
                    // body_force[1 + ii] -= 4 * stiff * g; // Force
                    // LUISA_INFO("Force of col {} = {}", 1 + ii, g);
                }
                for (uint ii = 0; ii < 3; ii++)
                {
                    for (uint jj = 0; jj < 3; jj++)
                    {
                        float3x3 hessian = Zero3x3;
                        if (ii == jj)
                        {
                            float3x3 qiqiT = outer_product(A[ii], A[ii]);
                            float    qiTqi = dot_vec(A[ii], A[ii]) - 1.0f;
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
                        // LUISA_INFO("hess of {} adj {} = {}", ii, jj, hessian);
                        cgA.block<3, 3>(3 + 3 * ii, 3 + 3 * jj) += 4.0f * stiff * float3x3_to_eigen3x3(hessian);
                        // body_hessian[ii][jj] = body_hessian[ii][jj] + 4.0f * stiff * hessian; idx += 1;
                    }
                }
            }
        });

    float3*   affine_body_cgB      = &host_sim_data->sa_cgB[host_sim_data->num_verts_soft];
    float3x3* affine_body_cgA_diag = &host_sim_data->sa_cgA_diag[host_sim_data->num_verts_soft];

    const uint body_idx = 0;
    affine_body_cgB[4 * body_idx + 0] += eigen3_to_float3(cgB.block<3, 1>(0, 0));
    affine_body_cgB[4 * body_idx + 1] += eigen3_to_float3(cgB.block<3, 1>(3, 0));
    affine_body_cgB[4 * body_idx + 2] += eigen3_to_float3(cgB.block<3, 1>(6, 0));
    affine_body_cgB[4 * body_idx + 3] += eigen3_to_float3(cgB.block<3, 1>(9, 0));
    affine_body_cgA_diag[4 * body_idx + 0] =
        affine_body_cgA_diag[4 * body_idx + 0] + eigen3x3_to_float3x3(cgA.block<3, 3>(0, 0));
    affine_body_cgA_diag[4 * body_idx + 1] =
        affine_body_cgA_diag[4 * body_idx + 1] + eigen3x3_to_float3x3(cgA.block<3, 3>(3, 3));
    affine_body_cgA_diag[4 * body_idx + 2] =
        affine_body_cgA_diag[4 * body_idx + 2] + eigen3x3_to_float3x3(cgA.block<3, 3>(6, 6));
    affine_body_cgA_diag[4 * body_idx + 3] =
        affine_body_cgA_diag[4 * body_idx + 3] + eigen3x3_to_float3x3(cgA.block<3, 3>(9, 9));
    host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 0] =
        host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 0] + eigen3x3_to_float3x3(cgA.block<3, 3>(0, 3));
    host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 1] =
        host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 1] + eigen3x3_to_float3x3(cgA.block<3, 3>(0, 6));
    host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 2] =
        host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 2] + eigen3x3_to_float3x3(cgA.block<3, 3>(0, 9));
    host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 3] =
        host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 3] + eigen3x3_to_float3x3(cgA.block<3, 3>(3, 6));
    host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 4] =
        host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 4] + eigen3x3_to_float3x3(cgA.block<3, 3>(3, 9));
    host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 5] =
        host_sim_data->sa_cgA_offdiag_affine_body[6 * body_idx + 5] + eigen3x3_to_float3x3(cgA.block<3, 3>(6, 9));


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
    CpuParallel::parallel_for(0,
                              host_sim_data->sa_stretch_springs.size(),
                              [sa_x     = host_sim_data->sa_x.data(),
                               sa_edges = host_sim_data->sa_stretch_springs.data(),
                               sa_rest_length = host_sim_data->sa_stretch_spring_rest_state_length.data(),
                               output_gradient_ptr = host_sim_data->sa_stretch_springs_gradients.data(),
                               output_hessian_ptr  = host_sim_data->sa_stretch_springs_hessians.data(),
                               stiffness_stretch   = get_scene_params().stiffness_spring](const uint eid)
                              {
                                  uint2 edge = sa_edges[eid];

                                  float3   vert_pos[2]  = {sa_x[edge[0]], sa_x[edge[1]]};
                                  float3   gradients[2] = {Zero3, Zero3};
                                  float3x3 He           = luisa::make_float3x3(0.0f);

                                  const float L                        = sa_rest_length[eid];
                                  const float stiffness_stretch_spring = stiffness_stretch;

                                  float3 diff = vert_pos[0] - vert_pos[1];
                                  float  l    = max_scalar(length_vec(diff), Epsilon);
                                  float  l0   = L;
                                  float  C    = l - l0;

                                  float3 dir = diff / l;
                                  // float3 dir = normalize_vec(diff);
                                  float3x3 nnT           = outer_product(dir, dir);
                                  float    x_inv         = 1.f / l;
                                  float    x_squared_inv = x_inv * x_inv;

                                  gradients[0] = stiffness_stretch_spring * dir * C;
                                  gradients[1] = -gradients[0];
                                  He           = stiffness_stretch_spring * nnT
                                       + stiffness_stretch_spring * max_scalar(1.0f - L * x_inv, 0.0f)
                                             * (luisa::make_float3x3(1.0f) - nnT);

                                  // Output
                                  {
                                      output_gradient_ptr[eid * 2 + 0] = gradients[0];
                                      output_gradient_ptr[eid * 2 + 1] = gradients[1];

                                      output_hessian_ptr[eid * 4 + 0] = He;
                                      output_hessian_ptr[eid * 4 + 1] = He;
                                      output_hessian_ptr[eid * 4 + 2] = -1.0f * He;
                                      output_hessian_ptr[eid * 4 + 3] = -1.0f * He;
                                  }
                              });
}
void NewtonSolver::host_evaluete_bending()
{
    CpuParallel::parallel_for(
        0,
        host_sim_data->sa_bending_edges.size(),
        [sa_x                = host_sim_data->sa_x.data(),
         sa_bending_edges    = host_sim_data->sa_bending_edges.data(),
         sa_bending_edges_Q  = host_sim_data->sa_bending_edges_Q.data(),
         output_gradient_ptr = host_sim_data->sa_bending_edges_gradients.data(),
         output_hessian_ptr  = host_sim_data->sa_bending_edges_hessians.data(),
         stiffness_bending   = get_scene_params().get_stiffness_quadratic_bending()](const uint eid)
        {
            uint4    edge = sa_bending_edges[eid];
            float4x4 m_Q  = sa_bending_edges_Q[eid];

            float3 vert_pos[4] = {sa_x[edge[0]], sa_x[edge[1]], sa_x[edge[2]], sa_x[edge[3]]};

            float3 gradients[4] = {Zero3, Zero3, Zero3, Zero3};
            for (uint ii = 0; ii < 4; ii++)
            {
                for (uint jj = 0; jj < 4; jj++)
                {
                    gradients[ii] += m_Q[ii][jj] * vert_pos[jj];  // -Qx
                }
                gradients[ii] = stiffness_bending * gradients[ii];
            }

            // Output
            {
                output_gradient_ptr[eid * 4 + 0] = gradients[0];
                output_gradient_ptr[eid * 4 + 1] = gradients[1];
                output_gradient_ptr[eid * 4 + 2] = gradients[2];
                output_gradient_ptr[eid * 4 + 3] = gradients[3];

                auto hess = stiffness_bending * luisa::make_float3x3(1.0f);
                ;
                output_hessian_ptr[eid * 16 + 0] = m_Q[0][0] * hess;
                output_hessian_ptr[eid * 16 + 1] = m_Q[1][1] * hess;
                output_hessian_ptr[eid * 16 + 2] = m_Q[2][2] * hess;
                output_hessian_ptr[eid * 16 + 3] = m_Q[3][3] * hess;

                uint idx = 4;
                for (uint ii = 0; ii < 4; ii++)
                {
                    for (uint jj = 0; jj < 4; jj++)
                    {
                        if (ii != jj)
                        {
                            output_hessian_ptr[eid * 16 + idx] = m_Q[jj][ii] * hess;
                            idx += 1;
                        }
                    }
                }
            }
        });
}
void NewtonSolver::host_material_energy_assembly()
{
    // Assemble spring
    {
        CpuParallel::parallel_for(
            0,
            host_sim_data->num_verts_soft,
            [&](const uint vid)
            {
                const uint curr_prefix = host_sim_data->sa_vert_adj_material_force_verts_csr[vid];
                const uint next_prefix = host_sim_data->sa_vert_adj_material_force_verts_csr[vid + 1];
                if (next_prefix - curr_prefix != 0)
                {
                    float3                total_gradiant = Zero3;
                    float3x3              total_diag_A   = Zero3x3;
                    std::vector<float3x3> total_offdiag_A(next_prefix - curr_prefix, Zero3x3);
                    const auto&           adj_springs = host_sim_data->vert_adj_stretch_springs[vid];
                    for (const uint adj_eid : adj_springs)
                    {
                        auto       edge   = host_sim_data->sa_stretch_springs[adj_eid];
                        const uint offset = vid == edge[0] ? 0 : 1;
                        // const float3* gradient_ptr = host_sim_data->sa_stretch_springs_gradients.data() + adj_eid * 2;
                        // const float3x3* hessian_ptr = host_sim_data->sa_stretch_springs_hessians.data() + adj_eid * 4;
                        float3 grad = host_sim_data->sa_stretch_springs_gradients[adj_eid * 2 + offset];
                        float3x3 diag_hess = host_sim_data->sa_stretch_springs_hessians[adj_eid * 4 + offset];
                        total_gradiant = total_gradiant + grad;
                        total_diag_A   = total_diag_A + diag_hess;

                        float3x3 offdiag_hess =
                            host_sim_data->sa_stretch_springs_hessians[adj_eid * 4 + 2 + offset * 1 + 0];
                        auto* offsets_in_adjlist_ptr =
                            host_sim_data->sa_stretch_springs_offsets_in_adjlist.data() + adj_eid * 2 + offset * 1;
                        uint offdiag_offset             = offsets_in_adjlist_ptr[0];
                        total_offdiag_A[offdiag_offset] = total_offdiag_A[offdiag_offset] + offdiag_hess;
                    }
                    const auto& adj_bending_edges = host_sim_data->vert_adj_bending_edges[vid];
                    for (const uint adj_eid : adj_bending_edges)
                    {
                        auto edge   = host_sim_data->sa_bending_edges[adj_eid];
                        uint offset = 0;
                        if (vid == edge[0])
                        {
                            offset = 0;
                        }
                        else if (vid == edge[1])
                        {
                            offset = 1;
                        }
                        else if (vid == edge[2])
                        {
                            offset = 2;
                        }
                        else if (vid == edge[3])
                        {
                            offset = 3;
                        }
                        // const float3* gradient_ptr = host_sim_data->sa_bending_edges_gradients.data() + adj_eid * 4;
                        // const float3x3* diag_hessian_ptr = host_sim_data->sa_bending_edges_hessians.data() + adj_eid * 16;
                        // const float3x3* offdiag_hessian_ptr = host_sim_data->sa_bending_edges_hessians.data() + adj_eid * 16 + 4;
                        float3 grad = host_sim_data->sa_bending_edges_gradients[adj_eid * 4 + offset];
                        float3x3 diag_hess = host_sim_data->sa_bending_edges_hessians[adj_eid * 16 + offset];
                        total_gradiant = total_gradiant + grad;
                        total_diag_A   = total_diag_A + diag_hess;

                        // auto* offsets_in_adjlist_ptr = host_sim_data->sa_bending_edges_offsets_in_adjlist.data() + adj_eid * 4 + offset * 3;
                        for (uint ii = 0; ii < 3; ii++)
                        {
                            uint offdiag_offset =
                                host_sim_data->sa_bending_edges_offsets_in_adjlist[adj_eid * 12 + offset * 3 + ii];
                            // float3x3 offdiag_hess = offdiag_hessian_ptr[offset * 3 + ii];
                            float3x3 offdiag_hess =
                                host_sim_data->sa_bending_edges_hessians[adj_eid * 16 + 4 + offset * 3 + ii];
                            total_offdiag_A[offdiag_offset] = total_offdiag_A[offdiag_offset] + offdiag_hess;
                            if (offdiag_offset >= total_offdiag_A.size())
                            {
                                LUISA_ERROR("Bending hessian offset out of range! {} vs {}",
                                            offdiag_offset,
                                            total_offdiag_A.size());
                            }
                        }
                    }
                    host_sim_data->sa_cgB[vid]      = host_sim_data->sa_cgB[vid] - total_gradiant;
                    host_sim_data->sa_cgA_diag[vid] = host_sim_data->sa_cgA_diag[vid] + total_diag_A;
                    for (uint ii = curr_prefix; ii < next_prefix; ii++)
                    {
                        uint offset  = ii - curr_prefix;
                        auto triplet = host_sim_data->sa_cgA_fixtopo_offdiag_triplet[ii];
                        add_triplet_matrix(triplet, total_offdiag_A[offset]);
                        host_sim_data->sa_cgA_fixtopo_offdiag_triplet[ii] = triplet;
                    }
                }
            },
            32);
    }
}

// Device functions
void NewtonSolver::device_broadphase_ccd(luisa::compute::Stream& stream)
{
    const float thickness       = get_scene_params().thickness;
    const float ccd_query_range = thickness + 0;  // + d_hat ???

    mp_narrowphase_detector->reset_broadphase_count(stream);

    mp_lbvh_face->update_face_tree_leave_aabb(
        stream, thickness, sim_data->sa_x_iter_start, sim_data->sa_x, mesh_data->sa_faces);
    mp_lbvh_face->refit(stream);
    mp_lbvh_face->broad_phase_query_from_verts(
        stream,
        sim_data->sa_x_iter_start,
        sim_data->sa_x,
        collision_data->broad_phase_collision_count.view(collision_data->get_vf_count_offset(), 1),
        collision_data->broad_phase_list_vf,
        ccd_query_range);

    mp_lbvh_edge->update_edge_tree_leave_aabb(
        stream, thickness, sim_data->sa_x_iter_start, sim_data->sa_x, mesh_data->sa_edges);
    mp_lbvh_edge->refit(stream);
    mp_lbvh_edge->broad_phase_query_from_edges(
        stream,
        sim_data->sa_x_iter_start,
        sim_data->sa_x,
        mesh_data->sa_edges,
        collision_data->broad_phase_collision_count.view(collision_data->get_ee_count_offset(), 1),
        collision_data->broad_phase_list_ee,
        ccd_query_range);
}
void NewtonSolver::device_broadphase_dcd(luisa::compute::Stream& stream)
{
    const float thickness       = get_scene_params().thickness;
    const float d_hat           = get_scene_params().d_hat;
    const float dcd_query_range = d_hat + thickness;

    mp_lbvh_face->update_face_tree_leave_aabb(
        stream, thickness, sim_data->sa_x, sim_data->sa_x, mesh_data->sa_faces);
    mp_lbvh_face->refit(stream);
    mp_lbvh_face->broad_phase_query_from_verts(
        stream,
        sim_data->sa_x,
        sim_data->sa_x,
        collision_data->broad_phase_collision_count.view(collision_data->get_vf_count_offset(), 1),
        collision_data->broad_phase_list_vf,
        dcd_query_range);

    mp_lbvh_edge->update_edge_tree_leave_aabb(
        stream, thickness, sim_data->sa_x, sim_data->sa_x, mesh_data->sa_edges);
    mp_lbvh_edge->refit(stream);
    mp_lbvh_edge->broad_phase_query_from_edges(
        stream,
        sim_data->sa_x,
        sim_data->sa_x,
        mesh_data->sa_edges,
        collision_data->broad_phase_collision_count.view(collision_data->get_ee_count_offset(), 1),
        collision_data->broad_phase_list_ee,
        dcd_query_range);
}
void NewtonSolver::device_narrowphase_ccd(luisa::compute::Stream& stream)
{
    const float thickness = get_scene_params().thickness;
    const float d_hat     = get_scene_params().d_hat;
    // mp_narrowphase_detector->reset_narrowphase_count(stream);
    mp_narrowphase_detector->reset_toi(stream);

    mp_narrowphase_detector->vf_ccd_query(stream,
                                          sim_data->sa_x_iter_start,
                                          sim_data->sa_x_iter_start,
                                          sim_data->sa_x,
                                          sim_data->sa_x,
                                          mesh_data->sa_faces,
                                          d_hat,
                                          thickness);

    mp_narrowphase_detector->ee_ccd_query(stream,
                                          sim_data->sa_x_iter_start,
                                          sim_data->sa_x_iter_start,
                                          sim_data->sa_x,
                                          sim_data->sa_x,
                                          mesh_data->sa_edges,
                                          mesh_data->sa_edges,
                                          d_hat,
                                          thickness);
}
void NewtonSolver::device_narrowphase_dcd(luisa::compute::Stream& stream)
{
    const float thickness = get_scene_params().thickness;
    const float d_hat     = get_scene_params().d_hat;
    const float kappa     = get_scene_params().stiffness_collision;

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
                                                    d_hat,
                                                    thickness,
                                                    kappa);

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
                                                    d_hat,
                                                    thickness,
                                                    kappa);
}
void NewtonSolver::device_update_contact_list(luisa::compute::Stream& stream)
{
    mp_narrowphase_detector->reset_broadphase_count(stream);
    mp_narrowphase_detector->reset_narrowphase_count(stream);
    mp_narrowphase_detector->reset_pervert_collision_count(stream);

    if (get_scene_params().use_self_collision)
        device_broadphase_dcd(stream);

    mp_narrowphase_detector->download_broadphase_collision_count(stream);

    if (get_scene_params().use_self_collision)
        device_narrowphase_dcd(stream);

    mp_narrowphase_detector->download_narrowphase_collision_count(stream);
    mp_narrowphase_detector->construct_pervert_adj_list(stream);
}
void NewtonSolver::device_ccd_line_search(luisa::compute::Stream& stream)
{
    device_broadphase_ccd(stream);

    mp_narrowphase_detector->download_broadphase_collision_count(stream);

    device_narrowphase_ccd(stream);
}
float NewtonSolver::device_compute_contact_energy(luisa::compute::Stream&               stream,
                                                  const luisa::compute::Buffer<float3>& curr_x)
{
    // stream << sim_data->sa_x.copy_from(sa_x.data());
    const float thickness = get_scene_params().thickness;
    const float d_hat     = get_scene_params().d_hat;
    const float kappa     = get_scene_params().stiffness_collision;

    mp_narrowphase_detector->reset_energy(stream);
    mp_narrowphase_detector->compute_contact_energy_from_iter_start_list(stream,
                                                                         curr_x,
                                                                         curr_x,
                                                                         mesh_data->sa_rest_x,
                                                                         mesh_data->sa_rest_x,
                                                                         mesh_data->sa_rest_vert_area,
                                                                         mesh_data->sa_rest_face_area,
                                                                         mesh_data->sa_faces,
                                                                         d_hat,
                                                                         thickness,
                                                                         kappa);

    return mp_narrowphase_detector->download_energy(stream);
    // return 0.0f;
}
void NewtonSolver::device_SpMV(luisa::compute::Stream&               stream,
                               const luisa::compute::Buffer<float3>& input_ptr,
                               luisa::compute::Buffer<float3>&       output_ptr)
{
    stream << fn_pcg_spmv_diag(input_ptr, output_ptr).dispatch(input_ptr.size());

    // stream << fn_pcg_spmv_offdiag_perVert(input_ptr, output_ptr).dispatch(host_sim_data->num_verts_soft);

    // stream << fn_pcg_spmv_offdiag_material_part_perTriplet(input_ptr, output_ptr)
    //               .dispatch(sim_data->sa_cgA_fixtopo_offdiag_triplet.size());

    // stream << fn_pcg_spmv_offdiag_warp_rbk(input_ptr, output_ptr)
    //               .dispatch(sim_data->sa_cgA_fixtopo_offdiag_triplet.size());

    stream << fn_pcg_spmv_offdiag_block_rbk(sim_data->sa_cgA_fixtopo_offdiag_triplet, input_ptr, output_ptr)
                  .dispatch(sim_data->sa_cgA_fixtopo_offdiag_triplet.size());

    // const uint num_pairs             = host_count.front();
    // const uint aligned_diaptch_count = get_dispatch_block(num_pairs * 12, 256) * 256;
    // stream << fn_pcg_spmv_offdiag_block_rbk(collision_data->sa_cgA_contact_offdiag_triplet, input_ptr, output_ptr)
    //               .dispatch(aligned_diaptch_count);

    const auto& host_count            = host_collision_data->narrow_phase_collision_count;
    const uint  reduced_triplet       = host_count[5];
    const uint  aligned_diaptch_count = get_dispatch_block(reduced_triplet, 256) * 256;
    stream << fn_pcg_spmv_offdiag_block_rbk(collision_data->sa_cgA_contact_offdiag_triplet, input_ptr, output_ptr)
                  .dispatch(aligned_diaptch_count);

    // mp_narrowphase_detector->device_perVert_spmv(stream, input_ptr, output_ptr);
    // mp_narrowphase_detector->device_perPair_spmv(stream, input_ptr, output_ptr);
}

void NewtonSolver::host_SpMV(luisa::compute::Stream&    stream,
                             const std::vector<float3>& input_ptr,
                             std::vector<float3>&       output_ptr)
{
    constexpr bool use_eigen          = ConjugateGradientSolver::use_eigen;
    constexpr bool use_upper_triangle = ConjugateGradientSolver::use_upper_triangle;

    // Diag
    CpuParallel::parallel_for(0,
                              input_ptr.size(),
                              [&](const uint vid)
                              {
                                  float3x3 A_diag      = host_sim_data->sa_cgA_diag[vid];
                                  float3   input_vec   = input_ptr[vid];
                                  float3   diag_output = A_diag * input_vec;
                                  output_ptr[vid]      = diag_output;
                              });
    // Off-Diag
    {
        if (false)
        {
            auto& sa_edges             = host_sim_data->sa_stretch_springs;
            auto& off_diag_hessian_ptr = host_sim_data->sa_stretch_springs_hessians;
            CpuParallel::single_thread_for(0,
                                           sa_edges.size(),
                                           [&](const uint eid)
                                           {
                                               const uint2 edge = sa_edges[eid];
                                               float3x3 offdiag_hessian1 = off_diag_hessian_ptr[4 * eid + 2];
                                               float3x3 offdiag_hessian2 = off_diag_hessian_ptr[4 * eid + 3];
                                               float3 output_vec0 = offdiag_hessian1 * input_ptr[edge[1]];
                                               float3 output_vec1 = offdiag_hessian2 * input_ptr[edge[0]];
                                               output_ptr[edge[0]] += output_vec0;
                                               output_ptr[edge[1]] += output_vec1;
                                           });
            return;
        }

        // Material Energy
        {
            CpuParallel::parallel_for(
                0,
                host_sim_data->num_verts_soft,
                [&](const uint vid)
                {
                    const uint curr_prefix = host_sim_data->sa_vert_adj_material_force_verts_csr[vid];
                    const uint next_prefix = host_sim_data->sa_vert_adj_material_force_verts_csr[vid + 1];
                    float3 output_vec = Zero3;
                    for (uint j = curr_prefix; j < next_prefix; j++)
                    {
                        const auto triplet = host_sim_data->sa_cgA_fixtopo_offdiag_triplet[j];
                        const uint adj_vid = triplet.get_col_idx();
                        output_vec += triplet.get_matrix() * input_ptr[adj_vid];
                    }
                    output_ptr[vid] += output_vec;
                });
        }

        // Affine body 12x12 block
        {
            const uint affine_body_dof_prefix = host_sim_data->num_verts_soft;

            auto& off_diag_hessian_ptr = host_sim_data->sa_cgA_offdiag_affine_body;
            CpuParallel::parallel_for(
                0,
                host_sim_data->sa_affine_bodies.size(),
                [&](const uint body_idx)
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
                            float3x3 hessian = off_diag_hessian_ptr[6 * body_idx + offset];
                            offset += 1;
                            output_vec[j] += (hessian)*input_vec[jj];
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
    mp_narrowphase_detector->host_perVert_spmv(stream, input_ptr, output_ptr);
}
void NewtonSolver::host_solve_eigen(luisa::compute::Stream& stream,
                                    std::function<double(const std::vector<float3>&)> func_compute_energy)
{

    EigenFloat12 eigen_b;
    eigen_b.setZero();
    EigenFloat12x12 eigen_A;
    eigen_A.setZero();
    for (uint vid = 0; vid < 4; vid++)
    {
        eigen_b.block<3, 1>(vid * 3, 0)       = float3_to_eigen3(host_sim_data->sa_cgB[vid]);
        eigen_A.block<3, 3>(vid * 3, vid * 3) = float3x3_to_eigen3x3(host_sim_data->sa_cgA_diag[vid]);
    }
    for (uint eid = 0; eid < 5; eid++)
    {
        float3   grad = host_sim_data->sa_stretch_springs_gradients[2 * eid + 0];
        float3x3 He   = host_sim_data->sa_stretch_springs_hessians[4 * eid + 0];
        uint2    edge = host_sim_data->colored_data.sa_merged_stretch_springs[eid];
        eigen_A.block<3, 3>(edge[0] * 3, edge[0] * 3) += float3x3_to_eigen3x3(He);
        eigen_A.block<3, 3>(edge[1] * 3, edge[1] * 3) += float3x3_to_eigen3x3(He);
        eigen_A.block<3, 3>(edge[0] * 3, edge[1] * 3) += float3x3_to_eigen3x3(-1.0f * He);
        eigen_A.block<3, 3>(edge[1] * 3, edge[0] * 3) += float3x3_to_eigen3x3(-1.0f * He);
        eigen_b.block<3, 1>(edge[0] * 3, 0) += float3_to_eigen3(grad);
        eigen_b.block<3, 1>(edge[1] * 3, 0) -= float3_to_eigen3(-grad);
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
    constexpr bool print_energy = false;
    double         curr_energy  = 0.0;
    if constexpr (print_energy)
    {
        host_apply_dx(1.0f);
        curr_energy = func_compute_energy(host_sim_data->sa_x);
    }
    const float infinity_norm = fast_infinity_norm(host_sim_data->sa_cgX);
    if (luisa::isnan(infinity_norm) || luisa::isinf(infinity_norm))
    {
        LUISA_ERROR("cgX exist NAN/INF value : {}", infinity_norm);
    }
    LUISA_INFO("  In non-linear iter {:2}, EigenSolve error = {:7.6f}, max_element(p) = {:6.5f}{}",
               get_scene_params().current_nonlinear_iter,
               (eigen_b - eigen_A * eigen_dx).norm(),
               infinity_norm,
               print_energy ? luisa::format(", energy = {:6.5f}", curr_energy) : "");
}
void NewtonSolver::host_line_search(luisa::compute::Stream& stream)
{
}


void NewtonSolver::host_apply_dx(const float alpha)
{
    if (alpha < 0.0f || alpha > 1.0f)
    {
        LUISA_ERROR("Alpha is not safe : {}", alpha);
    }

    // Update affine-body q
    float3* affine_body_cgX = &host_sim_data->sa_cgX[host_sim_data->num_verts_soft];
    CpuParallel::parallel_for(0,
                              host_sim_data->sa_affine_bodies.size() * 4,
                              [&](const uint block_idx)
                              {
                                  host_sim_data->sa_affine_bodies_q[block_idx] =
                                      host_sim_data->sa_affine_bodies_q_iter_start[block_idx]
                                      + alpha * affine_body_cgX[block_idx];
                                  // LUISA_INFO("Rigid body {}: q_{} = {}", block_idx / 4, block_idx % 4, host_sim_data->sa_affine_bodies_q[block_idx]);
                              });

    // Update sa_x
    CpuParallel::parallel_for(
        0,
        host_mesh_data->num_verts,
        [&](const uint vid)
        {
            const bool is_rigid_body = host_mesh_data->sa_vert_mesh_type[vid] == Initializer::ShellTypeRigid;
            if (is_rigid_body)
            {
                const uint body_idx = host_sim_data->sa_vert_affine_bodies_id[vid];
                float3     p;
                float3x3   A;
                AffineBodyDynamics::extract_Ap_from_q(&host_sim_data->sa_affine_bodies_q[4 * body_idx], A, p);
                const float3 rest_x      = host_mesh_data->sa_scaled_model_x[vid];
                const float3 affine_x    = A * rest_x + p;  // Affine position
                host_sim_data->sa_x[vid] = affine_x;
                // LUISA_INFO("Rigid Body {}'s Vert {} apply transform, from {} to {}", body_idx, vid, rest_x, affine_x);
            }
            else
            {
                host_sim_data->sa_x[vid] =
                    host_sim_data->sa_x_iter_start[vid] + alpha * host_sim_data->sa_cgX[vid];
            }
        });
}

void NewtonSolver::physics_step_CPU(luisa::compute::Device& device, luisa::compute::Stream& stream)
{
    // Input
    {
        lcs::SolverInterface::physics_step_prev_operation();
        CpuParallel::parallel_for(0,
                                  host_sim_data->sa_x.size(),
                                  [&](const uint vid)
                                  {
                                      host_sim_data->sa_x_step_start[vid] = host_mesh_data->sa_x_frame_outer[vid];
                                      host_sim_data->sa_v_step_start[vid] = host_mesh_data->sa_v_frame_outer[vid];
                                      host_sim_data->sa_x[vid] = host_mesh_data->sa_x_frame_outer[vid];
                                      host_sim_data->sa_v[vid] = host_mesh_data->sa_v_frame_outer[vid];
                                  });
        CpuParallel::parallel_for(0,
                                  host_sim_data->sa_affine_bodies_q.size(),
                                  [&](const uint vid)
                                  {
                                      host_sim_data->sa_affine_bodies_q_step_start[vid] =
                                          host_sim_data->sa_affine_bodies_q_outer[vid];
                                      host_sim_data->sa_affine_bodies_q[vid] =
                                          host_sim_data->sa_affine_bodies_q_outer[vid];
                                      host_sim_data->sa_affine_bodies_q_v[vid] =
                                          host_sim_data->sa_affine_bodies_q_v_outer[vid];
                                  });
    }

    constexpr bool use_eigen          = ConjugateGradientSolver::use_eigen;
    constexpr bool use_upper_triangle = ConjugateGradientSolver::use_upper_triangle;

    auto pcg_spmv = [&](const std::vector<float3>& input_ptr, std::vector<float3>& output_ptr) -> void
    { host_SpMV(stream, input_ptr, output_ptr); };

    const float thickness = get_scene_params().thickness;
    const float d_hat     = get_scene_params().d_hat;
    const float kappa     = 1e5;

    auto update_contact_set = [&]()
    {
        stream
            // << sim_data->sa_x_iter_start.copy_from(host_sim_data->sa_x_iter_start.data())
            << sim_data->sa_x.copy_from(host_sim_data->sa_x.data());

        device_update_contact_list(stream);
        mp_narrowphase_detector->download_narrowphase_list(stream);
        mp_narrowphase_detector->download_pervert_adjacent_list(stream);
    };
    auto evaluate_contact = [&]()
    {
        stream << sim_data->sa_cgB.copy_from(host_sim_data->sa_cgB.data())
               << sim_data->sa_cgA_diag.copy_from(host_sim_data->sa_cgA_diag.data());

        mp_narrowphase_detector->device_perVert_evaluate_gradient_hessian(
            stream, sim_data->sa_x, sim_data->sa_x, d_hat, thickness, sim_data->sa_cgB, sim_data->sa_cgA_diag);

        stream << sim_data->sa_cgB.copy_to(host_sim_data->sa_cgB.data())
               << sim_data->sa_cgA_diag.copy_to(host_sim_data->sa_cgA_diag.data())
               << luisa::compute::synchronize();
    };
    auto ccd_get_toi = [&]() -> float
    {
        stream << sim_data->sa_x_iter_start.copy_from(host_sim_data->sa_x_iter_start.data())
               << sim_data->sa_x.copy_from(host_sim_data->sa_x.data())
            // << luisa::compute::synchronize()
            ;

        device_ccd_line_search(stream);

        float toi = mp_narrowphase_detector->get_global_toi(stream);
        return toi;  // 0.9f * toi
        // return 1.0f;
    };

    auto compute_energy_interface = [&](const std::vector<float3>& curr_x)
    {
        stream << sim_data->sa_x_tilde.copy_from(host_sim_data->sa_x_tilde.data());
        stream << sim_data->sa_x.copy_from(curr_x.data());
        // auto material_energy = host_compute_elastic_energy(curr_x);
        auto material_energy = device_compute_elastic_energy(stream, sim_data->sa_x);
        auto barrier_energy  = device_compute_contact_energy(stream, sim_data->sa_x);
        // LUISA_INFO(".       Energy = {} + {}", material_energy, barrier_energy);
        auto total_energy = material_energy + barrier_energy;
        if (is_nan_scalar(material_energy) || is_inf_scalar(material_energy))
        {
            LUISA_ERROR("Material energy is not valid : {}", material_energy);
        }
        if (is_nan_scalar(barrier_energy) || is_inf_scalar(barrier_energy))
        {
            LUISA_ERROR("Barrier energy is not valid : {}", material_energy);
        }
        return total_energy;
    };
    auto linear_solver_interface = [&]()
    {
        if constexpr (false)
        {
            host_solve_eigen(stream, compute_energy_interface);
        }
        else
        {
            // simple_solve();
            pcg_solver->host_solve(stream, pcg_spmv, compute_energy_interface);
        }
    };

    const float substep_dt            = lcs::get_scene_params().get_substep_dt();
    const bool  use_energy_linesearch = get_scene_params().use_energy_linesearch;
    const bool  use_ccd_linesearch    = get_scene_params().use_ccd_linesearch;

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
        LUISA_INFO("=== In frame {} ===", get_scene_params().current_frame);

        host_predict_position();

        // double barrier_nergy = compute_barrier_energy_from_broadphase_list();
        double prev_state_energy = Float_max;

        for (uint iter = 0; iter < get_scene_params().nonlinear_iter_count; iter++)
        {
            get_scene_params().current_nonlinear_iter = iter;

            host_reset_cgB_cgX_diagA();

            host_reset_off_diag();

            // if constexpr (false)
            {
                host_evaluate_inertia();

                host_evaluate_ground_collision();

                host_evaluate_orthogonality();

                host_evaluete_spring();

                host_evaluete_bending();

                host_material_energy_assembly();

                update_contact_set();

                evaluate_contact();

                // host_evaluate_dirichlet();

                // if (iter == 0) // Always refresh for collision count is variant
                if (use_energy_linesearch)
                {
                    prev_state_energy = compute_energy_interface(host_sim_data->sa_x);
                }
            }

            // host_test_affine_body(stream);

            linear_solver_interface();  // Solve Ax=b

            float alpha   = 1.0f;
            float ccd_toi = 1.0f;
            host_apply_dx(alpha);

            if (use_ccd_linesearch)
            {
                ccd_toi = ccd_get_toi();
                alpha   = ccd_toi;
                host_apply_dx(alpha);
            }

            // Non-linear iteration break condition
            {
                float max_move      = 1e-2;
                float curr_max_step = fast_infinity_norm(host_sim_data->sa_cgX);
                if (curr_max_step < max_move * substep_dt)
                {
                    LUISA_INFO("  In non-linear iter {:2}: Iteration break for small searching direction {} < {}",
                               iter,
                               curr_max_step,
                               max_move * substep_dt);
                    break;
                }
            }  // That means: If the step is too small, then we dont need energy line-search (energy may not be descent in small step)

            if (use_energy_linesearch)
            {
                // Energy after CCD or just solving Axb
                auto curr_energy = compute_energy_interface(host_sim_data->sa_x);
                if (is_nan_scalar(curr_energy) || is_inf_scalar(curr_energy))
                {
                    LUISA_ERROR("Energy is not valid : {}", curr_energy);
                }

                uint line_search_count = 0;
                while (line_search_count < 20)  // Compare energy
                {
                    if (curr_energy < prev_state_energy + Epsilon)
                    {
                        if (alpha != 1.0f)
                        {
                            LUISA_INFO("     Line search {} break : alpha = {:6.5f}, curr energy = {:12.10f} , prev energy {:12.10f} , {}",
                                       line_search_count,
                                       alpha,
                                       curr_energy,
                                       prev_state_energy,
                                       ccd_toi != 1.0f ? "CCD toi = " + std::to_string(ccd_toi) : "");
                        }
                        break;
                    }
                    if (line_search_count == 0)
                    {
                        LUISA_INFO("     Line search {} : alpha = {:6.5f}, energy = {:12.10f} , prev state energy {:12.10f} {}",
                                   line_search_count,
                                   alpha,
                                   curr_energy,
                                   prev_state_energy,
                                   ccd_toi != 1.0f ? ", CCD toi = " + std::to_string(ccd_toi) : "");
                    }
                    alpha /= 2;
                    host_apply_dx(alpha);

                    curr_energy = compute_energy_interface(host_sim_data->sa_x);
                    LUISA_INFO("     Line search {} : alpha = {:6.5f}, energy = {:12.10f}", line_search_count, alpha, curr_energy);

                    if (alpha < 1e-4)
                    {
                        LUISA_ERROR("  Line search failed, energy = {}, prev state energy = {}", curr_energy, prev_state_energy);
                    }
                    line_search_count++;
                }
                prev_state_energy = curr_energy;  // E_prev = E
            }

            // CpuParallel::parallel_copy(host_sim_data->sa_x.data(), host_sim_data->sa_x_iter_start.data(), host_sim_data->num_verts_soft); // x_prev = x
            CpuParallel::parallel_copy(host_sim_data->sa_x, host_sim_data->sa_x_iter_start);  // x_prev = x
            CpuParallel::parallel_copy(host_sim_data->sa_affine_bodies_q,
                                       host_sim_data->sa_affine_bodies_q_iter_start);  // q_prev = q
        }
        host_update_velocity();
    }

    // Output
    {
        CpuParallel::parallel_for(0,
                                  host_sim_data->sa_x.size(),
                                  [&](const uint vid)
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
    constexpr bool profile_time = false;
    using SystemClock           = std::chrono::high_resolution_clock;
    using Tick                  = std::chrono::high_resolution_clock::time_point;
    std::vector<std::pair<std::string, Tick>> time_stamps;

    auto ADD_DEVICE_TIME_STAMP = [&](const std::string& task_name)
    {
        if constexpr (profile_time)
            stream << [&] { time_stamps.emplace_back(std::make_pair(task_name, SystemClock::now())); };
    };
    auto ADD_HOST_TIME_STAMP = [&](const std::string& task_name)
    {
        if constexpr (profile_time)
            time_stamps.emplace_back(std::make_pair(task_name, SystemClock::now()));
    };

    ADD_HOST_TIME_STAMP("Init");
    lcs::SolverInterface::physics_step_prev_operation();
    // Get frame start position and velocity
    CpuParallel::parallel_for(0,
                              host_sim_data->sa_x.size(),
                              [&](const uint vid)
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
    const uint  num_substep          = lcs::get_scene_params().num_substep;
    const uint  nonlinear_iter_count = lcs::get_scene_params().nonlinear_iter_count;
    const float substep_dt           = lcs::get_scene_params().get_substep_dt();

    auto device_apply_dx = [&](const float alpha)
    { stream << fn_apply_dx(alpha).dispatch(sim_data->sa_cgX.size()); };

    const float thickness          = get_scene_params().thickness;
    const float d_hat              = get_scene_params().d_hat;
    auto        update_contact_set = [&]() { device_update_contact_list(stream); };
    auto        evaluate_contact   = [&]()
    {
        // mp_narrowphase_detector->device_perPair_evaluate_gradient_hessian(
        mp_narrowphase_detector->device_perVert_evaluate_gradient_hessian(
            stream, sim_data->sa_x, sim_data->sa_x, d_hat, thickness, sim_data->sa_cgB, sim_data->sa_cgA_diag);
    };
    auto ccd_get_toi = [&]() -> float
    {
        device_ccd_line_search(stream);
        float toi = mp_narrowphase_detector->get_global_toi(stream);
        return toi;  // 0.9f * toi
    };

    const bool use_ipc   = true;
    const uint num_verts = host_mesh_data->num_verts;

    auto pcg_spmv = [&](const luisa::compute::Buffer<float3>& input_ptr, luisa::compute::Buffer<float3>& output_ptr) -> void
    { device_SpMV(stream, input_ptr, output_ptr); };
    auto compute_energy_interface = [&](const luisa::compute::Buffer<float3>& curr_x)
    {
        // stream << sim_data->sa_x_tilde.copy_to(host_sim_data->sa_x_tilde.data());
        // auto material_energy = host_compute_elastic_energy(host_sim_data->sa_x);

        // stream << sim_data->sa_x.copy_to(host_sim_data->sa_x.data());
        // stream << luisa::compute::synchronize();

        auto material_energy = device_compute_elastic_energy(stream, curr_x);
        auto barrier_energy  = device_compute_contact_energy(stream, curr_x);
        ;
        // LUISA_INFO(".       Energy = {} + {}", material_energy, barrier_energy);
        auto total_energy = material_energy + barrier_energy;
        if (is_nan_scalar(material_energy) || is_inf_scalar(material_energy))
        {
            LUISA_ERROR("Material energy is not valid : {}", material_energy);
        }
        if (is_nan_scalar(barrier_energy) || is_inf_scalar(barrier_energy))
        {
            LUISA_ERROR("Barrier energy is not valid : {}", material_energy);
        }

        return total_energy;
    };
    // Init LBVH
    {
        ADD_HOST_TIME_STAMP("Init LBVH");
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
        stream << fn_predict_position(substep_dt, get_scene_params().gravity).dispatch(num_verts)
               // << sim_data->sa_x_step_start.copy_to(host_x_step_start.data())
               << sim_data->sa_x_tilde.copy_to(host_sim_data->sa_x_tilde.data())  // For calculate inertia energy
               << luisa::compute::synchronize();

        double prev_state_energy = Float_max;

        LUISA_INFO("=== In frame {} ===", get_scene_params().current_frame);

        for (uint iter = 0; iter < get_scene_params().nonlinear_iter_count; iter++)
        {
            ADD_HOST_TIME_STAMP("Calc Force");
            get_scene_params().current_nonlinear_iter = iter;

            stream << sim_data->sa_x_iter_start.copy_to(host_sim_data->sa_x_iter_start.data())
                   << fn_reset_vector(sim_data->sa_cgX).dispatch(sim_data->sa_cgX.size())
                   << fn_reset_vector(sim_data->sa_cgB).dispatch(sim_data->sa_cgB.size())
                   << fn_reset_float3x3(sim_data->sa_cgA_diag).dispatch(sim_data->sa_cgA_diag.size())
                   << fn_reset_cgA_offdiag_triplet().dispatch(sim_data->sa_cgA_fixtopo_offdiag_triplet.size())
                // << fn_reset_float3x3(sim_data->sa_cgA_offdiag_stretch_spring).dispatch(sim_data->sa_cgA_offdiag_stretch_spring.size())
                // << fn_reset_float3x3(sim_data->sa_cgA_offdiag_bending).dispatch(sim_data->sa_cgA_offdiag_bending.size())
                ;

            {
                stream << fn_evaluate_inertia(substep_dt, get_scene_params().stiffness_dirichlet).dispatch(num_verts);

                stream << fn_evaluate_ground_collision(get_scene_params().floor.y,
                                                       get_scene_params().use_floor,
                                                       1e7f,
                                                       get_scene_params().d_hat,
                                                       get_scene_params().thickness)
                              .dispatch(num_verts);

                stream << fn_evaluate_spring(get_scene_params().stiffness_spring)
                              .dispatch(host_sim_data->sa_stretch_springs.size());

                stream << fn_evaluate_bending(get_scene_params().get_stiffness_quadratic_bending())
                              .dispatch(host_sim_data->sa_bending_edges.size());

                stream << fn_material_energy_assembly().dispatch(host_sim_data->num_verts_soft);

                update_contact_set();

                evaluate_contact();

                // stream << fn_evaluate_dirichlet(substep_dt, get_scene_params().stiffness_dirichlet).dispatch(num_verts);

                if (get_scene_params().use_energy_linesearch)
                    prev_state_energy = compute_energy_interface(sim_data->sa_x);
            }

            stream << luisa::compute::synchronize();
            ADD_HOST_TIME_STAMP("PCG");
            pcg_solver->device_solve(stream, pcg_spmv, compute_energy_interface);

            float alpha   = 1.0f;
            float ccd_toi = 1.0f;
            host_apply_dx(alpha);
            device_apply_dx(alpha);

            ADD_HOST_TIME_STAMP("CCD");
            if (get_scene_params().use_ccd_linesearch)
            {
                ccd_toi = ccd_get_toi();
                alpha   = ccd_toi;
                host_apply_dx(alpha);
                device_apply_dx(alpha);
            }
            ADD_HOST_TIME_STAMP("End CCD");

            // Non-linear iteration break condition
            {
                float max_move      = 1e-2;
                float curr_max_step = fast_infinity_norm(host_sim_data->sa_cgX);
                if (curr_max_step < max_move * substep_dt)
                {
                    LUISA_INFO("  In non-linear iter {:2}: Iteration break for small searching direction {} < {}",
                               iter,
                               curr_max_step,
                               max_move * substep_dt);
                    break;
                }
            }  // That means: If the step is too small, then we dont need energy line-search (energy may not be descent in small step)

            if (get_scene_params().use_energy_linesearch)
            {
                // Energy after CCD or just solving Axb
                auto curr_energy = compute_energy_interface(sim_data->sa_x);
                if (is_nan_scalar(curr_energy) || is_inf_scalar(curr_energy))
                {
                    LUISA_ERROR("Energy is not valid : {}", curr_energy);
                }

                uint line_search_count = 0;
                while (line_search_count < 20)  // Compare energy
                {
                    if (curr_energy < prev_state_energy + Epsilon)
                    {
                        if (alpha != 1.0f)
                        {
                            LUISA_INFO("     Line search {} break : alpha = {:6.5f}, curr energy = {:12.10f} , prev energy {:12.10f} , {}",
                                       line_search_count,
                                       alpha,
                                       curr_energy,
                                       prev_state_energy,
                                       ccd_toi != 1.0f ? "CCD toi = " + std::to_string(ccd_toi) : "");
                        }
                        break;
                    }
                    if (line_search_count == 0)
                    {
                        LUISA_INFO("     Line search {} : alpha = {:6.5f}, energy = {:12.10f} , prev state energy {:12.10f} {}",
                                   line_search_count,
                                   alpha,
                                   curr_energy,
                                   prev_state_energy,
                                   ccd_toi != 1.0f ? ", CCD toi = " + std::to_string(ccd_toi) : "");
                    }
                    alpha /= 2;
                    host_apply_dx(alpha);
                    device_apply_dx(alpha);

                    curr_energy = compute_energy_interface(sim_data->sa_x);
                    LUISA_INFO("     Line search {} : alpha = {:6.5f}, energy = {:12.10f}", line_search_count, alpha, curr_energy);

                    if (alpha < 1e-4)
                    {
                        LUISA_ERROR("  Line search failed, energy = {}, prev state energy = {}", curr_energy, prev_state_energy);
                    }
                    line_search_count++;
                }
                prev_state_energy = curr_energy;  // E_prev = E
            }

            stream << sim_data->sa_x.copy_from(host_sim_data->sa_x.data())
                   << sim_data->sa_x_iter_start.copy_from(host_sim_data->sa_x_iter_start.data());

            stream << sim_data->sa_x.copy_to(sim_data->sa_x_iter_start) << luisa::compute::synchronize();
        }
        stream << fn_update_velocity(substep_dt, get_scene_params().fix_scene, get_scene_params().damping_cloth)
                      .dispatch(num_verts);
    }

    stream << luisa::compute::synchronize();

    // Copy to host
    {
        stream << sim_data->sa_x.copy_to(host_sim_data->sa_x.data())
               << sim_data->sa_v.copy_to(host_sim_data->sa_v.data()) << luisa::compute::synchronize();
    }

    // Return frame end position and velocity
    CpuParallel::parallel_for(0,
                              host_sim_data->sa_x.size(),
                              [&](const uint vid)
                              {
                                  host_mesh_data->sa_x_frame_outer[vid] = host_sim_data->sa_x[vid];
                                  host_mesh_data->sa_v_frame_outer[vid] = host_sim_data->sa_v[vid];
                              });
    lcs::SolverInterface::physics_step_post_operation();

    {
        if constexpr (profile_time)
        {
            if (!time_stamps.empty())
            {
                // Aggregate durations (ms) per task name
                std::unordered_map<std::string, double> agg;
                double                                  total_ms = 0.0;
                for (size_t i = 0; i + 1 < time_stamps.size(); ++i)
                {
                    const auto& curr = time_stamps[i];
                    const auto& next = time_stamps[i + 1];
                    double delta = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                                       next.second - curr.second)
                                       .count();
                    agg[curr.first] += delta;
                    total_ms += delta;
                }

                LUISA_INFO("Profiling merged timestamps (sum of deltas per task):");
                for (const auto& p : agg)
                {
                    LUISA_INFO("  {:<30} : {:8.3f} ms", p.first, p.second);
                }
                LUISA_INFO("  {:<30} : {:8.3f} ms (total)", "TOTAL", total_ms);
            }
        }
    }
}

}  // namespace lcs