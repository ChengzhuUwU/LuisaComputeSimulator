#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include "CollisionDetector/libuipc/codim_ipc_simplex_normal_contact_function.h"
#include "Core/affine_position.h"
#include "Core/float_n.h"
#include "Energy/bending_energy.h"
#include "Energy/stretch_energy.h"
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
void buffer_add(const Var<luisa::compute::BufferView<T>>& buffer, const Var<uint> dest, const Var<T>& value)
{
    buffer->write(dest, buffer->read(dest) + value);
}
template <typename T>
void buffer_add(const luisa::compute::BufferView<T>& buffer, const Var<uint> dest, const Var<T>& value)
{
    buffer->write(dest, buffer->read(dest) + value);
}
template <typename T>
void buffer_add(std::vector<T>& buffer, const uint dest, const T& value)
{
    buffer[dest] = buffer[dest] + value;
}
template <typename T>
void buffer_add(T* buffer, const uint dest, const T& value)
{
    buffer[dest] = buffer[dest] + value;
}

void atomic_buffer_add(const Var<luisa::compute::BufferView<float3>>& buffer, const Var<uint> dest, const Var<float3>& value)
{
    buffer->atomic(dest)[0].fetch_add(value[0]);
    buffer->atomic(dest)[1].fetch_add(value[1]);
    buffer->atomic(dest)[2].fetch_add(value[2]);
}
void atomic_buffer_add(const Var<luisa::compute::BufferView<float3x3>>& buffer,
                       const Var<uint>                                  dest,
                       const Var<float3x3>&                             value)
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

void atomic_buffer_add(const luisa::compute::BufferView<float3>& buffer, const Var<uint> dest, const Var<float3>& value)
{
    buffer->atomic(dest)[0].fetch_add(value[0]);
    buffer->atomic(dest)[1].fetch_add(value[1]);
    buffer->atomic(dest)[2].fetch_add(value[2]);
}
void atomic_buffer_add(const luisa::compute::BufferView<float3x3>& buffer, const Var<uint> dest, const Var<float3x3>& value)
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

template <uint N>
struct EigenTripletBlock
{
    std::array<uint, N>                indices;
    Eigen::Matrix<float, 3 * N, 3 * N> matrix;
};

struct VarientEigenTripletBlock
{
    std::vector<uint> left_indices;
    std::vector<uint> right_indices;
    Eigen::MatrixXf   matrix;
};

static void convert_triplets_to_sparse_matrix(Eigen::SparseMatrix<float>&          matrix,
                                              const std::vector<MatrixTriplet3x3>& triplets)
{
    std::vector<Eigen::Triplet<float>> eigen_triplets;
    eigen_triplets.reserve(triplets.size() * 9);
    for (const auto& triplet : triplets)
    {
        if (MatrixTriplet::is_valid(triplet.get_matrix_property()))
        {
            auto row  = triplet.get_row_idx();
            auto col  = triplet.get_col_idx();
            auto hess = triplet.get_matrix();
            for (uint i = 0; i < 3; i++)
            {
                for (uint j = 0; j < 3; j++)
                {
                    eigen_triplets.emplace_back(Eigen::Triplet<float>(row * 3 + i, col * 3 + j, hess[i][j]));
                }
            }
        }
    }
    matrix.setFromTriplets(eigen_triplets.begin(), eigen_triplets.end());
}

template <uint N>
static void convert_triplets_to_sparse_matrix(Eigen::SparseMatrix<float>&              matrix,
                                              const std::vector<EigenTripletBlock<N>>& hessians)
{
    Eigen::SparseMatrix<float> local_matrix(matrix.rows(), matrix.cols());
    local_matrix.setZero();
    std::vector<Eigen::Triplet<float>> local_triplets;
    local_triplets.reserve(hessians.size() * N * N * 9);
    for (size_t t_idx = 0; t_idx < hessians.size(); t_idx++)
    {
        auto        triplet = hessians[t_idx];
        const auto& hess    = triplet.matrix;
        const auto& idxs    = triplet.indices;
        for (uint ii = 0; ii < N; ii++)
        {
            const uint vid = idxs[ii];
            for (uint jj = 0; jj < N; jj++)
            {
                const uint adj_vid = idxs[jj];
                for (uint i = 0; i < 3; i++)
                {
                    for (uint j = 0; j < 3; j++)
                    {
                        local_triplets.emplace_back(
                            Eigen::Triplet<float>(vid * 3 + i, adj_vid * 3 + j, hess(ii * 3 + i, jj * 3 + j)));
                    }
                }
            }
        }
    }
    local_matrix.setFromTriplets(local_triplets.begin(), local_triplets.end());
    matrix += local_matrix;
}

static void convert_triplets_to_sparse_matrix(Eigen::SparseMatrix<float>&                  matrix,
                                              const std::vector<VarientEigenTripletBlock>& hessians)
{
    Eigen::SparseMatrix<float> local_matrix(matrix.rows(), matrix.cols());
    local_matrix.setZero();
    std::vector<Eigen::Triplet<float>> local_triplets;
    for (size_t t_idx = 0; t_idx < hessians.size(); t_idx++)
    {
        auto        triplet    = hessians[t_idx];
        const auto& hess       = triplet.matrix;
        const auto& left_idxs  = triplet.left_indices;
        const auto& right_idxs = triplet.right_indices;

        for (uint ii = 0; ii < left_idxs.size(); ii++)
        {
            const uint vid = left_idxs[ii];
            for (uint jj = 0; jj < right_idxs.size(); jj++)
            {
                const uint adj_vid = right_idxs[jj];
                for (uint i = 0; i < 3; i++)
                {
                    for (uint j = 0; j < 3; j++)
                    {
                        local_triplets.emplace_back(
                            Eigen::Triplet<float>(vid * 3 + i, adj_vid * 3 + j, hess(ii * 3 + i, jj * 3 + j)));
                    }
                }
            }
        }
    }
    local_matrix.setFromTriplets(local_triplets.begin(), local_triplets.end());
    matrix += local_matrix;
}

void NewtonSolver::compile(AsyncCompiler& compiler)
{
    const bool use_debug_info = false;
    using namespace luisa::compute;

    luisa::compute::ShaderOption default_option = {.enable_debug_info = false};

    compile_advancing(compiler, default_option);

    compile_assembly(compiler, default_option);

    compile_evaluate(compiler, default_option);


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
            const Uint vid = dispatch_id().x;
            Float3     new_x;

            new_x = sa_x_iter_start->read(vid) + alpha * sa_cgX->read(vid);
            sa_x->write(vid, sa_x_iter_start->read(vid) + alpha * sa_cgX->read(vid));

            sa_x->write(vid, new_x);
        },
        default_option);

    if (host_sim_data->sa_affine_bodies_q.size() != 0)
        compiler.compile<1>(
            fn_apply_dx_affine_bodies,
            [sa_x               = sim_data->sa_x.view(),
             sa_scaled_model_x  = mesh_data->sa_scaled_model_x.view(),
             sa_x_iter_start    = sim_data->sa_x_iter_start.view(),
             sa_cgX             = sim_data->sa_cgX.view(),
             sa_affine_bodies_q = sim_data->sa_affine_bodies_q.view(),
             sa_vert_affine_bodies_id = sim_data->sa_vert_affine_bodies_id.view()](const Float alpha, const Uint prefix)
            {
                const Uint vid      = prefix + dispatch_id().x;
                const Uint body_idx = sa_vert_affine_bodies_id->read(vid);
                Float3     new_x;
                Float3     p;
                Float3x3   A;
                AffineBodyDynamics::extract_Ap_from_q(sa_affine_bodies_q, body_idx, A, p);
                const Float3 rest_x = sa_scaled_model_x->read(vid);
                new_x               = A * rest_x + p;  // Affine position

                sa_x->write(vid, new_x);
            },
            default_option);

    if (host_sim_data->sa_affine_bodies_q.size() != 0)
        compiler.compile<1>(
            fn_apply_dq,
            [sa_q            = sim_data->sa_affine_bodies_q.view(),
             sa_q_iter_start = sim_data->sa_affine_bodies_q_iter_start.view(),
             sa_cgX          = sim_data->sa_cgX.view()](const Float alpha, const Uint prefix)
            {
                const UInt vid = dispatch_id().x;
                sa_q->write(vid, sa_q_iter_start->read(vid) + alpha * sa_cgX->read(prefix + vid));
            },
            default_option);

    compiler.compile<1>(
        fn_apply_dx_non_constant,
        [sa_x            = sim_data->sa_x.view(),
         sa_cgX          = sim_data->sa_cgX.view(),
         sa_x_iter_start = sim_data->sa_x_iter_start.view()](Var<BufferView<float>> alpha_buffer)
        {
            const Float alpha = alpha_buffer.read(0);
            const Uint  vid   = dispatch_id().x;
            Float3      new_x;

            new_x = sa_x_iter_start->read(vid) + alpha * sa_cgX->read(vid);
            sa_x->write(vid, sa_x_iter_start->read(vid) + alpha * sa_cgX->read(vid));

            sa_x->write(vid, new_x);
        },
        default_option);

    if (host_sim_data->sa_affine_bodies_q.size() != 0)
        compiler.compile<1>(
            fn_apply_dq_non_constant,
            [sa_q            = sim_data->sa_affine_bodies_q.view(),
             sa_q_iter_start = sim_data->sa_affine_bodies_q_iter_start.view(),
             sa_cgX = sim_data->sa_cgX.view()](Var<BufferView<float>> alpha_buffer, const Uint prefix)
            {
                const Float alpha = alpha_buffer.read(0);
                const UInt  vid   = dispatch_id().x;
                sa_q->write(vid, sa_q_iter_start->read(vid) + alpha * sa_cgX->read(prefix + vid));
            },
            default_option);
}

void NewtonSolver::compile_advancing(AsyncCompiler& compiler, const luisa::compute::ShaderOption& default_option)
{
    using namespace luisa::compute;

    if (host_sim_data->num_verts_soft != 0)
    {
        compiler.compile<1>(
            fn_predict_position,
            [sa_x            = sim_data->sa_x.view(),
             sa_x_step_start = sim_data->sa_x_step_start.view(),
             sa_x_iter_start = sim_data->sa_x_iter_start.view(),
             sa_x_tilde      = sim_data->sa_x_tilde.view(),
             sa_v            = sim_data->sa_v.view(),
             sa_cgX          = sim_data->sa_cgX.view(),
             sa_is_fixed = mesh_data->sa_is_fixed.view()](const Float substep_dt, const Float3 gravity)
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
            [sa_x = sim_data->sa_x.view(),
             sa_v = sim_data->sa_v.view(),
             sa_x_step_start = sim_data->sa_x_step_start.view()](const Float substep_dt, const Bool fix_scene, const Float damping)
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
                sa_x_step_start->write(vid, x_step_end);
            },
            default_option);
    }

    if (host_sim_data->num_affine_bodies != 0)
    {
        compiler.compile<1>(
            fn_abd_predict_position,
            [sa_x            = sim_data->sa_affine_bodies_q.view(),
             sa_x_step_start = sim_data->sa_affine_bodies_q_step_start.view(),
             sa_x_iter_start = sim_data->sa_affine_bodies_q_iter_start.view(),
             sa_x_tilde      = sim_data->sa_affine_bodies_q_tilde.view(),
             sa_v            = sim_data->sa_affine_bodies_q_v.view(),
             sa_cgX          = sim_data->sa_cgX.view(),
             sa_is_fixed = sim_data->sa_affine_bodies_is_fixed.view()](const Float substep_dt, const Float3 gravity)
            {
                const UInt vid    = dispatch_id().x;
                Float3     x_prev = sa_x_step_start->read(vid);
                Float3     v_prev = sa_v->read(vid);
                Float3     v_pred = v_prev;

                Float3 outer_acceleration = gravity;
                $if(vid % 4 == 0)
                {
                    v_pred += substep_dt * outer_acceleration;
                };

                const Uint body_idx = vid / 4;
                const Bool is_fixed = sa_is_fixed->read(body_idx);
                $if(is_fixed)
                {
                    v_pred = v_prev;
                };

                sa_x_iter_start->write(vid, x_prev);
                Float3 x_pred = x_prev + substep_dt * v_pred;
                sa_x_tilde->write(vid, x_pred);
                sa_x->write(vid, x_prev);
            },
            default_option);

        compiler.compile<1>(
            fn_abd_update_velocity,
            [sa_x            = sim_data->sa_affine_bodies_q.view(),
             sa_x_step_start = sim_data->sa_affine_bodies_q_step_start.view(),
             sa_v = sim_data->sa_affine_bodies_q_v.view()](const Float substep_dt, const Bool fix_scene, const Float damping)
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
                sa_x_step_start->write(vid, x_step_end);
            },
            default_option);
    }
}
void NewtonSolver::compile_assembly(AsyncCompiler& compiler, const luisa::compute::ShaderOption& default_option)
{
    using namespace luisa::compute;

    auto assembly_template =
        [sa_vert_adj_material_force_verts_csr = sim_data->sa_vert_adj_material_force_verts_csr.view(),
         sa_cgB                               = sim_data->sa_cgB.view(),
         sa_cgA_diag                          = sim_data->sa_cgA_diag.view(),
         sa_cgA_offdiag_triplet               = sim_data->sa_cgA_fixtopo_offdiag_triplet.view()](
            const uint                  N,
            const Uint                  vid,
            const BufferView<uint>&     vert_adj_constraints_csr,
            const auto&                 constaints,
            const BufferView<float3>&   constaint_gradients,
            const BufferView<float3x3>& constaint_hessians,
            const BufferView<ushort>&   constaint_offsets_in_adjlist)
    {
        const Uint curr_prefix = sa_vert_adj_material_force_verts_csr->read(vid);
        const Uint next_prefix = sa_vert_adj_material_force_verts_csr->read(vid + 1);

        const Uint curr_prefix_bending = vert_adj_constraints_csr->read(vid);
        const Uint next_prefix_bending = vert_adj_constraints_csr->read(vid + 1);

        $for(j, curr_prefix_bending, next_prefix_bending)
        {
            const Uint adj_eid = vert_adj_constraints_csr->read(j);
            const auto edge    = constaints->read(adj_eid);
            Uint       offset  = -1u;
            for (uint k = 0; k < N; k++)
            {
                $if(vid == edge[k])
                {
                    offset = k;
                };
            }
            device_assert(offset != -1u, "Error in assembly: offset not found.");

            const Float3   grad      = constaint_gradients->read(adj_eid * N + offset);
            const Float3x3 diag_hess = constaint_hessians->read(adj_eid * (N * N) + offset);

            buffer_add(sa_cgB, vid, -grad);
            buffer_add(sa_cgA_diag, vid, diag_hess);

            const uint N_off = N - 1;
            for (uint ii = 0; ii < N_off; ii++)  // For each off-diagonal in curr row
            {
                Float3x3 offdiag_hess = constaint_hessians->read(adj_eid * (N * N) + N + offset * N_off + ii);
                Uint offdiag_offset =
                    constaint_offsets_in_adjlist->read(adj_eid * (N * N_off) + offset * N_off + ii);
                auto triplet = sa_cgA_offdiag_triplet->read(curr_prefix + offdiag_offset);
                add_triplet_matrix(triplet, offdiag_hess);
                sa_cgA_offdiag_triplet->write(curr_prefix + offdiag_offset, triplet);
            }
        };
    };

    // Assembly
    if (host_sim_data->sa_stretch_springs.size() != 0)
        compiler.compile(
            fn_material_energy_assembly_stretch_spring,
            [vert_adj_constraints_csr     = sim_data->sa_vert_adj_stretch_springs_csr.view(),
             constaints                   = sim_data->sa_stretch_springs.view(),
             constaint_gradients          = sim_data->sa_stretch_springs_gradients.view(),
             constaint_hessians           = sim_data->sa_stretch_springs_hessians.view(),
             constaint_offsets_in_adjlist = sim_data->sa_stretch_springs_offsets_in_adjlist.view(),
             assembly_template]()
            {
                const Uint vid = dispatch_x();
                assembly_template(2, vid, vert_adj_constraints_csr, constaints, constaint_gradients, constaint_hessians, constaint_offsets_in_adjlist);
            });

    if (host_sim_data->sa_stretch_faces.size() != 0)
        compiler.compile(fn_material_energy_assembly_stretch_face,
                         [vert_adj_constraints_csr = sim_data->sa_vert_adj_stretch_faces_csr.view(),
                          constaints               = sim_data->sa_stretch_faces.view(),
                          constaint_gradients      = sim_data->sa_stretch_faces_gradients.view(),
                          constaint_hessians       = sim_data->sa_stretch_faces_hessians.view(),
                          constaint_offsets_in_adjlist = sim_data->sa_stretch_faces_offsets_in_adjlist.view(),
                          assembly_template]()
                         {
                             const Uint vid = dispatch_x();
                             assembly_template(3, vid, vert_adj_constraints_csr, constaints, constaint_gradients, constaint_hessians, constaint_offsets_in_adjlist);
                         });

    if (host_sim_data->sa_bending_edges.size() != 0)
        compiler.compile(fn_material_energy_assembly_bending,
                         [vert_adj_constraints_csr = sim_data->sa_vert_adj_bending_edges_csr.view(),
                          constaints               = sim_data->sa_bending_edges.view(),
                          constaint_gradients      = sim_data->sa_bending_edges_gradients.view(),
                          constaint_hessians       = sim_data->sa_bending_edges_hessians.view(),
                          constaint_offsets_in_adjlist = sim_data->sa_bending_edges_offsets_in_adjlist.view(),
                          assembly_template]()
                         {
                             const Uint vid = dispatch_x();
                             assembly_template(4, vid, vert_adj_constraints_csr, constaints, constaint_gradients, constaint_hessians, constaint_offsets_in_adjlist);
                         });

    if (host_sim_data->sa_affine_bodies.size() != 0)
        compiler.compile(fn_material_energy_assembly_affine_body,
                         [vert_adj_constraints_csr = sim_data->sa_vert_adj_affine_bodies_csr.view(),
                          constaints               = sim_data->sa_affine_bodies.view(),
                          constaint_gradients      = sim_data->sa_affine_bodies_gradients.view(),
                          constaint_hessians       = sim_data->sa_affine_bodies_hessians.view(),
                          constaint_offsets_in_adjlist = sim_data->sa_affine_bodies_offsets_in_adjlist.view(),
                          assembly_template](const Uint prefix)
                         {
                             const Uint vid = prefix + dispatch_x();
                             assembly_template(4, vid, vert_adj_constraints_csr, constaints, constaint_gradients, constaint_hessians, constaint_offsets_in_adjlist);
                         });
}
void NewtonSolver::compile_evaluate(AsyncCompiler& compiler, const luisa::compute::ShaderOption& default_option)
{
    using namespace luisa::compute;

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

    if (host_sim_data->sa_stretch_springs.size() != 0)
        compiler.compile<1>(
            fn_evaluate_spring,
            [sa_x = sim_data->sa_x.view(),
             // sa_cgB = sim_data->sa_cgB.view(),
             // sa_cgA_diag = sim_data->sa_cgA_diag.view(),
             output_gradient_ptr = sim_data->sa_stretch_springs_gradients.view(),
             output_hessian_ptr  = sim_data->sa_stretch_springs_hessians.view(),
             sa_edges            = sim_data->sa_stretch_springs.view(),
             sa_rest_length      = sim_data->sa_stretch_spring_rest_state_length.view(),
             sa_stretch_spring_stiffness = sim_data->sa_stretch_spring_stiffness.view()](const Float stiffness_stretch)
            {
                const UInt eid  = dispatch_id().x;
                UInt2      edge = sa_edges->read(eid);

                Float3   vert_pos[2]  = {sa_x->read(edge.x), sa_x->read(edge.y)};
                Float3   gradients[2] = {make_float3(0.0f), make_float3(0.0f)};
                Float3x3 He           = make_float3x3(0.0f);

                const Float L                = sa_rest_length->read(eid);
                const Float stiffness_spring = sa_stretch_spring_stiffness->read(eid);
                // const Float stiffness_spring = stiffness_stretch;

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

    if (host_sim_data->sa_stretch_faces.size() != 0)
        compiler.compile<1>(
            fn_evaluate_stretch_face,
            [sa_x                       = sim_data->sa_x.view(),
             output_gradient_ptr        = sim_data->sa_stretch_faces_gradients.view(),
             output_hessian_ptr         = sim_data->sa_stretch_faces_hessians.view(),
             sa_faces                   = sim_data->sa_stretch_faces.view(),
             sa_stretch_faces_Dm_inv    = sim_data->sa_stretch_faces_Dm_inv.view(),
             sa_stretch_faces_rest_area = sim_data->sa_stretch_faces_rest_area.view(),
             sa_stretch_faces_mu_lambda = sim_data->sa_stretch_faces_mu_lambda.view()]()
            {
                const UInt  fid  = dispatch_id().x;
                const UInt3 face = sa_faces->read(fid);

                Float3   vert_pos[3] = {sa_x->read(face[0]), sa_x->read(face[1]), sa_x->read(face[2])};
                Float3x3 gradients;
                Float9x9 hessians;

                Float2x2 Dm_inv = sa_stretch_faces_Dm_inv->read(fid);
                Float    area   = sa_stretch_faces_rest_area->read(fid);

                Float2 mu_lambda    = sa_stretch_faces_mu_lambda->read(fid);
                Float  mu_cloth     = mu_lambda[0];
                Float  lambda_cloth = mu_lambda[1];

                StretchEnergy::compute_gradient_hessian(
                    vert_pos[0], vert_pos[1], vert_pos[2], Dm_inv, mu_cloth, lambda_cloth, area, gradients, hessians);

                // Output
                {
                    output_gradient_ptr->write(fid * 3 + 0, gradients[0]);
                    output_gradient_ptr->write(fid * 3 + 1, gradients[1]);
                    output_gradient_ptr->write(fid * 3 + 2, gradients[2]);
                }
                {
                    output_hessian_ptr->write(fid * 9 + 0, hessians->block(0, 0));
                    output_hessian_ptr->write(fid * 9 + 1, hessians->block(1, 1));
                    output_hessian_ptr->write(fid * 9 + 2, hessians->block(2, 2));

                    output_hessian_ptr->write(fid * 9 + 3, hessians->block(1, 0));  // lower triangle
                    output_hessian_ptr->write(fid * 9 + 4, hessians->block(2, 0));
                    output_hessian_ptr->write(fid * 9 + 5, hessians->block(0, 1));
                    output_hessian_ptr->write(fid * 9 + 6, hessians->block(2, 1));
                    output_hessian_ptr->write(fid * 9 + 7, hessians->block(0, 2));
                    output_hessian_ptr->write(fid * 9 + 8, hessians->block(1, 2));
                }
            },
            default_option);

    if (host_sim_data->sa_bending_edges.size() != 0)
        compiler.compile<1>(
            fn_evaluate_bending,
            [sa_x = sim_data->sa_x.view(),
             // sa_cgB = sim_data->sa_cgB.view(),
             // sa_cgA_diag = sim_data->sa_cgA_diag.view(),
             // sa_cgA_offdiag_bending = sim_data->sa_cgA_offdiag_bending.view(),
             output_gradient_ptr         = sim_data->sa_bending_edges_gradients.view(),
             output_hessian_ptr          = sim_data->sa_bending_edges_hessians.view(),
             sa_edges                    = sim_data->sa_bending_edges.view(),
             sa_bending_edges_Q          = sim_data->sa_bending_edges_Q.view(),
             sa_bending_edges_rest_area  = sim_data->sa_bending_edges_rest_area.view(),
             sa_bending_edges_rest_angle = sim_data->sa_bending_edges_rest_angle.view(),
             sa_bending_edges_stiffness = sim_data->sa_bending_edges_stiffness.view()](const Float scaling)
            {
                const UInt  eid  = dispatch_id().x;
                const UInt4 edge = sa_edges->read(eid);

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

                // Refers to ppf-contact-solver
                const Float rest_angle = sa_bending_edges_rest_angle->read(eid);
                const Float angle =
                    BendingEnergy::compute_d_theta_d_x(vert_pos[0], vert_pos[1], vert_pos[2], vert_pos[3], gradients);
                const Float delta_angle = angle - rest_angle;

                const Float area  = sa_bending_edges_rest_area->read(eid);
                const Float stiff = sa_bending_edges_stiffness->read(eid) * scaling * area;

                {
                    output_gradient_ptr->write(eid * 4 + 0, stiff * delta_angle * gradients[0]);
                    output_gradient_ptr->write(eid * 4 + 1, stiff * delta_angle * gradients[1]);
                    output_gradient_ptr->write(eid * 4 + 2, stiff * delta_angle * gradients[2]);
                    output_gradient_ptr->write(eid * 4 + 3, stiff * delta_angle * gradients[3]);

                    auto outer = [&](const uint ii, const uint jj) -> Float3x3
                    {
                        // Use simple Gauss-Newton approximation
                        return stiff * outer_product(gradients[ii], gradients[jj]);
                    };
                    output_hessian_ptr->write(eid * 16 + 0, outer(0, 0));
                    output_hessian_ptr->write(eid * 16 + 1, outer(1, 1));
                    output_hessian_ptr->write(eid * 16 + 2, outer(2, 2));
                    output_hessian_ptr->write(eid * 16 + 3, outer(3, 3));

                    uint idx = 4;
                    for (uint ii = 0; ii < 4; ii++)
                    {
                        for (uint jj = 0; jj < 4; jj++)
                        {
                            if (ii != jj)
                            {
                                output_hessian_ptr->write(eid * 16 + idx, outer(ii, jj));
                                idx += 1;
                            }
                        }
                    }
                }
            },
            default_option);

    if (host_sim_data->num_affine_bodies != 0)
    {
        compiler.compile<1>(
            fn_evaluate_abd_inertia,
            [affine_bodies = sim_data->sa_affine_bodies.view(),
             abd_q         = sim_data->sa_affine_bodies_q.view(),
             abd_q_tilde   = sim_data->sa_affine_bodies_q_tilde.view(),
             abd_gradients = sim_data->sa_affine_bodies_gradients.view(),
             abd_hessians  = sim_data->sa_affine_bodies_hessians.view(),
             abd_is_fixed  = sim_data->sa_affine_bodies_is_fixed.view(),
             sa_affine_bodies_mass_matrix =
                 sim_data->sa_affine_bodies_mass_matrix.view()](const Float substep_dt, const Float stiffness_dirichlet)
            {
                const UInt  body_idx = dispatch_id().x;
                const Float h        = substep_dt;
                const Float h_2_inv  = 1.0f / (h * h);

                Float3 delta_q[4] = {abd_q->read(4 * body_idx + 0) - abd_q_tilde->read(4 * body_idx + 0),
                                     abd_q->read(4 * body_idx + 1) - abd_q_tilde->read(4 * body_idx + 1),
                                     abd_q->read(4 * body_idx + 2) - abd_q_tilde->read(4 * body_idx + 2),
                                     abd_q->read(4 * body_idx + 3) - abd_q_tilde->read(4 * body_idx + 3)};

                // const Uint4 body = affine_bodies->read(body_idx);
                //  abd_q->read(body[0]) - abd_q_tilde->read(body[0]),
                //  abd_q->read(body[1]) - abd_q_tilde->read(body[1]),
                //  abd_q->read(body[2]) - abd_q_tilde->read(body[2]),
                //  abd_q->read(body[3]) - abd_q_tilde->read(body[3])

                Float4x4 mass_matrix = sa_affine_bodies_mass_matrix->read(body_idx);
                Float3   gradient[4] = {Zero3, Zero3, Zero3, Zero3};

                for (uint ii = 0; ii < 4; ii++)
                {
                    for (uint jj = 0; jj < 4; jj++)
                    {
                        gradient[ii] += mass_matrix[ii][jj] * delta_q[jj];
                    }
                }

                Float alpha = 1.0f;
                $if(abd_is_fixed->read(body_idx) != 0)
                {
                    alpha = (1.0f + stiffness_dirichlet);
                };

                abd_gradients->write(4 * body_idx + 0, alpha * h_2_inv * gradient[0]);
                abd_gradients->write(4 * body_idx + 1, alpha * h_2_inv * gradient[1]);
                abd_gradients->write(4 * body_idx + 2, alpha * h_2_inv * gradient[2]);
                abd_gradients->write(4 * body_idx + 3, alpha * h_2_inv * gradient[3]);

                abd_hessians->write(16 * body_idx + 0, make_eye3x3(alpha * h_2_inv * mass_matrix[0][0]));
                abd_hessians->write(16 * body_idx + 1, make_eye3x3(alpha * h_2_inv * mass_matrix[1][1]));
                abd_hessians->write(16 * body_idx + 2, make_eye3x3(alpha * h_2_inv * mass_matrix[2][2]));
                abd_hessians->write(16 * body_idx + 3, make_eye3x3(alpha * h_2_inv * mass_matrix[3][3]));

                uint idx = 4;
                for (uint ii = 0; ii < 4; ii++)
                {
                    for (uint jj = 0; jj < 4; jj++)
                    {
                        if (ii != jj)
                        {
                            abd_hessians->write(16 * body_idx + idx,
                                                make_eye3x3(alpha * h_2_inv * mass_matrix[ii][jj]));
                            idx += 1;
                        }
                    }
                }
            },
            default_option);

        compiler.compile<1>(
            fn_evaluate_abd_orthogonality,
            [affine_bodies = sim_data->sa_affine_bodies.view(),
             abd_q         = sim_data->sa_affine_bodies_q.view(),
             abd_gradients = sim_data->sa_affine_bodies_gradients.view(),
             abd_hessians  = sim_data->sa_affine_bodies_hessians.view(),
             abd_is_fixed  = sim_data->sa_affine_bodies_is_fixed.view(),
             abd_volume    = sim_data->sa_affine_bodies_volume.view(),
             abd_kappa     = sim_data->sa_affine_bodies_kappa.view()]()
            {
                const UInt body_idx = dispatch_id().x;

                Float3   ortho_gradient[3] = {Zero3};
                Float3x3 ortho_hessian[6]  = {Zero3x3};

                Float3x3 A = make_float3x3(abd_q->read(4 * body_idx + 1),
                                           abd_q->read(4 * body_idx + 2),
                                           abd_q->read(4 * body_idx + 3));

                const Float kappa = abd_kappa->read(body_idx);
                const Float V     = abd_volume->read(body_idx);

                Float stiff = kappa * V;
                for (uint ii = 0; ii < 3; ii++)
                {
                    Float3 grad = (-1.0f) * A[ii];
                    for (uint jj = 0; jj < 3; jj++)
                    {
                        grad += dot_vec(A[ii], A[jj]) * A[jj];
                    }
                    // cgB.block<3, 1>(3 + 3 * ii, 0) -= 4 * stiff * float3_to_eigen3(grad);
                    ortho_gradient[ii] += 4.0f * stiff * grad;
                }
                uint idx = 0;
                for (uint ii = 0; ii < 3; ii++)
                {
                    for (uint jj = ii; jj < 3; jj++)
                    {
                        Float3x3 hessian = Zero3x3;
                        if (ii == jj)
                        {
                            Float3x3 qiqiT = outer_product(A[ii], A[ii]);
                            Float3x3 qiTqi = (dot_vec(A[ii], A[ii]) - 1.0f) * Identity3x3;
                            hessian        = qiqiT + qiTqi;
                            for (uint kk = 0; kk < 3; kk++)
                            {
                                hessian = hessian + outer_product(A[kk], A[kk]);
                            }
                        }
                        else
                        {
                            hessian = outer_product(A[jj], A[ii]) + dot_vec(A[ii], A[jj]) * Identity3x3;
                        }
                        ortho_hessian[idx] = ortho_hessian[idx] + 4.0f * stiff * hessian;
                        idx += 1;
                    }
                }

                auto add_to_grad = [&](const uint i)
                {
                    Float3 orig_grad = abd_gradients->read(4 * body_idx + 1 + i);
                    abd_gradients->write(4 * body_idx + 1 + i, orig_grad + ortho_gradient[i]);
                };
                auto add_to_hess = [&](const uint i, const uint j, const bool use_transpose)
                {
                    Float3x3 orig_hess  = abd_hessians->read(16 * body_idx + i);
                    Float3x3 ortho_hess = use_transpose ? transpose(ortho_hessian[j]) : ortho_hessian[j];
                    abd_hessians->write(16 * body_idx + i, orig_hess + ortho_hess);
                };

                //                 0   4   5   6
                //   0   1   2     7   1   8   9
                //  t1   3   4    10  11   2  12
                //  t2  t4   5    13  14  15   3
                add_to_grad(0);
                add_to_grad(1);
                add_to_grad(2);

                add_to_hess(1, 0, false);
                add_to_hess(8, 1, false);
                add_to_hess(9, 2, false);
                add_to_hess(11, 1, true);
                add_to_hess(2, 3, false);
                add_to_hess(12, 4, false);
                add_to_hess(14, 2, true);
                add_to_hess(15, 4, true);
                add_to_hess(3, 5, false);
            },
            default_option);

        compiler.compile<1>(
            fn_evaluate_abd_ground_collision,
            [sa_scaled_model_x   = mesh_data->sa_scaled_model_x.view(),
             sa_x                = sim_data->sa_x.view(),
             abd_q               = sim_data->sa_affine_bodies_q.view(),
             abd_perVert_body_id = sim_data->sa_vert_affine_bodies_id.view(),
             sa_rest_vert_area   = mesh_data->sa_rest_vert_area.view(),
             abd_gradients       = sim_data->sa_affine_bodies_gradients.view(),
             abd_hessians        = sim_data->sa_affine_bodies_hessians.view(),
             sa_is_fixed         = mesh_data->sa_is_fixed.view()](
                Float floor_y, Bool use_ground_collision, Float stiffness, Float d_hat, Float thickness, Uint vid_start)
            {
                const UInt vid = vid_start + dispatch_id().x;

                $if(use_ground_collision)
                {
                    $if(!sa_is_fixed->read(vid))
                    {
                        Float3 x_k  = sa_x->read(vid);
                        Float  diff = x_k.y - floor_y;

                        $if(diff < d_hat + thickness)
                        {
                            Float    C        = diff - (d_hat + thickness);
                            float3   normal   = luisa::make_float3(0, 1, 0);
                            Float    area     = sa_rest_vert_area->read(vid);
                            Float    stiff    = stiffness * area;
                            Float3   gradient = stiff * C * normal;
                            Float3x3 hessian  = stiff * outer_product(normal, normal);

                            const Uint   body_idx = abd_perVert_body_id->read(vid);
                            const Float4 weight   = make_float4(1.0f, sa_scaled_model_x->read(vid));

                            uint idx = 4;
                            for (uint ii = 0; ii < 4; ii++)
                            {
                                Float  wi          = weight[ii];
                                Float3 affine_grad = wi * gradient;
                                atomic_buffer_add(abd_gradients, 4 * body_idx + ii, affine_grad);
                                for (uint jj = 0; jj < 4; jj++)
                                {
                                    Float    wj          = weight[jj];
                                    Float3x3 affine_hess = wi * wj * hessian;
                                    if (ii == jj)
                                    {
                                        atomic_buffer_add(abd_hessians, 16 * body_idx + ii, affine_hess);
                                    }
                                    else
                                    {
                                        atomic_buffer_add(abd_hessians, 16 * body_idx + idx, affine_hess);
                                        idx += 1;
                                    }
                                }
                            }
                        };
                    };
                };
            },
            default_option);
    }
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
                                  //   LUISA_INFO("Body {}'s block_{} : q = {}, v = {} , dt = {} => q_tilde = {}",
                                  //              block_idx / 4,
                                  //              block_idx % 4,
                                  //              q_prev,
                                  //              q_v,
                                  //              substep_dt,
                                  //              q_pred);
                              });
}
void NewtonSolver::host_update_velocity()
{
    CpuParallel::parallel_for(0,
                              host_sim_data->num_verts_soft,
                              [sa_x            = host_sim_data->sa_x.data(),
                               sa_v            = host_sim_data->sa_v.data(),
                               sa_x_step_start = host_sim_data->sa_x_step_start.data(),
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
void NewtonSolver::host_evaluate_inertia()
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
                                  float3   gradient = mass * h_2_inv * (x_k - x_tilde);
                                  float3x3 hessian  = luisa::make_float3x3(1.0f) * mass * h_2_inv;

                                  if (sa_is_fixed[vid])
                                  {
                                      gradient = stiffness_dirichlet * gradient;
                                      hessian  = stiffness_dirichlet * hessian;
                                  }
                                  {
                                      // sa_cgX[vid] = dx_0;
                                      sa_cgB[vid]      = -gradient;
                                      sa_cgA_diag[vid] = hessian;
                                  }
                              });

    float3*   abd_gradients = host_sim_data->sa_affine_bodies_gradients.data();
    float3x3* abd_hessians  = host_sim_data->sa_affine_bodies_hessians.data();

    const auto& abd_q       = host_sim_data->sa_affine_bodies_q;
    const auto& abd_q_tilde = host_sim_data->sa_affine_bodies_q_tilde;

    CpuParallel::parallel_for(
        0,
        host_sim_data->sa_affine_bodies.size(),
        [&](const uint body_idx)
        {
            const float substep_dt = get_scene_params().get_substep_dt();
            const float h          = substep_dt;
            const float h_2_inv    = 1.f / (h * h);

            float3   delta_q[4]  = {abd_q[4 * body_idx + 0] - abd_q_tilde[4 * body_idx + 0],
                                    abd_q[4 * body_idx + 1] - abd_q_tilde[4 * body_idx + 1],
                                    abd_q[4 * body_idx + 2] - abd_q_tilde[4 * body_idx + 2],
                                    abd_q[4 * body_idx + 3] - abd_q_tilde[4 * body_idx + 3]};
            float4x4 mass_matrix = host_sim_data->sa_affine_bodies_mass_matrix[body_idx];
            float3   gradient[4] = {Zero3, Zero3, Zero3, Zero3};

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
        });
}
void NewtonSolver::host_evaluate_orthogonality()
{
    float3*   abd_gradients = host_sim_data->sa_affine_bodies_gradients.data();
    float3x3* abd_hessians  = host_sim_data->sa_affine_bodies_hessians.data();

    CpuParallel::parallel_for(
        0,
        host_sim_data->sa_affine_bodies.size(),
        [&](const uint body_idx)
        {
            float3   ortho_gradient[3] = {Zero3};
            float3x3 ortho_hessian[6]  = {Zero3x3};

            const float substep_dt = get_scene_params().get_substep_dt();
            const float h          = substep_dt;
            const float h_2_inv    = 1.f / (h * h);

            float3x3 A = luisa::make_float3x3(host_sim_data->sa_affine_bodies_q[4 * body_idx + 1],
                                              host_sim_data->sa_affine_bodies_q[4 * body_idx + 2],
                                              host_sim_data->sa_affine_bodies_q[4 * body_idx + 3]);

            const float kappa = host_sim_data->sa_affine_bodies_kappa[body_idx];
            const float V     = host_sim_data->sa_affine_bodies_volume[body_idx];

            float stiff = kappa * V;
            for (uint ii = 0; ii < 3; ii++)
            {
                float3 grad = (-1.0f) * A[ii];
                for (uint jj = 0; jj < 3; jj++)
                {
                    grad += dot_vec(A[ii], A[jj]) * A[jj];
                }
                // cgB.block<3, 1>(3 + 3 * ii, 0) -= 4 * stiff * float3_to_eigen3(grad);
                ortho_gradient[ii] += 4.0f * stiff * grad;
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
                        float3x3 qiTqi = (dot_vec(A[ii], A[ii]) - 1.0f) * Identity3x3;
                        hessian        = qiqiT + qiTqi;
                        for (uint kk = 0; kk < 3; kk++)
                        {
                            hessian = hessian + outer_product(A[kk], A[kk]);
                        }
                    }
                    else
                    {
                        hessian = outer_product(A[jj], A[ii]) + dot_vec(A[ii], A[jj]) * Identity3x3;
                    }
                    // LUISA_INFO("hess of {} adj {} = {}", ii, jj, hessian);
                    // cgA.block<3, 3>(3 + 3 * ii, 3 + 3 * jj) += 4.0f * stiff * float3x3_to_eigen3x3(hessian);
                    ortho_hessian[idx] = ortho_hessian[idx] + 4.0f * stiff * hessian;
                    idx += 1;
                }
            }

            auto* body_grad_ptr = &abd_gradients[4 * body_idx];
            auto* body_hess_ptr = &abd_hessians[16 * body_idx];

            //                 0   4   5   6
            //   0   1   2     7   1   8   9
            //  t1   3   4    10  11   2  12
            //  t2  t4   5    13  14  15   3
            body_grad_ptr[1] += ortho_gradient[0];
            body_grad_ptr[2] += ortho_gradient[1];
            body_grad_ptr[3] += ortho_gradient[2];

            body_hess_ptr[1]  = body_hess_ptr[1] + ortho_hessian[0];
            body_hess_ptr[8]  = body_hess_ptr[8] + ortho_hessian[1];
            body_hess_ptr[9]  = body_hess_ptr[9] + ortho_hessian[2];
            body_hess_ptr[11] = body_hess_ptr[11] + luisa::transpose(ortho_hessian[1]);
            body_hess_ptr[2]  = body_hess_ptr[2] + ortho_hessian[3];
            body_hess_ptr[12] = body_hess_ptr[12] + ortho_hessian[4];
            body_hess_ptr[14] = body_hess_ptr[14] + luisa::transpose(ortho_hessian[2]);
            body_hess_ptr[15] = body_hess_ptr[15] + luisa::transpose(ortho_hessian[4]);
            body_hess_ptr[3]  = body_hess_ptr[3] + ortho_hessian[5];
        },
        32);
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
    float       stiffness_ground = get_scene_params().stiffness_collision;

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

    float3*    abd_gradients = host_sim_data->sa_affine_bodies_gradients.data();
    float3x3*  abd_hessians  = host_sim_data->sa_affine_bodies_hessians.data();
    const uint num_bodies    = host_sim_data->sa_affine_bodies.size();
    CpuParallel::single_thread_for(
        0,
        host_sim_data->sa_affine_bodies.size(),
        [&](const uint body_idx)
        {
            if (get_scene_params().use_floor)
            {
                const uint mesh_idx    = host_sim_data->sa_affine_bodies_mesh_id[body_idx];
                const uint curr_prefix = host_mesh_data->prefix_num_verts[mesh_idx];
                const uint next_prefix = host_mesh_data->prefix_num_verts[mesh_idx + 1];

                for (uint vid = curr_prefix; vid < next_prefix; vid++)
                {
                    float3 x_k  = host_sim_data->sa_x[vid];
                    float  diff = x_k.y - get_scene_params().floor.y;

                    if (diff < d_hat + thickness)
                    {
                        float    C        = diff - (d_hat + thickness);
                        float3   normal   = luisa::make_float3(0, 1, 0);
                        float    area     = host_mesh_data->sa_rest_vert_area[vid];
                        float    stiff    = stiffness_ground * area;
                        float    k1       = stiff * C;
                        float3   model_x  = host_mesh_data->sa_scaled_model_x[vid];
                        float3   gradient = stiff * C * normal;
                        float3x3 hessian  = stiff * outer_product(normal, normal);

                        // Friction

                        uint   idx    = 4;
                        float4 weight = luisa::make_float4(1.0f, model_x.x, model_x.y, model_x.z);
                        for (uint ii = 0; ii < 4; ii++)
                        {
                            float  wi          = weight[ii];
                            float3 affine_grad = wi * gradient;
                            abd_gradients[4 * body_idx + ii] += affine_grad;
                            for (uint jj = 0; jj < 4; jj++)
                            {
                                float    wj          = weight[jj];
                                float3x3 affine_hess = wi * wj * hessian;
                                if (ii == jj)
                                {
                                    abd_hessians[16 * body_idx + ii] = abd_hessians[16 * body_idx + ii] + affine_hess;
                                }
                                else
                                {
                                    abd_hessians[16 * body_idx + idx] = abd_hessians[16 * body_idx + idx] + affine_hess;
                                    idx += 1;
                                }
                            }
                        }

                        // auto J = AffineBodyDynamics::get_jacobian_dxdq(model_x);
                        // cgB.block<12, 1>(body_idx * 12, 0) -= J.transpose() * float3_to_eigen3(gradient);
                        // cgA.block<12, 12>(body_idx * 12, body_idx * 12) +=
                        //     J.transpose() * float3x3_to_eigen3x3(hessian) * J;
                    }
                }
            }
        });
}
void NewtonSolver::host_test_dynamics(luisa::compute::Stream& stream)
{
    stream << luisa::compute::synchronize();

    const uint                 num_dof    = host_sim_data->num_dof;
    const uint                 num_bodies = host_sim_data->sa_affine_bodies.size();
    Eigen::SparseMatrix<float> cgA(3 * num_dof, 3 * num_dof);
    Eigen::MatrixXf            cgB(3 * num_dof, 1);
    cgA.setZero();
    cgB.setZero();

    const uint prefix = host_sim_data->num_verts_soft;

    // Soft inertia
    // if constexpr (false)
    {
        std::vector<EigenTripletBlock<1>> hessian_blocks(host_sim_data->num_verts_soft);
        CpuParallel::single_thread_for(0,
                                       host_sim_data->num_verts_soft,
                                       [sa_cgB              = host_sim_data->sa_cgB.data(),
                                        sa_cgA_diag         = host_sim_data->sa_cgA_diag.data(),
                                        sa_x                = host_sim_data->sa_x.data(),
                                        sa_x_tilde          = host_sim_data->sa_x_tilde.data(),
                                        sa_is_fixed         = host_mesh_data->sa_is_fixed.data(),
                                        sa_vert_mass        = host_mesh_data->sa_vert_mass.data(),
                                        substep_dt          = get_scene_params().get_substep_dt(),
                                        stiffness_dirichlet = get_scene_params().stiffness_dirichlet,
                                        &cgB,
                                        &hessian_blocks](const uint vid)
                                       {
                                           const float h       = substep_dt;
                                           const float h_2_inv = 1.f / (h * h);

                                           float3 x_k     = sa_x[vid];
                                           float3 x_tilde = sa_x_tilde[vid];
                                           // float3 v_0 = sa_v[vid];

                                           float  mass     = sa_vert_mass[vid];
                                           float3 gradient = mass * h_2_inv * (x_k - x_tilde);
                                           float3x3 hessian = luisa::make_float3x3(1.0f) * mass * h_2_inv;

                                           if (sa_is_fixed[vid])
                                           {
                                               gradient = stiffness_dirichlet * gradient;
                                               hessian  = stiffness_dirichlet * hessian;
                                           }
                                           // LUISA_INFO("vid {}: is_fixed? {}, x_k = {}, x_tilde = {}, mass = {} inertia gradient = {}",
                                           //            vid,
                                           //            sa_is_fixed[vid] == 1,
                                           //            x_k,
                                           //            x_tilde,
                                           //            mass,
                                           //            gradient);
                                           {
                                               cgB.block<3, 1>(vid * 3, 0) = -float3_to_eigen3(gradient);
                                               hessian_blocks[vid]         = {.indices = {vid},
                                                                              .matrix = float3x3_to_eigen3x3(hessian)};
                                           }
                                       });

        convert_triplets_to_sparse_matrix(cgA, hessian_blocks);
    }

    // Soft stretch face
    // if constexpr (false)
    {
        std::vector<EigenTripletBlock<3>> hessian_blocks(host_sim_data->sa_stretch_faces.size());
        CpuParallel::single_thread_for(
            0,
            host_sim_data->sa_stretch_faces.size(),
            [sa_x                       = host_sim_data->sa_x.data(),
             sa_rest_x                  = host_mesh_data->sa_rest_x.data(),
             sa_faces                   = host_sim_data->sa_stretch_faces.data(),
             sa_stretch_faces_rest_area = host_sim_data->sa_stretch_faces_rest_area.data(),
             sa_stretch_faces_Dm_inv    = host_sim_data->sa_stretch_faces_Dm_inv.data(),
             sa_stretch_faces_mu_lambda = host_sim_data->sa_stretch_faces_mu_lambda.data(),
             youngs_modulus_cloth       = get_scene_params().youngs_modulus_cloth,
             poisson_ratio_cloth        = get_scene_params().poisson_ratio_cloth,
             &hessian_blocks,
             &cgB](const uint fid)
            {
                uint3 face = sa_faces[fid];

                float3 vert_pos[3] = {sa_x[face[0]], sa_x[face[1]], sa_x[face[2]]};
                float3 x_bars[3]   = {sa_rest_x[face[0]], sa_rest_x[face[1]], sa_rest_x[face[2]]};
                // float3   gradients[3] = {Zero3, Zero3, Zero3};
                // float3x3 hessians[3][3];
                // for (auto& tmp : hessians)
                // {
                //     for (auto& hess : tmp)
                //         hess = Zero3x3;
                // }
                float2x2    Dm_inv    = sa_stretch_faces_Dm_inv[fid];
                float2      mu_lambda = sa_stretch_faces_mu_lambda[fid];  // {lambda, mu}
                const float lambda    = mu_lambda[1];
                const float mu        = mu_lambda[0];
                float       area      = sa_stretch_faces_rest_area[fid];

                Eigen::Matrix<float, 9, 1> G;
                Eigen::Matrix<float, 9, 9> H;

                float3x3 gradients;
                float9x9 hessians;
                StretchEnergy::compute_gradient_hessian(
                    vert_pos[0], vert_pos[1], vert_pos[2], Dm_inv, mu, lambda, area, gradients, hessians);
                // LUISA_INFO("Face {}: grad = {}", face, gradients);
                cgB.block<3, 1>(3 * face[0], 0) -= float3_to_eigen3(gradients[0]);
                cgB.block<3, 1>(3 * face[1], 0) -= float3_to_eigen3(gradients[1]);
                cgB.block<3, 1>(3 * face[2], 0) -= float3_to_eigen3(gradients[2]);
                EigenFloat9x9 tmpH = hessians.to_eigen_matrix();
                // EigenFloat9x9 tmpH;
                // for (uint ii = 0; ii < 3; ii++)
                // {
                //     for (uint jj = 0; jj < 3; jj++)
                //     {
                //         tmpH.block<3, 3>(ii * 3, jj * 3) = float3x3_to_eigen3x3(hessians[ii][jj]);
                //     }
                // }
                // StretchEnergy::libuipc::compute_gradient_hessian(
                //     vert_pos[0], vert_pos[1], vert_pos[2], x_bars[0], x_bars[1], x_bars[2], 1e7f, 0.46f, area, G, H);
                // cgB.block<3, 1>(3 * face[0], 0) -= G.block<3, 1>(0, 0);
                // cgB.block<3, 1>(3 * face[1], 0) -= G.block<3, 1>(3, 0);
                // cgB.block<3, 1>(3 * face[2], 0) -= G.block<3, 1>(6, 0);
                // EigenFloat9x9 tmpH  = H;

                hessian_blocks[fid] = {.indices = {face[0], face[1], face[2]}, .matrix = tmpH};
            });
        convert_triplets_to_sparse_matrix(cgA, hessian_blocks);
    }

    // Soft stretch spring
    if constexpr (false)
    {
        std::vector<EigenTripletBlock<2>> hessian_blocks(host_sim_data->sa_stretch_springs.size());
        CpuParallel::single_thread_for(
            0,
            host_sim_data->sa_stretch_springs.size(),
            [sa_x                        = host_sim_data->sa_x.data(),
             sa_edges                    = host_sim_data->sa_stretch_springs.data(),
             sa_rest_length              = host_sim_data->sa_stretch_spring_rest_state_length.data(),
             sa_stretch_spring_stiffness = host_sim_data->sa_stretch_spring_stiffness.data(),
             output_gradient_ptr         = host_sim_data->sa_stretch_springs_gradients.data(),
             output_hessian_ptr          = host_sim_data->sa_stretch_springs_hessians.data(),
             stiffness_stretch           = get_scene_params().stiffness_spring,
             &cgB,
             &hessian_blocks](const uint eid)
            {
                uint2 edge = sa_edges[eid];

                float3   vert_pos[2]  = {sa_x[edge[0]], sa_x[edge[1]]};
                float3   gradients[2] = {Zero3, Zero3};
                float3x3 He           = luisa::make_float3x3(0.0f);

                const float L                        = sa_rest_length[eid];
                const float stiffness_stretch_spring = sa_stretch_spring_stiffness[eid];
                // const float stiffness_stretch_spring = stiffness_stretch;

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
                    cgB.block<3, 1>(3 * edge[0], 0) -= float3_to_eigen3(gradients[0]);
                    cgB.block<3, 1>(3 * edge[1], 0) -= float3_to_eigen3(gradients[1]);

                    EigenFloat6x6 tmpH;
                    tmpH.block<3, 3>(0, 0) = float3x3_to_eigen3x3(He);
                    tmpH.block<3, 3>(0, 3) = float3x3_to_eigen3x3(-1.0f * He);
                    tmpH.block<3, 3>(3, 0) = float3x3_to_eigen3x3(-1.0f * He);
                    tmpH.block<3, 3>(3, 3) = float3x3_to_eigen3x3(He);
                    hessian_blocks[eid]    = {.indices = {edge[0], edge[1]}, .matrix = tmpH};
                }
            });

        convert_triplets_to_sparse_matrix(cgA, hessian_blocks);
    }

    // Bending
    // if constexpr (false)
    {
        std::vector<EigenTripletBlock<4>> hessian_blocks(host_sim_data->sa_bending_edges.size());
        CpuParallel::single_thread_for(
            0,
            host_sim_data->sa_bending_edges.size(),
            [sa_x                        = host_sim_data->sa_x.data(),
             sa_bending_edges            = host_sim_data->sa_bending_edges.data(),
             sa_bending_edges_Q          = host_sim_data->sa_bending_edges_Q.data(),
             sa_bending_edges_rest_angle = host_sim_data->sa_bending_edges_rest_angle.data(),
             sa_bending_edges_rest_area  = host_sim_data->sa_bending_edges_rest_area.data(),
             sa_bending_edges_stiffness  = host_sim_data->sa_bending_edges_stiffness.data(),
             output_gradient_ptr         = host_sim_data->sa_bending_edges_gradients.data(),
             output_hessian_ptr          = host_sim_data->sa_bending_edges_hessians.data(),
             scaling                     = get_scene_params().get_bending_stiffness_scaling(),
             &hessian_blocks,
             &cgB](const uint eid)
            {
                uint4  edge        = sa_bending_edges[eid];
                float3 vert_pos[4] = {sa_x[edge[0]], sa_x[edge[1]], sa_x[edge[2]], sa_x[edge[3]]};


                const float area       = sa_bending_edges_rest_area[eid];
                const float stiff      = sa_bending_edges_stiffness[eid] * area * scaling;
                const float rest_angle = sa_bending_edges_rest_angle[eid];

                float3 gradients[4] = {Zero3, Zero3, Zero3, Zero3};
                float  angle =
                    BendingEnergy::compute_d_theta_d_x(vert_pos[0], vert_pos[1], vert_pos[2], vert_pos[3], gradients);
                const float delta_angle = angle - rest_angle;

                EigenFloat12x12 H = EigenFloat12x12::Zero();
                cgB.block<3, 1>(3 * edge[0], 0) -= stiff * delta_angle * float3_to_eigen3(gradients[0]);
                cgB.block<3, 1>(3 * edge[1], 0) -= stiff * delta_angle * float3_to_eigen3(gradients[1]);
                cgB.block<3, 1>(3 * edge[2], 0) -= stiff * delta_angle * float3_to_eigen3(gradients[2]);
                cgB.block<3, 1>(3 * edge[3], 0) -= stiff * delta_angle * float3_to_eigen3(gradients[3]);
                for (uint ii = 0; ii < 4; ii++)
                    for (uint jj = 0; jj < 4; jj++)
                    {
                        H.block<3, 3>(3 * ii, 3 * jj) =
                            stiff * float3x3_to_eigen3x3(outer_product(gradients[ii], gradients[jj]));
                    }
                hessian_blocks[eid] = {.indices = {edge[0], edge[1], edge[2], edge[3]}, .matrix = H};
            });
        convert_triplets_to_sparse_matrix(cgA, hessian_blocks);
    }

    // ABD Ground Collision
    if constexpr (false)
    {
        std::vector<EigenTripletBlock<4>> hessian_blocks(num_bodies);
        for (uint body_idx = 0; body_idx < host_sim_data->sa_affine_bodies.size(); body_idx++)
        {
            hessian_blocks[body_idx] = {.indices =
                                            {
                                                prefix + 4 * body_idx + 0,
                                                prefix + 4 * body_idx + 1,
                                                prefix + 4 * body_idx + 2,
                                                prefix + 4 * body_idx + 3,
                                            },
                                        .matrix = EigenFloat12x12::Zero()};
        }

        const float d_hat     = get_scene_params().d_hat;
        const float thickness = get_scene_params().thickness;
        CpuParallel::single_thread_for(
            0,
            host_sim_data->sa_affine_bodies.size(),
            [&](const uint body_idx)
            {
                if (get_scene_params().use_floor)
                {
                    const uint mesh_idx    = host_sim_data->sa_affine_bodies_mesh_id[body_idx];
                    const uint curr_prefix = host_mesh_data->prefix_num_verts[mesh_idx];
                    const uint next_prefix = host_mesh_data->prefix_num_verts[mesh_idx + 1];

                    for (uint vid = curr_prefix; vid < next_prefix; vid++)
                    {
                        float3 x_k  = host_sim_data->sa_x[vid];
                        float  diff = x_k.y - get_scene_params().floor.y;

                        if (diff < d_hat + thickness)
                        {
                            float  C      = d_hat + thickness - diff;
                            float3 normal = luisa::make_float3(0, 1, 0);
                            float  area   = host_mesh_data->sa_rest_vert_area[vid];
                            // float    area    = 1.0f;
                            float    stiff   = get_scene_params().stiffness_collision * area;
                            float    k1      = stiff * C;
                            float3   model_x = host_mesh_data->sa_scaled_model_x[vid];
                            float3   force   = stiff * C * normal;
                            float3x3 hessian = stiff * outer_product(normal, normal);

                            // Friction

                            auto J = AffineBodyDynamics::get_jacobian_dxdq(model_x);
                            cgB.block<12, 1>(body_idx * 12, 0) += J.transpose() * float3_to_eigen3(force);
                            hessian_blocks[body_idx].matrix += J.transpose() * float3x3_to_eigen3x3(hessian) * J;
                            // cgA.block<12, 12>(body_idx * 12, body_idx * 12) +=
                            //     J.transpose() * float3x3_to_eigen3x3(hessian) * J;

                            //  0   1   2   3
                            // t1   4   5   6
                            // t2  t5   7   8
                            // t3  t6  t8   9
                        }
                    }
                }
            });
        convert_triplets_to_sparse_matrix(cgA, hessian_blocks);
    }

    // ABD Inertia
    if constexpr (false)
    {
        const auto& abd_q       = host_sim_data->sa_affine_bodies_q;
        const auto& abd_q_tilde = host_sim_data->sa_affine_bodies_q_tilde;

        std::vector<EigenTripletBlock<4>> hessian_blocks(num_bodies);

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
                    float3_to_eigen3(abd_q[4 * body_idx + 0] - abd_q_tilde[4 * body_idx + 0]);
                delta.block<3, 1>(3, 0) =
                    float3_to_eigen3(abd_q[4 * body_idx + 1] - abd_q_tilde[4 * body_idx + 1]);
                delta.block<3, 1>(6, 0) =
                    float3_to_eigen3(abd_q[4 * body_idx + 2] - abd_q_tilde[4 * body_idx + 2]);
                delta.block<3, 1>(9, 0) =
                    float3_to_eigen3(abd_q[4 * body_idx + 3] - abd_q_tilde[4 * body_idx + 3]);

                EigenFloat12    gradient = h_2_inv * M * delta;
                EigenFloat12x12 hessian  = h_2_inv * M;

                // std::cout << "Body " << body_idx << " inertia gradient: " << std::endl
                //           << gradient << std::endl;
                // std::cout << "Body " << body_idx << " inertia hessian: " << std::endl
                //           << hessian << std::endl;

                hessian_blocks[body_idx] = {.indices = {prefix + 4 * body_idx + 0,
                                                        prefix + 4 * body_idx + 1,
                                                        prefix + 4 * body_idx + 2,
                                                        prefix + 4 * body_idx + 3},
                                            .matrix  = hessian};
                cgB.block<12, 1>(body_idx * 12, 0) -= gradient;
                // cgA.block<12, 12>(body_idx * 12, body_idx * 12) += hessian;
            });

        convert_triplets_to_sparse_matrix(cgA, hessian_blocks);
    }

    // Orthogonality
    if constexpr (false)
    {
        std::vector<EigenTripletBlock<4>> hessian_blocks(num_bodies);

        const float d_hat     = get_scene_params().d_hat;
        const float thickness = get_scene_params().thickness;
        CpuParallel::single_thread_for(
            0,
            host_sim_data->sa_affine_bodies.size(),
            [&](const uint body_idx)
            {
                // Orthogonality potential
                // if constexpr (false)
                {
                    float3x3 A = luisa::make_float3x3(host_sim_data->sa_affine_bodies_q[4 * body_idx + 1],
                                                      host_sim_data->sa_affine_bodies_q[4 * body_idx + 2],
                                                      host_sim_data->sa_affine_bodies_q[4 * body_idx + 3]);
                    // A          = luisa::transpose(A);

                    const float kappa = host_sim_data->sa_affine_bodies_kappa[body_idx];
                    const float V     = host_sim_data->sa_affine_bodies_volume[body_idx];

                    float stiff = kappa * V;
                    for (uint ii = 0; ii < 3; ii++)
                    {
                        float3 grad = (-1.0f) * A[ii];
                        for (uint jj = 0; jj < 3; jj++)
                        {
                            grad += dot_vec(A[ii], A[jj]) * A[jj];
                        }
                        // LUISA_INFO("grad of col {} = {}", ii, grad);
                        cgB.block<3, 1>(12 * body_idx + 3 + 3 * ii, 0) -= 4 * stiff * float3_to_eigen3(grad);
                        // body_force[1 + ii] -= 4 * stiff * g; // Force
                        // LUISA_INFO("Force of col {} = {}", 1 + ii, g);
                    }
                    EigenFloat12x12 local_H = EigenFloat12x12::Zero();
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
                            // cgA.block<3, 3>(12 * body_idx + 3 + 3 * ii, 12 * body_idx + 3 + 3 * jj) +=
                            //     4.0f * stiff * float3x3_to_eigen3x3(hessian);
                            local_H.block<3, 3>(3 + 3 * ii, 3 + 3 * jj) +=
                                4.0f * stiff * float3x3_to_eigen3x3(hessian);
                        }
                    }
                    hessian_blocks[body_idx] = {.indices = {prefix + 4 * body_idx + 0,
                                                            prefix + 4 * body_idx + 1,
                                                            prefix + 4 * body_idx + 2,
                                                            prefix + 4 * body_idx + 3},
                                                .matrix  = local_H};
                }
            });

        convert_triplets_to_sparse_matrix(cgA, hessian_blocks);
    }

    // Contact
    if constexpr (false)
    {
        narrow_phase_detector->download_narrowphase_list(stream);

        const auto& host_count = host_collision_data->narrow_phase_collision_count;
        const uint  num_pairs  = host_count.front();

        std::vector<VarientEigenTripletBlock> hessian_blocks;

        for (uint pair_idx = 0; pair_idx < num_pairs; pair_idx++)
        {
            auto pair = host_collision_data->narrow_phase_list[pair_idx];

            const uint4  indices        = pair.get_indices();
            const float4 weight         = pair.get_weight();
            const float3 normal         = pair.get_normal();
            const float  k1             = pair.get_k1();  // dBdD
            const float  k2             = pair.get_k2();  // ddBddD
            const auto   collision_type = pair.get_collision_type();
            float        D;

            float3 positions[4] = {
                host_sim_data->sa_x[indices[0]],
                host_sim_data->sa_x[indices[1]],
                host_sim_data->sa_x[indices[2]],
                host_sim_data->sa_x[indices[3]],
            };
            bool is_rigid[4] = {
                indices[0] >= prefix,
                indices[1] >= prefix,
                indices[2] >= prefix,
                indices[3] >= prefix,
            };

            if constexpr (false)
            {
                for (uint ii = 0; ii < 4; ii++)
                {
                    EigenFloat3 grad = float3_to_eigen3(k1 * weight[ii] * normal);
                    cgB.block<3, 1>(indices[ii] * 3, 0) += grad;
                    for (uint jj = 0; jj < 4; jj++)
                    {
                        const float3x3 hess = k2 * weight[ii] * weight[jj] * outer_product(normal, normal);
                        hessian_blocks.push_back({.left_indices  = {indices[ii]},
                                                  .right_indices = {indices[jj]},
                                                  .matrix        = float3x3_to_eigen3x3(hess)});
                    }
                }
            }

            // if constexpr (false)
            {
                for (uint ii = 0; ii < 4; ii++)
                {
                    std::vector<uint> left_indices;
                    std::vector<uint> right_indices;

                    EigenFloat3     grad = float3_to_eigen3(k1 * weight[ii] * normal);
                    Eigen::MatrixXf J1;  // = Eigen::Matrix<float, 3, 12>::Zero();
                    if (is_rigid[ii])
                    {
                        const float3 Xi   = host_mesh_data->sa_scaled_model_x[indices[ii]];
                        J1                = AffineBodyDynamics::get_jacobian_dxdq(Xi);
                        const uint body_i = host_sim_data->sa_vert_affine_bodies_id[indices[ii]];

                        cgB.block<12, 1>(3 * prefix + body_i * 12, 0) += J1.transpose() * grad;
                        left_indices = {
                            prefix + 4 * body_i + 0,
                            prefix + 4 * body_i + 1,
                            prefix + 4 * body_i + 2,
                            prefix + 4 * body_i + 3,
                        };
                    }
                    else
                    {
                        J1 = EigenFloat3x3::Identity();
                        cgB.block<3, 1>(indices[ii] * 3, 0) += grad;
                        left_indices = {indices[ii]};
                    }

                    for (uint jj = 0; jj < 4; jj++)
                    {
                        const float3x3 hess = k2 * weight[ii] * weight[jj] * outer_product(normal, normal);
                        Eigen::MatrixXf J2;
                        if (is_rigid[jj])
                        {
                            const float3 Xj   = host_mesh_data->sa_scaled_model_x[indices[jj]];
                            J2                = AffineBodyDynamics::get_jacobian_dxdq(Xj);
                            const uint body_j = host_sim_data->sa_vert_affine_bodies_id[indices[jj]];
                            right_indices     = {
                                prefix + 4 * body_j + 0,
                                prefix + 4 * body_j + 1,
                                prefix + 4 * body_j + 2,
                                prefix + 4 * body_j + 3,
                            };
                        }
                        else
                        {
                            J2            = EigenFloat3x3::Identity();
                            right_indices = {indices[jj]};
                        }
                        const Eigen::MatrixXf JtHJ = J1.transpose() * float3x3_to_eigen3x3(hess) * J2;

                        hessian_blocks.push_back(
                            {.left_indices = left_indices, .right_indices = right_indices, .matrix = JtHJ});
                        // LUISA_INFO("Host generate offdiag triplet ({}, {}): {}", left_indices, right_indices, hess);
                    }
                }
            }
        }
        // convert_triplets_to_sparse_matrix(cgA, hessian_blocks);
    }

    if constexpr (false)
    {
        stream << fn_reset_vector(sim_data->sa_cgB).dispatch(num_dof)
               << fn_reset_float3x3(sim_data->sa_cgA_diag).dispatch(num_dof);
        narrow_phase_detector->device_perPair_evaluate_gradient_hessian(stream,
                                                                        sim_data->sa_x,
                                                                        sim_data->sa_x,
                                                                        get_scene_params().d_hat,
                                                                        get_scene_params().thickness,
                                                                        sim_data->sa_vert_affine_bodies_id,
                                                                        mesh_data->sa_scaled_model_x,
                                                                        host_sim_data->num_verts_soft,
                                                                        sim_data->sa_cgB,
                                                                        sim_data->sa_cgA_diag);
        narrow_phase_detector->device_assemble_contact_triplet(
            stream, mesh_data->sa_scaled_model_x, host_sim_data->num_verts_soft);
        narrow_phase_detector->download_contact_triplet(stream);

        {
            std::vector<float3>   contact_force(num_dof);
            std::vector<float3x3> contact_diagA(num_dof);
            stream << sim_data->sa_cgB.copy_to(contact_force.data())
                   << sim_data->sa_cgA_diag.copy_to(contact_diagA.data()) << luisa::compute::synchronize();
            std::vector<EigenTripletBlock<1>> hessian_blocks(num_dof);
            for (uint vid = 0; vid < num_dof; vid++)
            {
                cgB.block<3, 1>(vid * 3, 0) += float3_to_eigen3(contact_force[vid]);
                hessian_blocks[vid] = {.indices = {vid}, .matrix = float3x3_to_eigen3x3(contact_diagA[vid])};
            }
            convert_triplets_to_sparse_matrix(cgA, hessian_blocks);
        }

        const uint triplet_count =
            host_collision_data->narrow_phase_collision_count[CollisionPair::CollisionCount::total_adj_verts_offset()];
        std::vector<MatrixTriplet3x3> hessian_blocks(triplet_count);
        CpuParallel::parallel_copy(host_collision_data->sa_cgA_contact_offdiag_triplet, hessian_blocks, triplet_count);
        Eigen::SparseMatrix<float> cgA_contact_offdiag(num_dof * 3, num_dof * 3);
        cgA_contact_offdiag.setZero();
        convert_triplets_to_sparse_matrix(cgA_contact_offdiag, hessian_blocks);
        cgA += cgA_contact_offdiag;
    }

    Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower> solver;  // Eigen::IncompleteCholesky<float>
    solver.setTolerance(1e-6f);
    solver.compute(cgA);

    if (solver.info() != Eigen::Success)
    {
        LUISA_ERROR("Eigen PCG decomposition failed!");
    }
    auto cgX = solver.solve(cgB).eval();
    CpuParallel::single_thread_for(0,
                                   host_sim_data->sa_cgX.size(),
                                   [&](const uint vid) {
                                       host_sim_data->sa_cgX[vid] = eigen3_to_float3(cgX.block<3, 1>(3 * vid, 0));
                                   });

    const float error = (cgB - cgA * cgX).norm();
    LUISA_INFO("  In non-linear iter {:2}, EigenSolve error = {:7.6f}, max_element(p) = {:6.5f}",
               get_scene_params().current_nonlinear_iter,
               error,
               fast_infinity_norm(host_sim_data->sa_cgX));
    if (std::isnan(error) || std::isinf(error))
    {
        LUISA_ERROR("NaN/INF detected in Eigen PCG solve!");
    }
}
void NewtonSolver::host_evaluete_stretch_spring()
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
void NewtonSolver::host_evaluete_stretch_face()
{
    CpuParallel::parallel_for(
        0,
        host_sim_data->sa_stretch_faces.size(),
        [sa_x                       = host_sim_data->sa_x.data(),
         sa_faces                   = host_sim_data->sa_stretch_faces.data(),
         sa_stretch_faces_rest_area = host_sim_data->sa_stretch_faces_rest_area.data(),
         sa_stretch_faces_Dm_inv    = host_sim_data->sa_stretch_faces_Dm_inv.data(),
         sa_stretch_faces_mu_lambda = host_sim_data->sa_stretch_faces_mu_lambda.data(),
         output_gradient_ptr        = host_sim_data->sa_stretch_faces_gradients.data(),
         output_hessian_ptr         = host_sim_data->sa_stretch_faces_hessians.data()](const uint fid)
        {
            uint3 face = sa_faces[fid];

            float3 vert_pos[3] = {sa_x[face[0]], sa_x[face[1]], sa_x[face[2]]};
            // float3   gradients[3] = {Zero3, Zero3, Zero3};
            // float3x3 hessians[9]  = {Zero3x3};
            float3x3 gradients;
            float9x9 hessians;
            float2x2 Dm_inv = sa_stretch_faces_Dm_inv[fid];
            float    area   = sa_stretch_faces_rest_area[fid];

            auto [mu_cloth, lambda_cloth] = sa_stretch_faces_mu_lambda[fid];

            StretchEnergy::compute_gradient_hessian(
                vert_pos[0], vert_pos[1], vert_pos[2], Dm_inv, mu_cloth, lambda_cloth, area, gradients, hessians);

            // Output
            {
                output_gradient_ptr[fid * 3 + 0] = gradients[0];
                output_gradient_ptr[fid * 3 + 1] = gradients[1];
                output_gradient_ptr[fid * 3 + 2] = gradients[2];
            }
            {
                // 0 1 2
                // 3 4 5
                // 6 7 8
                output_hessian_ptr[fid * 9 + 0] = hessians.block(0, 0);
                output_hessian_ptr[fid * 9 + 1] = hessians.block(1, 1);
                output_hessian_ptr[fid * 9 + 2] = hessians.block(2, 2);

                output_hessian_ptr[fid * 9 + 3] = hessians.block(1, 0);  // lower triangle
                output_hessian_ptr[fid * 9 + 4] = hessians.block(2, 0);
                output_hessian_ptr[fid * 9 + 5] = hessians.block(0, 1);
                output_hessian_ptr[fid * 9 + 6] = hessians.block(2, 1);
                output_hessian_ptr[fid * 9 + 7] = hessians.block(0, 2);
                output_hessian_ptr[fid * 9 + 8] = hessians.block(1, 2);

                // output_hessian_ptr[fid * 9 + 0] = hessians[0];
                // output_hessian_ptr[fid * 9 + 1] = hessians[4];
                // output_hessian_ptr[fid * 9 + 2] = hessians[8];

                // output_hessian_ptr[fid * 9 + 3] = hessians[1];
                // output_hessian_ptr[fid * 9 + 4] = hessians[2];
                // output_hessian_ptr[fid * 9 + 5] = hessians[3];
                // output_hessian_ptr[fid * 9 + 6] = hessians[5];
                // output_hessian_ptr[fid * 9 + 7] = hessians[6];
                // output_hessian_ptr[fid * 9 + 8] = hessians[7];
            }
        });
}
void NewtonSolver::host_evaluete_bending()
{
    CpuParallel::parallel_for(
        0,
        host_sim_data->sa_bending_edges.size(),
        [sa_x                        = host_sim_data->sa_x.data(),
         sa_bending_edges            = host_sim_data->sa_bending_edges.data(),
         sa_bending_edges_Q          = host_sim_data->sa_bending_edges_Q.data(),
         sa_bending_edges_rest_angle = host_sim_data->sa_bending_edges_rest_angle.data(),
         sa_bending_edges_rest_area  = host_sim_data->sa_bending_edges_rest_area.data(),
         sa_bending_edges_stiffness  = host_sim_data->sa_bending_edges_stiffness.data(),
         output_gradient_ptr         = host_sim_data->sa_bending_edges_gradients.data(),
         output_hessian_ptr          = host_sim_data->sa_bending_edges_hessians.data(),
         scaling = get_scene_params().get_bending_stiffness_scaling()](const uint eid)
        {
            uint4  edge         = sa_bending_edges[eid];
            float3 vert_pos[4]  = {sa_x[edge[0]], sa_x[edge[1]], sa_x[edge[2]], sa_x[edge[3]]};
            float3 gradients[4] = {Zero3, Zero3, Zero3, Zero3};

            if constexpr (true)  // Dehedral angle beding
            {
                const float rest_angle = sa_bending_edges_rest_angle[eid];
                const float angle =
                    BendingEnergy::compute_d_theta_d_x(vert_pos[0], vert_pos[1], vert_pos[2], vert_pos[3], gradients);
                const float delta_angle = angle - rest_angle;

                const float area                 = sa_bending_edges_rest_area[eid];
                const float stiff                = sa_bending_edges_stiffness[eid] * scaling * area;
                output_gradient_ptr[eid * 4 + 0] = stiff * delta_angle * gradients[0];
                output_gradient_ptr[eid * 4 + 1] = stiff * delta_angle * gradients[1];
                output_gradient_ptr[eid * 4 + 2] = stiff * delta_angle * gradients[2];
                output_gradient_ptr[eid * 4 + 3] = stiff * delta_angle * gradients[3];

                auto outer = [&gradients, stiff](uint ii, uint jj) -> float3x3
                {
                    // Use simple Gauss-Newton approximation
                    return outer_product(stiff * gradients[ii], gradients[jj]);
                };

                output_hessian_ptr[eid * 16 + 0] = outer(0, 0);
                output_hessian_ptr[eid * 16 + 1] = outer(1, 1);
                output_hessian_ptr[eid * 16 + 2] = outer(2, 2);
                output_hessian_ptr[eid * 16 + 3] = outer(3, 3);

                uint idx = 4;
                for (uint ii = 0; ii < 4; ii++)
                {
                    for (uint jj = 0; jj < 4; jj++)
                    {
                        if (ii != jj)
                        {
                            output_hessian_ptr[eid * 16 + idx] = outer(ii, jj);
                            idx += 1;
                        }
                    }
                }
            }
            if constexpr (false)  // Quadratic bending
            {
                float4x4 m_Q = sa_bending_edges_Q[eid];

                for (uint ii = 0; ii < 4; ii++)
                {
                    for (uint jj = 0; jj < 4; jj++)
                    {
                        gradients[ii] += m_Q[ii][jj] * vert_pos[jj];  // -Qx
                    }
                    gradients[ii] = scaling * gradients[ii];
                }

                output_gradient_ptr[eid * 4 + 0] = gradients[0];
                output_gradient_ptr[eid * 4 + 1] = gradients[1];
                output_gradient_ptr[eid * 4 + 2] = gradients[2];
                output_gradient_ptr[eid * 4 + 3] = gradients[3];
                auto hess                        = scaling * luisa::make_float3x3(1.0f);
                output_hessian_ptr[eid * 16 + 0] = m_Q[0][0] * hess;
                output_hessian_ptr[eid * 16 + 1] = m_Q[1][1] * hess;
                output_hessian_ptr[eid * 16 + 2] = m_Q[2][2] * hess;
                output_hessian_ptr[eid * 16 + 3] = m_Q[3][3] * hess;
                uint idx                         = 4;
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
    // Assemble material forces and stiffness matrix
    {
        auto assembly_template =
            [sa_vert_adj_material_force_verts_csr = host_sim_data->sa_vert_adj_material_force_verts_csr.data(),
             sa_cgB      = host_sim_data->sa_cgB.data(),
             sa_cgA_diag = host_sim_data->sa_cgA_diag.data(),
             sa_cgA_offdiag_triplet =
                 host_sim_data->sa_cgA_fixtopo_offdiag_triplet.data()](const uint N,
                                                                       const uint vid,
                                                                       const uint* vert_adj_constraints_csr,
                                                                       const auto&   constaints,
                                                                       const float3* constaint_gradients,
                                                                       const float3x3* constaint_hessians,
                                                                       const ushort* constaint_offsets_in_adjlist)
        {
            const uint curr_prefix = sa_vert_adj_material_force_verts_csr[vid];
            const uint next_prefix = sa_vert_adj_material_force_verts_csr[vid + 1];

            const uint curr_prefix_bending = vert_adj_constraints_csr[vid];
            const uint next_prefix_bending = vert_adj_constraints_csr[vid + 1];

            for (uint j = curr_prefix_bending; j < next_prefix_bending; j++)
            {
                const uint adj_eid = vert_adj_constraints_csr[j];
                const auto edge    = constaints[adj_eid];
                uint       offset  = -1u;
                for (uint k = 0; k < N; k++)
                {
                    if (vid == edge[k])
                    {
                        offset = k;
                    }
                }
                LUISA_ASSERT(offset != -1u, "Error in assembly: offset not found.");

                const float3   grad      = constaint_gradients[adj_eid * N + offset];
                const float3x3 diag_hess = constaint_hessians[adj_eid * (N * N) + offset];

                buffer_add(sa_cgB, vid, -grad);
                buffer_add(sa_cgA_diag, vid, diag_hess);

                const uint N_off = N - 1;
                for (uint ii = 0; ii < N_off; ii++)  // For each off-diagonal in curr row
                {
                    float3x3 offdiag_hess = constaint_hessians[adj_eid * (N * N) + N + offset * N_off + ii];
                    uint offdiag_offset =
                        constaint_offsets_in_adjlist[adj_eid * (N * N_off) + offset * N_off + ii];
                    auto triplet = sa_cgA_offdiag_triplet[curr_prefix + offdiag_offset];
                    add_triplet_matrix(triplet, offdiag_hess);
                    sa_cgA_offdiag_triplet[curr_prefix + offdiag_offset] = triplet;
                }
            };
        };


        if (host_sim_data->sa_stretch_springs.size() != 0 && false)
            CpuParallel::parallel_for(
                0,
                host_sim_data->num_verts_soft,
                [vert_adj_constraints_csr = host_sim_data->sa_vert_adj_stretch_springs_csr.data(),
                 constaints               = host_sim_data->sa_stretch_springs.data(),
                 constaint_gradients      = host_sim_data->sa_stretch_springs_gradients.data(),
                 constaint_hessians       = host_sim_data->sa_stretch_springs_hessians.data(),
                 constaint_offsets_in_adjlist = host_sim_data->sa_stretch_springs_offsets_in_adjlist.data(),
                 assembly_template](const uint vid)
                {
                    assembly_template(2, vid, vert_adj_constraints_csr, constaints, constaint_gradients, constaint_hessians, constaint_offsets_in_adjlist);
                });

        if (host_sim_data->sa_stretch_faces.size() != 0)
            CpuParallel::parallel_for(
                0,
                host_sim_data->num_verts_soft,
                [vert_adj_constraints_csr = host_sim_data->sa_vert_adj_stretch_faces_csr.data(),
                 constaints               = host_sim_data->sa_stretch_faces.data(),
                 constaint_gradients      = host_sim_data->sa_stretch_faces_gradients.data(),
                 constaint_hessians       = host_sim_data->sa_stretch_faces_hessians.data(),
                 constaint_offsets_in_adjlist = host_sim_data->sa_stretch_faces_offsets_in_adjlist.data(),
                 assembly_template](const uint vid)
                {
                    assembly_template(3, vid, vert_adj_constraints_csr, constaints, constaint_gradients, constaint_hessians, constaint_offsets_in_adjlist);
                });

        if (host_sim_data->sa_bending_edges.size() != 0)
            CpuParallel::parallel_for(
                0,
                host_sim_data->num_verts_soft,
                [vert_adj_constraints_csr = host_sim_data->sa_vert_adj_bending_edges_csr.data(),
                 constaints               = host_sim_data->sa_bending_edges.data(),
                 constaint_gradients      = host_sim_data->sa_bending_edges_gradients.data(),
                 constaint_hessians       = host_sim_data->sa_bending_edges_hessians.data(),
                 constaint_offsets_in_adjlist = host_sim_data->sa_bending_edges_offsets_in_adjlist.data(),
                 assembly_template](const uint vid)
                {
                    assembly_template(4, vid, vert_adj_constraints_csr, constaints, constaint_gradients, constaint_hessians, constaint_offsets_in_adjlist);
                });

        if (host_sim_data->sa_affine_bodies.size() != 0)
            CpuParallel::parallel_for(
                0,
                host_sim_data->sa_affine_bodies.size() * 4,
                [vert_adj_constraints_csr = host_sim_data->sa_vert_adj_affine_bodies_csr.data(),
                 constaints               = host_sim_data->sa_affine_bodies.data(),
                 constaint_gradients      = host_sim_data->sa_affine_bodies_gradients.data(),
                 constaint_hessians       = host_sim_data->sa_affine_bodies_hessians.data(),
                 constaint_offsets_in_adjlist = host_sim_data->sa_affine_bodies_offsets_in_adjlist.data(),
                 prefix = host_sim_data->num_verts_soft,
                 assembly_template](const uint block_idx)
                {
                    assembly_template(4,
                                      prefix + block_idx,
                                      vert_adj_constraints_csr,
                                      constaints,
                                      constaint_gradients,
                                      constaint_hessians,
                                      constaint_offsets_in_adjlist);
                });
    }
}

// Device functions
void NewtonSolver::device_broadphase_ccd(luisa::compute::Stream& stream)
{
    narrow_phase_detector->reset_broadphase_count(stream);

    lbvh_face->update_face_tree_leave_aabb(stream,
                                           sim_data->sa_contact_active_verts_offset,
                                           sim_data->sa_x_iter_start,
                                           sim_data->sa_x,
                                           mesh_data->sa_faces);
    lbvh_face->refit(stream);
    lbvh_face->broad_phase_query_from_verts(
        stream,
        sim_data->sa_x_iter_start,
        sim_data->sa_x,
        collision_data->broad_phase_collision_count.view(collision_data->get_vf_count_offset(), 1),
        collision_data->broad_phase_list_vf,
        sim_data->sa_contact_active_verts_d_hat,
        sim_data->sa_contact_active_verts_offset);

    lbvh_edge->update_edge_tree_leave_aabb(stream,
                                           sim_data->sa_contact_active_verts_offset,
                                           sim_data->sa_x_iter_start,
                                           sim_data->sa_x,
                                           mesh_data->sa_edges);
    lbvh_edge->refit(stream);
    lbvh_edge->broad_phase_query_from_edges(
        stream,
        sim_data->sa_x_iter_start,
        sim_data->sa_x,
        mesh_data->sa_edges,
        collision_data->broad_phase_collision_count.view(collision_data->get_ee_count_offset(), 1),
        collision_data->broad_phase_list_ee,
        sim_data->sa_contact_active_verts_d_hat,
        sim_data->sa_contact_active_verts_offset);
}
void NewtonSolver::device_broadphase_dcd(luisa::compute::Stream& stream)
{
    lbvh_face->update_face_tree_leave_aabb(
        stream, sim_data->sa_contact_active_verts_offset, sim_data->sa_x, sim_data->sa_x, mesh_data->sa_faces);
    lbvh_face->refit(stream);
    lbvh_face->broad_phase_query_from_verts(
        stream,
        sim_data->sa_x,
        sim_data->sa_x,
        collision_data->broad_phase_collision_count.view(collision_data->get_vf_count_offset(), 1),
        collision_data->broad_phase_list_vf,
        sim_data->sa_contact_active_verts_d_hat,
        sim_data->sa_contact_active_verts_offset);

    lbvh_edge->update_edge_tree_leave_aabb(
        stream, sim_data->sa_contact_active_verts_offset, sim_data->sa_x, sim_data->sa_x, mesh_data->sa_edges);
    lbvh_edge->refit(stream);
    lbvh_edge->broad_phase_query_from_edges(
        stream,
        sim_data->sa_x,
        sim_data->sa_x,
        mesh_data->sa_edges,
        collision_data->broad_phase_collision_count.view(collision_data->get_ee_count_offset(), 1),
        collision_data->broad_phase_list_ee,
        sim_data->sa_contact_active_verts_d_hat,
        sim_data->sa_contact_active_verts_offset);
}
void NewtonSolver::device_narrowphase_ccd(luisa::compute::Stream& stream)
{
    const float thickness = get_scene_params().thickness;
    const float d_hat     = get_scene_params().d_hat;
    // narrow_phase_detector->reset_narrowphase_count(stream);
    narrow_phase_detector->reset_toi(stream);

    narrow_phase_detector->vf_ccd_query(stream,
                                        sim_data->sa_x_iter_start,
                                        sim_data->sa_x_iter_start,
                                        sim_data->sa_x,
                                        sim_data->sa_x,
                                        mesh_data->sa_faces,
                                        sim_data->sa_contact_active_verts_d_hat,
                                        sim_data->sa_contact_active_verts_offset);

    narrow_phase_detector->ee_ccd_query(stream,
                                        sim_data->sa_x_iter_start,
                                        sim_data->sa_x_iter_start,
                                        sim_data->sa_x,
                                        sim_data->sa_x,
                                        mesh_data->sa_edges,
                                        mesh_data->sa_edges,
                                        sim_data->sa_contact_active_verts_d_hat,
                                        sim_data->sa_contact_active_verts_offset);
}
void NewtonSolver::device_narrowphase_dcd(luisa::compute::Stream& stream)
{
    const float thickness = get_scene_params().thickness;
    const float d_hat     = get_scene_params().d_hat;
    const float kappa     = get_scene_params().stiffness_collision;

    narrow_phase_detector->vf_dcd_query_repulsion(stream,
                                                  sim_data->sa_x,
                                                  sim_data->sa_x,
                                                  mesh_data->sa_rest_x,
                                                  mesh_data->sa_rest_x,
                                                  mesh_data->sa_rest_vert_area,
                                                  mesh_data->sa_rest_face_area,
                                                  mesh_data->sa_faces,
                                                  sim_data->sa_vert_affine_bodies_id,
                                                  sim_data->sa_vert_affine_bodies_id,
                                                  sim_data->sa_contact_active_verts_d_hat,
                                                  sim_data->sa_contact_active_verts_offset,
                                                  kappa);

    narrow_phase_detector->ee_dcd_query_repulsion(stream,
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
                                                  sim_data->sa_contact_active_verts_d_hat,
                                                  sim_data->sa_contact_active_verts_offset,
                                                  kappa);
}
void NewtonSolver::device_update_contact_list(luisa::compute::Stream& stream)
{
    narrow_phase_detector->reset_broadphase_count(stream);
    narrow_phase_detector->reset_narrowphase_count(stream);
    narrow_phase_detector->reset_pervert_collision_count(stream);

    if (get_scene_params().use_self_collision)
        device_broadphase_dcd(stream);

    narrow_phase_detector->download_broadphase_collision_count(stream);

    if (get_scene_params().use_self_collision)
        device_narrowphase_dcd(stream);

    narrow_phase_detector->download_narrowphase_collision_count(stream);
    narrow_phase_detector->construct_pervert_adj_list(
        stream, sim_data->sa_vert_affine_bodies_id, host_sim_data->num_verts_soft);
}
void NewtonSolver::device_ccd_line_search(luisa::compute::Stream& stream)
{
    device_broadphase_ccd(stream);

    narrow_phase_detector->download_broadphase_collision_count(stream);

    device_narrowphase_ccd(stream);
}
void NewtonSolver::device_compute_contact_energy(luisa::compute::Stream& stream, std::map<std::string, double>& energy_list)
{
    // stream << sim_data->sa_x.copy_from(sa_x.data());
    const float thickness = get_scene_params().thickness;
    const float d_hat     = get_scene_params().d_hat;
    const float kappa     = get_scene_params().stiffness_collision;

    narrow_phase_detector->reset_energy(stream);
    narrow_phase_detector->compute_contact_energy_from_iter_start_list(stream,
                                                                       sim_data->sa_x,
                                                                       sim_data->sa_x,
                                                                       mesh_data->sa_rest_x,
                                                                       mesh_data->sa_rest_x,
                                                                       mesh_data->sa_rest_vert_area,
                                                                       mesh_data->sa_rest_face_area,
                                                                       mesh_data->sa_faces,
                                                                       d_hat,
                                                                       thickness,
                                                                       kappa);

    auto contact_energy = narrow_phase_detector->download_energy(stream);
    energy_list.insert(std::make_pair("Contact", contact_energy));
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
    // const uint aligned_diaptch_count = get_dispatch_threads(num_pairs * 12, 256);
    // stream << fn_pcg_spmv_offdiag_block_rbk(collision_data->sa_cgA_contact_offdiag_triplet, input_ptr, output_ptr)
    //               .dispatch(aligned_diaptch_count);

    const auto& host_count      = host_collision_data->narrow_phase_collision_count;
    const uint  reduced_triplet = host_count[CollisionPair::CollisionCount::total_adj_verts_offset()];
    const uint  aligned_diaptch_count = get_dispatch_threads(reduced_triplet, 256);
    stream << fn_pcg_spmv_offdiag_block_rbk(collision_data->sa_cgA_contact_offdiag_triplet, input_ptr, output_ptr)
                  .dispatch(aligned_diaptch_count);

    // narrow_phase_detector->device_perVert_spmv(stream, input_ptr, output_ptr);
    // narrow_phase_detector->device_perPair_spmv(stream, input_ptr, output_ptr);
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
        if constexpr (false)
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
        auto fn_SpMV_reduce_by_key =
            [&](const std::vector<MatrixTriplet3x3>& sa_cgA_offdiag_triplet, const uint gridDim, const uint triplet_count)
        {
            CpuParallel::parallel_for_each_core(
                0,
                gridDim,
                [&](const uint blockIdx)
                {
                    const uint blockDim    = 256;
                    const uint blockPrefix = blockIdx * blockDim;

                    float3 sum_contrib = luisa::make_float3(0.0f);
                    for (uint triplet_idx = blockPrefix;
                         triplet_idx < min_scalar(blockPrefix + blockDim, triplet_count);
                         triplet_idx++)
                    {
                        auto       triplet         = sa_cgA_offdiag_triplet[triplet_idx];
                        const uint vid             = triplet.get_row_idx();
                        const uint adj_vid         = triplet.get_col_idx();
                        const uint matrix_property = triplet.get_matrix_property();

                        float3 contrib = luisa::make_float3(0.0f);
                        if (MatrixTriplet::is_valid(matrix_property))
                        {
                            const float3x3 mat   = read_triplet_matrix(triplet);
                            const float3   input = input_ptr[adj_vid];
                            contrib              = mat * input;
                            sum_contrib += contrib;
                            // output_ptr[vid] += contrib;

                            // if (get_scene_params().current_pcg_it == 0)
                            //     LUISA_INFO("Triplet {} : vid {}, adj_vid {}, is_first {}, is_last {}, contrib {}, sum_contrib {}",
                            //                triplet_idx,
                            //                vid,
                            //                adj_vid,
                            //                MatrixTriplet::is_first_col_in_row(matrix_property),
                            //                MatrixTriplet::is_last_col_in_row(matrix_property),
                            //                contrib,
                            //                sum_contrib);
                        };
                        // continue;
                        if (MatrixTriplet::is_last_col_in_row(matrix_property))
                        {
                            // if (get_scene_params().current_pcg_it == 0)
                            //     LUISA_INFO("Triplet {} try to add to {}, with {}", triplet_idx, vid, sum_contrib);

                            if (MatrixTriplet::write_use_atomic(matrix_property))
                            {
                                CpuParallel::spin_atomic<float3>::fetch_add(output_ptr[vid], sum_contrib);
                                // auto atomic_view = (CpuParallel::spin_atomic<float3>*)(&output_ptr[vid]);
                                // output_ptr[vid] += sum_contrib;
                            }
                            else
                            {
                                output_ptr[vid] += sum_contrib;
                            };
                            sum_contrib = luisa::make_float3(0.0f);
                        };
                    }
                });
        };

        fn_SpMV_reduce_by_key(host_sim_data->sa_cgA_fixtopo_offdiag_triplet,
                              get_dispatch_block(sim_data->sa_cgA_fixtopo_offdiag_triplet.size(), 256),
                              sim_data->sa_cgA_fixtopo_offdiag_triplet.size());

        const auto& host_count     = host_collision_data->narrow_phase_collision_count;
        const uint reduced_triplet = host_count[CollisionPair::CollisionCount::total_adj_verts_offset()];
        fn_SpMV_reduce_by_key(host_collision_data->sa_cgA_contact_offdiag_triplet,
                              get_dispatch_block(reduced_triplet, 256),
                              reduced_triplet);
    }
}
void NewtonSolver::host_solve_eigen(luisa::compute::Stream& stream, std::function<double()> func_compute_energy)
{

    const uint                 num_dof = host_sim_data->num_dof;
    Eigen::SparseMatrix<float> cgA(num_dof * 3, num_dof * 3);
    Eigen::VectorXf            cgB(num_dof * 3, 1);
    cgA.setZero();
    cgB.setZero();

    for (uint vid = 0; vid < num_dof; vid++)
    {
        cgB.block<3, 1>(vid * 3, 0) = float3_to_eigen3(host_sim_data->sa_cgB[vid]);
    }
    std::vector<MatrixTriplet3x3> new_triplet;
    {
        // Diag part
        for (uint vid = 0; vid < num_dof; vid++)
        {
            new_triplet.push_back(
                make_matrix_triplet(vid, vid, MatrixTriplet::is_valid(), host_sim_data->sa_cgA_diag[vid]));
        }
        // Material hessian off-diag part
        for (uint i = 0; i < host_sim_data->sa_cgA_fixtopo_offdiag_triplet.size(); i++)
        {
            new_triplet.push_back(host_sim_data->sa_cgA_fixtopo_offdiag_triplet[i]);
        }
        // Contact hessian off-diag part
        const auto& host_count     = host_collision_data->narrow_phase_collision_count;
        const uint reduced_triplet = host_count[CollisionPair::CollisionCount::total_adj_verts_offset()];
        for (uint i = 0; i < reduced_triplet; i++)
        {
            new_triplet.push_back(host_collision_data->sa_cgA_contact_offdiag_triplet[i]);
        }
    }
    convert_triplets_to_sparse_matrix(cgA, new_triplet);


    Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower> solver;  // Eigen::IncompleteCholesky<float>
    solver.setTolerance(1e-6f);
    solver.compute(cgA);

    if (solver.info() != Eigen::Success)
    {
        LUISA_ERROR("Eigen PCG decomposition failed!");
    }
    auto cgX = solver.solve(cgB).eval();
    // auto cgX = eigen_pcg(eigen_A, eigen_b);

    // std::cout << "Eigen A = \n" << eigen_A << std::endl;
    // std::cout << "Eigen b = \n" << eigen_b.transpose() << std::endl;
    // std::cout << "Eigen result = " << eigen_dx.transpose() << std::endl;
    for (uint vid = 0; vid < num_dof; vid++)
    {
        host_sim_data->sa_cgX[vid] = eigen3_to_float3(cgX.block<3, 1>(vid * 3, 0));
    }

    const float infinity_norm = fast_infinity_norm(host_sim_data->sa_cgX);
    if (luisa::isnan(infinity_norm) || luisa::isinf(infinity_norm))
    {
        LUISA_ERROR("cgX exist NAN/INF value : {}", infinity_norm);
    }
    LUISA_INFO("  In non-linear iter {:2}, EigenSolve error = {:7.6f}, max_element(p) = {:6.5f}",
               get_scene_params().current_nonlinear_iter,
               (cgB - cgA * cgX).norm(),
               infinity_norm);
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
    lcs::SolverInterface::physics_step_prev_operation();


    constexpr bool use_eigen          = ConjugateGradientSolver::use_eigen;
    constexpr bool use_upper_triangle = ConjugateGradientSolver::use_upper_triangle;

    const float thickness = get_scene_params().thickness;
    const float d_hat     = get_scene_params().d_hat;

    auto update_contact_set = [&]()
    {
        stream << sim_data->sa_x.copy_from(host_sim_data->sa_x.data());

        device_update_contact_list(stream);
        // narrow_phase_detector->download_narrowphase_list(stream);
        // narrow_phase_detector->download_pervert_adjacent_list(stream);
        narrow_phase_detector->device_sort_contact_triplet(stream);
    };
    auto evaluate_contact = [&]()
    {
        stream << sim_data->sa_cgB.copy_from(host_sim_data->sa_cgB.data())
               << sim_data->sa_cgA_diag.copy_from(host_sim_data->sa_cgA_diag.data());

        narrow_phase_detector->device_perPair_evaluate_gradient_hessian(stream,
                                                                        sim_data->sa_x,
                                                                        sim_data->sa_x,
                                                                        d_hat,
                                                                        thickness,
                                                                        sim_data->sa_vert_affine_bodies_id,
                                                                        mesh_data->sa_scaled_model_x,
                                                                        host_sim_data->num_verts_soft,
                                                                        sim_data->sa_cgB,
                                                                        sim_data->sa_cgA_diag);
        narrow_phase_detector->device_assemble_contact_triplet(
            stream, mesh_data->sa_scaled_model_x, host_sim_data->num_verts_soft);
        narrow_phase_detector->download_contact_triplet(stream);

        stream << sim_data->sa_cgB.copy_to(host_sim_data->sa_cgB.data())
               << sim_data->sa_cgA_diag.copy_to(host_sim_data->sa_cgA_diag.data())
               << luisa::compute::synchronize();
    };
    auto ccd_get_toi = [&]() -> float
    {
        for (uint vid = 0; vid < host_sim_data->sa_x.size(); vid++)
        {
            float3 delta = host_sim_data->sa_x[vid] - host_sim_data->sa_x_iter_start[vid];
            if (length(delta) > 1.0f)
            {
                LUISA_ERROR("Large delta detected at vert {} : delta = {}, start = {}, end = {}",
                            vid,
                            delta,
                            host_sim_data->sa_x_iter_start[vid],
                            host_sim_data->sa_x[vid]);
            }
        }

        stream << sim_data->sa_x_iter_start.copy_from(host_sim_data->sa_x_iter_start.data())
               << sim_data->sa_x.copy_from(host_sim_data->sa_x.data());

        device_ccd_line_search(stream);

        float toi = narrow_phase_detector->get_global_toi(stream);
        return toi;  // 0.9f * toi
        // return 1.0f;
    };

    auto compute_energy_interface = [&]()
    {
        stream << sim_data->sa_x.copy_from(host_sim_data->sa_x.data());
        if (host_sim_data->num_verts_soft != 0)
        {
            stream << sim_data->sa_x_tilde.copy_from(host_sim_data->sa_x_tilde.data());
        }
        if (host_sim_data->num_affine_bodies != 0)
        {
            stream
                << sim_data->sa_affine_bodies_q.copy_from(host_sim_data->sa_affine_bodies_q.data())
                << sim_data->sa_affine_bodies_q_tilde.copy_from(host_sim_data->sa_affine_bodies_q_tilde.data());
        }

        std::map<std::string, double> energy_list;
        device_compute_elastic_energy(stream, energy_list);
        device_compute_contact_energy(stream, energy_list);

        auto total_energy = std::accumulate(energy_list.begin(),
                                            energy_list.end(),
                                            0.0,
                                            [](double sum, const auto& pair) { return sum + pair.second; });
        if (get_scene_params().print_system_energy)
        {
            for (const auto& pair : energy_list)
            {
                LUISA_INFO("Energy {} = {}", pair.first, pair.second);
            }
        }
        for (const auto& pair : energy_list)
        {
            if (is_nan_scalar(pair.second) || is_inf_scalar(pair.second))
            {
                LUISA_ERROR("{} energy is not valid : {}", pair.first, pair.second);
            }
        }
        return total_energy;
    };
    auto pcg_spmv_interface = [&](const std::vector<float3>& input_ptr, std::vector<float3>& output_ptr) -> void
    {
        //
        host_SpMV(stream, input_ptr, output_ptr);
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
            pcg_solver->host_solve(stream, pcg_spmv_interface, compute_energy_interface);
        }
    };

    const float substep_dt            = lcs::get_scene_params().get_substep_dt();
    const bool  use_energy_linesearch = get_scene_params().use_energy_linesearch;
    const bool  use_ccd_linesearch    = get_scene_params().use_ccd_linesearch;

    // Init LBVH
    {
        stream << sim_data->sa_x_step_start.copy_from(host_sim_data->sa_x_step_start.data());
        lbvh_face->reduce_face_tree_aabb(stream, sim_data->sa_x_step_start, mesh_data->sa_faces);
        lbvh_edge->reduce_edge_tree_aabb(stream, sim_data->sa_x_step_start, mesh_data->sa_edges);
        lbvh_face->construct_tree(stream);
        lbvh_edge->construct_tree(stream);
        stream << luisa::compute::synchronize();
    }
    // for (uint substep = 0; substep < get_scene_params().num_substep; substep++)
    {
        LUISA_INFO("=== In frame {} ===", get_scene_params().current_frame);

        host_predict_position();

        CpuParallel::parallel_copy(host_sim_data->sa_x_step_start, host_sim_data->sa_x_iter_start);  // For CCD line search

        // double barrier_nergy = compute_barrier_energy_from_broadphase_list();
        double prev_state_energy = Float_max;

        // for (uint iter = 0; iter < get_scene_params().nonlinear_iter_count; iter++)
        uint iter                      = 0;
        bool direchlet_point_converged = false;
        for (iter = 0; iter < 100; iter++)
        {
            if (iter >= get_scene_params().nonlinear_iter_count && direchlet_point_converged)
            {
                break;
            }
            get_scene_params().current_nonlinear_iter = iter;

            host_reset_cgB_cgX_diagA();

            host_reset_off_diag();

            update_contact_set();

            // if (iter == 0) // Always refresh for collision count is variant
            if (use_energy_linesearch)
            {
                prev_state_energy = compute_energy_interface();
            }
            if constexpr (true)
            {
                host_evaluate_inertia();

                host_evaluate_ground_collision();

                host_evaluate_orthogonality();

                host_evaluete_stretch_spring();

                host_evaluete_stretch_face();

                host_evaluete_bending();

                host_material_energy_assembly();

                evaluate_contact();

                // host_evaluate_dirichlet();

                linear_solver_interface();  // Solve Ax=b
            }
            else
            {
                host_test_dynamics(stream);
            }

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

            // Check dirichlet point target
            {
                const float direchlet_max_delta = CpuParallel::parallel_for_and_reduce(
                    0,
                    host_sim_data->sa_x.size(),
                    [&](const uint vid)
                    {
                        if (host_mesh_data->sa_is_fixed[vid])
                        {
                            float3 delta = host_sim_data->sa_x[vid] - host_sim_data->sa_x_tilde[vid];
                            return luisa::length(delta);
                        }
                        return 0.0f;  // Non-fixed point
                    },
                    [](const float left, const float right) { return max_scalar(left, right); },
                    -1e9f);
                direchlet_point_converged = direchlet_max_delta < 1e-3;
                if (!direchlet_point_converged)
                {
                    LUISA_INFO("  In non-linear iter {:2}: Dirichlet point not converged, max delta = {}", iter, direchlet_max_delta);
                }
            }

            if (use_energy_linesearch)
            {
                // Energy after CCD or just solving Axb
                auto curr_energy = compute_energy_interface();
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

                    curr_energy = compute_energy_interface();
                    LUISA_INFO("     Line search {} : alpha = {:6.5f}, energy = {:12.10f}", line_search_count, alpha, curr_energy);

                    if (alpha < 1e-4)
                    {
                        LUISA_ERROR("  Line search failed, energy = {}, prev state energy = {}", curr_energy, prev_state_energy);
                    }
                    line_search_count++;
                }
                prev_state_energy = curr_energy;  // E_prev = E
            }

            CpuParallel::parallel_copy(host_sim_data->sa_x, host_sim_data->sa_x_iter_start);  // x_prev = x
            CpuParallel::parallel_copy(host_sim_data->sa_affine_bodies_q,
                                       host_sim_data->sa_affine_bodies_q_iter_start);  // q_prev = q
        }
        host_update_velocity();
    }

    // Output
    lcs::SolverInterface::physics_step_post_operation();
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

    // Read frame start position and velocity
    lcs::SolverInterface::physics_step_prev_operation();

    // Upload to GPU
    stream << sim_data->sa_x_step_start.copy_from(host_sim_data->sa_x_step_start.data())
           << sim_data->sa_x.copy_from(host_sim_data->sa_x_step_start.data())
           << sim_data->sa_v.copy_from(host_sim_data->sa_v.data());

    if (host_sim_data->num_affine_bodies != 0)
    {
        stream << sim_data->sa_affine_bodies_q_step_start.copy_from(
            host_sim_data->sa_affine_bodies_q_step_start.data())
               << sim_data->sa_affine_bodies_q.copy_from(host_sim_data->sa_affine_bodies_q.data())
               << sim_data->sa_affine_bodies_q_v.copy_from(host_sim_data->sa_affine_bodies_q_v.data());
    }
    stream << luisa::compute::synchronize();

    const uint  num_substep          = lcs::get_scene_params().num_substep;
    const uint  nonlinear_iter_count = lcs::get_scene_params().nonlinear_iter_count;
    const float substep_dt           = lcs::get_scene_params().get_substep_dt();

    auto device_apply_dx = [&](const float alpha)
    {
        if (sim_data->num_verts_soft != 0)
        {
            stream << fn_apply_dx(alpha).dispatch(host_sim_data->num_verts_soft);
        }
        if (sim_data->num_affine_bodies != 0)
        {
            stream << fn_apply_dq(alpha, sim_data->num_verts_soft).dispatch(sim_data->num_affine_bodies * 4);
            stream << fn_apply_dx_affine_bodies(alpha, host_sim_data->num_verts_soft).dispatch(sim_data->num_verts_rigid);
        }
    };

    const float thickness          = get_scene_params().thickness;
    const float d_hat              = get_scene_params().d_hat;
    auto        update_contact_set = [&]()
    {
        device_update_contact_list(stream);
        narrow_phase_detector->device_sort_contact_triplet(stream);
    };
    auto evaluate_contact = [&]()
    {
        narrow_phase_detector->device_perPair_evaluate_gradient_hessian(stream,
                                                                        sim_data->sa_x,
                                                                        sim_data->sa_x,
                                                                        d_hat,
                                                                        thickness,
                                                                        sim_data->sa_vert_affine_bodies_id,
                                                                        mesh_data->sa_scaled_model_x,
                                                                        host_sim_data->num_verts_soft,
                                                                        sim_data->sa_cgB,
                                                                        sim_data->sa_cgA_diag);

        narrow_phase_detector->device_assemble_contact_triplet(
            stream, mesh_data->sa_scaled_model_x, host_sim_data->num_verts_soft);
    };
    auto ccd_get_toi = [&]() -> float
    {
        device_ccd_line_search(stream);
        float toi = narrow_phase_detector->get_global_toi(stream);
        return toi;  // 0.9f * toi
    };

    const uint num_verts = host_mesh_data->num_verts;

    auto pcg_spmv_interface = [&](const luisa::compute::Buffer<float3>& input_ptr,
                                  luisa::compute::Buffer<float3>&       output_ptr) -> void
    {
        //
        device_SpMV(stream, input_ptr, output_ptr);
    };
    auto compute_energy_interface = [&]()
    {
        std::map<std::string, double> energy_list;
        device_compute_elastic_energy(stream, energy_list);
        device_compute_contact_energy(stream, energy_list);

        auto total_energy = std::accumulate(energy_list.begin(),
                                            energy_list.end(),
                                            0.0,
                                            [](double sum, const auto& pair) { return sum + pair.second; });
        if (get_scene_params().print_system_energy)
        {
            for (const auto& pair : energy_list)
            {
                LUISA_INFO("Energy {} = {}", pair.first, pair.second);
            }
        }
        for (const auto& pair : energy_list)
        {
            if (is_nan_scalar(pair.second) || is_inf_scalar(pair.second))
            {
                LUISA_ERROR("Energy {} is not valid : {}", pair.first, pair.second);
            }
        }
        return total_energy;
    };
    // Init LBVH
    {
        ADD_HOST_TIME_STAMP("Init LBVH");
        stream << sim_data->sa_x_step_start.copy_from(host_sim_data->sa_x_step_start.data());
        lbvh_face->reduce_face_tree_aabb(stream, sim_data->sa_x_step_start, mesh_data->sa_faces);
        lbvh_edge->reduce_edge_tree_aabb(stream, sim_data->sa_x_step_start, mesh_data->sa_edges);
        lbvh_face->construct_tree(stream);
        lbvh_edge->construct_tree(stream);
        stream << luisa::compute::synchronize();
    }
    // for (uint substep = 0; substep < get_scene_params().num_substep; substep++)
    {
        if (host_sim_data->num_verts_soft != 0)
        {
            stream << fn_predict_position(substep_dt, get_scene_params().gravity).dispatch(host_sim_data->num_verts_soft)
                   << sim_data->sa_x_tilde.copy_to(host_sim_data->sa_x_tilde.data());
        }
        if (host_sim_data->num_affine_bodies != 0)
        {
            stream
                << fn_abd_predict_position(substep_dt, get_scene_params().gravity).dispatch(host_sim_data->num_affine_bodies * 4)
                << sim_data->sa_affine_bodies_q_tilde.copy_to(host_sim_data->sa_affine_bodies_q_tilde.data());
        }

        stream << luisa::compute::synchronize();


        double prev_state_energy = Float_max;

        LUISA_INFO("=== In frame {} ===", get_scene_params().current_frame);

        uint iter                      = 0;
        bool direchlet_point_converged = false;
        for (iter = 0; iter < 100; iter++)
        {
            if (iter >= get_scene_params().nonlinear_iter_count && direchlet_point_converged)
            {
                break;
            }
            ADD_HOST_TIME_STAMP("Calc Force");
            get_scene_params().current_nonlinear_iter = iter;


            stream << sim_data->sa_x_iter_start.copy_to(host_sim_data->sa_x_iter_start.data());  // For host apply dx
            if (host_sim_data->num_affine_bodies != 0)
            {
                stream << sim_data->sa_affine_bodies_q_iter_start.copy_to(
                    host_sim_data->sa_affine_bodies_q_iter_start.data());
            }
            stream << fn_reset_vector(sim_data->sa_cgX).dispatch(sim_data->sa_cgX.size())
                   << fn_reset_vector(sim_data->sa_cgB).dispatch(sim_data->sa_cgB.size())
                   << fn_reset_float3x3(sim_data->sa_cgA_diag).dispatch(sim_data->sa_cgA_diag.size())
                   << fn_reset_cgA_offdiag_triplet().dispatch(sim_data->sa_cgA_fixtopo_offdiag_triplet.size())
                // << fn_reset_float3x3(sim_data->sa_cgA_offdiag_stretch_spring).dispatch(sim_data->sa_cgA_offdiag_stretch_spring.size())
                // << fn_reset_float3x3(sim_data->sa_cgA_offdiag_bending).dispatch(sim_data->sa_cgA_offdiag_bending.size())
                ;

            {
                if (host_sim_data->num_verts_soft != 0)
                {
                    stream << fn_evaluate_inertia(substep_dt, get_scene_params().stiffness_dirichlet)
                                  .dispatch(host_sim_data->num_verts_soft);

                    stream << fn_evaluate_ground_collision(get_scene_params().floor.y,
                                                           get_scene_params().use_floor,
                                                           get_scene_params().stiffness_collision,
                                                           get_scene_params().d_hat,
                                                           get_scene_params().thickness)
                                  .dispatch(host_sim_data->num_verts_soft);
                }

                if (host_sim_data->sa_stretch_springs.size() != 0)
                {
                    stream << fn_evaluate_spring(get_scene_params().stiffness_spring)
                                  .dispatch(host_sim_data->sa_stretch_springs.size());
                    stream << fn_material_energy_assembly_stretch_spring().dispatch(host_sim_data->num_verts_soft);
                }

                if (host_sim_data->sa_stretch_faces.size() != 0)
                {
                    stream << fn_evaluate_stretch_face().dispatch(host_sim_data->sa_stretch_faces.size());
                    stream << fn_material_energy_assembly_stretch_face().dispatch(host_sim_data->num_verts_soft);
                }

                if (host_sim_data->sa_bending_edges.size() != 0)
                {
                    stream << fn_evaluate_bending(get_scene_params().get_bending_stiffness_scaling())
                                  .dispatch(host_sim_data->sa_bending_edges.size());
                    stream << fn_material_energy_assembly_bending().dispatch(host_sim_data->num_verts_soft);
                }

                if (host_sim_data->num_affine_bodies != 0)
                {
                    stream << fn_evaluate_abd_inertia(substep_dt, get_scene_params().stiffness_dirichlet)
                                  .dispatch(host_sim_data->num_affine_bodies);

                    stream << fn_evaluate_abd_orthogonality().dispatch(host_sim_data->num_affine_bodies);

                    stream << fn_evaluate_abd_ground_collision(get_scene_params().floor.y,
                                                               get_scene_params().use_floor,
                                                               get_scene_params().stiffness_collision,
                                                               get_scene_params().d_hat,
                                                               get_scene_params().thickness,
                                                               host_sim_data->num_verts_soft)
                                  .dispatch(host_sim_data->num_verts_rigid);

                    stream << fn_material_energy_assembly_affine_body(host_sim_data->num_verts_soft)
                                  .dispatch(host_sim_data->num_affine_bodies * 4);
                }

                update_contact_set();

                evaluate_contact();

                // stream << fn_evaluate_dirichlet(substep_dt, get_scene_params().stiffness_dirichlet).dispatch(num_verts);

                if (get_scene_params().use_energy_linesearch)
                    prev_state_energy = compute_energy_interface();
            }

            stream << luisa::compute::synchronize();
            ADD_HOST_TIME_STAMP("PCG");
            pcg_solver->device_solve(stream, pcg_spmv_interface, compute_energy_interface);

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
                auto curr_energy = compute_energy_interface();
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

                    curr_energy = compute_energy_interface();
                    LUISA_INFO("     Line search {} : alpha = {:6.5f}, energy = {:12.10f}", line_search_count, alpha, curr_energy);

                    if (alpha < 1e-4)
                    {
                        LUISA_ERROR("  Line search failed, energy = {}, prev state energy = {}", curr_energy, prev_state_energy);
                    }
                    line_search_count++;
                }
                prev_state_energy = curr_energy;  // E_prev = E
            }

            // Check dirichlet point target
            {
                const float direchlet_max_delta = CpuParallel::parallel_for_and_reduce(
                    0,
                    host_sim_data->sa_x.size(),
                    [&](const uint vid)
                    {
                        if (host_mesh_data->sa_is_fixed[vid])
                        {
                            float3 delta = host_sim_data->sa_x[vid] - host_sim_data->sa_x_tilde[vid];
                            return luisa::length(delta);
                        }
                        return 0.0f;  // Non-fixed point
                    },
                    [](const float left, const float right) { return max_scalar(left, right); },
                    -1e9f);
                direchlet_point_converged = direchlet_max_delta < 1e-3;
                if (!direchlet_point_converged)
                {
                    LUISA_INFO("  In non-linear iter {:2}: Dirichlet point not converged, max delta = {}", iter, direchlet_max_delta);
                }
            }

            stream << sim_data->sa_x.copy_to(sim_data->sa_x_iter_start);
            if (sim_data->num_affine_bodies != 0)
            {
                stream << sim_data->sa_affine_bodies_q.copy_to(sim_data->sa_affine_bodies_q_iter_start);
            }
            stream << luisa::compute::synchronize();
        }

        if (host_sim_data->num_verts_soft != 0)
        {
            stream << fn_update_velocity(substep_dt, get_scene_params().fix_scene, get_scene_params().damping_cloth)
                          .dispatch(host_sim_data->num_verts_soft);
        }
        if (host_sim_data->num_affine_bodies != 0)
        {
            stream << fn_abd_update_velocity(substep_dt, get_scene_params().fix_scene, get_scene_params().damping_cloth)
                          .dispatch(host_sim_data->num_affine_bodies * 4);
        }
    }

    stream << luisa::compute::synchronize();

    // Copy to host
    {
        stream << sim_data->sa_x.copy_to(host_sim_data->sa_x.data())
               << sim_data->sa_v.copy_to(host_sim_data->sa_v.data());

        if (host_sim_data->num_affine_bodies != 0)
        {
            stream << sim_data->sa_affine_bodies_q.copy_to(host_sim_data->sa_affine_bodies_q.data())
                   << sim_data->sa_affine_bodies_q_v.copy_to(host_sim_data->sa_affine_bodies_q_v.data());
        }

        stream << luisa::compute::synchronize();
    }

    // Return frame end position and velocity
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