#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include "CollisionDetector/cipc_kernel.hpp"
#include "CollisionDetector/distance.hpp"
#include "CollisionDetector/friction_kernel.hpp"
#include "Core/affine_position.h"
#include "Core/float_n.h"
#include "Energy/bending_energy.h"
#include "Energy/stretch_energy.h"
#include "Initializer/init_mesh_data.h"
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

template <typename T>
static void buffer_copy(const std::vector<T>& src, std::vector<T>& dst)
{
    if (src.size() != dst.size())
    {
        LUISA_ERROR("Buffer size mismatch {} != {}", src.size(), dst.size());
    }
    std::memcpy(dst.data(), src.data(), sizeof(T) * src.size());
    // CpuParallel::parallel_copy(src, dst);
}
template <typename T>
static void buffer_copy(luisa::compute::Stream&          stream,
                        const luisa::compute::Buffer<T>& src,
                        luisa::compute::Buffer<T>&       dst)
{
    stream << src.copy_to(dst);
}
template <typename T>
static void buffer_upload(luisa::compute::Stream& stream, const std::vector<T>& src, luisa::compute::Buffer<T>& dst)
{
    stream << dst.copy_from(src.data());
}

template <typename T>
static void buffer_download(luisa::compute::Stream&          stream,
                            const luisa::compute::Buffer<T>& src,
                            std::vector<T>&                  dst,
                            const bool                       wait = false)
{
    stream << src.copy_to(dst.data());
    if (wait)
        stream << luisa::compute::synchronize();
}
// template <typename T>
// [[nodiscard]] static auto buffer_download(const luisa::compute::Buffer<T>& src, std::vector<T>& dst, const bool wait = false)
// {
//     if (wait)
//         return src.copy_to(dst.data()) << luisa::compute::synchronize();
//     else
//         return src.copy_to(dst.data());
// }

template <typename T>
static void buffer_add(const luisa::compute::BufferView<T>& buffer, const Var<uint> dest, const Var<T>& value)
{
    buffer->write(dest, buffer->read(dest) + value);
}
template <typename T>
static void buffer_add(const luisa::compute::BufferVar<T>& buffer, const Var<uint> dest, const Var<T>& value)
{
    buffer->write(dest, buffer->read(dest) + value);
}

template <typename T>
static void buffer_add(std::vector<T>& buffer, const uint dest, const T& value)
{
    buffer[dest] = buffer[dest] + value;
}
template <typename T>
static void buffer_add(const std::span<T>& buffer, const uint dest, const T& value)
{
    buffer[dest] = buffer[dest] + value;
}
template <typename T>
static void atomic_buffer_add(const std::vector<T>& buffer, const uint dest, const T& value)
{
    CpuParallel::spin_atomic<T>::fetch_add(buffer[dest], value);
}
template <typename T>
static void atomic_buffer_add(const std::span<T>& buffer, const uint dest, const T& value)
{
    CpuParallel::spin_atomic<T>::fetch_add(buffer[dest], value);
}

static void atomic_buffer_add(const luisa::compute::BufferVar<float3>& buffer, const Var<uint> dest, const Var<float3>& value)
{
    buffer->atomic(dest)[0].fetch_add(value[0]);
    buffer->atomic(dest)[1].fetch_add(value[1]);
    buffer->atomic(dest)[2].fetch_add(value[2]);
}
static void atomic_buffer_add(const luisa::compute::BufferVar<float3x3>& buffer,
                              const Var<uint>                            dest,
                              const Var<float3x3>&                       value)
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

    luisa::compute::ShaderOption default_option = {.enable_debug_info = true};

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

    auto fn_apply_template = [sa_x_to_dof_map = sim_data->sa_x_to_dof_map.view(),
                              sa_scaled_model_x = sim_data->sa_scaled_model_x.view()](const auto& input_q)
    {
        const Uint vid = dispatch_id().x;
        Float3     new_dx;
        const Uint map_info   = sa_x_to_dof_map->read(vid);
        const Uint mapped_dof = map_info & Attributions::RIGID_BODY_MASK;
        $if((map_info & Attributions::RIGID_BODY_FLAG) == 0)  // Soft body
        {
            new_dx = input_q->read(mapped_dof);
        }
        $else  // Rigid body
        {
            const Float3 rest_x = sa_scaled_model_x->read(vid);
            Float3       p;
            Float3x3     A;
            p      = input_q->read(mapped_dof + 0);
            A[0]   = input_q->read(mapped_dof + 1);
            A[1]   = input_q->read(mapped_dof + 2);
            A[2]   = input_q->read(mapped_dof + 3);
            new_dx = A * rest_x + p;  // Affine position
        };
        return new_dx;
    };

    // Map from status `q` to position `x`
    compiler.compile<1>(
        fn_apply_q_to_x_template,
        [fn_apply_template](BufferVar<float3> input_q, BufferVar<float3> out_x)
        {
            const Uint vid   = dispatch_id().x;
            Float3     new_x = fn_apply_template(input_q);
            out_x->write(vid, new_x);
        },
        default_option);

    // Linear interpolation
    compiler.compile<1>(
        fn_interpolate_template,
        [](BufferVar<float3> output_buffer, BufferVar<float3> input_start, BufferVar<float3> input_delta, Float alpha)
        {
            const UInt   vid     = dispatch_id().x;
            const Float3 init_q  = input_start->read(vid);
            const Float3 delta_q = input_delta->read(vid);
            const Float3 new_q   = init_q + alpha * delta_q;
            output_buffer->write(vid, new_q);
        },
        default_option);

    // Why this failed?????
    // compiler.compile<1>(
    //     fn_apply_dq,
    //     [sa_q            = sim_data->sa_q.view(),
    //      sa_dq           = sim_data->sa_dq.view(),
    //      sa_q_iter_start = sim_data->sa_q_iter_start.view()](const Float alpha)
    //     {
    //         const UInt vid = dispatch_id().x;
    //         sa_q->write(vid, sa_q_iter_start->read(vid) + alpha * sa_dq->read(vid));
    //     },
    //     default_option);

    // compiler.compile<1>(
    //     fn_apply_dx,
    //     [sa_x            = sim_data->sa_x.view(),
    //      sa_dx           = sim_data->sa_dx.view(),
    //      sa_x_iter_start = sim_data->sa_x_iter_start.view()](const Float alpha)
    //     {
    //         const UInt vid = dispatch_id().x;
    //         sa_x->write(vid, sa_x_iter_start->read(vid) + alpha * sa_dx->read(vid));
    //     },
    //     default_option);
}

void NewtonSolver::compile_advancing(AsyncCompiler& compiler, const luisa::compute::ShaderOption& default_option)
{
    using namespace luisa::compute;

    // if (host_sim_data->num_verts_soft != 0)
    {
        compiler.compile<1>(
            fn_predict_position,
            [sa_q_step_start = sim_data->sa_q_step_start.view(),  // Input
             sa_q_v          = sim_data->sa_q_v.view(),           // Input
             //  sa_q            = sim_data->sa_q.view(),             // Output
             //  sa_q_iter_start = sim_data->sa_q_iter_start.view(),  // Output
             sa_q_tilde    = sim_data->sa_q_tilde.view(),     // Output
             sa_is_fixed   = sim_data->sa_q_is_fixed.view(),  // Constant
             sa_q_property = sim_data->sa_q_property.view()   // Constant
        ](const Float substep_dt, const Float3 gravity)
            {
                const UInt vid                = dispatch_id().x;
                const UInt property           = sa_q_property->read(vid);
                const Bool is_fixed           = sa_is_fixed->read(vid);
                const Bool is_rigid           = (property & Attributions::RIGID_BODY_FLAG) != 0;
                const Bool is_translation_dof = (property & Attributions::ABD_Is_Translation_DOF) != 0;

                Float3 x_prev = sa_q_step_start->read(vid);
                Float3 v_prev = sa_q_v->read(vid);
                Float3 v_pred = v_prev;

                // Only apply gravity to the translation part of rigid body
                $if(!is_fixed &                       //
                    (!is_rigid |                      // Soft body
                     (is_rigid & is_translation_dof)  // Rigid body translation dof
                     ))
                {
                    v_pred += substep_dt * gravity;
                };
                Float3 x_pred = x_prev + substep_dt * v_pred;

                sa_q_tilde->write(vid, x_pred);
            },
            default_option);

        compiler.compile<1>(
            fn_update_velocity,
            [sa_q   = sim_data->sa_q.view(),
             sa_q_v = sim_data->sa_q_v.view(),
             sa_q_step_start = sim_data->sa_q_step_start.view()](const Float substep_dt, const Bool fix_scene, const Float damping)
            {
                const UInt vid       = dispatch_id().x;
                Float3     q_step_bg = sa_q_step_start->read(vid);
                Float3     q_step_ed = sa_q->read(vid);

                Float3 dq  = q_step_ed - q_step_bg;
                Float3 vel = dq / substep_dt;

                $if(fix_scene)
                {
                    dq  = make_float3(0.0f);
                    vel = make_float3(0.0f);
                    sa_q->write(vid, q_step_bg);
                    $return();
                };

                vel *= exp(-damping * substep_dt);

                sa_q_v->write(vid, vel);
                sa_q_step_start->write(vid, q_step_ed);  // Optional output
            },
            default_option);
    }
}


template <typename T>
void assembly_template3(const luisa::compute::Var<uint>       vid,
                        const luisa::compute::Var<T>&         constaints,
                        luisa::compute::BufferView<uint>&     sa_vert_adj_material_force_verts_csr,
                        luisa::compute::BufferView<float3>&   sa_cgB,
                        luisa::compute::BufferView<float3x3>& sa_cgA_diag,
                        luisa::compute::BufferView<MatrixTriplet3x3>& sa_cgA_offdiag_triplet)
{
    constexpr uint N = T::get_num_verts_per_constaint();

    // auto& sa_vert_adj_material_force_verts_csr = constaints.get_vert_adj_material_force_verts_csr();
    auto& vert_adj_constraints_csr     = constaints.vert_adj_constraints_csr;
    auto& constaint_gradients          = constaints.constraint_gradients;
    auto& constaint_hessians           = constaints.constraint_hessians;
    auto& constaint_offsets_in_adjlist = constaints.constraint_offsets_in_adjlist;
    auto& constaint_indices            = constaints.constraint_indices;

    using namespace luisa::compute;

    // const Uint vid         = dispatch_x();
    const Uint curr_prefix = sa_vert_adj_material_force_verts_csr->read(vid);
    const Uint next_prefix = sa_vert_adj_material_force_verts_csr->read(vid + 1);

    const Uint curr_prefix_bending = vert_adj_constraints_csr->read(vid);
    const Uint next_prefix_bending = vert_adj_constraints_csr->read(vid + 1);

    $for(j, curr_prefix_bending, next_prefix_bending)
    {
        const Uint adj_eid = vert_adj_constraints_csr->read(j);
        const auto edge    = constaint_indices->read(adj_eid);
        Uint       offset  = -1u;
        if constexpr (N == 1)
        {
            offset = 0;
        }
        else
        {
            for (uint k = 0; k < N; k++)
            {
                $if(vid == edge[k])
                {
                    offset = k;
                };
            }
            device_assert(offset != -1u, "Error in assembly: offset not found.");
        }

        const Float3   grad      = constaint_gradients->read(adj_eid * N + offset);
        const Float3x3 diag_hess = constaint_hessians->read(adj_eid * (N * N) + offset);

        buffer_add(sa_cgB, vid, -grad);
        buffer_add(sa_cgA_diag, vid, diag_hess);

        const uint N_off = N - 1;
        for (uint ii = 0; ii < N_off; ii++)  // For each off-diagonal in curr row
        {
            Float3x3 offdiag_hess = constaint_hessians->read(adj_eid * (N * N) + N + offset * N_off + ii);
            Uint offdiag_offset = constaint_offsets_in_adjlist->read(adj_eid * (N * N_off) + offset * N_off + ii);
            auto triplet = sa_cgA_offdiag_triplet->read(curr_prefix + offdiag_offset);
            add_triplet_matrix(triplet, offdiag_hess);
            sa_cgA_offdiag_triplet->write(curr_prefix + offdiag_offset, triplet);
        }
    };
};

void NewtonSolver::compile_assembly(AsyncCompiler& compiler, const luisa::compute::ShaderOption& default_option)
{
    using namespace luisa::compute;
    auto sa_vert_adj_material_force_verts_csr = sim_data->sa_vert_adj_material_force_verts_csr.view();
    auto sa_cgB                               = sim_data->sa_cgB.view();
    auto sa_cgA_diag                          = sim_data->sa_cgA_diag.view();
    auto sa_cgA_offdiag_triplet               = sim_data->sa_cgA_fixtopo_offdiag_triplet.view();

    auto perform_assembly_interface = [&](const auto& constraint, const Uint prefix_dof)
    {
        assembly_template3(prefix_dof + dispatch_x(), constraint, sa_vert_adj_material_force_verts_csr, sa_cgB, sa_cgA_diag, sa_cgA_offdiag_triplet);
    };

    // Assembly
    compiler.compile(fn_material_energy_assembly_stretch_spring,
                     [&](Var<Constitutions::StretchSpring<luisa::compute::Buffer>> constraint)
                     { perform_assembly_interface(constraint, 0); });

    compiler.compile(fn_material_energy_assembly_stretch_face,
                     [&](Var<Constitutions::StretchFace<luisa::compute::Buffer>> constraint)
                     { perform_assembly_interface(constraint, 0); });

    compiler.compile(fn_material_energy_assembly_bending,
                     [&](Var<Constitutions::BendingEdge<luisa::compute::Buffer>> constraint)
                     { perform_assembly_interface(constraint, 0); });

    compiler.compile(fn_material_energy_assembly_soft_inertia,
                     [&](Var<Constitutions::SoftInertia<luisa::compute::Buffer>> constraint)
                     { perform_assembly_interface(constraint, 0); });

    compiler.compile(fn_material_energy_assembly_abd_inertia,
                     [&](Var<Constitutions::AbdInertia<luisa::compute::Buffer>> constraint, const Uint prefix)
                     { perform_assembly_interface(constraint, prefix); });

    compiler.compile(fn_material_energy_assembly_abd_ortho,
                     [&](Var<Constitutions::AbdOrthogonality<luisa::compute::Buffer>> constraint, const Uint prefix)
                     { perform_assembly_interface(constraint, prefix); });
}
void NewtonSolver::compile_evaluate(AsyncCompiler& compiler, const luisa::compute::ShaderOption& default_option)
{
    using namespace luisa::compute;

    compiler.compile<1>(
        fn_evaluate_soft_inertia,
        [sa_x        = sim_data->sa_q.view(),
         sa_x_tilde  = sim_data->sa_q_tilde.view(),
         sa_is_fixed = sim_data->sa_q_is_fixed.view()](
            Var<Constitutions::SoftInertia<luisa::compute::Buffer>> contraint, const Float substep_dt)
        {
            auto& sa_vert_mass           = contraint.sa_soft_vert_mass;
            auto& sa_stiffness_dirichlet = contraint.sa_stiffness_dirichlet;
            auto& output_gradient        = contraint.constraint_gradients;
            auto& output_hessian         = contraint.constraint_hessians;

            const UInt  vid     = dispatch_id().x;
            const Float h       = substep_dt;
            const Float h_2_inv = 1.0f / (h * h);

            Float3 x_k     = sa_x->read(vid);
            Float3 x_tilde = sa_x_tilde->read(vid);
            Float  mass    = sa_vert_mass->read(vid);

            Float3   gradient = mass * h_2_inv * (x_k - x_tilde);
            Float3x3 hessian  = make_float3x3(1.0f) * mass * h_2_inv;

            // $if(sa_is_fixed->read(vid) != 0)
            {
                const Float stiffness_dirichlet = sa_stiffness_dirichlet->read(vid);

                gradient = stiffness_dirichlet * gradient;
                hessian  = stiffness_dirichlet * hessian;
                // hessian = make_float3x3(1.0f) * 1e9f;
                // gradient = make_float3(0.0f);
            };

            output_gradient->write(vid, gradient);
            output_hessian->write(vid, hessian);
        },
        default_option);

    compiler.compile<1>(
        fn_evaluate_abd_inertia,
        [abd_q         = sim_data->sa_q.view(),
         abd_q_tilde   = sim_data->sa_q_tilde.view(),
         sa_q_is_fixed = sim_data->sa_q_is_fixed.view()](
            Var<Constitutions::AbdInertia<luisa::compute::Buffer>> constraint, const Float substep_dt)
        {
            auto& abd_indices                  = constraint.constraint_indices;
            auto& abd_gradients                = constraint.constraint_gradients;
            auto& abd_hessians                 = constraint.constraint_hessians;
            auto& sa_affine_bodies_mass_matrix = constraint.sa_affine_bodies_mass_matrix;
            auto& sa_stiffness_dirichlet       = constraint.sa_stiffness_dirichlet;

            const UInt  body_idx = dispatch_id().x;
            const Float h        = substep_dt;
            const Float h_2_inv  = 1.0f / (h * h);

            const Uint4 indices = abd_indices->read(body_idx);

            Float3 delta_q[4] = {abd_q->read(indices[0]) - abd_q_tilde->read(indices[0]),
                                 abd_q->read(indices[1]) - abd_q_tilde->read(indices[1]),
                                 abd_q->read(indices[2]) - abd_q_tilde->read(indices[2]),
                                 abd_q->read(indices[3]) - abd_q_tilde->read(indices[3])};

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
            // $if(sa_q_is_fixed->read(indices[0]) != 0)
            {
                const Float stiffness_dirichlet = sa_stiffness_dirichlet->read(body_idx);

                // alpha = (1.0f + stiffness_dirichlet);
                alpha = stiffness_dirichlet;
            };

            abd_gradients->write(4 * body_idx + 0, alpha * h_2_inv * gradient[0]);
            abd_gradients->write(4 * body_idx + 1, alpha * h_2_inv * gradient[1]);
            abd_gradients->write(4 * body_idx + 2, alpha * h_2_inv * gradient[2]);
            abd_gradients->write(4 * body_idx + 3, alpha * h_2_inv * gradient[3]);

            // Float3 max_grad     = max(max(gradient[0], gradient[1]), max(gradient[2], gradient[3]));
            // Float  max_grad_val = alpha * h_2_inv * max(max_grad.x, max(max_grad.y, max_grad.z));
            // device_log("Body {}, max force = {}", body_idx, -max_grad_val);

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

    auto calculate_per_vert_grad_hess_template =
        [sa_x                           = sim_data->sa_x.view(),
         sa_x_step_start                = sim_data->sa_x_step_start.view(),
         sa_rest_vert_area              = mesh_data->sa_rest_vert_area.view(),
         sa_is_fixed                    = mesh_data->sa_is_fixed.view(),
         sa_contact_active_verts_offset = sim_data->sa_contact_active_verts_offset.view(),
         sa_contact_active_verts_d_hat  = sim_data->sa_contact_active_verts_d_hat.view(),
         sa_contact_active_verts_friction_coeff =
             sim_data->sa_contact_active_verts_friction_coeff.view()](const Uint  vid,
                                                                      Float3&     out_gradient,
                                                                      Float3x3&   out_hessian,
                                                                      const Float floor_y,
                                                                      const Bool  use_ground_collision,
                                                                      const Float stiffness,
                                                                      const Uint  collision_type)
    {
        Bool collide = false;
        $if(!sa_is_fixed->read(vid) & use_ground_collision)
        {
            Float3 x_k = sa_x->read(vid);
            Float3 x_0 = sa_x_step_start->read(vid);

            Float d_hat     = sa_contact_active_verts_d_hat->read(vid);
            Float thickness = sa_contact_active_verts_offset->read(vid);

            float3 normal = luisa::make_float3(0, 1, 0);
            Float  area   = sa_rest_vert_area->read(vid);
            Float  stiff  = stiffness * area;

            // Repulsion
            Float curr_dist = x_k.y - floor_y;
            $if(curr_dist - thickness < d_hat)
            {
                Float k1;
                Float k2;
                $if(collision_type == 0)
                {
                    k1 = stiff * (curr_dist - thickness - d_hat);
                    k2 = stiff;
                }
                $else
                {
                    k1 = stiff * ipc::barrier_first_derivative(curr_dist - thickness, d_hat);
                    k2 = stiff * ipc::barrier_second_derivative(curr_dist - thickness, d_hat);
                };

                out_gradient = k1 * normal;
                out_hessian  = k2 * outer_product(normal, normal);
                collide      = true;
            };

            // Friction
            Float init_dist = x_0.y - floor_y;
            $if(init_dist - thickness < d_hat)
            {
                Float k1;
                $if(collision_type == 0)
                {
                    k1 = stiff * (init_dist - thickness - d_hat);
                }
                $else
                {
                    k1 = stiff * ipc::barrier_first_derivative(init_dist - thickness, d_hat);
                };

                Float3 rel_dx       = x_k - x_0;
                Float  friction_mu  = sa_contact_active_verts_friction_coeff->read(vid);
                Float  friction_eps = Friction::ando_barrier::friction_eps;
                auto   lambda_mu    = -k1 * friction_mu;
                auto   friction_grad_hess =
                    Friction::ipc_barrier::compute_friction_gradient_hessian(lambda_mu, normal, rel_dx, friction_eps);
                out_gradient += friction_grad_hess.first;
                out_hessian += friction_grad_hess.second;
                collide = true;
            };
        };
        return collide;
    };

    compiler.compile<1>(
        fn_evaluate_soft_ground_collision,
        [calculate_per_vert_grad_hess_template](Var<Constitutions::SoftInertia<luisa::compute::Buffer>> contraint,
                                                Float floor_y,
                                                Bool  use_ground_collision,
                                                Float stiffness,
                                                Uint  collision_type)
        {
            auto& output_gradient = contraint.constraint_gradients;
            auto& output_hessian  = contraint.constraint_hessians;

            const UInt vid = dispatch_id().x;

            Float3   grad    = make_float3(0.0f);
            Float3x3 hess    = make_float3x3(0.0f);
            Bool     collide = calculate_per_vert_grad_hess_template(
                vid, grad, hess, floor_y, use_ground_collision, stiffness, collision_type);

            $if(collide)
            {
                buffer_add(output_gradient, vid, grad);
                buffer_add(output_hessian, vid, hess);
            };
        },
        default_option);

    compiler.compile<1>(
        fn_evaluate_abd_ground_collision,
        [sa_scaled_model_x = mesh_data->sa_scaled_model_x.view(),
         sa_x_to_dof_map   = sim_data->sa_x_to_dof_map.view(),
         calculate_per_vert_grad_hess_template](Var<Constitutions::AbdInertia<luisa::compute::Buffer>> constraint,
                                                Float floor_y,
                                                Bool  use_ground_collision,
                                                Float stiffness,
                                                Uint  vid_start,
                                                Uint  collision_type)
        {
            auto& abd_gradients = constraint.constraint_gradients;
            auto& abd_hessians  = constraint.constraint_hessians;

            const UInt vid = vid_start + dispatch_id().x;

            Float3   grad    = make_float3(0.0f);
            Float3x3 hess    = make_float3x3(0.0f);
            Bool     collide = calculate_per_vert_grad_hess_template(
                vid, grad, hess, floor_y, use_ground_collision, stiffness, collision_type);

            const Uint dof_info = sa_x_to_dof_map->read(vid);
            const Uint dof_idx  = dof_info & Attributions::RIGID_BODY_MASK;
            const Uint body_idx = (dof_idx - vid_start) / 4;

            const Float4 weight = make_float4(1.0f, sa_scaled_model_x->read(vid));

            uint idx = 4;
            for (uint ii = 0; ii < 4; ii++)
            {
                Float  wi          = weight[ii];
                Float3 affine_grad = wi * grad;
                atomic_buffer_add(abd_gradients, 4 * body_idx + ii, affine_grad);
                for (uint jj = 0; jj < 4; jj++)
                {
                    Float    wj          = weight[jj];
                    Float3x3 affine_hess = wi * wj * hess;
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
        },
        default_option);

    compiler.compile<1>(
        fn_gound_collision_ccd,
        [sa_x_iter_start                = sim_data->sa_x_iter_start.view(),
         sa_x                           = sim_data->sa_x.view(),
         sa_contact_active_verts_offset = sim_data->sa_contact_active_verts_offset.view(),
         sa_contact_active_verts_d_hat  = sim_data->sa_contact_active_verts_d_hat.view(),
         toi_per_vert                   = collision_data->toi_per_vert.view(),
         sa_is_fixed = mesh_data->sa_is_fixed.view()](Float floor_y, Bool use_ground_collision)
        {
            const UInt vid = dispatch_id().x;

            Float toi = 1.0f;
            $if(use_ground_collision)
            {
                $if(!sa_is_fixed->read(vid))
                {
                    Float offset = sa_contact_active_verts_offset->read(vid);
                    Float curr_y = sa_x->read(vid).y;
                    $if(curr_y - offset < floor_y)
                    {
                        Float init_y  = sa_x_iter_start->read(vid).y;
                        Float curr_dy = curr_y - init_y;  // abs
                        toi = (init_y - offset - floor_y) / (init_y - curr_y) / accd::line_search_max_t;
                        // Float target_y = init_y + toi * curr_dy;
                        // $if(alpha < 0.0f | alpha > 1.0f | target_y - offset < floor_y){};
                        // $if(vid == 22)
                        // {
                        //     device_log("Vert {} penetrate in toi {} (From {} to {}, thickness = {} (After apply = {})",
                        //                vid,
                        //                alpha,
                        //                init_y,
                        //                curr_y,
                        //                offset,
                        //                init_y + alpha * curr_dy);
                        // };
                        // device_log("Vert {}'s toi = {} (After apply = {})", vid, alpha, init_y + alpha * curr_dy);
                        // toi_per_vert->atomic(0).fetch_min(alpha);
                    };
                };
            };

            toi = ParallelIntrinsic::block_intrinsic_reduce(vid, toi, ParallelIntrinsic::warp_reduce_op_min<float>);

            $if(vid % 256 == 0 & toi != 1.0f)
            {
                toi_per_vert->atomic(0).fetch_min(toi);
            };
        },
        default_option);

    // if (host_sim_data->sa_stretch_springs.size() != 0)
    compiler.compile<1>(
        fn_evaluate_spring,
        [sa_x = sim_data->sa_x.view()](Var<Constitutions::StretchSpring<luisa::compute::Buffer>> constraint)
        {
            auto& sa_edges                    = constraint.constraint_indices;
            auto& sa_rest_length              = constraint.sa_stretch_spring_rest_state_length;
            auto& sa_stretch_spring_stiffness = constraint.sa_stretch_spring_stiffness;
            auto& output_gradient_ptr         = constraint.constraint_gradients;
            auto& output_hessian_ptr          = constraint.constraint_hessians;


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

    // if (host_sim_data->sa_stretch_faces.size() != 0)
    compiler.compile<1>(
        fn_evaluate_stretch_face,
        [sa_x = sim_data->sa_x.view()](Var<Constitutions::StretchFace<luisa::compute::Buffer>> constraint)
        {
            auto& sa_faces                   = constraint.constraint_indices;
            auto& sa_stretch_faces_Dm_inv    = constraint.sa_stretch_faces_Dm_inv;
            auto& sa_stretch_faces_rest_area = constraint.sa_stretch_faces_rest_area;
            auto& sa_stretch_faces_mu_lambda = constraint.sa_stretch_faces_mu_lambda;
            auto& output_gradient_ptr        = constraint.constraint_gradients;
            auto& output_hessian_ptr         = constraint.constraint_hessians;

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

    compiler.compile<1>(
        fn_evaluate_bending,
        [sa_x = sim_data->sa_x.view()](Var<Constitutions::BendingEdge<luisa::compute::Buffer>> constraint, const Float scaling)
        {
            auto& sa_edges                    = constraint.constraint_indices;
            auto& sa_bending_edges_rest_angle = constraint.sa_bending_edges_rest_angle;
            auto& sa_bending_edges_rest_area  = constraint.sa_bending_edges_rest_area;
            auto& sa_bending_edges_stiffness  = constraint.sa_bending_edges_stiffness;
            auto& output_gradient_ptr         = constraint.constraint_gradients;
            auto& output_hessian_ptr          = constraint.constraint_hessians;

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

    compiler.compile<1>(
        fn_evaluate_abd_orthogonality,
        [abd_q = sim_data->sa_q.view()](Var<Constitutions::AbdOrthogonality<luisa::compute::Buffer>> constraint)
        {
            auto& abd_indices   = constraint.constraint_indices;
            auto& abd_gradients = constraint.constraint_gradients;
            auto& abd_hessians  = constraint.constraint_hessians;
            auto& abd_volume    = constraint.abd_volume;
            auto& abd_kappa     = constraint.abd_kappa;

            const UInt body_idx = dispatch_id().x;

            Float3   ortho_gradient[3] = {Zero3};
            Float3x3 ortho_hessian[6]  = {Zero3x3};

            const Uint3 indices = abd_indices->read(body_idx);

            Float3x3 A =
                make_float3x3(abd_q->read(indices[0]), abd_q->read(indices[1]), abd_q->read(indices[2]));

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

            auto write_to_grad = [&](const uint i)
            { abd_gradients->write(3 * body_idx + i, ortho_gradient[i]); };
            auto write_to_hess = [&](const uint i, const uint j, const bool use_transpose)
            {
                Float3x3 ortho_hess = use_transpose ? transpose(ortho_hessian[j]) : ortho_hessian[j];
                abd_hessians->write(9 * body_idx + i, ortho_hess);
            };

            write_to_grad(0);
            write_to_grad(1);
            write_to_grad(2);

            //  0   3   4  ||   0   1   2
            //  5   1   6  ||  t1   3   4
            //  7   8   2  ||  t2  t4   5
            write_to_hess(0, 0, false);
            write_to_hess(1, 3, false);
            write_to_hess(2, 5, false);
            write_to_hess(3, 1, false);
            write_to_hess(4, 2, false);
            write_to_hess(5, 1, true);
            write_to_hess(6, 4, false);
            write_to_hess(7, 2, true);
            write_to_hess(8, 4, true);
        },
        default_option);
}


// Host functions
// Outputs:
//          sa_q_tilde => predicted position
void NewtonSolver::host_predict_position()
{
    CpuParallel::parallel_for(0,
                              host_sim_data->num_dof,
                              [sa_q_v          = std::span(host_sim_data->sa_q_v),
                               sa_q_tilde      = std::span(host_sim_data->sa_q_tilde),
                               sa_q_step_start = std::span(host_sim_data->sa_q_step_start),
                               sa_q_property   = std::span(host_sim_data->sa_q_property),
                               sa_q_is_fixed   = std::span(host_sim_data->sa_q_is_fixed),
                               substep_dt      = get_scene_params().get_substep_dt(),
                               gravity         = get_scene_params().gravity](const uint vid)
                              {
                                  const uint dof_property = sa_q_property[vid];
                                  const bool is_fixed     = sa_q_is_fixed[vid];
                                  const bool is_rigid = (dof_property & Attributions::RIGID_BODY_FLAG) != 0;
                                  const bool is_translation_dof =
                                      (dof_property & Attributions::ABD_Is_Translation_DOF) != 0;

                                  float3 x_prev = sa_q_step_start[vid];
                                  float3 v_prev = sa_q_v[vid];
                                  float3 v_pred = v_prev;


                                  // Only apply gravity to the translation part of rigid body
                                  if ((!is_fixed) & ((!is_rigid) | (is_rigid & is_translation_dof)))
                                  {
                                      v_pred += substep_dt * gravity;
                                  };
                                  float3 x_pred   = x_prev + substep_dt * v_pred;
                                  sa_q_tilde[vid] = x_pred;
                              });
}
void NewtonSolver::host_update_velocity()
{
    CpuParallel::parallel_for(0,
                              host_sim_data->num_dof,
                              [sa_q            = std::span(host_sim_data->sa_q),
                               sa_qv           = std::span(host_sim_data->sa_q_v),
                               sa_q_step_start = std::span(host_sim_data->sa_q_step_start),
                               sa_is_fixed     = std::span(host_sim_data->sa_q_is_fixed),
                               substep_dt      = get_scene_params().get_substep_dt(),
                               fix_scene       = get_scene_params().fix_scene,
                               damping         = get_scene_params().damping_cloth](const uint vid)
                              {
                                  float3 x_step_begin = sa_q_step_start[vid];
                                  float3 x_step_end   = sa_q[vid];

                                  float3 dx  = x_step_end - x_step_begin;
                                  float3 vel = dx / substep_dt;

                                  if (fix_scene)
                                  {
                                      dx        = Zero3;
                                      vel       = Zero3;
                                      sa_q[vid] = x_step_begin;
                                      return;
                                  };

                                  vel *= luisa::exp(-damping * substep_dt);

                                  sa_qv[vid]           = vel;
                                  sa_q_step_start[vid] = x_step_end;
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
    auto& inertia_data = host_sim_data->get_soft_inertia_data();
    if (inertia_data.is_valid())
    {
        CpuParallel::parallel_for(0,
                                  host_sim_data->num_verts_soft,
                                  [sa_x          = std::span(host_sim_data->sa_x),
                                   sa_x_tilde    = std::span(host_sim_data->sa_q_tilde),
                                   sa_q_is_fixed = std::span(host_sim_data->sa_q_is_fixed),
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
                                      // float3 v_0 = sa_v[vid];

                                      float    mass     = sa_vert_mass[vid];
                                      float3   gradient = mass * h_2_inv * (x_k - x_tilde);
                                      float3x3 hessian  = mass * h_2_inv * luisa::float3x3::eye(1.0f);


                                      //   if (sa_q_is_fixed[vid])
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


    auto& abd_data = host_sim_data->get_abd_inertia_data();

    const uint  prefix      = host_sim_data->num_verts_soft;
    const auto& abd_q       = host_sim_data->sa_q;
    const auto& abd_q_tilde = host_sim_data->sa_q_tilde;

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
             abd_q                   = std::span(host_sim_data->sa_q),
             sa_q_is_fixed           = std::span(host_sim_data->sa_q_is_fixed),
             abd_q_tilde             = std::span(host_sim_data->sa_q_tilde)](const uint body_idx)
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

                // if (sa_q_is_fixed[indices[0]])
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
void NewtonSolver::host_evaluate_orthogonality()
{
    auto& abd_data = host_sim_data->get_abd_orthogonality_data();

    if (abd_data.is_valid())
    {
        CpuParallel::parallel_for(
            0,
            abd_data.get_num_indices(),
            [abd_q                   = std::span(host_sim_data->sa_q),
             abd_gradients           = std::span(abd_data.constraint_gradients),
             abd_hessians            = std::span(abd_data.constraint_hessians),
             sa_affine_bodies_kappa  = std::span(abd_data.abd_kappa),
             sa_affine_bodies_volume = std::span(abd_data.abd_volume),
             abd_ortho_indices       = std::span(abd_data.constraint_indices)](const uint body_idx)
            {
                float3   ortho_gradient[3] = {Zero3};
                float3x3 ortho_hessian[6]  = {Zero3x3};
                // float3x3 ortho_hessian[3][3] = {{Zero3x3, Zero3x3, Zero3x3},
                //                                 {Zero3x3, Zero3x3, Zero3x3},
                //                                 {Zero3x3, Zero3x3, Zero3x3}};

                const float substep_dt = get_scene_params().get_substep_dt();
                const float h          = substep_dt;
                const float h_2_inv    = 1.f / (h * h);

                const uint3 indices = abd_ortho_indices[body_idx];
                float3x3 A = luisa::make_float3x3(abd_q[indices[0]], abd_q[indices[1]], abd_q[indices[2]]);

                const float kappa = sa_affine_bodies_kappa[body_idx];
                const float V     = sa_affine_bodies_volume[body_idx];

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
                        // ortho_hessian[ii][jj] = 4.0f * stiff * hessian;
                        idx += 1;
                    }
                }

                //
                //   0   1   2  |  0   3   4
                //  t1   3   4  |  5   1   6
                //  t2  t4   5  |  7   8   2
                auto* body_grad_ptr = &abd_gradients[3 * body_idx];
                auto* body_hess_ptr = &abd_hessians[9 * body_idx];
                body_grad_ptr[0]    = ortho_gradient[0];
                body_grad_ptr[1]    = ortho_gradient[1];
                body_grad_ptr[2]    = ortho_gradient[2];

                body_hess_ptr[0] = ortho_hessian[0];
                body_hess_ptr[1] = ortho_hessian[3];
                body_hess_ptr[2] = ortho_hessian[5];
                body_hess_ptr[3] = ortho_hessian[1];
                body_hess_ptr[4] = ortho_hessian[2];
                body_hess_ptr[5] = transpose(ortho_hessian[1]);
                body_hess_ptr[6] = ortho_hessian[4];
                body_hess_ptr[7] = transpose(ortho_hessian[2]);
                body_hess_ptr[8] = transpose(ortho_hessian[4]);
            },
            32);
    }
}
void NewtonSolver::host_evaluate_dirichlet()
{
    return;
    LUISA_ERROR("Dirichlet BC not implemented on host solver yet.");
}
void NewtonSolver::host_evaluate_ground_collision()
{
    if (!get_scene_params().use_floor)
        return;

    auto calculate_per_vert_grad_hess_template =
        [sa_x                           = std::span(host_sim_data->sa_x),
         sa_x_step_start                = std::span(host_sim_data->sa_x_step_start),
         sa_contact_active_verts_offset = std::span(host_sim_data->sa_contact_active_verts_offset),
         sa_contact_active_verts_d_hat  = std::span(host_sim_data->sa_contact_active_verts_d_hat),
         sa_contact_active_verts_friction_coeff = std::span(host_sim_data->sa_contact_active_verts_friction_coeff),
         sa_is_fixed       = std::span(host_mesh_data->sa_is_fixed),
         sa_rest_vert_area = std::span(host_mesh_data->sa_rest_vert_area),
         floor_y           = get_scene_params().floor.y,
         stiffness_ground  = get_scene_params().stiffness_collision,
         collision_type = get_scene_params().contact_energy_type](const uint vid, float3& out_gradient, float3x3& out_hessian)
    {
        bool collide = false;
        if (!sa_is_fixed[vid] && get_scene_params().use_floor)
        {
            float3 x_k = sa_x[vid];
            float3 x_0 = sa_x_step_start[vid];

            float thickness = sa_contact_active_verts_offset[vid];
            float d_hat     = sa_contact_active_verts_d_hat[vid];
            float curr_dist = x_k.y - floor_y;
            float init_dist = x_0.y - floor_y;

            float3 normal = luisa::make_float3(0, 1, 0);
            float  area   = sa_rest_vert_area[vid];
            float  stiff  = stiffness_ground * area;

            // Repulsion
            if (curr_dist - thickness < d_hat)
            {
                float k1;
                float k2;
                if (collision_type == 0)
                {
                    k1 = stiff * (curr_dist - thickness - d_hat);
                    k2 = stiff;
                }
                else
                {
                    k1 = stiff * ipc::barrier_first_derivative(curr_dist - thickness, d_hat);
                    k2 = stiff * ipc::barrier_second_derivative(curr_dist - thickness, d_hat);
                }
                if (luisa::isnan(k1 * k2) || luisa::isinf(k1 * k2))
                {
                    LUISA_ERROR("NaN detected in ground collision computation: dist = {}, thickness = {}, d_hat = {}, k1 = {}, k2 = {}",
                                curr_dist,
                                thickness,
                                d_hat,
                                k1,
                                k2);
                }
                out_gradient = k1 * normal;
                out_hessian  = k2 * outer_product(normal, normal);
                collide      = true;
            }
            // Friction
            if (init_dist - thickness < d_hat)
            {
                float k1;
                if (collision_type == 0)
                {
                    k1 = stiff * (init_dist - thickness - d_hat);
                }
                else
                {
                    k1 = stiff * ipc::barrier_first_derivative(init_dist - thickness, d_hat);
                }
                float3 rel_dx       = x_k - x_0;
                float  friction_mu  = sa_contact_active_verts_friction_coeff[vid];
                float  friction_eps = Friction::ando_barrier::friction_eps;
                auto   lambda_mu    = -k1 * friction_mu;
                auto   friction_grad_hess =
                    Friction::ipc_barrier::compute_friction_gradient_hessian(lambda_mu, normal, rel_dx, friction_eps);
                out_gradient += friction_grad_hess.first;
                out_hessian = out_hessian + friction_grad_hess.second;
                collide     = true;
            }
        }
        return collide;
    };

    // Shared the gradient and hessian buffer with inertia: using the same per-dof constitution
    auto& inertia_data = host_sim_data->get_soft_inertia_data();
    if (inertia_data.is_valid())
    {
        CpuParallel::parallel_for(0,
                                  inertia_data.get_num_indices(),
                                  [output_gradient = std::span(inertia_data.constraint_gradients),
                                   output_hessian  = std::span(inertia_data.constraint_hessians),
                                   &calculate_per_vert_grad_hess_template](const uint vid)
                                  {
                                      float3   gradient = Zero3;
                                      float3x3 hessian  = Zero3x3;
                                      bool collide = calculate_per_vert_grad_hess_template(vid, gradient, hessian);
                                      if (collide)
                                      {
                                          buffer_add(output_gradient, vid, gradient);
                                          buffer_add(output_hessian, vid, hessian);
                                      }
                                  });
    }

    const uint prefix   = host_sim_data->num_verts_soft;
    auto&      abd_data = host_sim_data->get_abd_inertia_data();

    if (abd_data.is_valid())
    {
        CpuParallel::parallel_for(
            0,
            host_sim_data->num_verts_rigid,
            [output_gradient   = std::span(abd_data.constraint_gradients),
             output_hessian    = std::span(abd_data.constraint_hessians),
             sa_scaled_model_x = std::span(host_sim_data->sa_scaled_model_x),
             sa_x_to_dof_map   = std::span(host_sim_data->sa_x_to_dof_map),
             prefix_vid        = host_sim_data->num_verts_soft,
             &calculate_per_vert_grad_hess_template](const uint index)
            {
                float3     gradient = Zero3;
                float3x3   hessian  = Zero3x3;
                const uint vid      = prefix_vid + index;
                bool       collide  = calculate_per_vert_grad_hess_template(vid, gradient, hessian);
                if (collide)
                {
                    const uint dof_idx  = sa_x_to_dof_map[vid] & Attributions::RIGID_BODY_MASK;
                    const uint body_idx = (dof_idx - prefix_vid) / 4;
                    float3     model_x  = sa_scaled_model_x[vid];
                    float4     weight   = luisa::make_float4(1.0f, model_x);
                    uint       idx      = 4;
                    for (uint ii = 0; ii < 4; ii++)
                    {
                        float  wi          = weight[ii];
                        float3 affine_grad = wi * gradient;
                        CpuParallel::spin_atomic<float3>::fetch_add(output_gradient[4 * body_idx + ii], affine_grad);
                        for (uint jj = 0; jj < 4; jj++)
                        {
                            float    wj          = weight[jj];
                            float3x3 affine_hess = wi * wj * hessian;
                            if (ii == jj)
                            {
                                atomic_buffer_add(output_hessian, 16 * body_idx + ii, affine_hess);
                            }
                            else
                            {
                                atomic_buffer_add(output_hessian, 16 * body_idx + idx, affine_hess);
                                idx += 1;
                            }
                        }
                    }
                }
            });
    }
}
void NewtonSolver::host_test_dynamics(luisa::compute::Stream& stream)
{
    stream << luisa::compute::synchronize();

    const uint                 num_dof    = host_sim_data->num_dof;
    const uint                 num_bodies = host_sim_data->num_affine_bodies;
    Eigen::SparseMatrix<float> cgA(3 * num_dof, 3 * num_dof);
    Eigen::MatrixXf            cgB(3 * num_dof, 1);
    cgA.setZero();
    cgB.setZero();

    // const uint prefix = host_sim_data->num_verts_soft;

    // Soft inertia
    // if constexpr (false)
    {
        std::vector<EigenTripletBlock<1>> hessian_blocks(host_sim_data->num_verts_soft);
        CpuParallel::single_thread_for(0,
                                       host_sim_data->num_verts_soft,
                                       [sa_cgB              = std::span(host_sim_data->sa_cgB),
                                        sa_cgA_diag         = std::span(host_sim_data->sa_cgA_diag),
                                        sa_x                = std::span(host_sim_data->sa_q),
                                        sa_x_tilde          = std::span(host_sim_data->sa_q_tilde),
                                        sa_is_fixed         = std::span(host_mesh_data->sa_is_fixed),
                                        sa_vert_mass        = std::span(host_mesh_data->sa_vert_mass),
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
        const auto& stretch_faces = host_sim_data->get_stretch_face_data();

        std::vector<EigenTripletBlock<3>> hessian_blocks(stretch_faces.get_num_indices());
        CpuParallel::single_thread_for(
            0,
            stretch_faces.get_num_indices(),
            [sa_x                       = std::span(host_sim_data->sa_x),
             sa_rest_x                  = std::span(host_mesh_data->sa_rest_x),
             sa_faces                   = std::span(stretch_faces.constraint_indices),
             sa_stretch_faces_rest_area = std::span(stretch_faces.sa_stretch_faces_rest_area),
             sa_stretch_faces_Dm_inv    = std::span(stretch_faces.sa_stretch_faces_Dm_inv),
             sa_stretch_faces_mu_lambda = std::span(stretch_faces.sa_stretch_faces_mu_lambda),
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
    // if constexpr (false)
    {
        const auto& stretch_springs = host_sim_data->get_stretch_spring_data();

        std::vector<EigenTripletBlock<2>> hessian_blocks(stretch_springs.get_num_indices());
        CpuParallel::single_thread_for(
            0,
            stretch_springs.get_num_indices(),
            [sa_x           = std::span(host_sim_data->sa_x),
             sa_edges       = std::span(stretch_springs.constraint_indices),
             sa_rest_length = std::span(stretch_springs.sa_stretch_spring_rest_state_length),
             sa_stretch_spring_stiffness = std::span(stretch_springs.sa_stretch_spring_stiffness),
             output_gradient_ptr         = std::span(stretch_springs.constraint_gradients),
             output_hessian_ptr          = std::span(stretch_springs.constraint_hessians),
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
        const auto& bending_edges = host_sim_data->get_bending_edge_data();

        std::vector<EigenTripletBlock<4>> hessian_blocks(bending_edges.get_num_indices());
        CpuParallel::single_thread_for(
            0,
            bending_edges.get_num_indices(),
            [sa_x                        = host_sim_data->sa_x.data(),
             sa_bending_edges            = std::span(bending_edges.constraint_indices),
             sa_bending_edges_Q          = std::span(bending_edges.sa_bending_edges_Q),
             sa_bending_edges_rest_angle = std::span(bending_edges.sa_bending_edges_rest_angle),
             sa_bending_edges_rest_area  = std::span(bending_edges.sa_bending_edges_rest_area),
             sa_bending_edges_stiffness  = std::span(bending_edges.sa_bending_edges_stiffness),
             output_gradient_ptr         = std::span(bending_edges.constraint_gradients),
             output_hessian_ptr          = std::span(bending_edges.constraint_hessians),
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

    // Soft ground collision
    // if constexpr (false)
    if (get_scene_params().use_floor)
    {
        std::vector<EigenTripletBlock<1>> hessian_blocks(host_sim_data->num_verts_soft);
        CpuParallel::parallel_for(
            0,
            host_sim_data->num_verts_soft,
            [sa_x                           = host_sim_data->sa_x.data(),
             sa_contact_active_verts_offset = host_sim_data->sa_contact_active_verts_offset.data(),
             sa_contact_active_verts_d_hat  = host_sim_data->sa_contact_active_verts_d_hat.data(),
             sa_is_fixed                    = host_mesh_data->sa_is_fixed.data(),
             sa_rest_vert_area              = host_mesh_data->sa_rest_vert_area.data(),
             &cgB,
             &hessian_blocks](const uint vid)
            {
                float3 x_k  = sa_x[vid];
                float  dist = x_k.y - get_scene_params().floor.y;

                const float d_hat            = sa_contact_active_verts_d_hat[vid];
                const float thickness        = sa_contact_active_verts_offset[vid];
                const float stiffness_ground = get_scene_params().stiffness_collision;
                const uint  collision_type   = get_scene_params().contact_energy_type;

                if (dist - thickness < d_hat)
                {
                    float3 normal = luisa::make_float3(0, 1, 0);
                    float  area   = sa_rest_vert_area[vid];
                    float  stiff  = stiffness_ground * area;

                    float k1;
                    float k2;
                    if (collision_type == 0)
                    {
                        k1 = stiff * (dist - thickness - d_hat);
                        k2 = stiff;
                    }
                    else
                    {
                        k1 = stiff * ipc::barrier_first_derivative(dist - thickness, d_hat);
                        k2 = stiff * ipc::barrier_second_derivative(dist - thickness, d_hat);
                    }
                    float3   grad    = k1 * normal;
                    float3x3 hessian = k2 * outer_product(normal, normal);

                    cgB.block<3, 1>(vid * 3, 0) -= float3_to_eigen3(grad);
                    hessian_blocks[vid] = {.indices = {vid}, .matrix = float3x3_to_eigen3x3(hessian)};
                }
            });

        convert_triplets_to_sparse_matrix(cgA, hessian_blocks);
    }

    // ABD Ground Collision
    // if constexpr (false)
    if (get_scene_params().use_floor)
    {
        const uint  prefix   = host_sim_data->num_verts_soft;
        const auto& abd_data = host_sim_data->get_abd_inertia_data();

        std::vector<EigenTripletBlock<4>> hessian_blocks(abd_data.get_num_indices());
        for (uint body_idx = 0; body_idx < abd_data.get_num_indices(); body_idx++)
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

        // const float d_hat     = get_scene_params().d_hat;
        // const float thickness = get_scene_params().thickness;
        CpuParallel::single_thread_for(
            0,
            abd_data.get_num_indices(),
            [&](const uint body_idx)
            {
                if (get_scene_params().use_floor)
                {
                    const uint mesh_idx    = host_sim_data->sa_affine_bodies_mesh_id[body_idx];
                    const uint curr_prefix = host_mesh_data->prefix_num_verts[mesh_idx];
                    const uint next_prefix = host_mesh_data->prefix_num_verts[mesh_idx + 1];

                    for (uint vid = curr_prefix; vid < next_prefix; vid++)
                    {
                        float3      x_k       = host_sim_data->sa_x[vid];
                        float       dist      = x_k.y - get_scene_params().floor.y;
                        const float d_hat     = host_sim_data->sa_contact_active_verts_d_hat[vid];
                        const float thickness = host_sim_data->sa_contact_active_verts_offset[vid];

                        if (dist - thickness < d_hat)
                        {
                            const float stiffness_ground = get_scene_params().stiffness_collision;
                            const uint  collision_type   = get_scene_params().contact_energy_type;

                            float3 normal  = luisa::make_float3(0, 1, 0);
                            float  area    = host_mesh_data->sa_rest_vert_area[vid];
                            float  stiff   = stiffness_ground * area;
                            float3 model_x = host_mesh_data->sa_scaled_model_x[vid];

                            float k1;
                            float k2;
                            if (collision_type == 0)
                            {
                                k1 = stiff * (dist - thickness - d_hat);
                                k2 = stiff;
                            }
                            else
                            {
                                k1 = stiff * ipc::barrier_first_derivative(dist - thickness, d_hat);
                                k2 = stiff * ipc::barrier_second_derivative(dist - thickness, d_hat);
                            }
                            float3   gradient = k1 * normal;
                            float3x3 hessian  = k2 * outer_product(normal, normal);

                            auto J = AffineBodyDynamics::get_jacobian_dxdq(model_x);
                            cgB.block<12, 1>(body_idx * 12, 0) -= J.transpose() * float3_to_eigen3(gradient);
                            hessian_blocks[body_idx].matrix += J.transpose() * float3x3_to_eigen3x3(hessian) * J;
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
    // if constexpr (false)
    {
        const uint  prefix      = host_sim_data->num_verts_soft;
        const auto& abd_q       = host_sim_data->sa_q;
        const auto& abd_q_tilde = host_sim_data->sa_q_tilde;
        const auto& abd_data    = host_sim_data->get_abd_inertia_data();

        std::vector<EigenTripletBlock<4>> hessian_blocks(abd_data.get_num_indices());

        CpuParallel::single_thread_for(
            0,
            abd_data.get_num_indices(),
            [&](const uint body_idx)
            {
                const float substep_dt = get_scene_params().get_substep_dt();
                const float h          = substep_dt;
                const float h_2_inv    = 1.f / (h * h);

                auto         M     = abd_data.sa_affine_bodies_mass_matrix_full[body_idx];
                EigenFloat12 delta = EigenFloat12::Zero();

                const uint4 indices     = abd_data.constraint_indices[body_idx];
                delta.block<3, 1>(0, 0) = float3_to_eigen3(abd_q[indices[0]] - abd_q_tilde[indices[0]]);
                delta.block<3, 1>(3, 0) = float3_to_eigen3(abd_q[indices[1]] - abd_q_tilde[indices[1]]);
                delta.block<3, 1>(6, 0) = float3_to_eigen3(abd_q[indices[2]] - abd_q_tilde[indices[2]]);
                delta.block<3, 1>(9, 0) = float3_to_eigen3(abd_q[indices[3]] - abd_q_tilde[indices[3]]);

                EigenFloat12    gradient = h_2_inv * M * delta;
                EigenFloat12x12 hessian  = h_2_inv * M;

                // std::cout << "Body " << body_idx << " inertia gradient: " << std::endl
                //           << gradient << std::endl;
                // std::cout << "Body " << body_idx << " inertia hessian: " << std::endl
                //           << hessian << std::endl;

                hessian_blocks[body_idx] = {.indices = {indices[0], indices[1], indices[2], indices[3]},
                                            .matrix  = hessian};
                cgB.block<12, 1>(body_idx * 12, 0) -= gradient;
                // cgA.block<12, 12>(body_idx * 12, body_idx * 12) += hessian;
            });

        convert_triplets_to_sparse_matrix(cgA, hessian_blocks);
    }

    // Orthogonality
    // if constexpr (false)
    {
        const auto& abd_data                = host_sim_data->get_abd_orthogonality_data();
        const auto& abd_q                   = host_sim_data->sa_q;
        const auto& abd_q_tilde             = host_sim_data->sa_q_tilde;
        const auto& abd_ortho_indices       = abd_data.constraint_indices;
        const auto& sa_affine_bodies_kappa  = abd_data.abd_kappa;
        const auto& sa_affine_bodies_volume = abd_data.abd_volume;
        const uint  prefix                  = host_sim_data->num_verts_soft;

        std::vector<EigenTripletBlock<4>> hessian_blocks(abd_data.get_num_indices());

        CpuParallel::single_thread_for(
            0,
            abd_data.get_num_indices(),
            [&](const uint body_idx)
            {
                // Orthogonality potential
                // if constexpr (false)
                {
                    uint3 indices = abd_ortho_indices[body_idx];
                    float3x3 A = luisa::make_float3x3(abd_q[indices[0]], abd_q[indices[1]], abd_q[indices[2]]);

                    const float kappa = sa_affine_bodies_kappa[body_idx];
                    const float V     = sa_affine_bodies_volume[body_idx];

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
    // if constexpr (false)
    {
        narrow_phase_detector->download_narrowphase_list(stream);

        // Host collision detection
        if (false && host_sim_data->num_dof < 32)
        {
            uint               num_pairs   = 0;
            std::atomic<uint>* atomic_view = (std::atomic<uint>*)&num_pairs;

            host_collision_data->narrow_phase_list.resize(collision_data->narrow_phase_list.size());

            CpuParallel::parallel_for(
                0,
                host_mesh_data->num_verts * host_mesh_data->num_faces,
                [&](const uint i)
                {
                    const uint vid = i / host_mesh_data->sa_faces.size();
                    const uint fid = i % host_mesh_data->sa_faces.size();

                    uint3 face = host_mesh_data->sa_faces[fid];
                    if (vid == face[0] || vid == face[1] || vid == face[2])
                        return;

                    float3 p    = host_sim_data->sa_x[vid];
                    float3 t0   = host_sim_data->sa_x[face[0]];
                    float3 t1   = host_sim_data->sa_x[face[1]];
                    float3 t2   = host_sim_data->sa_x[face[2]];
                    auto   bary = host_distance::point_triangle_distance_coeff_unclassified(
                        float3_to_eigen3(p), float3_to_eigen3(t0), float3_to_eigen3(t1), float3_to_eigen3(t2));

                    float3 x         = bary[0] * (p - t0) + bary[1] * (p - t1) + bary[2] * (p - t2);
                    float  d2        = length_squared_vec(x);
                    float  d_hat1    = host_sim_data->sa_contact_active_verts_d_hat[vid];
                    float  d_hat2    = host_sim_data->sa_contact_active_verts_d_hat[face[0]];
                    float  offset1   = host_sim_data->sa_contact_active_verts_offset[vid];
                    float  offset2   = host_sim_data->sa_contact_active_verts_offset[face[0]];
                    float  d_hat     = (d_hat1 + d_hat2) * 0.5f;
                    float  thickness = offset1 + offset2;

                    if (d2 < (d_hat + thickness) * (d_hat + thickness))
                    {
                        float  d      = sqrt_scalar(d2);
                        float3 normal = x / d;

                        float k1;
                        float k2;
                        float avg_area = 1.0f;

                        {
                            float area_a = host_mesh_data->sa_rest_vert_area[vid];
                            float area_b = host_mesh_data->sa_rest_face_area[fid];
                            avg_area     = 0.5f * (area_a + area_b);
                        }

                        const auto contact_energy_type = ContactEnergyType(get_scene_params().contact_energy_type);
                        const float kappa = get_scene_params().stiffness_collision;
                        if (contact_energy_type == ContactEnergyType::Quadratic)
                        {
                            float C     = thickness + d_hat - d;
                            float stiff = kappa * avg_area;
                            k1          = stiff * C;
                            k2          = stiff;
                        }
                        else if (contact_energy_type == ContactEnergyType::Barrier)
                        {
                            // float dBdD;
                            // float ddBddD;
                            // cipc::dKappaBarrierdD(dBdD, avg_area * kappa, d2, d_hat, thickness);
                            // cipc::ddKappaBarrierddD(ddBddD, avg_area * kappa, d2, d_hat, thickness);
                            // k1     = -dBdD;
                            // k2     = ddBddD;
                            // normal = 2.0f * x;
                            k1 = -avg_area * kappa * ipc::barrier_first_derivative(d - thickness, d_hat);
                            k2 = avg_area * kappa * ipc::barrier_second_derivative(d - thickness, d_hat);

                            // LUISA_INFO("Pair {}: k1 = {}, k2 = {}, d = {}, thickness = {}, d_hat = {}, area = {}, kappa = {}",
                            //            num_pairs,
                            //            k1,
                            //            k2,
                            //            d,
                            //            thickness,
                            //            d_hat,
                            //            avg_area,
                            //            kappa);
                        }
                        {
                            uint                                 idx = atomic_view->fetch_add(1);
                            CollisionPair::CollisionPairTemplate vf_pair;
                            vf_pair.make_vf_pair(luisa::make_uint4(vid, face[0], face[1], face[2]),
                                                 normal,
                                                 k1,
                                                 k2,
                                                 avg_area * kappa,
                                                 eigen3_to_float3(bary));
                            host_collision_data->narrow_phase_list[idx] = vf_pair;
                            // num_pairs += 1;
                            // LUISA_INFO("VF Pair {}: dist = {}, force = {}, idx = {}",
                            //            idx,
                            //            d,
                            //            k1 * normal,
                            //            vf_pair.get_indices());
                        }
                    }
                });
            CpuParallel::parallel_for(
                0,
                host_mesh_data->num_edges * host_mesh_data->num_edges,
                [&](const uint i)
                {
                    const uint eid1 = i / host_mesh_data->num_edges;
                    const uint eid2 = i % host_mesh_data->num_edges;

                    uint2  left_edge = host_mesh_data->sa_edges[eid1];
                    float3 ea_p0     = host_sim_data->sa_x[left_edge[0]];
                    float3 ea_p1     = host_sim_data->sa_x[left_edge[1]];

                    uint2 right_edge = host_mesh_data->sa_edges[eid2];
                    if (left_edge[0] == right_edge[0] || left_edge[0] == right_edge[1]
                        || left_edge[1] == right_edge[0] || left_edge[1] == right_edge[1])
                        return;

                    float3 eb_p0 = host_sim_data->sa_x[right_edge[0]];
                    float3 eb_p1 = host_sim_data->sa_x[right_edge[1]];
                    auto   bary =
                        host_distance::edge_edge_distance_coeff_unclassified(float3_to_eigen3(ea_p0),
                                                                             float3_to_eigen3(ea_p1),
                                                                             float3_to_eigen3(eb_p0),
                                                                             float3_to_eigen3(eb_p1));

                    float3 x0 = bary[0] * ea_p0 + bary[1] * ea_p1;
                    float3 x1 = bary[2] * eb_p0 + bary[3] * eb_p1;
                    float3 x  = x0 - x1;
                    float  d2 = length_squared_vec(x);

                    float d_hat1    = host_sim_data->sa_contact_active_verts_d_hat[left_edge[0]];
                    float d_hat2    = host_sim_data->sa_contact_active_verts_d_hat[right_edge[0]];
                    float offset1   = host_sim_data->sa_contact_active_verts_offset[left_edge[0]];
                    float offset2   = host_sim_data->sa_contact_active_verts_offset[right_edge[0]];
                    float d_hat     = (d_hat1 + d_hat2) * 0.5f;
                    float thickness = offset1 + offset2;

                    if (d2 < (d_hat + thickness) * (d_hat + thickness))
                    {
                        float  d      = sqrt_scalar(d2);
                        float3 normal = x / d;

                        float k1;
                        float k2;
                        float avg_area = 1.0f;

                        {
                            float area_a = host_mesh_data->sa_rest_edge_area[eid1];
                            float area_b = host_mesh_data->sa_rest_edge_area[eid2];
                            avg_area     = 0.5f * (area_a + area_b);
                        }

                        const auto contact_energy_type = ContactEnergyType(get_scene_params().contact_energy_type);
                        const float kappa = get_scene_params().stiffness_collision;
                        if (contact_energy_type == ContactEnergyType::Quadratic)
                        {
                            float C     = thickness + d_hat - d;
                            float stiff = kappa * avg_area;
                            k1          = stiff * C;
                            k2          = stiff;
                        }
                        else if (contact_energy_type == ContactEnergyType::Barrier)
                        {
                            // float dBdD;
                            // float ddBddD;
                            // cipc::dKappaBarrierdD(dBdD, avg_area * kappa, d2, d_hat, thickness);
                            // cipc::ddKappaBarrierddD(ddBddD, avg_area * kappa, d2, d_hat, thickness);
                            // k1     = -dBdD;
                            // k2     = ddBddD;
                            // normal = 2.0f * x;
                            k1 = -avg_area * kappa * ipc::barrier_first_derivative(d - thickness, d_hat);
                            k2 = avg_area * kappa * ipc::barrier_second_derivative(d - thickness, d_hat);
                            // k1 = -avg_area * kappa * squared_ipc::barrier_first_derivative(d - thickness, d_hat);
                            // k2 = avg_area * kappa * squared_ipc::barrier_second_derivative(d - thickness, d_hat);
                            // LUISA_INFO("Pair {}: k1 = {}, k2 = {}, d = {}, thickness = {}, d_hat = {}, area = {}, kappa = {}",
                            //            num_pairs,
                            //            k1,
                            //            k2,
                            //            d,
                            //            thickness,
                            //            d_hat,
                            //            avg_area,
                            //            kappa);
                        }
                        {
                            uint                                 idx = atomic_view->fetch_add(1);
                            CollisionPair::CollisionPairTemplate ee_pair;
                            auto                                 tmp_bary = eigen4_to_float4(bary);
                            ee_pair.make_ee_pair(
                                luisa::make_uint4(left_edge[0], left_edge[1], right_edge[0], right_edge[1]),
                                normal,
                                k1,
                                k2,
                                avg_area * kappa,
                                tmp_bary.xy(),
                                tmp_bary.zw());
                            host_collision_data->narrow_phase_list[idx] = ee_pair;
                            // num_pairs += 1;
                            // LUISA_INFO("EE Pair {}: dist = {}, force = {}, idx = {}",
                            //            idx,
                            //            d,
                            //            k1 * normal,
                            //            ee_pair.get_indices());
                        }
                    }
                });
            host_collision_data->narrow_phase_collision_count.front() = num_pairs;
        }

        const auto& host_count = host_collision_data->narrow_phase_collision_count;
        const uint  num_pairs  = host_count.front();
        const uint  prefix     = host_sim_data->num_verts_soft;

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

            D = luisa::length(weight[0] * positions[0] + weight[1] * positions[1]
                              + weight[2] * positions[2] + weight[3] * positions[3]);

            float d_hat = 0.5f
                          * (host_sim_data->sa_contact_active_verts_d_hat[indices[0]]
                             + host_sim_data->sa_contact_active_verts_d_hat[indices[2]]);
            float thickness = (host_sim_data->sa_contact_active_verts_offset[indices[0]]
                               + host_sim_data->sa_contact_active_verts_offset[indices[2]]);

            // LUISA_INFO("Contact pair {}: D = {:.6f}, (d_hat = {}, thickness = {}), indices = {}, force = {}",
            //            pair_idx,
            //            D,
            //            d_hat,
            //            thickness,
            //            indices,
            //            k1 * normal);

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

                    EigenFloat3     force = float3_to_eigen3(k1 * weight[ii] * normal);
                    Eigen::MatrixXf J1;  // = Eigen::Matrix<float, 3, 12>::Zero();
                    if (is_rigid[ii])
                    {
                        const float3 Xi   = host_mesh_data->sa_scaled_model_x[indices[ii]];
                        J1                = AffineBodyDynamics::get_jacobian_dxdq(Xi);
                        const uint body_i = host_sim_data->sa_vert_affine_bodies_id[indices[ii]];

                        cgB.block<12, 1>(3 * prefix + body_i * 12, 0) += J1.transpose() * force;
                        left_indices = {
                            prefix + 4 * body_i + 0,
                            prefix + 4 * body_i + 1,
                            prefix + 4 * body_i + 2,
                            prefix + 4 * body_i + 3,
                        };
                        // LUISA_INFO("Apply contact force to rigid vertex {}, force = {}, k1/k2 = {}/{}",
                        //            indices[ii],
                        //            force,
                        //            k1,
                        //            k2);
                    }
                    else
                    {
                        J1 = EigenFloat3x3::Identity();
                        cgB.block<3, 1>(indices[ii] * 3, 0) += force;
                        left_indices = {indices[ii]};

                        // LUISA_INFO("Apply contact force to soft vertex {}, force = {}", indices[ii], force);
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
        convert_triplets_to_sparse_matrix(cgA, hessian_blocks);
    }

    // Validate contact gradient and hessian
    if constexpr (false)
    {
        stream << fn_reset_vector(sim_data->sa_cgB).dispatch(num_dof)
               << fn_reset_float3x3(sim_data->sa_cgA_diag).dispatch(num_dof);
        narrow_phase_detector->device_perPair_evaluate_gradient_hessian(stream,
                                                                        sim_data->sa_x,
                                                                        sim_data->sa_x_step_start,
                                                                        sim_data->sa_contact_active_verts_friction_coeff,
                                                                        sim_data->sa_contact_active_verts_d_hat,
                                                                        sim_data->sa_contact_active_verts_offset,
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
        CpuParallel::parallel_copy(host_collision_data->triplet_data.sa_cgA_contact_offdiag_triplet,
                                   hessian_blocks,
                                   triplet_count);
        Eigen::SparseMatrix<float> cgA_contact_offdiag(num_dof * 3, num_dof * 3);
        cgA_contact_offdiag.setZero();
        convert_triplets_to_sparse_matrix(cgA_contact_offdiag, hessian_blocks);
        cgA += cgA_contact_offdiag;
    }

    Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower> solver;  // Eigen::IncompleteCholesky<float>
    solver.setTolerance(1e-10f);
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
    LUISA_INFO("  In newton iter {:2}, EigenSolve error = {:7.6f}, max_element(p) = {:6.5f}",
               get_scene_params().current_nonlinear_iter,
               error,
               fast_infinity_norm(host_sim_data->sa_cgX));
    if (luisa::isnan(error) || luisa::isinf(error))
    {
        LUISA_ERROR("NaN/INF detected in Eigen PCG solve!");
    }
}
void NewtonSolver::host_evaluete_stretch_spring()
{
    auto& stretch_springs = host_sim_data->get_stretch_spring_data();

    CpuParallel::parallel_for(0,
                              stretch_springs.get_num_indices(),
                              [sa_x     = std::span(host_sim_data->sa_x),
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
    auto& stretch_faces = host_sim_data->get_stretch_face_data();

    CpuParallel::parallel_for(
        0,
        stretch_faces.get_num_indices(),
        [sa_x                       = std::span(host_sim_data->sa_x),
         sa_faces                   = std::span(stretch_faces.constraint_indices),
         sa_stretch_faces_rest_area = std::span(stretch_faces.sa_stretch_faces_rest_area),
         sa_stretch_faces_Dm_inv    = std::span(stretch_faces.sa_stretch_faces_Dm_inv),
         sa_stretch_faces_mu_lambda = std::span(stretch_faces.sa_stretch_faces_mu_lambda),
         output_gradient_ptr        = std::span(stretch_faces.constraint_gradients),
         output_hessian_ptr         = std::span(stretch_faces.constraint_hessians)](const uint fid)
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
    auto& bending_edges = host_sim_data->get_bending_edge_data();

    CpuParallel::parallel_for(
        0,
        bending_edges.get_num_indices(),
        [sa_x                        = std::span(host_sim_data->sa_x),
         sa_bending_edges            = std::span(bending_edges.constraint_indices),
         sa_bending_edges_Q          = std::span(bending_edges.sa_bending_edges_Q),
         sa_bending_edges_rest_angle = std::span(bending_edges.sa_bending_edges_rest_angle),
         sa_bending_edges_rest_area  = std::span(bending_edges.sa_bending_edges_rest_area),
         sa_bending_edges_stiffness  = std::span(bending_edges.sa_bending_edges_stiffness),
         output_gradient_ptr         = std::span(bending_edges.constraint_gradients),
         output_hessian_ptr          = std::span(bending_edges.constraint_hessians),
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


template <typename Derived>
void assembly_template2(const uint                                                  vid,
                        Constitutions::ConstitutionInterface<std::vector, Derived>& constaints,
                        std::span<uint>&             sa_vert_adj_material_force_verts_csr,
                        std::span<float3>&           sa_cgB,
                        std::span<float3x3>&         sa_cgA_diag,
                        std::span<MatrixTriplet3x3>& sa_cgA_offdiag_triplet)
{
    // constexpr size_t N = constaints.get_num_verts_per_constaint();
    constexpr size_t N = Derived::get_num_verts_per_constaint();

    // auto& sa_vert_adj_material_force_verts_csr = constaints.get_vert_adj_material_force_verts_csr();
    auto& vert_adj_constraints_csr     = constaints.get_vert_adj_constraints_csr();
    auto& constaint_gradients          = constaints.get_constraint_gradients();
    auto& constaint_hessians           = constaints.get_constraint_hessians();
    auto& constaint_offsets_in_adjlist = constaints.get_constraint_offsets_in_adjlist();
    auto& indices                      = constaints.get_indices();

    //  sa_vert_adj_material_force_verts_csr = std::span(host_sim_data->sa_vert_adj_material_force_verts_csr),
    //  sa_cgB      = std::span(host_sim_data->sa_cgB),
    //  sa_cgA_diag = std::span(host_sim_data->sa_cgA_diag),
    //  sa_cgA_offdiag_triplet = std::span(host_sim_data->sa_cgA_fixtopo_offdiag_triplet)

    const uint curr_prefix = sa_vert_adj_material_force_verts_csr[vid];
    const uint next_prefix = sa_vert_adj_material_force_verts_csr[vid + 1];

    const uint curr_prefix_bending = vert_adj_constraints_csr[vid];
    const uint next_prefix_bending = vert_adj_constraints_csr[vid + 1];

    for (uint j = curr_prefix_bending; j < next_prefix_bending; j++)
    {
        const uint adj_eid = vert_adj_constraints_csr[j];
        const auto edge    = indices[adj_eid];
        uint       offset  = -1u;
        if constexpr (N == 1)
        {
            offset = 0;
        }
        else
        {
            for (uint k = 0; k < N; k++)
            {
                if (vid == edge[k])
                {
                    offset = k;
                }
            }
        }
        if (offset == -1u)
        {
            LUISA_ERROR("Error in assembly: Can not find vertex {} in constraint {}, eid = {}, elements = {}",
                        vid,
                        constaints.get_constitution_name(),
                        adj_eid,
                        edge);
        }
        // LUISA_ASSERT(offset != -1u, "Error in assembly: offset not found.");

        const float3&   grad      = constaint_gradients[adj_eid * N + offset];
        const float3x3& diag_hess = constaint_hessians[adj_eid * (N * N) + offset];

        if (is_nan_vec(grad) || is_inf_vec(grad))
        {
            LUISA_ERROR("NaN/INF detected in assembly: grad of constraint {} (eid = {}, vertex = {}) is {}",
                        constaints.get_constitution_name(),
                        adj_eid,
                        vid,
                        grad);
        }

        buffer_add(sa_cgB, vid, -grad);
        buffer_add(sa_cgA_diag, vid, diag_hess);

        if constexpr (N != 1)
        {
            constexpr uint N_off = N - 1;
            for (uint ii = 0; ii < N_off; ii++)  // For each off-diagonal in curr row
            {
                float3x3 offdiag_hess = constaint_hessians[adj_eid * (N * N) + N + offset * N_off + ii];
                uint offdiag_offset = constaint_offsets_in_adjlist[adj_eid * (N * N_off) + offset * N_off + ii];
                uint triplet_idx = curr_prefix + offdiag_offset;
                if (triplet_idx >= sa_cgA_offdiag_triplet.size())
                {
                    LUISA_ERROR("Error in assembly: triplet_idx {} out of range (size = {})",
                                triplet_idx,
                                sa_cgA_offdiag_triplet.size());
                }
                auto& triplet = sa_cgA_offdiag_triplet[curr_prefix + offdiag_offset];
                add_triplet_matrix(triplet, offdiag_hess);
            }
        }
    };
};

void NewtonSolver::host_material_energy_assembly()
{
    // Assemble material forces and stiffness matrix
    {
        auto sa_cgB                 = std::span(host_sim_data->sa_cgB);
        auto sa_cgA_diag            = std::span(host_sim_data->sa_cgA_diag);
        auto sa_cgA_offdiag_triplet = std::span(host_sim_data->sa_cgA_fixtopo_offdiag_triplet);
        auto adj_verts              = std::span(host_sim_data->sa_vert_adj_material_force_verts_csr);

        const uint num_dof_soft     = host_sim_data->num_verts_soft;
        const uint num_dof_rigid    = host_sim_data->num_affine_bodies * 4;
        const uint prefix_dof_rigid = num_dof_soft;

        auto& soft_inertia = host_sim_data->get_soft_inertia_data();
        if (soft_inertia.is_valid())
            CpuParallel::parallel_for(0,
                                      num_dof_soft,
                                      [&](const uint vid) {
                                          assembly_template2(vid, soft_inertia, adj_verts, sa_cgB, sa_cgA_diag, sa_cgA_offdiag_triplet);
                                      });

        auto& stretch_springs = host_sim_data->get_stretch_spring_data();
        if (stretch_springs.is_valid())
            CpuParallel::parallel_for(0,
                                      num_dof_soft,
                                      [&](const uint vid) {
                                          assembly_template2(vid, stretch_springs, adj_verts, sa_cgB, sa_cgA_diag, sa_cgA_offdiag_triplet);
                                      });

        auto& stretch_faces = host_sim_data->get_stretch_face_data();
        if (stretch_faces.is_valid())
            CpuParallel::parallel_for(0,
                                      num_dof_soft,
                                      [&](const uint vid) {
                                          assembly_template2(vid, stretch_faces, adj_verts, sa_cgB, sa_cgA_diag, sa_cgA_offdiag_triplet);
                                      });

        auto& bending_edges = host_sim_data->get_bending_edge_data();
        if (bending_edges.is_valid())
            CpuParallel::parallel_for(0,
                                      num_dof_soft,
                                      [&](const uint vid) {
                                          assembly_template2(vid, bending_edges, adj_verts, sa_cgB, sa_cgA_diag, sa_cgA_offdiag_triplet);
                                      });

        auto& abd_inertia = host_sim_data->get_abd_inertia_data();
        if (abd_inertia.is_valid())
            CpuParallel::parallel_for(0,
                                      num_dof_rigid,
                                      [&](const uint block_idx)
                                      {
                                          const uint vid = prefix_dof_rigid + block_idx;
                                          assembly_template2(vid, abd_inertia, adj_verts, sa_cgB, sa_cgA_diag, sa_cgA_offdiag_triplet);
                                      });

        auto& abd_ortho = host_sim_data->get_abd_orthogonality_data();
        if (abd_ortho.is_valid())
            CpuParallel::parallel_for(0,
                                      num_dof_rigid,
                                      [&](const uint block_idx)
                                      {
                                          const uint vid = prefix_dof_rigid + block_idx;
                                          assembly_template2(vid, abd_ortho, adj_verts, sa_cgB, sa_cgA_diag, sa_cgA_offdiag_triplet);
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
    narrow_phase_detector->reset_toi(stream);

    stream << fn_gound_collision_ccd(get_scene_params().floor.y, get_scene_params().use_floor)
                  .dispatch(sim_data->sa_x.size());

    // stream << collision_data->toi_per_vert.view(0, 1).copy_to(host_collision_data->toi_per_vert.data())
    //        << luisa::compute::synchronize();
    // LUISA_INFO("  Min TOI after ground collision check: {:7.6f}", host_collision_data->toi_per_vert.front());

    if (get_scene_params().use_self_collision)
    {
        narrow_phase_detector->vf_ccd_query(stream,
                                            sim_data->sa_x_iter_start,
                                            sim_data->sa_x,
                                            mesh_data->sa_faces,
                                            sim_data->sa_vert_affine_bodies_id,
                                            sim_data->sa_contact_active_verts_d_hat,
                                            sim_data->sa_contact_active_verts_offset);

        // stream << collision_data->toi_per_vert.view(0, 1).copy_to(host_collision_data->toi_per_vert.data())
        //        << luisa::compute::synchronize();
        // LUISA_INFO("  Min TOI after VF CCD check: {:7.6f}", host_collision_data->toi_per_vert.front());

        narrow_phase_detector->ee_ccd_query(stream,
                                            sim_data->sa_x_iter_start,
                                            sim_data->sa_x,
                                            mesh_data->sa_edges,
                                            sim_data->sa_vert_affine_bodies_id,
                                            sim_data->sa_contact_active_verts_d_hat,
                                            sim_data->sa_contact_active_verts_offset);
    }

    // stream << collision_data->toi_per_vert.view(0, 1).copy_to(host_collision_data->toi_per_vert.data())
    //        << luisa::compute::synchronize();
    // LUISA_INFO("  Min TOI after EE CCD check: {:7.6f}", host_collision_data->toi_per_vert.front());
}
void NewtonSolver::device_narrowphase_dcd(luisa::compute::Stream& stream)
{
    const float kappa = get_scene_params().stiffness_collision;

    narrow_phase_detector->vf_dcd_query_repulsion(stream,
                                                  sim_data->sa_x,
                                                  mesh_data->sa_rest_x,
                                                  mesh_data->sa_rest_vert_area,
                                                  mesh_data->sa_rest_face_area,
                                                  mesh_data->sa_faces,
                                                  sim_data->sa_vert_affine_bodies_id,
                                                  sim_data->sa_contact_active_verts_d_hat,
                                                  sim_data->sa_contact_active_verts_offset,
                                                  kappa);

    narrow_phase_detector->ee_dcd_query_repulsion(stream,
                                                  sim_data->sa_x,
                                                  mesh_data->sa_rest_x,
                                                  mesh_data->sa_rest_edge_area,
                                                  mesh_data->sa_edges,
                                                  sim_data->sa_vert_affine_bodies_id,
                                                  sim_data->sa_contact_active_verts_d_hat,
                                                  sim_data->sa_contact_active_verts_offset,
                                                  kappa);
}
void NewtonSolver::device_update_contact_list(luisa::compute::Device& device, luisa::compute::Stream& stream)
{
    narrow_phase_detector->reset_broadphase_count(stream);
    narrow_phase_detector->reset_narrowphase_count(stream);
    narrow_phase_detector->reset_pervert_collision_count(stream);

    if (get_scene_params().use_self_collision)
        device_broadphase_dcd(stream);

    bool succ_broad = narrow_phase_detector->download_broadphase_collision_count(stream);
    if (!succ_broad)
    {
        narrow_phase_detector->resize_buffers(device, stream);  // Resize broadphase buffers
        narrow_phase_detector->reset_broadphase_count(stream);
        device_broadphase_dcd(stream);
        narrow_phase_detector->download_broadphase_collision_count(stream);
    }

    // lbvh_face->check_health(stream);
    // lbvh_edge->check_health(stream);

    if (get_scene_params().use_self_collision)
        device_narrowphase_dcd(stream);

    bool succ_narrow = narrow_phase_detector->download_narrowphase_collision_count(stream);
    if (!succ_narrow)
    {
        narrow_phase_detector->resize_buffers(device, stream);  // Resize narrowphase buffers
        narrow_phase_detector->reset_narrowphase_count(stream);
        device_narrowphase_dcd(stream);
        narrow_phase_detector->download_narrowphase_collision_count(stream);
    }

    // narrow_phase_detector->download_narrowphase_list(stream);
    // auto&      host_count        = host_collision_data->narrow_phase_collision_count;
    // auto&      narrow_phase_list = host_collision_data->narrow_phase_list;
    // const uint num_pairs         = host_count.front();
    // float      min_dist          = CpuParallel::parallel_for_and_reduce(
    //     0,
    //     num_pairs,
    //     [&](const uint pair_idx)
    //     {
    //         const auto& pair    = narrow_phase_list[pair_idx];
    //         float4      weight  = pair.get_weight();
    //         uint4       indices = pair.get_indices();
    //         float3      diff =
    //             weight[0] * host_sim_data->sa_x[indices[0]] + weight[1] * host_sim_data->sa_x[indices[1]]
    //             + weight[2] * host_sim_data->sa_x[indices[2]] + weight[3] * host_sim_data->sa_x[indices[3]];
    //         return length_vec(diff);
    //     },
    //     [](float a, float b) { return min_scalar(a, b); },
    //     std::numeric_limits<float>::max());
    // LUISA_INFO("    Narrowphase collision pairs: {}, min_dist = {:7.6f}", num_pairs, min_dist);
    // float max_k1 = std::min_element(host_collision_data->narrow_phase_list.begin(),
    //                                 host_collision_data->narrow_phase_list.begin() + num_pairs,
    //                                 [](const auto& a, const auto& b) { return a.get_k1() < b.get_k1(); })
    //                    ->get_k1();
    // LUISA_INFO("    Narrowphase collision pairs: {}, max_k1 = {:7.6f}", num_pairs, max_k1);
}
void NewtonSolver::device_ccd_line_search(luisa::compute::Device& device, luisa::compute::Stream& stream)
{
    if (get_scene_params().use_self_collision)
        device_broadphase_ccd(stream);

    bool succ = narrow_phase_detector->download_broadphase_collision_count(stream);
    if (!succ)
    {
        LUISA_INFO("Broadphase collision count out of range, reallocate buffers and retry.");
        narrow_phase_detector->resize_buffers(device, stream);  // Resize broadphase buffers
        device_broadphase_ccd(stream);
        narrow_phase_detector->download_broadphase_collision_count(stream);
    }

    // lbvh_face->check_health(stream);
    // lbvh_edge->check_health(stream);

    device_narrowphase_ccd(stream);
}
void NewtonSolver::device_post_dist_check(luisa::compute::Stream& stream)
{
    narrow_phase_detector->reset_narrowphase_count(stream);
    if (get_scene_params().use_self_collision)
        device_narrowphase_dcd(stream);
    stream << collision_data->toi_per_vert.view(0, 1).copy_to(host_collision_data->toi_per_vert.data())
           << luisa::compute::synchronize();
    if (host_collision_data->toi_per_vert.front() == 0.0f)
    {
        LUISA_ERROR("Exist penetration after step!");
    }
}
void NewtonSolver::device_compute_contact_energy(luisa::compute::Stream& stream, std::map<std::string, double>& energy_list)
{
    // stream << sim_data->sa_x.copy_from(sa_x.data());
    const float kappa = get_scene_params().stiffness_collision;

    narrow_phase_detector->reset_energy(stream);
    narrow_phase_detector->compute_contact_energy_from_iter_start_list(stream,
                                                                       sim_data->sa_x,
                                                                       sim_data->sa_x_step_start,
                                                                       mesh_data->sa_rest_vert_area,
                                                                       mesh_data->sa_rest_face_area,
                                                                       mesh_data->sa_faces,
                                                                       sim_data->sa_contact_active_verts_d_hat,
                                                                       sim_data->sa_contact_active_verts_offset,
                                                                       sim_data->sa_contact_active_verts_friction_coeff,
                                                                       kappa);

    auto contact_energy = narrow_phase_detector->download_energy(stream);
    energy_list.insert(std::make_pair("Contact", contact_energy.x));
    energy_list.insert(std::make_pair("Friction", contact_energy.y));
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
    // stream << fn_pcg_spmv_offdiag_block_rbk(collision_data->triplet_data.sa_cgA_contact_offdiag_triplet, input_ptr, output_ptr)
    //               .dispatch(aligned_diaptch_count);

    const auto& host_count      = host_collision_data->narrow_phase_collision_count;
    const uint  reduced_triplet = host_count[CollisionPair::CollisionCount::total_adj_verts_offset()];
    if (reduced_triplet != 0)
    {
        const uint aligned_diaptch_count = get_dispatch_threads(reduced_triplet, 256);
        stream << fn_pcg_spmv_offdiag_block_rbk(collision_data->triplet_data.sa_cgA_contact_offdiag_triplet, input_ptr, output_ptr)
                      .dispatch(aligned_diaptch_count);
    }

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
            // auto& sa_edges             = host_sim_data->sa_stretch_springs;
            // auto& off_diag_hessian_ptr = host_sim_data->sa_stretch_springs_hessians;
            // CpuParallel::single_thread_for(0,
            //                                sa_edges.size(),
            //                                [&](const uint eid)
            //                                {
            //                                    const uint2 edge = sa_edges[eid];
            //                                    float3x3 offdiag_hessian1 = off_diag_hessian_ptr[4 * eid + 2];
            //                                    float3x3 offdiag_hessian2 = off_diag_hessian_ptr[4 * eid + 3];
            //                                    float3 output_vec0 = offdiag_hessian1 * input_ptr[edge[1]];
            //                                    float3 output_vec1 = offdiag_hessian2 * input_ptr[edge[0]];
            //                                    output_ptr[edge[0]] += output_vec0;
            //                                    output_ptr[edge[1]] += output_vec1;
            //                                });
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
        fn_SpMV_reduce_by_key(host_collision_data->triplet_data.sa_cgA_contact_offdiag_triplet,
                              get_dispatch_block(reduced_triplet, 256),
                              reduced_triplet);
    }
}
void NewtonSolver::host_solve_eigen(luisa::compute::Stream& stream)
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
            new_triplet.push_back(host_collision_data->triplet_data.sa_cgA_contact_offdiag_triplet[i]);
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
    LUISA_INFO("  In newton iter {:2}, EigenSolve error = {:7.6f}, max_element(p) = {:6.5f}",
               get_scene_params().current_nonlinear_iter,
               (cgB - cgA * cgX).norm(),
               infinity_norm);
}


void NewtonSolver::host_apply_dq_dx(const float alpha)
{
    if (alpha < 0.0f || alpha > 1.0f)
    {
        LUISA_ERROR("Alpha is not safe : {}", alpha);
    }

    const uint num_dof         = host_sim_data->num_dof;
    const uint num_verts_total = host_sim_data->num_verts_total;

    CpuParallel::parallel_for(0,
                              num_dof,
                              [&](const uint dof_idx)
                              {
                                  host_sim_data->sa_q[dof_idx] = host_sim_data->sa_q_iter_start[dof_idx]
                                                                 + alpha * host_sim_data->sa_dq[dof_idx];
                              });
    CpuParallel::parallel_for(0,
                              num_verts_total,
                              [&](const uint vid) {
                                  host_sim_data->sa_x[vid] = host_sim_data->sa_x_iter_start[vid]
                                                             + alpha * host_sim_data->sa_dx[vid];
                              });
}
void NewtonSolver::device_apply_dq_dx(luisa::compute::Stream& stream, const float alpha)
{
    if (alpha < 0.0f || alpha > 1.0f)
    {
        LUISA_ERROR("Alpha is not safe : {}", alpha);
    }

    const uint num_dof         = host_sim_data->num_dof;
    const uint num_verts_total = host_sim_data->num_verts_total;

    stream
        << fn_interpolate_template(sim_data->sa_q, sim_data->sa_q_iter_start, sim_data->sa_dq, alpha).dispatch(num_dof)
        << fn_interpolate_template(sim_data->sa_x, sim_data->sa_x_iter_start, sim_data->sa_dx, alpha).dispatch(num_verts_total);
    // stream << fn_apply_dq(alpha).dispatch(num_dof) << fn_apply_dx(alpha).dispatch(num_verts_total);
}

// void host_apply_dq_dx(const float alpha);
// void device_apply_dq_dx(luisa::compute::Stream& stream, const float alpha);

// void device_apply_x_from_q(luisa::compute::Stream& stream);
// void device_apply_x_from_q(luisa::compute::Stream&               stream,
//                             const luisa::compute::Buffer<float3>& input_q,
//                             luisa::compute::Buffer<float3>&       output_x);
// void host_apply_x_from_q(luisa::compute::Stream& stream);
// void host_apply_x_from_q(luisa::compute::Stream&    stream,
//                             const std::vector<float3>& input_q,
//                             std::vector<float3>&       output_x);

void NewtonSolver::device_apply_q_to_x(luisa::compute::Stream&               stream,
                                       const luisa::compute::Buffer<float3>& input_q,
                                       luisa::compute::Buffer<float3>&       output_x)
{
    stream << fn_apply_q_to_x_template(input_q, output_x).dispatch(output_x.size());
}
float3 fn_apply_template(const std::vector<uint>&   sa_x_to_dof_map,
                         const std::vector<float3>& sa_scaled_model_x,
                         const std::vector<float3>& input_q,
                         const uint                 vid)
{

    float3     new_dx;
    const uint map_info = sa_x_to_dof_map[vid];
    const uint dof_idx  = map_info & (~Attributions::RIGID_BODY_FLAG);
    if ((map_info & Attributions::RIGID_BODY_FLAG) == 0)  // Soft body
    {
        new_dx = input_q[dof_idx];
    }
    else  // Rigid body
    {
        const float3 rest_x = sa_scaled_model_x[vid];
        float3       p;
        float3x3     A;
        p      = input_q[dof_idx + 0];
        A[0]   = input_q[dof_idx + 1];
        A[1]   = input_q[dof_idx + 2];
        A[2]   = input_q[dof_idx + 3];
        new_dx = A * rest_x + p;  // Affine position
    };
    return new_dx;
};
void NewtonSolver::host_apply_q_to_x(const std::vector<float3>& input_q, std::vector<float3>& output_x)
{
    const uint num_verts_total = host_sim_data->num_verts_total;
    if (output_x.size() != num_verts_total)
    {
        LUISA_ERROR("Output x size mismatch {} != {}", output_x.size(), num_verts_total);
    }
    CpuParallel::parallel_for(0,
                              num_verts_total,
                              [&](const uint vid)
                              {
                                  output_x[vid] = fn_apply_template(host_sim_data->sa_x_to_dof_map,
                                                                    host_sim_data->sa_scaled_model_x,
                                                                    input_q,
                                                                    vid);
                              });
}

// Required data on BOTH devices:
//          dx, x_iter_start,
//          dq, q_iter_start,
//          device_q_tilde
void NewtonSolver::line_search(luisa::compute::Device& device,
                               luisa::compute::Stream& stream,
                               bool&                   dirichlet_converged,
                               bool&                   global_converged)
{
    const uint iter = get_scene_params().current_nonlinear_iter;

    auto ccd_get_toi = [&]() -> float
    {
        device_ccd_line_search(device, stream);

        stream << collision_data->toi_per_vert.view().copy_to(host_collision_data->toi_per_vert.data())
               << luisa::compute::synchronize();

        float toi = host_collision_data->toi_per_vert.front();
        return toi == 1.0f ? 1.0f : 0.9f * toi;  // 0.9f * toi
    };

    auto compute_energy_interface = [&](const float alpha)
    {
        // stream << sim_data->sa_x.copy_from(host_sim_data->sa_x.data());
        // if (host_sim_data->num_verts_soft != 0)
        // {
        //     stream << sim_data->sa_x_tilde.copy_from(host_sim_data->sa_x_tilde.data());
        // }
        // if (host_sim_data->num_affine_bodies != 0)
        // {
        //     stream
        //         << sim_data->sa_affine_bodies_q.copy_from(host_sim_data->sa_affine_bodies_q.data())
        //         << sim_data->sa_affine_bodies_q_tilde.copy_from(host_sim_data->sa_affine_bodies_q_tilde.data());
        // }
        std::map<std::string, double> energy_list;
        device_compute_elastic_energy(stream, energy_list);
        device_compute_contact_energy(stream, energy_list);

        auto total_energy = std::accumulate(energy_list.begin(),
                                            energy_list.end(),
                                            0.0,
                                            [](double sum, const auto& pair) { return sum + pair.second; });
        if (get_scene_params().print_system_energy)
        {
            LUISA_INFO("  +----------------------+----------------------+");
            LUISA_INFO("  | {:<20} | {:>20} |", luisa::format("Energy in step {:4.3f}", alpha), "Value");
            LUISA_INFO("  +----------------------+----------------------+");
            for (const auto& pair : energy_list)
            {
                LUISA_INFO("  | {:<20} | {:>20.10f} |", pair.first, pair.second);
            }
            LUISA_INFO("  +----------------------+----------------------+");
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

    auto apply_final_dx = [&](const float alpha)
    {
        // stream << sim_data->sa_x.copy_to(sim_data->sa_x_iter_start);
        // if (sim_data->num_affine_bodies != 0)
        // {
        //     stream << sim_data->sa_affine_bodies_q.copy_to(sim_data->sa_affine_bodies_q_iter_start);
        // }
        // stream << luisa::compute::synchronize();
        device_apply_dq_dx(stream, alpha);
        host_apply_dq_dx(alpha);
        stream << luisa::compute::synchronize();
    };

    float alpha   = 1.0f;
    float ccd_toi = 1.0f;

    if (get_scene_params().use_ccd_linesearch)
    {
        device_apply_dq_dx(stream, alpha);

        ccd_toi = ccd_get_toi();
        alpha   = ccd_toi;

        // if (ccd_toi < 1.0f)
        // {
        //     LUISA_INFO("  In newton iter {:2}: CCD line search applied, toi = {:6.5f}",
        //                get_scene_params().current_nonlinear_iter,
        //                ccd_toi);
        // }
        if (ccd_toi < 0.0f || ccd_toi > 1.0f)
        {
            LUISA_ERROR("Invalid Toi {}", ccd_toi);
        }
        LUISA_INFO(
            "  In newton iter {:2}: CCD toi = {:6.5f}, BroadPhase VF/EE = {} / {}, NarrowPhase = {} (Friction = {}), HessTriplet = {}",
            iter,
            ccd_toi,
            host_collision_data->broad_phase_collision_count[CollisionPair::CollisionCount::vf_offset()],
            host_collision_data->broad_phase_collision_count[CollisionPair::CollisionCount::ee_offset()],
            host_collision_data->narrow_phase_collision_count.front(),
            host_collision_data->num_pairs_in_first_iter.front(),
            host_collision_data->narrow_phase_collision_count[CollisionPair::CollisionCount::total_adj_verts_offset()]);
    }

    // Non-linear iteration break condition
    {
        float max_move      = 1e-2;
        float curr_max_step = fast_infinity_norm(host_sim_data->sa_dq);
        if (curr_max_step < max_move * get_scene_params().implicit_dt)
        {
            LUISA_INFO("  In newton iter {:2}: Iteration break for small searching direction {} < {}",
                       iter,
                       curr_max_step,
                       max_move * get_scene_params().implicit_dt);

            apply_final_dx(alpha);
            dirichlet_converged = true;
            global_converged    = true;
            return;
        }
        global_converged = false;
    }  // That means: If the step is too small, then we dont need energy line-search (energy may not be descent in small step)

    if (get_scene_params().use_energy_linesearch)
    {
        device_apply_dq_dx(stream, 0.0f);
        float prev_state_energy = compute_energy_interface(0.0f);

        device_apply_dq_dx(stream, alpha);

        // Energy after CCD or just solving Axb
        auto curr_energy = compute_energy_interface(alpha);
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

            device_apply_dq_dx(stream, alpha);

            curr_energy = compute_energy_interface(alpha);
            LUISA_INFO("     Line search {} : alpha = {:6.5f}, energy = {:12.10f}", line_search_count, alpha, curr_energy);

            if (alpha < 1e-4)
            {
                LUISA_ERROR("  Line search failed, energy = {}, prev state energy = {}", curr_energy, prev_state_energy);
            }
            line_search_count++;
        }
        prev_state_energy = curr_energy;  // E_prev = E
    }

    apply_final_dx(alpha);

    // Check dirichlet point target
    {
        const float direchlet_max_delta = CpuParallel::parallel_for_and_reduce(
            0,
            host_sim_data->sa_q.size(),
            [&](const uint vid)
            {
                if (host_sim_data->sa_q_is_fixed[vid])
                {
                    // Converged condition: ||x - x_tilde|| < tol, or ||x - x_iter_start|| < tol
                    float3 delta1 = host_sim_data->sa_q[vid] - host_sim_data->sa_q_tilde[vid];
                    float3 delta2 = host_sim_data->sa_q[vid] - host_sim_data->sa_q_iter_start[vid];
                    // LUISA_INFO("     Dirichlet vertex {} : current pos {}, target pos {}, start pos {}, delta1 {}, delta2 {}",
                    //            vid,
                    //            host_sim_data->sa_q[vid],
                    //            host_sim_data->sa_q_tilde[vid],
                    //            host_sim_data->sa_q_iter_start[vid],
                    //            delta1,
                    //            delta2);
                    float delta = luisa::min(luisa::length(delta1), luisa::length(delta2));
                    return delta;
                }
                return 0.0f;  // Non-fixed point
            },
            [](const float left, const float right) { return max_scalar(left, right); },
            -1e9f);

        dirichlet_converged = direchlet_max_delta < 1e-3;
        if (!dirichlet_converged)
        {
            LUISA_INFO("  In newton iter {:2}: Dirichlet point not converged, max delta = {}", iter, direchlet_max_delta);
        }
    }
    return;  // Since max_p is larger than epsilon
}
void NewtonSolver::physics_step_CPU(luisa::compute::Device& device, luisa::compute::Stream& stream)
{
    // Input
    lcs::SolverInterface::physics_step_prev_operation();

    constexpr bool use_eigen          = ConjugateGradientSolver::use_eigen;
    constexpr bool use_upper_triangle = ConjugateGradientSolver::use_upper_triangle;

    auto update_contact_set = [&]()
    {
        stream << sim_data->sa_x.copy_from(host_sim_data->sa_x.data());

        device_update_contact_list(device, stream);
        narrow_phase_detector->prescan_pervert_adj_list(
            stream, sim_data->sa_vert_affine_bodies_id, host_sim_data->num_verts_soft);
        narrow_phase_detector->download_narrowphase_collision_count(stream);
        narrow_phase_detector->resize_buffers(device, stream);  // Resize adj pairs buffers
        narrow_phase_detector->construct_pervert_adj_list(
            stream, sim_data->sa_vert_affine_bodies_id, host_sim_data->num_verts_soft);
        narrow_phase_detector->device_sort_contact_triplet(stream);
        narrow_phase_detector->resize_buffers(device, stream);  // Resize adj verts buffers (Assembled triplet buffers)
    };
    auto evaluate_contact = [&]()
    {
        stream << sim_data->sa_cgB.copy_from(host_sim_data->sa_cgB.data())
               << sim_data->sa_cgA_diag.copy_from(host_sim_data->sa_cgA_diag.data());

        narrow_phase_detector->device_perPair_evaluate_gradient_hessian(stream,
                                                                        sim_data->sa_x,
                                                                        sim_data->sa_x_step_start,
                                                                        sim_data->sa_contact_active_verts_friction_coeff,
                                                                        sim_data->sa_contact_active_verts_d_hat,
                                                                        sim_data->sa_contact_active_verts_offset,
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
    auto pcg_spmv_interface = [&](const std::vector<float3>& input_ptr, std::vector<float3>& output_ptr) -> void
    {
        //
        host_SpMV(stream, input_ptr, output_ptr);
    };
    auto linear_solver_interface = [&]()
    {
        if constexpr (false)
        {
            host_solve_eigen(stream);
        }
        else
        {
            pcg_solver->host_solve(stream, pcg_spmv_interface, []() { return 0.0; });
        }
    };

    const float substep_dt            = lcs::get_scene_params().get_substep_dt();
    const bool  use_energy_linesearch = get_scene_params().use_energy_linesearch;
    const bool  use_ccd_linesearch    = get_scene_params().use_ccd_linesearch;

    host_apply_q_to_x(host_sim_data->sa_q_step_start, host_sim_data->sa_x_step_start);
    buffer_copy(host_sim_data->sa_q_step_start, host_sim_data->sa_q);
    buffer_copy(host_sim_data->sa_x_step_start, host_sim_data->sa_x);
    // buffer_copy(host_sim_data->sa_q_step_start, host_sim_data->sa_q_iter_start);
    // buffer_copy(host_sim_data->sa_x_step_start, host_sim_data->sa_x_iter_start);

    // Init LBVH
    {
        buffer_upload(stream, host_sim_data->sa_x_step_start, sim_data->sa_x_step_start);
        lbvh_face->reduce_face_tree_aabb(stream, sim_data->sa_x_step_start, mesh_data->sa_faces);
        lbvh_edge->reduce_edge_tree_aabb(stream, sim_data->sa_x_step_start, mesh_data->sa_edges);
        lbvh_face->construct_tree(stream);
        lbvh_edge->construct_tree(stream);
        stream << luisa::compute::synchronize();
    }
    // for (uint substep = 0; substep < get_scene_params().num_substep; substep++)
    {
        LUISA_INFO("=== In frame {} ===", get_scene_params().current_frame);

        host_predict_position();  // => q_tilde

        if (get_scene_params().use_energy_linesearch)
        {
            stream << sim_data->sa_q_tilde.copy_from(host_sim_data->sa_q_tilde.data());
        }

        // for (uint iter = 0; iter < get_scene_params().nonlinear_iter_count; iter++)
        uint iter                = 0;
        bool global_converged    = false;
        bool dirichlet_converged = false;
        for (iter = 0; iter < 100; iter++)
        {
            if (global_converged || (iter >= get_scene_params().nonlinear_iter_count && dirichlet_converged))
            {
                break;
            }
            get_scene_params().current_nonlinear_iter = iter;

            // Record position at iteration start, for linear interpolation: x = x_iter_start + alpha * dx
            {
                buffer_copy(host_sim_data->sa_x, host_sim_data->sa_x_iter_start);
                buffer_copy(host_sim_data->sa_q, host_sim_data->sa_q_iter_start);
                buffer_upload(stream, host_sim_data->sa_x_iter_start, sim_data->sa_x_iter_start);
                buffer_upload(stream, host_sim_data->sa_q_iter_start, sim_data->sa_q_iter_start);
                buffer_upload(stream, host_sim_data->sa_x, sim_data->sa_x);
            }

            host_reset_cgB_cgX_diagA();

            host_reset_off_diag();

            update_contact_set();

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

                linear_solver_interface();  // Solve Ax=b => cgX
            }
            else
            {
                host_test_dynamics(stream);
            }

            {
                buffer_copy(host_sim_data->sa_cgX, host_sim_data->sa_dq);  // dq = cgX
                host_apply_q_to_x(host_sim_data->sa_dq, host_sim_data->sa_dx);
                buffer_upload(stream, host_sim_data->sa_dq, sim_data->sa_dq);
                buffer_upload(stream, host_sim_data->sa_dx, sim_data->sa_dx);
            }

            line_search(device, stream, dirichlet_converged, global_converged);

            narrow_phase_detector->resize_buffers(device, stream);  // Pre-allocatation

            if (iter == 99)
                LUISA_ERROR("Solver is not converged in 100 iters");
        }
        host_update_velocity();
        host_apply_q_to_x(host_sim_data->sa_q_v, host_sim_data->sa_v);
    }

    // Output
    lcs::SolverInterface::physics_step_post_operation();
}
void NewtonSolver::physics_step_GPU(luisa::compute::Device& device, luisa::compute::Stream& stream)
{
    // Read frame start position and velocity
    lcs::SolverInterface::physics_step_prev_operation();  // => sa_q_step_start, sa_q_v

    auto update_contact_set = [&]()
    {
        device_update_contact_list(device, stream);
        narrow_phase_detector->prescan_pervert_adj_list(
            stream, sim_data->sa_vert_affine_bodies_id, host_sim_data->num_verts_soft);
        narrow_phase_detector->download_narrowphase_collision_count(stream);
        narrow_phase_detector->resize_buffers(device, stream);  // Resize adj pairs buffers
        narrow_phase_detector->construct_pervert_adj_list(
            stream, sim_data->sa_vert_affine_bodies_id, host_sim_data->num_verts_soft);
        narrow_phase_detector->device_sort_contact_triplet(stream);
        narrow_phase_detector->resize_buffers(device, stream);  // Resize adj verts buffers (Assembled triplet buffers)
    };
    auto evaluate_contact = [&]()
    {
        narrow_phase_detector->device_perPair_evaluate_gradient_hessian(stream,
                                                                        sim_data->sa_x,
                                                                        sim_data->sa_x_step_start,
                                                                        sim_data->sa_contact_active_verts_friction_coeff,
                                                                        sim_data->sa_contact_active_verts_d_hat,
                                                                        sim_data->sa_contact_active_verts_offset,
                                                                        sim_data->sa_vert_affine_bodies_id,
                                                                        mesh_data->sa_scaled_model_x,
                                                                        host_sim_data->num_verts_soft,
                                                                        sim_data->sa_cgB,
                                                                        sim_data->sa_cgA_diag);

        narrow_phase_detector->device_assemble_contact_triplet(
            stream, mesh_data->sa_scaled_model_x, host_sim_data->num_verts_soft);
    };

    auto pcg_spmv_interface = [&](const luisa::compute::Buffer<float3>& input_ptr,
                                  luisa::compute::Buffer<float3>&       output_ptr) -> void
    {
        //
        device_SpMV(stream, input_ptr, output_ptr);
    };

    // return dq, dx
    auto linear_solver_interface = [&]()
    {
        if constexpr (false)
        {
            stream << sim_data->sa_cgB.copy_to(host_sim_data->sa_cgB.data())
                   << sim_data->sa_cgA_diag.copy_to(host_sim_data->sa_cgA_diag.data())
                   << sim_data->sa_cgA_fixtopo_offdiag_triplet.copy_to(
                          host_sim_data->sa_cgA_fixtopo_offdiag_triplet.data());
            narrow_phase_detector->download_contact_triplet(stream);
            host_solve_eigen(stream);
            CpuParallel::parallel_copy(host_sim_data->sa_cgX, host_sim_data->sa_dq);
            stream << sim_data->sa_dq.copy_from(host_sim_data->sa_dq.data());
        }
        else
        {
            pcg_solver->device_solve(stream, pcg_spmv_interface, []() { return 0.0; });
            buffer_copy(stream, sim_data->sa_cgX, sim_data->sa_dq);  // dq = cgX
            stream << fn_apply_q_to_x_template(sim_data->sa_dq, sim_data->sa_dx).dispatch(sim_data->sa_dx.size());
            buffer_download(stream, sim_data->sa_dq, host_sim_data->sa_dq);
            buffer_download(stream, sim_data->sa_dx, host_sim_data->sa_dx);
            stream << luisa::compute::synchronize();
        }
    };

    // Upload frame start information
    buffer_upload(stream, host_sim_data->sa_q_step_start, sim_data->sa_q_step_start);
    buffer_upload(stream, host_sim_data->sa_q_v, sim_data->sa_q_v);

    device_apply_q_to_x(stream, sim_data->sa_q_step_start, sim_data->sa_x_step_start);
    buffer_copy(stream, sim_data->sa_x_step_start, sim_data->sa_x);
    buffer_copy(stream, sim_data->sa_q_step_start, sim_data->sa_q);
    buffer_download(stream, sim_data->sa_x, host_sim_data->sa_x);
    buffer_download(stream, sim_data->sa_q, host_sim_data->sa_q);
    stream << luisa::compute::synchronize();

    const uint  num_substep          = lcs::get_scene_params().num_substep;
    const uint  nonlinear_iter_count = lcs::get_scene_params().nonlinear_iter_count;
    const float substep_dt           = lcs::get_scene_params().get_substep_dt();

    // Init LBVH
    {
        lbvh_face->reduce_face_tree_aabb(stream, sim_data->sa_x_step_start, mesh_data->sa_faces);
        lbvh_edge->reduce_edge_tree_aabb(stream, sim_data->sa_x_step_start, mesh_data->sa_edges);
        lbvh_face->construct_tree(stream);
        lbvh_edge->construct_tree(stream);
        stream << luisa::compute::synchronize();
    }

    // for (uint substep = 0; substep < get_scene_params().num_substep; substep++)
    {
        {
            stream << fn_predict_position(substep_dt, get_scene_params().gravity).dispatch(host_sim_data->num_dof);
            buffer_download(stream, sim_data->sa_q_tilde, host_sim_data->sa_q_tilde, /*wait=*/false);
        }

        double prev_state_energy = Float_max;

        LUISA_INFO("=== In frame {} ===", get_scene_params().current_frame);

        uint iter                = 0;
        bool global_converged    = false;
        bool dirichlet_converged = false;
        for (iter = 0; iter < 100; iter++)
        {
            if (global_converged || (iter >= get_scene_params().nonlinear_iter_count && dirichlet_converged))
            {
                break;
            }
            get_scene_params().current_nonlinear_iter = iter;

            // TODO: If we use predict position, the start position may not in safe region
            buffer_copy(stream, sim_data->sa_x, sim_data->sa_x_iter_start);
            buffer_copy(stream, sim_data->sa_q, sim_data->sa_q_iter_start);
            buffer_download(stream, sim_data->sa_x_iter_start, host_sim_data->sa_x_iter_start);  // For host apply dx
            buffer_download(stream, sim_data->sa_q_iter_start, host_sim_data->sa_q_iter_start);  // For host apply dq
            stream << luisa::compute::synchronize();

            stream << fn_reset_vector(sim_data->sa_cgX).dispatch(sim_data->sa_cgX.size())
                   << fn_reset_vector(sim_data->sa_cgB).dispatch(sim_data->sa_cgB.size())
                   << fn_reset_float3x3(sim_data->sa_cgA_diag).dispatch(sim_data->sa_cgA_diag.size())
                   << fn_reset_cgA_offdiag_triplet().dispatch(sim_data->sa_cgA_fixtopo_offdiag_triplet.size());

            {
                const uint num_dof_soft  = host_sim_data->num_verts_soft;
                const uint num_dof_rigid = host_sim_data->num_affine_bodies * 4;

                auto& soft_inertia = sim_data->get_soft_inertia_data();
                if (soft_inertia.is_valid())
                {
                    stream << fn_evaluate_soft_inertia(soft_inertia, substep_dt).dispatch(soft_inertia.get_num_indices());

                    stream << fn_evaluate_soft_ground_collision(soft_inertia,
                                                                get_scene_params().floor.y,
                                                                get_scene_params().use_floor,
                                                                get_scene_params().stiffness_collision,
                                                                get_scene_params().contact_energy_type)
                                  .dispatch(host_sim_data->num_verts_soft);
                    stream << fn_material_energy_assembly_soft_inertia(soft_inertia).dispatch(num_dof_soft);
                }

                const auto& stretch_springs = sim_data->get_stretch_spring_data();
                if (stretch_springs.is_valid())
                {
                    stream << fn_evaluate_spring(stretch_springs).dispatch(stretch_springs.get_num_indices());
                    stream << fn_material_energy_assembly_stretch_spring(stretch_springs).dispatch(num_dof_soft);
                }

                const auto& stretch_faces = sim_data->get_stretch_face_data();
                if (stretch_faces.is_valid())
                {
                    stream << fn_evaluate_stretch_face(stretch_faces).dispatch(stretch_faces.get_num_indices());
                    stream << fn_material_energy_assembly_stretch_face(stretch_faces).dispatch(num_dof_soft);
                }

                const auto& bending_data = sim_data->get_bending_edge_data();
                if (bending_data.is_valid())
                {
                    stream << fn_evaluate_bending(bending_data, get_scene_params().get_bending_stiffness_scaling())
                                  .dispatch(bending_data.get_num_indices());
                    stream << fn_material_energy_assembly_bending(bending_data).dispatch(num_dof_soft);
                }

                const auto& abd_inertia_data = sim_data->get_abd_inertia_data();
                if (abd_inertia_data.is_valid())
                {
                    stream << fn_evaluate_abd_inertia(abd_inertia_data, substep_dt)
                                  .dispatch(abd_inertia_data.get_num_indices());

                    stream << fn_evaluate_abd_ground_collision(abd_inertia_data,
                                                               get_scene_params().floor.y,
                                                               get_scene_params().use_floor,
                                                               get_scene_params().stiffness_collision,
                                                               host_sim_data->num_verts_soft,
                                                               get_scene_params().contact_energy_type)
                                  .dispatch(host_sim_data->num_verts_rigid);

                    stream << fn_material_energy_assembly_abd_inertia(abd_inertia_data, num_dof_soft).dispatch(num_dof_rigid);
                }

                const auto& abd_orthogonality_data = sim_data->get_abd_orthogonality_data();
                if (abd_orthogonality_data.is_valid())
                {
                    stream << fn_evaluate_abd_orthogonality(abd_orthogonality_data)
                                  .dispatch(abd_orthogonality_data.get_num_indices());
                    stream << fn_material_energy_assembly_abd_ortho(abd_orthogonality_data, num_dof_soft).dispatch(num_dof_rigid);
                }

                update_contact_set();

                evaluate_contact();

                // stream << fn_evaluate_dirichlet(substep_dt, get_scene_params().stiffness_dirichlet).dispatch(num_verts);
            }

            stream << luisa::compute::synchronize();

            linear_solver_interface();  // => dq, dx

            line_search(device, stream, dirichlet_converged, global_converged);

            narrow_phase_detector->resize_buffers(device, stream);  // Pre-allocatation

            if (iter == 99)
                LUISA_ERROR("Solver is not converged in 100 iters");
        }

        stream << fn_update_velocity(substep_dt, get_scene_params().fix_scene, get_scene_params().damping_cloth)
                      .dispatch(host_sim_data->num_dof);
        device_apply_q_to_x(stream, sim_data->sa_q_v, sim_data->sa_v);
    }

    stream << luisa::compute::synchronize();

    // Copy to host
    {
        buffer_download(stream, sim_data->sa_q, host_sim_data->sa_q);
        buffer_download(stream, sim_data->sa_q_v, host_sim_data->sa_q_v);
        buffer_download(stream, sim_data->sa_x, host_sim_data->sa_x);
        buffer_download(stream, sim_data->sa_v, host_sim_data->sa_v);
        stream << luisa::compute::synchronize();
    }

    // Return frame end position and velocity
    lcs::SolverInterface::physics_step_post_operation();
}

}  // namespace lcs