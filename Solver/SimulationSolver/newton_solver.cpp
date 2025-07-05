#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include "CollisionDetector/libuipc/codim_ipc_simplex_normal_contact_function.h"
#include "SimulationSolver/descent_solver.h"
#include "SimulationSolver/newton_solver.h"
#include "Core/float_nxn.h"
#include "Core/scalar.h"
#include "Core/xbasic_types.h"
#include "SimulationCore/scene_params.h"
#include "Utils/cpu_parallel.h"
#include "Utils/reduce_helper.h"
#include "luisa/backends/ext/pinned_memory_ext.hpp"
#include "luisa/runtime/stream.h"
#include <luisa/dsl/sugar.h>
#include "CollisionDetector/accd.hpp"

namespace lcsv 
{

template<typename T>
void buffer_add(luisa::compute::BufferView<T> buffer, const Var<uint> dest, const Var<T>& value)
{
    buffer->write(dest, buffer->read(dest) + value);
}
template<typename T>
void buffer_add(Var<luisa::compute::BufferView<T>>& buffer, const Var<uint> dest, const Var<T>& value)
{
    buffer->write(dest, buffer->read(dest) + value);
}

void NewtonSolver::compile(luisa::compute::Device& device)
{
    const bool use_debug_info = false;
    using namespace luisa::compute;

    luisa::compute::ShaderOption default_option = {.enable_debug_info = false};

    auto& sa_x_tilde = sim_data->sa_x_tilde;
    auto& sa_x = sim_data->sa_x;
    auto& sa_v = sim_data->sa_v;
    auto& sa_x_step_start = sim_data->sa_x_step_start;
    auto& sa_x_iter_start = sim_data->sa_x_iter_start;
    auto& sa_v_step_start = sim_data->sa_v_step_start;
    
    auto& sa_cgX = sim_data->sa_cgX;
    auto& sa_cgB = sim_data->sa_cgB;
    auto& sa_cgA_diag = sim_data->sa_cgA_diag;
    auto& sa_cgA_offdiag = sim_data->sa_cgA_offdiag;
    
    auto& sa_cgMinv = sim_data->sa_cgMinv;
    auto& sa_cgP = sim_data->sa_cgP;
    auto& sa_cgQ = sim_data->sa_cgQ;
    auto& sa_cgR = sim_data->sa_cgR;
    auto& sa_cgZ = sim_data->sa_cgZ;

    fn_reset_vector = device.compile<1>([](Var<BufferView<float3>> buffer, Float3 target)
    {
        const UInt vid = dispatch_id().x;
        // buffer->write(vid, target);
        buffer->write(vid, make_float3(0.0f));
    });

    fn_reset_offdiag = device.compile<1>(
        [
            sa_cgA_offdiag = sa_cgA_offdiag.view()
        ](){
            sa_cgA_offdiag->write(dispatch_id().x, makeFloat3x3(Float(0.0f)));
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
        $if (sa_is_fixed->read(vid) != 0) { v_pred = make_float3(0.0f); };

        sa_x_iter_start->write(vid, x_prev);
        Float3 x_pred = x_prev + substep_dt * v_pred;
        sa_x_tilde->write(vid, x_pred);
        sa_x->write(vid, x_prev);
        sa_cgX->write(vid, make_float3(0.0f));
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

        $if (fix_scene) {
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
        ](const Float substep_dt)
    {
        const UInt vid = dispatch_id().x;
        const Float h = substep_dt;
        const Float h_2_inv = 1.0f / (h * h);

        Float3 x_k = sa_x->read(vid);
        Float3 x_tilde = sa_x_tilde->read(vid);
        Int is_fixed = sa_is_fixed->read(vid);
        Float mass = sa_vert_mass->read(vid);

        Float3 gradient = -mass * h_2_inv * (x_k - x_tilde);
        Float3x3 hessian = make_float3x3(1.0f) * mass * h_2_inv;

        $if (is_fixed != 0) {
            hessian = make_float3x3(1.0f) * 1e9f;
            gradient = make_float3(0.0f);
        };

        sa_cgB->write(vid, gradient); 
        sa_cgA_diag->write(vid, hessian);
    }, default_option);

    fn_evaluate_spring = device.compile<1>(
        [
            sa_x = sim_data->sa_x.view(),
            sa_cgB = sim_data->sa_cgB.view(),
            sa_cgA_diag = sim_data->sa_cgA_diag.view(),
            sa_cgA_offdiag = sim_data->sa_cgA_offdiag.view(),
            culster = sim_data->sa_prefix_merged_springs.view(),
            sa_edges = sim_data->sa_merged_edges.view(),
            sa_rest_length = sim_data->sa_merged_edges_rest_length.view()
        ](const Float stiffness_stretch, const Uint curr_prefix)
    {
        // const Uint curr_prefix = culster->read(cluster_idx);
        const UInt eid = curr_prefix + dispatch_id().x;

        // const UInt eid = dispatch_id().x;
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

        buffer_add(sa_cgB, edge[0], force[0]);
        buffer_add(sa_cgB, edge[1], force[1]);
        buffer_add(sa_cgA_diag, edge[0], He);
        buffer_add(sa_cgA_diag, edge[1], He);
        buffer_add(sa_cgA_offdiag, eid * 2 + 0, -1.0f * He);
        buffer_add(sa_cgA_offdiag, eid * 2 + 1, -1.0f * He);
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

    fn_pcg_spmv_offdiag = device.compile<1>(
        [
            sa_edges = sim_data->sa_merged_edges.view(),
            sa_cgA_offdiag = sim_data->sa_cgA_offdiag.view(),
            culster = sim_data->sa_prefix_merged_springs.view()
        ](
            Var<luisa::compute::BufferView<float3>> sa_input_vec, 
            Var<luisa::compute::BufferView<float3>> sa_output_vec,
            const Uint curr_prefix
        )
    {
        // const Uint curr_prefix = culster->read(cluster_idx);
        const UInt eid = curr_prefix + dispatch_id().x;

        // const UInt eid = dispatch_id().x;
        UInt2 edge = sa_edges->read(eid);
        Float3x3 offdiag_hessian1 = sa_cgA_offdiag->read(2 * eid + 0);
        Float3x3 offdiag_hessian2 = sa_cgA_offdiag->read(2 * eid + 1);

        Float3 input1 = sa_input_vec->read(edge[1]);
        Float3 input2 = sa_input_vec->read(edge[0]);
        
        Float3 output1 = offdiag_hessian1 * input1;
        Float3 output2 = offdiag_hessian2 * input2;

        buffer_add(sa_output_vec, edge[0], output1);
        buffer_add(sa_output_vec, edge[1], output2);
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
                buffer_add(sa_convergence, 7, energy);
            };
        }
    );

    fn_calc_energy_inertia = device.compile<1>(
        [
            sa_x_tilde = sim_data->sa_x_tilde.view(),
            sa_vert_mass = mesh_data->sa_vert_mass.view(),
            sa_block_result = sim_data->sa_block_result.view()
        ](
            Var<BufferView<float3>> sa_x, 
            Float substep_dt
        )
    {
        const Uint vid = dispatch_id().x;

        Float energy = 0.0f;

        {
            Float3 x_new = sa_x->read(vid);
            Float3 x_tilde = sa_x_tilde->read(vid);
            Float mass = sa_vert_mass->read(vid);
            energy = length_squared_vec(x_new - x_tilde) * mass / (2.0f * substep_dt * substep_dt);
        };

        energy = ParallelIntrinsic::block_intrinsic_reduce(vid, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
        $if(vid % 256 == 0)
        {
            sa_block_result->write(vid / 256, energy);
        };
    }, default_option);

    fn_calc_energy_spring = device.compile<1>(
        [
            sa_edges = mesh_data->sa_edges.view(),
            sa_edge_rest_state_length = mesh_data->sa_edges_rest_state_length.view(),
            sa_block_result = sim_data->sa_block_result.view()
        ](
            Var<BufferView<float3>> sa_x,
            Float stiffness_spring
        )
    {
        const Uint eid = dispatch_id().x;

        Float energy = 0.0f;
        {
            const Uint2 edge = sa_edges->read(eid);
            const Float rest_edge_length = sa_edge_rest_state_length->read(eid);
            Float3 diff = sa_x->read(edge[1]) - sa_x->read(edge[0]);
            Float orig_lengthsqr = length_squared_vec(diff);
            Float l = sqrt_scalar(orig_lengthsqr);
            Float l0 = rest_edge_length;
            Float C = l - l0;
            // if (C > 0.0f)
                energy = 0.5f * stiffness_spring * C * C;
        };

        energy = ParallelIntrinsic::block_intrinsic_reduce(eid, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
        $if (eid % 256 == 0)
        {
            sa_block_result->write(eid / 256, energy);
        };

    }, default_option);

    

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

    
}

using EigenFloat3x3 = Eigen::Matrix<float, 3, 3>;
using EigenFloat6x6 = Eigen::Matrix<float, 6, 6>;
using EigenFloat9x9 = Eigen::Matrix<float, 9, 9>;
using EigenFloat12x12 = Eigen::Matrix<float, 12, 12>;
using EigenFloat3   = Eigen::Matrix<float, 3, 1>;
using EigenFloat4   = Eigen::Matrix<float, 4, 1>;

static inline EigenFloat3x3 float3x3_to_eigen3x3(const float3x3& input)
{
    EigenFloat3x3 mat; mat << 
        input[0][0], input[1][0], input[2][0], 
        input[0][1], input[1][1], input[2][1], 
        input[0][2], input[1][2], input[2][2]; 
    return mat;
};
static inline float3x3 eigen3x3_to_float3x3(const EigenFloat3x3& input)
{
    return luisa::make_float3x3(
        input(0, 0), input(1, 0), input(2, 0), 
        input(0, 1), input(1, 1), input(2, 1), 
        input(0, 2), input(1, 2), input(2, 2));
};
static inline EigenFloat6x6 float6x6_to_eigen6x6(const float6x6& input)
{
    EigenFloat6x6 output;
    for (uint i = 0; i < 2; ++i) 
    {
        for (uint j = 0; j < 2; ++j) 
        {
            output.block<3, 3>(i * 3, j * 3) = float3x3_to_eigen3x3(input.mat[i][j]);
        }
    }
    return output;
};
static inline float6x6 eigen6x6_to_float6x6(const EigenFloat6x6& input)
{
    float6x6 output;
    for (uint i = 0; i < 2; ++i) 
    {
        for (uint j = 0; j < 2; ++j) 
        {
            output.mat[i][j] = eigen3x3_to_float3x3(input.block<3, 3>(i * 3, j * 3));
        }
    }
    return output;
};
static inline EigenFloat9x9 float9x9_to_eigen9x9(const float9x9& input)
{
    EigenFloat9x9 output;
    for (uint i = 0; i < 3; ++i) 
    {
        for (uint j = 0; j < 3; ++j) 
        {
            output.block<3, 3>(i * 3, j * 3) = float3x3_to_eigen3x3(input.mat[i][j]);
        }
    }
    return output;
};
static inline float9x9 eigen9x9_to_float9x9(const EigenFloat9x9& input)
{
    float9x9 output;
    for (uint i = 0; i < 3; ++i) 
    {
        for (uint j = 0; j < 3; ++j) 
        {
            output.mat[i][j] = eigen3x3_to_float3x3(input.block<3, 3>(i * 3, j * 3));
        }
    }
    return output;
};
static inline EigenFloat12x12 float12x12_to_eigen12x12(const float12x12 input)
{
    EigenFloat12x12 output;
    for (uint i = 0; i < 4; ++i) 
    {
        for (uint j = 0; j < 4; ++j) 
        {
            output.block<3, 3>(i * 3, j * 3) = float3x3_to_eigen3x3(input.mat[i][j]);
        }
    }
    return output;
};
static inline float12x12 eigen12x12_to_float12x12(const EigenFloat9x9& input)
{
    float12x12 output;
    for (uint i = 0; i < 4; ++i) 
    {
        for (uint j = 0; j < 4; ++j) 
        {
            output.mat[i][j] = eigen3x3_to_float3x3(input.block<3, 3>(i * 3, j * 3));
        }
    }
    return output;
};
static inline EigenFloat3 float3_to_eigen3(const float3& input) { EigenFloat3 vec; vec << input[0], input[1], input[2]; return vec; };
static inline EigenFloat4 float4_to_eigen4(const float4& input) { EigenFloat4 vec; vec << input[0], input[1], input[2], input[3]; return vec; };
static inline float3 eigen3_to_float3(const EigenFloat3& input) { return luisa::make_float3(input(0, 0), input(1, 0), input(2, 0)); };
static inline float4 eigen4_to_float4(const EigenFloat4& input) { return luisa::make_float4(input(0, 0), input(1, 0), input(2, 0), input(3, 0)); };

// SPD projection
template<int N>
Eigen::Matrix<float, N, N> spd_projection(const Eigen::Matrix<float, N, N>& orig_matrix)
{
    // Ensure the matrix is symmetric
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, N, N>> eigensolver(orig_matrix);
    Eigen::Matrix<float, N, 1> eigenvalues = eigensolver.eigenvalues();
    Eigen::Matrix<float, N, N> eigenvectors = eigensolver.eigenvectors();

    // Set negative eigenvalues to zero (or abs, as in your python code)
    for (int i = 0; i < N; ++i) 
    {
        eigenvalues[i] = std::max(0.0f, eigenvalues[i]);
        // eigenvalues(i) = std::abs(eigenvalues(i));
    }

    // Reconstruct the matrix: V * diag(lam) * V^T
    Eigen::Matrix<float, N, N> D = eigenvalues.asDiagonal();
    return eigenvectors * D * eigenvectors.transpose();
}

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
        lcsv::SolverInterface::physics_step_prev_operation(); 
        CpuParallel::parallel_for(0, sa_x.size(), [&](const uint vid)
        {
            sa_x_step_start[vid] = host_mesh_data->sa_x_frame_outer[vid];
            sa_v_step_start[vid] = host_mesh_data->sa_v_frame_outer[vid];
            sa_x[vid] = host_mesh_data->sa_x_frame_outer[vid];
            sa_v[vid] = host_mesh_data->sa_v_frame_outer[vid];
        });
        std::fill(host_mesh_data->sa_system_energy.begin(), host_mesh_data->sa_system_energy.end(), 0.0f);
    }
    
    std::vector<float3>& sa_cgX = host_sim_data->sa_cgX;
    std::vector<float3>& sa_cgB = host_sim_data->sa_cgB;
    std::vector<float3x3>& sa_cgA_diag = host_sim_data->sa_cgA_diag;
    std::vector<float3x3>& sa_cgA_offdiag = host_sim_data->sa_cgA_offdiag; // Row-major for simplier SpMV
 
    std::vector<float3x3>& sa_cgMinv = host_sim_data->sa_cgMinv;
    std::vector<float3>& sa_cgP = host_sim_data->sa_cgP;
    std::vector<float3>& sa_cgQ = host_sim_data->sa_cgQ;
    std::vector<float3>& sa_cgR = host_sim_data->sa_cgR;
    std::vector<float3>& sa_cgZ = host_sim_data->sa_cgZ;

    auto fast_dot = [](const std::vector<float3>& left_ptr, const std::vector<float3>& right_ptr) -> float
    {
        return CpuParallel::parallel_for_and_reduce_sum<float>(0, left_ptr.size(), [&](const uint vid)
        {
            return luisa::dot(left_ptr[vid], right_ptr[vid]);
        });
    };
    auto fast_norm = [](const std::vector<float3>& ptr) -> float
    {
        float tmp = CpuParallel::parallel_for_and_reduce_sum<float>(0, ptr.size(), [&](const uint vid)
        {
            return luisa::dot(ptr[vid], ptr[vid]);
        });
        return sqrt(tmp);
    };
    auto fast_infinity_norm = [](const std::vector<float3>& ptr) -> float // Min value in array
    {
        return CpuParallel::parallel_for_and_reduce(0, ptr.size(), [&](const uint vid)
        {
            return luisa::length(ptr[vid]);
        }, [](const float left, const float right) { return max_scalar(left, right); }, -1e9f); 
    };
    
    constexpr bool use_eigen = ConjugateGradientSolver::use_eigen;
    constexpr bool use_upper_triangle = ConjugateGradientSolver::use_upper_triangle;

    static Eigen::SparseMatrix<float> eigen_cgA;
    static Eigen::SparseMatrix<float> springA;
    static std::vector<Eigen::Triplet<float>> triplets_springA;
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
            springA.resize(mesh_data->num_verts * 3, mesh_data->num_verts * 3);
            springA.reserve(mesh_data->num_edges * 9 * 4);
            triplets_springA.resize(mesh_data->num_edges * 9 * 4);
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
    auto predict_position = [&](const float substep_dt)
    {
        auto* sa_is_fixed = host_mesh_data->sa_is_fixed.data();

        CpuParallel::parallel_for(0, host_mesh_data->num_verts, [&](const uint vid)
        {   
            const float3 gravity(0, -9.8f, 0);
            float3 x_prev = sa_x_step_start[vid];
            float3 v_prev = sa_v[vid];
            float3 outer_acceleration = gravity;
            // If we consider gravity energy here, then we will not consider it in potential energy 
            float3 v_pred = v_prev + substep_dt * outer_acceleration;
            if (sa_is_fixed[vid] != 0) { v_pred = Zero3; };

            sa_x_iter_start[vid] = x_prev;
            const float3 x_pred = x_prev + substep_dt * v_pred; 
            sa_x_tilde[vid] = x_pred;

            // sa_x[vid] = x_pred;
            // sa_cgX[vid] = v_prev * substep_dt;
            sa_x[vid] = x_prev;
            sa_cgX[vid] = luisa::make_float3(0.0f);
        });
    };
    auto update_velocity = [&](const float substep_dt, const bool fix_scene, const float damping)
    {
        CpuParallel::parallel_for(0, host_mesh_data->num_verts, [&](const uint vid)
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
    };
    auto evaluate_inertia = [&](const float substep_dt)
    {
        auto* sa_is_fixed = host_mesh_data->sa_is_fixed.data();
        auto* sa_vert_mass = host_mesh_data->sa_vert_mass.data();

        static std::vector<Eigen::Triplet<float>> triplets_A;
        if (triplets_A.empty())
        {
            triplets_A.resize(mesh_data->num_verts * 9);
        }

        const uint num_verts = host_mesh_data->num_verts;

        CpuParallel::parallel_for(0, num_verts, [&](const uint vid)
        {
            const float3 gravity(0, -9.8f, 0);
            const float h = substep_dt;
            const float h_2_inv = 1.f / (h * h);

            float3 x_k = sa_x[vid];
            float3 x_tilde = sa_x_tilde[vid];
            // float3 v_0 = sa_v[vid];

            auto is_fixed = sa_is_fixed[vid];
            float mass = sa_vert_mass[vid];
            
            float3 gradient = -mass * h_2_inv * (x_k - x_tilde) ; // !should not add gravity
            float3x3 hessian = luisa::make_float3x3(1.0f) * mass * h_2_inv;

            if (is_fixed != 0)
            {
                hessian = luisa::make_float3x3(1.0f) * float(1E9);
                // mat = luisa::make_float3x3(1.0f);
                gradient = luisa::make_float3(0.0f);
            };
            
            // float3 dx_0 = substep_dt * v_0;

            if constexpr (use_eigen) 
            {
                const uint prefix_triplets_A = 9 * vid;
                const uint prefix_triplets_b = 3 * vid;

                // Assemble diagonal 3x3 block for vertex vid
                for (int ii = 0; ii < 3; ++ii)
                {
                    for (int jj = 0; jj < 3; ++jj)
                    {
                        triplets_A[prefix_triplets_A + ii * 3 + jj] = Eigen::Triplet<float>(3 * vid + ii, 3 * vid + jj, hessian[jj][ii]); // mat[i][j] is ok???
                    }
                }
                // Assemble gradient
                eigen_cgB.segment<3>(prefix_triplets_b) = float3_to_eigen3(gradient);
                // eigen_cgX.segment<3>(prefix_triplets_b) = float3_to_eigen3(dx_0);
            }
            else 
            {  
                // sa_cgX[vid] = dx_0;
                sa_cgB[vid] = gradient;
                sa_cgA_diag[vid] = hessian;
            }
        });
        if constexpr (use_eigen) { eigen_cgA.setFromTriplets(triplets_A.begin(), triplets_A.end()); }
    };
    auto reset_off_diag = [&]()
    {
        if constexpr (use_eigen)
        {
            springA.setZero();
        }
        else 
        {
            CpuParallel::parallel_set(sa_cgA_offdiag, luisa::make_float3x3(0.0f));
        }
    };
    auto reset_diag_and_cgb = [&]()
    {
        if constexpr (use_eigen)
        {
            eigen_cgA.setZero();
            eigen_cgB.setZero();
            eigen_cgX.setZero();
        }
        else 
        {
            CpuParallel::parallel_set(sa_cgA_diag, luisa::make_float3x3(0.0f));
            CpuParallel::parallel_set(sa_cgB, luisa::make_float3(0.0f));
            CpuParallel::parallel_set(sa_cgX, luisa::make_float3(0.0f));
        }
    };
    auto evaluete_spring = [&](const float stiffness_stretch)
    {
        const uint num_verts = host_mesh_data->num_verts;
        const uint num_edges = host_mesh_data->num_edges;
        
        // auto& culster = host_xpbd_data->sa_clusterd_springs;
        // auto& sa_edges = host_mesh_data->sa_edges;
        // auto& sa_rest_length = host_mesh_data->sa_edges_rest_state_length;
        
        auto& culster = host_sim_data->sa_prefix_merged_springs;
        auto& sa_edges = host_sim_data->sa_merged_edges;
        auto& sa_rest_length = host_sim_data->sa_merged_edges_rest_length;

        for (uint cluster_idx = 0; cluster_idx < host_sim_data->num_clusters_springs; cluster_idx++) 
        {
            const uint curr_prefix = culster[cluster_idx];
            const uint next_prefix = culster[cluster_idx + 1];
            const uint num_elements_clustered = next_prefix - curr_prefix;

            // CpuParallel::single_thread_for(0, mesh_data->num_edges, [&](const uint eid)
            CpuParallel::parallel_for(0, num_elements_clustered, [&](const uint index)
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

                if constexpr (use_eigen) 
                {
                    const uint prefix_triplets_A = 9 * eid * 4;
                    const uint prefix_triplets_b = 3 * eid * 2;

                    // Assemble 3x3 blocks for edge (off-diagonal and diagonal)
                    for (int ii = 0; ii < 3; ++ii)
                    {
                        for (int jj = 0; jj < 3; ++jj) 
                        {
                            // Diagonal blocks
                            triplets_springA[prefix_triplets_A + 4 * (ii * 3 + jj) + 0] = Eigen::Triplet<float>(3 * edge[0] + ii, 3 * edge[0] + jj, He[ii][jj]);
                            triplets_springA[prefix_triplets_A + 4 * (ii * 3 + jj) + 1] = Eigen::Triplet<float>(3 * edge[1] + ii, 3 * edge[1] + jj, He[ii][jj]);

                            // Off-diagonal blocks
                            triplets_springA[prefix_triplets_A + 4 * (ii * 3 + jj) + 2] = Eigen::Triplet<float>(3 * edge[0] + ii, 3 * edge[1] + jj, -He[ii][jj]);
                            triplets_springA[prefix_triplets_A + 4 * (ii * 3 + jj) + 3] = Eigen::Triplet<float>(3 * edge[1] + ii, 3 * edge[0] + jj, -He[ii][jj]);
                        }
                    }
                    // Assemble force to gradient
                    eigen_cgB.segment<3>(3 * edge[0]) += float3_to_eigen3(force[0]);
                    eigen_cgB.segment<3>(3 * edge[1]) += float3_to_eigen3(force[1]);
                }
                else
                {
                    sa_cgB[edge[0]] = sa_cgB[edge[0]] + force[0];
                    sa_cgB[edge[1]] = sa_cgB[edge[1]] + force[1];
                    sa_cgA_diag[edge[0]] = sa_cgA_diag[edge[0]] + He;
                    sa_cgA_diag[edge[1]] = sa_cgA_diag[edge[1]] + He;
                    // sa_cgA_diag[edge[0]] = sa_cgA_diag[edge[0]] + eigen3x3_to_float3x3(projected_hessian.block<3, 3>(0, 0));
                    // sa_cgA_diag[edge[1]] = sa_cgA_diag[edge[1]] + eigen3x3_to_float3x3(projected_hessian.block<3, 3>(3, 3));
                    
                    if constexpr (use_upper_triangle)
                    {
                        for (uint ii = 0; ii < 2; ii++)
                        {
                            for (uint jj = ii + 1; jj < 2; jj++)
                            {
                                const uint hessian_index = host_sim_data->sa_hessian_slot_per_edge[eid];
                                sa_cgA_offdiag[hessian_index] = sa_cgA_offdiag[hessian_index] - He;
                            }
                        }
                    }
                    else 
                    {
                        sa_cgA_offdiag[eid * 2 + 0] = -1.0f * He;
                        sa_cgA_offdiag[eid * 2 + 1] = -1.0f * He;
                        // sa_cgA_offdiag[eid * 2 + 0] = eigen3x3_to_float3x3(projected_hessian.block<3, 3>(0, 3));
                        // sa_cgA_offdiag[eid * 2 + 1] = eigen3x3_to_float3x3(projected_hessian.block<3, 3>(3, 0));
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
                    //         sa_cgA_offdiag[hessian_index] = sa_cgA_offdiag[hessian_index] + need_transpose ? luisa::transpose(offdiag_hessian) : offdiag_hessian;
                    //     }
                    // }
                    
                }
            }, 32);
        }

        if constexpr (use_eigen) 
        {
            // Add spring contributions to cgA and cg_b_vec
            springA.setFromTriplets(triplets_springA.begin(), triplets_springA.end());
            eigen_cgA += springA;
        }
    };
    auto pcg_spmv = [&](const std::vector<float3>& input_ptr, std::vector<float3>& output_ptr) -> void
    {   
        // Diag
        CpuParallel::parallel_for(0, input_ptr.size(), [&](const uint vid)
        {
            float3x3 A_diag = sa_cgA_diag[vid];
            float3 input_vec = input_ptr[vid];
            float3 diag_output = A_diag * input_vec;
            output_ptr[vid] = diag_output;
        });
        // Off-diag: Material energy hessian
        if constexpr (use_upper_triangle)
        {
            auto& cluster = host_sim_data->sa_clusterd_hessian_pairs;
            auto& sa_hessian_set = host_sim_data->sa_hessian_pairs;
            
            for (uint cluster_idx = 0; cluster_idx < host_sim_data->num_clusters_hessian_pairs; cluster_idx++) 
            {
                const uint curr_prefix = cluster[cluster_idx];
                const uint next_prefix = cluster[cluster_idx + 1];
                const uint num_elements_clustered = next_prefix - curr_prefix;

                CpuParallel::parallel_for(0, num_elements_clustered, [&](const uint index)
                {
                    const uint pair_idx = cluster[curr_prefix + index];
                    const uint2 pair = sa_hessian_set[pair_idx];
                    float3x3 offdiag_hessian = sa_cgA_offdiag[pair_idx];
                    float3 output_vec0 = offdiag_hessian * input_ptr[pair[1]];
                    float3 output_vec1 = luisa::transpose(offdiag_hessian) * input_ptr[pair[0]];
                    output_ptr[pair[0]] += output_vec0;
                    output_ptr[pair[1]] += output_vec1;
                });
            }
        }
        else
        {
            // Hessian free
            // auto& sa_edges = host_mesh_data->sa_edges;
            // auto& cluster = host_xpbd_data->sa_clusterd_springs;

            auto& sa_edges = host_sim_data->sa_merged_edges;
            auto& cluster = host_sim_data->sa_prefix_merged_springs;
            
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

                    float3x3 offdiag_hessian1 = sa_cgA_offdiag[2 * eid + 0];
                    float3x3 offdiag_hessian2 = sa_cgA_offdiag[2 * eid + 1];
                    float3 output_vec0 = offdiag_hessian1 * input_ptr[edge[1]];
                    float3 output_vec1 = offdiag_hessian2 * input_ptr[edge[0]];
                    output_ptr[edge[0]] += output_vec0;
                    output_ptr[edge[1]] += output_vec1;
                });
            }
        }

        // Off-diag: Collision hessian
        mp_narrowphase_detector->host_spmv(stream, input_ptr, output_ptr);

    };


    const float thickness = 0;
    const float d_hat = 1e-2;
    const float kappa = 1e5;

    // Init LBVH
    {
        mp_lbvh_face->reduce_face_tree_aabb(stream, sim_data->sa_x, mesh_data->sa_faces);
        mp_lbvh_edge->reduce_edge_tree_aabb(stream, sim_data->sa_x, mesh_data->sa_edges);
        mp_lbvh_face->construct_tree(stream);
        mp_lbvh_edge->construct_tree(stream);
    }
    auto broadphase_ccd = [&]()
    {
        const float ccd_query_range = d_hat + thickness;

        // if (lcsv::get_scene_params().current_nonlinear_iter == 0)
        // {
        //     mp_lbvh_face->reduce_face_tree_aabb(stream, sim_data->sa_x, mesh_data->sa_faces);
        //     mp_lbvh_face->construct_tree(stream);
        // }
        mp_lbvh_face->update_face_tree_leave_aabb(stream, sim_data->sa_x_iter_start, sim_data->sa_x, mesh_data->sa_faces);
        mp_lbvh_face->refit(stream);
        mp_lbvh_face->broad_phase_query_from_verts(stream, 
            sim_data->sa_x_iter_start, 
            sim_data->sa_x, 
            collision_data->broad_phase_collision_count.view(collision_data->get_vf_count_offset(), 1), 
            collision_data->broad_phase_list_vf, ccd_query_range);

        // if (lcsv::get_scene_params().current_nonlinear_iter == 0)
        // {
        //     mp_lbvh_edge->reduce_edge_tree_aabb(stream, sim_data->sa_x, mesh_data->sa_edges);
        //     mp_lbvh_edge->construct_tree(stream);
        // }
        mp_lbvh_edge->update_edge_tree_leave_aabb(stream, sim_data->sa_x_iter_start, sim_data->sa_x, mesh_data->sa_edges);
        mp_lbvh_edge->refit(stream);
        mp_lbvh_edge->broad_phase_query_from_edges(stream, 
            sim_data->sa_x_iter_start, 
            sim_data->sa_x, 
            mesh_data->sa_edges, 
            collision_data->broad_phase_collision_count.view(collision_data->get_ee_count_offset(), 1), 
            collision_data->broad_phase_list_ee, ccd_query_range);
    };
    auto narrowphase_ccd = [&]()
    {
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
    };
    auto ccd_line_search = [&]() -> float
    {
        stream 
            << sim_data->sa_x_iter_start.copy_from(host_sim_data->sa_x_iter_start.data())
            << sim_data->sa_x.copy_from(host_sim_data->sa_x.data())
            // << luisa::compute::synchronize()
            ;
        mp_narrowphase_detector->reset_broadphase_count(stream);
        mp_narrowphase_detector->reset_narrowphase_count(stream);

        broadphase_ccd();

        mp_narrowphase_detector->reset_toi(stream);
        mp_narrowphase_detector->download_broadphase_collision_count(stream);
        
        narrowphase_ccd();
        
        float toi = mp_narrowphase_detector->get_global_toi(stream);
        return toi; // 0.9f * toi
        // return 1.0f;
    };

    auto broadphase_dcd = [&]()
    {
        const float dcd_query_range = d_hat + thickness;

        mp_lbvh_face->update_face_tree_leave_aabb(stream, sim_data->sa_x, sim_data->sa_x, mesh_data->sa_faces);
        mp_lbvh_face->refit(stream);
        mp_lbvh_face->broad_phase_query_from_verts(stream, 
            sim_data->sa_x, 
            sim_data->sa_x, 
            collision_data->broad_phase_collision_count.view(collision_data->get_vf_count_offset(), 1), 
            collision_data->broad_phase_list_vf, dcd_query_range);

        mp_lbvh_edge->update_edge_tree_leave_aabb(stream, sim_data->sa_x, sim_data->sa_x, mesh_data->sa_edges);
        mp_lbvh_edge->refit(stream);
        mp_lbvh_edge->broad_phase_query_from_edges(stream, 
            sim_data->sa_x, 
            sim_data->sa_x, 
            mesh_data->sa_edges, 
            collision_data->broad_phase_collision_count.view(collision_data->get_ee_count_offset(), 1), 
            collision_data->broad_phase_list_ee, dcd_query_range);
    };
    auto narrowphase_dcd = [&]()
    {
        mp_narrowphase_detector->download_broadphase_collision_count(stream);
        
        mp_narrowphase_detector->vf_dcd_query(stream, 
            sim_data->sa_x, 
            sim_data->sa_x, 
            mesh_data->sa_faces, 
            d_hat, thickness, kappa);
        mp_narrowphase_detector->ee_dcd_query(stream, 
            sim_data->sa_x, 
            sim_data->sa_x, 
            mesh_data->sa_edges, 
            mesh_data->sa_edges, 
            d_hat, thickness, kappa);
    };
    auto On2_update_barrier_set = [&]()
    {
        mp_narrowphase_detector->host_ON2_dcd_query_libuipc(eigen_cgA, eigen_cgB, 
            host_sim_data->sa_x, 
            host_sim_data->sa_x, 
            host_mesh_data->sa_rest_x, 
            host_mesh_data->sa_rest_x, 
            host_mesh_data->sa_faces, 
            host_mesh_data->sa_faces, 
            host_mesh_data->sa_edges, 
            host_mesh_data->sa_edges, 
            d_hat, thickness, kappa);
    };
    auto update_barrier_set = [&]()
    {
        if constexpr (use_eigen)
        {
            On2_update_barrier_set();
            return;
        }
        stream 
            // << sim_data->sa_x_iter_start.copy_from(host_sim_data->sa_x_iter_start.data())
            << sim_data->sa_x.copy_from(host_sim_data->sa_x.data())
            // << luisa::compute::synchronize()
            ;

        mp_narrowphase_detector->reset_broadphase_count(stream);
        mp_narrowphase_detector->reset_narrowphase_count(stream);

        broadphase_dcd();
        
        narrowphase_dcd();

        mp_narrowphase_detector->download_narrowphase_collision_count(stream);
        mp_narrowphase_detector->download_narrowphase_list(stream);  
        
        mp_narrowphase_detector->host_barrier_hessian_spd_projection(stream);
        mp_narrowphase_detector->upload_spd_narrowphase_list(stream);

        stream 
            << sim_data->sa_cgB.copy_from(sa_cgB.data())
            << sim_data->sa_cgA_diag.copy_from(sa_cgA_diag.data());

        mp_narrowphase_detector->barrier_hessian_assemble(stream, sim_data->sa_cgB, sim_data->sa_cgA_diag);

        stream 
            << sim_data->sa_cgB.copy_to(sa_cgB.data())
            << sim_data->sa_cgA_diag.copy_to(sa_cgA_diag.data())
            << luisa::compute::synchronize();
    };
    auto compute_barrier_energy_from_ccd_list = [&]() -> float
    {
        stream 
            << sim_data->sa_x.copy_from(sa_x.data());

        mp_narrowphase_detector->reset_energy(stream);

        mp_narrowphase_detector->compute_barrier_energy_from_vf(stream, 
            sim_data->sa_x, 
            sim_data->sa_x, 
            mesh_data->sa_faces, 
            d_hat, thickness, kappa);

        mp_narrowphase_detector->compute_barrier_energy_from_ee(stream, 
            sim_data->sa_x, 
            sim_data->sa_x, 
            mesh_data->sa_edges, 
            mesh_data->sa_edges, 
            d_hat, thickness, kappa);

        return mp_narrowphase_detector->download_energy(stream, kappa);
        // return 0.0f;
    };

    
    auto On2_compute_barrier_energy = [&]() -> float
    {
        return mp_narrowphase_detector->host_ON2_compute_barrier_energy_uipc(
            host_sim_data->sa_x, 
            host_sim_data->sa_x, 
            host_mesh_data->sa_rest_x, 
            host_mesh_data->sa_rest_x, 
            host_mesh_data->sa_faces, 
            host_mesh_data->sa_faces, 
            host_mesh_data->sa_edges, 
            host_mesh_data->sa_edges, 
            d_hat, thickness, kappa
        );
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
                float dt = lcsv::get_scene_params().get_substep_dt();
                float3 vel =  dx / dt;
                sa_x[vid] += dx;
            };
        }); 
    };
    
    
    auto compute_energy_with_barrier = [&](const std::vector<float3> &curr_x, const std::vector<float3> &curr_x_tilde)
    {
        auto material_energy = host_compute_energy(curr_x, curr_x_tilde);
        auto barrier_energy = On2_compute_barrier_energy();;
        return material_energy + barrier_energy;
    };
    auto linear_solver_interface = [&]()
    {
        if constexpr (use_eigen) 
        {
            pcg_solver->eigen_solve(eigen_cgA, eigen_cgX, eigen_cgB, compute_energy_with_barrier);
            // eigen_decompose_solve();
        } 
        else 
        {
            // simple_solve();
            pcg_solver->host_solve(stream, pcg_spmv, compute_energy_with_barrier);
        }
    };
    

    const float substep_dt = lcsv::get_scene_params().get_substep_dt();
    const bool use_ipc = true;

    
    for (uint substep = 0; substep < get_scene_params().num_substep; substep++)
    {
        predict_position(substep_dt);
        
        luisa::log_info("In frame {} : ", get_scene_params().current_frame); 

        // double barrier_nergy = compute_barrier_energy_from_broadphase_list();
        double prev_state_energy = compute_energy_with_barrier(sa_x_step_start, sa_x_tilde);

        for (uint iter = 0; iter < get_scene_params().nonlinear_iter_count; iter++)
        {   get_scene_params().current_nonlinear_iter = iter;
            
            CpuParallel::parallel_set(sa_cgX, luisa::make_float3(0.0f)); // 

            reset_diag_and_cgb();
            reset_off_diag();
            {
                evaluate_inertia(substep_dt);

                evaluete_spring(1e4);

                update_barrier_set();
            }
            linear_solver_interface(); // Solve Ax=b

            float alpha = 1.0f;
            host_apply_dx(alpha);            
            if constexpr (use_ipc)
            { 
                const float ccd_toi = ccd_line_search();
                alpha = ccd_toi;
                host_apply_dx(alpha);   

                auto curr_energy = compute_energy_with_barrier(sa_x, sa_x_tilde); 
                if (is_nan_scalar(curr_energy) || is_inf_scalar(curr_energy)) { luisa::log_error("Energy is not valid : {}", curr_energy); }
                
                uint line_search_count = 0;
                while (line_search_count < 20)
                {
                    if (curr_energy < prev_state_energy + Epsilon) 
                    { 
                        break; 
                    }
                    if (line_search_count == 0)
                    {
                        luisa::log_info("     Line search {} : alpha = {:6.5f}, energy = {:12.10f} , prev state energy {:12.10f} , {}", 
                            line_search_count, alpha, curr_energy, prev_state_energy, 
                            ccd_toi != 1.0f ? "CCD toi = " + std::to_string(ccd_toi) : "");
                    }
                    alpha /= 2; host_apply_dx(alpha);
                    line_search_count++;

                    curr_energy = compute_energy_with_barrier(sa_x, sa_x_tilde);
                    luisa::log_info("     Line search {} : alpha = {:6.5f}, energy = {:12.10f}", 
                        line_search_count, alpha, curr_energy);
                    
                    if (alpha < 1e-4) 
                    {
                        luisa::log_error("  Line search failed, energy = {}, prev state energy = {}", 
                            curr_energy, prev_state_energy);
                    }
                }

                prev_state_energy = curr_energy; // E_prev = E
            }

            // Non-linear iteration break condition
            {
                float max_move = 1e-2;
                float curr_max_step = fast_infinity_norm(sa_cgX); 
                if (curr_max_step < max_move * substep_dt) 
                {
                    // luisa::log_info("  In non-linear iter {:2}: Iteration break for small searching direction {} < {}", iter, curr_max_step, max_move * substep_dt);
                    break;
                }
            }

            CpuParallel::parallel_copy(sa_x, sa_x_iter_start); // x_prev = x
        }
        update_velocity(substep_dt, false, lcsv::get_scene_params().damping_cloth);
    }

    // Output
    {
        CpuParallel::parallel_for(0, sa_x.size(), [&](const uint vid)
        {
            host_mesh_data->sa_x_frame_outer[vid] = sa_x[vid];
            host_mesh_data->sa_v_frame_outer[vid] = sa_v[vid];
        });
        lcsv::SolverInterface::physics_step_post_operation(); 
    }
}
void NewtonSolver::physics_step_GPU(luisa::compute::Device& device, luisa::compute::Stream& stream)
{
    lcsv::SolverInterface::physics_step_prev_operation(); 
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
        
           << mp_buffer_filler->fill(device, mesh_data->sa_system_energy, 0.0f)
           << luisa::compute::synchronize();
    
    // const uint num_substep = lcsv::get_scene_params().print_xpbd_convergence ? 1 : lcsv::get_scene_params().num_substep;
    const uint num_substep = lcsv::get_scene_params().num_substep;
    const uint nonlinear_iter_count = lcsv::get_scene_params().nonlinear_iter_count;
    const float substep_dt = lcsv::get_scene_params().get_substep_dt();

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
    auto host_dot = [](const std::vector<float3>& left_ptr, const std::vector<float3>& right_ptr) -> float
    {
        return CpuParallel::parallel_for_and_reduce_sum<float>(0, left_ptr.size(), [&](const uint vid)
        {
            return luisa::dot(left_ptr[vid], right_ptr[vid]);
        });
    };
    auto host_norm = [](const std::vector<float3>& ptr) -> float
    {
        float tmp = CpuParallel::parallel_for_and_reduce_sum<float>(0, ptr.size(), [&](const uint vid)
        {
            return luisa::dot(ptr[vid], ptr[vid]);
        });
        return sqrt(tmp);
    };
    auto host_infinity_norm = [](const std::vector<float3>& ptr) -> float // Min value in array
    {
        return CpuParallel::parallel_for_and_reduce(0, ptr.size(), [&](const uint vid)
        {
            return luisa::length(ptr[vid]);
        }, [](const float left, const float right) { return max_scalar(left, right); }, -1e9f); 
    };
    

    const bool use_ipc = true;
    const uint num_verts = host_mesh_data->num_verts;
    const uint num_edges = host_mesh_data->num_edges;
    const uint num_faces = host_mesh_data->num_faces;
    const uint num_blocks_verts = get_dispatch_block(num_verts, 256);

    
    auto pcg_spmv = [&](const luisa::compute::Buffer<float3>& input_ptr, luisa::compute::Buffer<float3>& output_ptr) -> void
    {   
        stream 
            << fn_pcg_spmv_diag(input_ptr, output_ptr).dispatch(num_verts);

        auto& culster = host_sim_data->sa_prefix_merged_springs;
        for (uint cluster_idx = 0; cluster_idx < host_sim_data->num_clusters_springs; cluster_idx++) 
        {
            const uint curr_prefix = culster[cluster_idx];
            const uint next_prefix = culster[cluster_idx + 1];
            const uint num_elements_clustered = next_prefix - curr_prefix;
            stream << fn_pcg_spmv_offdiag(input_ptr, output_ptr, curr_prefix).dispatch(num_elements_clustered);
        }
    };
    auto host_compute_barrier_energy = []() { return 0.0f; };
    auto compute_energy_with_barrier = [&](const luisa::compute::Buffer<float3>& curr_x, const luisa::compute::Buffer<float3>& curr_x_tilde)
    {
        auto material_energy = host_compute_energy(host_x, host_x_tilde);
        auto barrier_energy = host_compute_barrier_energy();;
        return material_energy + barrier_energy;
    };

    for (uint substep = 0; substep < get_scene_params().num_substep; substep++)
    {
        stream 
            << fn_predict_position(substep_dt).dispatch(num_verts)
            << sim_data->sa_x_step_start.copy_to(host_x_step_start.data())
            << sim_data->sa_x_tilde.copy_to(host_x_tilde.data())
            << luisa::compute::synchronize();
        
        auto prev_state_energy = host_compute_energy(host_x_step_start, host_x_tilde);
        luisa::log_info("In frame {} : ", get_scene_params().current_frame);

        for (uint iter = 0; iter < get_scene_params().nonlinear_iter_count; iter++)
        {   get_scene_params().current_nonlinear_iter = iter;
            
            stream 
                << sim_data->sa_x_iter_start.copy_to(host_x_iter_start.data())
                << fn_reset_vector(sim_data->sa_cgX, makeFloat3(0.0f)).dispatch(num_verts)
                << fn_reset_offdiag().dispatch(sim_data->sa_cgA_offdiag.size())
                << fn_evaluate_inertia(substep_dt).dispatch(num_verts);

            {
                auto& culster = host_sim_data->sa_prefix_merged_springs;
                for (uint cluster_idx = 0; cluster_idx < host_sim_data->num_clusters_springs; cluster_idx++) 
                {
                    const uint curr_prefix = culster[cluster_idx];
                    const uint next_prefix = culster[cluster_idx + 1];
                    const uint num_elements_clustered = next_prefix - curr_prefix;
                    stream << fn_evaluate_spring(1e4, curr_prefix).dispatch(num_elements_clustered);
                }
            }
                
            // device_pcg();
            pcg_solver->device_solve(stream, pcg_spmv, compute_energy_with_barrier);

            // Do line search on the host
            float alpha = 1.0f;
            host_apply_dx(alpha);

            if constexpr (use_ipc)
            { 
                // alpha = ccd_line_search();
                // apply_dx(alpha);   

                auto curr_energy = host_compute_energy(host_x, host_x_tilde);
                uint line_search_count = 0;
                while (line_search_count < 20)
                {
                    if (curr_energy < prev_state_energy) { break; }
                    if (line_search_count == 0)
                    {
                        luisa::log_info("     Line search {} : alpha = {:6.5f}, energy = {:12.10f} , prev state energy {:12.10f}", 
                            line_search_count, alpha, curr_energy, prev_state_energy);
                    }
                    alpha /= 2; host_apply_dx(alpha);
                    line_search_count++;

                    curr_energy = host_compute_energy(host_x, host_x_tilde);
                    luisa::log_info("     Line search {} : alpha = {:6.5f}, energy = {:12.10f}", 
                        line_search_count, alpha, curr_energy);
                }
                prev_state_energy = curr_energy;
            }

            // Non-linear iteration break condition
            {
                float max_move = 1e-2;
                float curr_max_step = host_infinity_norm(host_cgX); 
                if (curr_max_step < max_move * substep_dt) 
                {
                    luisa::log_info("  In non-linear iter {:2}: Iteration break for small searching direction {} < {}", iter, curr_max_step, max_move * substep_dt);
                    break;
                }
            }

            CpuParallel::parallel_copy(host_x, host_x_iter_start);

            stream
                << sim_data->sa_x.copy_from(host_x.data())
                << sim_data->sa_x_iter_start.copy_from(host_x_iter_start.data());
        }

        stream
            << fn_update_velocity(substep_dt, false, lcsv::get_scene_params().damping_cloth).dispatch(num_verts)
            // << luisa::compute::synchronize();
            ;
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
    lcsv::SolverInterface::physics_step_post_operation(); 
}

float NewtonSolver::device_compute_energy(luisa::compute::Stream& stream, const luisa::compute::BufferView<float3>& curr_x)
{
    
    // luisa::log_info("    Energy {} = inertia {} + stretch {}", energy_inertia + energy_spring, energy_inertia, energy_spring);
    // return energy_inertia + energy_spring;
    return 0.0f;
};

} // namespace lcsv