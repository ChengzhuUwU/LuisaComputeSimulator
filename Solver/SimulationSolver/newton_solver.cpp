#include <iostream>
#include <Eigen/Sparse>
#include "SimulationSolver/descent_solver.h"
#include "SimulationSolver/newton_solver.h"
#include "Core/float_nxn.h"
#include "Core/scalar.h"
#include "Core/xbasic_types.h"
#include "SimulationCore/scene_params.h"
#include "Utils/cpu_parallel.h"
#include <luisa/dsl/sugar.h>

namespace lcsv 
{

void NewtonSolver::compile(luisa::compute::Device& device)
{
    const bool use_debug_info = false;
    using namespace luisa::compute;

    luisa::compute::ShaderOption default_option = {.enable_debug_info = false};

    auto fn_init_force = device.compile<1>(
        [
            sa_x = sim_data->sa_x.view(),
            sa_v = sim_data->sa_v.view(),
            sa_x_start = sim_data->sa_x_step_start.view(),
            sa_is_fixed = mesh_data->sa_is_fixed.view()
        ](const Float substep_dt)
    {
        const UInt vid = dispatch_id().x;
        const Float3 gravity(0, -9.8f, 0);
        Float3 x_prev = sa_x_start->read(vid);
        Float3 v_prev = sa_v->read(vid);
        Float3 outer_acceleration = gravity;
        Float3 v_pred = v_prev + substep_dt * outer_acceleration;
        $if (sa_is_fixed->read(vid) != 0) { outer_acceleration = Zero3; v_pred = Zero3; };
        const Float3 x_pred = x_prev + substep_dt * v_pred;
        sa_x->write(vid, x_pred);
    }, default_option);

    auto fn_init_hessian = device.compile<1>(
        [
            sa_x = sim_data->sa_x.view(),
            sa_v = sim_data->sa_v.view(),
            sa_iter_start_position = sim_data->sa_x_step_start.view(),
            sa_iter_position = sim_data->sa_x.view(),
            sa_velocity_start = sim_data->sa_v_step_start.view(),
            sa_vert_velocity = sim_data->sa_v.view(),
            sa_x_start = sim_data->sa_x_step_start.view()
        ](const Float substep_dt, const Bool fix_scene, const Float damping)
        {
            const UInt vid = dispatch_id().x;

            Float3 x_k_init = sa_iter_start_position->read(vid);
            Float3 x_k = sa_iter_position->read(vid);

            Float3 dx = x_k - x_k_init;
            Float3 vel = dx / substep_dt;

            $if (fix_scene) 
            {
                dx = Zero3;
                vel = Zero3;
                sa_iter_position->write(vid, sa_x_start->read(vid));
                return;
            };

            vel *= exp(-damping * substep_dt);

            sa_vert_velocity->write(vid, vel);
            sa_velocity_start->write(vid, vel);
            sa_iter_start_position->write(vid, x_k);
        }
    );
}
void NewtonSolver::physics_step_newton_CPU(luisa::compute::Device& device, luisa::compute::Stream& stream)
{
    // Input
    {
        lcsv::SolverInterface::physics_step_prev_operation(); 
        // CpuParallel::parallel_for(0, sa_x.size(), [&](const uint vid)
        // {
        //     sa_x[vid] = host_mesh_data->sa_x_frame_start[vid];
        //     sa_v[vid] = host_mesh_data->sa_v_frame_start[vid];
        //     sa_x_start[vid] = host_mesh_data->sa_x_frame_start[vid];
        //     sa_v_start[vid] = host_mesh_data->sa_v_frame_start[vid];
        // });
        std::fill(host_mesh_data->sa_system_energy.begin(), host_mesh_data->sa_system_energy.end(), 0.0f);
    }

    std::vector<float3>& sa_x_tilde = host_sim_data->sa_x_tilde;
    std::vector<float3>& sa_x = host_sim_data->sa_x;
    std::vector<float3>& sa_v = host_sim_data->sa_v;
    std::vector<float3>& sa_x_step_start = host_sim_data->sa_x_step_start;
    std::vector<float3>& sa_x_iter_start = host_sim_data->sa_x_iter_start;
    std::vector<float3>& sa_v_step_start = host_sim_data->sa_v_step_start;
    
    static std::vector<float3> sa_cgX;
    static std::vector<float3> sa_cgB;
    static std::vector<float3x3> sa_cgA_diag;
    static std::vector<float3x3> sa_cgA_offdiag; // Row-major for simplier SpMV
 
    static std::vector<float3x3> sa_cgMinv;
    static std::vector<float3> sa_cgP;
    static std::vector<float3> sa_cgQ;
    static std::vector<float3> sa_cgR;
    static std::vector<float3> sa_cgZ;
    
    constexpr bool use_eigen = false;
    constexpr bool use_upper_triangle = false;

    const uint curr_frame = get_scene_params().current_frame;
    if (sa_cgB.empty())
    {
        const uint num_verts = host_mesh_data->num_verts;
        const uint num_edges = host_mesh_data->num_edges;
        const uint num_faces = host_mesh_data->num_faces;

        sa_cgX.resize(num_verts);
        sa_cgB.resize(num_verts);
        sa_cgA_diag.resize(num_verts);
        if constexpr (use_upper_triangle)
            sa_cgA_offdiag.resize(host_sim_data->sa_hessian_pairs.size());
        else
            sa_cgA_offdiag.resize(num_edges * 2);

        // sa_x_tilde.resize(num_verts);
        // sa_x.resize(num_verts);
        // sa_v.resize(num_verts);
        // sa_v_start.resize(num_verts);
        // sa_x_step_start.resize(num_verts);
        // sa_x_iter_start.resize(num_verts);

        sa_cgMinv.resize(num_verts);
        sa_cgP.resize(num_verts);
        sa_cgQ.resize(num_verts);
        sa_cgR.resize(num_verts);
        sa_cgZ.resize(num_verts);
        
    }

    CpuParallel::parallel_for(0, sa_x.size(), [&](const uint vid)
    {
        sa_x[vid] = host_mesh_data->sa_x_frame_start[vid];
        sa_v[vid] = host_mesh_data->sa_v_frame_start[vid];
        sa_x_step_start[vid] = host_mesh_data->sa_x_frame_start[vid];
        sa_v_step_start[vid] = host_mesh_data->sa_v_frame_start[vid];
    });


    using EigenFloat3x3 = Eigen::Matrix<float, 3, 3>;
    using EigenFloat3   = Eigen::Matrix<float, 3, 1>;
    auto float3x3_to_eigen3x3 = [](const float3x3& input)
    {
        EigenFloat3x3 mat; mat << 
            input[0][0], input[1][0], input[2][0], 
            input[0][1], input[1][1], input[2][1], 
            input[0][2], input[1][2], input[2][2]; return mat;
    };
    auto eigen3x3_to_float3x3 = [](const EigenFloat3x3& input)
    {
        return luisa::make_float3x3(
            input(0, 0), input(1, 0), input(2, 0), 
            input(0, 1), input(1, 1), input(2, 1), 
            input(0, 2), input(1, 2), input(2, 2));
    };
    auto float3_to_eigen3 = [](const float3& input) { EigenFloat3 vec; vec << input[0], input[1], input[2]; return vec; };
    auto eigen3_to_float3 = [](const EigenFloat3& input) { return luisa::make_float3(input(0, 0), input(1, 0), input(2, 0)); };

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

    auto apply_dx = [&](const float alpha)
    {
        // Update sa_x
        CpuParallel::parallel_for(0, mesh_data->num_verts, [&](const uint vid)
        {
            sa_x[vid] = sa_x_iter_start[vid] + alpha * sa_cgX[vid];
        });
    };

    // Predict Position
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
            const float3 x_pred = x_prev + substep_dt * v_pred; 
            sa_x_tilde[vid] = x_pred;

            // sa_x[vid] = x_pred;
            sa_x[vid] = x_prev; // TODO: Profiling convergence of sa_x and sa_cgX
            // sa_cgX[vid] = v_prev * substep_dt;
            sa_cgX[vid] = luisa::make_float3(0.0f);
            sa_x_iter_start[vid] = x_prev;
        });
    };

    // Update Velocity
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

    // Init vert
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
            float3x3 mat = luisa::make_float3x3(1.0f) * mass * h_2_inv;
            float3 outer_acceleration = gravity;
            float3 outer_force = mass * outer_acceleration;

            if (is_fixed != 0)
            {
                mat = mat + luisa::make_float3x3(1.0f) * float(1E9);
            };

            float3 gradient = -mass * h_2_inv * (x_k - x_tilde) + outer_force;
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
                        triplets_A[prefix_triplets_A + ii * 3 + jj] = Eigen::Triplet<float>(3 * vid + ii, 3 * vid + jj, mat[jj][ii]); // mat[i][j] is ok???
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
                sa_cgA_diag[vid] = mat;
            }
        });
        if constexpr (use_eigen) { eigen_cgA.setFromTriplets(triplets_A.begin(), triplets_A.end()); }
    };
    
    // Init energy
    auto reset_energy = [&]()
    {
        CpuParallel::parallel_set(sa_cgA_offdiag, luisa::make_float3x3(0.0f));
    };

    // Evaluate Spring
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
                force[0] = stiffness_stretch_spring * dir * C;
                force[1] = -force[0];

                float3x3 xxT = outer_product(diff, diff);
                float x_inv = 1.f / l;
                float x_squared_inv = x_inv * x_inv;
                He = stiffness_stretch_spring * x_squared_inv * xxT + stiffness_stretch_spring * max_scalar(1 - L * x_inv, 0.0f) * (luisa::make_float3x3(1.0f) - x_squared_inv * xxT);

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

    

    // Solve
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
    auto simple_solve = [&]()
    {
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
    auto simple_pcg = [&]()
    {
        auto get_dot_rz_rr = [&]() -> float2 // [0] = r^T z, [1] = r^T r
        {
            return CpuParallel::parallel_for_and_reduce_sum<float2>(0, sa_cgR.size(), [&](const uint vid) -> float2
            {
                float3 r = sa_cgR[vid];
                float3 z = sa_cgZ[vid];
                return luisa::make_float2(luisa::dot(r, z), luisa::dot(r, r));
            });
        };
        auto read_beta = [](const uint vid, std::vector<float>& sa_converage) -> float
        {
            float delta_old = sa_converage[0];
            float delta = sa_converage[2];
            float beta = delta_old == 0.0f ? 0.0f : delta / delta_old;
            if (vid == 0)  
            { 
                sa_converage[1] = 0; 
                uint iteration_idx = uint(sa_converage[8]);
                sa_converage[9 + iteration_idx] = delta;
                sa_converage[8] = float(iteration_idx + 1); 
            }
            return beta;
        };
        auto save_dot_pq = [](const uint blockIdx, std::vector<float>& sa_converage, const float dot_pq) -> void
        {
            sa_converage[1] = dot_pq; /// <= reduce
            if (blockIdx == 0)
            {
                float delta_old = sa_converage[2];
                float delta_old_old = sa_converage[0];
                sa_converage[2] = 0;
                sa_converage[0] = delta_old;
                sa_converage[4] = delta_old_old;
            }
        };
        auto read_alpha = [](std::vector<float>& sa_converage) -> float
        {
            float delta = sa_converage[0];
            float dot_pq = sa_converage[1];
            float alpha = dot_pq == 0.0f ? 0.0f : delta / dot_pq;
            return alpha;
        };
        auto save_dot_rz = [](const uint blockIdx, std::vector<float>& sa_converage, const float dot_rz) -> void
        {
            sa_converage[2] = dot_rz; /// <= reduce
        };

        const uint num_verts = mesh_data->num_verts;
        auto pcg_spmv = [&](const std::vector<float3>& input_ptr, std::vector<float3>& output_ptr) -> void
        {   
            // Diag
            CpuParallel::parallel_for(0, num_verts, [&](const uint vid)
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
        };

        auto pcg_make_preconditioner_jacobi = [&]()
        {
            CpuParallel::parallel_for(0, num_verts, [&](const uint vid)
            {
                float3x3 diagA = sa_cgA_diag[vid];
                float3x3 inv_M = luisa::inverse(diagA);
                // float3x3 inv_M = luisa::make_float3x3(
                //     luisa::make_float3(1.0f / diagA[0][0], 0.0f, 0.0f), 
                //     luisa::make_float3(0.0f, 1.0f / diagA[1][1], 0.0f), 
                //     luisa::make_float3(0.0f, 0.0f, 1.0f / diagA[2][2])
                // );
                sa_cgMinv[vid] = inv_M;
            });
        };
        auto pcg_apply_preconditioner_jacobi = [&]()
        {
            CpuParallel::parallel_for(0, num_verts, [&](const uint vid)
            {
                const float3 r = sa_cgR[vid];
                const float3x3 inv_M = sa_cgMinv[vid];
                float3 z = inv_M * r;
                sa_cgZ[vid] = z;
            });
        };

        auto pcg_init = [&]()
        {
            CpuParallel::parallel_for(0, num_verts, [&](const uint vid)
            {
                const float3 b = sa_cgB[vid];
                const float3 q = sa_cgQ[vid];
                const float3 r = b - q;  // r = b - q = b - A * x
                sa_cgR[vid] = r;
                sa_cgP[vid] = Zero3;
                sa_cgQ[vid] = Zero3;
            });
        };
        auto pcg_update_p = [&](const float beta)
        {
            CpuParallel::parallel_for(0, num_verts, [&](const uint vid)
            {
                const float3 p = sa_cgP[vid];
                sa_cgP[vid] = sa_cgZ[vid] + beta * p;
            });
        };
        auto pcg_step = [&](const float alpha)
        {
            CpuParallel::parallel_for(0, num_verts, [&](const uint vid)
            {
                sa_cgX[vid] = sa_cgX[vid] + alpha * sa_cgP[vid];
                sa_cgR[vid] = sa_cgR[vid] - alpha * sa_cgQ[vid];
            });
        };

        auto& sa_convergence = host_mesh_data->sa_pcg_convergence;
        std::fill(sa_convergence.begin(), sa_convergence.end(), 0.0f);

        pcg_spmv(sa_cgX, sa_cgQ);

        pcg_init();
        
        pcg_make_preconditioner_jacobi();

        float normR_0 = 0.0f;

        for (uint iter = 0; iter < lcsv::get_scene_params().pcg_iter_count; iter++)
        {
            lcsv::get_scene_params().current_pcg_it = iter;

            // if (get_scene_params().print_system_energy)
            // {
            //     update_position_for_energy();
            //     compute_system_energy(it);
            //     compute_pcg_residual(it);
            // }

            pcg_apply_preconditioner_jacobi();
            float dot_rz = fast_dot(sa_cgR, sa_cgZ);

            float2 dot_rr_rz = get_dot_rz_rr(); 
            dot_rz = dot_rr_rz[0];
            float normR = std::sqrt(dot_rr_rz[1]); if (iter == 0) normR_0 = normR;
            save_dot_rz(0, sa_convergence, dot_rz);

            const float beta = read_beta(0, sa_convergence);
            pcg_update_p(beta);
        
            pcg_spmv(sa_cgP, sa_cgQ);
            float dot_pq = fast_dot(sa_cgP, sa_cgQ);
            save_dot_pq(0, sa_convergence, dot_pq);

            // luisa::log_info("     PCG iter {} : rTz = {}, rTr = {}, pTq = {}",  iter, dot_rz, normR, dot_pq);
            
            // 
            // luisa::log_info("     PCG iter {} : energy = {}", iter, compute_energy(sa_x));


            if (normR < 5e-3 * normR_0 || dot_rz == 0.0f) 
            {
                apply_dx(1.0f);
                luisa::log_info("  In non-linear iter {:2}, PCG : iter-count = {:3}, error = {:6.5f}, infinity norm = {:6.5f}, energy = {:6.3f}", 
                    get_scene_params().current_nonlinear_iter,
                    iter, normR / normR_0, fast_infinity_norm(sa_cgX), host_compute_energy(sa_x)); // from normR_0 -> normR
                break;
            }

            const float alpha = read_alpha(sa_convergence);
            pcg_step(alpha);
        }

        /*
        for (uint iter = 0; iter < lcsv::get_scene_params().pcg_iter_count; iter++)
        {
            lcsv::get_scene_params().current_pcg_it = iter;
            pcg_apply_preconditioner_jacobi();
            delta_old = delta;
            float2 dot_rr_rz = get_dot_rz_rr(); delta = dot_rr_rz[0];
            float normR = std::sqrt(dot_rr_rz[1]); if (iter == 0) normR_0 = normR;
            float beta = delta_old == 0.0f ? 0.0f : delta / delta_old;
            pcg_update_p(beta);
            pcg_spmv(sa_cgP, sa_cgQ);
            float dot_pq = fast_dot(sa_cgP, sa_cgQ);
            if (normR < 5e-3 * normR_0) 
            {
                break;
            }
            const float alpha = dot_pq == 0.0f ? 0.0f : delta / dot_pq;
            pcg_step(alpha);   
        }
        */
    };

    auto eigen_iter_solve = [&]()
    {
        // Solve cgA * dx = cg_b_vec for dx using Conjugate Gradient
        Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower> solver; // Eigen::IncompleteCholesky<float>

        // solver.setMaxIterations(128);
        solver.setTolerance(1e-2f);
        solver.compute(eigen_cgA);

        // 计算Jacobi预条件子的对角线逆
        Eigen::VectorXf eigen_cgR = eigen_cgB - eigen_cgA * eigen_cgX;
        Eigen::VectorXf eigen_cgM_inv(eigen_cgR.rows());
        for (int i = 0; i < eigen_cgR.rows(); ++i) {
            float diag = eigen_cgA.coeff(i, i);
            eigen_cgM_inv[i] = (std::abs(diag) > 1e-12f) ? (1.0f / diag) : 0.0f;
        }
        Eigen::VectorXf eigen_cgZ = eigen_cgR.cwiseProduct(eigen_cgM_inv);
        Eigen::VectorXf eigen_cgQ = eigen_cgA * eigen_cgZ;
        luisa::log_info("initB = {}, initR = {}, initM = {}, initZ = {}, initQ = {}",
            eigen_cgB.norm(), eigen_cgR.norm(), eigen_cgM_inv.norm(), eigen_cgZ.norm(), eigen_cgQ.norm());

        solver._solve_impl(eigen_cgB, eigen_cgX);
        if (solver.info() != Eigen::Success) { luisa::log_error("Eigen: Solve failed in {} iterations", solver.iterations()); }
        else 
        {
            luisa::log_info("  In non-linear iter {}, Eigen-PCG : iter-count = {}, relative error = {}", 
                    get_scene_params().current_nonlinear_iter,
                    solver.iterations(), solver.error());
        }
    };
    auto eigen_decompose_solve = [&]()
    {
        // Solve cgA * dx = cg_b_vec for dx using SimplicialLDLT decomposition
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
        solver.compute(eigen_cgA);
        if (solver.info() != Eigen::Success)
        {
            luisa::log_error("Eigen: SimplicialLDLT decomposition failed!");
            return;
        }
        solver._solve_impl(eigen_cgB, eigen_cgX);
        if (solver.info() != Eigen::Success)
        {
            luisa::log_error("Eigen: SimplicialLDLT solve failed!");
            return;
        }
        else
        {
            float error = (eigen_cgB - eigen_cgA * eigen_cgX).norm();
            luisa::log_info("  In non-linear iter {}, Eigen-Decompose: ", 
                    get_scene_params().current_nonlinear_iter);
            // luisa::log_info("  In non-linear iter {}, Eigen-Decompose : relative error = {}", 
            //         get_scene_params().current_nonlinear_iter,
            //         error);
        }
    };
    auto linear_solver_interface = [&]()
    {
        if constexpr (use_eigen) 
        {
            // eigen_iter_solve();
            eigen_decompose_solve();
            CpuParallel::parallel_for(0, mesh_data->num_verts, [&](const uint vid)
            {
                sa_cgX[vid] = eigen3_to_float3(eigen_cgX.segment<3>(3 * vid));
            });
        } 
        else 
        {
            // simple_solve();
            simple_pcg();
        }
    };

    const float substep_dt = lcsv::get_scene_params().get_substep_dt();
    const bool use_ipc = true;

    
    for (uint substep = 0; substep < get_scene_params().num_substep; substep++)
    {
        predict_position(substep_dt);
        
        const auto step_start_energy = host_compute_energy(sa_x_step_start);

        luisa::log_info("In frame {} : Frame init energy = {}", get_scene_params().current_frame, step_start_energy);

        for (uint iter = 0; iter < get_scene_params().nonlinear_iter_count; iter++)
        {   get_scene_params().current_nonlinear_iter = iter;
            
            CpuParallel::parallel_set(sa_cgX, luisa::make_float3(0.0f)); // 

            reset_energy();
            {
                evaluate_inertia(substep_dt);
                evaluete_spring(1e4);
            }
            linear_solver_interface();
            
            if constexpr (use_ipc)
            {
                float max_move = 1e-2;
                float curr_max_step = fast_infinity_norm(sa_cgX); 
                if (curr_max_step < max_move * substep_dt) 
                {
                    luisa::log_info("  Non-linear iteration break for small searching direction {} < {}", curr_max_step, max_move * substep_dt);
                    break;
                }
            }

            float alpha = 1.0f;
            apply_dx(alpha);
            if constexpr (use_ipc)
            { 
                auto curr_energy = host_compute_energy(sa_x);
                uint line_search_count = 0;
                while (line_search_count < 12)
                {
                    if (curr_energy < step_start_energy + 0.00001) { break; }
                    if (line_search_count == 0)
                    {
                        luisa::log_info("     Line search {} : alpha = 1/{}, energy = {:6.3f} Frame-start-energy = {:6.3f}", 
                            line_search_count, (1 << line_search_count), curr_energy, step_start_energy);
                    }
                    alpha /= 2; apply_dx(alpha);
                    line_search_count++;

                    curr_energy = host_compute_energy(sa_x);
                    luisa::log_info("     Line search {} : alpha = 1/{}, energy = {:6.3f}", 
                        line_search_count, (1 << line_search_count), curr_energy);
                }
            }

            CpuParallel::parallel_copy(sa_x, sa_x_iter_start);
        }
        update_velocity(substep_dt, false, lcsv::get_scene_params().damping_cloth);
    }

    // Output
    {
        CpuParallel::parallel_for(0, sa_x.size(), [&](const uint vid)
        {
            host_mesh_data->sa_x_frame_end[vid] = sa_x[vid];
            host_mesh_data->sa_v_frame_end[vid] = sa_v[vid];
        });
        lcsv::SolverInterface::physics_step_post_operation(); 
    }
}
void NewtonSolver::physics_step_newton_GPU(luisa::compute::Device& device, luisa::compute::Stream& stream)
{
    
}

} // namespace lcsv