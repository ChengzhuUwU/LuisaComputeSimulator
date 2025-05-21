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
            sa_x = xpbd_data->sa_x.view(),
            sa_v = xpbd_data->sa_v.view(),
            sa_x_start = xpbd_data->sa_x_start.view(),
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
            sa_x = xpbd_data->sa_x.view(),
            sa_v = xpbd_data->sa_v.view(),
            sa_iter_start_position = xpbd_data->sa_x_start.view(),
            sa_iter_position = xpbd_data->sa_x.view(),
            sa_velocity_start = xpbd_data->sa_v_start.view(),
            sa_vert_velocity = xpbd_data->sa_v.view(),
            sa_x_start = xpbd_data->sa_x_start.view()
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

    static std::vector<float3> sa_x_tilde;
    static std::vector<float3> sa_x;
    static std::vector<float3> sa_v;
    static std::vector<float3> sa_v_start;
    static std::vector<float3> sa_x_start;
    
    static std::vector<float3> sa_dx;
    static std::vector<float3> sa_b;
    static std::vector<float3x3> sa_diagA;
    static std::vector<float3x3> sa_offdiagA; // Row-major for simplier SpMV
 
    static std::vector<float3x3> sa_cgMinv;
    static std::vector<float3> sa_cgP;
    static std::vector<float3> sa_cgQ;
    static std::vector<float3> sa_cgR;
    static std::vector<float3> sa_cgZ;
    
 
    const uint curr_frame = get_scene_params().current_frame;
    if (sa_b.empty())
    {
        const uint num_verts = host_mesh_data->num_verts;
        const uint num_edges = host_mesh_data->num_edges;
        const uint num_faces = host_mesh_data->num_faces;

        sa_dx.resize(num_verts);
        sa_b.resize(num_verts);
        sa_diagA.resize(num_verts);
        sa_offdiagA.resize(num_edges * 2);

        sa_x_tilde.resize(num_verts);
        sa_x.resize(num_verts);
        sa_v.resize(num_verts);
        sa_v_start.resize(num_verts);
        sa_x_start.resize(num_verts);

        sa_cgMinv.resize(num_verts);
        sa_cgP.resize(num_verts);
        sa_cgQ.resize(num_verts);
        sa_cgR.resize(num_verts);
        sa_cgZ.resize(num_verts);
        
    }

    constexpr bool use_eigen = true;
    using EigenFloat3x3 = Eigen::Matrix<float, 3, 3>;
    using EigenFloat3   = Eigen::Matrix<float, 3, 1>;
    auto float3x3_to_eigen3x3 = [](const float3x3& input)
    {
        EigenFloat3x3 mat;
        mat << 
            input[0][0], input[1][0], input[2][0], 
            input[0][1], input[1][1], input[2][1], 
            input[0][2], input[1][2], input[2][2];
        return mat;
    };
    auto eigen3x3_to_float3x3 = [](const EigenFloat3x3& input)
    {
        return luisa::make_float3x3(
            input(0, 0), input(1, 0), input(2, 0), 
            input(0, 1), input(1, 1), input(2, 1), 
            input(0, 2), input(1, 2), input(2, 2)
        );
    };
    auto float3_to_eigen3 = [](const float3& input)
    {
        EigenFloat3 vec;
        vec << input[0], input[1], input[2];
        return vec;
    };
    auto eigen3_to_float3 = [](const EigenFloat3& input)
    {
        return luisa::make_float3(
            input(0, 0), input(1, 0), input(2, 0)
        );
    };
    static Eigen::SparseMatrix<float> cgA;
    static Eigen::VectorXf cg_b_vec;
    static Eigen::VectorXf cg_x_vec;
    static Eigen::SparseMatrix<float> springA(mesh_data->num_verts * 3, mesh_data->num_verts * 3);
    if constexpr (use_eigen) 
    {
        if (cg_b_vec.size() == 0)
        {
            cg_b_vec.resize(mesh_data->num_verts * 3); cg_b_vec.setOnes();
            cg_x_vec.resize(mesh_data->num_verts * 3);
            cgA.resize(mesh_data->num_verts * 3, mesh_data->num_verts * 3);
            cgA.reserve(mesh_data->num_verts * 9 + mesh_data->num_edges * 9 * 2);
            springA.resize(mesh_data->num_verts * 3, mesh_data->num_verts * 3);
            springA.reserve(mesh_data->num_edges * 9 * 4);
        }
    }

    CpuParallel::parallel_for(0, sa_x.size(), [&](const uint vid)
    {
        sa_x[vid] = host_mesh_data->sa_x_frame_start[vid];
        sa_v[vid] = host_mesh_data->sa_v_frame_start[vid];
        sa_x_start[vid] = host_mesh_data->sa_x_frame_start[vid];
        sa_v_start[vid] = host_mesh_data->sa_v_frame_start[vid];
    });

    // Predict Position
    auto predict_position = [&](const float substep_dt)
    {
        auto* sa_is_fixed = host_mesh_data->sa_is_fixed.data();

        CpuParallel::parallel_for(0, host_mesh_data->num_verts, [&](const uint vid)
        {   
            const float3 gravity(0, -9.8f, 0);
            float3 x_prev = sa_x_start[vid];
            float3 v_prev = sa_v[vid];
            float3 outer_acceleration = gravity;
            float3 v_pred = v_prev + substep_dt * outer_acceleration;
            if (sa_is_fixed[vid] != 0) { outer_acceleration = Zero3; v_pred = Zero3; };
            const float3 x_pred = x_prev + substep_dt * v_pred;
            sa_x[vid] = x_pred;
            sa_x_tilde[vid] = x_pred;
        });
    };

    // Init vert
    auto init_vert = [&](const float substep_dt)
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
            float3 x_0 = sa_x_start[vid];
            float3 v_0 = sa_v[vid];

            auto is_fixed = sa_is_fixed[vid];
            float mass = sa_vert_mass[vid];
            float3x3 mat = luisa::make_float3x3(1.0f) * mass * h_2_inv;
            float3 outer_acceleration = gravity;
            float3 outer_force = mass * outer_acceleration;

            if (is_fixed != 0)
            {
                mat = mat + luisa::make_float3x3(1.0f) * float(1E9);
            };

            float3 gradient = -mass * h_2_inv * (x_k - x_0 - v_0 * h) + outer_force;
            float3 dx_0 = substep_dt * v_0;

            if constexpr (use_eigen) 
            {
                const uint prefix_triplets_A = 9 * vid;
                const uint prefix_triplets_b = 3 * vid;

                // Assemble diagonal 3x3 block for vertex vid
                // cgA.block<3, 3>(3 * vid, 3 * vid) = float3x3_to_eigen3x3(mat).sparseView();
                for (int ii = 0; ii < 3; ++ii)
                {
                    for (int jj = 0; jj < 3; ++jj)
                    {
                        triplets_A[prefix_triplets_A + ii * 3 + jj] = Eigen::Triplet<float>(3 * vid + ii, 3 * vid + jj, mat[jj][ii]); // mat[i][j] is ok???
                    }
                }
                // Assemble gradient
                cg_b_vec.segment<3>(prefix_triplets_b) = float3_to_eigen3(gradient);
            }
            // else 
            {  
                sa_dx[vid] = dx_0;
                sa_b[vid] = gradient;
                sa_diagA[vid] = mat;
            }
        });

        if constexpr (use_eigen) 
        {
            cgA.setFromTriplets(triplets_A.begin(), triplets_A.end());

            // luisa::log_info("nnz of A = {} (desire for {})", cgA.nonZeros(), 9 * num_verts);
            // luisa::log_info("Luisa = {} , and {}", sa_diagA[0], sa_b[0]);
            // auto mat = cgA.block<3, 3>(0, 0, 3, 3).toDense();
            // auto vec = cg_b_vec.segment<3>(0);
            // std::cout << "Eigen = \n" << mat << " , and \n" << vec << std::endl;
        }
    };

    // Evaluate Spring
    auto evaluete_spring = [&](const float stiffness_stretch)
    {
        auto* sa_iter_position = sa_x.data();
        auto* sa_edges = host_mesh_data->sa_edges.data();
        auto* sa_rest_length = host_mesh_data->sa_edges_rest_state_length.data();

        const uint num_verts = host_mesh_data->num_verts;
        const uint num_edges = host_mesh_data->num_edges;

        static std::vector<Eigen::Triplet<float>> triplets_A;
        if (triplets_A.empty())
        {
            triplets_A.resize(num_edges * 9 * 4);
        }

        for (uint cluster_idx = 0; cluster_idx < host_xpbd_data->num_clusters_stretch_mass_spring; cluster_idx++) 
        {
            const uint curr_prefix = host_xpbd_data->prefix_stretch_mass_spring[cluster_idx];
            const uint next_prefix = host_xpbd_data->prefix_stretch_mass_spring[cluster_idx + 1];
            const uint num_elements_clustered = next_prefix - curr_prefix;

            // CpuParallel::single_thread_for(0, mesh_data->num_edges, [&](const uint eid)
            CpuParallel::parallel_for(0, num_elements_clustered, [&](const uint index)
            {
                const uint eid = curr_prefix + index;
                uint2 edge = sa_edges[eid];

                float3 vert_pos[2] = {
                    sa_iter_position[edge[0]],
                    sa_iter_position[edge[1]],
                };
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
                            triplets_A[prefix_triplets_A + 4 * (ii * 3 + jj) + 0] = Eigen::Triplet<float>(3 * edge[0] + ii, 3 * edge[0] + jj, He[ii][jj]);
                            triplets_A[prefix_triplets_A + 4 * (ii * 3 + jj) + 1] = Eigen::Triplet<float>(3 * edge[1] + ii, 3 * edge[1] + jj, He[ii][jj]);

                            // Off-diagonal blocks
                            triplets_A[prefix_triplets_A + 4 * (ii * 3 + jj) + 2] = Eigen::Triplet<float>(3 * edge[0] + ii, 3 * edge[1] + jj, -He[ii][jj]);
                            triplets_A[prefix_triplets_A + 4 * (ii * 3 + jj) + 3] = Eigen::Triplet<float>(3 * edge[1] + ii, 3 * edge[0] + jj, -He[ii][jj]);
                        }
                    }
                    // Assemble force to gradient
                    cg_b_vec.segment<3>(3 * edge[0]) += float3_to_eigen3(force[0]);
                    cg_b_vec.segment<3>(3 * edge[1]) += float3_to_eigen3(force[1]);
                }
                
                {
                    sa_b[edge[0]] = sa_b[edge[0]] + force[0];
                    sa_b[edge[1]] = sa_b[edge[1]] + force[1];
                    sa_diagA[edge[0]] = sa_diagA[edge[0]] + He;
                    sa_diagA[edge[1]] = sa_diagA[edge[1]] + He;
                    sa_offdiagA[eid * 2 + 0] = -1.0f * He;
                    sa_offdiagA[eid * 2 + 1] = -1.0f * He;
                }
            }, 32);
        }

        if constexpr (use_eigen) 
        {
            // Add spring contributions to cgA and cg_b_vec
            springA.setFromTriplets(triplets_A.begin(), triplets_A.end());
            cgA += springA;

            // luisa::log_info("nnz of A = {} (desire for {})", cgA.nonZeros(), mesh_data->num_verts * 9 + mesh_data->num_edges * 9 * 2);
            // luisa::log_info("Luisa = {} , and {}", sa_diagA[0], sa_b[0]);
            // auto mat = cgA.block<3, 3>(0, 0, 3, 3).toDense();
            // auto vec = cg_b_vec.segment<3>(0);
            // std::cout << "Eigen = \n" << mat << " , and \n" << vec << std::endl;

        }
    };

    // Solve
    auto simple_solve = [&]()
    {
        CpuParallel::parallel_for(0, mesh_data->num_verts, [&](const uint vid)
        {
            float3x3 hessian = sa_diagA[vid];
            float3 f = sa_b[vid];

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

    auto eigen_iter_solve = [&]()
    {
        // Solve cgA * dx = cg_b_vec for dx using Conjugate Gradient
        Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower> solver;
        // solver.setMaxIterations(128);
        solver.setTolerance(1e-2f);
        solver.compute(cgA);
        if (solver.info() != Eigen::Success) 
        {
            luisa::log_error("Eigen: Decomposition failed!");
            return;
        }
        // Eigen::VectorXf dx = solver.solveWithGuess(cg_b_vec, cg_x_vec);
        Eigen::VectorXf dx = solver.solve(cg_b_vec); 
        if (solver.info() != Eigen::Success) 
        {
            luisa::log_error("Eigen: Solve failed in {} iterations", solver.iterations());
            // std::cerr << solver.error() << std::endl;
            // return;
        }
        else 
        {
            luisa::log_info("Eigen CG: nonZero = {}, lenColumn = {}, Iterations = {}, Estimated Error = {}", 
                cgA.nonZeros(), cg_b_vec.size(), solver.iterations(), solver.error());
        }
        // Update sa_x
        CpuParallel::parallel_for(0, mesh_data->num_verts, [&](const uint vid)
        {
            sa_x[vid].x += dx[3 * vid + 0];
            sa_x[vid].y += dx[3 * vid + 1];
            sa_x[vid].z += dx[3 * vid + 2];
        });
    };
    auto eigen_decompose_solve = [&]()
    {
        // Solve cgA * dx = cg_b_vec for dx using SimplicialLDLT decomposition
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
        solver.compute(cgA);
        if (solver.info() != Eigen::Success)
        {
            luisa::log_error("Eigen: SimplicialLDLT decomposition failed!");
            return;
        }
        Eigen::VectorXf dx = solver.solve(cg_b_vec);
        if (solver.info() != Eigen::Success)
        {
            luisa::log_error("Eigen: SimplicialLDLT solve failed!");
            return;
        }
        else
        {
            luisa::log_info("Eigen SimplicialLDLT: nonZero = {}, lenColumn = {}", 
                cgA.nonZeros(), cg_b_vec.size());
        }
        // Update sa_x
        CpuParallel::parallel_for(0, mesh_data->num_verts, [&](const uint vid)
        {
            sa_x[vid].x += dx[3 * vid + 0];
            sa_x[vid].y += dx[3 * vid + 1];
            sa_x[vid].z += dx[3 * vid + 2];
        });
    };

    auto linear_solver_interface = [&]()
    {
        if constexpr (use_eigen) 
        {
            eigen_iter_solve();
            // eigen_decompose_solve();
        } 
        else 
        {
            simple_solve();
        }
    };

    // Update Velocity
    auto update_velocity = [&](const float substep_dt, const bool fix_scene, const float damping)
    {
        CpuParallel::parallel_for(0, host_mesh_data->num_verts, [&](const uint vid)
        {   
            float3 x_k_init = sa_x_start[vid];
            float3 x_k = sa_x[vid];

            float3 dx = x_k - x_k_init;
            float3 vel = dx / substep_dt;

            if (fix_scene) 
            {
                dx = Zero3;
                vel = Zero3;
                sa_x[vid] = sa_x_start[vid];
                return;
            };

            vel *= exp(-damping * substep_dt);

            sa_v[vid] = vel;
            sa_v_start[vid] = vel;
            sa_x_start[vid] = x_k;
        });
    };

    const float substep_dt = lcsv::get_scene_params().get_substep_dt();

    for (uint substep = 0; substep < get_scene_params().num_substep; substep++)
    {
        predict_position(substep_dt);
        for (uint iter = 0; iter < get_scene_params().nonlinear_iter_count; iter++)
        {
            init_vert(substep_dt);
            evaluete_spring(1e4);
            linear_solver_interface();
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