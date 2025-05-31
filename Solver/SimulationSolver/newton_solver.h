#pragma once

#include "Core/float_n.h"
#include "SimulationCore/solver_interface.h"

namespace lcsv
{

class NewtonSolver : public lcsv::SolverInterface
{

template<typename T>
using Buffer = luisa::compute::Buffer<T>;


public:
    NewtonSolver() : lcsv::SolverInterface() {}
    ~NewtonSolver() {}

public:    
    void physics_step_GPU(luisa::compute::Device& device, luisa::compute::Stream& stream);
    void physics_step_CPU(luisa::compute::Device& device, luisa::compute::Stream& stream);
    void compile(luisa::compute::Device& device);

private:
    // void compile_force(luisa::compute::Device& device);
    // void compile_cg(luisa::compute::Device& device);

private:
    void collision_detection(luisa::compute::Stream& stream);
    void predict_position(luisa::compute::Stream& stream);
    void update_velocity(luisa::compute::Stream& stream);
    float device_compute_energy(luisa::compute::Stream& stream, const luisa::compute::BufferView<float3>& curr_x);

private:
    template<typename... Args>
    using Shader = luisa::compute::Shader<1, Args...>;
    
    luisa::compute::Shader<1, luisa::compute::BufferView<float3>, float3> fn_reset_vector;

    luisa::compute::Shader<1> fn_reset_offdiag ;
    luisa::compute::Shader<1, float> fn_predict_position ; // const Float substep_dt
    luisa::compute::Shader<1, float, bool, float> fn_update_velocity; // const Float substep_dt, const Bool fix_scene, const Float damping
    luisa::compute::Shader<1, float> fn_evaluate_inertia; // Float substep_dt
    luisa::compute::Shader<1, float, uint> fn_evaluate_spring; // Float stiffness_stretch, Uint cluster_idx

    luisa::compute::Shader<1, luisa::compute::BufferView<float3>, float> fn_calc_energy_inertia; // Var<BufferView<float3>> sa_x, Float substep_dt
    luisa::compute::Shader<1, luisa::compute::BufferView<float3>, float> fn_calc_energy_spring; // Var<BufferView<float3>> sa_x, Float stiffness_spring

    luisa::compute::Shader<1, float> fn_apply_dx;

    luisa::compute::Shader<1> fn_pcg_init;
    luisa::compute::Shader<1> fn_pcg_init_second_pass;
    luisa::compute::Shader<1, luisa::compute::BufferView<float3>, luisa::compute::BufferView<float3>> fn_pcg_spmv_diag ;
    luisa::compute::Shader<1, luisa::compute::BufferView<float3>, luisa::compute::BufferView<float3>, uint> fn_pcg_spmv_offdiag;
    luisa::compute::Shader<1> fn_dot_pq;
    luisa::compute::Shader<1> fn_dot_pq_second_pass;
    luisa::compute::Shader<1> fn_pcg_update_p;
    luisa::compute::Shader<1> fn_pcg_step;

    luisa::compute::Shader<1> fn_pcg_make_preconditioner;
    luisa::compute::Shader<1> fn_pcg_apply_preconditioner;
    luisa::compute::Shader<1> fn_pcg_apply_preconditioner_second_pass;
    

    
};


} // namespace lcsv 