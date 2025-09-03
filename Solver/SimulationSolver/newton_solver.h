#pragma once

#include "Core/float_n.h"
#include "SimulationCore/solver_interface.h"
#include "LinearSolver/precond_cg.h"
#include "luisa/runtime/buffer.h"
#include "luisa/runtime/device.h"
#include "luisa/runtime/stream.h"

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
    
private:
    // Host functions
    void host_predict_position();
    void host_update_velocity();
    void host_evaluate_inertia() ;
    void host_evaluate_ground_collision();
    void host_evaluate_dirichlet();
    void host_reset_off_diag();
    void host_reset_cgB_cgX_diagA();
    void host_evaluete_spring();
    void host_line_search(luisa::compute::Stream& stream);
    
    // Device functions
    void device_broadphase_ccd(luisa::compute::Stream& stream);
    void device_broadphase_dcd(luisa::compute::Stream& stream);
    void device_narrowphase_ccd(luisa::compute::Stream& stream);
    void device_narrowphase_dcd(luisa::compute::Stream& stream);
    void device_update_contact_list(luisa::compute::Stream& stream) ;
    void device_ccd_line_search(luisa::compute::Stream& stream);
    float device_compute_contact_energy(luisa::compute::Stream& stream, const luisa::compute::Buffer<float3>& curr_x);
    // void device_line_search(luisa::compute::Stream& stream);

private:
    void collision_detection(luisa::compute::Stream& stream);
    void predict_position(luisa::compute::Stream& stream);
    void update_velocity(luisa::compute::Stream& stream);

private:
    template<typename... Args>
    using Shader = luisa::compute::Shader<1, Args...>;
    
    luisa::compute::Shader<1, luisa::compute::BufferView<float3>> fn_reset_vector;
    luisa::compute::Shader<1, luisa::compute::BufferView<float3x3>> fn_reset_float3x3;

    luisa::compute::Shader<1> fn_reset_offdiag ;
    luisa::compute::Shader<1, float> fn_predict_position ; // const Float substep_dt
    luisa::compute::Shader<1, float, bool, float> fn_update_velocity; // const Float substep_dt, const Bool fix_scene, const Float damping
    luisa::compute::Shader<1, float> fn_evaluate_inertia; // Float substep_dt
    luisa::compute::Shader<1, float, float> fn_evaluate_dirichlet; // Float substep_dt, stiffness_dirichlet
    luisa::compute::Shader<1, float, bool, float, float, float> fn_evaluate_ground_collision; // Float substep_dt
    luisa::compute::Shader<1, float> fn_evaluate_spring; // Float stiffness_stretch

    luisa::compute::Shader<1, luisa::compute::BufferView<float3>, luisa::compute::BufferView<float3>> fn_pcg_spmv_diag ;
    luisa::compute::Shader<1, luisa::compute::BufferView<float3>, luisa::compute::BufferView<float3>> fn_pcg_spmv_offdiag;

    luisa::compute::Shader<1, float> fn_apply_dx;
    luisa::compute::Shader<1, luisa::compute::BufferView<float>> fn_apply_dx_non_constant;;
    

    
};


} // namespace lcsv 