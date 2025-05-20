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
    void physics_step_newton_GPU(luisa::compute::Device& device, luisa::compute::Stream& stream);
    void physics_step_newton_CPU(luisa::compute::Device& device, luisa::compute::Stream& stream);
    void compile(luisa::compute::Device& device);

private:
    void compile_force(luisa::compute::Device& device);
    void compile_cg(luisa::compute::Device& device);

    private:
    void collision_detection(luisa::compute::Stream& stream);
    void predict_position(luisa::compute::Stream& stream);
    void update_velocity(luisa::compute::Stream& stream);

private:
    template<typename... Args>
    using Shader = luisa::compute::Shader<1, Args...>;
    
    Shader<float> fn_predict_position ; // const Float substep_dt
    Shader<float, bool, float> fn_update_velocity; // const Float substep_dt, const Bool fix_scene, const Float damping
    
};


} // namespace lcsv 