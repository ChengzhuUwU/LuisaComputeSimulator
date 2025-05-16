#pragma once

#include "Core/float_n.h"
#include "SimulationCore/solver_interface.h"

namespace lcsv
{

class DescentSolverCPU : public lcsv::SolverInterface
{

template<typename T>
using Buffer = luisa::compute::Buffer<T>;


public:
    DescentSolverCPU() : lcsv::SolverInterface() {}
    ~DescentSolverCPU() {}

public:    
    void physics_step_vbd(luisa::compute::Device& device, luisa::compute::Stream& stream);
    void physics_step_xpbd(luisa::compute::Device& device, luisa::compute::Stream& stream);
    // void compute_energy(const Buffer<float3>& curr_cloth_position);
    void compile(luisa::compute::Device& device);

private:
    void collision_detection(luisa::compute::Stream& stream);
    void predict_position(luisa::compute::Stream& stream);
    void update_velocity(luisa::compute::Stream& stream);
    void reset_constrains(luisa::compute::Stream& stream);
    void reset_collision_constrains(luisa::compute::Stream& stream);

private:
    // Buffer<float4x3>& get_Hf();
    // void solve_constraints_XPBD();
    // void solve_constraint_stretch_spring(Buffer<float3>& curr_cloth_position, const uint cluster_idx);
    // void solve_constraint_bending(Buffer<float3>& curr_cloth_position, const uint cluster_idx);

private:
    void solve_constraints_VBD(luisa::compute::Stream& stream);
    void vbd_evaluate_inertia(luisa::compute::Stream& stream, Buffer<lcsv::float3>& curr_cloth_position, const uint cluster_idx);
    void vbd_evaluate_stretch_spring(luisa::compute::Stream& stream, Buffer<lcsv::float3>& curr_cloth_position, const uint cluster_idx);
    void vbd_evaluate_bending(luisa::compute::Stream& stream, Buffer<lcsv::float3>& curr_cloth_position, const uint cluster_idx);
    void vbd_step(luisa::compute::Stream& stream, Buffer<lcsv::float3>& curr_cloth_position, const uint cluster_idx);

private:
    using Shader = luisa::compute::Shader<1>;
    Shader fn_predict_position;
    Shader fn_update_velocity;
    
    Shader fn_evaluate_inertia;
    Shader fn_evaluate_stretch_spring;
    Shader fn_evaluate_bending;
    Shader fn_step;
};


}