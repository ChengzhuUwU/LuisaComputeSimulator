#pragma once

#include "Core/float_n.h"
#include "SimulationSolver/solver_interface.h"
#include "LinearSolver/precond_cg.h"
#include "luisa/runtime/buffer.h"
#include "luisa/runtime/device.h"
#include "luisa/runtime/stream.h"
#include <vector>

namespace lcs
{

class NewtonSolver : public lcs::SolverInterface
{

    template <typename T>
    using Buffer = luisa::compute::Buffer<T>;


  public:
    NewtonSolver()
        : lcs::SolverInterface()
    {
    }
    ~NewtonSolver() {}

  public:
    void physics_step_GPU(luisa::compute::Device& device, luisa::compute::Stream& stream);
    void physics_step_CPU(luisa::compute::Device& device, luisa::compute::Stream& stream);
    void init_solver(luisa::compute::Device&                   device,
                     luisa::compute::Stream&                   stream,
                     std::vector<lcs::Initializer::WorldData>& shell_list)
    {
        LUISA_INFO("Init mesh data...");
        SolverInterface::init_data(device, stream, shell_list);

        luisa::compute::Clock clk;
        {
            AsyncCompiler compiler(device);
            {
                SolverInterface::compile(compiler);
                this->compile(compiler);
            }
            compiler.wait();
        }
        LUISA_INFO("Shader compile done with time {} seconds.", clk.toc() * 1e-3);

        SolverInterface::restart_system();
        LUISA_INFO("Simulation begin...");
    }

  private:
    void compile(AsyncCompiler& compiler);
    void compile_advancing(AsyncCompiler& compiler, const luisa::compute::ShaderOption& default_option);
    void compile_assembly(AsyncCompiler& compiler, const luisa::compute::ShaderOption& default_option);
    void compile_evaluate(AsyncCompiler& compiler, const luisa::compute::ShaderOption& default_option);

  private:
    // Host functions
    void host_apply_dx(const float alpha);
    void device_apply_dx(luisa::compute::Stream& stream, const float alpha);
    void host_predict_position();
    void host_update_velocity();
    void host_evaluate_inertia();
    void host_evaluate_orthogonality();
    void host_evaluate_ground_collision();
    void host_evaluate_dirichlet();
    void host_reset_off_diag();
    void host_reset_cgB_cgX_diagA();
    void host_evaluete_stretch_spring();
    void host_evaluete_stretch_face();
    void host_evaluete_bending();
    void host_material_energy_assembly();
    void host_solve_eigen(luisa::compute::Stream& stream);
    void host_SpMV(luisa::compute::Stream& stream, const std::vector<float3>& input_array, std::vector<float3>& output_array);
    void line_search(luisa::compute::Device& device, luisa::compute::Stream& stream, bool& dirichlet_converged, bool& global_converged);

    // Device functions
    void device_broadphase_ccd(luisa::compute::Stream& stream);
    void device_broadphase_dcd(luisa::compute::Stream& stream);
    void device_narrowphase_ccd(luisa::compute::Stream& stream);
    void device_narrowphase_dcd(luisa::compute::Stream& stream);
    void device_update_contact_list(luisa::compute::Device& device, luisa::compute::Stream& stream);
    void device_post_dist_check(luisa::compute::Stream& stream);
    void device_ccd_line_search(luisa::compute::Device& device, luisa::compute::Stream& stream);
    void device_SpMV(luisa::compute::Stream&               stream,
                     const luisa::compute::Buffer<float3>& input_array,
                     luisa::compute::Buffer<float3>&       output_array);
    void device_compute_contact_energy(luisa::compute::Stream& stream, std::map<std::string, double>& energy_list);
    // void device_line_search(luisa::compute::Stream& stream);

    void host_test_dynamics(luisa::compute::Stream& stream);

  private:
    void collision_detection(luisa::compute::Stream& stream);
    void predict_position(luisa::compute::Stream& stream);
    void update_velocity(luisa::compute::Stream& stream);

  private:
    template <typename... Args>
    using Shader = luisa::compute::Shader<1, Args...>;

    luisa::compute::Shader<1, luisa::compute::BufferView<float3>>   fn_reset_vector;
    luisa::compute::Shader<1, luisa::compute::BufferView<float3x3>> fn_reset_float3x3;
    luisa::compute::Shader<1>                                       fn_reset_cgA_offdiag_triplet;

    luisa::compute::Shader<1, float, float3> fn_predict_position;  // const Float substep_dt
    luisa::compute::Shader<1, float, bool, float> fn_update_velocity;  // const Float substep_dt, const Bool fix_scene, const Float damping
    luisa::compute::Shader<1, float, float> fn_evaluate_inertia;  // Float substep_dt, Float stiffness_dirichlet
    luisa::compute::Shader<1, float, float> fn_evaluate_dirichlet;  // Float substep_dt, stiffness_dirichlet
    luisa::compute::Shader<1, float, bool, float, uint> fn_evaluate_ground_collision;
    luisa::compute::Shader<1, float, bool>              fn_gound_collision_ccd;
    luisa::compute::Shader<1, float>                    fn_evaluate_spring;  // Float stiffness_stretch
    luisa::compute::Shader<1>                           fn_evaluate_stretch_face;
    luisa::compute::Shader<1, float>                    fn_evaluate_bending;  // Float stiffness_bending

    luisa::compute::Shader<1, float, float3> fn_abd_predict_position;
    luisa::compute::Shader<1, float, bool, float> fn_abd_update_velocity;  // const Float substep_dt, const Bool fix_scene, const Float damping
    luisa::compute::Shader<1, float, float> fn_evaluate_abd_inertia;  // Float substep_dt, Float stiffness_dirichlet
    luisa::compute::Shader<1>                                 fn_evaluate_abd_orthogonality;
    luisa::compute::Shader<1, float, bool, float, uint, uint> fn_evaluate_abd_ground_collision;

    luisa::compute::Shader<1>       fn_material_energy_assembly;
    luisa::compute::Shader<1>       fn_material_energy_assembly_stretch_spring;
    luisa::compute::Shader<1>       fn_material_energy_assembly_stretch_face;
    luisa::compute::Shader<1>       fn_material_energy_assembly_bending;
    luisa::compute::Shader<1, uint> fn_material_energy_assembly_affine_body;

    luisa::compute::Shader<1, luisa::compute::Buffer<float3>, luisa::compute::Buffer<float3>> fn_pcg_spmv_diag;
    luisa::compute::Shader<1, luisa::compute::Buffer<float3>, luisa::compute::Buffer<float3>> fn_pcg_spmv_offdiag_perVert;
    luisa::compute::Shader<1, luisa::compute::Buffer<float3>, luisa::compute::Buffer<float3>> fn_pcg_spmv_offdiag_warp_rbk;
    luisa::compute::Shader<1, luisa::compute::Buffer<MatrixTriplet3x3>, luisa::compute::Buffer<float3>, luisa::compute::Buffer<float3>> fn_pcg_spmv_offdiag_block_rbk;

    luisa::compute::Shader<1, float>                                   fn_apply_dx;
    luisa::compute::Shader<1, float, uint>                             fn_apply_dx_affine_bodies;
    luisa::compute::Shader<1, float, uint>                             fn_apply_dq;
    luisa::compute::Shader<1, luisa::compute::BufferView<float>>       fn_apply_dx_non_constant;
    luisa::compute::Shader<1, luisa::compute::BufferView<float>, uint> fn_apply_dq_non_constant;
    ;
};


}  // namespace lcs