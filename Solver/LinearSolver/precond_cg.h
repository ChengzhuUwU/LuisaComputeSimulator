#pragma once

#include "SimulationCore/base_mesh.h"
#include "SimulationCore/simulation_data.h"
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Utils/async_compiler.h>

namespace lcs
{

// struct PcgData
// {
// };

class ConjugateGradientSolver
{

  public:
    constexpr static bool use_eigen          = false;
    constexpr static bool use_upper_triangle = false;

  public:
    void set_data(MeshData<std::vector>*                  host_mesh_data,
                  MeshData<luisa::compute::Buffer>*       mesh_data,
                  SimulationData<std::vector>*            host_sim_data,
                  SimulationData<luisa::compute::Buffer>* sim_data)
    {
        this->host_mesh_data = host_mesh_data;
        this->mesh_data      = mesh_data;
        this->host_sim_data  = host_sim_data;
        this->sim_data       = sim_data;
    }
    void compile(AsyncCompiler& compiler);

  public:
    void host_solve(luisa::compute::Stream&                                               stream,
                    std::function<void(const std::vector<float3>&, std::vector<float3>&)> func_spmv,
                    std::function<double(const std::vector<float3>&)> func_compute_energy);
    void device_solve(luisa::compute::Stream& stream,
                      std::function<void(const luisa::compute::Buffer<float3>&, luisa::compute::Buffer<float3>&)> func_spmv,
                      std::function<double(const luisa::compute::Buffer<float3>&)> func_compute_energy);
    void eigen_solve(const Eigen::SparseMatrix<float>&                 eigen_cgA,
                     Eigen::VectorXf&                                  eigen_cgX,
                     const Eigen::VectorXf&                            eigen_cgB,
                     std::function<double(const std::vector<float3>&)> func_compute_energy);

  private:
    luisa::compute::Shader<1, luisa::compute::BufferView<float3>> fn_reset_float3;
    luisa::compute::Shader<1, luisa::compute::BufferView<float>>  fn_reset_float;
    luisa::compute::Shader<1, luisa::compute::BufferView<uint>>   fn_reset_uint;

    luisa::compute::Shader<1> fn_pcg_init;
    luisa::compute::Shader<1> fn_pcg_init_second_pass;

    luisa::compute::Shader<1> fn_dot_pq;
    luisa::compute::Shader<1> fn_dot_pq_second_pass;
    luisa::compute::Shader<1> fn_pcg_update_p;
    luisa::compute::Shader<1> fn_pcg_step;

    luisa::compute::Shader<1> fn_pcg_make_preconditioner;
    luisa::compute::Shader<1> fn_pcg_apply_preconditioner;
    luisa::compute::Shader<1> fn_pcg_apply_preconditioner_second_pass;

  private:
    MeshData<std::vector>*                  host_mesh_data;
    MeshData<luisa::compute::Buffer>*       mesh_data;
    SimulationData<std::vector>*            host_sim_data;
    SimulationData<luisa::compute::Buffer>* sim_data;
};


}  // namespace lcs