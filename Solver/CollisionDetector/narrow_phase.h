#pragma once

#include "CollisionDetector/lbvh.h"
#include "Core/scalar.h"
#include "SimulationCore/simulation_data.h"
#include "SimulationCore/simulation_type.h"
#include <vector>
#include <string>
#include <luisa/luisa-compute.h>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>


namespace lcsv 
{

class NarrowPhasesDetector
{
    template<typename T>
    using Buffer = luisa::compute::Buffer<T>;
    using Stream = luisa::compute::Stream;
    using Device = luisa::compute::Device;

private:
    void compile_ccd(luisa::compute::Device& device);
    void compile_dcd(luisa::compute::Device& device);
    void compile_energy(luisa::compute::Device& device);
    
public:
    void unit_test(luisa::compute::Device& device, luisa::compute::Stream& stream);
    void compile(luisa::compute::Device& device);
    void set_collision_data(
        CollisionData<std::vector>* host_ccd_ptr,
        CollisionData<luisa::compute::Buffer>* ccd_ptr
    ) 
    { 
        host_collision_data = host_ccd_ptr; 
        collision_data = ccd_ptr; 
    }

public:
    void reset_energy(Stream& stream);
    float download_energy(Stream& stream, const float kappa);

    void reset_toi(Stream& stream);
    void host_reset_toi(Stream& stream);
    void reset_broadphase_count(Stream& stream);
    void reset_narrowphase_count(Stream& stream);
    float get_global_toi(Stream& stream);
    void download_broadphase_collision_count(Stream& stream);
    void download_narrowphase_collision_count(Stream& stream);
    void download_narrowphase_list(Stream& stream);

public:
    // CCD 
    void vf_ccd_query(Stream& stream, 
        const Buffer<float3>& sa_x_begin_left, 
        const Buffer<float3>& sa_x_begin_right, 
        const Buffer<float3>& sa_x_end_left,
        const Buffer<float3>& sa_x_end_right,
        const Buffer<uint3>& sa_faces_right,
        const float d_hat, 
        const float thickness);

    void ee_ccd_query(Stream& stream, 
        const Buffer<float3>& sa_x_begin_left, 
        const Buffer<float3>& sa_x_begin_right, 
        const Buffer<float3>& sa_x_end_left,
        const Buffer<float3>& sa_x_end_right,
        const Buffer<uint2>& sa_edges_left,
        const Buffer<uint2>& sa_edges_right,
        const float d_hat, 
        const float thickness);
    
    void host_vf_ccd_query(Stream& stream, 
        const std::vector<float3>& sa_x_begin_left, 
        const std::vector<float3>& sa_x_begin_right, 
        const std::vector<float3>& sa_x_end_left,
        const std::vector<float3>& sa_x_end_right,
        const std::vector<uint3>& sa_faces_right,
        const float d_hat,
        const float thickness);

    void host_ee_ccd_query(Stream& stream, 
        const std::vector<float3>& sa_x_begin_left, 
        const std::vector<float3>& sa_x_begin_right, 
        const std::vector<float3>& sa_x_end_left,
        const std::vector<float3>& sa_x_end_right,
        const std::vector<uint2>& sa_edges_left,
        const std::vector<uint2>& sa_edges_right,
        const float d_hat,
        const float thickness);
public:
    // DCD
    void vf_dcd_query(Stream& stream, 
        const Buffer<float3>& sa_x_left, 
        const Buffer<float3>& sa_x_right, 
        const Buffer<uint3>& sa_faces_right,
        const float d_hat, 
        const float thickness, const float kappa);

    void ee_dcd_query(Stream& stream, 
        const Buffer<float3>& sa_x_left, 
        const Buffer<float3>& sa_x_right, 
        const Buffer<uint2>& sa_edges_left,
        const Buffer<uint2>& sa_edges_right,
        const float d_hat, 
        const float thickness, const float kappa);

    void host_dcd_query_libuipc(
        Eigen::SparseMatrix<float>& eigen_cgA,
        Eigen::VectorXf& eigen_cgB,
        const std::vector<float3>& sa_x_left, 
        const std::vector<float3>& sa_x_right, 
        const std::vector<float3>& sa_rest_x_left, 
        const std::vector<float3>& sa_rest_x_right, 
        const std::vector<uint3>& sa_faces_left,
        const std::vector<uint3>& sa_faces_right,
        const std::vector<uint2>& sa_edges_left,
        const std::vector<uint2>& sa_edges_right,
        const float d_hat, 
        const float thickness,
        const float kappa);

    void host_barrier_hessian_spd_projection(Stream& stream, Eigen::SparseMatrix<float>& eigen_cgA, Eigen::VectorXf& eigen_cgB);

public:
    // Compute barrier energy
    void compute_barrier_energy_from_vf(Stream& stream, 
        const Buffer<float3>& sa_x_left, 
        const Buffer<float3>& sa_x_right, 
        const Buffer<uint3>& sa_faces_right,
        const float d_hat,
        const float thickness,
        const float kappa);

    void compute_barrier_energy_from_ee(Stream& stream, 
        const Buffer<float3>& sa_x_left, 
        const Buffer<float3>& sa_x_right, 
        const Buffer<uint2>& sa_edges_left,
        const Buffer<uint2>& sa_edges_right,
        const float d_hat,
        const float thickness,
        const float kappa);

    double host_compute_barrier_energy_uipc(
        const std::vector<float3>& sa_x_left, 
        const std::vector<float3>& sa_x_right,
        const std::vector<float3>& sa_rest_x_left,
        const std::vector<float3>& sa_rest_x_right, 
        const std::vector<uint3>& sa_faces_left,
        const std::vector<uint3>& sa_faces_right,
        const std::vector<uint2>& sa_edge_left,
        const std::vector<uint2>& sa_edge_right,
        const float d_hat,
        const float thickness,
        const float kappa);
    

public:
    
    
    

public:
    CollisionData<luisa::compute::Buffer>* collision_data;
    CollisionData<std::vector>* host_collision_data;

private:
    luisa::compute::Shader<1, 
        luisa::compute::BufferView<float3>,
        luisa::compute::BufferView<float3>,
        luisa::compute::BufferView<float3>,
        luisa::compute::BufferView<float3>,
        luisa::compute::BufferView<uint3>, float, float> fn_narrow_phase_vf_ccd_query;

    luisa::compute::Shader<1, 
        luisa::compute::BufferView<float3>,
        luisa::compute::BufferView<float3>,
        luisa::compute::BufferView<float3>,
        luisa::compute::BufferView<float3>,
        luisa::compute::BufferView<uint2>,
        luisa::compute::BufferView<uint2>, float, float> fn_narrow_phase_ee_ccd_query ;

    luisa::compute::Shader<1, luisa::compute::BufferView<float>> fn_reset_toi;
    luisa::compute::Shader<1, luisa::compute::BufferView<float>> fn_reset_energy;
    luisa::compute::Shader<1, luisa::compute::BufferView<uint>> fn_reset_uint;
    luisa::compute::Shader<1, luisa::compute::BufferView<float>> fn_reset_float;


    luisa::compute::Shader<1, 
        luisa::compute::BufferView<float3>,
        luisa::compute::BufferView<float3>,
        luisa::compute::BufferView<uint3>, 
        float, 
        float,
        float
        > fn_narrow_phase_vf_dcd_query;
    
    luisa::compute::Shader<1, 
        luisa::compute::BufferView<float3>,
        luisa::compute::BufferView<float3>,
        luisa::compute::BufferView<uint2>,
        luisa::compute::BufferView<uint2>, 
        float, 
        float,
        float
        > fn_narrow_phase_ee_dcd_query;

    luisa::compute::Shader<1, 
        luisa::compute::BufferView<float3>, 
        luisa::compute::BufferView<float3>, 
        luisa::compute::BufferView<uint3>, 
        float, 
        float,
        float
        > fn_narrow_phase_vf_dcd_for_barrier_energy;

    luisa::compute::Shader<1, 
        luisa::compute::BufferView<float3>, 
        luisa::compute::BufferView<float3>, 
        luisa::compute::BufferView<uint2>, 
        luisa::compute::BufferView<uint2>, 
        float, 
        float,
        float
        >  fn_narrow_phase_ee_dcd_for_barrier_energy;
};


// class AccdDetector
// {
// private:
//     CollisionDataCCD<luisa::compute::Buffer>* ccd_data;
// };


}