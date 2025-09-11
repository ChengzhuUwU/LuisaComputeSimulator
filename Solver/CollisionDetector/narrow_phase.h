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


namespace lcs 
{

class NarrowPhasesDetector
{
    template<typename T>
    using Buffer = luisa::compute::Buffer<T>;
    template<typename T>
    using BufferView = luisa::compute::BufferView<T>;
    using Stream = luisa::compute::Stream;
    using Device = luisa::compute::Device;

private:
    void compile_ccd(luisa::compute::Device& device);
    void compile_dcd(luisa::compute::Device& device);
    void compile_assemble(luisa::compute::Device& device);
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
    float download_energy(Stream& stream);

    void reset_toi(Stream& stream);
    void host_reset_toi(Stream& stream);
    void reset_broadphase_count(Stream& stream);
    void reset_narrowphase_count(Stream& stream);
    float get_global_toi(Stream& stream);
    void download_broadphase_collision_count(Stream& stream);
    void download_narrowphase_collision_count(Stream& stream);
    void download_narrowphase_list(Stream& stream);
    void upload_spd_narrowphase_list(Stream& stream);

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
    void vf_dcd_query_repulsion(Stream& stream, 
        const Buffer<float3>& sa_x_left, 
        const Buffer<float3>& sa_x_right, 
        const Buffer<float3>& sa_rest_x_left, 
        const Buffer<float3>& sa_rest_x_right, 
        const Buffer<float>& sa_rest_area_left, 
        const Buffer<float>& sa_rest_area_right, 
        const Buffer<uint3>& sa_faces_right,
        const float d_hat, 
        const float thickness, const float kappa);

    void ee_dcd_query_repulsion(Stream& stream, 
        const Buffer<float3>& sa_x_left, 
        const Buffer<float3>& sa_x_right, 
        const Buffer<float3>& sa_rest_x_left, 
        const Buffer<float3>& sa_rest_x_right, 
        const Buffer<float>& sa_rest_area_left, 
        const Buffer<float>& sa_rest_area_right, 
        const Buffer<uint2>& sa_edges_left,
        const Buffer<uint2>& sa_edges_right,
        const float d_hat, 
        const float thickness, const float kappa);

    void compute_repulsion_gradiant_hessian_and_assemble(Stream& stream, 
        const Buffer<float3>& sa_x_left, 
        const Buffer<float3>& sa_x_right, 
        const float d_hat,
        const float thickness,
        Buffer<float3>& sa_cgB, Buffer<float3x3>& sa_cgA_diag);

    void host_spmv_repulsion(Stream& stream, const std::vector<float3>& input_array, std::vector<float3>& output_array);
    void device_spmv(Stream& stream, const Buffer<float3>& input_array, Buffer<float3>& output_array);

public:
    // Compute barrier energy
    void compute_penalty_energy_from_vf(Stream& stream, 
        const Buffer<float3>& sa_x_left, 
        const Buffer<float3>& sa_x_right, 
        const Buffer<float3>& sa_rest_x_left, 
        const Buffer<float3>& sa_rest_x_right, 
        const Buffer<float>& sa_rest_area_left, 
        const Buffer<float>& sa_rest_area_right, 
        const Buffer<uint3>& sa_faces_right,
        const float d_hat,
        const float thickness,
        const float kappa);

    void compute_penalty_energy_from_ee(Stream& stream, 
        const Buffer<float3>& sa_x_left, 
        const Buffer<float3>& sa_x_right, 
        const Buffer<float3>& sa_rest_x_left, 
        const Buffer<float3>& sa_rest_x_right, 
        const Buffer<float>& sa_rest_area_left, 
        const Buffer<float>& sa_rest_area_right, 
        const Buffer<uint2>& sa_edges_left,
        const Buffer<uint2>& sa_edges_right,
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
        luisa::compute::BufferView<float3>,
        luisa::compute::BufferView<float3>,
        luisa::compute::BufferView<float>,
        luisa::compute::BufferView<float>,
        luisa::compute::BufferView<uint3>, 
        float, 
        float,
        float
        > fn_narrow_phase_vf_dcd_query_penalty;
    
    luisa::compute::Shader<1, 
        luisa::compute::BufferView<float3>,
        luisa::compute::BufferView<float3>,
        luisa::compute::BufferView<float3>,
        luisa::compute::BufferView<float3>,
        luisa::compute::BufferView<float>,
        luisa::compute::BufferView<float>,
        luisa::compute::BufferView<uint2>,
        luisa::compute::BufferView<uint2>, 
        float, 
        float,
        float
        > fn_narrow_phase_ee_dcd_query_penalty;

    luisa::compute::Shader<1, 
        luisa::compute::BufferView<float3>, 
        luisa::compute::BufferView<float3>, 
        float, 
        float,
        float
        > fn_compute_repulsion_energy_from_vf;

    luisa::compute::Shader<1, 
        luisa::compute::BufferView<float3>, 
        luisa::compute::BufferView<float3>, 
        float, 
        float,
        float
        >  fn_compute_repulsion_energy_from_ee;

    // Assemble
    luisa::compute::Shader<1, BufferView<float3>, BufferView<float3>, float, float, BufferView<float3>, BufferView<float3x3>> fn_assemble_repulsion_hessian_gradient_vf;
    luisa::compute::Shader<1, BufferView<float3>, BufferView<float3>, float, float, BufferView<float3>, BufferView<float3x3>> fn_assemble_repulsion_hessian_gradient_ee;
    
    // AtomicAdd SpMV
    luisa::compute::Shader<1, BufferView<float3>, BufferView<float3>> fn_atomic_add_spmv_vf;
    luisa::compute::Shader<1, BufferView<float3>, BufferView<float3>> fn_atomic_add_spmv_ee;
    
};


// class AccdDetector
// {
// private:
//     CollisionDataCCD<luisa::compute::Buffer>* ccd_data;
// };


}