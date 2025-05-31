#pragma once

#include <luisa/luisa-compute.h>
#include <string>
#include "CollisionDetector/lbvh.h"
#include "Core/xbasic_types.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/simulation_data.h"
#include "Utils/buffer_filler.h"
#include "Utils/device_parallel.h"

namespace lcsv 
{

class SolverInterface
{

// public:
//     template<typename T>
//     using Buffer = luisa::compute::Buffer<T>;

public:
    SolverInterface() {}
    ~SolverInterface() {}  


    void set_data_pointer(
        MeshData<std::vector>*                      host_mesh_ptr, 
        MeshData<luisa::compute::Buffer>*           mesh_ptr, 
        SimulationData<std::vector>*                host_xpbd_ptr, 
        SimulationData<luisa::compute::Buffer>*     xpbd_ptr, 
        CollisionDataCCD<std::vector>*              host_ccd_ptr, 
        CollisionDataCCD<luisa::compute::Buffer>*   ccd_ptr, 

        LBVH* lbvh_face_ptr,
        LBVH* lbvh_edge_ptr,
        BufferFiller* buffer_filler_ptr, 
        DeviceParallel* device_parallel_ptr
    )
    {
        // Data pointer
        host_mesh_data = host_mesh_ptr;
        host_sim_data = host_xpbd_ptr;

        mesh_data = mesh_ptr;
        sim_data = xpbd_ptr;

        host_ccd_data = host_ccd_ptr;
        ccd_data = ccd_ptr;

        // Tool class pointer
        mp_lbvh_face = lbvh_face_ptr;
        mp_lbvh_edge = lbvh_edge_ptr;
        mp_device_parallel = device_parallel_ptr;
        mp_buffer_filler = buffer_filler_ptr;
    }

public:
    void physics_step_prev_operation();
    void physics_step_post_operation();
    void restart_system();
    void save_current_frame_state();
    void save_current_frame_state_to_host(const uint frame, const std::string& addition_str);
    void load_saved_state();
    void load_saved_state_from_host(const uint frame, const std::string& addition_str);
    void save_mesh_to_obj(const uint frame, const std::string& addition_str = "");
    double host_compute_energy(const std::vector<float3>& curr_x, const std::vector<float3>& curr_x_tilde);

protected:
    MeshData<std::vector>*                      host_mesh_data;
    MeshData<luisa::compute::Buffer>*           mesh_data;

    SimulationData<std::vector>*                host_sim_data;
    SimulationData<luisa::compute::Buffer>*     sim_data;

    CollisionDataCCD<std::vector>*              host_ccd_data;
    CollisionDataCCD<luisa::compute::Buffer>*   ccd_data;

    BufferFiller*   mp_buffer_filler;
    DeviceParallel* mp_device_parallel;
    LBVH* mp_lbvh_face;
    LBVH* mp_lbvh_edge;
    // lcsv::LBVH* collision_detector_narrow_phase;
};


}