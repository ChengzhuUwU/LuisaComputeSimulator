#pragma once

#include <luisa/luisa-compute.h>
#include <string>
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
        MeshData<std::vector>* host_mesh_ptr, 
        MeshData<luisa::compute::Buffer>* mesh_ptr, 
        XpbdData<std::vector>* host_xpbd_ptr, 
        XpbdData<luisa::compute::Buffer>* xpbd_ptr, 
        lcsv::BufferFiller* buffer_filler_ptr, 
        lcsv::DeviceParallel* device_parallel_ptr)
    {
        // Host data pointer
        host_mesh_data = host_mesh_ptr;
        host_xpbd_data = host_xpbd_ptr;

        // Device data pointer
        mesh_data = mesh_ptr;
        xpbd_data = xpbd_ptr;

        // Tool class pointer
        mp_device_parallel = device_parallel_ptr;
        mp_buffer_filler = buffer_filler_ptr;
    }

public:
    void physics_step_prev_operation();
    void physics_step_post_operation();
    void restart_system();
    void save_current_frame_state();
    void load_saved_state();
    void save_mesh_to_obj(const uint frame, const std::string& addition_str = "");

protected:
    MeshData<std::vector>* host_mesh_data;
    XpbdData<std::vector>* host_xpbd_data;
    
    MeshData<luisa::compute::Buffer>* mesh_data;
    XpbdData<luisa::compute::Buffer>* xpbd_data;

    lcsv::BufferFiller*   mp_buffer_filler;
    lcsv::DeviceParallel* mp_device_parallel;
};


}