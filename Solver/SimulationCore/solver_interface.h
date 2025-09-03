#pragma once

#include <luisa/luisa-compute.h>
#include <string>
#include "CollisionDetector/lbvh.h"
#include "CollisionDetector/narrow_phase.h"
#include "Core/xbasic_types.h"
#include "LinearSolver/precond_cg.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/simulation_data.h"
#include "Utils/buffer_filler.h"
#include "Utils/device_parallel.h"
#include "luisa/runtime/buffer.h"
#include "luisa/runtime/device.h"
#include "luisa/runtime/shader.h"
#include "luisa/runtime/stream.h"

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
        CollisionData<std::vector>*              host_ccd_ptr, 
        CollisionData<luisa::compute::Buffer>*   ccd_ptr, 

        LBVH* lbvh_face_ptr,
        LBVH* lbvh_edge_ptr,
        BufferFiller* buffer_filler_ptr, 
        DeviceParallel* device_parallel_ptr,
        NarrowPhasesDetector* narrowphase_detector_ptr,
        ConjugateGradientSolver* pcg_solver_ptr
    )
    {
        // Data pointer
        host_mesh_data = host_mesh_ptr;
        host_sim_data = host_xpbd_ptr;

        mesh_data = mesh_ptr;
        sim_data = xpbd_ptr;

        host_collision_data = host_ccd_ptr;
        collision_data = ccd_ptr;

        // Tool class pointer
        mp_lbvh_face = lbvh_face_ptr;
        mp_lbvh_edge = lbvh_edge_ptr;
        mp_device_parallel = device_parallel_ptr;
        mp_buffer_filler = buffer_filler_ptr;
        mp_narrowphase_detector = narrowphase_detector_ptr;
        pcg_solver = pcg_solver_ptr;
    }
    void compile(luisa::compute::Device& device)
    {
        compile_compute_energy(device);
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
    double host_compute_elastic_energy(const std::vector<float3>& curr_x);
    double device_compute_elastic_energy(luisa::compute::Stream& stream, const luisa::compute::Buffer<float3>& curr_x);
    void compile_compute_energy(luisa::compute::Device& device);

protected:
    MeshData<std::vector>*                      host_mesh_data;
    MeshData<luisa::compute::Buffer>*           mesh_data;

    SimulationData<std::vector>*                host_sim_data;
    SimulationData<luisa::compute::Buffer>*     sim_data;

    CollisionData<std::vector>*              host_collision_data;
    CollisionData<luisa::compute::Buffer>*   collision_data;

    BufferFiller*   mp_buffer_filler;
    DeviceParallel* mp_device_parallel;
    LBVH* mp_lbvh_face;
    LBVH* mp_lbvh_edge;
    NarrowPhasesDetector* mp_narrowphase_detector;
    ConjugateGradientSolver* pcg_solver;
    // lcsv::LBVH* collision_detector_narrow_phase;

private:
    luisa::compute::Shader<1, luisa::compute::BufferView<float>> fn_reset_float;
    luisa::compute::Shader<1, 
        luisa::compute::BufferView<float3>, // sa_x
        float, // substep_dt
        float // stiffness_dirichlet
        > fn_calc_energy_inertia; 
    luisa::compute::Shader<1, 
        luisa::compute::BufferView<float3>, // sa_x
        float // stiffness_spring
        > fn_calc_energy_spring;
    luisa::compute::Shader<1, 
        luisa::compute::BufferView<float3>, // sa_x
        float, // floor_y
        bool, // use_ground_collision
        float, // stiffness
        float, // d_hat
        float // thickness
        > fn_calc_energy_ground_collision; 
    // luisa::compute::Shader<1, 
    //     luisa::compute::BufferView<float3>, 
    //     luisa::compute::BufferView<float3>, 
    //     float, 
    //     float,
    //     float
    //     > fn_compute_repulsion_energy_from_vf;
    // luisa::compute::Shader<1, 
    //     luisa::compute::BufferView<float3>, 
    //     luisa::compute::BufferView<float3>, 
    //     float, 
    //     float,
    //     float
    //     >  fn_compute_repulsion_energy_from_ee;
};


}