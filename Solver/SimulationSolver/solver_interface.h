#pragma once

#include <luisa/luisa-compute.h>
#include <string>
#include "CollisionDetector/lbvh.h"
#include "CollisionDetector/narrow_phase.h"
#include "Core/xbasic_types.h"
#include "Initializer/init_mesh_data.h"
#include "LinearSolver/precond_cg.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/simulation_data.h"
#include "SimulationCore/collision_data.h"
#include "Utils/buffer_filler.h"
#include "Utils/device_parallel.h"
#include "luisa/runtime/buffer.h"
#include "luisa/runtime/device.h"
#include "luisa/runtime/shader.h"
#include "luisa/runtime/stream.h"
#include <Utils/async_compiler.h>

namespace lcs
{


class SolverInterface
{
  private:
    struct SolverData
    {
        lcs::MeshData<std::vector>            host_mesh_data;
        lcs::MeshData<luisa::compute::Buffer> mesh_data;

        lcs::SimulationData<std::vector>            host_sim_data;
        lcs::SimulationData<luisa::compute::Buffer> sim_data;

        lcs::LbvhData<luisa::compute::Buffer> lbvh_data_face;
        lcs::LbvhData<luisa::compute::Buffer> lbvh_data_edge;

        lcs::CollisionData<std::vector>            host_collision_data;
        lcs::CollisionData<luisa::compute::Buffer> collision_data;
    };

    struct SolverHelper
    {
        lcs::BufferFiller   buffer_filler;
        lcs::DeviceParallel device_parallel;

        lcs::LBVH lbvh_face;
        lcs::LBVH lbvh_edge;

        lcs::NarrowPhasesDetector    narrow_phase_detector;
        lcs::ConjugateGradientSolver pcg_solver;
    };

    // public:
    //     template<typename T>
    //     using Buffer = luisa::compute::Buffer<T>;

  public:
    SolverInterface() {}
    ~SolverInterface() {}

  protected:
    void init_data(luisa::compute::Device&                   device,
                   luisa::compute::Stream&                   stream,
                   std::vector<lcs::Initializer::ShellInfo>& shell_list);
    void compile(AsyncCompiler& compiler);
    void set_data_pointer(SolverData& solver_data, SolverHelper& solver_helper);

  public:
    void restart_system();
    void save_current_frame_state_to_host(const uint frame, const std::string& addition_str);
    void load_saved_state_from_host(const uint frame, const std::string& addition_str);
    void save_mesh_to_obj(const uint frame, const std::string& addition_str = "");
    void host_compute_elastic_energy(std::map<std::string, double>& energy_list);
    void device_compute_elastic_energy(luisa::compute::Stream& stream, std::map<std::string, double>& energy_list);
    void compile_compute_energy(AsyncCompiler& compiler);

  public:
    template <typename T>
    void get_simulation_results_to_host(std::vector<std::vector<T>>& output_positions)
    {
        for (uint meshIdx = 0; meshIdx < host_mesh_data->num_meshes; meshIdx++)
        {
            CpuParallel::parallel_for(
                0,
                host_mesh_data->prefix_num_verts[meshIdx + 1] - host_mesh_data->prefix_num_verts[meshIdx],
                [&](const uint vid)
                {
                    auto pos = host_mesh_data->sa_x_frame_outer[vid + host_mesh_data->prefix_num_verts[meshIdx]];
                    output_positions[meshIdx][vid] = {pos.x, pos.y, pos.z};
                });
        }
    }
    template <typename T>
    void update_pinned_verts_information(const uint               meshIdx,
                                         const std::vector<uint>& pinned_verts,
                                         const std::vector<T>&    pinned_verts_target_position)
    {
        const uint prefix = host_mesh_data->prefix_num_verts[meshIdx];
        CpuParallel::parallel_for(0,
                                  pinned_verts.size(),
                                  [&](const uint index)
                                  {
                                      const uint local_vid = pinned_verts[index];
                                      const auto target    = pinned_verts_target_position[index];
                                      const uint vid = host_sim_data->fixed_verts_map[meshIdx][local_vid];
                                      host_sim_data->sa_target_positions[vid] =
                                          luisa::make_float3(target[0], target[1], target[2]);
                                  });
    }

  protected:
    void physics_step_prev_operation();
    void physics_step_post_operation();

  private:
    SolverData   solver_data;
    SolverHelper solver_helper;

  protected:
    MeshData<std::vector>*            host_mesh_data;
    MeshData<luisa::compute::Buffer>* mesh_data;

    SimulationData<std::vector>*            host_sim_data;
    SimulationData<luisa::compute::Buffer>* sim_data;

    lcs::LbvhData<luisa::compute::Buffer>* lbvh_data_face;
    lcs::LbvhData<luisa::compute::Buffer>* lbvh_data_edge;

    CollisionData<std::vector>*            host_collision_data;
    CollisionData<luisa::compute::Buffer>* collision_data;

    BufferFiller*            buffer_filler;
    DeviceParallel*          device_parallel;
    LBVH*                    lbvh_face;
    LBVH*                    lbvh_edge;
    NarrowPhasesDetector*    narrow_phase_detector;
    ConjugateGradientSolver* pcg_solver;
    // lcs::LBVH* collision_detector_narrow_phase;

  public:
    MeshData<std::vector>&       get_host_mesh_data() const { return *host_mesh_data; }
    SimulationData<std::vector>& get_host_sim_data() const { return *host_sim_data; }
    CollisionData<std::vector>&  get_host_collision_data() const { return *host_collision_data; }
    CollisionData<luisa::compute::Buffer>& get_device_collision_data() const { return *collision_data; }

  private:
    luisa::compute::Shader<1, luisa::compute::BufferView<float>> fn_reset_float;
    luisa::compute::Shader<1,
                           luisa::compute::BufferView<float3>,  // sa_x
                           float,                               // substep_dt
                           float                                // stiffness_dirichlet
                           >
        fn_calc_energy_inertia;
    luisa::compute::Shader<1,
                           luisa::compute::BufferView<float3>,  // sa_q
                           float,                               // substep_dt
                           float                                // stiffness_dirichlet
                           >
        fn_calc_energy_abd_inertia;
    // luisa::compute::Shader<1,
    //     luisa::compute::BufferView<float3>, // sa_x
    //     float, // substep_dt
    //     float // stiffness_dirichlet
    //     > fn_calc_energy_dirichlet;
    luisa::compute::Shader<1,
                           luisa::compute::BufferView<float3>,  // sa_x
                           float                                // stiffness_spring
                           >
        fn_calc_energy_spring;
    luisa::compute::Shader<1,
                           luisa::compute::BufferView<float3>,  // sa_q
                           float                                // stiffness_spring
                           >
        fn_calc_energy_abd_ortho;
    luisa::compute::Shader<1,
                           luisa::compute::BufferView<float3>,  // sa_x
                           float                                // stiffness_bending
                           >
        fn_calc_energy_bending;
    luisa::compute::Shader<1,
                           luisa::compute::BufferView<float3>,  // sa_x
                           float,                               // floor_y
                           bool,                                // use_ground_collision
                           float,                               // stiffness
                           float,                               // d_hat
                           float                                // thickness
                           >
        fn_calc_energy_ground_collision;
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


}  // namespace lcs