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
                   std::vector<lcs::Initializer::WorldData>& shell_list);
    void compile(AsyncCompiler& compiler);
    void set_data_pointer(SolverData& solver_data, SolverHelper& solver_helper);

  public:
    void restart_system();
    void save_current_frame_state_to_host(const uint frame, const std::string& addition_str);
    void load_saved_state_from_host(const uint frame, const std::string& addition_str);
    void save_mesh_to_obj(const uint frame, const std::string& addition_str = "");
    void device_compute_elastic_energy(luisa::compute::Stream& stream, std::map<std::string, double>& energy_list);
    void compile_compute_energy(AsyncCompiler& compiler);

  public:
    void get_simulation_results_to_host(std::vector<std::vector<std::array<float, 3>>>& output_positions);
    void update_pinned_verts_position(const uint                  meshIdx,
                                      const uint                  local_vid,
                                      const std::array<float, 3>& pinned_verts_target_position);
    void update_pinned_body_state(const uint                  body_id,
                                  const std::array<float, 3>& translation = {0.0f, 0.0f, 0.0f},
                                  const std::array<float, 4>& rotation    = {0.0f, 0.0f, 0.0f, 0.0f});

  protected:
    void physics_step_prev_operation();
    void physics_step_post_operation();

  private:
    SolverData   solver_data;
    SolverHelper solver_helper;

    std::vector<Animation::PerVertexAnimation> per_vertex_animations;
    std::vector<Animation::PerBodyAnimation>   per_body_animations;

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
                           Constitutions::SoftInertia<luisa::compute::Buffer>,
                           luisa::compute::BufferView<float3>,  // sa_x
                           float                                // substep_dt
                           >
        fn_calc_energy_inertia;
    luisa::compute::Shader<1,
                           Constitutions::AbdInertia<luisa::compute::Buffer>,
                           luisa::compute::BufferView<float3>,  // sa_q
                           float                                // substep_dt
                           >
        fn_calc_energy_abd_inertia;

    luisa::compute::Shader<1,
                           Constitutions::StretchSpring<luisa::compute::Buffer>,  // stretch_spring_constitution
                           luisa::compute::BufferView<float3>,                    // sa_x
                           float                                                  // stiffness_spring
                           >
        fn_calc_energy_spring;
    luisa::compute::Shader<1,
                           Constitutions::StretchFace<luisa::compute::Buffer>,  // stretch_face_constitution
                           luisa::compute::BufferView<float3>                   // sa_x
                           >
        fn_calc_energy_stretch_face;
    luisa::compute::Shader<1,
                           Constitutions::AbdOrthogonality<luisa::compute::Buffer>,  // abd_ortho_constitution
                           luisa::compute::BufferView<float3>                        // sa_q
                           >
        fn_calc_energy_abd_ortho;
    luisa::compute::Shader<1,
                           Constitutions::BendingEdge<luisa::compute::Buffer>,  // bending_edge_constitution
                           luisa::compute::BufferView<float3>,                  // sa_x
                           float                                                // stiffness_bending
                           >
        fn_calc_energy_bending;
    luisa::compute::Shader<1,
                           luisa::compute::BufferView<float3>,  // sa_x
                           float,                               // floor_y
                           bool,                                // use_ground_collision
                           float,                               // stiffness
                           uint                                 // collision_type
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