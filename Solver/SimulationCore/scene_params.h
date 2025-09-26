#pragma once

#include "Core/float_n.h"

namespace lcs 
{

enum SolverType
{
    SolverTypeNewton,
    SolverTypeXPBD,
    SolverTypeVBD,
};

struct SceneParams 
{
// public:
    bool simulate_cloth = false;
    bool simulate_tet = false;
    bool simulate_obstacle = false;

    bool draw_line = true;
    bool draw_obstacle = true;
    bool draw_cloth = true;
    bool draw_deformable_body = true;
    bool disable_enable_fixed_cloth = true;
    bool disable_enable_fixed_tet = true;

    // Solver type
    bool use_newton_solver = false;
    bool use_xpbd_solver = true;
    bool use_vbd_solver = false;
    bool use_sod_solver = false;
    bool sod_use_jacobi = false;
    
    // Global Config
    bool use_explicit = false;
    bool use_gpu = false;
    bool fix_scene = false;
    bool use_substep = true;
    bool use_energy_linesearch = false;
    bool use_ccd_linesearch = false;
    
    bool print_system_energy = false;
    bool print_pcg_convergence = false;
    bool print_xpbd_convergence = false;

    bool use_small_timestep = false;
    bool use_big_damping = false;

    // Collision
    bool use_floor = true;
    bool use_self_collision = true;
    bool use_obstacle_collision = true;
    bool use_ccd_constraint = false;
    
    bool use_collision_vf = false;
    bool use_collision_ee = false;

    bool print_collision_count = false;

    // Bending
    bool use_bending = true;
    bool use_quadratic_bending_model = false;

    // Animation
    bool use_stretch_animation = false;
    bool use_obstacle_animation = true;

    // Iteration info
    uint num_iteration = 96;   
    uint num_substep = 1;
    uint nonlinear_iter_count = 8;
    uint pcg_iter_count = 50;

    uint current_frame = 0;
    uint current_nonlinear_iter = 0;
    uint current_pcg_it = 0;
    uint current_substep = 0;

    uint load_state_frame = -1u;

    uint collision_detection_frequece = 1;
    uint animation_start_frame = 9999;

    uint contact_energy_type = 0;

    // uint max_vv_per_vert_broad_self_collision = 32;
    // uint max_vf_per_vert_broad_self_collision = 32;
    // uint max_vv_per_vert_narrow_self_collision = 16;
    // uint max_vf_per_vert_narrow_self_collision = 16;
    // uint max_vv_per_vert_broad_obstacle_collision = 32;
    // uint max_vf_per_vert_broad_obstacle_collision = 32;
    // uint max_vv_per_vert_narrow_obstacle_collision = 16;
    // uint max_vf_per_vert_narrow_obstacle_collision = 16;

    uint laplasion_damping_cloth_count = 1;
    uint laplasion_damping_tet_count = 1;
    float laplasion_damping_cloth_weight = 0.5;
    float laplasion_damping_tet_weight = 0.5;

    uint use_big_damping_frame = 999999;
    uint dag_mode = -1u; // -1u:non, 0:full cloth, 1:full tet, 2:hybrid

    float implicit_dt = 1.f/30.f;
    float explicit_dt = 1E-4;
    float dt = implicit_dt;
    float dt_inv = 1.0f / dt;
    float dt_2_inv = dt_inv * dt_inv;
    

    // Mass
    float default_mass = 1.0f;
    float density_cloth = 0.01;
    float density_tet = 10.0;

    // Stiffness
    float stiffness_spring = 1e4;
    float youngs_modulus_cloth = 1e7;
    float youngs_modulus_tet = 1e7;
    float poisson_ratio_cloth = 0.2f;
    float poisson_ratio_tet = 0.2f;

    float stiffness_bending_ui = 0.5f;
    float stiffness_quadratic_bending = 5e-3f;
    float stiffness_DAB_bending = stiffness_quadratic_bending;
    float stiffness_collision = 1e9;
    float stiffness_dirichlet = 1e9;

    // Balloon constraint
    float balloon_scale_rate = 1.0f; 
    float stiffness_pressure = 1e5; 

    // Damping
    float damping_cloth = 2.0f; 
    float damping_tet = 0.0f; 

    // Thickness & Friction
    float thickness = 0.0f;
    float d_hat = 3e-3f;
    // float thickness_vv_obstacle = 0.01f;
    // float thickness_vv_cloth = 0.01f;
    // float thickness_vv_tet = 0.01f;
    // float thickness_vv_cross = 0.01f;

    // float friction_cloth = 0.25;
    // float friction_tet = 0.1;
    // float friction_cross = 0.1;
    // float friction_obstacle_cloth = 0.25;
    // float friction_obstacle_tet = 0.1;
    float collision_query_range_rate = lcs::Sqrt_2;
    // float collision_query_range_vv = thickness_vv * Sqrt_2;

    
    float chebyshev_omega = 1.0f;
    
    lcs::float3 gravity{0, -9.8f, 0};
    lcs::float3 floor{0, 0, 0};

    // constant
    bool use_fixed_verts = false;
    bool use_attach = false;
    SolverType solver_type = SolverTypeVBD;

    SceneParams(){}
    
    void update_dt(const float input_dt)
    {
        dt = input_dt;
        dt_inv = 1.0f / dt;
        dt_2_inv = dt_inv * dt_inv;
    }
    uint get_curr_iteration_with_substep() { return num_substep * current_substep + current_nonlinear_iter; }
    float get_substep_dt() { return implicit_dt / float(num_substep); }
    float get_stiffness_quadratic_bending() { return stiffness_quadratic_bending * stiffness_bending_ui; }
    float get_stiffness_DAB_bending() { return stiffness_DAB_bending * stiffness_bending_ui; }
    
};

#ifndef  METAL_CODE

void init_scene_params();
SceneParams& get_scene_params();
// std::vector<SceneParams>& get_scene_params_array();

#endif


} // namespace lcs