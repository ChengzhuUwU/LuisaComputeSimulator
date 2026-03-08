#pragma once

#include "Core/float_n.h"
#include <memory>

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
		bool simulate_rigid = false;
		bool simulate_obstacle = false;

		bool draw_line = true;
		bool draw_obstacle = true;
		bool draw_cloth = true;
		bool draw_deformable_body = true;

		// Global Config
		// bool use_explicit = false;
		// bool use_substep = true;
		bool use_gpu = true;
		bool fix_scene = false;
		bool use_energy_linesearch = false;
		bool use_ccd_linesearch = true;

		bool print_system_energy = false;
		bool print_pcg_info = false;
		bool print_collision_info = false;

		// bool use_small_timestep = false;
		// bool use_big_damping = false;

		// Collision
		bool use_floor = true;
		bool use_self_collision = true;

		// Animation
		bool output_per_frame = false;
		bool output_per_iteration = false;
		uint scene_id = 0;

		// Iteration info
		uint num_substep = 1;
		uint nonlinear_iter_count = 3;
		uint pcg_iter_count = 100;

		uint current_frame = 0;
		uint current_nonlinear_iter = 0;
		uint current_pcg_it = 0;
		uint current_substep = 0;

		uint collision_detection_frequece = 1;

		uint contact_energy_type = 1; // 0 for quadratic, 1 for log-barrier

		float implicit_dt = 1.f / 60.f;
		float explicit_dt = 1E-4;
		float dt = implicit_dt;
		float dt_inv = 1.0f / dt;
		float dt_2_inv = dt_inv * dt_inv;

		// Stiffness
		float stiffness_bending_ui = 1.0f;
		float stiffness_collision = 1e8;
		float stiffness_dirichlet = 1e9;

		// Damping
		float damping_rate = 2.0f;

		// Thickness & Friction
		float d_hat = 1e-3f;

		lcs::float3 gravity{ 0, -9.8f, 0 };
		lcs::float3 floor{ 0, 0, 0 };

		SceneParams() {}

		void update_dt(const float input_dt)
		{
			dt = input_dt;
			dt_inv = 1.0f / dt;
			dt_2_inv = dt_inv * dt_inv;
		}
		float get_substep_dt() { return implicit_dt / float(num_substep); }
		float get_bending_stiffness_scaling() { return stiffness_bending_ui; }
	};

	void						 set_scene_params_ptr(const std::shared_ptr<SceneParams>& scene_params_ptr);
	std::shared_ptr<SceneParams> get_scene_params_ptr();
	SceneParams&				 get_scene_params();
	// std::vector<SceneParams>& get_scene_params_array();

} // namespace lcs