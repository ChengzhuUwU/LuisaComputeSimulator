#include <iostream>
#include <luisa/luisa-compute.h>
#include <string>

#include "CollisionDetector/lbvh.h"
#include "CollisionDetector/narrow_phase.h"
#include "Core/constant_value.h"
#include "Initializer/init_collision_data.h"
#include "MeshOperation/default_mesh.h"
#include "MeshOperation/mesh_reader.h"
#include "SimulationSolver/newton_solver.h"
#include "Utils/cpu_parallel.h"
#include "Utils/device_parallel.h"
#include "Utils/buffer_filler.h"

#include "SimulationCore/scene_params.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationSolver/solver_interface.h"

#include "Initializer/init_mesh_data.h"
#include "Initializer/init_sim_data.h"
#include "app_simulation_demo_config.h"
#include "luisa/core/basic_types.h"

int main(int argc, char** argv)
{
	luisa::log_level_info();

	// Init GPU system
	const std::string binary_path(argv[0]);
	std::string		  backend;
	if (argc >= 2)
	{
		backend = argv[1];
	}

	lcs::NewtonSolver solver;
	solver.create_device(binary_path, backend);

	auto upper_square = solver.register_world_data_from_file_path("upper square", std::string(LCSV_RESOURCE_PATH) + "/InputMesh/square2.obj")
							.set_simulation_type(lcs::Initializer::SimulationType::Cloth)
							.set_physics_material(lcs::Initializer::ClothMaterial{
								.stretch_model = lcs::Initializer::ConstitutiveStretchModelCloth::Spring,
								.thickness = 0.1f })
							.add_fixed_point_info({ .method = lcs::Initializer::FixedPointsType::LeftBack });

	std::vector<std::array<float, 3>> square_mesh_vertices{ { -0.5, 0, -0.5 }, { 0.5, 0, -0.5 }, { -0.5, 0, 0.5 }, { 0.5, 0, 0.5 } };
	std::vector<std::array<uint, 3>>  square_mesh_faces{ { 0, 3, 1 }, { 0, 2, 3 } };

	auto lower_square = solver.register_world_data_from_array("lower square", square_mesh_vertices, square_mesh_faces)
							.set_simulation_type(lcs::Initializer::SimulationType::Cloth)
							.set_physics_material(lcs::Initializer::ClothMaterial{
								.thickness = 0.1f })
							.set_scale(1.0f)
							.set_translation({ 0.1f, -0.2f, 0.0f })
							.add_fixed_point_info({ .method = lcs::Initializer::FixedPointsType::Left })
							.add_fixed_point_info({ .method = lcs::Initializer::FixedPointsType::Right });

	auto config = solver.get_config();
	config.use_floor = false;
	config.implicit_dt = 0.2;
	config.use_energy_linesearch = true;

	// Init Solver
	solver.init_solver();

	// Init rendering data
	std::vector<std::vector<std::array<float, 3>>> sa_rendering_vertices;
	solver.get_simulation_results_to_host(sa_rendering_vertices);

	// Main application
	for (uint ii = 0; ii < 20; ii++)
	{
		// Update animation states

		solver.physics_step_GPU();

		solver.get_simulation_results_to_host(sa_rendering_vertices);

		// Display or other processing
	}

	return 0;
}