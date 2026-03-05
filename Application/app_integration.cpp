#include <string>
#include <vector>
#include "SimulationSolver/newton_solver.h"

int main(int argc, char** argv)
{
	luisa::log_level_info();

	// Init GPU system
	std::string backend;
	if (argc >= 2)
	{
		backend = argv[1];
	}

	lcs::NewtonSolver solver;
	solver.create_device(argv[0], backend);

	auto upper_square = solver.register_world_data_from_file_path("upper square", std::string(LCSV_RESOURCE_PATH) + "/InputMesh/square2.obj")
							.set_material_type(lcs::Initializer::MaterialType::Cloth)
							.set_physics_material(lcs::Initializer::ClothMaterial{
								.stretch_model = lcs::Initializer::ConstitutiveStretchModelCloth::Spring,
							})
							.set_translation({ 0.0f, 0.4f, 0.0f })
							.add_fixed_point_info({ .method = lcs::Initializer::FixedPointsType::LeftBack });

	std::vector<std::array<float, 3>> square_mesh_vertices{ { -0.5, 0, -0.5 }, { 0.5, 0, -0.5 }, { -0.5, 0, 0.5 }, { 0.5, 0, 0.5 } };
	std::vector<std::array<uint, 3>>  square_mesh_faces{ { 0, 3, 1 }, { 0, 2, 3 } };

	auto lower_square = solver.register_world_data_from_array("lower square", square_mesh_vertices, square_mesh_faces)
							.set_material_type(lcs::Initializer::MaterialType::Cloth)
							.set_physics_material(lcs::Initializer::ClothMaterial{})
							.set_scale(0.8f)
							.set_translation({ 0.1f, 0.2f, 0.0f })
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
	solver.get_curr_vertices_to_host(sa_rendering_vertices);

	// Main application
	for (uint ii = 0; ii < 20; ii++)
	{
		// Update animation states

		solver.physics_step_GPU();

		solver.get_curr_vertices_to_host(sa_rendering_vertices);

		// Display or other processing
	}

	solver.save_mesh_to_obj(luisa::format("{}/OutputMesh/sample.obj", LCSV_RESOURCE_PATH));

	return 0;
}