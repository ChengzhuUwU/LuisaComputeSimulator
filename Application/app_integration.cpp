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

	auto upper_square = lcs::Initializer::WorldData()
							.set_name("upper square")
							.load_mesh_from_path(std::string(LCSV_RESOURCE_PATH) + "/InputMesh/square2.obj")
							.set_material_type(lcs::Material::MaterialType::Cloth)
							.set_physics_material(lcs::Material::ClothMaterial{
								.stretch_model = lcs::Material::ConstitutiveStretchModelCloth::Spring,
							})
							.set_translation({ 0.0f, 0.4f, 0.0f })
							.add_fixed_point_info({ .method = lcs::Initializer::FixedPointsType::LeftBack });
	uint upper_square_id = solver.register_world_data(upper_square);

	std::vector<std::array<float, 3>> square_mesh_vertices{ { -0.5, 0, -0.5 }, { 0.5, 0, -0.5 }, { -0.5, 0, 0.5 }, { 0.5, 0, 0.5 } };
	std::vector<std::array<uint, 3>>  square_mesh_faces{ { 0, 3, 1 }, { 0, 2, 3 } };
	auto							  lower_square = lcs::Initializer::WorldData()
							.set_name("lower square")
							.load_mesh_from_array(square_mesh_vertices, square_mesh_faces)
							.set_material_type(lcs::Material::MaterialType::Cloth)
							.set_physics_material(lcs::Material::ClothMaterial{})
							.set_scale(0.8f)
							.set_translation({ 0.1f, 0.2f, 0.0f })
							.add_fixed_point_info({ .method = lcs::Initializer::FixedPointsType::Left })
							.add_fixed_point_info({ .method = lcs::Initializer::FixedPointsType::Right });
	uint lower_square_id = solver.register_world_data(lower_square);

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