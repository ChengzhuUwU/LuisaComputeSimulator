/**
 * LuisaComputeSimulator - C++ Integration Example
 *
 * This file demonstrates how to use the LuisaComputeSimulator C++ API
 * for soft body / cloth simulation with different constitutive models.
 *
 * Build: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
 *        cmake --build build -j
 * Run:   ./build/bin/app_integration cuda
 */

#include <string>
#include <vector>
#include "SimulationSolver/newton_solver.h"

int main(int argc, char** argv)
{
	// Set log level
	luisa::log_level_info();

	// Parse backend from command line (cuda, dx, vk, metal, or empty for CPU)
	std::string backend;
	if (argc >= 2)
	{
		backend = argv[1];
	}

	// Create solver instance
	lcs::NewtonSolver solver;
	solver.create_device(argv[0], backend);

	// ========================================================================
	// Example 1: Cloth with FEM_BW98 (Finite Element Method)
	// Suitable for large deformations, physically-based materials
	// ========================================================================
	auto upper_square = lcs::Initializer::WorldData()
							.set_name("upper square")
							.load_mesh_from_path(std::string(LCSV_RESOURCE_PATH) + "/InputMesh/square2.obj")
							.set_material_type(lcs::Material::MaterialType::Cloth)
							.set_physics_material(lcs::Material::ClothMaterial{
								.stretch_model = lcs::Material::ConstitutiveStretchModelCloth::FEM_BW98, // Finite strain energy
								.bending_model = lcs::Material::ConstitutiveBendingModelCloth::QuadraticBending,
								.thickness = 0.001f,
								.youngs_modulus = 1e6f,
								.poisson_ratio = 0.3f,
							})
							.set_translation({ 0.0f, 0.4f, 0.0f })
							.add_fixed_point_from_method({ .method = lcs::Initializer::FixedPointsType::LeftBack });
	uint upper_square_id = solver.register_world_data(upper_square);

	// ========================================================================
	// Example 2: Cloth with Spring model (linear, fast)
	// Suitable for real-time applications with basic cloth behavior
	// ========================================================================
	std::vector<std::array<float, 3>> square_mesh_vertices{ { -0.5, 0, -0.5 }, { 0.5, 0, -0.5 }, { -0.5, 0, 0.5 }, { 0.5, 0, 0.5 } };
	std::vector<std::array<uint, 3>>  square_mesh_faces{ { 0, 3, 1 }, { 0, 2, 3 } };
	auto							  lower_square = lcs::Initializer::WorldData()
							.set_name("lower square")
							.load_mesh_from_array(square_mesh_vertices, square_mesh_faces)
							.set_material_type(lcs::Material::MaterialType::Cloth)
							.set_physics_material(lcs::Material::ClothMaterial{
								.stretch_model = lcs::Material::ConstitutiveStretchModelCloth::Spring, // Linear spring
								.bending_model = lcs::Material::ConstitutiveBendingModelCloth::DihedralAngle,
								.thickness = 0.001f,
								.youngs_modulus = 1e5f,
							})
							.set_scale(0.8f)
							.set_translation({ 0.1f, 0.2f, 0.0f })
							.add_fixed_point_from_method({ .method = lcs::Initializer::FixedPointsType::Left })
							.add_fixed_point_from_method({ .method = lcs::Initializer::FixedPointsType::Right });
	uint lower_square_id = solver.register_world_data(lower_square);

	// ========================================================================
	// Example 3: Rigid Body (Affine Body Dynamics)
	// ========================================================================
	// auto rigid_body = lcs::Initializer::WorldData()
	// 						.set_name("rigid cube")
	// 						.load_mesh_from_path(std::string(LCSV_RESOURCE_PATH) + "/InputMesh/cube.obj")
	// 						.set_material_type(lcs::Material::MaterialType::Rigid)
	// 						.set_translation({ 0.0f, 0.34f, 0.0f })
	// 						.set_rotation({ 0.5235988f, 0.0f, 0.5235988f })  // Euler angles
	// 						.set_scale(0.1f);
	// uint rigid_id = solver.register_world_data(rigid_body);

	// ========================================================================
	// Configure solver parameters
	// ========================================================================
	auto& config = solver.get_config();
	config.use_floor = false;
	config.use_self_collision = true;
	config.implicit_dt = 1.0 / 60.0; // 60 FPS timestep
	config.use_energy_linesearch = true;
	config.nonlinear_iter_count = 10;
	config.pcg_iter_count = 100;

	// ========================================================================
	// Initialize solver (compiles shaders, allocates GPU buffers)
	// ========================================================================
	solver.init_solver();

	// Get initial vertices for rendering
	std::vector<std::vector<std::array<float, 3>>> sa_rendering_vertices;
	solver.get_curr_vertices_to_host(sa_rendering_vertices);

	// ========================================================================
	// Main simulation loop
	// ========================================================================
	for (uint ii = 0; ii < 20; ii++)
	{
		// Step physics (GPU or CPU)
		solver.physics_step_GPU(); // or solver.physics_step_CPU()

		// Retrieve updated vertices
		solver.get_curr_vertices_to_host(sa_rendering_vertices);

		// TODO: Render or process vertices here
	}

	// Save final result to OBJ
	solver.save_mesh_to_obj(luisa::format("{}/OutputMesh/sample.obj", LCSV_RESOURCE_PATH));

	return 0;
}
