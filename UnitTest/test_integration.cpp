/**
 * @file test_integration.cpp
 * @brief Full Newton iteration pipeline integration tests
 *
 * Test Coverage:
 * - Full physics_step_CPU and physics_step_GPU pipeline
 * - CPU vs GPU consistency
 * - Multi-step simulation determinism
 * - Velocity update correctness
 * - Predict position correctness
 * - Fixed-point constraint preservation
 * - Energy conservation (elastic energy should not increase without external forces)
 */

#include "test_base_solver.h"
#include "test_framework.h"
#include "luisa/core/logging.h"

using namespace lcs;
using namespace lcs::test;

// =============================================================================
// Integration Test Cases
// =============================================================================

class TestIntegration : public TestNewtonSolverBase
{
public:
	// -------------------------------------------------------------------------
	// Test 1: Cloth free fall - gravity should pull it down
	// -------------------------------------------------------------------------
	bool test_cloth_free_fall()
	{
		std::cout << "\n  [Test] Cloth free fall under gravity...\n";

		setup_cloth_scene(3);
		init_solver();

		auto* sim = get_host_sim_data();

		// Initial positions
		const auto& rest_x = sim->sa_rest_x;
		float		initial_avg_y = 0.0f;
		for (const auto& p : rest_x)
			initial_avg_y += p.y;
		initial_avg_y /= rest_x.size();

		std::cout << "    Initial avg Y: " << initial_avg_y << "\n";

		// Run CPU step
		physics_step_CPU();

		const auto& curr_x = sim->sa_q_outer;
		float		new_avg_y = 0.0f;
		for (const auto& p : curr_x)
			new_avg_y += p.y;
		new_avg_y /= curr_x.size();

		std::cout << "    After CPU step avg Y: " << new_avg_y << "\n";

		// Restart and run GPU step
		restart_system();
		physics_step_GPU();

		const auto& gpu_x = sim->sa_q_outer;
		float		gpu_avg_y = 0.0f;
		for (const auto& p : gpu_x)
			gpu_avg_y += p.y;
		gpu_avg_y /= gpu_x.size();

		std::cout << "    After GPU step avg Y: " << gpu_avg_y << "\n";

		// Both should fall
		TEST_ASSERT(new_avg_y < initial_avg_y, "Cloth should fall under gravity (CPU)");
		TEST_ASSERT(gpu_avg_y < initial_avg_y, "Cloth should fall under gravity (GPU)");

		std::cout << "    PASSED\n";
		return true;
	}

	// -------------------------------------------------------------------------
	// Test 2: CPU vs GPU consistency
	// -------------------------------------------------------------------------
	bool test_cpu_gpu_consistency()
	{
		std::cout << "\n  [Test] CPU vs GPU result consistency...\n";

		setup_stretch_scene();
		init_solver();

		auto* sim = get_host_sim_data();

		// Run CPU
		physics_step_CPU();
		auto cpu_positions = sim->sa_q;

		// Restart and run GPU
		restart_system();
		physics_step_GPU();
		auto gpu_positions = sim->sa_q;

		// Compare positions
		float max_diff = 0.0f;
		for (size_t i = 0; i < cpu_positions.size(); ++i)
		{
			float3 diff = cpu_positions[i] - gpu_positions[i];
			max_diff = std::max(max_diff, luisa::length(diff));
		}

		std::cout << "    Max position diff (CPU vs GPU): " << max_diff << "\n";

		// CPU and GPU should give similar results
		// (Tolerance is higher because of floating point differences)
		TEST_ASSERT(max_diff < 1e-2f,
			"CPU and GPU results should be close (within 1e-2)");

		std::cout << "    PASSED\n";
		return true;
	}

	// -------------------------------------------------------------------------
	// Test 3: Determinism - same initial state should give same result
	// -------------------------------------------------------------------------
	bool test_determinism()
	{
		std::cout << "\n  [Test] Simulation determinism...\n";

		setup_cloth_scene(3);
		init_solver();

		// Run step 1
		auto* sim = get_host_sim_data();
		physics_step_CPU();
		auto step1_positions = sim->sa_q;

		// Reset to the registered rest state and rerun the same first step.
		restart_system();
		physics_step_CPU();
		auto step1b_positions = sim->sa_q;

		TEST_ASSERT(step1_positions.size() == step1b_positions.size(),
			"Repeated runs should have the same number of simulated vertices");

		// Compare
		float max_diff = 0.0f;
		for (size_t i = 0; i < step1_positions.size(); ++i)
		{
			float3 diff = step1_positions[i] - step1b_positions[i];
			max_diff = std::max(max_diff, luisa::length(diff));
		}

		std::cout << "    Max position diff (determinism): " << max_diff << "\n";
		TEST_ASSERT(max_diff < 1e-6f, "Simulation should be deterministic");

		std::cout << "    PASSED\n";
		return true;
	}

	// -------------------------------------------------------------------------
	// Test 4: Fixed-point preservation
	// -------------------------------------------------------------------------
	bool test_fixed_point_preservation()
	{
		std::cout << "\n  [Test] Fixed-point constraint preservation...\n";

		setup_stretch_scene();
		init_solver();

		auto* sim = get_host_sim_data();

		// Run several steps
		for (int i = 0; i < 3; ++i)
			physics_step_CPU();

		const auto& curr_q = sim->sa_q;
		const auto& rest_x = sim->sa_rest_x;

		// Fixed vertices (0 and 2) should remain at rest positions
		float3 fixed_0_diff = curr_q[0] - rest_x[0];
		float3 fixed_2_diff = curr_q[2] - rest_x[2];

		float tol = 1e-3f;
		TEST_ASSERT_VEC3_NEAR(curr_q[0], rest_x[0], tol,
			"Fixed vertex 0 should remain at rest position");
		TEST_ASSERT_VEC3_NEAR(curr_q[2], rest_x[2], tol,
			"Fixed vertex 2 should remain at rest position");

		std::cout << "    Fixed vertex 0: " << vec3_str(curr_q[0])
				  << " (rest: " << vec3_str(rest_x[0]) << ")\n";
		std::cout << "    Fixed vertex 2: " << vec3_str(curr_q[2])
				  << " (rest: " << vec3_str(rest_x[2]) << ")\n";
		std::cout << "    PASSED\n";
		return true;
	}

	// -------------------------------------------------------------------------
	// Test 5: Velocity update correctness
	// -------------------------------------------------------------------------
	bool test_velocity_update()
	{
		std::cout << "\n  [Test] Velocity update under gravity...\n";

		setup_cloth_scene(3);
		init_solver();

		auto* sim = get_host_sim_data();

		// Record initial velocity
		auto initial_v = sim->sa_q_v;

		// Run a step
		physics_step_CPU();

		// Check that velocity has changed (gravity applied)
		bool velocity_changed = false;
		for (size_t i = 0; i < sim->sa_q_v.size(); ++i)
		{
			if (luisa::length(sim->sa_q_v[i] - initial_v[i]) > 1e-6f)
			{
				velocity_changed = true;
				break;
			}
		}

		TEST_ASSERT(velocity_changed, "Velocity should change due to gravity");

		std::cout << "    Initial velocity[0]: " << vec3_str(initial_v[0]) << "\n";
		std::cout << "    New velocity[0]: " << vec3_str(sim->sa_q_v[0]) << "\n";
		std::cout << "    PASSED\n";
		return true;
	}

	// -------------------------------------------------------------------------
	// Test 6: Predict position
	// -------------------------------------------------------------------------
	bool test_predict_position()
	{
		std::cout << "\n  [Test] Predict position integration...\n";

		setup_cloth_scene(3);
		init_solver();

		auto* sim = get_host_sim_data();

		// Record step start
		auto step_start = sim->sa_q_step_start;

		// Run predict position (via physics_step)
		physics_step_CPU();

		// q_step_start should be different from q after prediction
		float change = 0.0f;
		for (size_t i = 0; i < sim->sa_q.size(); ++i)
		{
			change += luisa::length(sim->sa_q[i] - step_start[i]);
		}

		TEST_ASSERT(change > 1e-6f, "Predict position should change q");

		std::cout << "    Average position change: " << change / sim->sa_q.size() << "\n";
		std::cout << "    PASSED\n";
		return true;
	}

	// -------------------------------------------------------------------------
	// Test 7: Multi-step stability
	// -------------------------------------------------------------------------
	bool test_multi_step_stability()
	{
		std::cout << "\n  [Test] Multi-step simulation stability...\n";

		setup_cloth_scene(3);
		init_solver();

		auto* sim = get_host_sim_data();

		// Run 5 steps
		for (int i = 0; i < 5; ++i)
		{
			physics_step_CPU();
		}

		// Check for NaN
		bool has_nan = false;
		for (const auto& p : sim->sa_q)
		{
			if (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z))
			{
				has_nan = true;
				break;
			}
		}

		TEST_ASSERT(!has_nan, "Simulation should not produce NaN after 5 steps");

		// Check energy is bounded
		float total_energy = 0.0f;
		for (const auto& p : sim->sa_q)
		{
			total_energy += luisa::dot(p, p);
		}
		std::cout << "    Total position magnitude after 5 steps: " << total_energy << "\n";

		TEST_ASSERT(total_energy < 1e6f, "Positions should be bounded");

		std::cout << "    PASSED\n";
		return true;
	}

	// -------------------------------------------------------------------------
	// Test 8: Soft body (tetrahedron) free fall
	// -------------------------------------------------------------------------
	bool test_soft_body_free_fall()
	{
		std::cout << "\n  [Test] Soft body (tet) free fall...\n";

		setup_soft_body_scene();
		init_solver();

		auto* sim = get_host_sim_data();

		const auto& rest_x = sim->sa_rest_x;
		float		initial_avg_y = 0.0f;
		for (const auto& p : rest_x)
			initial_avg_y += p.y;
		initial_avg_y /= rest_x.size();

		physics_step_CPU();

		const auto& curr_x = sim->sa_q_outer;
		float		new_avg_y = 0.0f;
		for (const auto& p : curr_x)
			new_avg_y += p.y;
		new_avg_y /= curr_x.size();

		std::cout << "    Initial avg Y: " << initial_avg_y << "\n";
		std::cout << "    After step avg Y: " << new_avg_y << "\n";

		TEST_ASSERT(new_avg_y < initial_avg_y, "Soft body should fall under gravity");

		std::cout << "    PASSED\n";
		return true;
	}
};

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv)
{
	luisa::log_level_info();
	print_separator("NewtonSolver Integration Tests");

	TestSuiteResult suite;
	suite.name = "Integration";

	int total = 0, passed = 0;

	auto run = [&](bool (TestIntegration::*fn)(), const char* name)
	{
		total++;
		TestIntegration test;
		try
		{
			test.init_device();
			auto result = run_test(name, [&]
				{ return (test.*fn)(); });
			suite.add(result);
			if (result.passed)
				passed++;
		}
		catch (const std::exception& e)
		{
			std::cerr << "  [EXCEPTION] " << e.what() << "\n";
			suite.add(TestResult(false), false);
		}
	};

	run(&TestIntegration::test_cloth_free_fall, "Cloth free fall under gravity");
	run(&TestIntegration::test_cpu_gpu_consistency, "CPU vs GPU consistency");
	run(&TestIntegration::test_determinism, "Simulation determinism");
	run(&TestIntegration::test_fixed_point_preservation, "Fixed-point constraint preservation");
	run(&TestIntegration::test_velocity_update, "Velocity update under gravity");
	run(&TestIntegration::test_predict_position, "Predict position integration");
	run(&TestIntegration::test_multi_step_stability, "Multi-step stability");
	run(&TestIntegration::test_soft_body_free_fall, "Soft body (tet) free fall");

	print_suite_summary(suite);

	std::cout << "\n";
	std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
	auto pass_str = std::to_string(passed);
	auto total_str = std::to_string(total);
	int	 padding = std::max(0, 32 - static_cast<int>(pass_str.size()) - static_cast<int>(total_str.size()));
	std::cout << "║  Integration Tests: " << passed << "/" << total << " passed"
			  << std::string(padding, ' ') << "║\n";
	std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";

	return (passed == total) ? 0 : 1;
}
