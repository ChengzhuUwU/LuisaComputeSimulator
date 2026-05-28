/**
 * @file test_pcg_solver.cpp
 * @brief Tests for the project's PCG solver in Solver/LinearSolver/precond_cg.cpp.
 *
 * These tests use Eigen only as the reference solver. The solution under test is
 * produced by ConjugateGradientSolver::host_solve(), the same host PCG path used
 * by NewtonSolver when solving assembled Newton systems.
 */

#include "test_base_solver.h"
#include "test_framework.h"
#include "Core/lc_to_eigen.h"
#include "Core/matrix_triplet.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <cmath>
#include <functional>
#include <vector>

using namespace lcs;
using namespace lcs::test;

namespace
{

constexpr float kResidualTolerance = 2e-4f;
constexpr float kSolutionTolerance = 3e-3f;

struct BlockSystem
{
	uint					  num_blocks = 0;
	Eigen::SparseMatrix<float> A;
	Eigen::VectorXf			  b;
};

float3 eigen_segment_to_float3(const Eigen::VectorXf& v, const uint block)
{
	return luisa::make_float3(v[3 * block + 0], v[3 * block + 1], v[3 * block + 2]);
}


float3x3 make_block(const Eigen::Matrix3f& m)
{
	return eigen3x3_to_float3x3(m);
}

BlockSystem make_spd_block_system(const uint num_blocks)
{
	BlockSystem system;
	system.num_blocks = num_blocks;
	system.A.resize(static_cast<int>(num_blocks * 3), static_cast<int>(num_blocks * 3));
	system.b.resize(static_cast<int>(num_blocks * 3));

	std::vector<Eigen::Triplet<float>> triplets;
	triplets.reserve(num_blocks * 9 + (num_blocks - 1) * 18);

	for (uint block = 0; block < num_blocks; ++block)
	{
		Eigen::Matrix3f diag;
		diag << 7.0f + 0.25f * block, 0.25f, -0.10f,
			0.25f, 6.0f + 0.15f * block, 0.20f,
			-0.10f, 0.20f, 5.0f + 0.10f * block;

		for (int r = 0; r < 3; ++r)
		{
			for (int c = 0; c < 3; ++c)
			{
				triplets.emplace_back(static_cast<int>(block * 3 + r), static_cast<int>(block * 3 + c), diag(r, c));
			}
		}

		if (block + 1 < num_blocks)
		{
			Eigen::Matrix3f offdiag;
			offdiag << -0.45f, 0.04f, 0.00f,
				0.02f, -0.35f, 0.03f,
				0.00f, 0.01f, -0.25f;

			for (int r = 0; r < 3; ++r)
			{
				for (int c = 0; c < 3; ++c)
				{
					triplets.emplace_back(static_cast<int>(block * 3 + r), static_cast<int>((block + 1) * 3 + c), offdiag(r, c));
					triplets.emplace_back(static_cast<int>((block + 1) * 3 + c), static_cast<int>(block * 3 + r), offdiag(c, r));
				}
			}
		}
	}

	system.A.setFromTriplets(triplets.begin(), triplets.end());

	for (int i = 0; i < system.b.size(); ++i)
	{
		system.b[i] = 0.15f + 0.07f * static_cast<float>((i * 5 + 3) % 11);
	}

	return system;
}

} // namespace

class TestPCGSolver : public TestNewtonSolverBase
{
public:
	void setup_pcg_harness(const BlockSystem& system)
	{
		init_device();
		setup_cloth_scene(2);
		get_config().pcg_iter_count = 128;
		get_config().print_pcg_info = false;
		init_solver();

		auto* sim = get_host_sim_data();
		auto* mesh = get_host_mesh_data();
		const uint n = system.num_blocks;

		sim->num_dof = n;
		sim->sa_cgX.assign(n, luisa::make_float3(0.0f));
		sim->sa_cgB.assign(n, luisa::make_float3(0.0f));
		sim->sa_cgA_diag.assign(n, luisa::make_float3x3(0.0f));
		sim->sa_cgMinv.assign(n, luisa::make_float3x3(0.0f));
		sim->sa_cgP.assign(n, luisa::make_float3(0.0f));
		sim->sa_cgQ.assign(n, luisa::make_float3(0.0f));
		sim->sa_cgR.assign(n, luisa::make_float3(0.0f));
		sim->sa_cgZ.assign(n, luisa::make_float3(0.0f));
		sim->sa_block_result.assign(2 * n + 8, 0.0f);
		sim->sa_convergence.assign(256, 0.0f);
		sim->sa_cgA_fixtopo_offdiag_triplet.clear();
		sim->sa_cgA_fixtopo_offdiag_triplet_info.clear();

		mesh->sa_is_fixed.assign(n, 0u);

		for (uint block = 0; block < n; ++block)
		{
			sim->sa_cgB[block] = eigen_segment_to_float3(system.b, block);
			sim->sa_cgA_diag[block] = make_block(Eigen::Matrix3f(system.A.block(static_cast<int>(3 * block), static_cast<int>(3 * block), 3, 3)));
		}

		for (uint row_block = 0; row_block < n; ++row_block)
		{
			for (uint col_block = 0; col_block < n; ++col_block)
			{
				if (row_block == col_block)
				{
					continue;
				}

				Eigen::Matrix3f block = Eigen::Matrix3f(system.A.block(
					static_cast<int>(3 * row_block), static_cast<int>(3 * col_block), 3, 3));
				if (block.cwiseAbs().maxCoeff() == 0.0f)
				{
					continue;
				}

				const uint property = MatrixTriplet::is_valid();
				sim->sa_cgA_fixtopo_offdiag_triplet.push_back(
					make_matrix_triplet(row_block, col_block, property, make_block(block)));
				sim->sa_cgA_fixtopo_offdiag_triplet_info.push_back(make_matrix_triplet_info(row_block, col_block, property));
			}
		}
	}

	std::function<void(const std::vector<float3>&, std::vector<float3>&)> make_spmv(const Eigen::SparseMatrix<float>& A) const
	{
		return [A](const std::vector<float3>& input, std::vector<float3>& output)
		{
			Eigen::VectorXf x = float3_vector_to_eigen(input);
			Eigen::VectorXf y = A * x;
			output.resize(input.size());
			for (uint i = 0; i < output.size(); ++i)
			{
				output[i] = eigen_segment_to_float3(y, i);
			}
		};
	}

	void upload_pcg_harness_to_device()
	{
		auto* sim = get_host_sim_data();
		stream() << sim_data->sa_cgX.copy_from(sim->sa_cgX.data())
				 << sim_data->sa_cgB.copy_from(sim->sa_cgB.data())
				 << sim_data->sa_cgA_diag.copy_from(sim->sa_cgA_diag.data())
				 << sim_data->sa_cgMinv.copy_from(sim->sa_cgMinv.data())
				 << sim_data->sa_cgP.copy_from(sim->sa_cgP.data())
				 << sim_data->sa_cgQ.copy_from(sim->sa_cgQ.data())
				 << sim_data->sa_cgR.copy_from(sim->sa_cgR.data())
				 << sim_data->sa_cgZ.copy_from(sim->sa_cgZ.data())
				 << sim_data->sa_convergence.copy_from(sim->sa_convergence.data())
				 << luisa::compute::synchronize();
	}

	bool test_pcg_host_solve_matches_eigen_ldlt()
	{
		std::cout << "\n  [Test] Project host PCG vs Eigen SimplicialLDLT...\n";

		BlockSystem system = make_spd_block_system(6);
		setup_pcg_harness(system);

		Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> ldlt;
		ldlt.compute(system.A);
		TEST_ASSERT(ldlt.info() == Eigen::Success, "Reference SPD system must factorize with Eigen LDLT");

		Eigen::VectorXf x_ref = ldlt.solve(system.b);

		get_pcg_solver()->host_solve(stream(), make_spmv(system.A), [] { return 0.0; });

		Eigen::VectorXf x_pcg = float3_vector_to_eigen(get_host_sim_data()->sa_cgX);
		float residual = (system.A * x_pcg - system.b).norm() / std::max(system.b.norm(), 1.0f);
		float solution_error = (x_pcg - x_ref).norm() / std::max(x_ref.norm(), 1.0f);

		std::cout << "    Relative residual: " << residual << "\n";
		std::cout << "    Relative solution error: " << solution_error << "\n";

		TEST_ASSERT(residual < kResidualTolerance, "Project PCG residual should match the Eigen reference system");
		TEST_ASSERT(solution_error < kSolutionTolerance, "Project PCG solution should match Eigen LDLT");

		std::cout << "    PASSED\n";
		return true;
	}

	bool test_pcg_matches_eigen_cg_on_same_iterations()
	{
		std::cout << "\n  [Test] Project host PCG vs Eigen ConjugateGradient...\n";

		BlockSystem system = make_spd_block_system(8);
		setup_pcg_harness(system);

		Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower | Eigen::Upper> eigen_cg;
		eigen_cg.setMaxIterations(static_cast<int>(get_config().pcg_iter_count));
		eigen_cg.setTolerance(1e-7f);
		eigen_cg.compute(system.A);
		TEST_ASSERT(eigen_cg.info() == Eigen::Success, "Eigen CG setup must succeed");

		Eigen::VectorXf x_eigen = eigen_cg.solve(system.b);
		TEST_ASSERT(eigen_cg.info() == Eigen::Success, "Eigen CG solve must succeed");

		get_pcg_solver()->host_solve(stream(), make_spmv(system.A), [] { return 0.0; });

		Eigen::VectorXf x_pcg = float3_vector_to_eigen(get_host_sim_data()->sa_cgX);
		float eigen_residual = (system.A * x_eigen - system.b).norm() / std::max(system.b.norm(), 1.0f);
		float pcg_residual = (system.A * x_pcg - system.b).norm() / std::max(system.b.norm(), 1.0f);
		float solution_error = (x_pcg - x_eigen).norm() / std::max(x_eigen.norm(), 1.0f);

		std::cout << "    Eigen CG iterations: " << eigen_cg.iterations() << "\n";
		std::cout << "    Eigen relative residual: " << eigen_residual << "\n";
		std::cout << "    Project PCG relative residual: " << pcg_residual << "\n";
		std::cout << "    Relative solution error: " << solution_error << "\n";

		TEST_ASSERT(pcg_residual < kResidualTolerance, "Project PCG should converge on the same SPD system");
		TEST_ASSERT(solution_error < kSolutionTolerance, "Project PCG should agree with Eigen CG");

		std::cout << "    PASSED\n";
		return true;
	}

	bool test_project_pcg_uses_supplied_spmv()
	{
		std::cout << "\n  [Test] Project PCG calls supplied SpMV operator...\n";

		BlockSystem system = make_spd_block_system(4);
		setup_pcg_harness(system);

		int spmv_call_count = 0;
		auto spmv = make_spmv(system.A);
		get_pcg_solver()->host_solve(
			stream(),
			[&](const std::vector<float3>& input, std::vector<float3>& output)
			{
				++spmv_call_count;
				spmv(input, output);
			},
			[] { return 0.0; });

		Eigen::VectorXf x_pcg = float3_vector_to_eigen(get_host_sim_data()->sa_cgX);
		float residual = (system.A * x_pcg - system.b).norm() / std::max(system.b.norm(), 1.0f);

		std::cout << "    SpMV calls: " << spmv_call_count << "\n";
		std::cout << "    Relative residual: " << residual << "\n";

		TEST_ASSERT(spmv_call_count > 0, "Project PCG must invoke the supplied SpMV callback");
		TEST_ASSERT(residual < kResidualTolerance, "Project PCG should solve through the supplied SpMV callback");

		std::cout << "    PASSED\n";
		return true;
	}

	bool test_pcg_device_solve_matches_eigen_ldlt()
	{
		std::cout << "\n  [Test] Project device PCG vs Eigen SimplicialLDLT...\n";

		constexpr uint n = 4;
		BlockSystem system = make_spd_block_system(n);
		setup_pcg_harness(system);
		upload_pcg_harness_to_device();

		Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> ldlt;
		ldlt.compute(system.A);
		TEST_ASSERT(ldlt.info() == Eigen::Success, "Reference SPD system must factorize with Eigen LDLT");
		Eigen::VectorXf x_ref = ldlt.solve(system.b);

		std::vector<float3x3> host_blocks(n * n, luisa::make_float3x3(0.0f));
		for (uint row = 0; row < n; ++row)
		{
			for (uint col = 0; col < n; ++col)
			{
				host_blocks[row * n + col] = make_block(Eigen::Matrix3f(system.A.block(
					static_cast<int>(3 * row), static_cast<int>(3 * col), 3, 3)));
			}
		}

		auto device_blocks = device().create_buffer<float3x3>(host_blocks.size());
		stream() << device_blocks.copy_from(host_blocks.data()) << luisa::compute::synchronize();

		using namespace luisa::compute;
		auto fn_test_spmv = device().compile<1>(
			[device_blocks = device_blocks.view(), n](BufferVar<float3> input, BufferVar<float3> output)
			{
				const UInt row = dispatch_id().x;
				Float3 result = make_float3(0.0f);
				$for(col, 0u, n)
				{
					const Float3x3 block = device_blocks->read(row * n + col);
					result += block * input->read(col);
				};
				output->write(row, result);
			});

		int spmv_call_count = 0;
		get_pcg_solver()->device_solve(
			stream(),
			[&](const luisa::compute::Buffer<float3>& input, luisa::compute::Buffer<float3>& output)
			{
				++spmv_call_count;
				stream() << fn_test_spmv(input, output).dispatch(n);
			},
			[] { return 0.0; });

		Eigen::VectorXf x_pcg = float3_vector_to_eigen(get_host_sim_data()->sa_cgX);
		float residual = (system.A * x_pcg - system.b).norm() / std::max(system.b.norm(), 1.0f);
		float solution_error = (x_pcg - x_ref).norm() / std::max(x_ref.norm(), 1.0f);

		std::cout << "    Device SpMV calls: " << spmv_call_count << "\n";
		std::cout << "    Relative residual: " << residual << "\n";
		std::cout << "    Relative solution error: " << solution_error << "\n";

		TEST_ASSERT(spmv_call_count > 0, "Project device PCG must invoke the supplied SpMV callback");
		TEST_ASSERT(residual < kResidualTolerance, "Project device PCG residual should match the Eigen reference system");
		TEST_ASSERT(solution_error < kSolutionTolerance, "Project device PCG solution should match Eigen LDLT");

		std::cout << "    PASSED\n";
		return true;
	}
};

int main(int argc, char** argv)
{
	luisa::log_level_info();
	std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
	std::cout << "║  Project PCG Solver Tests                                    ║\n";
	std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";

	TestSuiteResult suite;
	suite.name = "Project PCG Solver";

	int total = 0;
	int passed = 0;

	auto run = [&](auto test_func, const char* name)
	{
		++total;
		try
		{
			TestPCGSolver test;
			if ((test.*test_func)())
			{
				++passed;
				suite.add(TestResult(true), false);
				return;
			}
		}
		catch (const std::exception& e)
		{
			std::cerr << "    FAILED: " << name << ": " << e.what() << "\n";
		}
		catch (...)
		{
			std::cerr << "    FAILED: " << name << ": unknown exception\n";
		}
		suite.add(TestResult(false), false);
	};

	run(&TestPCGSolver::test_pcg_host_solve_matches_eigen_ldlt, "Project PCG vs Eigen LDLT");
	run(&TestPCGSolver::test_pcg_matches_eigen_cg_on_same_iterations, "Project PCG vs Eigen CG");
	run(&TestPCGSolver::test_project_pcg_uses_supplied_spmv, "Project PCG uses SpMV callback");
	run(&TestPCGSolver::test_pcg_device_solve_matches_eigen_ldlt, "Project device PCG vs Eigen LDLT");

	std::cout << "\n";
	std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
	auto pass_str = std::to_string(passed);
	auto total_str = std::to_string(total);
	int	 padding = std::max(0, 25 - static_cast<int>(pass_str.size()) - static_cast<int>(total_str.size()));
	std::cout << "║  Project PCG Tests: " << passed << "/" << total << " passed" << std::string(padding, ' ') << "║\n";
	std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";

	return (passed == total) ? 0 : 1;
}
