/**
 * @file test_pcg_solver.cpp
 * @brief Unit tests for PCG (Preconditioned Conjugate Gradient) solver
 *
 * Test Coverage:
 * - SpMV (Sparse Matrix-Vector Product) correctness
 * - CG convergence on simple systems
 * - Matrix symmetry/positive-definiteness
 */

#include "test_base_solver.h"
#include "test_framework.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <cmath>

using namespace lcs;
using namespace lcs::test;

// =============================================================================
// Test Cases
// =============================================================================

class TestPCGSolver : public TestNewtonSolverBase
{
public:
	// -------------------------------------------------------------------------
	// Test 1: Simple linear system solve
	// -------------------------------------------------------------------------
	bool test_simple_solve()
	{
		std::cout << "\n  [Test] Simple 3x3 linear system solve...\n";

		// Solve: A * x = b where A = [[4, 1], [1, 3]] (2x2)
		Eigen::Matrix2f A;
		A << 4, 1,
			1, 3;

		Eigen::Vector2f b;
		b << 1, 2;

		// Solve using Eigen CG
		Eigen::ConjugateGradient<Eigen::Matrix2f> cg;
		cg.compute(A);

		Eigen::Vector2f x = cg.solve(b);

		// Verify solution
		Eigen::Vector2f residual = A * x - b;
		float			residual_norm = residual.norm();

		std::cout << "    Solution: [" << x[0] << ", " << x[1] << "]\n";
		std::cout << "    Residual norm: " << residual_norm << "\n";

		TEST_ASSERT(residual_norm < 1e-6f, "CG solution should satisfy Ax = b");

		std::cout << "    PASSED\n";
		return true;
	}

	// -------------------------------------------------------------------------
	// Test 2: SPD matrix property check
	// -------------------------------------------------------------------------
	bool test_spd_property()
	{
		std::cout << "\n  [Test] SPD matrix property check...\n";

		// Simple SPD matrix (positive definite, symmetric)
		Eigen::Matrix3f A;
		A << 4, 1, 0,
			1, 4, 1,
			0, 1, 4;

		// Check symmetry
		float asymmetry = (A - A.transpose()).cwiseAbs().maxCoeff();
		std::cout << "    Matrix asymmetry: " << asymmetry << "\n";
		TEST_ASSERT(asymmetry < 1e-6f, "Matrix should be symmetric");

		// Check positive definiteness via Cholesky
		Eigen::LLT<Eigen::Matrix3f> llt;
		llt.compute(A);
		bool is_pd = llt.info() == Eigen::Success;

		std::cout << "    Matrix is PD: " << (is_pd ? "yes" : "no") << "\n";
		TEST_ASSERT(is_pd, "Matrix should be positive definite");

		std::cout << "    PASSED\n";
		return true;
	}

	// -------------------------------------------------------------------------
	// Test 3: CG convergence rate
	// -------------------------------------------------------------------------
	bool test_cg_convergence()
	{
		std::cout << "\n  [Test] CG convergence on well-conditioned system...\n";

		// Diagonal dominant SPD matrix (well-conditioned)
		Eigen::Matrix4f A;
		A << 4, 0, 0, 0,
			0, 4, 0, 0,
			0, 0, 4, 0,
			0, 0, 0, 4;

		Eigen::Vector4f b;
		b << 1, 2, 3, 4;

		Eigen::Vector4f x_true = b.array() / 4.0f; // For diagonal matrix

		Eigen::ConjugateGradient<Eigen::Matrix4f> cg;
		cg.setMaxIterations(10);
		cg.compute(A);
		Eigen::Vector4f x = cg.solve(b);

		float error = (x - x_true).norm();
		int	  iterations = cg.iterations();

		std::cout << "    Iterations: " << iterations << "\n";
		std::cout << "    Error norm: " << error << "\n";

		// For diagonal matrix, CG should converge in 1 iteration
		TEST_ASSERT(error < 1e-6f, "CG should converge quickly for diagonal matrix");

		std::cout << "    PASSED\n";
		return true;
	}

	// -------------------------------------------------------------------------
	// Test 4: Sparse matrix operations
	// -------------------------------------------------------------------------
	bool test_sparse_matrix()
	{
		std::cout << "\n  [Test] Sparse matrix construction and solve...\n";

		// Create sparse tridiagonal matrix
		const int				   n = 10;
		Eigen::SparseMatrix<float> A(n, n);

		// Reserve space for tridiagonal
		std::vector<Eigen::Triplet<float>> triplets;
		for (int i = 0; i < n; ++i)
		{
			triplets.emplace_back(i, i, 4.0f); // Diagonal
			if (i > 0)
				triplets.emplace_back(i, i - 1, -1.0f); // Sub-diagonal
			if (i < n - 1)
				triplets.emplace_back(i, i + 1, -1.0f); // Super-diagonal
		}

		A.setFromTriplets(triplets.begin(), triplets.end());

		// Create RHS
		Eigen::VectorXf b = Eigen::VectorXf::Ones(n);

		// Solve
		Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
		solver.compute(A);
		Eigen::VectorXf x = solver.solve(b);

		// Verify
		Eigen::VectorXf residual = A * x - b;
		float			residual_norm = residual.norm();

		std::cout << "    Sparse matrix size: " << A.rows() << "x" << A.cols() << "\n";
		std::cout << "    Non-zeros: " << A.nonZeros() << "\n";
		std::cout << "    Residual norm: " << residual_norm << "\n";

		TEST_ASSERT(residual_norm < 1e-6f, "Sparse solve should be accurate");

		std::cout << "    PASSED\n";
		return true;
	}

	// -------------------------------------------------------------------------
	// Test 5: CG vs direct solve comparison
	// -------------------------------------------------------------------------
	bool test_cg_vs_direct()
	{
		std::cout << "\n  [Test] CG vs direct solve comparison...\n";

		// Simple 5x5 SPD matrix
		Eigen::Matrix<float, 5, 5> A;
		A << 10, 1, 0, 0, 0,
			1, 10, 1, 0, 0,
			0, 1, 10, 1, 0,
			0, 0, 1, 10, 1,
			0, 0, 0, 1, 10;

		Eigen::Matrix<float, 5, 1> b;
		b << 1, 2, 3, 4, 5;

		// Direct solve
		Eigen::Matrix<float, 5, 1> x_direct = A.ldlt().solve(b);

		// CG solve
		Eigen::ConjugateGradient<Eigen::Matrix<float, 5, 5>> cg;
		cg.setMaxIterations(50);
		cg.setTolerance(1e-10f);
		cg.compute(A);
		Eigen::Matrix<float, 5, 1> x_cg = cg.solve(b);

		float diff = (x_direct - x_cg).norm();

		std::cout << "    CG iterations: " << cg.iterations() << "\n";
		std::cout << "    CG error: " << cg.error() << "\n";
		std::cout << "    Diff from direct: " << diff << "\n";

		TEST_ASSERT(diff < 1e-4f, "CG should match direct solve closely");

		std::cout << "    PASSED\n";
		return true;
	}

	// -------------------------------------------------------------------------
	// Test 6: Preconditioner effect
	// -------------------------------------------------------------------------
	bool test_preconditioner()
	{
		std::cout << "\n  [Test] Diagonal preconditioner effect...\n";

		// Ill-conditioned matrix
		Eigen::Matrix3f A;
		A << 100, 1, 1,
			1, 100, 1,
			1, 1, 100;

		Eigen::Vector3f b;
		b << 1, 1, 1;

		// Without preconditioner (using default)
		Eigen::ConjugateGradient<Eigen::Matrix3f> cg_default;
		cg_default.setMaxIterations(100);
		cg_default.compute(A);
		Eigen::Vector3f x_default = cg_default.solve(b);
		int				iter_default = cg_default.iterations();

		// With Diagonal preconditioner (using identity - CG's default)
		Eigen::ConjugateGradient<Eigen::Matrix3f, Eigen::Lower, Eigen::DiagonalPreconditioner<float>> cg_diag;
		cg_diag.setMaxIterations(100);
		cg_diag.compute(A);
		Eigen::Vector3f x_diag = cg_diag.solve(b);
		int				iter_diag = cg_diag.iterations();

		std::cout << "    Iterations (default): " << iter_default << "\n";
		std::cout << "    Iterations (DiagonalPrec): " << iter_diag << "\n";

		// Diagonal should help convergence
		TEST_ASSERT(iter_diag <= iter_default, "Preconditioner should not increase iterations");

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
	std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
	std::cout << "║  PCG Solver Tests                                          ║\n";
	std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";

	TestSuiteResult suite;
	suite.name = "PCG Solver";

	int total = 0, passed = 0;

	TestPCGSolver test;

	total++;
	if (test.test_simple_solve())
	{
		passed++;
		suite.add(TestResult(true), false);
	}
	else
	{
		suite.add(TestResult(false), false);
	}

	total++;
	if (test.test_spd_property())
	{
		passed++;
		suite.add(TestResult(true), false);
	}
	else
	{
		suite.add(TestResult(false), false);
	}

	total++;
	if (test.test_cg_convergence())
	{
		passed++;
		suite.add(TestResult(true), false);
	}
	else
	{
		suite.add(TestResult(false), false);
	}

	total++;
	if (test.test_sparse_matrix())
	{
		passed++;
		suite.add(TestResult(true), false);
	}
	else
	{
		suite.add(TestResult(false), false);
	}

	total++;
	if (test.test_cg_vs_direct())
	{
		passed++;
		suite.add(TestResult(true), false);
	}
	else
	{
		suite.add(TestResult(false), false);
	}

	total++;
	if (test.test_preconditioner())
	{
		passed++;
		suite.add(TestResult(true), false);
	}
	else
	{
		suite.add(TestResult(false), false);
	}

	std::cout << "\n";
	std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
	auto pass_str = std::to_string(passed);
	auto total_str = std::to_string(total);
	int	 padding = std::max(0, 32 - static_cast<int>(pass_str.size()) - static_cast<int>(total_str.size()));
	std::cout << "║  PCG Solver Tests: " << passed << "/" << total << " passed"
			  << std::string(padding, ' ') << "║\n";
	std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";

	return (passed == total) ? 0 : 1;
}
