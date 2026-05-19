/**
 * @file test_framework.h
 * @brief Testing framework for IPC physics simulation
 *
 * Provides:
 * - TEST_ASSERT / TEST_ASSERT_NEAR / TEST_ASSERT_MATRIX_NEAR macros
 * - Finite difference gradient/hessian validation
 * - Test suite organization and reporting
 */
#pragma once

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <array>
#include <string>
#include <functional>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <luisa/luisa-compute.h>

namespace lcs::test
{

// =============================================================================
// Test Result Types
// =============================================================================

struct TestResult
{
	bool   passed = false;
	double elapsed_ms = 0.0;
	std::string message;

	TestResult() = default;
	explicit TestResult(bool p) : passed(p) {}
	TestResult(bool p, double ms) : passed(p), elapsed_ms(ms) {}
	TestResult(bool p, double ms, const std::string& msg) : passed(p), elapsed_ms(ms), message(msg) {}
};

struct TestSuiteResult
{
	std::string name;
	int			passed = 0;
	int			failed = 0;
	int			skipped = 0;
	double		total_ms = 0.0;

	void add(const TestResult& r, bool is_skipped = false)
	{
		if (is_skipped)
		{
			skipped++;
		}
		else if (r.passed)
		{
			passed++;
		}
		else
		{
			failed++;
		}
		total_ms += r.elapsed_ms;
	}
};

// =============================================================================
// Test Macros
// =============================================================================

#define TEST_ASSERT(condition, message)                                              \
	do                                                                               \
	{                                                                                \
		if (!(condition))                                                            \
		{                                                                            \
			std::cerr << "[ASSERT FAILED] " << message << " at " << __FILE__ << ":"   \
					  << __LINE__ << std::endl;                                      \
			return false;                                                            \
		}                                                                            \
	}                                                                                \
	while (0)

#define TEST_ASSERT_NEAR(actual, expected, tolerance, message)                      \
	do                                                                               \
	{                                                                                \
		double diff = std::abs((double)(actual) - (double)(expected));                \
		if (diff > (tolerance))                                                       \
		{                                                                            \
			std::cerr << "[ASSERT FAILED] " << message << std::endl;                 \
			std::cerr << "  Actual:   " << (actual) << std::endl;                    \
			std::cerr << "  Expected: " << (expected) << std::endl;                 \
			std::cerr << "  Diff:     " << diff << " > " << (tolerance) << std::endl; \
			return false;                                                            \
		}                                                                            \
	}                                                                                \
	while (0)

#define TEST_ASSERT_VEC3_NEAR(actual, expected, tolerance, message)                      \
	do                                                                                  \
	{                                                                                   \
		auto diff3 = (actual) - (expected);                                             \
		float len = luisa::length(diff3);                                               \
		if (len > (tolerance))                                                          \
		{                                                                               \
			std::cerr << "[ASSERT FAILED] " << message << std::endl;                    \
			std::cerr << "  Actual:   [" << (actual).x << ", " << (actual).y            \
					  << ", " << (actual).z << "]" << std::endl;                       \
			std::cerr << "  Expected: [" << (expected).x << ", " << (expected).y        \
					  << ", " << (expected).z << "]" << std::endl;                      \
			std::cerr << "  Diff len: " << len << " > " << (tolerance) << std::endl;     \
			return false;                                                              \
		}                                                                              \
	}                                                                                  \
	while (0)

#define TEST_ASSERT_MATRIX_NEAR(actual, expected, tolerance, message)                     \
	do                                                                                    \
	{                                                                                     \
		Eigen::MatrixXf a = (actual);                                                     \
		Eigen::MatrixXf e = (expected);                                                    \
		if (a.rows() != e.rows() || a.cols() != e.cols())                                 \
		{                                                                                 \
			std::cerr << "[ASSERT FAILED] " << message << std::endl;                      \
			std::cerr << "  Matrix dimensions mismatch: " << a.rows() << "x" << a.cols()  \
					  << " vs " << e.rows() << "x" << e.cols() << std::endl;              \
			return false;                                                                 \
		}                                                                                 \
		float diff = (a - e).cwiseAbs().maxCoeff();                                       \
		if (diff > (tolerance))                                                           \
		{                                                                                 \
			std::cerr << "[ASSERT FAILED] " << message << std::endl;                      \
			std::cerr << "  Max |diff|: " << diff << " > " << (tolerance) << std::endl;   \
			return false;                                                                 \
		}                                                                                 \
	}                                                                                    \
	while (0)

#define TEST_ASSERT_SPARSE_NEAR(actual, expected, tolerance, message)                        \
	do                                                                                         \
	{                                                                                          \
		Eigen::SparseMatrix<float> a = (actual);                                                \
		Eigen::SparseMatrix<float> e = (expected);                                              \
		if (a.rows() != e.rows() || a.cols() != e.cols())                                      \
		{                                                                                      \
			std::cerr << "[ASSERT FAILED] " << message << std::endl;                           \
			return false;                                                                     \
		}                                                                                      \
		float diff = 0.0f;                                                                     \
		for (int k = 0; k < a.outerSize(); ++k)                                                \
		{                                                                                      \
			for (Eigen::SparseMatrix<float>::InnerIterator it_a(a, k), it_e(e, k); it_a && it_e; ++it_a, ++it_e) \
			{                                                                                  \
				diff = std::max(diff, std::abs(it_a.value() - it_e.value()));                  \
			}                                                                                  \
		}                                                                                      \
		if (diff > (tolerance))                                                                \
		{                                                                                      \
			std::cerr << "[ASSERT FAILED] " << message << std::endl;                          \
			std::cerr << "  Max |diff|: " << diff << " > " << (tolerance) << std::endl;        \
			return false;                                                                     \
		}                                                                                      \
	}                                                                                          \
	while (0)

// =============================================================================
// Helper: Time Measurement
// =============================================================================

class ScopedTimer
{
public:
	explicit ScopedTimer(const std::string& name = "") : name_(name)
	{
		start_ = std::chrono::high_resolution_clock::now();
	}
	~ScopedTimer()
	{
		auto end = std::chrono::high_resolution_clock::now();
		double ms = std::chrono::duration<double, std::milli>(end - start_).count();
		if (!name_.empty())
		{
			std::cout << "    [" << name_ << "] " << std::fixed << std::setprecision(2) << ms << " ms\n";
		}
	}
	double elapsed_ms() const
	{
		auto end = std::chrono::high_resolution_clock::now();
		return std::chrono::duration<double, std::milli>(end - start_).count();
	}

private:
	std::string										 name_;
	std::chrono::high_resolution_clock::time_point start_;
};

// =============================================================================
// Finite Difference Utilities
// =============================================================================

namespace fd
{

/**
 * @brief Compute gradient via central difference
 * @tparam Scalar scalar type (float/double)
 * @tparam N dimension
 * @param func scalar function to differentiate
 * @param x input point
 * @param h step size (default 1e-6)
 * @return Eigen::Matrix<Scalar, N, 1> gradient
 */
template <typename Scalar, int N = Eigen::Dynamic>
Eigen::Matrix<Scalar, N, 1> central_difference_gradient(
	const std::function<Scalar(const Eigen::Matrix<Scalar, N, 1>&)>& func,
	const Eigen::Matrix<Scalar, N, 1>&								x,
	Scalar															h = Scalar(1e-6))
{
	int							 n = x.size();
	Eigen::Matrix<Scalar, N, 1> grad(n);
	grad.setZero();

	for (int i = 0; i < n; ++i)
	{
		Eigen::Matrix<Scalar, N, 1> x_plus = x;
		Eigen::Matrix<Scalar, N, 1> x_minus = x;
		x_plus(i) += h;
		x_minus(i) -= h;
		grad(i) = (func(x_plus) - func(x_minus)) / (Scalar(2) * h);
	}
	return grad;
}

/**
 * @brief Compute full gradient and Hessian via central difference
 * @param func scalar function
 * @param x input point
 * @param gradient output gradient
 * @param hessian output Hessian
 * @param h step size
 */
template <typename Scalar, int N = Eigen::Dynamic>
void compute_gradient_and_hessian(
	const std::function<Scalar(const Eigen::Matrix<Scalar, N, 1>&)>& func,
	const Eigen::Matrix<Scalar, N, 1>&								x,
	Eigen::Matrix<Scalar, N, 1>&									gradient,
	Eigen::Matrix<Scalar, N, N>&									hessian,
	Scalar															h = Scalar(1e-4))
{
	const int n = x.size();
	gradient.resize(n);
	hessian.resize(n, n);
	gradient.setZero();
	hessian.setZero();

	std::vector<Scalar> f_plus(n), f_minus(n);
	for (int i = 0; i < n; ++i)
	{
		Eigen::Matrix<Scalar, N, 1> x_plus = x;
		Eigen::Matrix<Scalar, N, 1> x_minus = x;
		x_plus(i) += h;
		x_minus(i) -= h;
		f_plus[i] = func(x_plus);
		f_minus[i] = func(x_minus);
		gradient(i) = (f_plus[i] - f_minus[i]) / (Scalar(2) * h);
		hessian(i, i) = (f_plus[i] - Scalar(2) * func(x) + f_minus[i]) / (h * h);
	}

	// Off-diagonal Hessian entries
	for (int i = 0; i < n; ++i)
	{
		for (int j = i + 1; j < n; ++j)
		{
			Eigen::Matrix<Scalar, N, 1> x_pp = x, x_pm = x, x_mp = x, x_mm = x;
			x_pp(i) += h;
			x_pp(j) += h;
			x_pm(i) += h;
			x_pm(j) -= h;
			x_mp(i) -= h;
			x_mp(j) += h;
			x_mm(i) -= h;
			x_mm(j) -= h;
			Scalar mixed = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (Scalar(4) * h * h);
			hessian(i, j) = mixed;
			hessian(j, i) = mixed;
		}
	}
}

/**
 * @brief Validate analytic gradient against finite difference
 * @return max absolute difference
 */
template <typename Scalar, int N = Eigen::Dynamic>
float validate_gradient(
	const std::function<Scalar(const Eigen::Matrix<Scalar, N, 1>&)>& func,
	const Eigen::Matrix<Scalar, N, 1>&								x,
	const Eigen::Matrix<Scalar, N, 1>&								analytic_grad,
	Scalar															h = Scalar(1e-4))
{
	auto numeric_grad = central_difference_gradient(func, x, h);
	float max_diff = (analytic_grad - numeric_grad).cwiseAbs().maxCoeff();
	return max_diff;
}

/**
 * @brief Validate analytic Hessian against finite difference (after PSD projection)
 * @return max absolute difference
 */
template <typename Scalar, int N = Eigen::Dynamic>
float validate_hessian(
	const std::function<Scalar(const Eigen::Matrix<Scalar, N, 1>&)>& func,
	const Eigen::Matrix<Scalar, N, 1>&								x,
	const Eigen::Matrix<Scalar, N, N>&								analytic_hess,
	Scalar															h = Scalar(1e-4))
{
	Eigen::Matrix<Scalar, N, 1> grad;
	Eigen::Matrix<Scalar, N, N> numeric_hess;
	compute_gradient_and_hessian(func, x, grad, numeric_hess, h);

	// Project to PSD for comparison (matching the test_gradient_hessian.cpp approach)
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(analytic_hess.template cast<float>());
	Eigen::VectorXf eigenvalues = solver.eigenvalues();
	Eigen::MatrixXf proj_hess = solver.eigenvectors() * eigenvalues.asDiagonal() * solver.eigenvectors().transpose();
	proj_hess = proj_hess.cast<Scalar>();

	float max_diff = (proj_hess - numeric_hess).cwiseAbs().maxCoeff();
	return max_diff;
}

} // namespace fd

// =============================================================================
// Eigen Conversions (mirrors Core/lc_to_eigen.h)
// =============================================================================

inline Eigen::Vector3f float3_to_eigen3(luisa::float3 v)
{
	return Eigen::Vector3f{v.x, v.y, v.z};
}

inline Eigen::Matrix3f float3x3_to_eigen3x3(luisa::float3x3 m)
{
	Eigen::Matrix3f e;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			e(i, j) = m[i][j];
	return e;
}

inline Eigen::VectorXf float3_vector_to_eigen(const std::vector<luisa::float3>& vec)
{
	Eigen::VectorXf e(vec.size() * 3);
	for (size_t i = 0; i < vec.size(); ++i)
	{
		e(i * 3 + 0) = vec[i].x;
		e(i * 3 + 1) = vec[i].y;
		e(i * 3 + 2) = vec[i].z;
	}
	return e;
}

inline std::vector<luisa::float3> eigen_to_float3_vector(const Eigen::VectorXf& e)
{
	std::vector<luisa::float3> vec(e.size() / 3);
	for (int i = 0; i < e.size() / 3; ++i)
	{
		vec[i] = luisa::make_float3(e(i * 3 + 0), e(i * 3 + 1), e(i * 3 + 2));
	}
	return vec;
}

// =============================================================================
// Test Printer Utilities
// =============================================================================

inline void print_separator(const std::string& title)
{
	std::cout << "\n============================================================\n";
	std::cout << title << "\n";
	std::cout << "============================================================\n";
}

inline void print_test_result(const std::string& name, bool passed, double ms, const std::string& detail = "")
{
	std::cout << "  " << std::left << std::setw(50) << name
			  << (passed ? "[PASS]" : "[FAIL]")
			  << "  " << std::fixed << std::setprecision(2) << ms << " ms\n";
	if (!detail.empty())
		std::cout << "    " << detail << "\n";
}

inline void print_suite_summary(const TestSuiteResult& suite)
{
	std::cout << "\n------------------------------------------------------------\n";
	std::cout << "Suite: " << suite.name << "\n";
	std::cout << "  Passed: " << suite.passed << "  Failed: " << suite.failed
			  << "  Skipped: " << suite.skipped << "\n";
	std::cout << "  Total time: " << std::fixed << std::setprecision(2)
			  << suite.total_ms << " ms\n";
	if (suite.failed > 0)
		std::cout << "  STATUS: FAILED\n";
	else
		std::cout << "  STATUS: ALL PASSED\n";
}

// =============================================================================
// Test Runner Helper
// =============================================================================

using TestFunc = std::function<bool()>;

inline TestResult run_test(const std::string& name, TestFunc func)
{
	std::cout << "\n  [Test] " << name << "...\n";
	auto start = std::chrono::high_resolution_clock::now();
	bool	passed = false;
	try
	{
		passed = func();
	}
	catch (const std::exception& e)
	{
		std::cerr << "  [EXCEPTION] " << e.what() << "\n";
		passed = false;
	}
	auto end = std::chrono::high_resolution_clock::now();
	double ms = std::chrono::duration<double, std::milli>(end - start).count();
	print_test_result(name, passed, ms);
	return TestResult(passed, ms);
}

// =============================================================================
// Mesh Generators (for test fixtures)
// =============================================================================

/**
 * @brief Generate a simple cloth grid mesh
 */
inline void generate_cloth_grid(
	int											 grid_size,
	std::vector<std::array<float, 3>>&			 vertices,
	std::vector<std::array<uint, 3>>&			 faces,
	float										 spacing = 0.1f)
{
	vertices.clear();
	faces.clear();

	for (int y = 0; y < grid_size; ++y)
	{
		for (int x = 0; x < grid_size; ++x)
		{
			vertices.push_back({(float) x * spacing, 0.5f, (float) y * spacing});
		}
	}

	for (int y = 0; y < grid_size - 1; ++y)
	{
		for (int x = 0; x < grid_size - 1; ++x)
		{
			uint i0 = y * grid_size + x;
			uint i1 = i0 + 1;
			uint i2 = i0 + grid_size;
			uint i3 = i2 + 1;
			faces.push_back({i0, i1, i2});
			faces.push_back({i1, i3, i2});
		}
	}
}

/**
 * @brief Generate a simple tetrahedron mesh
 */
inline void generate_tetrahedron(
	std::vector<std::array<float, 3>>&			 vertices,
	std::vector<std::array<uint, 4>>&			 tets)
{
	vertices = {
		{0.0f, 0.0f, 0.0f},
		{0.1f, 0.0f, 0.0f},
		{0.0f, 0.1f, 0.0f},
		{0.0f, 0.0f, 0.1f}};
	tets = {{0, 1, 2, 3}};
}

/**
 * @brief Generate a simple box mesh
 */
inline void generate_box(
	std::vector<std::array<float, 3>>& vertices,
	std::vector<std::array<uint, 3>>& faces,
	float							 size = 0.1f)
{
	float h = size / 2.0f;
	vertices = {
		{-h, -h, -h}, {h, -h, -h}, {h, h, -h}, {-h, h, -h},
		{-h, -h, h},  {h, -h, h},  {h, h, h},  {-h, h, h}};

	// 12 triangles (2 per face)
	faces = {
		{0, 1, 2}, {0, 2, 3},   // front
		{4, 6, 5}, {4, 7, 6},   // back
		{0, 4, 5}, {0, 5, 1},   // bottom
		{2, 6, 7}, {2, 7, 3},   // top
		{0, 3, 7}, {0, 7, 4},   // left
		{1, 5, 6}, {1, 6, 2}};  // right
}

/**
 * @brief Compute bounding box of vertices
 */
inline std::pair<luisa::float3, luisa::float3> compute_bbox(const std::vector<luisa::float3>& verts)
{
	luisa::float3 mn = verts[0], mx = verts[0];
	for (const auto& v : verts)
	{
		mn = luisa::min(mn, v);
		mx = luisa::max(mx, v);
	}
	return {mn, mx};
}

} // namespace lcs::test
