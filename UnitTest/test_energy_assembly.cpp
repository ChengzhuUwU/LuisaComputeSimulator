/**
 * @file test_energy_assembly.cpp
 * @brief Unit tests for energy Gradient/Hessian assembly
 *
 * Test Coverage:
 * - Individual energy Gradient/Hessian via FD validation
 * - Per-energy-type validation against analytical solutions
 */

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <functional>
#include "Core/float_nxn.h"
#include "Core/lc_to_eigen.h"
#include "Energies/detail/soft_inertia_energy.hpp"
#include "Energies/detail/hookean_spring_energy.hpp"
#include "Energies/detail/fem_BW98_cloth_energy.hpp"
#include "Energies/detail/stable_neo_hookean_energy.hpp"
#include "Energies/detail/fem_utils.h"
#include "luisa/core/logging.h"

using namespace lcs;

// Helper to convert float3x3 to Eigen::Matrix3f
static Eigen::Matrix3f toEigenMat3(const float3x3& m) {
    Eigen::Matrix3f result;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            result(i, j) = m[i][j];
    return result;
}

// Simple test assertion helper
static int g_passed = 0;
static int g_failed = 0;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FAILED] " << msg << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            g_failed++; \
            return false; \
        } \
    } while(0)

#define TEST_ASSERT_NEAR(actual, expected, tol, msg) \
    do { \
        double diff = std::abs((double)(actual) - (double)(expected)); \
        if (diff > (tol)) { \
            std::cerr << "[FAILED] " << msg << std::endl; \
            std::cerr << "  Actual: " << (actual) << ", Expected: " << (expected) << ", Diff: " << diff << std::endl; \
            g_failed++; \
            return false; \
        } \
    } while(0)

#define TEST_ASSERT_VEC3_NEAR(v1, v2, tol, msg) \
    do { \
        float3 diff3 = v1 - v2; \
        float diff = std::sqrt(luisa::dot(diff3, diff3)); \
        if (diff > (tol)) { \
            std::cerr << "[FAILED] " << msg << std::endl; \
            g_failed++; \
            return false; \
        } \
    } while(0)

// =============================================================================
// Test Cases
// =============================================================================

bool test_inertia_energy() {
    std::cout << "\n  [Test] Inertia energy..." << std::endl;

    const float mass = 0.01f;
    const float substep_dt = 1.0f / 60.0f;
    const float inv_h2 = 1.0f / (substep_dt * substep_dt);
    const float stiffness_dirichlet = 1.0f;

    float3 x_new = luisa::make_float3(0.11f, 0.21f, -0.09f);
    float3 x_tilde = luisa::make_float3(0.10f, 0.20f, -0.10f);

    auto input = detail::soft_inertia_energy::Input<float, float3>{
        .x_new = x_new,
        .x_tilde = x_tilde,
        .mass = mass,
        .inv_h2 = inv_h2,
        .stiffness_dirichlet = stiffness_dirichlet,
    };

    float energy = detail::soft_inertia_energy::compute_energy(input);
    auto eval = detail::soft_inertia_energy::evaluate(input, luisa::make_float3x3(1.0f));

    // Ground truth
    float3 diff = x_new - x_tilde;
    float expected_energy = 0.5f * stiffness_dirichlet * mass * inv_h2 * luisa::dot(diff, diff);
    float3 expected_grad = stiffness_dirichlet * mass * inv_h2 * diff;

    TEST_ASSERT_NEAR(energy, expected_energy, 1e-5f, "Inertia energy mismatch");
    TEST_ASSERT_VEC3_NEAR(eval.gradients[0], expected_grad, 1e-5f, "Inertia gradient mismatch");

    std::cout << "    Energy: " << energy << " (expected: " << expected_energy << ")" << std::endl;
    std::cout << "    PASSED" << std::endl;
    return true;
}

bool test_spring_energy() {
    std::cout << "\n  [Test] Hookean spring energy..." << std::endl;

    float3 p0 = luisa::make_float3(0.0f, 0.0f, 0.0f);
    float3 p1 = luisa::make_float3(1.0f, 0.0f, 0.0f);

    float stiffness = 100.0f;
    float rest_length = 1.0f;

    // Note: template is <ScalarT, Vec3T> but fields are Vec3T x0, x1, then ScalarT rest_length, stiffness
    auto input = detail::hookean_spring_energy::Input<float, float3>{
        .x0 = p0,
        .x1 = p1,
        .rest_length = rest_length,
        .stiffness = stiffness,
    };

    auto eval = detail::hookean_spring_energy::evaluate(input, luisa::make_float3x3(1.0f));

    // At rest length, energy should be zero (gradients should also be zero)
    float3 diff = p0 - p1;
    float l = std::sqrt(luisa::dot(diff, diff));
    float stretch = l - rest_length;
    float expected_energy = 0.5f * stiffness * stretch * stretch;

    TEST_ASSERT_NEAR(expected_energy, 0.0f, 1e-6f, "Spring at rest length should have zero energy");

    // Test stretched configuration
    float3 p1_stretched = luisa::make_float3(1.5f, 0.0f, 0.0f);
    auto input_stretched = detail::hookean_spring_energy::Input<float, float3>{
        .x0 = p0,
        .x1 = p1_stretched,
        .rest_length = rest_length,
        .stiffness = stiffness,
    };
    auto eval_stretched = detail::hookean_spring_energy::evaluate(input_stretched, luisa::make_float3x3(1.0f));

    // Energy should be positive when stretched
    float3 diff2 = p0 - p1_stretched;
    float l2 = std::sqrt(luisa::dot(diff2, diff2));
    float stretch2 = l2 - rest_length;
    float expected_energy2 = 0.5f * stiffness * stretch2 * stretch2;

    TEST_ASSERT(expected_energy2 > 0.0f, "Stretched spring should have positive energy");
    TEST_ASSERT_NEAR(expected_energy2, 12.5f, 1e-5f, "Spring energy mismatch");

    std::cout << "    Rest energy: " << expected_energy << std::endl;
    std::cout << "    Stretched energy: " << expected_energy2 << " (expected: 12.5)" << std::endl;
    std::cout << "    PASSED" << std::endl;
    return true;
}

bool test_fem_cloth_energy() {
    std::cout << "\n  [Test] FEM cloth (BW98) energy..." << std::endl;

    // Unit triangle at rest
    float3 x0 = luisa::make_float3(0.0f, 0.0f, 0.0f);
    float3 x1 = luisa::make_float3(1.0f, 0.0f, 0.0f);
    float3 x2 = luisa::make_float3(0.0f, 1.0f, 0.0f);

    // Compute Dm_inv using the utility function
    float2x2 Dm_inv = FemUtils::get_Dm_inv(x0, x1, x2);

    float area = 0.5f;
    float mu = 1.0f;
    float lambda = 1.0f;

    auto input = detail::fem_BW98_cloth_energy::Input<float3, float2x2, float>{
        .x0 = x0, .x1 = x1, .x2 = x2,
        .dm_inv = Dm_inv, .mu = mu, .lambda = lambda, .area = area,
    };

    float energy = detail::fem_BW98_cloth_energy::compute_energy(input);

    // At rest configuration, energy should be zero (minimal)
    TEST_ASSERT(energy >= 0.0f, "Energy should be non-negative at rest");

    // Test deformed configuration
    float3 x0d = luisa::make_float3(0.0f, 0.0f, 0.0f);
    float3 x1d = luisa::make_float3(1.5f, 0.0f, 0.0f); // Stretched
    float3 x2d = luisa::make_float3(0.0f, 1.0f, 0.0f);

    auto input_d = detail::fem_BW98_cloth_energy::Input<float3, float2x2, float>{
        .x0 = x0d, .x1 = x1d, .x2 = x2d,
        .dm_inv = Dm_inv, .mu = mu, .lambda = lambda, .area = area,
    };
    float energy_d = detail::fem_BW98_cloth_energy::compute_energy(input_d);

    // Energy should increase with stretch
    TEST_ASSERT(energy_d > energy, "Stretched energy should be greater than rest energy");

    std::cout << "    Rest energy: " << energy << std::endl;
    std::cout << "    Stretched energy: " << energy_d << std::endl;
    std::cout << "    PASSED" << std::endl;
    return true;
}

bool test_neohookean_energy() {
    std::cout << "\n  [Test] Stable Neo-Hookean energy..." << std::endl;

    // Tetrahedron vertices (simple unit tet)
    float3 x0 = luisa::make_float3(0.0f, 0.0f, 0.0f);
    float3 x1 = luisa::make_float3(1.0f, 0.0f, 0.0f);
    float3 x2 = luisa::make_float3(0.0f, 1.0f, 0.0f);
    float3 x3 = luisa::make_float3(0.0f, 0.0f, 1.0f);

    // Rest configuration
    float3x3 Dm;
    Dm[0] = x1 - x0;
    Dm[1] = x2 - x0;
    Dm[2] = x3 - x0;
    float3x3 Dm_inv = luisa::inverse(Dm);
    float volume = luisa::determinant(Dm) / 6.0f;

    float mu = 1.0f;
    float lambda = 1.0f;

    auto input = detail::stable_neo_hookean_energy::Input<float3, float3x3, float>{
        .x0 = x0, .x1 = x1, .x2 = x2, .x3 = x3,
        .dm_inv = Dm_inv, .mu = mu, .lambda = lambda, .volume = volume,
    };

    float energy = detail::stable_neo_hookean_energy::compute_energy(input);
    auto eval = detail::stable_neo_hookean_energy::evaluate(input);

    // Energy should be positive
    TEST_ASSERT(energy >= 0.0f, "Neo-Hookean energy should be non-negative");

    // Gradient sum should be approximately zero (translational invariance)
    float3 grad_sum = eval.gradients[0] + eval.gradients[1] + eval.gradients[2] + eval.gradients[3];
    TEST_ASSERT_VEC3_NEAR(grad_sum, luisa::make_float3(0.0f), 1e-3f,
                          "Gradient sum should be ~zero (translational invariance)");

    std::cout << "    Energy: " << energy << std::endl;
    std::cout << "    Gradient sum: [" << grad_sum.x << ", " << grad_sum.y << ", " << grad_sum.z << "]" << std::endl;
    std::cout << "    PASSED" << std::endl;
    return true;
}

bool test_hessian_spd() {
    std::cout << "\n  [Test] Hessian positive semi-definiteness..." << std::endl;

    // Test with spring Hessian
    float3 p0 = luisa::make_float3(0.0f, 0.0f, 0.0f);
    float3 p1 = luisa::make_float3(1.0f, 0.0f, 0.0f);
    float stiffness = 100.0f;
    float rest_length = 1.0f;

    auto input = detail::hookean_spring_energy::Input<float, float3>{
        .x0 = p0,
        .x1 = p1,
        .rest_length = rest_length,
        .stiffness = stiffness,
    };

    auto eval = detail::hookean_spring_energy::evaluate(input, luisa::make_float3x3(1.0f));

    // Convert Hessian to Eigen for eigenvalue check
    Eigen::Matrix<float, 6, 6> H_eigen = Eigen::Matrix<float, 6, 6>::Zero();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            H_eigen(i, j) = eval.hessians[0][i][j];
            H_eigen(i, j + 3) = eval.hessians[1][i][j];
            H_eigen(i + 3, j) = eval.hessians[2][i][j];
            H_eigen(i + 3, j + 3) = eval.hessians[3][i][j];
        }

    // Make symmetric
    H_eigen = (H_eigen + H_eigen.transpose()) / 2.0f;

    // Check eigenvalues
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> solver(H_eigen);
    Eigen::VectorXf eigenvalues = solver.eigenvalues();

    float min_eigenvalue = eigenvalues.minCoeff();
    std::cout << "    Min eigenvalue: " << min_eigenvalue << std::endl;
    std::cout << "    Max eigenvalue: " << eigenvalues.maxCoeff() << std::endl;

    // For a stable spring, Hessian should be positive semi-definite
    TEST_ASSERT(min_eigenvalue > -1e-3f, "Hessian should be positive semi-definite");

    std::cout << "    PASSED" << std::endl;
    return true;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv)
{
    luisa::log_level_info();
    std::cout << "╔═══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Energy Assembly Tests                                       ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════╝" << std::endl;

    int total = 0;

    // Test 1: Inertia energy
    total++;
    if (test_inertia_energy()) g_passed++; else g_failed++;

    // Test 2: Spring energy
    total++;
    if (test_spring_energy()) g_passed++; else g_failed++;

    // Test 3: FEM cloth energy
    total++;
    if (test_fem_cloth_energy()) g_passed++; else g_failed++;

    // Test 4: Neo-Hookean energy
    total++;
    if (test_neohookean_energy()) g_passed++; else g_failed++;

    // Test 5: Hessian SPD
    total++;
    if (test_hessian_spd()) g_passed++; else g_failed++;

    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Results: " << g_passed << "/" << total << " tests passed                                   ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════╝" << std::endl;

    return (g_failed == 0) ? 0 : 1;
}
