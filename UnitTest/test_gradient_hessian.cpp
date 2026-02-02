
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include "Core/float_nxn.h"
#include "Core/lc_to_eigen.h"
#include "Energy/bending_energy.h"
#include "Energy/stretch_energy.h"
#include "luisa/core/logging.h"
#include <luisa/dsl/sugar.h>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <functional>
#include <cmath>
#include <memory>
#include <unordered_map>

namespace FiniteDiff
{

template <typename Scalar>
using CacheMap = std::unordered_map<size_t, Scalar>;

/**
 * @brief Compute the gradient and hessian of a scalar function using central difference method
 * 
 * @tparam Scalar type
 * @tparam N Dimension
 * @param func Objective function
 * @param x Input point
 * @param h Difference step size
 * @param use_cache Whether to use caching
 * @return Eigen::Matrix<Scalar, N, 1> Gradient vector
 */
template <typename Scalar, int N = Eigen::Dynamic>
void computeGradientAndHessian(const std::function<Scalar(const Eigen::Matrix<Scalar, N, 1>&)>& func,
                               const Eigen::Matrix<Scalar, N, 1>&                               x,
                               Eigen::Matrix<Scalar, N, 1>&                                     gradient,
                               Eigen::Matrix<Scalar, N, N>&                                     hessian,
                               Scalar h         = Scalar(1e-6),
                               bool   use_cache = false)
{
    const int n = x.size();
    gradient.resize(n);
    hessian.resize(n, n);

    gradient.setZero();
    hessian.setZero();

    std::shared_ptr<CacheMap<Scalar>> cache;
    if (use_cache)
    {
        cache = std::make_shared<CacheMap<Scalar>>();
    }

    auto get_func_value = [&](const Eigen::Matrix<Scalar, N, 1>& point) -> Scalar
    {
        if (!use_cache)
        {
            return func(point);
        }

        size_t hash = 0;
        for (int i = 0; i < point.size(); ++i)
        {
            hash ^= std::hash<Scalar>{}(point[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }

        auto it = cache->find(hash);
        if (it != cache->end())
        {
            return it->second;
        }

        Scalar value   = func(point);
        (*cache)[hash] = value;
        return value;
    };

    Scalar f0 = get_func_value(x);

    std::vector<Scalar> f_plus_list(n);
    std::vector<Scalar> f_minus_list(n);

    for (int i = 0; i < n; ++i)
    {
        Eigen::Matrix<Scalar, N, 1> x_plus  = x;
        Eigen::Matrix<Scalar, N, 1> x_minus = x;

        x_plus(i) += h;
        x_minus(i) -= h;

        f_plus_list[i]  = get_func_value(x_plus);
        f_minus_list[i] = get_func_value(x_minus);

        gradient(i)   = (f_plus_list[i] - f_minus_list[i]) / (2 * h);
        hessian(i, i) = (f_plus_list[i] - 2 * f0 + f_minus_list[i]) / (h * h);
    }

    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            Eigen::Matrix<Scalar, N, 1> x_pp = x;
            Eigen::Matrix<Scalar, N, 1> x_pm = x;
            Eigen::Matrix<Scalar, N, 1> x_mp = x;
            Eigen::Matrix<Scalar, N, 1> x_mm = x;

            x_pp(i) += h;
            x_pp(j) += h;
            x_pm(i) += h;
            x_pm(j) -= h;
            x_mp(i) -= h;
            x_mp(j) += h;
            x_mm(i) -= h;
            x_mm(j) -= h;

            Scalar f_pp = get_func_value(x_pp);
            Scalar f_pm = get_func_value(x_pm);
            Scalar f_mp = get_func_value(x_mp);
            Scalar f_mm = get_func_value(x_mm);

            Scalar mixed = (f_pp - f_pm - f_mp + f_mm) / (4 * h * h);

            hessian(i, j) = mixed;
            hessian(j, i) = mixed;
        }
    }
}

/**
 * @brief Compute the Jacobian matrix using central difference method
 * 
 * @tparam Scalar type
 * @tparam N Input dimension
 * @tparam M Output dimension
 * @param func Objective function
 * @param x Input point
 * @param h Difference step size
 * @return Eigen::Matrix<Scalar, M, N> Jacobian matrix
 */
template <typename Scalar, int N = Eigen::Dynamic, int M = Eigen::Dynamic>
Eigen::Matrix<Scalar, M, N> computeJacobian(
    const std::function<Eigen::Matrix<Scalar, M, 1>(const Eigen::Matrix<Scalar, N, 1>&)>& func,
    const Eigen::Matrix<Scalar, N, 1>&                                                    x,
    Scalar h = Scalar(1e-6))
{
    const int                   n  = x.size();
    Eigen::Matrix<Scalar, M, 1> f0 = func(x);
    const int                   m  = f0.size();

    Eigen::Matrix<Scalar, M, N> jacobian(m, n);

    for (int j = 0; j < n; ++j)
    {
        Eigen::Matrix<Scalar, N, 1> x_plus  = x;
        Eigen::Matrix<Scalar, N, 1> x_minus = x;

        x_plus(j) += h;
        x_minus(j) -= h;

        Eigen::Matrix<Scalar, M, 1> f_plus  = func(x_plus);
        Eigen::Matrix<Scalar, M, 1> f_minus = func(x_minus);

        jacobian.col(j) = (f_plus - f_minus) / (2 * h);
    }

    return jacobian;
}

}  // namespace FiniteDiff


typedef Eigen::Matrix<float, 2, 1> Vector2;
typedef Eigen::Matrix<float, 3, 1> Vector3;
typedef Eigen::Matrix<float, 6, 1> Vector6;
typedef Eigen::Matrix<float, 3, 2> Matrix3x2;
typedef Eigen::Matrix<float, 6, 6> Matrix6x6;

luisa::float2x2 compute_Dm(const luisa::float3& X0, const luisa::float3& X1, const luisa::float3& X2)
{
    luisa::float3   r_1    = X1 - X0;
    luisa::float3   r_2    = X2 - X0;
    luisa::float3   cross  = luisa::cross(r_1, r_2);
    luisa::float3   axis_1 = luisa::normalize(r_1);
    luisa::float3   axis_2 = luisa::normalize(luisa::cross(cross, axis_1));
    luisa::float2   uv0    = luisa::float2(luisa::dot(axis_1, X0), luisa::dot(axis_2, X0));
    luisa::float2   uv1    = luisa::float2(luisa::dot(axis_1, X1), luisa::dot(axis_2, X1));
    luisa::float2   uv2    = luisa::float2(luisa::dot(axis_1, X2), luisa::dot(axis_2, X2));
    luisa::float2   duv0   = uv1 - uv0;
    luisa::float2   duv1   = uv2 - uv0;
    luisa::float2x2 Dm     = luisa::float2x2(duv0, duv1);
    return Dm;
}

auto hessian_proj_SPD(const Eigen::MatrixXf& Mat) -> Eigen::MatrixXf
{
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigen_solver(Mat);
    Eigen::VectorXf                                eigvals = eigen_solver.eigenvalues();
    Eigen::MatrixXf                                eigvecs = eigen_solver.eigenvectors();

    // Clamp negative eigenvalues to zero
    for (int i = 0; i < eigvals.size(); ++i)
    {
        if (eigvals(i) < 0.0f)
        {
            eigvals(i) = 0.0f;
        }
    }

    // Reconstruct the matrix
    Eigen::MatrixXf Mat_psd = eigvecs * eigvals.asDiagonal() * eigvecs.transpose();

    // Set small values to zero for numerical stability
    const float epsilon = 1e-6f;
    for (int i = 0; i < Mat_psd.rows(); ++i)
    {
        for (int j = 0; j < Mat_psd.cols(); ++j)
        {
            if (std::abs(Mat_psd(i, j)) < epsilon)
            {
                Mat_psd(i, j) = 0.0f;
            }
        }
    }
    return Mat_psd;
}


int main(int argc, char** argv)
{
    luisa::log_level_info();
    std::cout << "Hello, LuisaComputeSimulation!" << std::endl;

    // Init GPU system
#if defined(__APPLE__)
    std::string backend = "metal";
#else
    std::string backend = "cuda";
#endif
    const std::string            binary_path(argv[0]);
    luisa::compute::Context      context{binary_path};
    luisa::vector<luisa::string> device_names = context.backend_device_names(backend);
    if (device_names.empty())
    {
        LUISA_WARNING("No hardware device found.");
        exit(1);
    }
    for (size_t i = 0; i < device_names.size(); ++i)
    {
        LUISA_INFO("Device {}: {}", i, device_names[i]);
    }
    luisa::compute::Device device = context.create_device(backend);
    luisa::compute::Stream stream = device.create_stream(luisa::compute::StreamTag::COMPUTE);

    using namespace lcs;

    EigenFloat12x12 cgA = EigenFloat12x12::Zero();
    EigenFloat12    cgB = EigenFloat12::Zero();


    std::vector<luisa::float3> abd_q = {
        luisa::make_float3(0.0071148, 0.66069971, 0.0142296),
        luisa::make_float3(0.8271494, -0.55152995, -0.07775201),
        luisa::make_float3(0.49132671, 0.6148592, 0.61662802),
        luisa::make_float3(-0.28366761, -0.53456093, 0.79869019),
    };
    std::vector<luisa::float3> abd_q_tilde = {
        luisa::make_float3(0.0f, 0.0f, 0.0f),
        luisa::make_float3(1.0f, 0.0f, 0.0f),
        luisa::make_float3(0.0f, 1.0f, 0.0f),
        luisa::make_float3(0.0f, 0.0f, 1.0f),
    };

    // auto result = compute_ortho_gradient_hessian(abd_q, 1e5f);
    // std::cout << "Ortho gradient: " << std::endl << result.first << std::endl;
    // std::cout << "Ortho hessian: " << std::endl << result.second << std::endl;

    // Run finite-difference validation tests for several energies
    auto run_energy_fd_tests = [&](luisa::compute::Device& device, luisa::compute::Stream& stream)
    {
        using namespace lcs;
        using EigenVec = Eigen::Matrix<float, Eigen::Dynamic, 1>;

        auto print_diff = [](const std::string&     name,
                             const Eigen::VectorXf& g_num,
                             const Eigen::VectorXf& g_ana,
                             const Eigen::MatrixXf& H_num,
                             const Eigen::MatrixXf& H_ana,
                             float                  tol = 1e-3f)
        {
            std::cout << "########### Energy Test: " << name << " ###########" << std::endl;
            float g_diff = (g_num - g_ana).cwiseAbs().maxCoeff();
            float H_diff = (H_num - H_ana).cwiseAbs().maxCoeff();
            if (g_diff < tol && H_diff < tol)
            {
                std::cout << "Energy Test: " << name << " OK (within tol=" << tol << ")\n";
                return;
            }
            std::cout << "Energy Test: " << name << " MISMATCH" << std::endl;
            std::cout << "  gradient diff norm = " << g_diff << std::endl;
            std::cout << "  analytic gradient:\n" << g_ana.transpose() << std::endl;
            std::cout << "  numeric gradient:\n" << g_num.transpose() << std::endl;
            std::cout << "  Hessian diff norm = " << H_diff << std::endl;
            std::cout << "  analytic Hessian:\n" << H_ana << std::endl;
            std::cout << "  numeric Hessian:\n" << H_num << std::endl;
        };

        // finite-difference step (single-precision): avoid too-small h to reduce cancellation
        const float fd_h = 1e-2f;

        // 1) Inertia (single vertex)
        {
            const float     h                   = 0.01f;
            const float     mass                = 0.01f;
            const float     substep_dt          = 0.02f;
            const float     stiffness_dirichlet = 1.0f;
            const bool      is_fixed            = false;
            Eigen::Vector3f x_tilde;

            std::function<float(const EigenVec&)> inertia_func = [&](const EigenVec& xv) -> float
            {
                Eigen::Vector3f x(xv[0], xv[1], xv[2]);
                Eigen::Vector3f diff = x - x_tilde;
                float           inv2 = 1.0f / (substep_dt * substep_dt);
                float           e    = inv2 * diff.squaredNorm() * mass / 2.0f;
                // if (is_fixed)
                //     e *= stiffness_dirichlet;
                return e;
            };

            EigenVec x0(3);
            x0 << 0.12f, 0.22f, -0.08f;
            x_tilde << 0.1f, 0.2f, -0.1f;
            Eigen::VectorXf g_num;
            Eigen::MatrixXf H_num;
            FiniteDiff::computeGradientAndHessian<float, Eigen::Dynamic>(inertia_func, x0, g_num, H_num, fd_h, false);

            // analytic
            Eigen::VectorXf g_ana(3);
            Eigen::MatrixXf H_ana = Eigen::MatrixXf::Zero(3, 3);
            Eigen::Vector3f diff  = x0 - x_tilde;
            float           inv2  = 1.0f / (substep_dt * substep_dt);
            g_ana                 = (mass * inv2) * diff;
            // if (is_fixed)
            // {
            //     g_ana *= stiffness_dirichlet;
            //     H_ana = stiffness_dirichlet * mass * inv2 * Eigen::Matrix3f::Identity();
            // }
            // else
            H_ana = mass * inv2 * Eigen::Matrix3f::Identity();

            print_diff("Inertia", g_num, g_ana, H_num, H_ana);
        }

        // 2) Stretch Spring (edge)
        {
            luisa::float3 v0 = luisa::make_float3(0.0f, 0.0f, 0.0f);
            luisa::float3 v1 = luisa::make_float3(1.2f, 0.3f, -0.1f);
            const float   L0 = 1.0f;
            const float   k  = 2.0f;

            std::function<float(const EigenVec&)> spring_func = [&](const EigenVec& xv) -> float
            {
                Eigen::Vector3f a(xv[0], xv[1], xv[2]);
                Eigen::Vector3f b(xv[3], xv[4], xv[5]);
                Eigen::Vector3f diff = a - b;
                float           l    = std::max(diff.norm(), 1e-8f);
                float           C    = l - L0;
                return 0.5f * k * C * C;
            };

            EigenVec x0(6);
            x0 << v0.x, v0.y, v0.z, v1.x, v1.y, v1.z;
            Eigen::VectorXf g_num;
            Eigen::MatrixXf H_num;
            FiniteDiff::computeGradientAndHessian<float, Eigen::Dynamic>(spring_func, x0, g_num, H_num, fd_h, false);
            H_num = hessian_proj_SPD(H_num);

            // analytic
            Eigen::VectorXf g_ana(6);
            Eigen::MatrixXf H_ana = Eigen::MatrixXf::Zero(6, 6);
            Eigen::Vector3f a(x0[0], x0[1], x0[2]);
            Eigen::Vector3f b(x0[3], x0[4], x0[5]);
            Eigen::Vector3f diff  = a - b;
            float           l     = std::max(diff.norm(), 1e-8f);
            Eigen::Vector3f dir   = diff / l;
            float           C     = l - L0;
            Eigen::Matrix3f nnT   = dir * dir.transpose();
            float           coeff = k;
            Eigen::Matrix3f He =
                coeff * nnT + coeff * std::max(1.0f - L0 / l, 0.0f) * (Eigen::Matrix3f::Identity() - nnT);
            Eigen::Vector3f g0      = coeff * dir * C;
            Eigen::Vector3f g1      = -g0;
            g_ana.segment<3>(0)     = g0;
            g_ana.segment<3>(3)     = g1;
            H_ana.block<3, 3>(0, 0) = He;
            H_ana.block<3, 3>(0, 3) = -He;
            H_ana.block<3, 3>(3, 0) = -He;
            H_ana.block<3, 3>(3, 3) = He;

            print_diff("StretchSpring", g_num, g_ana, H_num, H_ana);
        }

        // 2a) Stretch component of StretchFace (triangle)
        {
            luisa::float3 x0     = luisa::make_float3(0.0f, 0.0f, 0.0f);
            luisa::float3 x1     = luisa::make_float3(2.0f, 0.0f, 0.0f);
            luisa::float3 x2     = luisa::make_float3(1.0f, 2.0f, 0.0f);
            luisa::float3 X0     = luisa::make_float3(0.0f, 0.0f, 0.0f);
            luisa::float3 X1     = luisa::make_float3(1.0f, 0.0f, 0.0f);
            luisa::float3 X2     = luisa::make_float3(0.0f, 1.0f, 0.0f);
            float2x2      Dm     = compute_Dm(X0, X1, X2);
            float2x2      Dm_inv = luisa::inverse(Dm);

            const float area = 1.0f;
            // auto [mu_tmp, lambda_tmp] = StretchEnergy::convert_prop(1e4f, 0.2f);
            // const float mu            = mu_tmp;
            // const float lambda        = lambda_tmp;
            const float mu     = 1.0f;
            const float lambda = 1.0f;

            std::function<float(const EigenVec&)> stretch_only = [&](const EigenVec& xv) -> float
            {
                luisa::float3 vx0 = luisa::make_float3(xv[0], xv[1], xv[2]);
                luisa::float3 vx1 = luisa::make_float3(xv[3], xv[4], xv[5]);
                luisa::float3 vx2 = luisa::make_float3(xv[6], xv[7], xv[8]);
                lcs::float2x3 F   = makeFloat2x3(vx1 - vx0, vx2 - vx0) * Dm_inv;
                return area * StretchEnergy::detail::stretch_energy(F, mu);
            };

            EigenVec xvec(9);
            xvec << x0.x, x0.y, x0.z, x1.x, x1.y, x1.z, x2.x, x2.y, x2.z;
            Eigen::VectorXf g_num;
            Eigen::MatrixXf H_num;
            FiniteDiff::computeGradientAndHessian<float, Eigen::Dynamic>(stretch_only, xvec, g_num, H_num, fd_h, false);
            H_num = hessian_proj_SPD(H_num);

            // analytic stretch-only
            lcs::float2x3 F =
                makeFloat2x3(luisa::make_float3(xvec[3] - xvec[0], xvec[4] - xvec[1], xvec[5] - xvec[2]),
                             luisa::make_float3(xvec[6] - xvec[0], xvec[7] - xvec[1], xvec[8] - xvec[2]))
                * Dm_inv;
            auto     dedF   = StretchEnergy::detail::stretch_gradient(F, mu);
            auto     d2eF   = StretchEnergy::detail::stretch_hessian(F, mu);
            float3x3 dedx   = area * StretchEnergy::detail::convert_force(dedF, Dm_inv);
            float9x9 d2edx2 = area * StretchEnergy::detail::convert_hessian(d2eF, Dm_inv);

            Eigen::VectorXf g_ana(9);
            Eigen::MatrixXf H_ana = d2edx2.to_eigen_matrix();
            for (int i = 0; i < 3; i++)
                g_ana.segment<3>(3 * i) = float3_to_eigen3(dedx[i]);

            print_diff("StretchFace_Stretch", g_num, g_ana, H_num, H_ana, 1e-2);
        }

        // 2b) Shear component of StretchFace (triangle)
        {
            luisa::float3   x0     = luisa::make_float3(0.0f, 0.0f, 0.0f);
            luisa::float3   x1     = luisa::make_float3(1.0f, 0.0f, 0.0f);
            luisa::float3   x2     = luisa::make_float3(0.0f, 1.0f, 0.0f);
            luisa::float3   X0     = luisa::make_float3(0.0f, 0.0f, 0.0f);
            luisa::float3   X1     = luisa::make_float3(0.0f, 1.0f, 0.0f);
            luisa::float3   X2     = luisa::make_float3(1.0f, 0.1f, 0.0f);
            luisa::float2x2 Dm     = compute_Dm(X0, X1, X2);
            luisa::float2x2 Dm_inv = luisa::inverse(Dm);

            const float area = 1.0f;
            // auto [mu_tmp, lambda_tmp] = StretchEnergy::convert_prop(1e4f, 0.2f);
            // const float mu            = mu_tmp;
            // const float lambda        = lambda_tmp;
            const float mu     = 1.0f;
            const float lambda = 1.0f;

            std::function<float(const EigenVec&)> shear_only = [&](const EigenVec& xv) -> float
            {
                luisa::float3 vx0 = luisa::make_float3(xv[0], xv[1], xv[2]);
                luisa::float3 vx1 = luisa::make_float3(xv[3], xv[4], xv[5]);
                luisa::float3 vx2 = luisa::make_float3(xv[6], xv[7], xv[8]);
                lcs::float2x3 F   = makeFloat2x3(vx1 - vx0, vx2 - vx0) * Dm_inv;
                return area * StretchEnergy::detail::shear_energy(F, lambda);
            };

            EigenVec xvec(9);
            xvec << x0.x, x0.y, x0.z, x1.x, x1.y, x1.z, x2.x, x2.y, x2.z;
            Eigen::VectorXf g_num;
            Eigen::MatrixXf H_num;
            FiniteDiff::computeGradientAndHessian<float, Eigen::Dynamic>(shear_only, xvec, g_num, H_num, fd_h, false);
            H_num = hessian_proj_SPD(H_num);

            // analytic shear-only
            lcs::float2x3 Fsh =
                makeFloat2x3(luisa::make_float3(xvec[3] - xvec[0], xvec[4] - xvec[1], xvec[5] - xvec[2]),
                             luisa::make_float3(xvec[6] - xvec[0], xvec[7] - xvec[1], xvec[8] - xvec[2]))
                * Dm_inv;
            auto     dedF_sh   = StretchEnergy::detail::shear_gradient(Fsh, lambda);
            auto     d2eF_sh   = StretchEnergy::detail::shear_hessian(Fsh, lambda);
            float3x3 dedx_sh   = area * StretchEnergy::detail::convert_force(dedF_sh, Dm_inv);
            float9x9 d2edx2_sh = area * StretchEnergy::detail::convert_hessian(d2eF_sh, Dm_inv);

            Eigen::VectorXf g_ana_sh(9);
            Eigen::MatrixXf H_ana_sh = d2edx2_sh.to_eigen_matrix();
            for (int i = 0; i < 3; i++)
                g_ana_sh.segment<3>(3 * i) = float3_to_eigen3(dedx_sh[i]);

            print_diff("StretchFace_Shear", g_num, g_ana_sh, H_num, H_ana_sh, 1e-2f);
        }


        // 4) Bending (hinge of 4 verts)
        {
            luisa::float3 q0    = luisa::make_float3(0.0f, 0.0f, 0.0f);
            luisa::float3 q1    = luisa::make_float3(1.0f, 0.0f, 0.0f);
            luisa::float3 q2    = luisa::make_float3(0.0f, 1.0f, 0.0f);
            luisa::float3 q3    = luisa::make_float3(0.0f, 0.0f, 1.0f);
            float         area  = 1.0f;
            float         stiff = 10.0f;

            std::function<float(const EigenVec&)> bend_func = [&](const EigenVec& xv) -> float
            {
                luisa::float3 v0         = luisa::make_float3(xv[0], xv[1], xv[2]);
                luisa::float3 v1         = luisa::make_float3(xv[3], xv[4], xv[5]);
                luisa::float3 v2         = luisa::make_float3(xv[6], xv[7], xv[8]);
                luisa::float3 v3         = luisa::make_float3(xv[9], xv[10], xv[11]);
                float         angle      = BendingEnergy::compute_theta(v0, v1, v2, v3);
                float         rest_angle = 0.0f;
                float         delta      = angle - rest_angle;
                return 0.5f * stiff * area * delta * delta;
            };

            EigenVec x0(12);
            x0 << q0.x, q0.y, q0.z, q1.x, q1.y, q1.z, q2.x, q2.y, q2.z, q3.x, q3.y, q3.z;
            Eigen::VectorXf g_num;
            Eigen::MatrixXf H_num;
            FiniteDiff::computeGradientAndHessian<float, Eigen::Dynamic>(bend_func, x0, g_num, H_num, fd_h, false);

            // analytic as implemented in host: use compute_d_theta_d_x and outer product approx
            float3 grad_arr[4];
            float  angle = BendingEnergy::compute_d_theta_d_x(luisa::make_float3(x0[0], x0[1], x0[2]),
                                                             luisa::make_float3(x0[3], x0[4], x0[5]),
                                                             luisa::make_float3(x0[6], x0[7], x0[8]),
                                                             luisa::make_float3(x0[9], x0[10], x0[11]),
                                                             grad_arr);
            float  rest_angle = 0.0f;
            float  delta      = angle - rest_angle;
            Eigen::VectorXf g_ana(12);
            Eigen::MatrixXf H_ana = Eigen::MatrixXf::Zero(12, 12);
            for (int ii = 0; ii < 4; ++ii)
            {
                Eigen::Vector3f gi       = float3_to_eigen3(grad_arr[ii]);
                g_ana.segment<3>(3 * ii) = stiff * delta * gi;  // host subtracts into cgB
                for (int jj = 0; jj < 4; ++jj)
                {
                    Eigen::Vector3f gj                = float3_to_eigen3(grad_arr[jj]);
                    H_ana.block<3, 3>(3 * ii, 3 * jj) = stiff * gi * gj.transpose();
                }
            }

            print_diff("Bending", g_num, g_ana, H_num, H_ana);
        }
        return;
        // 5) Ground collision (simple quadratic)
        {
            const float                           floor_y     = 0.0f;
            const float                           thickness   = 0.0f;
            const float                           d_hat       = 0.1f;
            const float                           area        = 1.0f;
            const float                           stiff       = 100.0f * area;
            std::function<float(const EigenVec&)> ground_func = [&](const EigenVec& xv) -> float
            {
                Eigen::Vector3f x(xv[0], xv[1], xv[2]);
                float           dist = x[1] - floor_y;
                if (dist - thickness < d_hat)
                {
                    float C = d_hat + thickness - dist;
                    return 0.5f * stiff * C * C;
                }
                return 0.0f;
            };

            EigenVec x0(3);
            x0 << 0.0f, -0.05f, 0.0f;  // slightly penetrating
            Eigen::VectorXf g_num;
            Eigen::MatrixXf H_num;
            FiniteDiff::computeGradientAndHessian<float, Eigen::Dynamic>(ground_func, x0, g_num, H_num, fd_h, false);

            Eigen::VectorXf g_ana(3);
            Eigen::MatrixXf H_ana = Eigen::MatrixXf::Zero(3, 3);
            float           dist  = x0[1] - floor_y;
            if (dist - thickness < d_hat)
            {
                float           C  = d_hat + thickness - dist;
                float           k1 = stiff * C;
                float           k2 = stiff;
                Eigen::Vector3f normal(0.0f, 1.0f, 0.0f);
                g_ana = -k1 * normal;
                H_ana = k2 * (normal * normal.transpose());
            }

            print_diff("GroundCollision", g_num, g_ana, H_num, H_ana);
        }
    };

    run_energy_fd_tests(device, stream);

    // float6 vec1;
    // vec1.vec[0] = luisa::make_float3(0, 1, 2);
    // vec1.vec[1] = luisa::make_float3(3, 4, 5);

    // float6 vec2;
    // vec2.vec[0] = luisa::make_float3(0, 1, 100);
    // vec2.vec[1] = luisa::make_float3(1000, 10000, 100000);
    // std::cout << "Val 1 = \n" << float6x6::outer_product(vec1, vec2).to_eigen_matrix() << std::endl;
    // std::cout << "Val 2 = \n"
    //           << vec1.to_eigen_matrix() * vec2.to_eigen_matrix().transpose() << std::endl;
    // std::cout << "Val 1 = \n" << float6x6::outer_product(vec2, vec1).to_eigen_matrix() << std::endl;
    // std::cout << "Val 2 = \n"
    //           << vec2.to_eigen_matrix() * vec1.to_eigen_matrix().transpose() << std::endl;
}