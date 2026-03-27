
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include "Core/float_nxn.h"
#include "Core/lc_to_eigen.h"
#include "Energies/detail/soft_inertia_energy.hpp"
#include "Energies/detail/hookean_spring_energy.hpp"
#include "Energies/detail/fem_BW98_cloth_energy.hpp"
#include "Energies/detail/arap_tet_energy.hpp"
#include "Energies/detail/bending_energy.hpp"
#include "Energies/detail/abd_inertia_energy.hpp"
#include "Energies/detail/abd_ortho_energy.hpp"
#include "Energies/detail/fixed_joint_constaint.hpp"
#include "Energies/detail/prismatic_joint_constaint.hpp"
#include "Energies/detail/revolute_joint_constaint.hpp"
#include "Energies/detail/stable_neo_hookean_energy.hpp"
#include "Energies/detail/ground_collision_energy.hpp"
#include "luisa/core/logging.h"
#include <luisa/dsl/sugar.h>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <functional>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <iomanip>

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
		const Eigen::Matrix<Scalar, N, 1>&															x,
		Eigen::Matrix<Scalar, N, 1>&																gradient,
		Eigen::Matrix<Scalar, N, N>&																hessian,
		Scalar																						h = Scalar(1e-6),
		bool																						use_cache = false)
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

			Scalar value = func(point);
			(*cache)[hash] = value;
			return value;
		};

		Scalar f0 = get_func_value(x);

		std::vector<Scalar> f_plus_list(n);
		std::vector<Scalar> f_minus_list(n);

		for (int i = 0; i < n; ++i)
		{
			Eigen::Matrix<Scalar, N, 1> x_plus = x;
			Eigen::Matrix<Scalar, N, 1> x_minus = x;

			x_plus(i) += h;
			x_minus(i) -= h;

			f_plus_list[i] = get_func_value(x_plus);
			f_minus_list[i] = get_func_value(x_minus);

			gradient(i) = (f_plus_list[i] - f_minus_list[i]) / (2 * h);
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
		const Eigen::Matrix<Scalar, N, 1>&													  x,
		Scalar																				  h = Scalar(1e-6))
	{
		const int					n = x.size();
		Eigen::Matrix<Scalar, M, 1> f0 = func(x);
		const int					m = f0.size();

		Eigen::Matrix<Scalar, M, N> jacobian(m, n);

		for (int j = 0; j < n; ++j)
		{
			Eigen::Matrix<Scalar, N, 1> x_plus = x;
			Eigen::Matrix<Scalar, N, 1> x_minus = x;

			x_plus(j) += h;
			x_minus(j) -= h;

			Eigen::Matrix<Scalar, M, 1> f_plus = func(x_plus);
			Eigen::Matrix<Scalar, M, 1> f_minus = func(x_minus);

			jacobian.col(j) = (f_plus - f_minus) / (2 * h);
		}

		return jacobian;
	}

} // namespace FiniteDiff

typedef Eigen::Matrix<float, 2, 1> Vector2;
typedef Eigen::Matrix<float, 3, 1> Vector3;
typedef Eigen::Matrix<float, 6, 1> Vector6;
typedef Eigen::Matrix<float, 3, 2> Matrix3x2;
typedef Eigen::Matrix<float, 6, 6> Matrix6x6;

luisa::float2x2 compute_Dm(const luisa::float3& X0, const luisa::float3& X1, const luisa::float3& X2)
{
	luisa::float3	r_1 = X1 - X0;
	luisa::float3	r_2 = X2 - X0;
	luisa::float3	cross = luisa::cross(r_1, r_2);
	luisa::float3	axis_1 = luisa::normalize(r_1);
	luisa::float3	axis_2 = luisa::normalize(luisa::cross(cross, axis_1));
	luisa::float2	uv0 = luisa::float2(luisa::dot(axis_1, X0), luisa::dot(axis_2, X0));
	luisa::float2	uv1 = luisa::float2(luisa::dot(axis_1, X1), luisa::dot(axis_2, X1));
	luisa::float2	uv2 = luisa::float2(luisa::dot(axis_1, X2), luisa::dot(axis_2, X2));
	luisa::float2	duv0 = uv1 - uv0;
	luisa::float2	duv1 = uv2 - uv0;
	luisa::float2x2 Dm = luisa::float2x2(duv0, duv1);
	return Dm;
}

auto hessian_proj_SPD(const Eigen::MatrixXf& Mat) -> Eigen::MatrixXf
{
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigen_solver(Mat);
	Eigen::VectorXf								   eigvals = eigen_solver.eigenvalues();
	Eigen::MatrixXf								   eigvecs = eigen_solver.eigenvectors();

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
	const std::string			 binary_path(argv[0]);
	luisa::compute::Context		 context{ binary_path };
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
	EigenFloat12	cgB = EigenFloat12::Zero();

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

		auto print_diff = [](const std::string&		 name,
							  const Eigen::VectorXf& g_num,
							  const Eigen::VectorXf& g_ana,
							  const Eigen::MatrixXf& H_num,
							  const Eigen::MatrixXf& H_ana,
							  float					 tol = 1e-3f)
		{
			const Eigen::IOFormat vec_fmt(3, 0, ", ", ", ", "[", "]");
			const Eigen::IOFormat mat_fmt(3, 0, ", ", "\n", "[", "]");
			Eigen::Index		  g_idx = 0;
			Eigen::Index		  h_row = 0;
			Eigen::Index		  h_col = 0;
			float				  g_diff = (g_num - g_ana).cwiseAbs().maxCoeff(&g_idx);
			float				  H_diff = (H_num - H_ana).cwiseAbs().maxCoeff(&h_row, &h_col);

			if (g_diff < tol && H_diff < tol)
			{
				std::cout << "Energy FD Check: " << std::format("{:20}", name) << " : PASS"
						  << "  (max |grad diff|=" << std::format("{:.3e}", g_diff)
						  << ", max |hess diff|=" << std::format("{:.3e}", H_diff)
						  << ", tol=" << tol << ")\n";
				return;
			}
			std::cout << "\n============================================================\n";
			std::cout << "Energy FD Check: " << name << "\n";
			std::cout << "------------------------------------------------------------\n";
			std::cout << std::left << std::setw(26) << "Max |grad diff|"
					  << ": " << g_diff
					  << "  (idx=" << g_idx << ")\n";
			std::cout << std::left << std::setw(26) << "Max |hessian diff|"
					  << ": " << H_diff
					  << "  (row=" << h_row << ", col=" << h_col << ")\n";
			std::cout << std::left << std::setw(26) << "Tolerance"
					  << ": " << tol << "\n";
			std::cout << "Result" << std::setw(21) << ""
					  << ": FAIL\n";
			if (g_diff >= tol)
			{
				std::cout << "Analytic gradient : " << g_ana.transpose().format(vec_fmt) << "\n";
				std::cout << "Numeric  gradient : " << g_num.transpose().format(vec_fmt) << "\n";
				for (int i = 0; i < g_num.size(); ++i)
				{
					if (std::abs(g_num[i] - g_ana[i]) > tol)
					{
						std::cout << "  idx=" << i
								  << ", grad_ana=" << g_ana[i]
								  << ", grad_num=" << g_num[i]
								  << ", diff=" << std::abs(g_num[i] - g_ana[i]) << "\n";
					}
				}
			}
			if (H_diff >= tol)
			{
				// std::cout << "Analytic hessian:\n"
				// 		  << H_ana.format(mat_fmt) << "\n";
				// std::cout << "Numeric  hessian:\n"
				// 		  << H_num.format(mat_fmt) << "\n";
				for (int i = 0; i < H_num.rows(); ++i)
				{
					for (int j = 0; j < H_num.cols(); ++j)
					{
						if (std::abs(H_num(i, j) - H_ana(i, j)) > tol)
						{
							std::cout << "  idx=(" << i << ", " << j << ")"
									  << ", hess_ana=" << H_ana(i, j)
									  << ", hess_num=" << H_num(i, j)
									  << ", diff=" << std::abs(H_num(i, j) - H_ana(i, j)) << "\n";
						}
					}
				}
			}
		};

		auto print_grad_only = [](const std::string&	  name,
								   const Eigen::VectorXf& g_num,
								   const Eigen::VectorXf& g_ana,
								   float				  tol = 1e-3f)
		{
			const Eigen::IOFormat vec_fmt(6, 0, ", ", ", ", "[", "]");
			Eigen::Index		  g_idx = 0;
			float				  g_diff = (g_num - g_ana).cwiseAbs().maxCoeff(&g_idx);

			if (g_diff < tol)
			{
				std::cout << "Energy FD Check: " << name << " (Gradient Only) : PASS"
						  << "  (max |grad diff|=" << g_diff << ", tol=" << tol << ")\n";
				return;
			}
			std::cout << "\n============================================================\n";
			std::cout << "Energy FD Check: " << name << " (Gradient Only)\n";
			std::cout << "------------------------------------------------------------\n";
			std::cout << std::left << std::setw(26) << "Max |grad diff|"
					  << ": " << g_diff
					  << "  (idx=" << g_idx << ")\n";
			std::cout << std::left << std::setw(26) << "Tolerance"
					  << ": " << tol << "\n";
			std::cout << "Result" << std::setw(21) << ""
					  << ": FAIL\n";
			std::cout << "Analytic gradient : " << g_ana.transpose().format(vec_fmt) << "\n";
			std::cout << "Numeric  gradient : " << g_num.transpose().format(vec_fmt) << "\n";
		};

		// finite-difference step (single-precision): avoid too-small h to reduce cancellation
		const float fd_h = 1e-2f;

		// 1) Inertia (single vertex)
		{
			const float		mass = 0.01f;
			const float		substep_dt = 0.02f;
			const float		stiffness_dirichlet = 1.0f;
			Eigen::Vector3f x_tilde;

			std::function<float(const EigenVec&)> inertia_func = [&](const EigenVec& xv) -> float
			{
				const auto input = detail::soft_inertia_energy::Input<float, float3>{
					.x_new = luisa::make_float3(xv[0], xv[1], xv[2]),
					.x_tilde = luisa::make_float3(x_tilde[0], x_tilde[1], x_tilde[2]),
					.mass = mass,
					.inv_h2 = 1.0f / (substep_dt * substep_dt),
					.stiffness_dirichlet = stiffness_dirichlet,
				};
				return detail::soft_inertia_energy::compute_energy(input);
			};

			EigenVec x0(3);
			x0 << 0.12f, 0.22f, -0.08f;
			x_tilde << 0.1f, 0.2f, -0.1f;
			Eigen::VectorXf g_num;
			Eigen::MatrixXf H_num;
			FiniteDiff::computeGradientAndHessian<float, Eigen::Dynamic>(inertia_func, x0, g_num, H_num, fd_h, false);

			// analytic (detail implementation)
			Eigen::VectorXf g_ana(3);
			const auto		input = detail::soft_inertia_energy::Input<float, float3>{
					 .x_new = luisa::make_float3(x0[0], x0[1], x0[2]),
					 .x_tilde = luisa::make_float3(x_tilde[0], x_tilde[1], x_tilde[2]),
					 .mass = mass,
					 .inv_h2 = 1.0f / (substep_dt * substep_dt),
					 .stiffness_dirichlet = stiffness_dirichlet,
			};
			auto eval = detail::soft_inertia_energy::evaluate(input, luisa::make_float3x3(1.0f));
			g_ana = float3_to_eigen3(eval.gradients[0]);
			Eigen::MatrixXf H_ana = float3x3_to_eigen3x3(eval.hessians[0]);

			print_diff("Inertia", g_num, g_ana, H_num, H_ana);
		}

		// 2) Stretch Spring (edge)
		{
			luisa::float3 v0 = luisa::make_float3(0.0f, 0.0f, 0.0f);
			luisa::float3 v1 = luisa::make_float3(1.2f, 0.3f, -0.1f);
			const float	  L0 = 1.0f;
			const float	  k = 2.0f;

			std::function<float(const EigenVec&)> spring_func = [&](const EigenVec& xv) -> float
			{
				Eigen::Vector3f a(xv[0], xv[1], xv[2]);
				Eigen::Vector3f b(xv[3], xv[4], xv[5]);
				Eigen::Vector3f diff = a - b;
				float			l = std::max(diff.norm(), 1e-8f);
				float			C = l - L0;
				return detail::hookean_spring_energy::compute_energy(k, C);
			};

			EigenVec x0(6);
			x0 << v0.x, v0.y, v0.z, v1.x, v1.y, v1.z;
			Eigen::VectorXf g_num;
			Eigen::MatrixXf H_num;
			FiniteDiff::computeGradientAndHessian<float, Eigen::Dynamic>(spring_func, x0, g_num, H_num, fd_h, false);
			H_num = hessian_proj_SPD(H_num);

			// analytic (detail implementation)
			Eigen::VectorXf g_ana(6);
			Eigen::MatrixXf H_ana = Eigen::MatrixXf::Zero(6, 6);
			Eigen::Vector3f a(x0[0], x0[1], x0[2]);
			Eigen::Vector3f b(x0[3], x0[4], x0[5]);

			auto eval = detail::hookean_spring_energy::evaluate(
				detail::hookean_spring_energy::Input<float, float3>{
					.x0 = luisa::make_float3(a[0], a[1], a[2]),
					.x1 = luisa::make_float3(b[0], b[1], b[2]),
					.rest_length = L0,
					.stiffness = k,
				},
				luisa::make_float3x3(1.0f));

			g_ana.segment<3>(0) = float3_to_eigen3(eval.gradients[0]);
			g_ana.segment<3>(3) = float3_to_eigen3(eval.gradients[1]);
			H_ana.block<3, 3>(0, 0) = float3x3_to_eigen3x3(eval.hessians[0]);
			H_ana.block<3, 3>(0, 3) = float3x3_to_eigen3x3(eval.hessians[1]);
			H_ana.block<3, 3>(3, 0) = float3x3_to_eigen3x3(eval.hessians[2]);
			H_ana.block<3, 3>(3, 3) = float3x3_to_eigen3x3(eval.hessians[3]);

			print_diff("StretchSpring", g_num, g_ana, H_num, H_ana);
		}

		// 2a) Stretch component of StretchFace (triangle)
		{
			luisa::float3 x0 = luisa::make_float3(0.0f, 0.0f, 0.0f);
			luisa::float3 x1 = luisa::make_float3(2.0f, 0.0f, 0.0f);
			luisa::float3 x2 = luisa::make_float3(1.0f, 2.0f, 0.0f);
			luisa::float3 X0 = luisa::make_float3(0.0f, 0.0f, 0.0f);
			luisa::float3 X1 = luisa::make_float3(1.0f, 0.0f, 0.0f);
			luisa::float3 X2 = luisa::make_float3(0.0f, 1.0f, 0.0f);
			float2x2	  Dm = compute_Dm(X0, X1, X2);
			float2x2	  Dm_inv = luisa::inverse(Dm);

			const float area = 1.0f;
			// auto [mu_tmp, lambda_tmp] = StretchEnergy::convert_prop(1e4f, 0.2f);
			// const float mu            = mu_tmp;
			// const float lambda        = lambda_tmp;
			const float mu = 1.0f;
			const float lambda = 1.0f;

			std::function<float(const EigenVec&)> stretch_only = [&](const EigenVec& xv) -> float
			{
				luisa::float3 vx0 = luisa::make_float3(xv[0], xv[1], xv[2]);
				luisa::float3 vx1 = luisa::make_float3(xv[3], xv[4], xv[5]);
				luisa::float3 vx2 = luisa::make_float3(xv[6], xv[7], xv[8]);
				lcs::float2x3 F = makeFloat2x3(vx1 - vx0, vx2 - vx0) * Dm_inv;
				return area * detail::fem_BW98_cloth_energy::stretch_energy(F, mu);
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
			auto	 dedF = detail::fem_BW98_cloth_energy::stretch_gradient(F, mu);
			auto	 d2eF = detail::fem_BW98_cloth_energy::stretch_hessian(F, mu);
			float3x3 dedx = area * FemUtils::convert_force(dedF, Dm_inv);
			float9x9 d2edx2 = area * FemUtils::convert_hessian(d2eF, Dm_inv);

			Eigen::VectorXf g_ana(9);
			Eigen::MatrixXf H_ana = d2edx2.to_eigen_matrix();
			for (int i = 0; i < 3; i++)
				g_ana.segment<3>(3 * i) = float3_to_eigen3(dedx[i]);

			print_diff("StretchFace_Stretch", g_num, g_ana, H_num, H_ana, 1e-2f);
		}

		// 2b) Shear component of StretchFace (triangle)
		{
			luisa::float3	x0 = luisa::make_float3(0.0f, 0.0f, 0.0f);
			luisa::float3	x1 = luisa::make_float3(1.0f, 0.0f, 0.0f);
			luisa::float3	x2 = luisa::make_float3(0.0f, 1.0f, 0.0f);
			luisa::float3	X0 = luisa::make_float3(0.0f, 0.0f, 0.0f);
			luisa::float3	X1 = luisa::make_float3(0.0f, 1.0f, 0.0f);
			luisa::float3	X2 = luisa::make_float3(1.0f, 0.1f, 0.0f);
			luisa::float2x2 Dm = compute_Dm(X0, X1, X2);
			luisa::float2x2 Dm_inv = luisa::inverse(Dm);

			const float area = 1.0f;
			// auto [mu_tmp, lambda_tmp] = StretchEnergy::convert_prop(1e4f, 0.2f);
			// const float mu            = mu_tmp;
			// const float lambda        = lambda_tmp;
			const float mu = 1.0f;
			const float lambda = 1.0f;

			std::function<float(const EigenVec&)> shear_only = [&](const EigenVec& xv) -> float
			{
				luisa::float3 vx0 = luisa::make_float3(xv[0], xv[1], xv[2]);
				luisa::float3 vx1 = luisa::make_float3(xv[3], xv[4], xv[5]);
				luisa::float3 vx2 = luisa::make_float3(xv[6], xv[7], xv[8]);
				lcs::float2x3 F = makeFloat2x3(vx1 - vx0, vx2 - vx0) * Dm_inv;
				return area * detail::fem_BW98_cloth_energy::shear_energy(F, lambda);
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
			auto	 dedF_sh = detail::fem_BW98_cloth_energy::shear_gradient(Fsh, lambda);
			auto	 d2eF_sh = detail::fem_BW98_cloth_energy::shear_hessian(Fsh, lambda);
			float3x3 dedx_sh = area * FemUtils::convert_force(dedF_sh, Dm_inv);
			float9x9 d2edx2_sh = area * FemUtils::convert_hessian(d2eF_sh, Dm_inv);

			Eigen::VectorXf g_ana_sh(9);
			Eigen::MatrixXf H_ana_sh = d2edx2_sh.to_eigen_matrix();
			for (int i = 0; i < 3; i++)
				g_ana_sh.segment<3>(3 * i) = float3_to_eigen3(dedx_sh[i]);

			print_diff("StretchFace_Shear", g_num, g_ana_sh, H_num, H_ana_sh, 1e-2f);
		}

		// 3) ARAP tet (4 vertices)
		{
			luisa::float3 X0 = luisa::make_float3(0.0f, 0.0f, 0.0f);
			luisa::float3 X1 = luisa::make_float3(1.0f, 0.0f, 0.0f);
			luisa::float3 X2 = luisa::make_float3(0.0f, 1.0f, 0.0f);
			luisa::float3 X3 = luisa::make_float3(0.0f, 0.0f, 1.0f);

			luisa::float3x3 Dm = luisa::make_float3x3(X1 - X0, X2 - X0, X3 - X0);
			luisa::float3x3 Dm_inv = luisa::inverse(Dm);
			const float		volume = std::abs(luisa::determinant(Dm)) / 6.0f;
			const float		mu = 2.0f;
			const float		lambda = 0.5f;

			std::function<float(const EigenVec&)> arap_func = [&](const EigenVec& xv) -> float
			{
				auto in = detail::arap_tet_energy::Input<float3, float3x3, float>{
					.x0 = luisa::make_float3(xv[0], xv[1], xv[2]),
					.x1 = luisa::make_float3(xv[3], xv[4], xv[5]),
					.x2 = luisa::make_float3(xv[6], xv[7], xv[8]),
					.x3 = luisa::make_float3(xv[9], xv[10], xv[11]),
					.dm_inv = Dm_inv,
					.mu = mu,
					.lambda = lambda,
					.volume = volume,
				};
				return detail::arap_tet_energy::compute_energy(in);
			};

			EigenVec x0(12);
			x0 << 0.05f, -0.02f, 0.01f,
				1.22f, 0.08f, -0.03f,
				0.11f, 1.10f, 0.06f,
				-0.04f, 0.15f, 1.18f;
			Eigen::VectorXf g_num;
			Eigen::MatrixXf H_num;
			FiniteDiff::computeGradientAndHessian<float, Eigen::Dynamic>(arap_func, x0, g_num, H_num, fd_h, false);
			H_num = hessian_proj_SPD(H_num);

			auto in = detail::arap_tet_energy::Input<float3, float3x3, float>{
				.x0 = luisa::make_float3(x0[0], x0[1], x0[2]),
				.x1 = luisa::make_float3(x0[3], x0[4], x0[5]),
				.x2 = luisa::make_float3(x0[6], x0[7], x0[8]),
				.x3 = luisa::make_float3(x0[9], x0[10], x0[11]),
				.dm_inv = Dm_inv,
				.mu = mu,
				.lambda = lambda,
				.volume = volume,
			};
			auto eval = detail::arap_tet_energy::evaluate_host(in);

			Eigen::VectorXf g_ana(12);
			Eigen::MatrixXf H_ana = Eigen::MatrixXf::Zero(12, 12);
			for (int a = 0; a < 4; ++a)
			{
				g_ana.segment<3>(3 * a) = float3_to_eigen3(eval.gradients[a]);
				for (int b = 0; b < 4; ++b)
				{
					H_ana.block<3, 3>(3 * a, 3 * b) = float3x3_to_eigen3x3(eval.hessians[a * 4 + b]);
				}
			}

			// ARAP uses SVD-based analytic modes and is compared against an FD Hessian
			// after PSD projection, so allow a slightly wider tolerance here.
			print_diff("TetARAP", g_num, g_ana, H_num, H_ana, 1.1e-2f);
		}

		// 4) Bending (gradient-only): Hessian uses Gauss-Newton approximation,
		// so we only check first-order consistency here.
		{
			luisa::float3 q0 = luisa::make_float3(0.0f, 0.0f, 0.0f);
			luisa::float3 q1 = luisa::make_float3(1.0f, 0.0f, 0.0f);
			luisa::float3 q2 = luisa::make_float3(0.0f, 1.0f, 0.0f);
			luisa::float3 q3 = luisa::make_float3(0.0f, 0.0f, 1.0f);
			const float	  area = 1.0f;
			const float	  stiff = 10.0f;

			std::function<float(const EigenVec&)> bend_func = [&](const EigenVec& xv) -> float
			{
				return detail::bending_energy::compute_energy(luisa::make_float3(xv[0], xv[1], xv[2]),
					luisa::make_float3(xv[3], xv[4], xv[5]),
					luisa::make_float3(xv[6], xv[7], xv[8]),
					luisa::make_float3(xv[9], xv[10], xv[11]), 0.0f, stiff * area);
			};

			EigenVec x0(12);
			x0 << q0.x, q0.y, q0.z, q1.x, q1.y, q1.z, q2.x, q2.y, q2.z, q3.x, q3.y, q3.z;
			Eigen::VectorXf g_num;
			Eigen::MatrixXf H_dummy;
			FiniteDiff::computeGradientAndHessian<float, Eigen::Dynamic>(bend_func, x0, g_num, H_dummy, fd_h, false);

			auto eval = detail::bending_energy::evaluate<float, float3, float3x3>(
				luisa::make_float3(x0[0], x0[1], x0[2]),
				luisa::make_float3(x0[3], x0[4], x0[5]),
				luisa::make_float3(x0[6], x0[7], x0[8]),
				luisa::make_float3(x0[9], x0[10], x0[11]),
				0.0f,
				stiff * area);

			Eigen::VectorXf g_ana(12);
			for (int ii = 0; ii < 4; ++ii)
			{
				g_ana.segment<3>(3 * ii) = float3_to_eigen3(eval.gradients[ii]);
			}
			print_grad_only("Bending_GradientOnly", g_num, g_ana, 2e-3f);
		}

		// 5) ABD inertia (4 control points / 12 dof)
		{
			const float			  scaled_stiffness = 3.0f;
			const luisa::float4x4 M = luisa::make_float4x4(
				0.50f, 0.10f, 0.08f, 0.02f,
				0.10f, 0.45f, 0.05f, 0.03f,
				0.08f, 0.05f, 0.40f, 0.04f,
				0.02f, 0.03f, 0.04f, 0.35f);
			luisa::float3 q_tilde[4] = {
				luisa::make_float3(0.0f, 0.0f, 0.0f),
				luisa::make_float3(1.0f, 0.0f, 0.0f),
				luisa::make_float3(0.0f, 1.0f, 0.0f),
				luisa::make_float3(0.0f, 0.0f, 1.0f),
			};

			std::function<float(const EigenVec&)> abd_inertia_func = [&](const EigenVec& xv) -> float
			{
				luisa::float3 delta[4];
				for (int i = 0; i < 4; ++i)
				{
					delta[i] = luisa::make_float3(xv[3 * i + 0] - q_tilde[i][0],
						xv[3 * i + 1] - q_tilde[i][1],
						xv[3 * i + 2] - q_tilde[i][2]);
				}
				return detail::abd_inertia_energy::compute_energy(delta, M, scaled_stiffness);
			};

			EigenVec x0(12);
			x0 << 0.02f, -0.01f, 0.03f,
				1.05f, 0.02f, -0.01f,
				-0.03f, 1.08f, 0.04f,
				0.01f, -0.02f, 1.10f;
			Eigen::VectorXf g_num;
			Eigen::MatrixXf H_num;
			FiniteDiff::computeGradientAndHessian<float, Eigen::Dynamic>(abd_inertia_func, x0, g_num, H_num, fd_h, false);

			luisa::float3 delta[4];
			for (int i = 0; i < 4; ++i)
			{
				delta[i] = luisa::make_float3(x0[3 * i + 0] - q_tilde[i][0],
					x0[3 * i + 1] - q_tilde[i][1],
					x0[3 * i + 2] - q_tilde[i][2]);
			}
			auto eval = detail::abd_inertia_energy::evaluate(delta, M, scaled_stiffness, luisa::make_float3x3(1.0f));

			Eigen::VectorXf g_ana(12);
			Eigen::MatrixXf H_ana = Eigen::MatrixXf::Zero(12, 12);
			for (int a = 0; a < 4; ++a)
			{
				g_ana.segment<3>(3 * a) = float3_to_eigen3(eval.gradients[a]);
				for (int b = 0; b < 4; ++b)
				{
					H_ana.block<3, 3>(3 * a, 3 * b) = float3x3_to_eigen3x3(eval.hessians[a * 4 + b]);
				}
			}
			print_diff("ABDInertia", g_num, g_ana, H_num, H_ana, 1e-3f);
		}

		// 6) ABD ortho (A in R^{3x3}, 9 dof)
		{
			const float							  stiff = 2.0f;
			std::function<float(const EigenVec&)> abd_ortho_func = [&](const EigenVec& xv) -> float
			{
				luisa::float3x3 A = luisa::make_float3x3(
					luisa::make_float3(xv[0], xv[1], xv[2]),
					luisa::make_float3(xv[3], xv[4], xv[5]),
					luisa::make_float3(xv[6], xv[7], xv[8]));
				return detail::abd_ortho_energy::compute_energy(A, stiff);
			};

			EigenVec x0(9);
			x0 << 1.02f, 0.03f, -0.01f,
				-0.02f, 0.98f, 0.04f,
				0.01f, -0.05f, 1.03f;
			Eigen::VectorXf g_num;
			Eigen::MatrixXf H_num;
			FiniteDiff::computeGradientAndHessian<float, Eigen::Dynamic>(abd_ortho_func, x0, g_num, H_num, fd_h, false);

			luisa::float3x3 A = luisa::make_float3x3(
				luisa::make_float3(x0[0], x0[1], x0[2]),
				luisa::make_float3(x0[3], x0[4], x0[5]),
				luisa::make_float3(x0[6], x0[7], x0[8]));
			auto eval = detail::abd_ortho_energy::evaluate<float, float3, float3x3>(A, stiff, luisa::make_float3x3(1.0f));

			Eigen::VectorXf g_ana(9);
			Eigen::MatrixXf H_ana = Eigen::MatrixXf::Zero(9, 9);
			for (int a = 0; a < 3; ++a)
			{
				g_ana.segment<3>(3 * a) = float3_to_eigen3(eval.gradients[a]);
				for (int b = 0; b < 3; ++b)
				{
					H_ana.block<3, 3>(3 * a, 3 * b) = float3x3_to_eigen3x3(eval.hessians[a * 3 + b]);
				}
			}
			print_diff("ABDOrtho", g_num, g_ana, H_num, H_ana, 1e-3f);
		}

		// 7) Fixed joint (two rigid bodies, 8 blocks = 24 dof)
		{
			const float	 k_pos = 20.0f;
			const float	 k_rot = 5.0f;
			const float3 anchor_a = luisa::make_float3(0.1f, -0.05f, 0.03f);
			const float3 anchor_b = luisa::make_float3(-0.02f, 0.08f, -0.04f);
			const float3 rest_position_delta = luisa::make_float3(0.03f, -0.01f, 0.02f);
			const float3 rest_rot_col0 = luisa::make_float3(0.98f, 0.05f, 0.00f);
			const float3 rest_rot_col1 = luisa::make_float3(-0.05f, 0.98f, 0.01f);
			const float3 rest_rot_col2 = luisa::make_float3(0.00f, -0.01f, 1.00f);

			std::function<float(const EigenVec&)> fixed_joint_func = [&](const EigenVec& xv) -> float
			{
				float3 q[8];
				for (int i = 0; i < 8; ++i)
				{
					q[i] = luisa::make_float3(xv[3 * i + 0], xv[3 * i + 1], xv[3 * i + 2]);
				}
				return detail::fixed_joint_constaint::compute_energy(
					q,
					anchor_a,
					anchor_b,
					rest_position_delta,
					rest_rot_col0,
					rest_rot_col1,
					rest_rot_col2,
					k_pos,
					k_rot,
					luisa::make_float3x3(1.0f));
			};

			EigenVec x0(24);
			x0 << 0.20f, 0.30f, -0.10f,
				1.02f, 0.01f, -0.02f,
				-0.03f, 0.98f, 0.04f,
				0.02f, -0.01f, 1.01f,
				0.26f, 0.24f, -0.08f,
				0.99f, -0.04f, 0.03f,
				0.02f, 1.01f, -0.02f,
				-0.01f, 0.03f, 0.97f;

			Eigen::VectorXf g_num;
			Eigen::MatrixXf H_num;
			FiniteDiff::computeGradientAndHessian<float, Eigen::Dynamic>(fixed_joint_func, x0, g_num, H_num, fd_h, false);

			float3 q[8];
			for (int i = 0; i < 8; ++i)
			{
				q[i] = luisa::make_float3(x0[3 * i + 0], x0[3 * i + 1], x0[3 * i + 2]);
			}
			auto eval = detail::fixed_joint_constaint::evaluate(
				q,
				anchor_a,
				anchor_b,
				rest_position_delta,
				rest_rot_col0,
				rest_rot_col1,
				rest_rot_col2,
				k_pos,
				k_rot,
				luisa::make_float3x3(1.0f));

			Eigen::VectorXf g_ana(24);
			Eigen::MatrixXf H_ana = Eigen::MatrixXf::Zero(24, 24);
			for (int a = 0; a < 8; ++a)
			{
				g_ana.segment<3>(3 * a) = float3_to_eigen3(eval.gradients[a]);
				for (int b = 0; b < 8; ++b)
				{
					H_ana.block<3, 3>(3 * a, 3 * b) = float3x3_to_eigen3x3(eval.hessians[a * 8 + b]);
				}
			}
			print_diff("JointFixed", g_num, g_ana, H_num, H_ana, 1e-3f);
		}

		// 8) Prismatic joint (slide axis in body-A local x)
		{
			const float	 k_pos = 12.0f;
			const float	 k_rot = 3.5f;
			const float3 anchor_a = luisa::make_float3(0.0f, 0.05f, -0.02f);
			const float3 anchor_b = luisa::make_float3(0.0f, -0.03f, 0.01f);
			const float3 rest_position_delta = luisa::make_float3(-0.02f, 0.04f, -0.01f);
			const float3 rest_rot_col0 = luisa::make_float3(1.00f, 0.02f, 0.00f);
			const float3 rest_rot_col1 = luisa::make_float3(-0.02f, 1.00f, 0.01f);
			const float3 rest_rot_col2 = luisa::make_float3(0.00f, -0.01f, 1.00f);
			const float3 axis_a_local = luisa::make_float3(1.0f, 0.0f, 0.0f);

			std::function<float(const EigenVec&)> prismatic_joint_func = [&](const EigenVec& xv) -> float
			{
				float3 q[8];
				for (int i = 0; i < 8; ++i)
				{
					q[i] = luisa::make_float3(xv[3 * i + 0], xv[3 * i + 1], xv[3 * i + 2]);
				}
				return detail::prismatic_joint_constaint::compute_energy(
					q,
					anchor_a,
					anchor_b,
					rest_position_delta,
					rest_rot_col0,
					rest_rot_col1,
					rest_rot_col2,
					axis_a_local,
					k_pos,
					k_rot,
					-std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
					luisa::make_float3x3(1.0f));
			};

			EigenVec x0(24);
			x0 << 0.20f, 0.10f, 0.00f,
				1.01f, 0.02f, -0.01f,
				-0.01f, 1.00f, 0.03f,
				0.00f, -0.02f, 0.99f,
				0.28f, 0.12f, 0.02f,
				0.98f, 0.01f, -0.03f,
				0.01f, 0.99f, 0.02f,
				0.02f, -0.01f, 1.02f;

			Eigen::VectorXf g_num;
			Eigen::MatrixXf H_num;
			FiniteDiff::computeGradientAndHessian<float, Eigen::Dynamic>(prismatic_joint_func, x0, g_num, H_num, fd_h, false);

			float3 q[8];
			for (int i = 0; i < 8; ++i)
			{
				q[i] = luisa::make_float3(x0[3 * i + 0], x0[3 * i + 1], x0[3 * i + 2]);
			}
			auto eval = detail::prismatic_joint_constaint::evaluate(
				q,
				anchor_a,
				anchor_b,
				rest_position_delta,
				rest_rot_col0,
				rest_rot_col1,
				rest_rot_col2,
				axis_a_local,
				k_pos,
				k_rot,
				-std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
				luisa::make_float3x3(1.0f));

			Eigen::VectorXf g_ana(24);
			Eigen::MatrixXf H_ana = Eigen::MatrixXf::Zero(24, 24);
			for (int a = 0; a < 8; ++a)
			{
				g_ana.segment<3>(3 * a) = float3_to_eigen3(eval.gradients[a]);
				for (int b = 0; b < 8; ++b)
				{
					H_ana.block<3, 3>(3 * a, 3 * b) = float3x3_to_eigen3x3(eval.hessians[a * 8 + b]);
				}
			}
			print_diff("JointPrismatic", g_num, g_ana, H_num, H_ana, 1e-3f);
		}

		// 9) Revolute joint (hinge axis in world-z)
		{
			const float	 k_pos = 15.0f;
			const float	 k_axis = 4.0f;
			const float3 anchor_a = luisa::make_float3(0.02f, -0.01f, 0.0f);
			const float3 anchor_b = luisa::make_float3(-0.01f, 0.02f, 0.0f);
			const float3 rest_position_delta = luisa::make_float3(0.01f, -0.03f, 0.02f);
			const float3 axis_a_local = luisa::make_float3(0.0f, 0.0f, 1.0f);
			const float3 axis_b_local = luisa::make_float3(0.0f, 0.0f, 1.0f);

			std::function<float(const EigenVec&)> revolute_joint_func = [&](const EigenVec& xv) -> float
			{
				float3 q[8];
				for (int i = 0; i < 8; ++i)
				{
					q[i] = luisa::make_float3(xv[3 * i + 0], xv[3 * i + 1], xv[3 * i + 2]);
				}
				return detail::revolute_joint_constaint::compute_energy(
					q,
					anchor_a,
					anchor_b,
					rest_position_delta,
					axis_a_local,
					axis_b_local,
					k_pos,
					k_axis,
					luisa::make_float3x3(1.0f));
			};

			EigenVec x0(24);
			x0 << -0.05f, 0.03f, 0.01f,
				0.99f, 0.02f, 0.00f,
				-0.03f, 1.01f, 0.01f,
				0.01f, -0.02f, 0.98f,
				0.01f, 0.05f, -0.02f,
				1.02f, -0.01f, 0.00f,
				0.01f, 0.97f, -0.03f,
				0.02f, 0.02f, 1.01f;

			Eigen::VectorXf g_num;
			Eigen::MatrixXf H_num;
			FiniteDiff::computeGradientAndHessian<float, Eigen::Dynamic>(revolute_joint_func, x0, g_num, H_num, fd_h, false);

			float3 q[8];
			for (int i = 0; i < 8; ++i)
			{
				q[i] = luisa::make_float3(x0[3 * i + 0], x0[3 * i + 1], x0[3 * i + 2]);
			}
			auto eval = detail::revolute_joint_constaint::evaluate(
				q,
				anchor_a,
				anchor_b,
				rest_position_delta,
				axis_a_local,
				axis_b_local,
				k_pos,
				k_axis,
				luisa::make_float3x3(1.0f));

			Eigen::VectorXf g_ana(24);
			Eigen::MatrixXf H_ana = Eigen::MatrixXf::Zero(24, 24);
			for (int a = 0; a < 8; ++a)
			{
				g_ana.segment<3>(3 * a) = float3_to_eigen3(eval.gradients[a]);
				for (int b = 0; b < 8; ++b)
				{
					H_ana.block<3, 3>(3 * a, 3 * b) = float3x3_to_eigen3x3(eval.hessians[a * 8 + b]);
				}
			}
			print_diff("JointRevolute", g_num, g_ana, H_num, H_ana, 1e-3f);
		}

		// 10) Stable Neo-Hookean tet (gradient-only)
		{
			luisa::float3 X0 = luisa::make_float3(0.0f, 0.0f, 0.0f);
			luisa::float3 X1 = luisa::make_float3(1.0f, 0.0f, 0.0f);
			luisa::float3 X2 = luisa::make_float3(0.0f, 1.0f, 0.0f);
			luisa::float3 X3 = luisa::make_float3(0.0f, 0.0f, 1.0f);

			luisa::float3x3 Dm = luisa::make_float3x3(X1 - X0, X2 - X0, X3 - X0);
			luisa::float3x3 Dm_inv = luisa::inverse(Dm);
			const float		volume = std::abs(luisa::determinant(Dm)) / 6.0f;
			const float		mu = 2.0f;
			const float		lambda = 0.5f;

			std::function<float(const EigenVec&)> arap_func = [&](const EigenVec& xv) -> float
			{
				auto in = detail::stable_neo_hookean_energy::Input<float3, float3x3, float>{
					.x0 = luisa::make_float3(xv[0], xv[1], xv[2]),
					.x1 = luisa::make_float3(xv[3], xv[4], xv[5]),
					.x2 = luisa::make_float3(xv[6], xv[7], xv[8]),
					.x3 = luisa::make_float3(xv[9], xv[10], xv[11]),
					.dm_inv = Dm_inv,
					.mu = mu,
					.lambda = lambda,
					.volume = volume,
				};
				return detail::stable_neo_hookean_energy::compute_energy(in);
			};

			EigenVec x0(12);
			x0 << 0.05f, -0.02f, 0.01f,
				1.22f, 0.08f, -0.03f,
				0.11f, 1.10f, 0.06f,
				-0.04f, 0.15f, 1.18f;
			Eigen::VectorXf g_num;
			Eigen::MatrixXf H_num;
			FiniteDiff::computeGradientAndHessian<float, Eigen::Dynamic>(arap_func, x0, g_num, H_num, fd_h, false);
			H_num = hessian_proj_SPD(H_num);

			auto in = detail::stable_neo_hookean_energy::Input<float3, float3x3, float>{
				.x0 = luisa::make_float3(x0[0], x0[1], x0[2]),
				.x1 = luisa::make_float3(x0[3], x0[4], x0[5]),
				.x2 = luisa::make_float3(x0[6], x0[7], x0[8]),
				.x3 = luisa::make_float3(x0[9], x0[10], x0[11]),
				.dm_inv = Dm_inv,
				.mu = mu,
				.lambda = lambda,
				.volume = volume,
			};
			lcs::detail::EnergyEvalResult<4, 16, float3, float3x3> eval{};
			for (int i = 0; i < 4; ++i)
			{
				eval.gradients[i] = luisa::make_float3(0.0f);
				for (int j = 0; j < 4; ++j)
				{
					eval.hessians[i * 4 + j] = luisa::make_float3x3(0.0f);
				}
			}

			eval = detail::stable_neo_hookean_energy::evaluate(in);
			// detail::TetElasticEnergyUtils::compute_gradient_hessian(
			// 	in.x0, in.x1, in.x2, in.x3, in.dm_inv, in.mu, in.lambda, in.volume, eval.gradients.data(), eval.hessians.data());

			Eigen::VectorXf g_ana(12);
			Eigen::MatrixXf H_ana = Eigen::MatrixXf::Zero(12, 12);
			for (int a = 0; a < 4; ++a)
			{
				g_ana.segment<3>(3 * a) = float3_to_eigen3(eval.gradients[a]);
				for (int b = 0; b < 4; ++b)
				{
					H_ana.block<3, 3>(3 * a, 3 * b) = float3x3_to_eigen3x3(eval.hessians[a * 4 + b]);
				}
			}

			print_diff("TetStableNeoHookean", g_num, g_ana, H_num, H_ana, 1e-2f);
		}

		// 11) Ground collision repulsive term (1 dof in y)
		{
			const float floor_y = 0.0f;
			const float thickness = 0.01f;
			const float d_hat = 0.1f;
			const float stiff = 100.0f;
			const uint	collision_type = 0u;

			std::function<float(const EigenVec&)> ground_func = [&](const EigenVec& xv) -> float
			{
				float dist = xv[0] - floor_y;
				return detail::ground_collision_energy::repulsive_energy(
					dist, thickness, d_hat, stiff, collision_type);
			};

			EigenVec x0(1);
			x0 << 0.04f;
			Eigen::VectorXf g_num;
			Eigen::MatrixXf H_num;
			FiniteDiff::computeGradientAndHessian<float, Eigen::Dynamic>(ground_func, x0, g_num, H_num, fd_h, false);

			Eigen::VectorXf g_ana(1);
			Eigen::MatrixXf H_ana(1, 1);
			float			dist = x0[0] - floor_y;
			g_ana[0] = detail::ground_collision_energy::repulsive_first_derivative(
				dist, thickness, d_hat, stiff, collision_type);
			H_ana(0, 0) = detail::ground_collision_energy::repulsive_second_derivative(
				dist, thickness, d_hat, stiff, collision_type);

			print_diff("GroundCollision_Repulsive", g_num, g_ana, H_num, H_ana, 1e-3f);
		}

		return;
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
