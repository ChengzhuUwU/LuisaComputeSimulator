#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <iomanip>
#include "Energies/tet_elastic_energy.h"

// Use implementations from tet_elastic_energy.h (SVD singular-value space functions)
using namespace lcs::TetElasticEnergyUtils;

using Vec3 = Eigen::Vector3f;
using Mat3 = Eigen::Matrix3f;

// Helper: Print eigenvalues and check positive-definiteness
// Returns: "PD" (all > eps), "PSD" (all >= -tol), or "INDEF" (has clearly negative)
std::string check_definiteness(float H[9], float eps_pd = 1e-7, float tol_psd = 1e-6)
{
	Eigen::Matrix3f mat;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			mat(i, j) = H[i * 3 + j];
		}
	}

	Eigen::SelfAdjointEigenSolver<Mat3> es(mat);
	Vec3								eigenvals = es.eigenvalues();

	float min_eig = eigenvals.minCoeff();
	std::cout << "  Hessian eigenvalues: " << eigenvals.transpose() << std::endl;

	if (min_eig > eps_pd)
	{
		std::cout << "  Min eigenvalue: " << std::scientific << min_eig << std::defaultfloat
				  << " → Positive Definite (PD)" << std::endl;
		return "PD";
	}
	else if (min_eig > -tol_psd)
	{
		std::cout << "  Min eigenvalue: " << std::scientific << min_eig << std::defaultfloat
				  << " → Positive Semi-Definite (PSD, numerical)" << std::endl;
		return "PSD";
	}
	else
	{
		std::cout << "  Min eigenvalue: " << std::scientific << min_eig << std::defaultfloat
				  << " → Indefinite (clearly negative)" << std::endl;
		return "INDEF";
	}
}

// Gauss-Newton approximation: H_GN = g*g^T / norm(g)^2
void gauss_newton_approx(float g[3], float H_gn[9])
{
	float norm_g_sq = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
	if (norm_g_sq < 1e-12f)
	{
		// If gradient is zero, return zero Hessian
		for (int i = 0; i < 9; i++)
			H_gn[i] = 0.0f;
	}
	else
	{
		// H_gn[i,j] = g[i] * g[j] / norm_g_sq
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				H_gn[i * 3 + j] = g[i] * g[j] / norm_g_sq;
			}
		}
	}
}

int main()
{
	// Create a proper 3D tetrahedron
	// Regular tet: vertices forming a real 3D volume
	// Vec3 x0(0.0f, 0.0f, 0.0f);
	// Vec3 x1(1.0f, 0.0f, 0.0f);
	// Vec3 x2(0.5f, 1.0f, 0.0f);
	// Vec3 x3(0.5f, 0.5f, 1.0f); // z != 0 for real 3D
	// Vec3 x0(0.2, 0.3, -0.2);
	// Vec3 x1(-0.2, 0.7, -0.2);
	// Vec3 x2(-0.2, 0.3, 0.2);
	// Vec3 x3(0.2, 0.7, 0.2);

	Vec3 x0(-0.2, 0.7, -0.2);
	Vec3 x1(-0.2, 0.3, 0.2);
	Vec3 x2(0.2, 0.7, 0.2);
	Vec3 x3(-0.2, 0.7, 0.2);

	// Compute Ds (rest configuration)
	Mat3 Ds;
	Ds.col(0) = x1 - x0;
	Ds.col(1) = x2 - x0;
	Ds.col(2) = x3 - x0;

	std::cout << "Rest Ds (deformation gradient basis):\n"
			  << Ds << std::endl;
	std::cout << "det(Ds) = " << Ds.determinant() << std::endl;

	// Compute Dm_inv (inverse of rest shape matrix)
	Mat3 Dm_inv = Ds.inverse();
	std::cout << "\nDm_inv:\n"
			  << Dm_inv << std::endl;
	std::cout << "det(Dm_inv) = " << Dm_inv.determinant() << std::endl;

	// Compute rest volume
	float rest_volume = std::abs(Ds.determinant()) / 6.0f;
	std::cout << "Rest volume: " << rest_volume << std::endl;

	// Material params
	float E = 1e6, nu = 0.4;
	float mu = E / (2.0f * (1.0f + nu));
	float lambda = E * nu / ((1.0f + nu) * (1.0f - 2.0f * nu));

	std::cout << "\nMaterial: E=" << E << " nu=" << nu << std::endl;
	std::cout << "mu=" << mu << " lambda=" << lambda << std::endl;

	// ===== Test 1: Identity (F = I) =====
	std::cout << "\n=== TEST 1: Identity deformation (F=I) ===" << std::endl;
	{
		Mat3				   F = Mat3::Identity();
		Eigen::JacobiSVD<Mat3> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Vec3				   S = svd.singularValues();

		std::cout << "F=I, S=" << S.transpose() << std::endl;
		std::cout << "det(F) = " << F.determinant() << std::endl;

		float energy = snhk_energy(S[0], S[1], S[2], mu, lambda);
		std::cout << "Energy: " << energy << " (should be 0)" << std::endl;

		float deda[3];
		snhk_deda(S[0], S[1], S[2], mu, lambda, deda);
		std::cout << std::fixed << std::setprecision(6);
		std::cout << "  SNHk grad: a=" << S[0] << " b=" << S[1] << " c=" << S[2]
				  << " -> grad=(" << deda[0] << ", " << deda[1] << ", " << deda[2] << ")" << std::endl;

		float d2[3][3];
		snhk_d2ed2a(S[0], S[1], S[2], mu, lambda, d2);
		float H[9];
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				H[i * 3 + j] = d2[i][j];
		std::cout << "  Full Hessian:" << std::endl;
		check_definiteness(H);
	}

	// ===== Test 2: Small stretch in x direction =====
	std::cout << "\n=== TEST 2: Small stretch (1.1 in x direction) ===" << std::endl;
	{
		// Only stretch x-coordinates by 1.1, keep y and z unchanged
		Vec3 y0(x0[0] * 1.1f, x0[1], x0[2]);
		Vec3 y1(x1[0] * 1.1f, x1[1], x1[2]);
		Vec3 y2(x2[0] * 1.1f, x2[1], x2[2]);
		Vec3 y3(x3[0] * 1.1f, x3[1], x3[2]);

		Mat3 Ds_def;
		Ds_def.col(0) = y1 - y0;
		Ds_def.col(1) = y2 - y0;
		Ds_def.col(2) = y3 - y0;

		Mat3				   F = Ds_def * Dm_inv;
		Eigen::JacobiSVD<Mat3> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Vec3				   S = svd.singularValues();

		std::cout << "F:\n"
				  << F << std::endl;
		std::cout << "S=" << S.transpose() << std::endl;
		std::cout << "det(F) = " << F.determinant() << " (should > 0)" << std::endl;

		float energy = snhk_energy(S[0], S[1], S[2], mu, lambda);
		std::cout << "Energy: " << energy << " (total: " << energy * rest_volume << ")" << std::endl;

		float deda[3];
		snhk_deda(S[0], S[1], S[2], mu, lambda, deda);
		std::cout << std::fixed << std::setprecision(6);
		std::cout << "  SNHk grad: a=" << S[0] << " b=" << S[1] << " c=" << S[2]
				  << " -> grad=(" << deda[0] << ", " << deda[1] << ", " << deda[2] << ")" << std::endl;

		float d2[3][3];
		snhk_d2ed2a(S[0], S[1], S[2], mu, lambda, d2);
		float H[9];
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				H[i * 3 + j] = d2[i][j];
		std::cout << "  Full Hessian:" << std::endl;
		check_definiteness(H);

		float H_gn[9];
		gauss_newton_approx(deda, H_gn);
		std::cout << "  Gauss-Newton approx (rank-1 PSD):" << std::endl;
		check_definiteness(H_gn);
	}

	// ===== Test 3: Compression (check inversion) =====
	std::cout << "\n=== TEST 3: Extreme compression (0.1x) ===" << std::endl;
	{
		Vec3 y0 = x0;
		Vec3 y1 = x1 * 0.1f;
		Vec3 y2 = x2 * 0.1f;
		Vec3 y3 = x3 * 0.1f;

		Mat3 Ds_def;
		Ds_def.col(0) = y1 - y0;
		Ds_def.col(1) = y2 - y0;
		Ds_def.col(2) = y3 - y0;

		Mat3				   F = Ds_def * Dm_inv;
		Eigen::JacobiSVD<Mat3> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Vec3				   S = svd.singularValues();

		std::cout << "S=" << S.transpose() << std::endl;
		std::cout << "det(F) = " << F.determinant() << " (should > 0 for compression)" << std::endl;

		float energy = snhk_energy(S[0], S[1], S[2], mu, lambda);
		std::cout << "Energy: " << energy << " (total: " << energy * rest_volume << ")" << std::endl;

		float deda[3];
		snhk_deda(S[0], S[1], S[2], mu, lambda, deda);
		std::cout << std::fixed << std::setprecision(6);
		std::cout << "  SNHk grad: a=" << S[0] << " b=" << S[1] << " c=" << S[2]
				  << " -> grad=(" << deda[0] << ", " << deda[1] << ", " << deda[2] << ")" << std::endl;

		float d2[3][3];
		snhk_d2ed2a(S[0], S[1], S[2], mu, lambda, d2);
		float H[9];
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				H[i * 3 + j] = d2[i][j];
		std::cout << "  Full Hessian:" << std::endl;
		check_definiteness(H);
	}

	// ===== Test 4: Near-inversion (J close to 0) =====
	std::cout << "\n=== TEST 4: Near-inversion (J near 0) ===" << std::endl;
	{
		Vec3 y0 = x0;
		Vec3 y1 = x1 * 0.01f;
		Vec3 y2 = x2 * 0.01f;
		Vec3 y3 = x3 * 0.01f;

		Mat3 Ds_def;
		Ds_def.col(0) = y1 - y0;
		Ds_def.col(1) = y2 - y0;
		Ds_def.col(2) = y3 - y0;

		Mat3				   F = Ds_def * Dm_inv;
		Eigen::JacobiSVD<Mat3> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Vec3				   S = svd.singularValues();

		std::cout << "S=" << S.transpose() << std::endl;
		std::cout << "det(F) = J = " << F.determinant() << " (should be small but > 0)" << std::endl;

		double detF = F.determinant();
		if (detF <= 0)
			std::cout << "WARNING: J non-positive (inversion!)." << std::endl;

		float energy = snhk_energy(S[0], S[1], S[2], mu, lambda);
		std::cout << "Energy: " << energy << " (total: " << energy * rest_volume << ")" << std::endl;

		float deda[3];
		snhk_deda(S[0], S[1], S[2], mu, lambda, deda);
		std::cout << std::fixed << std::setprecision(6);
		std::cout << "  SNHk grad: a=" << S[0] << " b=" << S[1] << " c=" << S[2]
				  << " -> grad=(" << deda[0] << ", " << deda[1] << ", " << deda[2] << ")" << std::endl;

		float d2[3][3];
		snhk_d2ed2a(S[0], S[1], S[2], mu, lambda, d2);
		float H[9];
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				H[i * 3 + j] = d2[i][j];
		std::cout << "  Full Hessian:" << std::endl;
		check_definiteness(H);
	}

	return 0;
}
