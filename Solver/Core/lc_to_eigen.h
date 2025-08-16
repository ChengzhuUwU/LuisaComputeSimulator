#pragma once

#include <Eigen/Dense>
#include "Core/float_n.h"
#include "Core/float_nxn.h"

namespace lcsv
{

using EigenFloat3x3 = Eigen::Matrix<float, 3, 3>;
using EigenFloat6x6 = Eigen::Matrix<float, 6, 6>;
using EigenFloat9x9 = Eigen::Matrix<float, 9, 9>;
using EigenFloat12x12 = Eigen::Matrix<float, 12, 12>;
using EigenFloat3   = Eigen::Matrix<float, 3, 1>;
using EigenFloat4   = Eigen::Matrix<float, 4, 1>;
using EigenFloat6   = Eigen::Matrix<float, 6, 1>;
using EigenFloat9   = Eigen::Matrix<float, 9, 1>;
using EigenFloat12   = Eigen::Matrix<float, 12, 1>;

inline EigenFloat3 float3_to_eigen3(const float3& input) { EigenFloat3 vec; vec << input[0], input[1], input[2]; return vec; };
inline EigenFloat4 float4_to_eigen4(const float4& input) { EigenFloat4 vec; vec << input[0], input[1], input[2], input[3]; return vec; };
inline float3 eigen3_to_float3(const EigenFloat3& input) { return luisa::make_float3(input(0, 0), input(1, 0), input(2, 0)); };
inline float4 eigen4_to_float4(const EigenFloat4& input) { return luisa::make_float4(input(0, 0), input(1, 0), input(2, 0), input(3, 0)); };


inline EigenFloat3x3 float3x3_to_eigen3x3(const float3x3& input)
{
    EigenFloat3x3 mat; mat << 
        input[0][0], input[1][0], input[2][0], 
        input[0][1], input[1][1], input[2][1], 
        input[0][2], input[1][2], input[2][2]; 
    return mat;
};
inline float3x3 eigen3x3_to_float3x3(const EigenFloat3x3& input)
{
    return luisa::make_float3x3(
        input(0, 0), input(1, 0), input(2, 0), 
        input(0, 1), input(1, 1), input(2, 1), 
        input(0, 2), input(1, 2), input(2, 2));
};
inline EigenFloat6x6 float6x6_to_eigen6x6(const float6x6& input)
{
    EigenFloat6x6 output;
    for (uint i = 0; i < 2; ++i) 
    {
        for (uint j = 0; j < 2; ++j) 
        {
            output.block<3, 3>(i * 3, j * 3) = float3x3_to_eigen3x3(input.mat[i][j]);
        }
    }
    return output;
};
inline float6x6 eigen6x6_to_float6x6(const EigenFloat6x6& input)
{
    float6x6 output;
    for (uint i = 0; i < 2; ++i) 
    {
        for (uint j = 0; j < 2; ++j) 
        {
            output.mat[i][j] = eigen3x3_to_float3x3(input.block<3, 3>(i * 3, j * 3));
        }
    }
    return output;
};
inline EigenFloat9x9 float9x9_to_eigen9x9(const float9x9& input)
{
    EigenFloat9x9 output;
    for (uint i = 0; i < 3; ++i) 
    {
        for (uint j = 0; j < 3; ++j) 
        {
            output.block<3, 3>(i * 3, j * 3) = float3x3_to_eigen3x3(input.mat[i][j]);
        }
    }
    return output;
};
inline float9x9 eigen9x9_to_float9x9(const EigenFloat9x9& input)
{
    float9x9 output;
    for (uint i = 0; i < 3; ++i) 
    {
        for (uint j = 0; j < 3; ++j) 
        {
            output.mat[i][j] = eigen3x3_to_float3x3(input.block<3, 3>(i * 3, j * 3));
        }
    }
    return output;
};
inline EigenFloat12x12 float12x12_to_eigen12x12(const float12x12 input)
{
    EigenFloat12x12 output;
    for (uint i = 0; i < 4; ++i) 
    {
        for (uint j = 0; j < 4; ++j) 
        {
            output.block<3, 3>(i * 3, j * 3) = float3x3_to_eigen3x3(input.mat[i][j]);
        }
    }
    return output;
};
inline float12x12 eigen12x12_to_float12x12(const EigenFloat9x9& input)
{
    float12x12 output;
    for (uint i = 0; i < 4; ++i) 
    {
        for (uint j = 0; j < 4; ++j) 
        {
            output.mat[i][j] = eigen3x3_to_float3x3(input.block<3, 3>(i * 3, j * 3));
        }
    }
    return output;
};

// SPD projection
template<int N>
Eigen::Matrix<float, N, N> spd_projection(const Eigen::Matrix<float, N, N>& orig_matrix)
{
    // Ensure the matrix is symmetric
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, N, N>> eigensolver(orig_matrix);
    Eigen::Matrix<float, N, 1> eigenvalues = eigensolver.eigenvalues();
    Eigen::Matrix<float, N, N> eigenvectors = eigensolver.eigenvectors();

    // Set negative eigenvalues to zero (or abs, as in your python code)
    for (int i = 0; i < N; ++i) 
    {
        eigenvalues[i] = std::max(0.0f, eigenvalues[i]);
        // eigenvalues(i) = std::abs(eigenvalues(i));
    }

    // Reconstruct the matrix: V * diag(lam) * V^T
    Eigen::Matrix<float, N, N> D = eigenvalues.asDiagonal();
    return eigenvectors * D * eigenvectors.transpose();
}

}